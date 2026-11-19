"""1.4 配方开关的护栏：默认必须逐位等于官方，打开后必须真的改变轨迹。

为什么值得一条测试：这三个开关（center 损失 / 严重度样本权重 / 末轮参数平均）是我们
唯一动到训练循环的地方。如果「默认关闭」不再等价，官方那一轮的 checkpoint 身份就变了，
审计链（parent sha256 → 本轮 sha256）会当场断掉。所以这里锁三件事：

1. 未设环境变量时 `_recipe_ext()` 全关；
2. `ZHIXUN_LOSS=center` 且 λ=1.0 与官方 MSE **逐位一致**（数学上等价，实现上不能走偏）；
3. 严重度列按名字定位（官方 schema 是排序后的 key 顺序，第 34 维才是 vertical spectral_rms）。

Run from `official/`:
    python -m pytest tests/test_recipe_switches.py -q
"""
import numpy as np
import pytest

from harness import baseline

PROFILE = "B"


@pytest.fixture
def tiny():
    """两段寿命、34 维的小数据集：够走完一个 epoch，且能触发分组与分位。"""
    _, _, model_mod = baseline._api(PROFILE)
    paths = [f'[["key","features"],["key","{"horizontal" if i < 17 else "vertical"}"],["key","f{i}"]]'
             for i in range(34)]
    # 把 vertical 的最后两维命名成真实特征名：spectral_rms 是官方排序里的第 34 维
    paths[-1] = '[["key","features"],["key","vertical"],["key","spectral_rms"]]'
    paths[-14] = '[["key","features"],["key","vertical"],["key","band3_energy_6k_12k"]]'
    rng = np.random.default_rng(0)
    rows, ids, ys = [], [], []
    for bearing in ("Bearing1_1", "Bearing1_2"):
        life = 24
        for index in range(life):
            values = rng.normal(0, 1, 34)
            values[-1] = index / life               # 分位随寿命单调
            rows.append(values)
            ids.append(f"xjtu_{bearing}_s{index + 1:06d}")
            ys.append(float((life - index) * 10))
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    return model_mod, paths, x, y, ids


def build(model_mod, paths, x, y, seed=7):
    return model_mod.Model("regression", paths, [], x, y, hidden=8, seed=seed,
                           learning_rate=.01, batch_size=8)


def test_defaults_are_off(monkeypatch, tiny):
    model_mod, *_ = tiny
    for key in ("ZHIXUN_LOSS", "ZHIXUN_CENTER", "ZHIXUN_SEV", "ZHIXUN_SEV_END", "ZHIXUN_SWA"):
        monkeypatch.delenv(key, raising=False)
    assert model_mod._recipe_ext() == {"loss": "mse", "center": 0.35, "sev": 0.0,
                                       "sev_end": False, "swa": 0}


def test_center_lambda_one_is_bitwise_mse(monkeypatch, tiny):
    model_mod, paths, x, y, ids = tiny
    for key in ("ZHIXUN_SEV", "ZHIXUN_SEV_END", "ZHIXUN_SWA"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ZHIXUN_LOSS", "mse")
    plain = build(model_mod, paths, x, y)
    plain.fit(x, y, 2, ids=ids)
    monkeypatch.setenv("ZHIXUN_LOSS", "center")
    monkeypatch.setenv("ZHIXUN_CENTER", "1.0")
    centred = build(model_mod, paths, x, y)
    centred.fit(x, y, 2, ids=ids)
    for name in plain.params:
        assert np.array_equal(plain.params[name], centred.params[name]), name


def test_ids_are_ignored_when_switches_are_off(monkeypatch, tiny):
    model_mod, paths, x, y, ids = tiny
    for key in ("ZHIXUN_LOSS", "ZHIXUN_CENTER", "ZHIXUN_SEV", "ZHIXUN_SEV_END", "ZHIXUN_SWA"):
        monkeypatch.delenv(key, raising=False)
    without = build(model_mod, paths, x, y)
    without.fit(x, y, 1)
    with_ids = build(model_mod, paths, x, y)
    with_ids.fit(x, y, 1, ids=ids)
    for name in without.params:
        assert np.array_equal(without.params[name], with_ids.params[name]), name


def test_severity_column_is_located_by_name(monkeypatch, tiny):
    model_mod, paths, *_ = tiny
    monkeypatch.delenv("ZHIXUN_SEV_COLUMN", raising=False)
    assert "spectral_rms" in paths[model_mod.severity_column(paths)]
    monkeypatch.setenv("ZHIXUN_SEV_COLUMN", "band3")
    assert "band3_energy" in paths[model_mod.severity_column(paths)]


def test_severity_weights_change_the_trajectory(monkeypatch, tiny):
    model_mod, paths, x, y, ids = tiny
    monkeypatch.delenv("ZHIXUN_LOSS", raising=False)
    monkeypatch.delenv("ZHIXUN_SWA", raising=False)
    monkeypatch.setenv("ZHIXUN_SEV", "2")
    monkeypatch.setenv("ZHIXUN_SEV_END", "1")
    weighted = build(model_mod, paths, x, y)
    weighted.fit(x, y, 2, ids=ids)
    monkeypatch.setenv("ZHIXUN_SEV", "0")
    monkeypatch.setenv("ZHIXUN_SEV_END", "0")
    plain = build(model_mod, paths, x, y)
    plain.fit(x, y, 2, ids=ids)
    assert any(not np.array_equal(weighted.params[name], plain.params[name]) for name in plain.params)


def test_swa_averages_last_snapshots(monkeypatch, tiny):
    model_mod, paths, x, y, ids = tiny
    monkeypatch.delenv("ZHIXUN_LOSS", raising=False)
    monkeypatch.delenv("ZHIXUN_SEV", raising=False)
    monkeypatch.setenv("ZHIXUN_SWA", "2")
    model = build(model_mod, paths, x, y)
    model.fit(x, y, 4, ids=ids)
    taken = model.finish_swa()
    assert taken == 2
    assert model.finish_swa() == 0        # 快照用完即清空，重复调用不会二次平均
    model.validate()