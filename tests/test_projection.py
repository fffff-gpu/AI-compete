"""投影包（平台每轮发的 current_round_projection）识别与回退行为。

平台反馈里说清了正式评测的喂法：每轮只给
`public/{train,validation}.jsonl` + `checkpoint.npz` + `baselines/`，
**没有** `baseline-info.json`，目录名也不含 cpu-x，父权重靠 `ZHIXUN_CKPT` 指。
这类目录没有身份文件，最容易出的事故不是"认不出"，而是**认错**或**悄悄回退到本地包**
（后者本机实测发生过：投影测试实际读的是工作区的完整包，complete 从 False 变 True）。

Run from `official/`:
    python -m pytest tests/test_projection.py -q
"""
import json

import pytest

from harness import baseline
from test_discovery import make_package, write_ckpt


@pytest.fixture
def platform_root(tmp_path, monkeypatch):
    """只挂一个显式根，禁掉一切隐式回退，PACKAGES 指向不存在的约定名目录。"""
    root = tmp_path / "platform"
    root.mkdir()
    empty = tmp_path / "workspace"
    empty.mkdir()
    monkeypatch.setattr(baseline, "WORKSPACE", empty)
    monkeypatch.setattr(baseline, "PACKAGES",
                        {p: empty / f"cpu-{p.lower()}-guide-20261110-v1" for p in baseline.PROFILES})
    for key in (*baseline.DATA_ENVS, *baseline.CKPT_ENVS, *baseline.ARTIFACT_ENVS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ZHIXUN_PROFILE", raising=False)
    monkeypatch.delenv("ZHIXUN_DATA_STRICT", raising=False)
    monkeypatch.setenv("ZHIXUN_DATA", str(root))
    baseline.discover_packages._cache = None
    return root


def projection(root, profile="B", name="current_round_projection", ckpt=True, info=False):
    """按平台描述造一个投影包目录：public 两份数据 + baselines/(三份 config) + 可选 checkpoint。"""
    directory = root / name
    public = directory / "public"
    public.mkdir(parents=True)
    (public / "train.jsonl").write_text('{"id":"t1","inputs":{},"answer":"a"}\n', encoding="utf-8")
    (public / "validation.jsonl").write_text('{"id":"v1","inputs":{},"answer":"a"}\n', encoding="utf-8")
    configs = directory / "baselines" / "configs"
    configs.mkdir(parents=True)
    for letter in baseline.PROFILES:                       # 官方每个包都同时带三份 config
        (configs / f"{letter}.json").write_text(json.dumps(
            {"profile": letter, "task": "regression" if letter == "B" else "classification",
             "target_unit": "minutes", "hidden": 24, "seed": 42, "classes": [],
             "learning_rate": 0.003, "batch_size": 128, "epochs": 15}), encoding="utf-8")
    if ckpt:
        write_ckpt(directory / baseline.ARTIFACT_NAME, profile=profile, task="regression")
    if info:
        (directory / "baseline-info.json").write_text(json.dumps({"dataset_key": profile}),
                                                      encoding="utf-8")
    return directory


def test_projection_with_checkpoint_is_recognized(platform_root):
    """checkpoint 的 metadata.profile 是官方训练器自己写的，认它不算猜。"""
    directory = projection(platform_root, "B")
    assert baseline.discover_packages(refresh=True)["B"] == directory
    assert baseline.package("B") == directory
    assert baseline.package_reason("B") == "discovered-in-explicit-root"


def test_identity_bearing_projection_uses_dataset_key(platform_root):
    directory = projection(platform_root, "C", ckpt=False, info=True)
    assert baseline.package("C") == directory


def test_bare_projection_needs_the_callers_intent(platform_root, monkeypatch):
    """三条识别路全断（无 info、无 ckpt、目录名不含 cpu-x）时：只有 ZHIXUN_PROFILE 能认，
    而且必须候选唯一——认不出就报错，而不是拿手边最像的东西顶上。"""
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "1")           # 只看显式根，排除根外回退的干扰
    directory = projection(platform_root, "B", ckpt=False, info=False)
    baseline.discover_packages._cache = None
    with pytest.raises(FileNotFoundError):
        baseline.package("B")                               # 没人说这轮是 B：不猜

    monkeypatch.setenv("ZHIXUN_PROFILE", "B")
    baseline.discover_packages._cache = None
    assert baseline.package("B") == directory
    assert baseline.package_reason("B") == "single-unknown-plus-zhixun-profile"

    # 说了是 B，就不该把同一个目录再当成 A 的包（这正是 configs 判据会犯的错）
    with pytest.raises(FileNotFoundError):
        baseline.package("A")


def test_all_three_configs_do_not_identify_a_package(platform_root, monkeypatch):
    """官方包/投影包都同时带 A.json/B.json/C.json，所以「有 baselines/configs/<P>.json」
    证明不了任何事。曾经用它消歧，会把 A 的包认成 C——这条测试就是钉住它不再回来。"""
    directory = projection(platform_root, "A", name="crp", ckpt=False, info=False)
    assert (directory / "baselines" / "configs" / "C.json").is_file()
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "1")
    baseline.discover_packages._cache = None
    with pytest.raises(FileNotFoundError):
        baseline.package("C")                               # 不给提示：认不出，报错
    monkeypatch.setenv("ZHIXUN_PROFILE", "A")
    baseline.discover_packages._cache = None
    assert baseline.package("A") == directory               # 给了提示且候选唯一：认


def test_strict_never_falls_back_to_a_package_outside_the_explicit_root(platform_root, monkeypatch):
    """约定名完整包**存在于根外**时，严格模式也不许用它——否则"我测的是投影包"就成了一句空话。

    这正是本机实际踩过的：投影目录认不出来，package() 回退到工作区的真实 B 包，
    于是声明 complete 从 False 变 True，测试全绿而测的根本不是投影。
    """
    elsewhere = platform_root.parent / "workspace_real"
    make_package(elsewhere, "B")                             # 完整包，在显式根之外
    monkeypatch.setattr(baseline, "PACKAGES",
                        {**baseline.PACKAGES, "B": elsewhere / "cpu-b-guide-20261110-v1"})
    projection(platform_root, "B", ckpt=False, info=False)   # 显式根里只有一个裸投影目录
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "1")
    baseline.discover_packages._cache = None
    with pytest.raises(FileNotFoundError, match="ZHIXUN_PROFILE"):
        baseline.package("B")


def test_non_strict_fallback_leaving_the_root_is_announced(platform_root, monkeypatch, capsys):
    """非严格模式允许回退到根外的包（老行为，测试与本机开发依赖它），但必须往 stderr 喊一声。"""
    elsewhere = platform_root.parent / "workspace_real2"
    make_package(elsewhere, "B")
    monkeypatch.setattr(baseline, "PACKAGES",
                        {**baseline.PACKAGES, "B": elsewhere / "cpu-b-guide-20261110-v1"})
    directory = projection(platform_root, "B", ckpt=False, info=False)
    baseline.discover_packages._cache = None
    chosen = baseline.package("B")
    assert chosen != directory                               # 确实离开了显式根
    assert "回退" in capsys.readouterr().err                   # 而且说出来了


def test_parent_checkpoint_accepts_a_bare_file_via_env(platform_root, monkeypatch, tmp_path):
    """投影包可以不带 checkpoint.npz，父亲完全由 ZHIXUN_CKPT 指过来（文件形态）。"""
    external = tmp_path / "rounds" / "r03"
    external.mkdir(parents=True)
    ckpt = write_ckpt(external / baseline.ARTIFACT_NAME, profile="B", steps=935, task="regression")
    projection(platform_root, "B", ckpt=False, info=True)
    monkeypatch.setenv("ZHIXUN_CKPT", str(ckpt))
    found, source = baseline.parent_checkpoint("B")
    assert source == "env:ZHIXUN_CKPT" and found == ckpt
