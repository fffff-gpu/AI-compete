"""记录层的两件事：本地 CSV 必须带 v2 判定列，记录层任何故障都不许中断训练。

为什么单独测：`rounds.csv` 是用 `csv.DictWriter(fieldnames=FIELDS, extrasaction="ignore")`
写的——**不在 FIELDS 里的键会被静默丢掉**。v2 的 `gain_lcb / worst_stage_lcb / zone / gate`
一度只在 SwanLab 和 rounds.jsonl 里有，CSV 里没有；照着 CSV 看会得出"这轮涨了 +0.01"，
而门禁其实按 `gain_lcb=-0.004` 判它是噪声。本地表与云端结论不一致，比没表更危险。
"""
import csv
import json

from harness.records import FIELDS, ExperimentRecorder

JUDGMENT = ("gain_lcb", "worst_stage_lcb", "zone", "gate")


def test_judgment_columns_are_in_the_local_csv_fields():
    """判定依据必须在 FIELDS 里，否则会被 extrasaction="ignore" 静默丢掉。"""
    for column in JUDGMENT:
        assert column in FIELDS, column


def test_record_writes_judgment_values_to_both_copies(tmp_path):
    rec = ExperimentRecorder(tmp_path, profile="B", mode="off")
    rec.record(round=1, candidate="wd_weighted", state="alarm", epochs=2, lr=0.003, wd=0.2,
               ls=0.0, jitter=0.0, clip=5.0, score_before=-681.16, score_after=-583.44,
               gain=97.72, accepted=True, worst_stage_gain=91.75,
               stage1_gain=8.66, stage2_gain=-15.53, stage3_gain=-75.52, train_seconds=0.046,
               labels_read=31173, labels_unique=8711, steps_initial=825, steps_final=935,
               gain_lcb=86.73, worst_stage_lcb=82.75, zone="clear_win", gate="strict",
               checkpoint_sha256="sha256:deadbeef", hypothesis="末端加权应该改善最差段")
    summary = rec.finish({"best": {"rmse": 583.44}, "baseline": {"rmse": 681.16}, "metric": "rmse",
                          "accepted_rounds": 1, "worst_stage_gain": 91.75,
                          "train_seconds_total": 0.046})

    row = next(csv.DictReader((tmp_path / "swanlab_export" / "rounds.csv").open(encoding="utf-8")))
    assert float(row["gain_lcb"]) == 86.73 and row["zone"] == "clear_win"
    jsonl = json.loads((tmp_path / "swanlab_export" / "rounds.jsonl").read_text(encoding="utf-8"))
    assert float(jsonl["worst_stage_lcb"]) == 82.75 and jsonl["gate"] == "strict"
    # 记录层的自述必须诚实：mode=off 时不该谎报云端链接，也不该留故障
    assert summary["mode"] == "off" and summary["rows"] == 1


def test_recorder_never_raises_when_swanlab_is_missing(tmp_path, monkeypatch):
    """封闭评测环境里 swanlab 可能装不上/没网：记录层只能记一条 error，绝不能把训练带崩。"""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "swanlab":
            raise ModuleNotFoundError("No module named 'swanlab'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    rec = ExperimentRecorder(tmp_path, profile="A", mode="offline")
    rec.record(round=1, candidate="calibrate", gain=0.01, accepted=True)   # 不该抛
    assert rec.error and "swanlab" in rec.error
    assert (tmp_path / "swanlab_export" / "rounds.csv").is_file()          # 本地副本照写
def test_round_number_is_not_charted_but_still_recorded(tmp_path, monkeypatch):
    """`round` 就是横轴（step），再发一遍只会多一张 45° 直线图；但本地记录里必须还在。"""
    sent = {}

    class FakeSwan:
        def init(self, **kw): return object()
        def log(self, data, step=None): sent.update(data)
        def finish(self): pass
        class config:
            @staticmethod
            def update(*a, **k): pass

    import sys, types
    inst = FakeSwan()
    fake = types.ModuleType("swanlab")
    fake.init, fake.log, fake.finish = inst.init, inst.log, inst.finish
    fake.config = inst.config
    monkeypatch.setitem(sys.modules, "swanlab", fake)

    rec = ExperimentRecorder(tmp_path, profile="B", mode="offline")
    rec.record(round=3, candidate="wd_weighted", gain=97.7, accepted=True, gain_lcb=86.7)
    assert "round" not in sent                      # 曲线里没有它
    assert sent["gain_lcb"] == 86.7                 # 判定依据必须发出去
    row = next(csv.DictReader((tmp_path / "swanlab_export" / "rounds.csv").open(encoding="utf-8")))
    assert row["round"] == "3"                      # 本地权威副本照旧有这一列
