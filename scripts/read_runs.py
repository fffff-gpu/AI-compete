"""把 runs/ 里的实验结果读成能直接判"好还是坏"的表。

    python scripts/read_runs.py                       # 全部 run 的总览（写 docs/实验总览.md）
    python scripts/read_runs.py runs/b_v2x            # 单个 run 的逐轮明细
    python scripts/read_runs.py --profile B           # 只看 B 的 run

为什么要这个脚本：`runs/*/**.csv` 与 `results.json` 里的 `score_before/score_after/gain`
是**统一成"越大越好"**的方向（B/C 取过负号），所以 CSV 里 `score_after=-583.4` 表示
RMSE 583.4。直接看 CSV 十有八九会把它读成"负数=变差"，这个坑我们已经在报告里踩过一次。
本脚本一律换算回各自真实的指标（F1 越大越好、RMSE 越小越好），并且把判定规则写在表里，
不是只把数字摊开。

判读口径（与《设计说明》§5.1、赛题"最差阶段"档一致）：
  ✓ 好    综合改善 ≥ min_gain，且最差阶段不后退（worst_stage_gain ≥ 0）
  ⚠ 可疑  综合改善但最差阶段后退——典型的"只降平均值、伤最后一段寿命"，不能当提交值
  ✗ 差    综合没改善（或变差）；`a_nogate` 就是故意留给评审看的反面证据
  —       冒烟/探针/宽度扫描等非主实验目录（名字带下划线前缀或 rounds<3）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "实验总览.md"

# 提交口径的最小改善量：与 harness/run.py 的 MIN_GAIN 一致（A 是 F1，B 是分钟，C 是循环）
MIN_GAIN = {"A": 0.002, "B": 5.0, "C": 0.5}
# A 的单轨迹噪声标准差 0.0268：涨得比噪声小就不叫涨。门禁看的是这个，不是"数字变没变"。
SIGMA = {"A": 0.0268, "B": 6.3, "C": 0.35}


def load(run: Path) -> dict | None:
    """读 results.json + rounds.csv；缺任何一个就返回 None（不猜数据）。"""
    sj = run / "results.json"
    if not sj.is_file():
        return None
    try:
        data = json.loads(sj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["_dir"] = run
    csv_path = run / "swanlab_export" / "rounds.csv"
    data["_rounds"] = list(csv.DictReader(csv_path.open(encoding="utf-8"))) if csv_path.is_file() else []
    return data


def real(profile: str, metrics: dict) -> float:
    """该 profile 的**真实主指标**（不做方向翻转）：A=Macro-F1，B/C=RMSE。"""
    return float(metrics["macro_f1"] if profile == "A" else metrics["rmse"])


def improves(profile: str, base: float, best: float) -> float:
    """改进幅度，一律正数（涨=正，跌=负），单位随 profile：A 是 F1 点数，B 分钟，C 循环。"""
    return (best - base) if profile == "A" else (base - best)


def verdict(data: dict) -> tuple[str, str]:
    """按 §设计说明 的口径给结论，并把理由写成一句人话（评审和队友都能看懂）。"""
    profile = data["profile"]
    base = real(profile, data["baseline"])
    best = real(profile, data["best"])
    gain = improves(profile, base, best)
    need = MIN_GAIN.get(profile, 0.0)
    worst = float(data.get("worst_stage_gain") or 0.0)
    rounds = int(data.get("rounds") or len(data["_rounds"]))
    name = data["_dir"].name
    if name.startswith("_") or rounds < 3:
        return "—", "校验/探针/冒烟类目录（下划线前缀或轮数<3），数字不作提交依据"
    if gain < need:
        return "✗", "综合改善 {:.4g} 没到提交口径 {}，按噪声 {} 也不算显著".format(gain, need, SIGMA.get(profile))
    if worst < 0:
        return "⚠", "综合改善 {:.4g} 但最差阶段后退 {:.4g}：只降平均、伤末段，不能当提交值".format(gain, worst)
    return "✓", "综合改善 {:.4g} ≥ {}，且最差阶段不退（{:+.4g}）".format(gain, need, worst)


def overview() -> list[dict]:
    rows = []
    for run in sorted(RUNS.glob("*/")):
        data = load(run)
        if data is None or data["profile"] not in MIN_GAIN:
            continue
        profile = data["profile"]
        base, best = real(profile, data["baseline"]), real(profile, data["best"])
        mark, why = verdict(data)
        accepted = int(data.get("accepted_rounds") or 0)
        ledger = data.get("ledger") or {}
        seconds = float(data.get("train_seconds_total") or 0.0)
        rows.append({
            "判": mark, "run": run.name, "profile": profile, "轮": data.get("rounds"),
            "口径": "v2" if data.get("gate_v2") else "v1",
            "基线": round(base, 4), "结果": round(best, 4),
            "改进幅度": round(improves(profile, base, best), 4),
            "相对": "{:+.1f}%".format(100.0 * improves(profile, base, best) / base if base else 0.0),
            "采纳": "{}/{}".format(accepted, data.get("rounds") or 0),
            "最差阶段": round(float(data.get("worst_stage_gain") or 0.0), 4),
            "标注账": ledger.get("labels"), "cost_units": round(float(ledger.get("cost_units") or 0.0), 1),
            "秒": round(seconds, 1), "理由": why})
    return rows


def main_metric(profile: str) -> str:
    return "macro_f1" if profile == "A" else "rmse"


def rounds_of(data: dict) -> tuple[list[dict], str]:
    """逐轮记录，**归一到真实指标空间**（A=Macro-F1，B/C=RMSE），返回 (行, 来源)。

    两个来源的表示法不一样，混着看必然读错：
    * `swanlab_export/rounds.csv` 的 `score_before/score_after` 是"越大越好"的统一方向（B/C 取过负号）；
    * `journal.jsonl` 的 round 事件里 `before/after` 是**指标字典** `{mae, rmse}`，另有一个 score 空间的 `gain`。
    42 个 run 里一半是 11-14 之前跑的 v1，只有 journal；只认 CSV 的话那些 run 会打印一张空表，
    看起来像"实验没跑"，其实只是工具没读对地方。缺失就留 None，表里显示 "-"，
    **不能填 0**——0 会被读成"这一段没退步"。
    """
    profile = data["profile"]
    key = main_metric(profile)
    flip = 1.0 if profile == "A" else -1.0
    rows = []
    if data["_rounds"]:
        for row in data["_rounds"]:
            def pick(name):
                value = row.get(name)
                try:
                    return float(value) * flip
                except (TypeError, ValueError):
                    return None
            rows.append({"round": row.get("round"), "candidate": row.get("candidate") or "-",
                         "epochs": row.get("epochs"), "before": pick("score_before"),
                         "after": pick("score_after"),
                         "accepted": str(row.get("accepted")).lower() == "true",
                         "worst_stage_gain": _num(row.get("worst_stage_gain")),
                         "seconds": _num(row.get("train_seconds"))})
        return rows, "swanlab_export/rounds.csv"

    journal = data["_dir"] / "journal.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") != "round":
                continue
            before, after = event.get("before"), event.get("after")
            rows.append({
                "round": event.get("round"), "candidate": event.get("candidate") or "-",
                "epochs": event.get("epochs"),
                "before": float(before[key]) if isinstance(before, dict) and key in before else None,
                "after": float(after[key]) if isinstance(after, dict) and key in after else None,
                "accepted": bool(event.get("accepted")),
                "worst_stage_gain": _num(event.get("worst_stage_gain")),
                "seconds": _num(event.get("train_seconds"))})
    return rows, "journal.jsonl"


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detail(run_dir: Path) -> int:
    data = load(run_dir)
    if data is None:
        # 探针 / 被中断的 run：只有 journal.jsonl，没有汇总。不猜 profile、不编判读，
        # 但也不能只甩一句"没有"——把能确定的事实说清楚，并告诉他去哪看结论。
        journal = run_dir / "journal.jsonl"
        kinds: dict = {}
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kinds[event.get("kind", "?")] = kinds.get(event.get("kind", "?"), 0) + 1
        print("✗ {} 没有 results.json（不是坏数据，是这个目录本来就没跑完，或只是探针）".format(run_dir))
        if kinds:
            print("  journal.jsonl 有 {} 个事件：{}".format(
                sum(kinds.values()), ", ".join("{}×{}".format(k, v) for k, v in sorted(kinds.items()))))
        print("  这类目录不参与判读。要看结论请用主实验目录（总览：`python scripts/read_runs.py`）；"
              "宽度/阶段探针的结论已经写成文档：docs/A最差阶段-不可达性论证.md、docs/漂移监测基准.md")
        return 1
    profile = data["profile"]
    unit = "F1 点数" if profile == "A" else ("分钟" if profile == "B" else "循环")
    print("=== {}（profile {}，{}，{} 轮，{}）===".format(
        run_dir.name, profile, data["metric"], data["rounds"], "v2" if data.get("gate_v2") else "v1"))
    base, best = real(profile, data["baseline"]), real(profile, data["best"])
    print("基线 {} → 交付 {}，改进幅度 {:+.4f} {}（相对 {:+.1f}%）".format(
        base, best, improves(profile, base, best), unit,
        100.0 * improves(profile, base, best) / base if base else 0.0))
    mark, why = verdict(data)
    print("判读 {} {}".format(mark, why))
    rows, source = rounds_of(data)
    print("每轮明细（真实指标：A=Macro-F1，B/C=RMSE；来源 {}）。"
          "CSV 里的 score_* 是取过负号的统一方向，本表已还原，别拿 CSV 的负数当『变差』读。".format(source))
    print("{:<4} {:<14} {:>9} {:>9} {:>9} {:>5} {:>9} {:>9} {:>6}".format(
        "轮", "候选", "轮前", "轮后", "改进", "采纳", "最差阶段", "耗时s", "epoch"))
    for row in rows:
        before, after = row["before"], row["after"]
        delta = "-" if (before is None or after is None) else \
            "{:+.4f}".format(improves(profile, before, after))
        def fmt(value, pattern="{:.4f}"):
            return "-" if value is None else pattern.format(value)
        print("{:<4} {:<14} {:>9} {:>9} {:>9} {:>5} {:>9} {:>9} {:>6}".format(
            row["round"], row["candidate"], fmt(before), fmt(after), delta,
            "是" if row["accepted"] else "否", fmt(row["worst_stage_gain"]),
            fmt(row["seconds"], "{:.3f}"), row["epochs"] if row["epochs"] not in (None, "") else "-"))
    stages = data.get("per_stage") or []
    if isinstance(stages, list) and stages and isinstance(stages[0], dict):
        # 字段是 score_gain（已是"越大越好"方向：正=比 static 好），不是 gain——
        # 读不存在的键会安静地给 0，那比报错更坏。
        print("三段（与 static 基线逐段对照；『最差阶段 10 分』看这一段的最小值）：")
        gains = [float(st.get("score_gain") or 0.0) for st in stages]
        worst_i = gains.index(min(gains)) if gains else -1
        for i, st in enumerate(stages):
            print("  段{} 行{}–{} static={} ours={} 改进 {:+.4f}{}".format(
                st.get("stage", i + 1), *(st.get("rows") or [0, 0]),
                real(profile, st["static"]), real(profile, st["ours"]), gains[i],
                "　← 最差段" if i == worst_i else ""))
        print("  最小段改进 = {}（results.json 的 worst_stage_gain = {}）".format(
            round(min(gains), 4), round(float(data.get("worst_stage_gain") or 0.0), 4)))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=None, help="runs/<name>；不给则出全部总览")
    parser.add_argument("--profile", default=None, choices=["A", "B", "C"])
    parser.add_argument("--no-write", action="store_true", help="只打印，不写 docs/实验总览.md")
    args = parser.parse_args(argv)

    if args.run:
        # 接受三种写法：runs/b_v2x、b_v2x、绝对路径。只拼 RUNS 的话，写 runs/b_v2x 会变成
        # runs/runs/b_v2x 然后报"没有 results.json"——读数据的工具自己先报错，最容易被误当成数据坏了。
        raw = Path(args.run)
        for cand in (raw, RUNS / raw, RUNS / raw.name):
            if cand.is_dir():
                return detail(cand)
        print("✗ 找不到 run 目录：{}（名字见总览表的 run 列）".format(args.run))
        return 1

    rows = [r for r in overview() if not args.profile or r["profile"] == args.profile]
    cols = ["判", "run", "profile", "轮", "口径", "基线", "结果", "改进幅度", "相对",
            "采纳", "最差阶段", "cost_units", "秒", "理由"]
    lines = ["# 实验总览（由 `python scripts/read_runs.py` 从 `runs/` 再生成，别手改）", "",
             "指标方向已还原成各自的真实口径：**A 的 Macro-F1 越大越好，B/C 的 RMSE 越小越好**；"
             "`改进幅度` 一律正数代表变好。`runs/*/swanlab_export/rounds.csv` 里的 "
             "`score_before/score_after` 是『越大越好』的统一方向（B/C 取过负号），不要直接读。", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    good = sum(1 for r in rows if r["判"] == "✓")
    suspect = sum(1 for r in rows if r["判"] == "⚠")
    bad = sum(1 for r in rows if r["判"] == "✗")
    lines += ["", "共 {} 个可读 run：✓ {}｜⚠ {}｜✗ {}｜其余是校验/探针类。".format(
        len(rows), good, suspect, bad),
        "判读规则：综合改善要过提交口径（A 0.002 F1 / B 5 分钟 / C 0.5 循环）且**最差阶段不退**；"
        "只降平均值但末段退步的（如旧版 `b_10r`）不能当提交值——这正是 v2 上重采样的原因。"]
    text = "\n".join(lines) + "\n"
    print(text)
    if not args.no_write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8", newline="\n")
        print("→", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
