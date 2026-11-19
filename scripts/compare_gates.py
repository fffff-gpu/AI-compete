"""v1 / v2 同数据同轮数对比表（交接说明 §5.2 的 P0 验收项）。

    python scripts/compare_gates.py
    python scripts/compare_gates.py --v1 runs/a_10r,runs/b_w2_10,runs/c_10r \
                                   --v2 runs/a_v2x,runs/b_v2x,runs/c_v2x

表里刻意并排三样东西，因为它们对应不同的分数：
- 综合性能（客观分主项）与最差阶段增益（10 分里的「最差阶段」档）；
- 裁决次数 / 抑制次数 / 三区分布 / 可检出轮数（说明门禁不是在演戏）；
- cost_units 与 unit_gain（客观分的分母是算力，多轨迹是有代价的，必须摊开写）。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path / "results.json").read_text(encoding="utf-8"))


def row(res: dict, label: str) -> dict:
    metric = res["metric"]
    direction = 1.0 if metric == "macro_f1" else -1.0
    delta = direction * (res["best"][metric] - res["baseline"][metric])
    rel = 100.0 * delta / max(abs(res["baseline"][metric]), 1e-12)
    gate = res.get("gate_v2") or {}
    ledger = res.get("ledger") or {}
    lineage = res.get("lineage") or {}
    suppressed = gate.get("rounds_suppressed") or {}
    return {"profile": res["profile"], "label": label, "metric": metric,
            "baseline": res["baseline"][metric], "best": res["best"][metric],
            "rel": rel, "accepted": f"{res['accepted_rounds']}/{res['rounds']}",
            "worst": res["worst_stage_gain"], "train_seconds": res["train_seconds_total"],
            "adjudicated": gate.get("rounds_adjudicated"), "suppressed": sum(suppressed.values()),
            "zones": gate.get("zones"), "detectable": gate.get("detectable_rounds"),
            "cost_units": ledger.get("cost_units"),
            "unit_gain": gate.get("unit_gain"),
            "labels": (ledger.get("used") or {}).get("labels"),
            "rollbacks": gate.get("rollbacks"),
            "chain": ("ok" if lineage.get("ok") else ("缺失" if not lineage else "BROKEN")),
            "events": lineage.get("events")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", default="runs/a_10r,runs/b_w2_10,runs/c_10r")
    parser.add_argument("--v2", default="runs/a_v2x,runs/b_v2x,runs/c_v2x")
    parser.add_argument("--out", default=None, help="把表写到文件（默认只打印）")
    args = parser.parse_args(argv)

    rows = []
    for tag, spec in (("v1", args.v1), ("v2", args.v2)):
        for item in [s.strip() for s in spec.split(",") if s.strip()]:
            try:
                rows.append(row(load(item), tag))
            except (OSError, ValueError, KeyError) as exc:
                print(f"· 跳过 {item}：{type(exc).__name__}: {exc}")

    lines = ["| profile | 门禁 | 指标 | 静态基线 | 本系统 | 相对 | 采纳 | 最差阶段增益 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['profile']} | {r['label']} | {r['metric']} | {r['baseline']:.4f} | "
                     f"{r['best']:.4f} | {r['rel']:+.1f}% | {r['accepted']} | {r['worst']:+.4f} |")
    lines += ["",
              "| profile | 门禁 | 裁决/抑制 | 三区分布 | 可检出轮 | 证据链 | 训练秒 | 标注行 | cost_units | unit_gain | 回滚 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        zones = "—"
        if r["zones"]:
            zones = "win {} / amb {} / loss {}".format(
                r["zones"].get("clear_win", 0), r["zones"].get("ambiguous", 0),
                r["zones"].get("clear_loss", 0))
        adjudicated = "—" if r["adjudicated"] is None else r["adjudicated"]
        suppressed = "—" if r["suppressed"] is None else r["suppressed"]
        detectable = "—" if r["detectable"] is None else r["detectable"]
        chain = r["chain"] + ("（%d 事件）" % r["events"] if r["events"] else "")
        labels = int(r["labels"]) if r["labels"] else "—"
        cost = "—" if r["cost_units"] is None else r["cost_units"]
        unit = "—" if r["unit_gain"] is None else "%.6f" % r["unit_gain"]
        rollbacks = "—" if r["rollbacks"] is None else r["rollbacks"]
        lines.append("| {} | {} | {}/{} | {} | {} | {} | {:.1f} | {} | {} | {} | {} |".format(
            r["profile"], r["label"], adjudicated, suppressed, zones, detectable, chain,
            r["train_seconds"], labels, cost, unit, rollbacks))
    lines += ["",
              "读法：**v2 不是「更准」，是「更敢说」。** 同一份数据上 v2 会给出置信下界、"
              "把效应标成可检出/不可检出，并拒绝那些只是种子抖动的候选；代价写在最后两列"
              "（多轨迹使 cost_units 上升，unit_gain 相应下降）。两个口径都留在表里，"
              "评审要的是这个取舍被明示，而不是被藏起来。"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        (ROOT / args.out).write_text(text, encoding="utf-8")
        print(f"→ {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
