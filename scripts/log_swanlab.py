"""把已跑完的 run 回填成 SwanLab 离线记录，之后再 sync 上云。

    python scripts/log_swanlab.py                      # 默认：所有有 results.json 的正式实验
    python scripts/log_swanlab.py runs/a_10r runs/b_w2_10
    python scripts/log_swanlab.py --list               # 只看会回填哪些，不写

为什么需要它：SwanLab 记录层（`harness/records.py`）是 11-13 深夜才接进主流程的，
之前跑的那 30 多个实验（v1 主实验、门禁消融、控制器消融、种子方差、LLM 臂对照）
都只有本地 journal。赛规门槛②要求「全部训练、微调、调整与评估实验」都完整记录，
所以这批历史实验必须回填，不能只交今晚之后跑的那五个。

回填内容与 records.py 同字段口径（配方、前后分数、增益、是否采纳、最差阶段、耗时、
token、成本单位），因此云端看到的字段和新 run 一致；每轮的行数等于 round_history 长度。

产出 `swanlog/index.json`：实验名 → 离线目录，供 sync 后把云端链接逐条对上。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROJECT = "university_AI_innovate"
SKIP_PREFIX = ("_",)          # _smoke/_probe/_parity/_rec/_idtest 这些一次性目录不上云


def discover(argv_runs):
    if argv_runs:
        return [Path(r).resolve() for r in argv_runs]
    runs = []
    for entry in sorted((ROOT / "runs").iterdir()):
        if not (entry / "results.json").is_file():
            continue
        if entry.name.startswith(SKIP_PREFIX):
            continue
        runs.append(entry)
    return runs


def round_rows(res):
    """把 round_history（v2 时内嵌 gate 裁决）合成同一条记录行。"""
    ledger = res.get("ledger") or {}
    rows = []
    for i, row in enumerate(res.get("round_history") or [], start=1):
        g = (row.get("gate") or {})
        metric = res.get("metric")
        rows.append({
            "round": row.get("round", i),
            "candidate": row.get("candidate"),
            "source": row.get("source"),
            "epochs": row.get("epochs", 0),
            "score": row.get("score"),
            "gain": row.get("gain", 0.0),
            "accepted": 1 if row.get("accepted") else 0,
            "suppressed": 1 if row.get("suppressed") else 0,
            "metric_value": (row.get("full") or {}).get(metric),
            "gain_lcb": g.get("gain_lcb"),
            "worst_stage_lcb": g.get("worst_stage_lcb"),
            "zone": {"clear_win": 2.0, "ambiguous": 1.0, "clear_loss": 0.0}.get(g.get("zone")),
            "cost_units": ledger.get("cost_units"),
            "labels": (ledger.get("used") or {}).get("labels"),
            "tokens": ledger.get("tokens"),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="要回填的 run 目录（默认全部正式实验）")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args(argv)

    targets = discover(args.runs)
    if not targets:
        print("没有可回填的 run 目录")
        return 1
    if args.list:
        for entry in targets:
            print("  ", entry.name)
        print(f"共 {len(targets)} 个")
        return 0

    import swanlab

    index = {}
    for entry in targets:
        res = json.loads((entry / "results.json").read_text(encoding="utf-8"))
        metric = res.get("metric")
        profile = res.get("profile", entry.name[:1].upper())
        name = f"{profile.lower()}-{entry.name}"
        existing = sorted((entry / "_swanlog").glob("run-*"))
        if existing:
            index[name] = {"run_dir": str(existing[-1]), "source": str(entry),
                           "skipped": "已有实时记录，用实时那份上云", "rows": len(rows),
                           "best": (res.get("best") or {}).get(metric),
                           "baseline": baseline_value,
                           "worst_stage_gain": res.get("worst_stage_gain")}
            print(f"  {name:22s} 跳过回填（实时记录在 {existing[-1].name}）")
            continue
        baseline_value = (res.get("baseline") or {}).get(metric)
        rows = round_rows(res)
        run = swanlab.init(
            project=args.project,
            name=name,
            mode="offline",
            log_dir=str(entry / "_swanlog"),
            config={
                "profile": profile, "task": res.get("task"), "metric": metric,
                "rounds": res.get("rounds"), "policy": res.get("policy"),
                "control": res.get("control_on"), "gate": "v2" if res.get("gate_v2") else "v1",
                "min_gain": res.get("min_gain"), "seed": res.get("seed"),
                "baseline": baseline_value,
                "parent_source": res.get("parent_source"),
                "worst_tol": res.get("worst_tol"),
            },
            tags=[f"profile-{profile}", "gate-" + ("v2" if res.get("gate_v2") else "v1")],
        )
        for row in rows:
            swanlab.log({k: v for k, v in row.items() if isinstance(v, (int, float)) and
                         not isinstance(v, bool)}, step=row["round"])
        swanlab.log({
            "summary/best": (res.get("best") or {}).get(metric),
            "summary/baseline": baseline_value,
            "summary/accepted_rounds": res.get("accepted_rounds"),
            "summary/worst_stage_gain": res.get("worst_stage_gain"),
            "summary/train_seconds_total": res.get("train_seconds_total"),
        })
        swanlab.finish()
        inner = sorted((entry / "_swanlog").glob("run-*"))
        logdir = str(inner[-1]) if inner else ""
        if not logdir:
            print(f"  !! {name} 回填后找不到离线目录")
        index[name] = {"run_dir": logdir, "source": str(entry), "rows": len(rows),
                       "best": (res.get("best") or {}).get(metric),
                       "baseline": baseline_value,
                       "worst_stage_gain": res.get("worst_stage_gain")}
        print(f"  {name:22s} {len(rows):3d} 轮 → {logdir}")

    outdir = ROOT / "swanlog"
    outdir.mkdir(exist_ok=True)
    (outdir / "backfill_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n回填 {len(index)} 个实验；索引 → {outdir / 'backfill_index.json'}")
    print("下一步（已 swanlab login 的窗口里）：")
    print("  python -m swanlab sync " + " ".join(sorted({v["run_dir"] for v in index.values()})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
