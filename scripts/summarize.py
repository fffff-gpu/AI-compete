"""汇总 runs/ 下所有 results.json，输出一张对比表（Markdown）。"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def row_of(path: Path):
    try:
        r = json.loads((path / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    profile, metric = r["profile"], r["metric"]
    base, best = r["baseline"][metric], r["best"][metric]
    if metric == "rmse":
        delta = base - best
        rel = delta / base * 100
    else:
        delta = best - base
        rel = delta / base * 100
    wall, fallback = 0.0, 0
    journal = path / "journal.jsonl"
    if journal.exists():
        ev = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        stamps = [e["t"] for e in ev if "t" in e]
        wall = (max(stamps) - min(stamps)) if stamps else 0.0
        fallback = sum(1 for e in ev if e.get("kind") == "decision" and e.get("source") == "rule_fallback")
    return {
        "run": path.name, "profile": profile, "metric": metric,
        "baseline": base, "best": best, "delta": delta, "rel": rel,
        "accepted": r["accepted_rounds"], "rounds": r["rounds"],
        "worst": r["worst_stage_gain"], "seconds": r["train_seconds_total"],
        "wall": wall, "policy": r["policy"], "fallback": fallback,
        "llm": (r.get("llm") or {}).get("calls", 0),
        "tokens": (r.get("llm") or {}).get("tokens_in", 0) + (r.get("llm") or {}).get("tokens_out", 0),
    }


def main():
    rows = [row_of(p) for p in sorted(RUNS.iterdir()) if (p / "results.json").exists()]
    rows = [r for r in rows if r]
    if not rows:
        print("没有可汇总的 run")
        return 1
    print("| run | 数据 | 指标 | 基线 | 最好 | 增益 | 相对 | 采纳/轮 | 最差阶段 | 训练秒 | 策略 | LLM调用 | token |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        fmt = "{:.4f}" if r["metric"] == "macro_f1" else "{:.2f}"
        print(f"| {r['run']} | {r['profile']} | {r['metric']} | {fmt.format(r['baseline'])} | "
              f"{fmt.format(r['best'])} | {fmt.format(r['delta'])} | {r['rel']:+.1f}% | "
              f"{r['accepted']}/{r['rounds']} | {r['worst']:+.4f} | {r['seconds']:.1f} | "
              f"{r['policy']} | {r['llm']} | {r['tokens']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())