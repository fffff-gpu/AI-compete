import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402

rows = [summarize.row_of(p) for p in sorted(summarize.RUNS.iterdir()) if (p / "results.json").exists()]
rows = [r for r in rows if r]
print("| run | 数据 | 指标 | 基线 | 最好 | 增益 | 相对 | 采纳 | 最差阶段 | 训练秒 | 墙钟 | 策略 | 降级 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    fmt = "{:.4f}" if r["metric"] == "macro_f1" else "{:.2f}"
    print("| {} | {} | {} | {} | {} | {} | {:+.1f}% | {}/{} | {:+.4f} | {:.1f} | {:.0f}s | {} | {} |".format(
        r["run"], r["profile"], r["metric"], fmt.format(r["baseline"]), fmt.format(r["best"]),
        fmt.format(r["delta"]), r["rel"], r["accepted"], r["rounds"], r["worst"],
        r["seconds"], r["wall"], r["policy"], r["fallback"]))