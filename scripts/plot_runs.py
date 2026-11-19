"""把 runs/ 里的逐轮训练记录画成图（loss 曲线、每轮验证指标、三段增益），输出 PNG 到 docs/charts/。

    python scripts/plot_runs.py                     # 三个 v2 主实验 + 一张总览
    python scripts/plot_runs.py runs/b_v2x --dpi 200
    python scripts/plot_runs.py --all               # 顺带把 a_10r 这类 v1 主实验也画了

为什么要这个脚本：赛题要看"实验记录"，而 `round-XX/history.jsonl` 里其实**有**逐 epoch 的
loss 与验证指标（官方训练器自己写的），只是没人画图，队友只能看 CSV 表格。图放 PNG 而不是
交互 HTML，是因为要进作品文档/PDF/PPT，必须能被静态引用。

三张图的读法（都是官方证据，不是我另算的口径）：
1. **每轮 loss 轨迹**：横轴是"轮内 epoch"，一条线 = 一轮。v2 一轮有三条轨迹（batch_size 阶梯），
   淡线全画、**下中位那条加粗**——因为我们交付的就是下中位轨迹，不是最好的那条（红线 C）。
2. **逐轮验证指标**：台阶 = 被采纳后的模型；空心叉 = 被拒的候选（它没成为下一轮起点）。
   这条线是"门禁在干活"最直观的证据：被拒的点画出来，才看得出我们没挑好看的。
3. **三段增益**：与 static 逐段对照，真实单位（B 分钟 / C 循环 / A F1 点数），最差段高亮——
   那是"最差阶段 10 分"看的东西。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                   # 无显示器环境（评测机/CI）也能出图
import matplotlib.pyplot as plt                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "charts"
MAIN = ["runs/a_v2x", "runs/b_v2x", "runs/c_v2x"]

# 本机没有 CJK 字体的 matplotlib 会把中文渲染成方框，图就成了假图。msyh/simhei 本机实测存在。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METRIC = {"A": ("macro_f1", "Macro-F1（越高越好）", 1.0)}
UNIT = {"B": ("rmse", "RMSE（分钟，越低越好）", -1.0), "C": ("rmse", "RMSE（循环，越低越好）", -1.0)}
PROFILE_METRIC = {"A": ("macro_f1", "Macro-F1", 1.0),
                  "B": ("rmse", "RMSE（分钟）", -1.0),
                  "C": ("rmse", "RMSE（循环）", -1.0)}


def metric_of(profile: str):
    key, label, sign = PROFILE_METRIC[profile]
    return key, label, sign


def score_to_real(profile: str, value: float) -> float:
    """CSV/journal 里 score 空间的数 → 真实指标。B/C 的 score 是 RMSE 取负，所以要翻回来。"""
    return value if profile == "A" else -value


def gain_to_improvement(profile: str, score_gain: float) -> float:
    """`per_stage[i].score_gain` **已经是"越大越好"空间**（正=比 static 好）。
    再乘一次符号就会把 +291 画成 −291、把最好的段标成"最差段"——本脚本第一版就栽在这里，
    所以这条转换单独定义并在图注里写明正负含义，不给调用方自由发挥的空间。"""
    return score_gain


def trajectories(round_dir: Path) -> list[Path]:
    """一轮里的 history.jsonl：v2 是 round-XX/t*/，v1 是 round-XX/history.jsonl。"""
    found = sorted(Path(p).parent for p in glob.glob(str(round_dir / "t*" / "history.jsonl")))
    if found:
        return found
    direct = round_dir / "history.jsonl"
    return [round_dir] if direct.is_file() else []


def read_history(path: Path) -> list[dict]:
    rows = []
    for line in (path / "history.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def plot_run(run_dir: Path, dpi: int = 140) -> list[Path]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    profile = results["profile"]
    key, label, _sign = metric_of(profile)
    lower_better = profile != "A"
    unit = label.replace("RMSE", "").strip("（）") or ""
    rounds = _rounds_csv(run_dir, results)
    round_dirs = sorted(p for p in run_dir.glob("round-*") if p.is_dir())

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    ax_loss, ax_val, ax_stage, ax_sum = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # ── 子图 1：每轮的 loss 轨迹 ────────────────────────────────────────────
    # 横轴用"轮内第 k 个 epoch"而不是 history 里的 epoch 字段：那个字段是**续训累计值**
    # （B 从 825 起、C 从 6630 起），直接画会把所有曲线挤到右上角，还看不出每轮内部的下降。
    for rd in round_dirs:
        trajs = trajectories(rd)
        finals = []
        for t in trajs:
            rows = read_history(t)
            if rows:
                finals.append((rows[-1]["validation"][key], t))
        # 排序取"下中位"：B/C 是 RMSE，越大越差，所以按真实值降序后取中间那条；
        # A 是 F1，越小越差，取升序的中间。这一步必须与门禁交付的口径一致，
        # 否则图上粗线标的是"最好那条"，而实际交付的是中位那条——那是给评审看错的图。
        # 统一排成 score 升序（最差在前）：RMSE 降序、F1 升序；再取下中位 (n-1)//2。
        # 偶数条时"下中位"是较差的那条——这正是门禁交付的规则，图上必须一致。
        finals.sort(key=lambda pair: (pair[0] if lower_better else -pair[0]), reverse=lower_better)
        median = finals[(len(finals) - 1) // 2][1] if finals else None
        for t in trajs:
            rows = read_history(t)
            if not rows:
                continue
            xs = list(range(1, len(rows) + 1))
            bold = (t == median)
            ax_loss.plot(xs, [r["loss"] for r in rows],
                         linewidth=2.6 if bold else 0.9,
                         color="crimson" if bold else "#9aa5b1",
                         alpha=1.0 if bold else 0.7,
                         label="本轮交付轨迹（下中位）" if bold and rd is round_dirs[0] else None)
    ax_loss.set_title("每轮的 loss 轨迹（一条线=一轮内的一个轨迹；粗红=该轮下中位交付轨迹）")
    ax_loss.set_xlabel("轮内第 k 个 epoch")
    ax_loss.set_ylabel("训练 loss（官方 history.jsonl 原值）")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.25)

    # ── 子图 2：逐轮验证指标（真实单位） ─────────────────────────────────────
    ax_val.plot([0], [results["baseline"][key]], marker="s", color="#7f8c8d", label="官方基线")
    prev = results["baseline"][key]
    # 按逐轮记录的**全部轮次**画，不按 round-XX 目录画：被治理层抑制的轮次（cancel/budget/hold）
    # 根本不产生目录，只按目录画会让"10 轮预算"在图上看起来只有 7 轮——那是给评审看的假象。
    seen_suppressed = False
    for row in sorted(rounds, key=lambda r: int(r["round"])):
        rnd = int(row["round"])
        rd = run_dir / "round-{:02d}".format(rnd)
        has_dir = rd.is_dir()
        after = score_to_real(profile, float(row["score_after"]))
        before = score_to_real(profile, float(row["score_before"]))
        if row["accepted"]:
            ax_val.plot([rnd - 1, rnd], [prev, after], color="#1f6f3f", linewidth=2.2)
            ax_val.plot([rnd], [after], marker="o", color="#1f6f3f", markersize=7,
                        label="采纳（成为下一轮起点）" if rnd == 2 else None)
            prev = after
        elif not has_dir:
            ax_val.plot([rnd], [before], marker="|", color="#8a8f98", markersize=12,
                        label="被抑制（本轮不训练，无产物目录）" if not seen_suppressed else None)
            seen_suppressed = True
        else:
            ax_val.plot([rnd], [after], marker="x", color="#c0392b", markersize=9,
                        label="被拒候选（没成为下一轮起点）" if rnd == 1 else None)
        for t in (trajectories(rd) if has_dir else []):
            rows = read_history(t)
            if rows:
                ax_val.plot([rnd] * len(rows), [r["validation"][key] for r in rows],
                            color="#9aa5b1", alpha=0.4, linewidth=0.9)
    ax_val.set_title("逐轮验证指标（绿线=采纳台阶；红叉=被拒候选；灰竖线=被抑制，共 {} 轮）".format(len(rounds) or len(round_dirs)))
    ax_val.set_xlabel("轮")
    ax_val.set_ylabel("{}{}".format("RMSE（" + unit + "）" if lower_better else "Macro-F1",
                                    "，越低越好" if lower_better else "，越高越好"))
    ax_val.set_xticks(range(0, (len(rounds) or len(round_dirs)) + 1))
    ax_val.legend(fontsize=8)
    ax_val.grid(alpha=0.25)

    # ── 子图 3：三段改进幅度 ───────────────────────────────────────────────
    stages = results.get("per_stage") or []
    if stages:
        gains = [gain_to_improvement(profile, float(s.get("score_gain") or 0.0)) for s in stages]
        labels = ["段{}（行 {}–{}）".format(s.get("stage", n + 1), (s.get("rows") or [0, 0])[0],
                                        (s.get("rows") or [0, 0])[1]) for n, s in enumerate(stages)]
        worst = min(range(len(gains)), key=lambda i: gains[i])
        ax_stage.bar(labels, gains, color=["#c0392b" if i == worst else "#2e7d32" for i in range(len(gains))])
        ax_stage.axhline(0, color="black", linewidth=0.8)
        for i, g in enumerate(gains):
            ax_stage.text(i, g, "{:+.4f}".format(g), ha="center",
                          va="bottom" if g >= 0 else "top", fontsize=9)
        ax_stage.set_title("三段改进幅度 vs static 基线（红=最差段；正数=比 static 好）")
        ax_stage.set_ylabel("改进幅度（{}）".format("分钟" if profile == "B" else
                                            "循环" if profile == "C" else "F1 点数"))
        ax_stage.grid(axis="y", alpha=0.25)
        for tick in ax_stage.get_xticklabels():
            tick.set_rotation(18)
    else:
        ax_stage.set_visible(False)

    # ── 子图 4：基线 vs 交付（真实指标，不取负） ─────────────────────────────
    base, best = results["baseline"][key], results["best"][key]
    bars = ax_sum.barh(["官方基线", "本系统交付"], [base, best], color=["#7f8c8d", "#2e7d32"])
    for b, v in zip(bars, [base, best]):
        ax_sum.annotate(" {:.4f}".format(v), (v, b.get_y() + b.get_height() / 2),
                        va="center", fontsize=10)
    adopted = sum(1 for r in rounds if r["accepted"])
    cost = (results.get("ledger") or {}).get("cost_units")
    rel = 100.0 * (base - best) / base if (lower_better and base) else (
        100.0 * (best - base) / base if base else 0.0)
    ax_sum.set_title("{}：{:.4f} → {:.4f}（{} {:.1f}%）｜采纳 {}/{}｜cost_units {}".format(
        profile, base, best, "误差降低" if lower_better else "提升", abs(rel),
        adopted, len(rounds) or len(round_dirs), "—" if cost is None else round(float(cost), 1)))
    ax_sum.set_xlabel("{}{}".format(label.split("（")[0], "（越低越好）" if lower_better else "（越高越好）"))
    ax_sum.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "{}.png".format(run_dir.name)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)

    # ── 单独一张：整体 loss 走势（横轴=续训累计 epoch，看"全程在降"） ──────────
    fig2, ax = plt.subplots(figsize=(11, 5.4))
    for rd in round_dirs:
        for t in trajectories(rd):
            rows = read_history(t)
            if rows:
                ax.plot([r["epoch"] for r in rows], [r["loss"] for r in rows],
                        linewidth=1.1, alpha=0.85,
                        label="round-{:02d}".format(int(rd.name.split("-")[1]))
                        if t == trajectories(rd)[0] else None)
    ax.set_title("{}：loss 随**累计 epoch** 的走势（横轴是续训不清零的 epoch 字段，一色=一轮）".format(run_dir.name))
    ax.set_xlabel("累计 epoch（续训保留，所以从父检查点的 epoch 起算）")
    ax.set_ylabel("训练 loss")
    ax.grid(alpha=0.25)
    if len(round_dirs) <= 12:
        ax.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    png2 = OUT / "{}_loss.png".format(run_dir.name)
    fig2.savefig(png2, dpi=dpi)
    plt.close(fig2)
    return [png, png2]


def _rounds_csv(run_dir: Path, results: dict) -> list[dict]:
    """逐轮的采纳情况：优先 rounds.csv；老 v1 run 没有就回读 journal 的 round 事件。"""
    import csv
    path = run_dir / "swanlab_export" / "rounds.csv"
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return [{"round": int(r["round"]), "accepted": r["accepted"] == "True",
                     "score_before": r["score_before"], "score_after": r["score_after"]}
                    for r in csv.DictReader(fh)]
    rows = []
    journal = run_dir / "journal.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") != "round":
                continue
            rows.append({"round": int(event["round"]), "accepted": bool(event.get("accepted")),
                         "score_before": event["before"][results["metric"]],
                         "score_after": event["after"][results["metric"]]})
    return rows


def plot_overview(runs: list[Path], dpi: int = 140) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for ax, run_dir in zip(axes, runs):
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        profile = results["profile"]
        key, label, _sign = metric_of(profile)
        base, best = results["baseline"][key], results["best"][key]        # 真实指标，不取负
        stages = [gain_to_improvement(profile, float(s.get("score_gain") or 0.0))
                  for s in results.get("per_stage") or []]
        adopted = _rounds_csv(run_dir, results)
        n_ok = sum(1 for r in adopted if r["accepted"])
        bars = ax.bar(["基线", "交付"], [base, best], color=["#7f8c8d", "#2e7d32"])
        for b, v in zip(bars, [base, best]):
            ax.text(b.get_x() + b.get_width() / 2, v, "{:.4f}".format(v), ha="center",
                    va="bottom", fontsize=9)
        rel = 100.0 * (base - best) / base if (profile != "A" and base) else (
            100.0 * (best - base) / base if base else 0.0)
        ax.set_title("profile {}｜{} {:.1f}%｜采纳 {}/{}｜最差段 {:+.4f}".format(
            profile, "误差降低" if profile != "A" else "提升", abs(rel), n_ok,
            len(adopted) or 1, min(stages) if stages else 0.0), fontsize=10)
        ax.set_ylabel("{}（{}）".format(key, "越低越好" if profile != "A" else "越高越好"))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("v2 主实验一览（数据源：runs/*/results.json 与官方 history.jsonl）", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "overview.png"
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=None, help="runs/<name>；不给则画三个 v2 主实验")
    parser.add_argument("--all", action="store_true", help="v1 主实验（a_10r/b_w2_10/c_10r）也画")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)

    targets: list[Path] = []
    if args.run:
        raw = Path(args.run)
        for cand in (raw, RUNS / raw, RUNS / raw.name):
            if (cand / "results.json").is_file():
                targets = [cand]
                break
        if not targets:
            print("✗ 找不到带 results.json 的 run：{}".format(args.run))
            return 1
    else:
        targets = [RUNS / name.split("/")[-1] for name in MAIN]
        if args.all:
            targets += [RUNS / n for n in ("a_10r", "b_w2_10", "c_10r")]
        targets = [t for t in targets if (t / "results.json").is_file()]

    made: list[Path] = []
    for run_dir in targets:
        made += plot_run(run_dir, dpi=args.dpi)
    if len(targets) > 1:
        made.append(plot_overview(targets, dpi=args.dpi))
    for path in made:
        print("✓ {}（{:.0f} KiB）".format(path, path.stat().st_size / 1024))
    print("→ 图目录：{}".format(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
