"""对照：开/关闭环控制器（--control）的十个指标。

用法（cwd = official/）：
  python scripts/compare_control.py A a_10r a_ctrl
  python scripts/compare_control.py B b_v2 b_ctrl
  python scripts/compare_control.py C c_v2 c_ctrl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTION = {"A": 1.0, "B": -1.0, "C": -1.0}
METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}


def load(name):
    path = ROOT / "runs" / name / "results.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def chatter(gains):
    signs = [1 if g > 0 else -1 for g in gains if abs(g) > 1e-12]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def stats(profile, res):
    metric = METRIC[profile]
    d = DIRECTION[profile]
    base = d * res["baseline"][metric]
    best = d * res["best"][metric]
    gains = [d * r["gain"] if not isinstance(r["gain"], dict) else 0.0
             for r in res["round_history"]]
    # round_history 里的 gain 已经是「越大越好」方向
    gains = [r["gain"] for r in res["round_history"]]
    acc = [r for r in res["round_history"] if r["accepted"]]
    rounding = sum(1 for r in res["round_history"] if r["gain"] < 0)
    return {
        "baseline": res["baseline"][metric],
        "best": res["best"][metric],
        "gain": best - base,
        "accepted": len(acc),
        "rejected": len(res["round_history"]) - len(acc),
        "wasted": rounding,
        "chatter": chatter(gains),
        "gain_sd": sd(gains),
        "worst_stage": res["worst_stage_gain"],
        "per_accepted": (best - base) / len(acc) if acc else 0.0,
        "train_seconds": res.get("train_seconds_total", 0.0),
        "control": res.get("control"),
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    profile, off_name, on_name = argv[1], argv[2], argv[3]
    off, on = stats(profile, load(off_name)), stats(profile, load(on_name))
    metric = METRIC[profile]
    print(f"== profile {profile}  metric={metric}  direction={'越大越好' if DIRECTION[profile] > 0 else '越小越好'}")
    rows = [
        ("run", off_name, on_name),
        (metric, f"{off['baseline']:.4f} → {off['best']:.4f}", f"{on['baseline']:.4f} → {on['best']:.4f}"),
        ("增益", f"{off['gain']:+.4f}", f"{on['gain']:+.4f}"),
        ("采纳轮", f"{off['accepted']}/10", f"{on['accepted']}/10"),
        ("被拒轮", f"{off['rejected']}", f"{on['rejected']}"),
        ("负增益轮（算力白花）", f"{off['wasted']}", f"{on['wasted']}"),
        ("抖振（符号翻转）", f"{off['chatter']}", f"{on['chatter']}"),
        ("增益标准差", f"{off['gain_sd']:.4f}", f"{on['gain_sd']:.4f}"),
        ("最差阶段增益", f"{off['worst_stage']:+.4f}", f"{on['worst_stage']:+.4f}"),
        ("每次采纳的增益", f"{off['per_accepted']:+.4f}", f"{on['per_accepted']:+.4f}"),
        ("训练秒", f"{off['train_seconds']:.1f}", f"{on['train_seconds']:.1f}"),
    ]
    if on["control"]:
        rows.append(("死区终值", "—", f"{on['control']['deadband_final']:.4f}"))
        mc = on["control"]["mode_counts"]
        rows.append(("exploit/probe/hold", "—",
                     f"{mc['exploit']}/{mc['probe']}/{mc['hold']}"))
        rows.append(("hold 轮（省算力）", "—", str(on["control"]["held_rounds"] or "无")))
    w0 = max(len(r[0]) for r in rows) + 2
    for a, b, c in rows:
        print(f"  {a:<{w0}}{b:<26}{c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))