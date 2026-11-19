"""A 的最差阶段为什么是 0：按时间切片看类别构成与混淆，再决定动哪个配方。

    python scripts/diagnose_a.py

动机：客观分里「最差阶段」这一档 A 拿到的是 0.0000（stage2、stage3 都没有增益），
而三段是 `validation.jsonl` 的**时间切片**（run.py 的 stages），不是随机划分。
gas drift 数据每个时段只有部分气体出现，所以最差阶段的瓶颈很可能不是「模型不够准」，
而是「类别构成变了 + macro-F1 对预测出该段不存在的类一样罚」。
先把这件事测出来，再决定是 class-balanced 采样、末端加权、还是权重平均。
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import baseline                                  # noqa: E402
from harness.run import METRIC, ref_stages                    # noqa: E402


def centroid_evidence() -> None:
    """结构证据：在 checkpoint 冻结的归一化空间里，漂移后的 Acetaldehyde 离谁更近。

    如果 stage2 的 Acetaldehyde 行在归一化空间里整体落到 Ethanol 质心一侧，
    那么「续训只能移动决策边界，改不了这个区域的可分性」就不是借口，而是可测量的事实：
    要修它必须重拟合 mean/std 或加特征，两样都被红线 B 禁掉。
    """
    import numpy as np

    cfg = baseline.config("A")
    classes = cfg["classes"]
    val = baseline.dataset("A", "validation")
    train = baseline.dataset("A", "train")
    ckpt = baseline.baseline_checkpoint("A")
    with np.load(ckpt, allow_pickle=False) as data:      # mean/scale 是独立张量，不在 metadata 里
        mean, scale = np.asarray(data["mean"]), np.asarray(data["scale"])
    z = lambda m: np.clip((m - mean) / scale, -10, 10)          # 官方 model.py 的归一化
    train_z, val_z = z(train["x"]), z(val["x"])
    centroids = {label: train_z[train["y"] == index].mean(axis=0)
                 for index, label in enumerate(classes) if np.any(train["y"] == index)}
    stage2 = val_z[622:933]
    stage2_y = val["y"][622:933]
    mask = stage2_y == classes.index("Acetaldehyde")
    rows = stage2[mask]
    if not len(rows) or "Acetaldehyde" not in centroids:
        print("· 质心分析跳过：类别缺失")
        return
    distances = {label: np.linalg.norm(rows - centre, axis=1) for label, centre in centroids.items()}
    stacked = np.vstack([distances[label] for label in centroids])
    winner = np.array(list(centroids))[stacked.argmin(axis=0)]
    counts = Counter(winner.tolist())
    print(f"· 冻结归一化空间里，stage2 的 {int(mask.sum())} 个 Acetaldehyde 行按最近质心归属：")
    print(f"  {dict(counts)}")
    early = z(train["x"][train["y"] == classes.index("Acetaldehyde")])
    print(f"  训练窗 Acetaldehyde 质心 ↔ stage2 Acetaldehyde 均值 的距离 = "
          f"{float(np.linalg.norm(centroids['Acetaldehyde'] - rows.mean(axis=0))):.2f}，"
          f"而 Ethanol 质心 ↔ stage2 均值 = "
          f"{float(np.linalg.norm(centroids['Ethanol'] - rows.mean(axis=0))):.2f}")


def cumulative_balanced(rounds: int = 8) -> None:
    """累计续训能不能把 Acetaldehyde 的召回抬起来（每次 5 epoch，class-balanced）。"""
    import numpy as np

    out = ROOT / "runs" / "_probe_a_cumulative"
    classes = baseline.config("A")["classes"]
    val = baseline.dataset("A", "validation")
    ckpt = baseline.baseline_checkpoint("A")
    for index in range(1, rounds + 1):
        step_dir = out / f"step-{index:02d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        report = baseline.continue_round("A", step_dir, ckpt, epochs=5, learning_rate=0.003,
                                         wd=0.0, ls=0.0, jitter=0.0, clip=5.0,
                                         weights="class_balanced", stage_id=f"cum-{index}")
        ckpt = step_dir / "checkpoint.npz"
        model = baseline.load_model("A", ckpt)
        full = baseline.metrics("A", ckpt, val)
        slice_val = {"x": val["x"][622:933], "y": val["y"][622:933]}
        metrics = baseline.metrics_model("A", model, slice_val)
        predicted = model.predict(slice_val["x"]).argmax(axis=1)
        mask = slice_val["y"] == classes.index("Acetaldehyde")
        recall = float(np.mean(predicted[mask] == classes.index("Acetaldehyde"))) if mask.sum() else float("nan")
        print(f"  累计 {index * 5:3d} epoch：全量 macro_f1={full['macro_f1']:.4f} "
              f"stage2 macro_f1={metrics['macro_f1']:.4f} Acetaldehyde 召回={recall:.3f} "
              f"steps={report['final_steps']}")


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "A"
    if "--structure" in sys.argv:
        centroid_evidence()
        print("\n· 累计 class-balanced 续训（每步 5 epoch，全部只读 public/train）：")
        cumulative_balanced()
        return 0
    cfg = baseline.config(profile)
    classes = cfg.get("classes") or []
    val = baseline.dataset(profile, "validation")
    n_val = len(val["y"])
    stages = ref_stages(n_val)
    ckpt = baseline.baseline_checkpoint(profile)
    model = baseline.load_model(profile, ckpt)
    predicted = model.predict(val["x"]).argmax(axis=1) if classes else None
    metric = METRIC[profile]

    print(f"profile {profile}　metric={metric}　验证 {n_val} 行　段={stages}")
    if classes:
        train = baseline.dataset(profile, "train")
        print(f"训练集类别构成：{dict(Counter(str(v) for v in train['y']))}")
    for index, (a, b) in enumerate(stages, start=1):
        y = val["y"][a:b]
        counts = Counter(str(v) for v in y)
        row = {"stage": index, "rows": b - a, "classes_present": len(counts),
               "composition": dict(counts)}
        if classes:
            labels = predicted[a:b]
            present = {classes[int(k)] for k in y}
            absent_preds = Counter(classes[int(k)] for k in labels)
            wrong_absent = {name: int(count) for name, count in absent_preds.items()
                            if name not in present}
            row["macro_f1"] = float(baseline.metrics_model(
                profile, model, {"x": val["x"][a:b], "y": y})[metric])
            row["acc"] = float(np.mean(labels == y))
            row["预测但该段不存在的类"] = wrong_absent
            row["该段有但没预测出的类"] = sorted(
                present - {classes[int(k)] for k in labels})
        else:
            row["rmse"] = float(baseline.metrics_model(
                profile, model, {"x": val["x"][a:b], "y": y})[metric])
        print(json.dumps(row, ensure_ascii=False))
    print("\n读法：如果某段 classes_present 远小于总类别数，而「预测但该段不存在的类」非空，"
          "那最差阶段的损失主要来自**类别构成漂移**，"
          "对症的配方是 class-balanced 采样（让少数类不被淹没）+ 末端加权，"
          "而不是继续加正则。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
