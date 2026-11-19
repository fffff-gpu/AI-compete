"""A 的三段各自对配方的反应（离线探针，不进闭环）。

    python scripts/probe_a_stages.py

为什么要有这个探针：闭环门禁现在的采纳条件是「总体增益 ≥ min_gain 且最差段不退」，
也就是说**任何「总体微跌、但最差段大涨」的候选都会被拒**。而客观分里
「最差阶段的归一化增益」是单独一档（10 分里的一半）。如果 A 存在这种候选，
那问题不在配方，而在我们的目标函数没对齐评分函数——这必须先有数据才能判断。

每个候选都从官方初始检查点续训（同一段 public/train，validation 只用于评估），
报三段的 macro-F1、stage2 的 Acetaldehyde 召回、stage3 的「该段不存在的类被预测了几次」。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import baseline                                  # noqa: E402
from harness.policy import candidates_for                     # noqa: E402
from harness.run import ref_stages, score_of                  # noqa: E402

PROFILE = "A"
EPOCH_MULTIPLIERS = (1, 2)


def stage_report(model, val, stages, classes):
    out = []
    for index, (a, b) in enumerate(stages, start=1):
        slice_val = {"x": val["x"][a:b], "y": val["y"][a:b]}
        metrics = baseline.metrics_model(PROFILE, model, slice_val)
        predicted = model.predict(val["x"][a:b]).argmax(axis=1)
        present = {int(k) for k in slice_val["y"]}
        absent_hits = {classes[int(k)]: int(count) for k, count in
                       Counter(predicted.tolist()).items() if int(k) not in present}
        recall = {}
        for label in sorted(present):
            mask = slice_val["y"] == label
            if mask.sum():
                recall[classes[label]] = float(np.mean(predicted[mask] == label))
        out.append({"stage": index, "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"], "classes_present": len(present),
                    "absent_class_predictions": absent_hits, "recall": recall})
    return out


def main() -> int:
    cfg = baseline.config(PROFILE)
    classes = cfg["classes"]
    val = baseline.dataset(PROFILE, "validation")
    stages = ref_stages(len(val["y"]))
    base_ckpt = baseline.baseline_checkpoint(PROFILE)
    base_model = baseline.load_model(PROFILE, base_ckpt)
    base_report = stage_report(base_model, val, stages, classes)
    base_full = baseline.metrics(PROFILE, base_ckpt, val)
    print(f"官方基线：全量 macro_f1={base_full['macro_f1']:.4f}")
    for row in base_report:
        print(f"  stage{row['stage']} f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f} "
              f"类别数={row['classes_present']} 不存在的类被预测={row['absent_class_predictions']}")
        print(f"    召回={ {k: round(v, 3) for k, v in row['recall'].items()} }")

    results = []
    work = ROOT / "runs" / "_probe_a_stages"
    for candidate in candidates_for(cfg["task"]):
        for multiplier in EPOCH_MULTIPLIERS:
            epochs = max(1, min(12, int(candidate.get("epochs", 5)) * multiplier))
            tag = f"{candidate['name']}-x{multiplier}"
            out_dir = work / tag.replace("/", "-")
            if out_dir.exists():
                import shutil

                shutil.rmtree(out_dir)
            try:
                baseline.continue_round(
                    PROFILE, out_dir, base_ckpt, epochs=epochs,
                    learning_rate=float(candidate.get("lr", 0.003)),
                    wd=float(candidate.get("wd", 0.0)), ls=float(candidate.get("ls", 0.0)),
                    jitter=float(candidate.get("jitter", 0.0)), clip=float(candidate.get("clip", 5.0)),
                    weights=candidate.get("weights"), stage_id=tag)
            except Exception as exc:                              # noqa: BLE001 探针要跑完整个网格
                print(f"{tag:24s} 训练失败 {type(exc).__name__}: {exc}")
                continue
            ckpt = out_dir / "checkpoint.npz"
            model = baseline.load_model(PROFILE, ckpt)
            full = baseline.metrics(PROFILE, ckpt, val)
            report = stage_report(model, val, stages, classes)
            gains = [row["macro_f1"] - base_report[i]["macro_f1"] for i, row in enumerate(report)]
            row_out = {"recipe": tag, "epochs": epochs, "weights": candidate.get("weights"),
                       "overall_gain": full["macro_f1"] - base_full["macro_f1"],
                       "stage_gains": gains, "worst_stage_gain": min(gains),
                       "stages": [{"f1": round(r["macro_f1"], 4),
                                   "absent_pred": sum(r["absent_class_predictions"].values()),
                                   "acetalddehyde_recall": r["recall"].get("Acetaldehyde")}
                                  for r in report]}
            results.append(row_out)
            flag = "  ← 总体跌但最差段涨" if (row_out["overall_gain"] < 0 <= row_out["worst_stage_gain"]) else ""
            print(f"{tag:24s} overall={row_out['overall_gain']:+.4f} "
                  f"stages={['%+.4f' % g for g in gains]} worst={row_out['worst_stage_gain']:+.4f}{flag}")
    (work / "probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    positive_worst = [item for item in results if item["worst_stage_gain"] > 0]
    print(f"\n共 {len(results)} 个候选；最差段有正增益的 {len(positive_worst)} 个："
          f"{[item['recipe'] for item in positive_worst][:8]}")
    print(f"→ {work / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
