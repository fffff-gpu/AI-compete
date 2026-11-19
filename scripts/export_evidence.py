"""把散在 runs/ 里的关键证据导出成受版本控制的 JSON，放进 docs/evidence/。

    python scripts/export_evidence.py

为什么需要这个脚本：`runs/_probe_a_stages/` 这类一次性探针目录被 .gitignore 排除
（`_*probe*`），而《A最差阶段-不可达性论证》写着"原始数据在 probe.json"——
克隆仓库的评审根本拿不到那个文件。**引用一个不在仓库里的路径，等于假引用。**

所以证据要走两条腿：盘上的 run 目录照旧（本地全量），
仓库里放一份**由脚本从 run 再生成的 JSON 摘要**，评审要么用我们的摘要，
要么按文档里的命令自己重跑，两边都可核。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence"
MAIN_RUNS = {"A": {"v2": "runs/a_v2x", "v1": "runs/a_10r"},
             "B": {"v2": "runs/b_v2x", "v1": "runs/b_w2_10"},
             "C": {"v2": "runs/c_v2x", "v1": "runs/c_10r"}}
ABLATIONS = ["runs/a_nogate", "runs/b_nogate", "runs/b_worst", "runs/b_w2_nc",
             "runs/a_ctrl4", "runs/b_ctrl4", "runs/c_ctrl4", "runs/c_ctrl2",
             "runs/b_seed1", "runs/b_seed2", "runs/b_seed3",
             "runs/a_llm2", "runs/a_llm_v1fix", "runs/a_cb_v2"]


def load(rel: str) -> dict:
    return json.loads((ROOT / rel / "results.json").read_text(encoding="utf-8"))


def headline(rel: str) -> dict:
    res = load(rel)
    metric = res["metric"]
    direction = 1.0 if metric == "macro_f1" else -1.0
    gain = direction * (res["best"][metric] - res["baseline"][metric])
    gate = res.get("gate_v2") or {}
    ledger = res.get("ledger") or {}
    lineage = res.get("lineage") or {}
    out = {
        "run": rel, "profile": res["profile"], "metric": metric,
        "rounds": res["rounds"], "policy": res["policy"],
        "gate": "v2" if gate else "v1",
        "control": bool(res.get("control_on")),
        "baseline": res["baseline"][metric], "best": res["best"][metric],
        "improvement_signed": round(gain, 6),
        "improvement_relative_pct": round(100 * gain / abs(res["baseline"][metric]), 3),
        "accepted_rounds": res["accepted_rounds"],
        "worst_stage_gain": res["worst_stage_gain"],
        "per_stage_gain": [row["score_gain"] for row in res["per_stage"]],
        "train_seconds_total": res["train_seconds_total"],
        "parent_source": res.get("parent_source"),
        "labels": res.get("labels"),
        "seed": res.get("seed"),
    }
    if gate:
        out["gate_v2"] = {"rounds_adjudicated": gate.get("rounds_adjudicated"),
                          "rounds_suppressed": gate.get("rounds_suppressed"),
                          "zones": gate.get("zones"), "detectable_rounds": gate.get("detectable_rounds"),
                          "modes": gate.get("modes"), "rollbacks": gate.get("rollbacks"),
                          "unit_gain": gate.get("unit_gain"),
                          "family_wise_note": gate.get("family_wise_note")}
    if ledger:
        out["ledger"] = {"used": ledger.get("used"), "cost_units": ledger.get("cost_units")}
    if lineage:
        out["lineage"] = {"ok": lineage.get("ok"), "events": lineage.get("events")}
    if res.get("llm"):
        out["llm"] = res["llm"]
    return out


def parity_check() -> dict:
    """v1 口径回归：新跑的 _parity_* 必须与历史 run 逐位相等（协议适配不许改变行为）。"""
    report = {"checked_at_run": "runs/_parity_a / _parity_b / _parity_c", "rows": []}
    for profile, paths in MAIN_RUNS.items():
        new_dir = ROOT / "runs" / f"_parity_{profile.lower()}"
        if not (new_dir / "results.json").is_file():
            report["rows"].append({"profile": profile, "status": "缺回归 run，跳过"})
            continue
        a, b = load(f"runs/_parity_{profile.lower()}"), load(paths["v1"])
        metric = a["metric"]
        same = (abs(a["best"][metric] - b["best"][metric]) < 1e-12
                and a["worst_stage_gain"] == b["worst_stage_gain"]
                and a["accepted_rounds"] == b["accepted_rounds"])
        report["rows"].append({"profile": profile, "metric": metric,
                               "new": a["best"][metric], "historical_v1": b["best"][metric],
                               "worst_new": a["worst_stage_gain"], "worst_v1": b["worst_stage_gain"],
                               "bit_identical": bool(same),
                               "status": "PASS" if same else "FAIL"})
    report["all_pass"] = all(row.get("bit_identical", False) for row in report["rows"])
    return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    main_evidence = {"note": "三组主实验的 v1/v2 双口径，全部字段取自 run 的 results.json",
                     "runs": {profile: {tag: headline(rel) for tag, rel in paths.items()}
                              for profile, paths in MAIN_RUNS.items()}}
    ablations = []
    for rel in ABLATIONS:
        if (ROOT / rel / "results.json").is_file():
            ablations.append(headline(rel))
        else:
            ablations.append({"run": rel, "status": "缺目录"})
    main_evidence["ablations"] = ablations
    (OUT / "main_results.json").write_text(
        json.dumps(main_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append("main_results.json")

    (OUT / "parity_v1.json").write_text(json.dumps(parity_check(), ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    written.append("parity_v1.json")

    probe = ROOT / "runs" / "_probe_a_stages" / "probe.json"
    if probe.is_file():
        data = json.loads(probe.read_text(encoding="utf-8"))
        (OUT / "probe_a_stages.json").write_text(
            json.dumps({"note": "24 配方 × 2 epoch 倍率网格；A 的三段各自增益（不可达性论证的证据）",
                        "generator": "python scripts/probe_a_stages.py",
                        "count": len(data), "rows": data}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        written.append("probe_a_stages.json")
    else:
        print("· 探针原始文件不在盘上（可重生成：python scripts/probe_a_stages.py）")

    for name in written:
        print(f"  ✓ docs/evidence/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
