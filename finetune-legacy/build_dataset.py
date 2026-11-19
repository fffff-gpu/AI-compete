"""把 harness 的 journal 变成 SFT 数据集（训练策略层用的 Qwen3.8-27B LoRA）。

一条样本 = 一个真实决策点：

    prompt     loop 在决策时刻写进 journal 的观测文本（原样复用，保证训练/上线同分布）
    completion 由**事后验证**给出的答案 JSON，而不是当初策略采取的动作

标签怎么来（这是这份数据集可信的原因）：
- 重训后门禁放行、且假设在下一个未见 batch 上被证实  -> retrain(candidate)
- 重训但门禁拒绝（增益不够）或假设被证伪              -> wait（别花算力）
- 原本就 wait 的步（冷却/未触发）                     -> wait

所以规则策略的错判不会被学进去，模型学的是"经系统验证过的判断"。

用法：
    python -m finetune.build_dataset                 # 读取 runs/，写出数据
    python -m finetune.build_dataset --holdout p2_llm_s3 seed3
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from harness.prompting import PROMPT_VERSION, SYSTEM_PROMPT  # noqa: E402

OUT_DIR = os.path.join(ROOT, "finetune", "data")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def _answer(action: str, candidates: List[str], hypothesis: str, reason: str) -> str:
    return json.dumps(
        {"action": action, "candidates": candidates, "hypothesis": hypothesis, "reason": reason},
        ensure_ascii=False,
    )


def _label_for(row: Dict[str, Any], hyp: Optional[Dict[str, Any]], decision: Dict[str, Any]) -> (str, str, str):
    """返回 (action, tag, 说明)。"""
    act = str(decision.get("action") or "").lower()
    promoted = str(row.get("promoted", "0")) in ("1", "True", "true")
    verdict = (hyp or {}).get("verdict")
    ks = _f(row, "ks_top3")
    acc = _f(row, "acc_harness")
    gain = _f(row, "gain")
    val = _f(row, "val_acc")

    if act == "retrain":
        cands = decision.get("candidates") or []
        cand = cands[0] if cands else "hgb_deep"
        if promoted and verdict != "refuted":
            why = (
                f"重训 {cand} 后验证精度 {val:.3f}，门禁放行"
                + (f"；下一未见 batch 上新旧对比 gap={_f(hyp or {}, 'gap'):+.3f}，假设成立" if verdict == "confirmed" else "")
                + "。"
            )
            return "retrain", "retrain_verified", why
        if promoted and verdict == "refuted":
            return "wait", "refuted", f"重训虽上线，但下一未见 batch 上新旧对比 gap={_f(hyp or {}, 'gap'):+.3f}，假设被证伪；先不继续训练。"
        return "wait", "gate_reject", f"候选增益不足（{gain:.4f} 低于门禁），本次不晋升，继续观察。"

    # 原本 wait：给人的理由要贴合当时的信号强度
    return "wait", "wait", f"KS Top-3 均值 {ks:.3f}，可见精度 {acc:.3f}；信号未持续超限，先不消耗训练算力，继续观察。"


def build(runs_dir: str, holdout: List[str], max_samples: int = 4000, include_cooldown: bool = False) -> Dict[str, Any]:
    run_dirs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d))
    train: List[Dict[str, Any]] = []
    evals: List[Dict[str, Any]] = []
    per_run: Dict[str, Any] = {}
    seen = set()
    skipped_cooldown = 0
    skipped_stale = 0
    skipped_policy = 0
    skipped_unverified = 0
    skipped_gate_wrong = 0
    skipped_old_template = 0

    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        if name.startswith("_"):
            continue  # 临时/诊断 run 不参与训练
        res_path = os.path.join(run_dir, "results.json")
        if not os.path.exists(res_path):
            continue
        results = json.load(open(res_path, encoding="utf-8"))
        journal = _read_jsonl(os.path.join(run_dir, "journal.jsonl"))
        # 模板卫生：只收"当前模板"产出的 run。旧模板的 prompt 与线上分布不一致，
        # 混进训练集等于教模型看一种格式、上线喂另一种（隐藏 B 集上会直接失效）。
        base = [e for e in journal if e.get("kind") == "baseline"]
        tpl = base[-1].get("template") if base else None
        if tpl != PROMPT_VERSION:
            skipped_old_template += 1
            continue
        rows = {int(r["step"]): r for r in _read_csv(os.path.join(run_dir, "metrics.csv")) if r.get("step")}
        decisions = {int(e["step"]): e for e in journal if e.get("kind") == "decision" and e.get("prompt")}
        hyps = {int(e["step"]): e for e in journal if e.get("kind") == "hypothesis" and e.get("step")}
        shadows = {int(e["for_step"]): e for e in journal if e.get("kind") == "shadow" and e.get("for_step")}

        kept = 0
        for step, dec in sorted(decisions.items()):
            row = rows.get(step)
            if not row:
                continue
            # 冷却期内的 wait 不是"判断"，是系统约束（而且线上根本不会问模型），剔掉
            if not include_cooldown and "cooldown" in str(dec.get("reason") or "").lower():
                skipped_cooldown += 1
                continue
            # "固定周期"基线（AlwaysPolicy）的动作不是判断，学它会把模型教成无脑重训
            if "固定周期" in str(dec.get("hypothesis") or "") or "periodic" in str(dec.get("reason") or ""):
                skipped_policy += 1
                continue
            action, tag, why = _label_for(row, hyps.get(step), dec)
            shadow = shadows.get(step)
            extra: Dict[str, Any] = {}
            if tag in ("wait", "gate_reject"):
                # 没有告警时的 wait 是"构造上正确"（系统认为一切正常，忍着当然对）
                if tag == "wait" and str(row.get("state")) == "normal":
                    tag = "wait_calm"
                elif shadow is None:
                    # 门禁拒绝本身就是系统判定证据（重训真的被拒了）；wait 则必须靠反事实复核
                    if tag == "gate_reject":
                        extra = {"verify": "gate"}
                    else:
                        skipped_unverified += 1
                        continue
                elif shadow.get("would_help"):
                    if tag == "gate_reject":
                        # 门禁误拒：当时重训其实有用，这条"wait"标签会教坏模型
                        skipped_gate_wrong += 1
                        continue
                    gain = float(shadow.get("gain") or 0.0)
                    action, tag = "retrain", "counterfactual_retrain"
                    why = f"反事实复核：当时重训在下一批上 {gain:+.3f}，超过门禁"
                    extra = {"shadow_gain": gain, "shadow_candidate": shadow.get("candidate")}
                else:
                    tag = "wait_verified"
                    extra = {"shadow_gain": float(shadow.get("gain") or 0.0)}
            if action == "retrain":
                hypothesis = dec.get("hypothesis") or why
                cand = (dec.get("candidates") or ["hgb_deep"])[0]
                if tag == "counterfactual_retrain" and extra.get("shadow_candidate"):
                    cand = extra["shadow_candidate"]
                    hypothesis = "该批漂移下重训可提升 %+.3f，应趁窗口训练" % extra["shadow_gain"]
                reason = (dec.get("reason") or "")[:120] or why
                completion = _answer("retrain", [cand], hypothesis, reason)
            else:
                completion = _answer(
                    "wait",
                    [],
                    "继续观察：当前信号不足以证明「重训能换来更高精度」，先保留算力。",
                    why,
                )

            prompt = dec["prompt"]
            # 旧模板的观测文本里参考集恒为 0（bug 已修），模板不一致的样本会教坏模型
            if "参考集 0.000" in prompt:
                skipped_stale += 1
                continue
            key = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            sample = {
                "id": f"{name}-s{step}",
                "run": name,
                "step": step,
                "batch": row.get("batch"),
                "task": results.get("adapter"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "completion": completion,
                "label": {"action": action, "candidates": [c for c in (dec.get("candidates") or [])[:1]] if action == "retrain" else []},
                "tag": tag,
                "meta": {
                    "state": row.get("state"),
                    "fired": row.get("fired"),
                    "ks_top3": _f(row, "ks_top3"),
                    "acc_harness": _f(row, "acc_harness"),
                    "promoted": row.get("promoted"),
                    "verdict": (hyps.get(step) or {}).get("verdict"),
                    "policy_source": dec.get("source"),
                    "verify": (
                        "ab" if tag in ("retrain_verified", "refuted")
                        else "gate" if tag == "gate_reject"
                        else "shadow" if extra else "construction"
                    ),
                    **extra,
                },
            }
            if name in holdout:
                evals.append(sample)
            else:
                train.append(sample)
            kept += 1

        if kept:
            per_run[name] = {
                "n": kept,
                "tags": {t: sum(1 for s in train + evals if s["run"] == name and s["tag"] == t) for t in ("retrain_verified", "counterfactual_retrain", "gate_reject", "refuted", "wait_verified", "wait_calm")},
            }

    train = train[:max_samples]
    evals = evals[: max(20, max_samples // 4)]
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {
        "train": len(train),
        "eval": len(evals),
        "skipped_cooldown_waits": skipped_cooldown,
        "skipped_stale_prompts": skipped_stale,
        "skipped_always_policy": skipped_policy,
        "skipped_unverified": skipped_unverified,
        "skipped_gate_wrong": skipped_gate_wrong,
        "skipped_old_template_runs": skipped_old_template,
        "prompt_version": PROMPT_VERSION,
        "by_tag_train": {t: sum(1 for s in train if s["tag"] == t) for t in ("retrain_verified", "counterfactual_retrain", "gate_reject", "refuted", "wait_verified", "wait_calm")},
        "by_tag_eval": {t: sum(1 for s in evals if s["tag"] == t) for t in ("retrain_verified", "counterfactual_retrain", "gate_reject", "refuted", "wait_verified", "wait_calm")},
        "holdout_runs": holdout,
        "runs": per_run,
    }

    with open(os.path.join(OUT_DIR, "policy_sft.jsonl"), "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "policy_sft_eval.jsonl"), "w", encoding="utf-8") as f:
        for s in evals:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 标准轻量微调格式：一条样本 = input / output 两个字段（人可读、可直接喂给
    # 「instruction 微调」类工具；PyTRIO 侧再由 train_lora.py 套 chat 模板）
    def _to_io(s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input": SYSTEM_PROMPT.rstrip() + "\n\n" + s["messages"][-1]["content"],
            "output": s["completion"],
            # 训练侧要按 chat 模板切 system/user，所以把两段分开留一份
            "system": SYSTEM_PROMPT.rstrip(),
            "user": s["messages"][-1]["content"],
            "meta": {"id": s["id"], "run": s["run"], "step": s["step"], "tag": s["tag"], **s["meta"]},
        }

    io_train = [_to_io(s) for s in train]
    io_eval = [_to_io(s) for s in evals]
    # 硬校验：output 必须是一行合法 JSON，action 必须在动作空间内
    for s in io_train + io_eval:
        obj = json.loads(s["output"])
        assert obj.get("action") in ("wait", "retrain"), f"非法 action: {s['output'][:80]}"

    with open(os.path.join(OUT_DIR, "policy_sft_io.jsonl"), "w", encoding="utf-8") as f:
        for s in io_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "policy_sft_io_eval.jsonl"), "w", encoding="utf-8") as f:
        for s in io_eval:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 人工抽检：20 条训练样本，input/output 成对展示
    review = ["# 微调样本抽检（input / output）\n"]
    for s in io_train[:20]:
        review.append(f"## {s['meta']['id']}　tag={s['meta']['tag']}　step={s['meta']['step']}\n")
        review.append("**input**\n\n```text\n" + s["input"] + "\n```\n")
        review.append("**output**\n\n```json\n" + s["output"] + "\n```\n")
    with open(os.path.join(OUT_DIR, "policy_sft_io.review.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(review))

    stats["io_train"] = len(io_train)
    stats["io_eval"] = len(io_eval)
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument(
        "--holdout",
        nargs="*",
        default=["v2_xjtu"],
        help="整组留出的 run：v2_xjtu 是跨数据集留出（回归、不同候选名、不同指标尺度）",
    )
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--include-cooldown", action="store_true", help="把冷却期 wait 也算作样本（默认剔除）")
    args = ap.parse_args()

    stats = build(args.runs, args.holdout, args.max_samples, args.include_cooldown)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["train"] == 0:
        print("[!] 没有样本：先跑几次 harness.loop（journal 里要有带 prompt 的 decision 事件）")


if __name__ == "__main__":
    main()