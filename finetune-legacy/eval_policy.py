"""离线评估策略模型：同一批观测，比较基座 Qwen3.8-27B 与 LoRA 微调版。

    python -m finetune.eval_policy                       # 只测基座
    python -m finetune.eval_policy --weights <path>      # 测微调版

指标：
- json_ok        输出能被解析成一行 JSON 的比例（格式纪律）
- action_ok      动作与"事后验证过的答案"一致的比例（判断力）
- exact_ok       动作 + 候选名都一致（更严）
- avg_tokens     平均输出 token（成本）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "finetune", "data")


def _parse(text: str) -> Optional[Dict[str, Any]]:
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i : j + 1])
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA, "policy_sft_eval.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--weights", default=None, help="LoRA 权重路径；不给就测基座")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    import pytrio as trio

    samples: List[Dict[str, Any]] = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit("评估集为空：先跑 finetune.build_dataset")

    svc = trio.ServiceClient()
    client = svc.create_sampling_client(base_model=args.base_model, model_path=args.weights)
    tok = client.get_tokenizer()

    def render(messages) -> List[int]:
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok.encode(text, add_special_tokens=False)

    ok_json = ok_action = ok_exact = 0
    tokens_out: List[int] = []
    rows = []
    for s in samples:
        ids = render(s["messages"])
        res = client.sample(
            prompt=trio.ModelInput.from_ints(ids),
            sampling_params=trio.SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
            num_samples=1,
        ).result()
        text = res.sequences[0].text
        obj = _parse(text)
        label = s["label"]
        act_ok = exact = False
        if obj is not None:
            ok_json += 1
            act = str(obj.get("action", "")).lower()
            act_ok = act == label["action"]
            cands = obj.get("candidates") or []
            if isinstance(cands, str):
                cands = [cands]
            exact = act_ok and sorted(cands) == sorted(label["candidates"])
            ok_action += int(act_ok)
            ok_exact += int(exact)
        n_tok = len(getattr(res.sequences[0], "tokens", []) or [])
        tokens_out.append(n_tok)
        rows.append(
            {
                "id": s["id"],
                "gold": label,
                "raw": text[:200],
                "action_ok": act_ok,
                "exact_ok": exact,
            }
        )

    n = len(samples)
    out = {
        "n": n,
        "weights": args.weights or "(base)",
        "json_ok": round(ok_json / n, 4),
        "action_ok": round(ok_action / n, 4),
        "exact_ok": round(ok_exact / n, 4),
        "avg_out_tokens": round(sum(tokens_out) / max(len(tokens_out), 1), 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    tag = "tuned" if args.weights else "base"
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"[saved] {os.path.join(DATA, f'eval_{tag}.json')}")


if __name__ == "__main__":
    main()