"""用 PyTRIO 给策略层做 LoRA SFT（Qwen3.8-27B）。

数据来自 finetune/build_dataset.py 产出的 policy_sft.jsonl：
    messages  = [system, user]      （与线上 harness 完全同一条观测文本）
    completion= 一行 JSON 决策

先干跑（不花钱、只建 Datum 看 token 账）：
    python -m finetune.train_lora --dry-run

真训：
    python -m finetune.train_lora --epochs 2 --rank 32 --lr 1e-4

训练结束后把 weights path 填进 configs/gas.json 的 policy.llm.model，
harness 就会用微调后的策略决策（同一套门禁/回滚不变）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "finetune", "data")
PRICE_PER_M_TOKENS = 4.5  # 元/百万训练 token，按官方 Jev 案例反推的量级，仅用于估算


def load_samples(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise SystemExit(f"数据集为空：{path}（先跑 finetune.build_dataset）")
    return out


def _prompt_text(sample: Dict[str, Any], tokenizer, fmt: str) -> str:
    """把样本还原成"模型真正看到的那段前缀"。

    messages 格式直接用归档的对话；io 格式用 system/user 两段重新套模板——
    两者必须与线上 harness（harness/prompting.py）逐字一致，否则训练/推理错配。
    """
    if fmt == "io":
        messages = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
        ]
    else:
        messages = sample["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_datum(sample: Dict[str, Any], tokenizer, max_length: int, fmt: str = "messages"):
    import numpy as np
    import pytrio as trio

    prompt_text = _prompt_text(sample, tokenizer, fmt)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion = sample["completion"] if fmt != "io" else sample["output"]
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    eos = tokenizer.eos_token_id
    if eos is not None:
        completion_tokens = completion_tokens + [eos]

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        weights = weights[:max_length]

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "weights": np.asarray(weights[1:], dtype=np.float32),
            "target_tokens": np.asarray(tokens[1:], dtype=np.int32),
        },
    )
    return datum, len(prompt_tokens), len(completion_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA, "policy_sft.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--save-name", default="zhixun-policy-v1")
    ap.add_argument("--format", choices=["messages", "io"], default="messages",
                    help="io = input/output 轻量微调格式（finetune/data/policy_sft_io.jsonl）")
    ap.add_argument("--dry-run", action="store_true", help="只建 Datum、打印 token 账，不训练")
    ap.add_argument("--no-swanlab", action="store_true")
    args = ap.parse_args()
    # --format io 时默认读 io 文件，免得忘了改 --data 却按 io 解析 messages 文件
    if args.format == "io" and os.path.basename(args.data) == "policy_sft.jsonl":
        args.data = os.path.join(DATA, "policy_sft_io.jsonl")

    import numpy as np
    import pytrio as trio

    samples = load_samples(args.data)
    random.Random(42).shuffle(samples)

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model, rank=args.rank
    )
    tokenizer = training_client.get_tokenizer()

    datums, prompt_lens, comp_lens = [], [], []
    for s in samples:
        d, pl, cl = build_datum(s, tokenizer, args.max_length, args.format)
        datums.append(d)
        prompt_lens.append(pl)
        comp_lens.append(cl)

    summary = {
        "samples": len(datums),
        "prompt_tokens_mean": round(float(np.mean(prompt_lens)), 1),
        "prompt_tokens_max": int(np.max(prompt_lens)),
        "completion_tokens_mean": round(float(np.mean(comp_lens)), 1),
        "trainable_tokens": int(sum(comp_lens)),
        "total_tokens_until": int(sum(prompt_lens) + sum(comp_lens)),
    }
    est = summary["total_tokens_until"] * args.epochs / 1e6 * PRICE_PER_M_TOKENS
    summary["est_cost_cny"] = round(est, 3)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("[dry-run] 未训练、未扣费。去掉 --dry-run 即开始训练。")
        return

    swanlab_run = None
    if not args.no_swanlab:
        try:
            import swanlab

            swanlab_run = swanlab.init(
                project=os.environ.get("SWANLAB_PROJECT", "university_AI_innovate"),
                mode=os.environ.get("SWANLAB_MODE", "offline"),
                experiment_name=f"sft-policy-{args.save_name}",
                config=summary | {"rank": args.rank, "lr": args.lr, "epochs": args.epochs, "batch": args.batch},
            )
        except Exception as exc:  # 记录不上不影响训练
            print(f"[warn] SwanLab 初始化失败：{type(exc).__name__}: {exc}")

    steps_per_epoch = (len(datums) + args.batch - 1) // args.batch
    print(f"开始训练：{len(datums)} 条 × {args.epochs} epoch，{steps_per_epoch} 步/epoch")
    global_step = 0
    for epoch in range(args.epochs):
        for start in range(0, len(datums), args.batch):
            batch = datums[start : start + args.batch]
            fwdbwd = training_client.forward_backward(batch, "cross_entropy")
            optim = training_client.optim_step(trio.AdamParams(learning_rate=args.lr))
            res = fwdbwd.result()
            optim.result()

            logprobs = np.concatenate([o["logprobs"].tolist() for o in res.loss_fn_outputs])
            weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
            loss = float(-np.dot(logprobs, weights) / max(weights.sum(), 1e-9))
            print(f"epoch {epoch+1}/{args.epochs} step {start//args.batch+1}/{steps_per_epoch} loss={loss:.4f}")
            if swanlab_run is not None:
                try:
                    import swanlab

                    swanlab.log({"loss": loss, "epoch": epoch + 1}, step=global_step)
                except Exception:
                    pass
            global_step += 1

    saved = training_client.save_weights_for_sampler(name=args.save_name).result()
    path = getattr(saved, "path", None) or (saved.get("path") if isinstance(saved, dict) else str(saved))
    print(f"[saved] weights path = {path}")

    meta = {
        "weights_path": path,
        "save_name": args.save_name,
        "base_model": args.base_model,
        "rank": args.rank,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch": args.batch,
        "data": args.data,
        "format": args.format,
        **summary,
    }
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "last_train.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta] 已写 {os.path.join(DATA, 'last_train.json')}")
    print('用法：configs/gas.json -> policy.llm.model 填上面的 weights path，然后 --policy llm 跑对照')


if __name__ == "__main__":
    main()