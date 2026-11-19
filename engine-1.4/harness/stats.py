"""v2 门禁的统计底座：多轨迹 + 配对 bootstrap。

**为什么不能按交接说明里的原方案做。** `harness-v2-设计.md` §5.3 与交接说明都写「候选用
3 seed 配对评估」，但官方 `baselines/train.py:73` 在 resume 时拒绝与 checkpoint 不同的
seed，`:77` 还要求 profile 的 seed 与 checkpoint 一致——续训协议下**训练种子是锁死的**，
多 seed 做不到。硬做就变成「从头训一个新模型冒充续训」，那是红线 B。

合法的方差来源有两个，这里都用：

1. **多轨迹**：`train.py:82-83` 明确允许在续训时改 `batch_size` 与 `learning_rate`。
   同父亲、同配方、只改 batch_size，优化轨迹（shuffle→分批的边界）不同，得到 n 个
   合法候选。第一档取 config 原值，所以 `--trajectories 1` 与 v1 逐位一致。
2. **配对 bootstrap**：把同一批验证行同时喂给新旧两个检查点，重采样行指标差。
   它测的是**评估噪声**（1244/1679/23049 行有限样本带来的抖动），与轨迹噪声互补。
   采纳用的置信下界取两者里更保守的一个。

红线 C 说「只报最好 seed 不报方差」要重罚，所以采纳后交付的是**中位轨迹**
（按分数取中间那条），选择规则事先写死、不是事后挑最好；每条轨迹的分数都进
lineage 和记录层。
"""
import numpy as np

MAX_BOOTSTRAP_ROWS = 4000


def batch_ladder(config_batch: int, n: int) -> list:
    """n 条轨迹的 batch_size：第一档=config 原值（保证与 v1 可比），其余 ±25%。

    batch_size 是官方在 resume 时允许改的两个超参之一，所以这是合规的方差来源；
    每一档都仍然只读 public/train.jsonl，不碰验证集。
    """
    n = max(1, int(n))
    base = int(config_batch)
    if n == 1:
        return [base]
    ladder = [base]
    offsets = (-0.25, 0.25, -0.5, 0.5, -0.125, 0.125)
    for ratio in offsets:
        if len(ladder) >= n:
            break
        value = max(1, int(round(base * (1.0 + ratio))))
        if value not in ladder:
            ladder.append(value)
    while len(ladder) < n:
        ladder.append(max(1, int(round(base * (1.0 - 0.25 * (len(ladder) + 1))))))
    return ladder[:n]


def paired_bootstrap_lcb(score_of, old_model, new_model, x, y, *, n_boot=200, seed=0,
                        alpha=0.10, max_rows=MAX_BOOTSTRAP_ROWS) -> dict:
    """同一批行同时过新旧模型，取指标差的 `alpha` 分位作保守下界。

    `score_of(metrics)` 必须已经统一成「越大越好」，否则 B/C 的 RMSE 会算反。
    只重采样行、不改数据内容：验证集在这里仍然只用于评估，一次梯度都没参与。
    """
    total = int(len(y))
    if total == 0:
        return {"n_boot": 0, "rows": 0, "lcb": None, "se": None, "ci": None}
    rng = np.random.default_rng(int(seed))
    if total > max_rows:                               # 大数据集在固定子样本上自助，成本可控
        keep = np.sort(rng.choice(total, size=max_rows, replace=False))
        x_sub, y_sub = x[keep], y[keep]
    else:
        x_sub, y_sub = x, y
    old_metrics = score_of(old_model, x_sub, y_sub)
    new_metrics = score_of(new_model, x_sub, y_sub)
    point = new_metrics - old_metrics
    rows = int(len(y_sub))
    diffs = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        pick = rng.integers(0, rows, rows)
        diffs[index] = score_of(new_model, x_sub[pick], y_sub[pick]) - \
                       score_of(old_model, x_sub[pick], y_sub[pick])
    se = float(np.std(diffs, ddof=1)) if n_boot > 1 else 0.0
    return {"n_boot": int(n_boot), "rows": rows, "point": float(point), "se": se,
            "lcb": float(np.quantile(diffs, alpha)),
            "ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}


def power_note(n: int, effect: float, sigma: float) -> dict:
    """把「这个效应在 n 条轨迹下有多大概率被检出」写进报告，而不是假装测得出来。"""
    from .governance import Gate

    return {"trajectories": int(n), "effect": float(effect), "sigma": float(sigma),
            "mde_80pct": Gate.mde(n, sigma), "power": Gate.power(n, effect, sigma),
            "detectable": bool(abs(float(effect)) >= Gate.mde(n, sigma))}
