"""漂移与性能信号：KS（有界、不用分箱）+ 置信度 + 指标变化。

与第一版 harness 同一套判据：每特征 KS 的 Top-3 均值做有限样本缩放，
阈值在参考集上对半自举得到，不手拍。
"""
import numpy as np


def _ks_scaled(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.sort(ref.ravel())
    cur = np.sort(cur.ravel())
    n, m = ref.size, cur.size
    if n == 0 or m == 0:
        return 0.0
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / n
    cdf_cur = np.searchsorted(cur, grid, side="right") / m
    return float(np.max(np.abs(cdf_ref - cdf_cur)) * np.sqrt(n * m / (n + m)))


def ks_topk(ref_x: np.ndarray, cur_x: np.ndarray, k: int = 3):
    scores = np.array([_ks_scaled(ref_x[:, j], cur_x[:, j]) for j in range(ref_x.shape[1])])
    order = np.argsort(scores)[::-1][:k]
    return float(scores[order].mean()), [{"feature": int(j), "ks": float(scores[j])} for j in order]


def bootstrap_thresholds(ref_x: np.ndarray, k: int = 3, n_boot: int = 24, seed: int = 42,
                         block: int = 0, compare_rows: int = 0) -> dict:
    """KS 告警阈值：默认 iid 对半自举（v1 口径，保持不变以便复现旧结果）。

    `block>0` → 连续块自举（针对工况聚簇的流）；
    `compare_rows>0` → **按实际比较的样本量配比**自举：真实比较是「大参考窗 vs 小测试窗」，
    而等大对半量出来的零分布偏窄，会让小窗口系统性误报。实测这是 C（turbofan，
    参考窗 7683 行 vs 测试窗 2561 行）把 clean 判成 covariate 的主因，
    而块自举修不了它（见 docs/漂移监测基准.md 的对照表）。
    """
    rng = np.random.default_rng(seed)
    n = ref_x.shape[0]
    vals = []
    span = int(compare_rows or 0)
    if span and span < n:
        for _ in range(n_boot):
            idx = rng.permutation(n)
            small = idx[:span]
            large = idx[span:]
            vals.append(ks_topk(ref_x[large], ref_x[small], k)[0])
    elif block and 2 * int(block) <= n:
        width = int(block)
        for _ in range(n_boot):
            first = rng.integers(0, n - width + 1)
            second = rng.integers(0, n - width + 1)
            vals.append(ks_topk(ref_x[first:first + width], ref_x[second:second + width], k)[0])
    else:
        for _ in range(n_boot):
            idx = rng.permutation(n)
            half = n // 2
            vals.append(ks_topk(ref_x[idx[:half]], ref_x[idx[half:2 * half]], k)[0])
    q95 = float(np.quantile(vals, 0.95))
    q99 = float(np.quantile(vals, 0.995))
    warn = max(q95 * 1.05, 0.05)
    alarm = max(q99 * 1.05, warn * 1.5)
    return {"warn": warn, "alarm": alarm, "null_q95": q95, "null_q99": q99,
            "null_mean": float(np.mean(vals)), "block": int(block or 0),
            "compare_rows": int(span)}


def bootstrap_window_thresholds(ref_x: np.ndarray, k: int = 3, n_boot: int = 16,
                                seed: int = 44, window_rows: int = 0) -> dict:
    """按「测试窗口 vs 参考窗」这个真实程序自举零分布。

    为什么需要：前面三种口径（等大对半 / 连续块 / 样本量配比）在 C 上都把 clean 流判成
    covariate——因为它们量的都是「同一大池子里两个子集的差异」，而基准里的比较是
    「1 个窗口 vs 若干个窗口的并」。窗口越小子集越独立，差异的抽样波动并不随样本量缩放，
    所以任何按等大对半定的阈值都会系统性偏低。
    正解是让**零分布由产生被测窗口的同一程序生成**：反复把参考池重排、切出一个与
    测试窗同尺寸的窗口，与剩下的部分算 KS。
    """
    rng = np.random.default_rng(seed)
    n = ref_x.shape[0]
    span = int(window_rows or max(16, n // 3))
    if span >= n:
        span = max(1, n // 3)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.permutation(n)
        test = idx[:span]
        rest = idx[span:]
        if len(rest) < 2 or len(test) < 2:
            continue
        vals.append(ks_topk(ref_x[rest], ref_x[test], k)[0])
    if not vals:
        return bootstrap_thresholds(ref_x, k=k, n_boot=n_boot, seed=seed)
    values = np.asarray(vals, dtype=float)
    q95 = float(np.quantile(values, 0.95))
    q99 = float(np.quantile(values, 0.995))
    warn = max(q95 * 1.05, 0.05)
    alarm = max(q99 * 1.05, warn * 1.5)
    return {"warn": warn, "alarm": alarm, "null_q95": q95, "null_q99": q99,
            "null_mean": float(values.mean()), "block": 0, "compare_rows": span,
            "procedure": "windowed"}


def confidence(proba) -> float:
    if proba is None:
        return float("nan")
    return float(np.mean(np.max(proba, axis=1)))


def classify_state(ks: float, thr: dict, metric_drop: float = 0.0,
                   conf_drop: float = 0.0, perf_warn: float = 0.02,
                   metric_thr: dict | None = None) -> str:
    """状态判定。给了 `metric_thr`（零漂移下指标跌幅的自举分布）就用它，
    否则退回历史固定 perf_warn=0.02（v1 口径，保持不变以便复现旧结果）。"""
    warn_drop = float(metric_thr["warn"]) if metric_thr else perf_warn
    alarm_drop = float(metric_thr["alarm"]) if metric_thr else perf_warn
    if ks >= thr["alarm"] or metric_drop >= alarm_drop or conf_drop >= 0.10:
        return "alarm"
    if ks >= thr["warn"] or metric_drop > 0 or conf_drop >= 0.05:
        return "warn"
    return "normal"


def bootstrap_metric_threshold(score_fn, ref_x, ref_y, n_boot: int = 16, seed: int = 44) -> dict:
    """零漂移时「性能跌幅」本来就能有多大——用对半自举把它量出来，而不是手拍 0.02。

    `score_fn(x, y)` 必须已经统一成「越大越好」，否则 B/C 的 RMSE 会把跌幅算反。
    小窗口下指标天然抖得厉害（A 的一个时间片里只有 2 类），所以这一项不标定，
    监测器就会把「窗口小、类别偏」误报成漂移：基准实测误触发 8/8。
    """
    rng = np.random.default_rng(seed)
    ref_x = np.asarray(ref_x, dtype=np.float64)
    ref_y = np.asarray(ref_y)
    n = len(ref_x)
    if n < 8:
        return {"warn": 0.02, "alarm": 0.02, "null_q95": 0.0, "null_mean": 0.0, "n_boot": 0}
    drops = []
    for _ in range(int(n_boot)):
        order = rng.permutation(n)
        half = n // 2
        left = score_fn(ref_x[order[:half]], ref_y[order[:half]])
        right = score_fn(ref_x[order[half:2 * half]], ref_y[order[half:2 * half]])
        base = max(abs(left), abs(right), 1e-9)
        drops.append(max(0.0, (max(left, right) - min(left, right)) / base))
    drops = np.asarray(drops, dtype=float)
    q95 = float(np.quantile(drops, 0.95))
    q995 = float(np.quantile(drops, 0.995))
    return {"warn": max(q95, 1e-6), "alarm": max(q995, q95 * 1.5, 1e-6),
            "null_q95": q95, "null_mean": float(drops.mean()), "n_boot": int(n_boot)}


# ── 漂移类型归因 ───────────────────────────────────────────────────────
# 「漂移监测与归因 8 分」要的是：不仅报警，还要说清是哪一类漂移，因为
# covariate（P(x) 变了）与 concept（P(y|x) 变了）该换的配方完全不同。
# 判据刻意只用公开 train/validation 上能算的量：特征侧 KS + 标签分布位移。

def label_shift(ref_y, cur_y) -> float:
    """标签分布位移：类别用总变差距离，回归用标量化的 KS。"""
    ref = np.asarray(ref_y).ravel()
    cur = np.asarray(cur_y).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    distinct = np.unique(np.concatenate([ref, cur]))
    if len(distinct) <= 32:                             # 分类：类频率的总变差距离
        props_ref = np.array([np.mean(ref == v) for v in distinct])
        props_cur = np.array([np.mean(cur == v) for v in distinct])
        return float(0.5 * np.sum(np.abs(props_ref - props_cur)))
    return _ks_scaled(ref.reshape(-1, 1), cur.reshape(-1, 1))   # 回归：RUL 值的分布位移


def bootstrap_label_threshold(ref_y, n_boot: int = 24, seed: int = 43) -> dict:
    """同样用对半自举得到「无漂移时标签位移能有多大」，不手拍阈值。"""
    rng = np.random.default_rng(seed)
    ref_y = np.asarray(ref_y).ravel()
    n = ref_y.size
    vals = [label_shift(ref_y[rng.permutation(n)[: n // 2]], ref_y[rng.permutation(n)[: n // 2]])
            for _ in range(n_boot)]
    q95 = float(np.quantile(vals, 0.95))
    return {"warn": max(q95 * 1.05, 0.02), "alarm": max(float(np.quantile(vals, 0.995)) * 1.05,
                                                      max(q95 * 1.05, 0.02) * 1.5),
            "null_q95": q95, "null_mean": float(np.mean(vals))}


def classify_drift(feature_ks: float, shift: float, thr: dict, label_thr: dict) -> str:
    """四类归因：none / covariate（只有 P(x) 变）/ concept（P(y) 或 P(y|x) 变）/ mixed。"""
    covariate = float(feature_ks) >= float(thr["warn"])
    concept = float(shift) >= float(label_thr["warn"])
    if covariate and concept:
        return "mixed"
    if covariate:
        return "covariate"
    if concept:
        return "concept"
    return "none"