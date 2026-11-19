"""合成漂移流：给监测器一个**有标准答案**的考试。

    from harness.driftsim import make_stream

为什么必须自己造：赛规说正式评测用「隐藏测试流」，但我们拿不到它，
而「漂移监测与归因 8 分」要的是*及时性、准确性、误触发/漏触发分析*——
这些指标在没有 ground truth 的数据上根本无法定义。所以这里在**公开验证集**上
按教科书定义造四种流，每种都知道 onset 在哪、变的是哪一维、变的是 P(x) 还是 P(y|x)。

四类（定义按漂移文献，不是我发明的口径）：

- `clean`     ：不打扰，原序排列。用来量**误触发率**。
- `covariate` ：对指定维度子集做固定仿射变换（缩放 + 平移）。P(x) 变、
                每个样本的 (x, y) 配对关系不变。用它量**归因准确率**：
                监测器报的 top-k 特征维，是否命中我们真正动过的那几维。
- `concept`   ：P(x) 不动，改 P(y|x)——分类按确定性规则把一部分行的标签搬到
                另一个类；回归把同一批 x 的目标值乘一个系数。
- `mixed`     ：两者同时。

⚠️ 边界（必须写清楚，否则会被误读成造假）：这些标签是我们**造出来的合成标签**，
只用于评估监测器本身的延迟/误报/漏报/归因，**绝不作为模型精度或增益的证据**，
也从不接触平台的隐藏标签（红线 A）。真实精度证据只在 `public/validation` 原序上算。
"""
import dataclasses

import numpy as np

CONDITIONS = ("clean", "covariate", "concept", "mixed")


@dataclasses.dataclass
class Window:
    index: int                      # 第几个窗口（0 起）
    x: np.ndarray
    y: np.ndarray
    start: int                      # 在原始验证流里的行起点（便于回溯）
    truth: dict                     # {"condition","onset","dims","relabelled"}

    @property
    def drifted(self) -> bool:
        return self.truth["condition"] != "clean" and self.index >= self.truth["onset"]


def _transform(x: np.ndarray, dims, scale=1.35, shift=0.6) -> np.ndarray:
    out = x.copy()
    out[:, dims] = out[:, dims] * scale + shift
    return out


def _concept_change(y: np.ndarray, rng, task: str, fraction=0.35):
    """改 P(y|x)：标签搬类（分类）或目标整体缩放（回归）。返回 (新 y, 改动数)。"""
    y = np.array(y, copy=True)
    count = max(1, int(len(y) * fraction))
    picked = rng.choice(len(y), size=count, replace=False)
    if task == "classification":
        labels = np.unique(y)
        if len(labels) < 2:
            return y, 0
        target = labels[(np.searchsorted(labels, y[picked]) + 1) % len(labels)]
        y[picked] = target
        return y, int(count)
    y[picked] = y[picked] * 1.25 + 8.0
    return y, int(count)


def make_stream(x, y, *, task="classification", condition="covariate", onset=2,
                windows=8, window_rows=None, dims=None, seed=7, ordering="iid") -> list:
    """把一条 (x, y) 流切成 `windows` 个窗口；从第 `onset` 个窗口起注入指定类型的漂移。

    `ordering="iid"`（默认）：每个窗口从池子里独立随机抽样。基准必须用这个口径——
    时序数据相邻片段本身就差很远，按时间切出来的「clean」流并不干净，
    实测把误触发率量成 8/8（连参考窗自己都被判 alarm），那测出来的是数据不是监测器。
    `ordering="sequential"`：连续时间片，用来复现闭环里「阶段」的时间语义。

    注入只作用于 onset 之后的窗口；参考窗（onset 之前）保持干净。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed))
    n = len(x)
    if not 1 <= int(onset) < int(windows):
        raise ValueError(f"onset 必须在 1..windows-1 之间，收到 onset={onset} windows={windows}")
    if ordering not in ("iid", "sequential"):
        raise ValueError(f"ordering 只能是 iid/sequential，收到 {ordering!r}")
    window_rows = int(window_rows or max(32, n // (windows + 1)))
    if ordering == "sequential":
        if window_rows * windows > n:                    # 不够切就缩小窗口
            window_rows = max(16, n // windows)
        offset = max(0, (n - window_rows * windows) // 2)
    else:
        window_rows = min(window_rows, n)
        offset = 0
    if dims is None:
        dims = sorted(rng.choice(x.shape[1], size=max(1, min(3, x.shape[1])),
                                replace=False).tolist())
    dims = [int(d) for d in dims][: max(1, min(3, x.shape[1]))]

    stream = []
    for index in range(windows):
        if ordering == "sequential":
            a = offset + index * window_rows
            b = a + window_rows
            wx, wy, start = x[a:b], y[a:b], a
        else:
            picked = rng.choice(n, size=window_rows, replace=False)
            wx, wy, start = x[picked], y[picked], int(picked.min())
        truth = {"condition": condition, "onset": int(onset), "dims": list(dims),
                 "relabelled": 0, "ordering": ordering}
        if condition in ("covariate", "mixed") and index >= onset:
            wx = _transform(wx, dims)
        if condition in ("concept", "mixed") and index >= onset:
            wy, changed = _concept_change(wy, rng, task)
            truth["relabelled"] = changed
        stream.append(Window(index=index, x=wx, y=wy, start=start, truth=truth))
    return stream


def reference_windows(stream: list, onset: int) -> list:
    return [w for w in stream if w.index < onset]


def detect(stream: list, monitor_state_fn, *, alarm_states=("alarm",)) -> list:
    """按窗口顺序问监测器，返回每个窗口的 (index, state, is_alarm)。"""
    rows = []
    for window in stream:
        state = monitor_state_fn(window)
        rows.append({"index": window.index, "state": state,
                     "alarm": state in alarm_states, "drifted": window.drifted})
    return rows


def score_table(rows: list, onset: int, horizon: int = None) -> dict:
    """把告警序列压成可写进报告的四个数：延迟、误触发、漏报、以及命中窗口。"""
    alarms = [r["index"] for r in rows if r["alarm"]]
    late_alarms = [i for i in alarms if i >= onset]
    early_alarms = [i for i in alarms if i < onset]
    limit = onset + horizon if horizon else len(rows)
    detected = [i for i in late_alarms if i <= limit]
    return {"onset": onset,
            "detected": bool(detected),
            "detection_delay_windows": (min(detected) - onset) if detected else None,
            "false_alarms_before_onset": len(early_alarms),
            "missed": not detected,
            "alarm_windows": alarms}


def attribution_overlap(top_features: list, true_dims: list, k: int = 3) -> float:
    """归因命中率：监测器点名的前 k 个特征维里，有几维是我们真正动过的。"""
    named = [int(item["feature"]) for item in top_features[:k]]
    if not named or not true_dims:
        return 0.0
    hits = len(set(named) & set(int(d) for d in true_dims))
    return hits / float(min(k, len(true_dims)))
