"""离散时间闭环控制器：死区 + 三级动作（利用 / 探测 / 保持）+ 抗积分饱和。

被控对象：一次续训（输入 = 配方 + epochs，输出 = 验证集指标）
传感器：每轮验证集增益 gain（已统一成「越大越好」）与门禁裁决 accepted
执行器：训练轮数（epochs 缩放）与动作类别（利用上一轮 / 换未试过的 / 本轮不训）

三条控制律：
  死区   band = min_gain + 0.25 × 近 5 轮增益标准差（噪声地板），低于它不算改进
  积分   连续被拒累加 reject_streak；到 2 次判定「执行器饱和」，本轮进入 hold
  保持   饱和时直接跳过训练（epochs=0），把这一轮的算力省下来，而不是继续乱试

hold 只保持一轮：下一轮积分清零重新进入 probe，避免把自己锁死。
"""
from __future__ import annotations

import math


def chatter(gains: list[float]) -> int:
    """增益符号翻转次数（忽略 0）：控制回路抖振的直接度量。"""
    signs = [1 if g > 0 else -1 for g in gains if abs(g) > 1e-12]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


class LoopController:
    def __init__(self, min_gain: float, *, k_noise: float = 0.25, window: int = 5,
                 hold_after: int = 2, exploit_scale: float = 1.0):
        """exploit_scale 默认 1.0：控制器只决定「花不花这一轮算力」和「多小的增益才算改进」，
        不擅自放大步长——实测放大步长在 C 上偶然更好、在 B 上把首轮从 +14.3 变成 −10.7，
        说明步长该由策略层按数据决定，而不是控制器的开环猜测。"""
        self.min_gain = float(min_gain)
        self.k_noise = k_noise
        self.window = window
        self.hold_after = hold_after
        self.exploit_scale = exploit_scale
        self.gains: list[float] = []
        self.reject_streak = 0
        self.log: list[dict] = []

    # ---- 传感器 -----------------------------------------------------------
    def deadband(self) -> float:
        """噪声地板：min_gain + 0.25σ，最多抬到 min_gain 的 3 倍。"""
        sd = stdev(self.gains[-self.window:])
        return self.min_gain + min(self.k_noise * sd, 2.0 * self.min_gain)

    def mode(self) -> str:
        """下一轮该干什么：exploit（沿上次方向加长）/ probe（换未试过的）/ hold（不训）。"""
        if not self.log:
            return "exploit"
        if self.log[-1]["accepted"]:
            return "exploit"
        if self.reject_streak >= self.hold_after:
            return "hold"
        return "probe"

    def epochs_scale(self) -> float:
        if not self.log:                      # 第一轮没有任何反馈，不做缩放
            return 1.0
        m = self.mode()
        if m == "hold":
            return 0.0
        return self.exploit_scale if m == "exploit" else 1.0

    # ---- 反馈 -------------------------------------------------------------
    def update(self, gain: float, accepted: bool, epochs_used: int) -> dict:
        mode_used = self.mode()
        self.gains.append(float(gain))
        if accepted:
            self.reject_streak = 0
        else:
            self.reject_streak += 1
        if mode_used == "hold":
            self.reject_streak = 0            # 保持只一轮，防止锁死

        rec = {"gain": round(float(gain), 6), "accepted": bool(accepted),
               "mode": mode_used, "deadband": round(self.deadband(), 6),
               "reject_streak": self.reject_streak, "epochs_used": int(epochs_used),
               "sd_recent": round(stdev(self.gains[-self.window:]), 6)}
        self.log.append(rec)
        return rec

    def summary(self) -> dict:
        modes = [r["mode"] for r in self.log]
        return {"deadband_final": round(self.deadband(), 6),
                "chatter": chatter(self.gains),
                "mode_counts": {m: modes.count(m) for m in ("exploit", "probe", "hold")},
                "held_rounds": [i + 1 for i, r in enumerate(self.log) if r["mode"] == "hold"],
                "rounds": self.log}