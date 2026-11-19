"""元控制：在两处调节自己 —— 预算分配与探索强度。

只在「允许的动作」内部调（轮数上限、探索系数），不碰数据、不碰门禁阈值：
门禁阈值属于评分口径，不能由策略层自己放松。
"""
from .policy import candidates_for


class MetaController:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.epoch_cap = int(cfg.get("epoch_cap", 12))
        self.explore = float(cfg.get("explore", 0.3))
        self.history = []
        self.updates = []

    def observe(self, gain: float, accepted: bool, seconds: float):
        self.history.append({"gain": float(gain), "accepted": bool(accepted), "seconds": float(seconds)})

    def maybe_update(self, round_no: int, rounds_left: int, worker=None) -> dict | None:
        """按最近 3 轮的接受率调探索强度；按累计耗时调轮数上限。"""
        if len(self.history) < 3:
            return None
        window = self.history[-3:]
        rate = sum(1 for h in window if h["accepted"]) / 3.0
        change = None
        if rate == 0.0 and self.explore > 0.15:
            change = {"key": "explore", "before": self.explore, "after": round(self.explore * 0.7, 3),
                      "why": "连续 3 轮被拒：先降探索强度，回到已知有效的一档"}
            self.explore = change["after"]
        elif rate >= 0.66 and self.explore < 0.6:
            change = {"key": "explore", "before": self.explore, "after": round(self.explore * 1.2, 3),
                      "why": "近 3 轮多数被采纳：可以稍微多试新动作"}
            self.explore = change["after"]
        if change is None:
            return None
        change["round"] = round_no
        self.updates.append(change)
        if worker is not None:
            worker(change)
        return change

    def clamp_epochs(self, epochs: int) -> int:
        return max(2, min(self.epoch_cap, int(epochs)))