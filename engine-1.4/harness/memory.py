"""策略记忆：每个（任务, 候选）一个 Beta 后验，跨轮次、跨运行累积。"""
import json
import math
from pathlib import Path


class PolicyMemory:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.stats = {}          # task -> {cand: {"alpha":, "beta":, "gain":, "n":}}
        self.episodes = []
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.stats = raw.get("stats", {})
                self.episodes = raw.get("episodes", [])
            except (ValueError, OSError):
                self.path.rename(self.path.with_suffix(".broken.json"))
                self.stats, self.episodes = {}, []

    # ---------------------------------------------------------- 记录
    def record(self, task, candidate, gain, accepted, round_no=None, note=None):
        per_task = self.stats.setdefault(task, {})
        cell = per_task.setdefault(candidate, {"alpha": 1.0, "beta": 1.0, "gain": 0.0, "n": 0})
        cell["alpha"] += 1.0 if accepted else 0.0
        cell["beta"] += 0.0 if accepted else 1.0
        cell["gain"] += float(gain)
        cell["n"] += 1
        self.episodes.append({"task": task, "candidate": candidate, "gain": float(gain),
                              "accepted": bool(accepted), "round": round_no, "note": note})
        self.save()

    # ---------------------------------------------------------- 读取
    def trials(self, task, candidate):
        cell = self.stats.get(task, {}).get(candidate)
        return 0 if not cell else int(cell["n"])

    def mean(self, task, candidate):
        cell = self.stats.get(task, {}).get(candidate)
        if not cell:
            return 0.5
        return cell["alpha"] / (cell["alpha"] + cell["beta"])

    def mean_gain(self, task, candidate):
        cell = self.stats.get(task, {}).get(candidate)
        if not cell or not cell["n"]:
            return 0.0
        return cell["gain"] / cell["n"]

    def ucb(self, task, candidate, explore=0.3):
        cell = self.stats.get(task, {}).get(candidate)
        if not cell or not cell["n"]:
            return 1.0  # 没试过的优先
        total = sum(c["n"] for c in self.stats.get(task, {}).values())
        return self.mean(task, candidate) + explore * math.sqrt(math.log(max(2, total)) / cell["n"])

    def best(self, task, names):
        best_name, best_score = None, -1e9
        for name in names:
            if self.trials(task, name) == 0:
                continue
            score = self.mean_gain(task, name)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    def summary(self):
        return {task: {cand: round(cell["alpha"] / (cell["alpha"] + cell["beta"]), 3)
                       for cand, cell in cells.items()}
                for task, cells in self.stats.items()}

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stats": self.stats, "episodes": self.episodes[-200:]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)