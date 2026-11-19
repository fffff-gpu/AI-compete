"""审计日志：每轮一个 JSON 事件，外加一份汇总。"""
import json
import time
from pathlib import Path


class Journal:
    def __init__(self, path, reset: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()

    def log(self, kind: str, **payload) -> dict:
        event = {"t": round(time.time(), 3), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def events(self, kind: str | None = None) -> list:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if kind is None or r.get("kind") == kind]


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")