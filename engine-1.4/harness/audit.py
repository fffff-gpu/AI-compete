"""Audit a run directory without retraining.

    cd official
    python -m harness.audit runs/a_10r
    python -m harness.audit runs/a_10r --json

Checks the SHA-256 lineage chain, then prints the decision/verdict/ledger summary
that the report needs. Exits 1 when the chain is broken, so CI can gate on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .governance import LineageLog


def collect(run: Path) -> dict:
    out: dict = {"run": str(run), "exists": run.exists(),
                 "lineage": None, "kinds": {}, "results": None, "ledger": None}

    lineage_path = run / "lineage.jsonl"
    if lineage_path.exists():
        try:
            log = LineageLog(lineage_path, reset=False)
            out["lineage"] = log.summary()
            out["kinds"] = out["lineage"].get("kinds", {})
        except (OSError, ValueError) as exc:
            out["lineage"] = {"path": str(lineage_path), "events": 0, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}", "kinds": {}}
            out["kinds"] = {}

    for name in ("ledger.json", "results.json"):
        path = run / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            payload = {"error": str(exc)}
        if name == "ledger.json":
            out["ledger"] = payload
        else:
            out["results"] = {key: payload.get(key) for key in
                              ("profile", "rounds", "accepted_rounds", "baseline",
                               "best", "worst_stage_gain", "train_seconds_total")}
    return out


def render(payload: dict) -> str:
    lines = [f"run: {payload['run']}"]
    if not payload["exists"]:
        return "\n".join(lines + ["  (目录不存在)"])

    lineage = payload.get("lineage")
    if lineage:
        status = "ok" if lineage["ok"] else f"BROKEN at {lineage['error']} (event {lineage['events']})"
        lines.append(f"lineage: {lineage['events']} events, chain {status}")
        lines.append("kinds: " + ", ".join(f"{key}={value}" for key, value in sorted(payload["kinds"].items())))
    else:
        lines.append("lineage: (missing lineage.jsonl)")

    results = payload.get("results")
    if results:
        lines.append(f"profile: {results.get('profile')}  rounds: {results.get('rounds')}  "
                     f"accepted: {results.get('accepted_rounds')}  "
                     f"worst_stage_gain: {results.get('worst_stage_gain')}")
        lines.append(f"baseline: {results.get('baseline')}")
        lines.append(f"best:     {results.get('best')}")

    ledger = payload.get("ledger")
    if ledger:
        lines.append(f"ledger: cost_units={ledger.get('cost_units')} tokens={ledger.get('tokens')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory, e.g. runs/a_10r")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    payload = collect(Path(args.run))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(render(payload))

    lineage = payload.get("lineage")
    return 1 if lineage and not lineage.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())