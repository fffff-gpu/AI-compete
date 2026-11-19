"""Re-run the platform train frame for every dataset x recipe.

Covers the three reported defects:
  1. Model.load + finish_swa dropped recipe_notes, so the C sev-skip path raised;
  2. the decision could hand a center-loss recipe to a classification round;
  3. the decision must always see a fresh measurement, even with no validation rows.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "official" / "sdk"))

import sim_sdk  # noqa: E402

SPEC = {
    "A": ("classification", "macro_f1", "higher"),
    "B": ("regression", "rmse", "lower"),
    "C": ("regression", "rmse", "lower"),
}
RECIPES = ["official", "center035_sev2_end", "center04_end", "long_train_swa"]


def run(dataset, recipe, epochs=2, validation=None):
    task, metric, direction = SPEC[dataset]
    parent = ROOT / "official" / "artifacts" / dataset / "checkpoint.npz"
    reply = json.dumps({"recipe": recipe, "epochs": epochs,
                        "hypothesis": "repro", "change_summary": "repro"}, ensure_ascii=False)
    rnd, host, work = sim_sdk.run_train(dataset, task, metric, direction, parent, reply,
                                        validation_override=validation)
    detail = json.loads((work / "candidate-r1" / "recipe.json").read_text(encoding="utf-8"))
    return detail


def empty_validation():
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    handle.close()
    return Path(handle.name)


def main():
    failures = 0
    for dataset in ("A", "B", "C"):
        for recipe in RECIPES:
            try:
                detail = run(dataset, recipe)
            except Exception as error:  # noqa: BLE001
                failures += 1
                print("FAIL %s %-20s %s: %s" % (dataset, recipe, type(error).__name__, error), flush=True)
                continue
            print("OK   %s %-20s -> %-20s requested=%-20s before=%s after=%s notes=%s" % (
                dataset, recipe, detail["recipe"], detail["requested_recipe"],
                json.dumps(detail["before"], ensure_ascii=False), json.dumps(detail["after"], ensure_ascii=False),
                detail["notes"]), flush=True)

    print("\n-- no validation rows mounted (probe path) --", flush=True)
    empty = empty_validation()
    for dataset in ("A", "B", "C"):
        try:
            detail = run(dataset, "official", validation=empty)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print("FAIL %s probe %s: %s" % (dataset, type(error).__name__, error), flush=True)
            continue
        print("OK   %s probe=%s rows_before=%s before=%s" % (
            dataset, detail["probe"], detail["validation_rows"] if "validation_rows" in detail else "-",
            json.dumps(detail["before"], ensure_ascii=False)), flush=True)

    print("\nfailures:", failures, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())