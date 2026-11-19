"""Continuation trainer for contest.harness.workspace.v2.

The platform runs this module through ``/app/harness.py``:

    python3 /app/harness.py describe
    python3 /app/harness.py train      # one round, one `start` frame on stdin
    python3 /app/harness.py predict    # one sample per stdin line

Only three things live here: how one round trains from the platform parent
checkpoint, how the recipe for that round is chosen (one real model call is
mandatory per round), and how a mounted checkpoint turns one sample into a
prediction. Measurement, metering and labels stay with the host.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from contest_harness import Round
from rulkit.data import flatten
from rulkit.model import Model

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

# Recipes are the delivered, data-mined configurations; the model call below may
# pick one of them or the official default, but never invent free-form knobs.
RECIPES = {
    "official": {"loss": "mse", "wd": 0.0, "swa": 0, "sev": 0.0, "sev_end": False, "epochs": {"A": 3, "B": 2, "C": 8}},
    "center035_sev2_end": {"loss": "center", "wd": 0.20, "swa": 2, "sev": 2.0, "sev_end": True, "epochs": {"A": 3, "B": 8, "C": 12}},
    "center04_end": {"loss": "center", "wd": 0.20, "swa": 0, "sev": 0.0, "sev_end": False, "epochs": {"A": 3, "B": 8, "C": 12}},
    "long_train_swa": {"loss": "mse", "wd": 0.0, "swa": 2, "sev": 0.0, "sev_end": False, "epochs": {"A": 5, "B": 6, "C": 14}},
}
DEFAULT_BY_DATASET = {"A": "official", "B": "center035_sev2_end", "C": "long_train_swa"}
MAX_EPOCHS = 16

# Center loss and the severity weighting are defined for regression only; a
# classification round must never be offered them or Model.fit raises.
ALLOWED_BY_TASK = {
    "classification": ("official", "long_train_swa"),
    "regression": tuple(RECIPES),
}

# When the host mounts no validation rows the round still needs a fresh
# measurement, so a deterministic slice of the training rows becomes the probe.
PROBE_RATIO = 5
PROBE_MAX = 512

SYSTEM_HINT = (
    "你是工业模型续训的决策器。只输出一个 JSON 对象，键为 recipe、epochs、hypothesis、"
    "change_summary。recipe 必须取自 allowed 列表；epochs 必须是 1..%d 的整数。"
    "state.before 是父检查点在当前数据上的实测指标，state.history 是前几轮的配方与实测，"
    "请据此选择，不要复述它们。hypothesis 与 change_summary 各不超过 300 字，用中文。" % MAX_EPOCHS
)


# --------------------------------------------------------------------------- data
def _ids_matrix(rows, task, classes, expected_paths):
    """Rows -> (ids, x, y, paths); the feature schema must match the checkpoint."""
    ids, features, targets = [], [], []
    paths = expected_paths
    for row in rows:
        row_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = row_paths
        if row_paths != paths:
            raise ValueError("feature schema changed inside the projection")
        ids.append(row["id"])
        features.append(values)
        if task == "classification":
            targets.append(classes.index(row["answer"]))
        else:
            targets.append(float(row["answer"]))
    x = np.asarray(features, dtype=np.float64) if features else np.zeros((0, len(paths or ())), dtype=np.float64)
    y = np.asarray(targets, dtype=float if task == "regression" else np.int64)
    return ids, x, y, paths


def _metric(model, x, y, task):
    if len(x) == 0:
        return {}
    predicted = model.predict(x)
    if task == "classification":
        labels = predicted.argmax(axis=1)
        labels_present = sorted(set(int(v) for v in np.unique(y)) | set(int(v) for v in np.unique(labels)))
        scores = []
        for label in labels_present:
            hit = np.sum((labels == label) & (y == label))
            predicted_hits = np.sum(labels == label)
            actual = np.sum(y == label)
            precision = hit / predicted_hits if predicted_hits else 0.0
            recall = hit / actual if actual else 0.0
            scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        return {"accuracy": float(np.mean(labels == y)),
                "macro_f1": float(np.mean(scores)) if scores else 0.0}
    residual = np.asarray(predicted, dtype=np.float64) - y
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.mean(np.abs(residual)))}


def _profile(y, task, classes):
    """Cheap shape summary of the mounted labels, so the decision sees the round."""
    if not len(y):
        return {}
    if task == "classification":
        counts = {classes[index]: int(np.sum(y == index)) for index in range(len(classes))}
        return {"class_counts": counts, "imbalance": round(float(max(counts.values()) / max(1, min(counts.values()))), 2)}
    values = np.asarray(y, dtype=np.float64)
    return {"target_mean": round(float(values.mean()), 3), "target_std": round(float(values.std()), 3),
            "target_min": round(float(values.min()), 3), "target_max": round(float(values.max()), 3)}


def _probe_split(rows):
    """Deterministic 1/PROBE_RATIO slice used when no validation rows are mounted."""
    def bucket(row):
        return int(hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()[:8], 16) % PROBE_RATIO

    probe = [row for row in rows if bucket(row) == 0][:PROBE_MAX]
    rest = [row for row in rows if bucket(row) != 0]
    return rest, probe


def _history(rnd, limit=3):
    """Recent round outcomes the host exposes; absent history is not an error."""
    try:
        records = rnd.history()
    except Exception:  # noqa: BLE001 - history is advisory only
        return []
    out = []
    for record in records[-limit:]:
        if not isinstance(record, dict):
            continue
        configuration = record.get("configuration") if isinstance(record.get("configuration"), dict) else {}
        out.append({"id": record.get("id"),
                    "recipe": configuration.get("recipe"),
                    "epochs": configuration.get("epochs"),
                    "before": configuration.get("validation_before"),
                    "after": configuration.get("validation_after")})
    return out


# ---------------------------------------------------------------------- parent ckpt
def _parent_checkpoint(rnd):
    declared = list(rnd.parent.declared_paths())
    preferred = [name for name in declared if name.endswith(".npz")] or declared
    if not preferred:
        raise RuntimeError("parent artifact declares no file")
    name = "checkpoint.npz" if "checkpoint.npz" in preferred else preferred[0]
    # CONTEST_PARENT_ROOT exists only for offline tests; the host mount is /previous.
    root = os.environ.get("CONTEST_PARENT_ROOT")
    return rnd.parent.path(name, root=root) if root else rnd.parent.path(name)


# --------------------------------------------------------------------- decision
def _parse_json(text, fallback):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return dict(fallback), "no-json"
    try:
        value = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return dict(fallback), "bad-json"
    if not isinstance(value, dict):
        return dict(fallback), "not-object"
    return value, "ok"


def _choose(rnd, state, allowed, fallback_recipe, fallback_epochs):
    """One mandatory real model call; returns (receipt, recipe, epochs, notes)."""
    receipt = rnd.model(
        [
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": json.dumps({
                "state": state,
                "allowed": [{"name": name, "loss": RECIPES[name]["loss"], "wd": RECIPES[name]["wd"],
                             "swa": RECIPES[name]["swa"], "sev": RECIPES[name]["sev"],
                             "sev_end": RECIPES[name]["sev_end"]} for name in allowed],
                "default": {"recipe": fallback_recipe, "epochs": fallback_epochs},
            }, ensure_ascii=False)},
        ],
        max_tokens=min(600, rnd.max_output_tokens),
        temperature=0,
    )
    parsed, status = _parse_json(receipt.text, {"recipe": fallback_recipe, "epochs": fallback_epochs})
    recipe = parsed.get("recipe")
    if recipe not in allowed:
        recipe = fallback_recipe
        status = status + ":recipe-defaulted"
    try:
        epochs = int(parsed.get("epochs", fallback_epochs))
    except (TypeError, ValueError):
        epochs = fallback_epochs
    epochs = max(1, min(MAX_EPOCHS, epochs))
    notes = {
        "parse": status,
        "model_text": (receipt.text or "")[:600],
        "hypothesis": str(parsed.get("hypothesis") or "沿用已知配方继续该数据集的续训。")[:900],
        "change_summary": str(parsed.get("change_summary") or "recipe=%s epochs=%d" % (recipe, epochs))[:900],
    }
    return receipt, recipe, epochs, notes


def _apply_recipe(body, epochs):
    os.environ["ZHIXUN_LOSS"] = body["loss"]
    os.environ["ZHIXUN_WD"] = str(body["wd"])
    os.environ["ZHIXUN_SWA"] = str(body["swa"])
    os.environ["ZHIXUN_SEV"] = str(body["sev"])
    os.environ["ZHIXUN_SEV_END"] = "1" if body["sev_end"] else "0"
    return epochs


def _attempt_plan(recipe, epochs, fallback_recipe, fallback_epochs, dataset):
    """Chosen recipe first, then the dataset default, then the official recipe."""
    plan = [(recipe, epochs)]
    if recipe != fallback_recipe:
        plan.append((fallback_recipe, fallback_epochs))
    if all(name != "official" for name, _ in plan):
        plan.append(("official", int(RECIPES["official"]["epochs"].get(dataset, 3))))
    return plan


# ------------------------------------------------------------------------ train
def train(rnd: Round):
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    candidate = work / ("candidate-r%d" % rnd.round_index)
    candidate.mkdir(parents=True, exist_ok=False)

    task = rnd.task["task"]
    checkpoint = _parent_checkpoint(rnd)
    model = Model.load(checkpoint)
    if model.meta.get("task") != task:
        raise RuntimeError("parent checkpoint task %r does not match round task %r" % (model.meta.get("task"), task))
    classes = list(model.meta["classes"]) if task == "classification" else None

    train_rows = list(rnd.data.training_rows())
    try:
        valid_rows = list(rnd.data.validation_rows())
    except (FileNotFoundError, OSError):
        valid_rows = []
    probe_source = "validation"
    if not valid_rows:
        # No public validation rows for this round: keep a deterministic slice of
        # the training rows aside so the decision still sees a real measurement.
        train_rows, valid_rows = _probe_split(train_rows)
        probe_source = "holdout-1/%d" % PROBE_RATIO
    rows_ids, x, y, paths = _ids_matrix(train_rows, task, classes, model.meta["paths"])
    v_ids, v_x, v_y, _ = _ids_matrix(valid_rows, task, classes, paths)
    before = _metric(model, v_x, v_y, task)
    if not before:
        probe_source = "none"

    dataset = rnd.dataset_key
    fallback_recipe = DEFAULT_BY_DATASET.get(dataset, "official")
    fallback_epochs = int(RECIPES[fallback_recipe]["epochs"].get(dataset, 3))
    allowed = ALLOWED_BY_TASK[task]

    state = {
        "dataset_key": dataset, "round_index": rnd.round_index, "task": task,
        "classes": classes, "rows": len(rows_ids), "validation_rows": len(v_ids),
        "probe": probe_source, "before": before, "profile": _profile(y, task, classes),
        "parent": {"steps": model.meta.get("steps"), "epochs": model.meta.get("epochs"),
                   "hidden": model.meta.get("hidden"), "parameters": model.meta.get("parameter_count")},
        "history": _history(rnd),
    }
    receipt, recipe, epochs, notes = _choose(rnd, state, allowed, fallback_recipe, fallback_epochs)
    requested, requested_epochs = recipe, epochs

    attempts, model, swa_rounds, losses = [], model, 0, []
    for index, (name, count) in enumerate(_attempt_plan(recipe, epochs, fallback_recipe, fallback_epochs, dataset)):
        body = RECIPES[name]
        if index:
            # A failed attempt may have left partial state; restart from the parent.
            model = Model.load(checkpoint)
        try:
            _apply_recipe(body, count)
            losses = model.fit(x, y, count, ids=rows_ids)
            swa_rounds = model.finish_swa()
        except Exception as error:  # noqa: BLE001 - a recipe must never abort the round
            attempts.append({"recipe": name, "epochs": count,
                             "error": "%s: %s" % (type(error).__name__, str(error)[:240])})
            continue
        recipe, epochs = name, count
        break

    notes["attempts"] = attempts
    if not attempts:
        notes["first_try"] = recipe
    if attempts:
        notes["change_summary"] = (notes["change_summary"] + "（%s 失败后回退到 %s）"
                                   % (attempts[0]["recipe"], recipe))[:900]
    after = _metric(model, v_x, v_y, task)

    output = candidate / "checkpoint.npz"
    model.save(output)
    recipe_notes = list(getattr(model, "recipe_notes", []))
    (candidate / "recipe.json").write_text(json.dumps({
        "recipe": recipe, "requested_recipe": requested, "requested_epochs": requested_epochs,
        "epochs": epochs, "swa_rounds": swa_rounds, "attempts": attempts,
        "notes": recipe_notes, "losses": [round(float(v), 6) for v in losses],
        "probe": probe_source, "before": before, "after": after,
        "call_id": receipt.call_id,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    configuration = {
        "recipe": recipe, "epochs": epochs, "requested_recipe": requested, "requested_epochs": requested_epochs,
        "attempts": attempts, "swa_rounds": swa_rounds, "recipe_notes": recipe_notes,
        "probe": probe_source, "validation_before": before, "validation_after": after,
        "rows": len(rows_ids), "validation_rows": len(v_ids),
        "parse": notes["parse"], "model_call_id": receipt.call_id,
    }
    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=notes["hypothesis"],
        change_summary=notes["change_summary"],
        configuration=configuration,
        seed=int(model.meta.get("seed", 42)),
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=notes["hypothesis"],
        change_summary=notes["change_summary"],
        reason=("平台公开验证集 %s -> %s，配方 %s、%d epoch。模型回复（%s）：%s"
                % (json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False),
                   recipe, epochs, receipt.call_id, notes["model_text"][:200])),
        configuration=configuration,
        seed=int(model.meta.get("seed", 42)),
        model_call_ids=[receipt.call_id],
    )


# ---------------------------------------------------------------------- predict
_CACHE = {}


def _checkpoint_path(model_dir):
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for pattern in ("*.npz", "checkpoint*/**/*.npz", "**/*.npz"):
        found = sorted(Path(model_dir).glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError("no checkpoint mounted under %s" % model_dir)


def _load(model_dir):
    path = _checkpoint_path(model_dir)
    stamp = (str(path), path.stat().st_mtime_ns)
    if stamp not in _CACHE:
        _CACHE[stamp] = Model.load(path)
    return _CACHE[stamp]


def _scores(probabilities):
    """One label must hold the top score exactly; break ties by index."""
    scores = [float(value) for value in probabilities]
    top = max(scores)
    holders = [index for index, value in enumerate(scores) if value == top]
    if len(holders) > 1:
        for index in holders[1:]:
            scores[index] = max(0.0, top - 1e-9)
    return scores


def predict(sample, model_dir):
    model = _load(model_dir)
    task = model.meta.get("task")
    if sample.task not in (None, task):
        raise ValueError("sample task %r does not match checkpoint task %r" % (sample.task, task))
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if task == "regression":
        return {"value": max(0.0, float(output))}
    return [{"label": label, "score": score}
            for label, score in zip(model.meta["classes"], _scores(output))]