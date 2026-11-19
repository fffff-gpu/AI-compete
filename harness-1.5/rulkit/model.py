"""Small NumPy MLP + Adam. Checkpoints preserve every state needed to resume."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import zipfile

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= _THREADS <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)
import numpy as np

PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


def _recipe():
    """Optional training-recipe knobs; 0 disables each one."""
    return (float(os.environ.get("ZHIXUN_WD", "0") or 0),
            float(os.environ.get("ZHIXUN_LS", "0") or 0),
            float(os.environ.get("ZHIXUN_JITTER", "0") or 0),
            float(os.environ.get("ZHIXUN_CLIP", "5.0") or 5.0))


def _recipe_ext():
    """Extended, default-off recipe knobs; all off => official behaviour exactly."""
    loss = (os.environ.get("ZHIXUN_LOSS") or "mse").strip().lower()
    if loss not in ("mse", "center"):
        raise ValueError("ZHIXUN_LOSS must be 'mse' or 'center'")
    center = float(os.environ.get("ZHIXUN_CENTER", "0.35") or 0.35)
    if not 0.0 <= center <= 1.0:
        raise ValueError("ZHIXUN_CENTER must be within 0..1")
    sev = float(os.environ.get("ZHIXUN_SEV", "0") or 0)
    if not 0.0 <= sev <= 10.0:
        raise ValueError("ZHIXUN_SEV must be within 0..10")
    sev_end = (os.environ.get("ZHIXUN_SEV_END", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
    swa = int(os.environ.get("ZHIXUN_SWA", "0") or 0)
    if not 0 <= swa <= 64:
        raise ValueError("ZHIXUN_SWA must be within 0..64")
    return {"loss": loss, "center": center, "sev": sev, "sev_end": sev_end, "swa": swa}


_GROUP = re.compile(r"Bearing(\d+)_(\d+)")


def group_of(identity):
    """Bearing id of one training row: xjtu_Bearing1_1_s000001 -> Bearing1_1."""
    if not isinstance(identity, str):
        return "unknown"
    match = _GROUP.search(identity)
    if match:
        return "Bearing{}_{}".format(match.group(1), match.group(2))
    return "unknown"


def severity_column(paths, name=None):
    """Index of the damage-severity feature inside this checkpoint's schema.

    The delivered recipe weights rows by the within-bearing percentile of the
    vertical spectral RMS (the last feature in the official sorted schema, index 33);
    `ZHIXUN_SEV_COLUMN=band3` switches the same weighting to vertical band3 energy,
    which the data-mining notes argue is the most transferable dimension.
    """
    wanted = (name or os.environ.get("ZHIXUN_SEV_COLUMN") or "spectral_rms").strip().lower()
    key = "band3_energy" if wanted.startswith("band3") else "spectral_rms"
    for index, path in enumerate(paths or ()):
        text = str(path)
        if key in text and "vertical" in text:
            return index
    return None          # 任务 C 的 24 维没有这一列：调用方跳过严重度，而不是崩溃


def severity_percentile(x, ids, paths=None):
    """Within-bearing percentile of one feature column (rank-based, scale free)."""
    column = severity_column(paths)
    if column is None or x.ndim != 2 or x.shape[1] <= column:
        raise ValueError("severity weighting needs a column carrying vertical spectral_rms")
    values = np.asarray(x[:, column], dtype=np.float64)
    groups = np.array([group_of(identity) for identity in ids])
    out = np.zeros(len(values), dtype=np.float64)
    for name in np.unique(groups):
        mask = groups == name
        sub = values[mask]
        if len(sub) < 2:
            out[mask] = 0.5
            continue
        out[mask] = np.argsort(np.argsort(sub)).astype(np.float64) / (len(sub) - 1)
    return out


class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42, learning_rate=.003, batch_size=128):
        if task not in ("classification", "regression") or not 1 <= hidden <= 128:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= 100000:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
                       "b1": np.zeros(hidden), "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
                       "b2": np.zeros(output)}
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self._swa = []
        self.recipe_notes = []    # 运行时才发现的问题（如严重度列缺失），只进报告、不进 meta
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {"format": FORMAT, "device": "cpu", "task": task, "paths": paths,
                     "classes": list(classes), "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                     "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
                     "target_mean": float(y.mean()) if task == "regression" else 0.0,
                     "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0}
        self.validate()

    def validate(self):
        m = self.meta
        if m.get("format") != FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= 128 or not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= m["seed"] < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096 or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"]):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size, hidden = len(m["paths"]), m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64") or self.mean.shape != (input_size,) or self.scale.shape != (input_size,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")

    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def fit(self, x, y, epochs, ids=None):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        _wd, _ls, _jitter, _clip = _recipe()
        extended = _recipe_ext()
        if not hasattr(self, "recipe_notes"):
            # A model rebuilt by load() must accept the same diagnostic notes as
            # a freshly constructed one; the sev-skip branch below appends here.
            self.recipe_notes = []
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        if extended["loss"] != "mse" and self.meta["task"] != "regression":
            raise ValueError("ZHIXUN_LOSS=center is only defined for regression")
        if extended["sev"] > 0 and self.meta["task"] != "regression":
            raise ValueError("ZHIXUN_SEV is only defined for regression")
        sample_weights, groups, weight_sum = None, None, 0.0
        if extended["sev"] > 0:
            if ids is None:
                raise ValueError("ZHIXUN_SEV requires training row ids")
            if severity_column(self.meta["paths"]) is None:
                # 这个任务的输入里没有 vertical spectral_rms（例如 C 的 24 维）：
                # 静默跳过比崩溃更难查，所以留一条给报告看的说明，权重退回等权。
                self.recipe_notes.append("sev_skipped:no-vertical-spectral-column")
            else:
                weights = 1.0 + extended["sev"] * severity_percentile(x, ids, paths=self.meta["paths"])
                if extended["sev_end"]:
                    weights = weights / np.sqrt(np.asarray(y, dtype=np.float64) + 50.0)
                sample_weights = weights / float(weights.mean())
        if extended["loss"] == "center":
            if ids is None:
                raise ValueError("ZHIXUN_LOSS=center requires training row ids")
            groups = np.array([group_of(identity) for identity in ids])
        epoch_losses = []
        for _ in range(epochs):
            # Epoch-indexed shuffling makes split/resumed execution reproduce a continuous run.
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                if _jitter > 0:
                    batch = batch + _jitter * rng.standard_normal(batch.shape)
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits -= logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    if _ls > 0:
                        delta = delta - _ls / probabilities.shape[1]
                        delta[np.arange(len(indices)), expected] -= 1 - _ls
                    else:
                        delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    if groups is not None:
                        batch_groups = groups[indices]
                        centered = error.copy()
                        for group in np.unique(batch_groups):
                            mask = batch_groups == group
                            centered[mask] = error[mask] - float(error[mask].mean())
                        lam = extended["center"]
                        lam_loss = lam * error ** 2 + (1.0 - lam) * centered ** 2
                        lam_delta = 2.0 * (lam * error + (1.0 - lam) * centered)
                        if sample_weights is not None:
                            batch_weights = sample_weights[indices]
                            weight_sum = float(batch_weights.sum())
                            loss_sum += float(np.sum(lam_loss * batch_weights))
                            delta = (lam_delta * batch_weights / weight_sum)[:, None]
                        else:
                            loss_sum += float(np.sum(lam_loss))
                            delta = (lam_delta / len(indices))[:, None]
                    elif sample_weights is not None:
                        batch_weights = sample_weights[indices]
                        weight_sum = float(batch_weights.sum())
                        loss_sum += float(np.sum(error ** 2 * batch_weights))
                        delta = (2 * error * batch_weights / weight_sum)[:, None]
                    else:
                        loss_sum += float(np.sum(error ** 2))
                        delta = (2 * error / len(indices))[:, None]
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                if _wd > 0:
                    gradients["w1"] = gradients["w1"] + _wd * self.params["w1"]
                    gradients["w2"] = gradients["w2"] + _wd * self.params["w2"]
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, _clip / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            if extended["swa"] > 0:
                self._swa.append({key: value.copy() for key, value in self.params.items()})
                while len(self._swa) > extended["swa"]:
                    self._swa.pop(0)
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses

    def finish_swa(self):
        """Average the last N epoch snapshots (ZHIXUN_SWA); 0 snapshots is a no-op.

        Diagnostics collected during fit stay on the object: the caller reports
        them, so clearing them here would silently drop the sev-skip note.
        """
        if not self._swa:
            return 0
        count = len(self._swa)
        for name in self.params:
            self.params[name] = np.mean([snap[name] for snap in self._swa], axis=0)
        self._swa = []
        if not hasattr(self, "recipe_notes"):
            self.recipe_notes = []
        self.validate()
        return count

    def predict(self, x):
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            return result / result.sum(axis=1, keepdims=True)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    def save(self, path):
        self.validate()
        arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()}, "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        # Fixed ZIP metadata avoids changing the identity solely because of wall-clock time.
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS), *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist()]
            if len(names) != len(expected) or set(names) != {name + ".npy" for name in expected} or sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            model.meta = json.loads(str(metadata))
            model.params = {k: arrays[k].copy() for k in PARAMETERS}
            model.first = {k: arrays["adam_first_" + k].copy() for k in PARAMETERS}
            model.second = {k: arrays["adam_second_" + k].copy() for k in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
            model._swa = []
            model.recipe_notes = []
        model.validate()
        return model


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
