"""宽度迁移探针：只改隐藏宽度（官方 checkpoint schema 允许 1..128），其余不动。

做法：
  1. 读官方 checkpoint；
  2. 新 w1 = [旧 w1 | 小随机噪声]，新 w2 = [旧 w2 / 小随机噪声]，b1 后面补 0；
  3. Adam 一阶/二阶清零（新参数从零动量开始），mean/scale 原样保留（标准化只由初始训练集拟合）；
  4. 用官方 Model.load 校验（schema、形状、参数量、float64），再续训 N 轮、在验证集上打分。

用法（cwd = official/）：
  python scripts/probe_width.py A 64 5
  python scripts/probe_width.py B 64 5
  python scripts/probe_width.py C 96 2
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PACKAGES = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def load_api(profile):
    pkg = WORKSPACE / PACKAGES[profile]
    sys.path.insert(0, str(pkg))
    for name in [m for m in sys.modules if m.startswith("baselines")]:
        del sys.modules[name]
    model_mod = importlib.import_module("baselines.model")
    data_mod = importlib.import_module("baselines.data")
    return pkg, model_mod, data_mod


def migrate(ckpt: dict, hidden: int) -> dict:
    w1, b1, w2, b2 = ckpt["w1"], ckpt["b1"], ckpt["w2"], ckpt["b2"]
    n_in, old_h = w1.shape
    out = w2.shape[1]
    if hidden == old_h:
        return dict(ckpt)
    rng = np.random.default_rng(20261112)
    extra = hidden - old_h
    pad_w1 = rng.normal(0, math_scale(n_in), (n_in, extra))
    pad_w2 = rng.normal(0, math_scale(hidden), (extra, out))
    new = {
        "w1": np.hstack([w1, pad_w1]),
        "b1": np.concatenate([b1, np.zeros(extra)]),
        "w2": np.vstack([w2, pad_w2]),
        "b2": b2.copy(),
        "adam_first_w1": np.zeros((n_in, hidden)),
        "adam_first_b1": np.zeros(hidden),
        "adam_first_w2": np.zeros((hidden, out)),
        "adam_first_b2": np.zeros(out),
        "adam_second_w1": np.zeros((n_in, hidden)),
        "adam_second_b1": np.zeros(hidden),
        "adam_second_w2": np.zeros((hidden, out)),
        "adam_second_b2": np.zeros(out),
        "mean": ckpt["mean"].copy(),
        "scale": ckpt["scale"].copy(),
    }
    meta = json.loads(str(ckpt["metadata"]))
    meta["hidden"] = hidden
    meta["parameter_count"] = int(n_in * hidden + hidden + hidden * out + out)
    new["metadata"] = np.array(json.dumps(meta, sort_keys=True))
    return new


def math_scale(fan_in: int) -> float:
    return float(np.sqrt(1.0 / max(fan_in, 1)))


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    profile, hidden, epochs = argv[1], int(argv[2]), int(argv[3])
    pkg, model_mod, data_mod = load_api(profile)
    Model = model_mod.Model
    task = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8")).get("task", "classification")
    meta_src = json.loads(str(np.load(pkg / "checkpoint.npz")["metadata"]))
    task = meta_src["task"]
    classes = meta_src["classes"] or None
    train = data_mod.read_dataset(pkg / "public/train.jsonl", task, classes, meta_src["paths"])
    val = data_mod.read_dataset(pkg / "public/validation.jsonl", task, classes, meta_src["paths"])

    src = dict(np.load(pkg / "checkpoint.npz"))
    mig = migrate(src, hidden)
    out_dir = ROOT / "runs" / f"_width_{profile}_{hidden}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "checkpoint.npz", **mig)

    model = Model.load(out_dir / "checkpoint.npz")
    before = score(Model.load(pkg / "checkpoint.npz"), val, task, model_mod)
    model.fit(train["x"], train["y"], epochs)
    after = score(model, val, task, model_mod)
    print(f"[{profile}] hidden {meta_src['hidden']} -> {hidden}, {epochs} epoch, "
          f"params {model.meta['parameter_count']}")
    print(f"  baseline(官方)  {fmt(before)}")
    print(f"  migrated+{epochs}ep {fmt(after)}")
    return 0


def score(model, data, task, model_mod):
    pred = model.predict(data["x"])
    if task == "classification":
        labels = pred.argmax(axis=1)
        y = data["y"]
        f1 = [2 * np.sum((labels == c) & (y == c)) / (np.sum(labels == c) + np.sum(y == c))
              for c in sorted(set(y.tolist()))]
        return {"macro_f1": float(np.mean(f1))}
    err = pred - data["y"]
    return {"rmse": float(np.sqrt(np.mean(err ** 2)))}


def fmt(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))