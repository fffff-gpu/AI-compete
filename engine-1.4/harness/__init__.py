"""自研 Harness（续训版）——不依赖官方 baselines 的调度代码，只调用它的训练/评测 API。

设计沿用第一版 harness 的四条不变量：
  I1 只从"被采纳的检查点"继续（拒绝就回退，绝不让坏模型滚雪球）
  I2 每轮只改一个变量，改了什么、依据什么，写进 journal
  I3 采纳要过门禁（验证集增益 >= min_gain），不是训练完就算
  I4 全程记账（训练秒、轮次、LLM token），并对每个决策记录可证伪假设
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("BASELINE_CPU_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # .../official
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import win_compat  # noqa: E402,F401  (must precede the official trainer)


def load_env(path=None):
    """把 official/.env 里的键读进 os.environ（不覆盖已有值，不打印值）。"""
    env = Path(path) if path else ROOT / ".env"
    if not env.exists():
        return {}
    loaded = {}
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = True
    return loaded


load_env()

__all__ = ["ROOT", "load_env"]