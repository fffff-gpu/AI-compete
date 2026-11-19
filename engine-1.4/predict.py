"""`/app/predict.py` = `python -m harness.predict` 的等价入口。

公告 11-13 要求镜像「提供独立预测入口」。平台可能按文件名找入口，也可能按模块找，
所以两条路径都给同一个实现，避免因为约定不同而判「不支持续训协议」：

    python -m harness.predict --selfcheck
    python predict.py --data /data/hidden_flow.jsonl -o /runs/preds.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.predict import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
