"""Assemble the enterprise-competition source package (zhixun-source-1.5.zip).

Layout (everything under submission/07-源代码/):
  zhixun-source-1.5/
    README.md              how to build, run, verify; what maps to which image layer
    THIRD_PARTY.md         open-source components and licenses
    requirements.txt       pinned runtime deps (numpy only)
    harness-1.5/           exactly what the image runs (SDK v2 entry + submission)
    engine-1.4/            full previous engine: drift monitor, control, governance
    configs/recipes.json   the four delivered recipes + defaults
    scripts/               experiment / eval / drift tooling
    tests/                 protocol + governance + projection tests
    finetune-legacy/       LoRA SFT code from the school stage (old data flow)
    LICENSES/              third-party license texts

No runs/, no *.jsonl, no *.npz, no credentials, no cloud-record index.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission" / "07-源代码"
STAGE = SUB / "zhixun-source-1.5"
ZIP = SUB / "zhixun-source-1.5.zip"
ARCH = ROOT.parent / "归档" / "zhixun-校内赛-20261119"

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo", ".jsonl", ".npz", ".tar", ".zip", ".env", ".swanlab"}


def copy_tree(src: Path, dst: Path, name_filter=None):
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        if any(part in SKIP_PARTS for part in item.parts):
            continue
        if item.suffix in SKIP_SUFFIX or item.name.startswith("."):
            continue
        rel = item.relative_to(src)
        if name_filter and not name_filter(rel):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # 1) 1.5 image content: exactly what Dockerfile copies
    h15 = STAGE / "harness-1.5"
    copy_tree(ROOT / "sdk", h15)
    for name in ("harness.py", "contest_submission.py"):
        pass  # already under sdk/
    shutil.copy2(ROOT / "Dockerfile", h15 / "Dockerfile")
    shutil.copy2(ROOT / "_export_image_15.ps1", h15 / "export-image.ps1")
    shutil.copy2(ROOT / "_verify_tar.py", h15 / "verify_tar.py")
    shutil.copy2(ROOT / "_strip_attestation.py", h15 / "strip_attestation.py")

    # 2) previous engine (monitoring / control / governance / journal)
    e14 = STAGE / "engine-1.4"
    copy_tree(ROOT / "harness", e14 / "harness")
    for name in ("predict.py", "win_compat.py"):
        shutil.copy2(ROOT / name, e14 / name)

    # 3) scripts / tests / deps
    copy_tree(ROOT / "scripts", STAGE / "scripts")
    copy_tree(ROOT / "tests", STAGE / "tests")
    shutil.copy2(ROOT / "requirements.txt", STAGE / "requirements.txt")

    # 4) finetune (legacy, code only)
    ft = STAGE / "finetune-legacy"
    ft.mkdir()
    for name in ("train_lora.py", "eval_policy.py", "build_dataset.py"):
        src = ARCH / "finetune" / name
        if src.exists():
            shutil.copy2(src, ft / name)
    (ft / "README.md").write_text(
        "# finetune-legacy\n\n"
        "校内赛阶段对 Qwen3.8-27B 做的 LoRA SFT（rank 32、2 epoch、93 条、约 7.4 万 token）。\n"
        "数据流是旧项目的 gas 数据，不是本赛道 A/B/C 投影；这里只留代码作“决策模型后训练”\n"
        "的方法记录，不参与 1.5 镜像默认决策。权重：trio://run_t46dhdn3u72b/sampler_weights/zhixun-policy-v1。\n",
        encoding="utf-8",
    )

    # 5) explicit config file
    cfg = STAGE / "configs"
    cfg.mkdir()
    sys.path.insert(0, str(ROOT / "sdk"))
    import contest_submission as cs

    (cfg / "recipes.json").write_text(
        json.dumps({"recipes": cs.RECIPES, "default_by_dataset": cs.DEFAULT_BY_DATASET,
                    "max_epochs": cs.MAX_EPOCHS, "system_hint": cs.SYSTEM_HINT},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 6) licenses
    lic = STAGE / "LICENSES"
    lic.mkdir()
    numpy_lic = Path(sys.prefix) / "Lib" / "site-packages" / "numpy" / "LICENSE.txt"
    if numpy_lic.exists():
        shutil.copy2(numpy_lic, lic / "numpy-BSD-3-Clause.txt")
    (STAGE / "THIRD_PARTY.md").write_text(
        "# 开源组件与许可证\n\n"
        "| 组件 | 版本 | 用途 | 许可证 |\n"
        "|---|---|---|---|\n"
        "| NumPy | 1.26.4 | 唯一的运行时第三方依赖（矩阵运算、随机数） | BSD-3-Clause，全文见 LICENSES/numpy-BSD-3-Clause.txt |\n"
        "| Python | 3.10（python:3.10-slim 基础镜像） | 运行时 | PSF License；Debian 基础层按其各自许可证分发 |\n"
        "| contest_harness | 1.0.4 | 组委会提供的 Harness SDK（入口协议与模型网关客户端） | 随赛题发布，按组委会使用条款使用，未修改 |\n"
        "| Qwen3.8-27B | — | 决策大模型（平台侧托管，不随本包分发） | Apache-2.0（模型方许可证） |\n\n"
        "本包不分发任何模型权重；`contest_harness/` 为组委会 SDK 原件，未改动。\n",
        encoding="utf-8",
    )

    # 7) README
    (STAGE / "README.md").write_text(
        """# zhixun-source-1.5

企业赛提交的完整源代码包（对应镜像 `zhixun-harness:1.5`）。不含权重、不含隐藏测试集、不含任何密钥。

## 目录

- `harness-1.5/`：镜像实际运行的内容。`harness.py` 是 SDK v2 入口（`python3 /app/harness.py describe|train|predict`），`contest_submission.py` 是选手模块（从父检查点续训 + 每轮一次真实模型调用选配方），`rulkit/` 是数据扁平化与 865 参数级 MLP，`contest_harness/` 是组委会 SDK 原件。
- `engine-1.4/`：上一代完整引擎（漂移监测 `monitor.py`、漂移合成 `driftsim.py`、门禁 `control.py`、治理 `governance.py`、记忆 `memory.py`、记录 `records.py`）。它用于离线实验和监测/归因论证，**不是** 1.5 镜像的入口。
- `scripts/`：A/B/C 三数据集的评估、对照、漂移探针、画图与 SwanLab 记录工具。
- `tests/`：协议、治理、投影、来源校验的单元测试。
- `configs/recipes.json`：四个交付配方、默认配方、epoch 上限与决策提示词。
- `finetune-legacy/`：校内赛阶段的 LoRA SFT 代码（旧数据流，仅作方法留档）。
- `Dockerfile`（在 `harness-1.5/` 内）：`python:3.10-slim` + `numpy==1.26.4`，构建期自检 `describe`。

## 构建与运行

```bash
# 构建（离线 wheel 已放在 official/_wheels；无 wheel 时走镜像源）
docker build --platform=linux/amd64 --provenance=false --sbom=false \
  -t zhixun-harness:1.5 harness-1.5/

# 本地协议自检
python3 harness-1.5/harness.py describe

# 平台入口（由平台注入 /data /previous /state /model /workspace）
python3 /app/harness.py train
python3 /app/harness.py predict
```

评测复现步骤、异常处理与回滚见提交包里的《系统运行说明》。

## 合规

- 训练只用 `/data/train.jsonl`，监测/门禁只用 `/data/validation.jsonl`；
- 未读取或推断隐藏标签；不含隐藏测试集；
- 镜像、源码、文档中不含 API Key、账号口令或任何凭据；模型调用密钥只在平台密钥栏；
- 驱动 Harness 的大模型为 PyTRIO 的 Qwen3.8-27B，或本队经该平台训练所得的模型。
""",
        encoding="utf-8",
    )

    # 8) zip
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for item in sorted(STAGE.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(SUB))

    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print("staged files:", sum(1 for f in STAGE.rglob("*") if f.is_file()))
    print("staged bytes:", size)
    print("zip bytes:", ZIP.stat().st_size)
    print("zip:", ZIP)


if __name__ == "__main__":
    main()