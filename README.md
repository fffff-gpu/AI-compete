# 智训引擎 · 工业模型自进化 Harness（企业赛 1.5.1）

参赛作品「工业模型自进化系统」的 Harness 侧源代码：在官方 CPU 基线上做**受控续训闭环**——
每轮先测当前检查点，再由大模型在封闭配方空间内选动作，训练后经门禁判定是否采纳，
被拒轮次不影响下一轮起点。A 气体六分类 / B 轴承 RUL / C 发动机 RUL 三个数据集的
全部训练、微调、调整与评估实验都以 SwanLab run 留痕，材料对照见 `docs/提交材料索引.md`。

交付版本 **zhixun-harness 1.5.1**，对应镜像 `zhixun-harness:1.5`
（`docker save` 归档 65,976,320 B，sha256 `9ff79a47…3345`，12 层，linux/amd64）。

## 结果（隐藏验证集口径）

| 数据 | 指标 | 静态基线 | 本系统 1.5.1 | 增益 | 最差阶段 | 采纳 |
|---|---|---|---|---|---|---|
| A 气体六分类 | macro-F1 ↑ | 0.7140 | **0.7258** | +0.0086（0.44σ，不显著） | 0（合规空间内不可达） | 1/10 |
| B 轴承 RUL | RMSE ↓ | 681.1598 | **497.7463** | +183.41（≈20σ） | +169.76 | 1/10 |
| C 发动机 RUL | RMSE ↓ | 52.8686 | **51.4884** | −2.61% | +1.25 | 12 轮 |

对照：同数据集同轮次只改采纳规则，A 无门禁掉到 0.6296（−9.6 点），B 平均 648.88 且末段 −108.6。

## 目录

```
harness-1.5/      镜像实际运行的内容：harness.py（SDK v2 入口）、contest_submission.py（选手模块）、
                  rulkit/（数据扁平化 + 865 参数级 MLP）、contest_harness/（组委会 SDK 1.0.4 原件）
engine-1.4/       上一代完整引擎（漂移监测 monitor.py、漂移合成 driftsim.py、门禁 control.py、
                  治理 governance.py、记忆 memory.py、记录 records.py）；用于离线实验与论证，不是镜像入口
configs/          四个交付配方与决策提示词（recipes.json）
scripts/          A/B/C 评估、对照、漂移探针、出表与 SwanLab 记录脚本
tests/            协议契约、投影校验、治理不变量、配方开关等本地测试
finetune-legacy/  决策模型后训练（Qwen3.8-27B LoRA SFT）的数据构建、训练与离线评估
evidence/         进程内 Harness 替身 + 3 数据集 × 4 配方回归脚本（1.5.1 缺陷回归）
docs/             系统运行说明、项目说明书（Markdown 源）及其 PDF 生成/版式检查脚本
tools/            镜像归档修补与校验、源码打包
```

## 快速开始

镜像 `harness-1.5/` 与镜像内 `/app` 逐字节一致，依赖只有 `numpy==1.26.4`（Python 3.10，线程固定 1）。

```bash
# 协议自检（无需数据、模型与网络）
python3 harness-1.5/harness.py describe
# {"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc",
#  "sdk":"contest_harness","sdk_version":"1.0.4","tasks":["classification","regression"]}

# 构建镜像（可选；离线 wheel 已随赛题提供）
docker build --platform=linux/amd64 --provenance=false --sbom=false \
  -t zhixun-harness:1.5 harness-1.5/

# 平台入口：由平台注入 /data /previous /state /model /workspace
python3 /app/harness.py train
python3 /app/harness.py predict
```

评测复现、异常处理与三级回滚见 `docs/系统运行说明-1.5.md`。

## 1.5.1 修了什么

平台预检指出三处缺陷，全部修复并有本地回归：

| 缺陷 | 修复 |
|---|---|
| C 集四个配方都抛 `AttributeError: 'Model' object has no attribute 'recipe_notes'` | `Model.fit` 懒初始化、`finish_swa` 不再清空诊断、`load` 显式置空 |
| A 集可能被要求跑中心化损失（分类任务非法） | `ALLOWED_BY_TASK`：分类只 `official` / `long_train_swa`，越界回落默认并标 `recipe-defaulted` |
| 决策只看行数与父元信息，没有当轮实测 | 决策前测当前检查点（accuracy+macro_f1 / rmse+mae；无验证集时按 id 哈希抽 20%、≤512 行作 `holdout-1/5` 探针） |

另加三级回退：所选配方异常 → 重载父检查点改跑默认配方 → 再失败跑 `official`，
`attempts` 落进 `recipe.json` 与 `configuration` 便于审计。回归：

```bash
python3 evidence/repro_recipe_failures.py
# 3 datasets × 4 recipes = 12/12 OK；probe path 3/3 OK；failures: 0
```

## 文档

- `docs/系统运行说明-1.5.md`：运行环境、依赖、启动命令、配置参数、输入输出、离线运行、评测复现、异常处理、回滚
- `docs/项目说明书-1.5.md`：系统架构、漂移监测与诊断、模型更新与受控自改进、静态基线对照、三本账（算力/标注/Token）、安全控制与工业落地
- `docs/提交材料索引.md`：企业赛七项材料对照与生成脚本

PDF 由 `docs/_md_pdf.py` 从 Markdown 直接生成（matplotlib 矢量输出，不依赖 Office）：

```bash
python3 docs/_md_pdf.py docs/系统运行说明-1.5.md --title "系统运行说明（企业赛 · 1.5.1）"
```

## 第三方组件与许可证

见 `THIRD_PARTY.md` 与 `LICENSES/`：`numpy`（BSD-3-Clause）、Python（PSF）、
组委会 `contest_harness` 1.0.4（随赛题发布，按其条款使用）、
`Qwen3.8-27B`（Apache-2.0，仅作决策模型，权重不在本仓库分发）。

## 合规

- 训练只用 `/data/train.jsonl`，监测/门禁用 `/data/validation.jsonl`；未读取或推断隐藏标签，不含隐藏测试集；
- 镜像、源码、文档中不含 API Key、账号口令或任何凭据；模型调用密钥只在平台密钥栏；
- 驱动 Harness 的大模型为 PyTRIO 的 Qwen3.8-27B，或本队经该平台训练所得的模型。

---

队伍：智训引擎　｜　赛题：人工智能+ 大赛 · 企业赛 · 工业模型自进化（CPU 赛道）