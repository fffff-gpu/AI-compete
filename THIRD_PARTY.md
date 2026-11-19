# 开源组件与许可证

| 组件 | 版本 | 用途 | 许可证 |
|---|---|---|---|
| NumPy | 1.26.4 | 唯一的运行时第三方依赖（矩阵运算、随机数） | BSD-3-Clause，全文见 LICENSES/numpy-BSD-3-Clause.txt |
| Python | 3.10（python:3.10-slim 基础镜像） | 运行时 | PSF License；Debian 基础层按其各自许可证分发 |
| contest_harness | 1.0.4 | 组委会提供的 Harness SDK（入口协议与模型网关客户端） | 随赛题发布，按组委会使用条款使用，未修改 |
| Qwen3.8-27B | — | 决策大模型（平台侧托管，不随本包分发） | Apache-2.0（模型方许可证） |

本包不分发任何模型权重；`contest_harness/` 为组委会 SDK 原件，未改动。
