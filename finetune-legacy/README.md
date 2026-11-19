# finetune-legacy

校内赛阶段对 Qwen3.8-27B 做的 LoRA SFT（rank 32、2 epoch、93 条、约 7.4 万 token）。
数据流是旧项目的 gas 数据，不是本赛道 A/B/C 投影；这里只留代码作“决策模型后训练”
的方法记录，不参与 1.5 镜像默认决策。权重：trio://run_t46dhdn3u72b/sampler_weights/zhixun-policy-v1。
