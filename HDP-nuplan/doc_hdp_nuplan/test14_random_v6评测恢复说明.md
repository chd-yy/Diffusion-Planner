# RL v6 `test14-random` 评测恢复说明

## 目的

恢复一次被中断的 RL v6 闭环评测，不重复运行已经完成的场景，也不改变模型、数据集、场景集合和评测配置。

## 评测对象

- 模型：RL v6，来自 HDP B Epoch 10。
- RL checkpoint：`rl_safety_progress_filtered_10k_seed2026_from_b10/.../model_epoch_2_trainloss_0.0010.pth`。
- 场景集合：原 `test14-random` 的 186 个场景。
- 原并行评测：完成 152 个场景后进程退出，没有生成完整最终报告。
- 恢复评测：只对缺失的 34 个场景使用 `test14-random-v6-missing34.yaml` 串行补跑。

## 为什么可以合并

NuPlan 的 metric 临时文件按场景保存。恢复任务使用相同的实验 UID 和输出目录，因此保留的 152 个临时结果与补跑的 34 个结果可以共同进入同一个聚合 parquet。脚本会检查最终数量为 186，并复用原 B Epoch 10 汇总，不重新评测 B 基线。

## 运行与结果文件

启动脚本：`HDP-nuplan/scripts/resume_v6_test14_random_missing34.sh`

结果文件：

`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_progress_filtered_10k_seed2026_test14_random_eval_parallel2/b10_vs_rl_v6_test14_random.json`

由于恢复时的 `runner_report.parquet` 只记录补跑的 34 个场景，最终 JSON 保留 186 个场景的聚合规划指标，但不报告不完整的运行时均值；脚本会将运行时字段置为 `null`，避免把 34 个场景的运行时间误当成 186 个场景的运行时间。
