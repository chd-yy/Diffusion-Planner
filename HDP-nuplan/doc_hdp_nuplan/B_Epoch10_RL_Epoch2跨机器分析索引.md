# B Epoch10 与 RL Epoch2 跨机器分析索引

## 用途

本索引用于在另一台电脑 clone/pull 仓库后，直接复盘 B Epoch10、RL Epoch2 的训练配置、数据范围和闭环评测结果。

## 首先阅读

1. `HDP_B_Epoch10训练过程详解.md`：B Epoch10 的权重继承、训练阶段和实际学习率；
2. `HDP_RL_Epoch2训练过程详解.md`：RL Epoch2 的 rollout、Replay Buffer、reward-weighted update 和专家 anchor；
3. `HDP_nuPlan训练评测与迭代矫正关键问题复盘.md`：训练与评测过程中确认的关键问题；
4. `Test14三模型参数与配置完整清单.md`：原始 DP、B Epoch10、RL Epoch2 的完整参数与 Test14 闭环配置。

## 已提交的参数与数据清单

- B Epoch10 参数：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/args.json`；
- B Epoch1～4 参数：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/args.json`；
- B 训练清单：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json`；
- RL Epoch2 参数：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json`；
- RL 训练清单：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json`；
- Test14 场景选择与分片：`HDP-nuplan/tmp/test14_full_remote_subset/` 下的 manifest 和 `config/scenario_filter/`。

这些训练清单只保存 NPZ 文件名，不包含 NPZ 本体。

## 已提交的核心结果

- fixed200：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_fixed200_eval_retry/`；
- Test14-random：`HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-random_three_models.json`；
- Test14-hard：`HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-hard_three_models.json`；
- B Epoch10 与 RL Epoch2 的逐场景配对分析：同一 `three_model_eval` 目录下的 `*_b10_vs_rl_epoch2_analysis.json` 和 Markdown；
- 对齐原始 Diffusion Planner 的 fixed200 结果：`HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/fixed200_eval/`。

## 未提交到普通 Git 的大文件

以下文件仍保留在本机，但由于当前仓库没有安装 Git LFS，没有纳入普通 Git commit：

- B Epoch10 checkpoint：`model_epoch_10_trainloss_0.0091.pth`，SHA256 为 `22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce`；
- RL Epoch2 checkpoint：`model_epoch_2_trainloss_0.0003.pth`，SHA256 为 `8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c`；
- 对齐原始 DP Epoch10 checkpoint：`model_epoch_10_trainloss_0.0618.pth`，SHA256 为 `37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786`；
- 306,801 个训练 NPZ、10,000 个 RL NPZ 和 Test14 DB。

只做代码、参数和结果分析时不需要这些大文件。若要在另一台电脑继续推理或闭环评测，需要单独传输对应 checkpoint、NuPlan DB 和地图，并用上述 SHA256 校验。
