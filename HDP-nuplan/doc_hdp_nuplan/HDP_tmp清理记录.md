# HDP-nuplan/tmp 清理记录

## 清理目标

仅保留与以下两个最终模型及其可复盘流程直接相关的文件：

- HDP B Epoch10；
- RL Epoch2（TTC 1 秒、expert anchor 0.1、10k、seed2026、从 B Epoch10 初始化）。

同时保留这两个模型使用的训练数据、参数、日志、最终 checkpoint、fixed200/Test14 评测数据与结果，以及用于公平对比的对齐原始 Diffusion Planner 结果。

## 清理前状态

- `HDP-nuplan/tmp`：约 136 GiB；
- `/home/yanjun/NewDisk` 可用空间：约 23 GiB；
- 清理前确认没有训练、评测或数据处理进程正在使用 `HDP-nuplan/tmp`。

## 保留的核心目录

- `mini_train_full_306801_seed3407_v1`：完整 mini-train 数据、B Epoch10 训练记录与 checkpoint；
- `mini_train_balanced_10000_seed3407_v1`：RL Epoch2 使用的 10k 数据、指定 RL checkpoint 与对应评测；
- `test14_full_remote_subset`：Test14 数据、配置和 B Epoch10/RL Epoch2 完整评测结果；
- `original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10`：公平对齐的原始 Diffusion Planner 对照实验。

## 目录内 checkpoint 精简规则

- B 分支只保留 `source_model_epoch_4_trainloss_0.0265.pth` 和 `model_epoch_10_trainloss_0.0091.pth`；
- B 分支删除 Epoch5～9、Epoch11～20 和指向 Epoch20 的 `latest.pth`；
- 共同阶段目录只保留 Epoch4 checkpoint、参数和训练证据；
- 对齐原始 DP 只保留最终 Epoch10 checkpoint、参数、日志和评测结果；
- RL 目录只保留指定 seed2026 的 RL Epoch2 checkpoint、训练日志，以及 fixed200、Test14-random、Test14-hard 结果。

## 删除内容

删除以下不属于最终 B Epoch10/RL Epoch2 主线的内容：

- smoke/pilot 数据与训练；
- 旧 10k 监督模型复现实验；
- 11,040、12,109、16,210、21,409 和 100k 等中间数据实验；
- 未对齐的原始 Diffusion Planner 实验；
- RL seed42、6 epoch、drivable-area、noise0.2、softmax、reference-relative、proxy-v2/v3 等替代或失败方案；
- 与上述试验对应的 checkpoint、PID、launcher 输出和评测目录；
- 已被最终结果替代的中间 checkpoint。

## 恢复说明

这些删除操作为直接释放磁盘空间，不经过回收站；未提交到 Git 或另行备份的被删缓存、checkpoint 和日志不能从当前工作区直接恢复。

## 清理后校验

- `HDP-nuplan/tmp` 从约 136 GiB 降至约 120 GiB；
- `/home/yanjun/NewDisk` 可用空间从约 23 GiB 增至约 40 GiB；
- B Epoch10 checkpoint SHA256：`22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce`；
- RL Epoch2 checkpoint SHA256：`8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c`；
- 对齐原始 DP Epoch10 checkpoint SHA256：`37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786`；
- fixed200、Test14-random、Test14-hard 的最终结果和复现配置均保留。
