# RL 训练集扩大操作日志

## 1. 目的

v8 使用 10,000 个 NPZ，在 fixed200 上得到 score `0.899675`，低于 B Epoch 10 的 `0.904567`。本实验先只扩大 RL 训练数据集，其他 reward、loss、学习率和训练轮数保持 v8 不变，用于隔离数据规模的影响。

## 2. 数据选择

本地已存在完整 mini-train 缓存：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache
```

共 `306,801` 个 NPZ。新实验选择 `100,000` 个：

- 保留旧 v8 的全部 `10,000` 个 NPZ；
- 从完整 mini-train 中排除这 10,000 个后，使用 seed `2026` 确定性抽取 `90,000` 个；
- 新清单不复制 NPZ，直接指向完整缓存目录，避免重复占用磁盘；
- manifest 内文件名唯一，并且全部能在完整缓存目录中找到。

生成并训练入口：

```bash
bash HDP-nuplan/scripts/run_rl_v8_proxy_v3_positive_advantage_100k_seed2026.sh
```

入口会先调用 `prepare_rl_expanded_manifest.py` 生成并审计清单，再启动训练。

## 3. 控制变量

沿用 v8：

```text
监督起点：B Epoch 10
train_epochs：2
reward：nuplan_score_proxy_v3
weighting：positive_advantage
learning_rate：4e-7
reference_anchor_weight：20.0
expert_anchor_weight：0.1
detach_window_size：0
rl_freeze_encoder：true
rl_group_size：32
rl_rollout_steps：6
rl_sampling_noise_scale：0.1
rl_min_reward_std：0.002
```

本次只改变数据规模及 buffer 容量：

```text
训练样本：10,000 → 100,000
rl_buffer_size：10,000 → 100,000
```

## 4. 当前状态

100k manifest 已生成并完成独立复核：

```text
target_size = 100000
unique_size = 100000
base_size = 10000
new_size = 90000
base_included = true
full_membership = true
duplicates = 0
```

生成文件：

```text
HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/diffusion_planner_training.json
HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/manifest_audit.json
```

城市分布：

```text
sg-one-north: 5484
us-ma-boston: 6999
us-nv-las-vegas-strip: 80104
us-pa-pittsburgh-hazelwood: 7413
```

目前尚未启动 100k RL 训练。启动命令为：

```bash
bash HDP-nuplan/scripts/run_rl_v8_proxy_v3_positive_advantage_100k_seed2026.sh
```

训练完成后，必须使用与 v8 相同的 fixed200 场景集合对 B Epoch 10 和新 checkpoint 做配对闭环评测。

## 5. 状态检查

2026-08-25 11:59 检查确认：

```text
100k manifest：存在，100000 条，唯一数 100000
RL 训练进程：未启动
100k checkpoint：尚未生成
100k 闭环评测：尚未开始
```

## 6. 训练与自动评测入口

为避免训练结束后遗漏评测，新增串行入口：

```bash
bash HDP-nuplan/scripts/run_rl_100k_and_eval.sh
```

该入口严格按以下顺序执行：

1. 使用 100k manifest 从 B Epoch 10 启动 RL；
2. 等待 Epoch 2 checkpoint 正常保存；
3. 自动使用与 v8 相同的 fixed200 场景集合评测新 checkpoint；
4. 复用已完成的 B Epoch 10 结果并生成逐场景配对分析。

## 7. 实际启动记录

2026-08-25 18:35:27 已使用后台串行入口正式启动：

```bash
nohup bash HDP-nuplan/scripts/run_rl_100k_and_eval.sh \
  > HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/train_and_eval_launcher.out 2>&1 &
```

后台主进程 PID：`12179`。

18:36:38 已完成 checkpoint 加载检查：

```text
Loaded pretrained model: missing=0, unexpected=0
```

随后进入 Epoch 1 rollout：

```text
NuPlan RL rollout: 50000 batches
batch_size: 2
目标 replay buffer: 100000 scenes
初始实测速率: 约 12.5 batch/s
```

按初始速度，rollout 约需 66 分钟；随后执行最多 500 个 update step、保存 Epoch 2 checkpoint，再自动开始 fixed200 闭环评测。实际时间仍取决于 replay buffer 增长后的 CPU 内存和 swap 压力。

当前监控记录：

```text
Epoch 1 rollout：约 6323 / 50000 batch
replay buffer：约 12648 / 100000
checkpoint：尚未保存，尚未进入 update
fixed200 评测：尚未开始
CPU available：约 3.4 GiB
swap used：约 644 KiB，尚未出现明显 swap 抖动或 OOM
```

## 8. 训练完成与自动评测进度

100k RL 训练于 2026-08-25 20:17:57 完成，随后自动启动 fixed200 评测。checkpoint：

```text
HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/rl_proxy_v3_positive_advantage_ref20_100k_seed2026_from_b10/training_log/hdp-rl-proxy_v3_positive_advantage_ref20_100k_seed2026-from-full-mini-b10/2026-08-25-18:36:38/model_epoch_2_trainloss_0.0006.pth
```

训练统计：

```text
total loss：0.0005633
reward-weighted diffusion loss：0.0001364
reward-weighted waypoint loss：0.0106917
reward mean：0.5435261
reward std p10 / p50 / p90：0.0005078 / 0.0017415 / 0.0029752
active group fraction：0.093
weighted RL loss：0.0002433
weighted expert anchor loss：0.0003138
weighted reference anchor loss：0.00000616
buffer size：100000
update steps：500
```

fixed200 当前进度：

```text
20 / 200（10%）
平均约 36.8 秒/场景
预计剩余约 1 小时 50 分钟
```

评测结束后会自动生成 B Epoch 10 与 100k RL 的 200 场景配对结果。

## 9. fixed200 最终结果

评测于 2026-08-25 23:07:44 完成：

```text
场景：200 / 200
成功：200
失败：0
场景集合一致：true
配对分析完整：true
```

| 指标 | B Epoch 10 | 100k RL | 平均变化 |
|---|---:|---:|---:|
| score | 0.90456672 | 0.90411358 | -0.00045314 |
| route progress | 0.91206773 | 0.91044633 | -0.00162141 |
| no at-fault collision | 0.95750000 | 0.95750000 | 0 |
| drivable area | 0.95000000 | 0.95000000 | 0 |
| making progress | 0.99000000 | 0.99000000 | 0 |
| TTC compliance | 0.93500000 | 0.93500000 | 0 |
| speed-limit compliance | 0.99543114 | 0.99550628 | +0.00007514 |
| comfort | 0.99500000 | 0.99500000 | 0 |

score 逐场景统计：`6 wins / 87 losses / 107 ties`；route progress 为 `0 wins / 95 losses / 105 ties`。

结论：数据从 10k 扩大到 100k 后，RL 相对 B Epoch 10 的 score 退化由 v8 的 `-0.004892` 缩小到 `-0.000453`，说明扩大数据显著减轻了策略退化，但仍未获得平均正收益。主要剩余退化来自 route progress；安全、可行驶区域、TTC、making progress 和 comfort 均与基线持平，限速指标有极小正收益。

结果文件：

```text
HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/rl_proxy_v3_positive_advantage_ref20_100k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_100k_fixed200_analysis.json
HDP-nuplan/tmp/mini_train_rl_100000_seed2026_v1/rl_proxy_v3_positive_advantage_ref20_100k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_100k_fixed200_analysis.md
```

最新监控：

```text
fixed200：86 / 200（43%）
已生成临时指标：86
当前评测进程：正常
失败/异常：未发现
预计剩余：约 1 小时 20–30 分钟，另加指标聚合时间
```

后续监控记录：

```text
Epoch 1 rollout：30460 / 50000 batch（61%）
replay buffer：60920 / 100000
当前速度：约 7 batch/s
剩余 rollout 预计：45–60 分钟
内存：available 约 2.7 GiB，swap used 约 202 MiB
checkpoint：尚未保存
```

按此速度，训练完成后还需要 update 和 fixed200 闭环评测；从该时刻到最终评测结果粗略预计约 3–4 小时。
