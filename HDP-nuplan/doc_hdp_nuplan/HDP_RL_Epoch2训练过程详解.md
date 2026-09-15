# HDP RL Epoch2 训练过程详解

本文说明当前 Test14 三模型对比中所称的 **RL Epoch2** 是如何训练得到的。这里专指 2026-08-22 训练的 `seed=2026`、`safetygate_ttc1_anchor01_10k` 模型，不是后续 v5、v6、v7、v8 等实验。

结论先说：它以 **HDP B Epoch10 的 EMA 权重**为起点，在 10,000 个平衡抽样的 mini-train NPZ 上执行了一次完整候选生成和一次奖励加权更新。配置中的 `train_epochs=2` 实际表示：

```text
Epoch 1：rollout，只生成候选、计算 reward、填充 Replay Buffer，不更新参数
Epoch 2：update，从 Replay Buffer 有放回采样，执行 500 次梯度更新
```

因此，它并不是对 10,000 个样本连续做了两遍梯度训练。

## 1. 模型身份

| 项目 | 内容 |
|---|---|
| 本文简称 | RL Epoch2 |
| 实验版本 | RL v4，`safetygate_ttc1_anchor01_10k_seed2026` |
| 模型类 | `hdp_nuplan.model.hyper_diffusion_planner.Hyper_Diffusion_Planner` |
| 网络结构 | 与 HDP B Epoch10 相同，没有增加 policy head、critic 或 value network |
| 训练入口 | `HDP-nuplan/train_predictor_rl.py` |
| rollout/update | `HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py` |
| 奖励函数 | `HDP-nuplan/hdp_nuplan/rl/reward.py` |
| 奖励加权损失 | `HDP-nuplan/hdp_nuplan/rl/loss.py` |
| Replay Buffer | `HDP-nuplan/hdp_nuplan/rl/replay_buffer.py` |
| 监督起点 | HDP B Epoch10 checkpoint 的 `ema_state_dict` |
| RL 数据量 | 10,000 个 NPZ |
| 随机种子 | 2026 |
| 总参数量 | 5,092,996 |
| 实际可训练参数 | 3,293,956，全部属于 Decoder |
| 冻结参数 | 1,799,040，全部属于 Encoder |
| 最终 epoch | 2 |
| 最终记录 loss | `0.000309404665179045` |

监督起点：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/
experiment_b_constant5e5_from_epoch4/
model_epoch_10_trainloss_0.0091.pth
```

起点 SHA256：

```text
22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce
```

最终 checkpoint：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/
hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/
2026-08-22-12:22:45/model_epoch_2_trainloss_0.0003.pth
```

最终 checkpoint SHA256：

```text
8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c
```

## 2. 严格来说，这是什么训练方法

项目中沿用 `RL` 和 `RL Epoch2` 的名称，但它的准确技术形式是：

```text
缓存场景上的分组候选采样
        +
张量化代理 reward
        +
reward-weighted diffusion self-distillation
        +
真实专家轨迹监督 anchor
```

它不是典型的在线强化学习，原因是：

- 训练时没有让自车在 NuPlan simulator 中执行动作并获得下一状态；
- 邻车未来轨迹来自 NPZ 中保存的真实未来，不会响应自车候选；
- 没有 critic、value function、TD target、PPO ratio 或策略梯度；
- closed-loop NuPlan 仿真只用于训练完成后的评测，不参与梯度更新。

所以更严谨的名称是“奖励加权的扩散模型后训练”。本文仍使用 `RL Epoch2`，只是为了与已有 checkpoint、脚本和评测表保持一致。

## 3. 权重继承关系

```text
原始 Diffusion Planner Encoder
        │
        ▼
完整 mini-train 监督训练
        │
        ▼
HDP B Epoch10 EMA
        │  加载全部 Encoder + Decoder 权重
        ▼
RL Epoch1 rollout
        │  模型参数不变
        ▼
RL Epoch2 update
        │  Encoder 冻结，只更新 Decoder
        ▼
RL Epoch2 model + EMA checkpoint
```

### 3.1 为什么加载 B Epoch10 的 EMA

`_load_pretrained()` 的优先级是：

```text
ema_state_dict > model > checkpoint 本身
```

B Epoch10 checkpoint 中存在 `ema_state_dict`，所以 RL 初始化使用的是监督模型的 EMA 权重，而不是瞬时 `model` 权重。

### 3.2 没有恢复 B Epoch10 的优化器

RL 入口只加载模型权重，然后重新创建 AdamW：

```text
AdamW(decoder_parameters, lr=4e-7, weight_decay=0.01)
```

因此：

- 继承了 B Epoch10 的网络能力；
- 没有继承 B Epoch10 的 AdamW 动量；
- 没有继承 B Epoch10 的学习率调度状态；
- RL 优化器和 scheduler 都从新状态开始。

### 3.3 Encoder 是否发生了训练

`rl_freeze_encoder=true`，Encoder 参数的 `requires_grad` 被设为 `False`，并且不进入优化器。checkpoint 对比确认：

| 权重 | Encoder 变化 | Decoder 变化 |
|---|---:|---:|
| 最终 `model` 相对 B10 EMA | 0 个参数变化 | 3,293,956 个参数发生更新 |
| 最终 `ema_state_dict` | 只有最大约 `3.58e-7` 的 EMA 浮点舍入差 | 随 Decoder 更新而变化 |

也就是说，真正学习行为变化的是 Decoder；EMA Encoder 的极小数值差不是梯度训练结果。

## 4. RL 训练数据

数据目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache
```

训练 manifest：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
diffusion_planner_training.json
```

数据审计结果：

| 项目 | 数值 |
|---|---:|
| 原始可选 mini-train 场景 | 306,801 |
| 最终 NPZ | 10,000 |
| manifest 条目 | 10,000 |
| 唯一条目 | 10,000 |
| 覆盖训练日志 | 44 |
| 每个日志抽样数 | 227～229 |
| 抽样策略 | `balanced_logs` |
| 数据抽样种子 | 3407 |
| 缓存大小 | 1,574,851,408 bytes，约 1.47 GiB |
| manifest SHA256 | `1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514` |

需要区分两个随机种子：

- `3407`：构建 10,000 NPZ 子集时的抽样种子；
- `2026`：本次 RL 候选噪声、Replay Buffer 抽样和更新加噪的训练种子。历史 `DistributedSampler` 没传 seed，实际使用默认 0，不能将它的排序归于 2026。

每个 NPZ 主要向本次训练提供：

- 当前自车状态；
- 自车未来 80 个点；
- 最多 32 辆邻车的历史状态；
- 最多 10 辆邻车的未来轨迹及有效掩码；
- lane、lane 边界、route lane 和限速信息；
- 最多 5 个静态物体；
- 场景名称，用于从 Replay Buffer 重新加载 NPZ。

候选轨迹的物理表示为：

```text
[x, y, cos(yaw), sin(yaw)]
```

形状为 `[B, G, 80, 4]`。80 个未来点、0.1 秒一个点，对应未来 8 秒。

## 5. 实际启动方式

外层多种子总控命令为：

```bash
RL_MULTI_LOG_TAG='rl_v4_ttc1_anchor01_10k_multiseed' \
RL_MULTI_VARIANT_PREFIX='safetygate_ttc1_anchor01_10k' \
RL_MULTI_EXPERT_ANCHOR_WEIGHT=0.1 \
bash HDP-nuplan/scripts/run_rl_v3_10k_multiseed.sh
```

总控先运行 `seed=42`，完成后再运行 `seed=2026`。本文的 RL Epoch2 是第二次运行得到的 seed2026 checkpoint。

内层实际训练命令等价于：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 1 \
  --standalone \
  HDP-nuplan/train_predictor_rl.py \
  --name hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10 \
  --save_dir HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10 \
  --train_set HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache \
  --train_set_list HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth \
  --normalization_file_path HDP-nuplan/normalization.json \
  --train_epochs 2 \
  --batch_size 2 \
  --learning_rate 4e-7 \
  --warm_up_epoch 1 \
  --save_utd 1 \
  --num_workers 2 \
  --planning_hybrid_loss 0.01 \
  --rl_group_size 32 \
  --rl_rollout_steps 6 \
  --rl_sampling_noise_scale 0.1 \
  --rl_trajectory_augmentation_std 0 \
  --rl_trajectory_augmentation_epochs 0 \
  --rl_buffer_update_epoch 2 \
  --rl_buffer_size 1024 \
  --rl_ema_update_rate 0.05 \
  --rl_reward_temperature 1.0 \
  --rl_advantage_clip 5.0 \
  --rl_min_reward_std 1e-6 \
  --rl_normalize_weights true \
  --rl_center_reward_weights true \
  --rl_rollout_loss_weight 1.0 \
  --rl_expert_anchor_weight 0.1 \
  --rl_max_update_steps_per_epoch 500 \
  --rl_detach_window_size 0 \
  --rl_grad_clip 5.0 \
  --rl_freeze_encoder true \
  --rl_deterministic_update true \
  --reward_progress_guard_weight 5.0 \
  --reward_progress_guard_stop_tolerance 0.2 \
  --reward_safety_gate_threshold 0.3 \
  --reward_safety_gate_margin 1.0 \
  --reward_safety_gate_min_ttc_seconds 1.0 \
  --seed 2026 \
  --use_wandb false
```

`torch.distributed.run` 虽然启用了 DDP 初始化，但 `--nproc-per-node 1` 表示本次只使用一张 GPU、一个训练进程。

运行前脚本检查了：

1. B Epoch10 checkpoint 的 `epoch` 是否等于 10；
2. checkpoint 是否同时包含 `model` 和 `ema_state_dict`；
3. NPZ 数和 manifest 数是否都等于 10,000；
4. 固定 20 场景和 39 场景的 B10 闭环基线是否已经存在。

## 6. 关键超参数

### 6.1 模型与扩散参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `future_len` | 80 | 输出未来 80 个轨迹点 |
| `hidden_dim` | 192 | 隐藏维度 |
| `encoder_depth` | 3 | Encoder 深度 |
| `decoder_depth` | 3 | Decoder 深度 |
| `num_heads` | 6 | 注意力头数 |
| `diffusion_model_type` | `x_start` | Decoder 输出解释为 clean action |
| `diffusion_supervision_type` | `x_start` | 在 clean action 空间监督 |
| `planning_hybrid_loss` | 0.01 | waypoint 积分重建损失权重 |
| `rl_detach_window_size` | 0 | 积分过程不截断梯度，即无 detach |

### 6.2 rollout 参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `rl_group_size` | 32 | 每个场景生成 32 条候选 |
| `rl_rollout_steps` | 6 | 每条候选执行 6 步扩散采样 |
| `rl_sampling_noise_scale` | 0.1 | 候选初始噪声尺度 |
| `rl_trajectory_augmentation_std` | 0 | 不做额外轨迹平移增强 |
| `rl_buffer_size` | 1024 | Replay Buffer 最多保留 1,024 个场景组 |
| `rl_buffer_update_epoch` | 2 | 每 2 个 epoch 重新 rollout 一次 |

### 6.3 update 参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `batch_size` | 2 | 每个 update step 从 Buffer 抽 2 个场景组 |
| `learning_rate` | `4e-7` | Decoder 的 AdamW 学习率 |
| `rl_max_update_steps_per_epoch` | 500 | Epoch2 最多更新 500 步 |
| `rl_reward_temperature` | 1.0 | 指数优势权重中的乘数 |
| `rl_advantage_clip` | 5.0 | 优势限制到 `[-5,5]` |
| `rl_min_reward_std` | `1e-6` | reward 方差过小的组跳过 rollout loss |
| `rl_normalize_weights` | `true` | 每个有效组的指数权重均值归一为 1 |
| `rl_center_reward_weights` | `true` | 实际回归权重使用 `w-1` |
| `rl_rollout_loss_weight` | 1.0 | rollout 自蒸馏项权重 |
| `rl_expert_anchor_weight` | 0.1 | 专家监督 anchor 权重 |
| `rl_grad_clip` | 5.0 | 全局梯度范数上限 |
| `rl_deterministic_update` | `true` | update 时使用 `model.eval()`，但仍保留梯度 |
| `rl_ema_update_rate` | 0.05 | EMA 每步吸收 5% 当前模型，等价 decay=0.95 |

## 7. 两个 epoch 的真实执行流程

```text
读取 B Epoch10 EMA
        │
        ├─ 冻结 Encoder
        ├─ 新建 Decoder AdamW
        ├─ 新建 EMA（初始等于 B10 EMA）
        └─ 新建 maxlen=1024 的 Replay Buffer
                    │
                    ▼
Epoch 1：rollout
遍历 10,000 场景，每场景生成 32 条候选并计算 reward
                    │
                    ▼
Buffer 只留下遍历顺序末尾的 1,024 个场景组
                    │
                    ▼
Epoch 2：update
从 1,024 个场景组中有放回抽样，执行 500 次更新
                    │
                    ▼
每次更新：reward-weighted rollout loss + 0.1 × expert anchor
                    │
                    ▼
更新 Decoder 和 EMA，保存 Epoch2 checkpoint
```

## 8. Epoch 1：rollout 阶段

### 8.1 遍历规模

DataLoader 长度为：

```text
10,000 scenes / batch_size 2 = 5,000 batches
```

每个 batch 的候选轨迹形状为：

```text
[2, 32, 80, 4]
```

整轮累计生成：

```text
10,000 × 32 = 320,000 条候选轨迹
```

这些轨迹全部用于统计 rollout reward，但不会全部留在 Replay Buffer。

### 8.2 候选由谁生成

rollout 使用 `ema.ema`，不是正在优化的普通 `model`。Epoch1 开始时 EMA 刚从 B Epoch10 EMA 初始化，而且 rollout 全程没有 optimizer step，因此 320,000 条候选都来自一份固定的 B Epoch10 策略。

每个场景复制 32 份条件，给不同随机扩散噪声，经过 6 步采样得到 32 条未来 8 秒轨迹。

### 8.3 训练时没有闭环交互

reward 使用的邻车未来来自 NPZ：

```text
候选自车轨迹 + 数据集中记录的邻车未来 + 地图/路线/静态物体
```

邻车不会因为候选自车轨迹改变行为。因此此处是缓存场景上的开环候选评分，不是 reactive-agent 闭环 rollout。

## 9. Reward 是怎样计算的

历史 RL Epoch2 使用的基础 reward 为：

```text
R_base
= 1.0 × R_risk
+ 3.0 × R_follow
+ 2.5 × R_lane
+ 5.0 × R_progress_guard
```

各项含义：

- `risk_reward`：综合候选与动态邻车、静态物体之间的 OBB 几何关系，以及 TTC、THW、occupancy 和碰撞风险；
- `follow_reward`：根据前车关系、时距、最小间距、相对速度和舒适加减速评估跟车质量；
- `lane_reward`：根据轨迹与附近 lane 中心线及边界的关系评估车道保持；
- `progress_guard_reward`：将候选路线进度与专家未来进度比较，防止模型通过停车获得表面安全收益。

`reward_progress_weight=1`、`reward_collision_weight=10`、`reward_route_weight=1`、`reward_comfort_weight=0.01` 等参数虽然保存在 `args.json` 中，但在这个历史版本里对应的 `progress`、`collision_cost`、`route_cost`、`comfort_cost` 主要用于日志诊断，**没有直接加入上面的最终 `R_base`**。

### 9.1 TTC safety gate

候选必须同时满足：

```text
risk_reward >= 0.3
min_ttc_seconds >= 1.0
```

其中 `min_ttc_seconds` 是候选轨迹在整个未来时域内，相对有效动态目标估算出的最小 TTC：

- 已经发生 OBB 重叠时为 0；
- 没有 closing speed 时为 `inf`；
- 没有有效动态目标时为 `inf`。

门控逻辑分两种情况：

1. 如果同一场景的 32 条候选中至少有一条合格，所有不合格候选的 reward 会被压到最低合格 reward 以下；
2. 如果 32 条候选全部不合格，则退化为优先按 `risk_reward` 排序，原 reward 只作为很小的 tie-breaker。

### 9.2 这里的“硬门”并没有删除危险候选

这是理解该 checkpoint 的关键。历史版本只把门控后的标量 reward 写入 Replay Buffer：

```text
scene_name + 32 条 trajectories + 32 个 gated rewards
```

它没有把 `safety_gate_eligible` mask 写入 update loss。因此 TTC 不达标候选仍会参与 Epoch2 的反向传播，只是 reward 排名更低。

后来新增的 `rl_filter_safety_eligible_candidates`、reference policy、drivable-area gate、`reward_objective_mode` 等参数，在本次历史 `args.json` 中都不存在，不能用它们解释 RL Epoch2。

### 9.3 Epoch1 实际 reward 统计

| 指标 | 数值 |
|---|---:|
| 门控后 mean reward | 5.613862 |
| mean base reward | 9.681606 |
| mean risk reward | 0.543807 |
| mean follow reward | 0.761471 |
| mean lane reward | 0.832060 |
| mean progress guard reward | 0.954647 |
| 合格候选比例 | 55.4209% |
| TTC 合格候选比例 | 56.1159% |
| 至少有一个合格候选的场景组比例 | 55.6500% |
| no-collision 候选比例 | 97.7250% |
| 最终 Buffer 场景组 | 1,024 |

这意味着约 44.35% 的场景组没有任何同时通过 `risk>=0.3` 和 `TTC>=1s` 的候选。

## 10. Replay Buffer 为什么只有 1,024 个场景

Replay Buffer 底层是：

```python
deque(maxlen=1024)
```

每处理一个场景，就追加一个条目。超过 1,024 后，最旧条目会自动被删除。因此 Epoch1 虽然遍历并评分了全部 10,000 个场景，最后真正可供更新的只有打乱顺序末尾的 1,024 个场景：

```text
1,024 scene groups × 32 candidates = 32,768 条候选
```

2026-09-07 复核更正：历史 `DistributedSampler` 没有显式传入 seed，默认值为 0。在相同 manifest、单进程和 `set_epoch(0)` 下，seed42 与 seed2026 留下的末尾 1,024 个场景相同。训练 seed 仍影响候选噪声、Replay 抽样和更新加噪，但不能把差异归因于 sampler 保留了不同场景。

## 11. Epoch 2：update 阶段

### 11.1 DataLoader 在这一轮的作用

Epoch2 仍遍历 DataLoader，但当前 `loader_batch` 的场景内容并不直接用于训练。它只负责：

- 提供本次 update 的 batch size；
- 提供最多 5,000 次循环的外壳。

随后代码会忽略该 batch 的场景身份，转而从 Replay Buffer 随机抽取真正的训练场景。

由于 `rl_max_update_steps_per_epoch=500`，循环在 500 步后停止。

### 11.2 有放回采样

每一步调用：

```python
random.choices(buffer_items, k=2)
```

这是有放回采样，所以：

- 同一个场景可以被多次抽到；
- 同一步的两个条目甚至可以是同一个场景；
- 500 步一共抽取 1,000 个场景条目，但不代表覆盖 1,000 个不同场景；
- 按均匀有放回抽样的期望，1,024 个 Buffer 场景中约覆盖 638 个不同场景；实际唯一数当时没有写入日志。

每一步从 2 个场景得到：

```text
trajectories: [2, 32, 80, 4]
rewards:      [2, 32]
```

将场景维和候选维展平后，扩散训练张量为：

```text
[2 × 32, 80, 4] = [64, 80, 4]
```

500 步累计处理 32,000 个“候选轨迹训练实例”，但其中存在 Replay 场景和候选组的重复使用。

### 11.3 为什么 update 使用 `model.eval()` 仍然能训练

`rl_deterministic_update=true` 会执行：

```python
model.eval()
```

它关闭 Dropout/DropPath 等训练随机性，避免 reward 自蒸馏目标之外的随机漂移，但不会关闭 autograd。只有 `torch.no_grad()` 才会关闭梯度，因此 Epoch2 仍正常执行：

```text
loss.backward()
optimizer.step()
```

## 12. Reward 如何变成训练权重

对于一个场景中的 32 条候选，先计算组内均值和总体标准差：

```text
μ_b = mean_g(r_b,g)
σ_b = std_g(r_b,g)
```

再计算标准化优势：

```text
A_b,g = (r_b,g - μ_b) / (σ_b + 1e-6)
```

随后：

```text
A_b,g = clip(A_b,g, -5, 5)
w_b,g = exp(A_b,g)
```

`rl_normalize_weights=true` 会把有效场景组内的 `w` 重新归一化到均值 1。若组内 reward 标准差小于 `1e-6`，该组 rollout 权重全部置 0。

### 12.1 本次真正使用的是 `w - 1`

因为：

```text
rl_center_reward_weights=true
```

实际回归权重不是 `w`，而是：

```text
w_reg = w - 1
```

因此：

- 高于组内平均 reward 的候选通常 `w_reg > 0`，最小化时会让模型更拟合它；
- 低于组内平均 reward 的候选通常 `w_reg < 0`，最小化时会主动增大它的拟合误差；
- 组内回归权重均值接近 0，只保留 reward 与回归误差之间的相关信号。

当时这样设计是为了消除所有候选共同携带的无权自蒸馏梯度。但后来分析确认：负回归权重会使目标不再是普通的有下界 MSE，可能把低奖励轨迹推向不可预测方向。这是历史 RL Epoch2 的已知限制，不应描述成“仅降低危险轨迹的学习强度”。

## 13. Reward-weighted diffusion loss

### 13.1 轨迹先转换成模型 action

Replay Buffer 保存的是具体 waypoint：

```text
[x_t, y_t, cos(yaw_t), sin(yaw_t)]
```

update 时转换成：

```text
[Δx_t, Δy_t, cos(yaw_t), sin(yaw_t)]
```

再使用 `state_normalizer` 归一化，作为 clean diffusion data `x_start`。

### 13.2 扩散监督

每条候选独立采样扩散时刻 `t` 和高斯噪声，构造带噪动作 `x_t`。场景条件重复 32 份，与 32 条候选一一对应。

本次：

```text
model_type       = x_start
supervision_type = x_start
```

所以 Decoder 的统一字段 `decoder_output["score"]` 在这里实际解释为预测 clean action，并计算：

```text
L_diff(b,g) = mean_t ||x_start_pred - x_start_target||²
```

奖励加权扩散项为：

```text
L_diff_weighted = mean_b,g[(w_b,g - 1) × L_diff(b,g)]
```

### 13.3 Hybrid waypoint loss

模型预测的 `[Δx, Δy]` 被积分还原为 `[x, y]` waypoint，再与 rollout 候选 waypoint 比较：

```text
L_wp(b,g) = mean_t ||waypoint_pred - waypoint_rollout||²
```

`rl_detach_window_size=0` 表示使用完整 `cumsum`，waypoint loss 的梯度可以通过积分传播到此前全部 displacement，没有 stop-gradient 窗口。

奖励加权 rollout 总损失为：

```text
L_rollout
= mean[(w-1) × L_diff]
+ 0.01 × mean[(w-1) × L_wp]
```

## 14. Expert anchor

只用 rollout 自蒸馏可能让模型偏离真实驾驶数据，所以同一个 Replay 场景还会读取 NPZ 中的真实自车未来轨迹，计算与监督训练相同的 HDP loss：

```text
L_expert
= L_ego_diffusion
+ 0.01 × L_ego_waypoint
```

最终优化目标为：

```text
L_total = 1.0 × L_rollout + 0.1 × L_expert
```

注意：`reward_imitation_weight=0` 只表示 reward 里不加入 imitation 项，不代表没有专家监督。`rl_expert_anchor_weight=0.1` 明确加入了真实轨迹监督。

本次也没有冻结一份 B Epoch10 reference policy 计算 reference loss；该功能是后续才加入的。

## 15. 每个 update step 发生什么

```text
1. 从 Replay Buffer 有放回抽 2 个场景组
2. 按 scene_name 重新加载对应 NPZ
3. 得到 [2,32,80,4] 候选和 [2,32] reward
4. 展平成 64 条候选并随机加扩散噪声
5. 计算每条候选的 diffusion loss 和 waypoint loss
6. 将 reward 转成组内 advantage、exp weight 和 centered weight
7. 计算 L_rollout
8. 用 2 条真实 ego future 计算 L_expert
9. 得到 L_total = L_rollout + 0.1 × L_expert
10. 清空旧梯度并反向传播
11. 将可训练参数的全局梯度范数裁剪到不超过 5
12. AdamW 只更新 Decoder
13. EMA 执行 ema = 0.95 × ema + 0.05 × model
```

以上过程重复 500 次。

## 16. Epoch2 实际训练日志

500 个 update step 的平均指标为：

| 指标 | 数值 | 解释 |
|---|---:|---|
| `reward_mean` | 5.604042 | Replay batch 中候选平均 reward |
| `reward_max` | 5.613615 | 每组最好候选 reward 的平均值 |
| `reward_min` | 5.591444 | 每组最差候选 reward 的平均值 |
| `reward_std_mean` | 0.005679 | 组内 reward 区分度很小 |
| `active_group_fraction` | 0.972 | 97.2% 的组通过 `std>=1e-6` |
| `weight_mean` | 0.972 | 无效组为 0，有效组均值约 1 |
| `regression_weight_mean` | 约 `-3.02e-10` | centered 后均值接近 0 |
| `reward_weighted_diffusion_loss` | `-6.8834e-6` | centered diffusion 项 |
| `reward_weighted_waypoint_loss` | `-0.001194009` | 尚未乘 0.01 的 centered waypoint 项 |
| `rl_loss` | `-1.882349e-5` | 上述两项按 0.01 合并 |
| `expert_anchor_loss` | `0.003282282` | 未乘 0.1 的真实轨迹监督项 |
| `weighted_expert_anchor_loss` | `0.000328228` | expert anchor 的实际贡献 |
| `loss` | `0.000309405` | 最终反向传播目标 |
| `buffer_size` | 1,024 | update 使用的场景组容量 |
| `update_steps` | 500 | 实际参数更新次数 |

数值关系可以直接核对：

```text
L_rollout
= -6.8834e-6 + 0.01 × (-0.001194009)
≈ -1.88235e-5

L_total
= -1.88235e-5 + 0.1 × 0.003282282
≈ 0.000309405
```

`rl_loss` 为负数不是日志错误，而是 `w-1` 产生负回归权重的直接结果。

## 17. 学习率和 EMA

本次 AdamW 的学习率始终为：

```text
4e-7
```

虽然配置了 `warm_up_epoch=1` 和两轮 scheduler，但只有 Epoch2 是 update epoch，而且 scheduler 只在 update 后推进一次。因此本次没有形成有实质区分的多阶段学习率曲线。

普通 `model` 在 500 步中直接更新；EMA 每步以 5% 的更新比例跟随。后续闭环 planner 加载 checkpoint 时优先使用 `ema_state_dict`，所以评测中的 RL Epoch2 指的是最终 EMA 策略。

## 18. checkpoint 保存内容

只有 update epoch 才保存 checkpoint。Epoch1 没有更新参数，因此不保存 `model_epoch_1`；Epoch2 保存：

```text
epoch
model
ema_state_dict
optimizer
schedule
loss
wandb_id
```

文件审计：

| 项目 | 数值 |
|---|---:|
| 保存时间 | 2026-08-22 12:31:36 CST |
| 文件大小 | 67,417,123 bytes |
| `model` 张量数 | 260 |
| `ema_state_dict` 张量数 | 260 |
| 参数量 | 5,092,996 |
| checkpoint 内 `epoch` | 2 |
| checkpoint 内 `loss` | `0.000309404665179045` |

训练从 12:22:42 左右开始，到 12:31:39 记录训练完成，约耗时 8 分 57 秒。后续 gate20、gate39 和 fixed200 是闭环评测时间，不属于这 9 分钟训练本身。

## 19. 这个模型实际学到了什么

可以确定的是：

- Encoder 未训练，场景表征保持 B Epoch10；
- Decoder 参数发生变化，候选轨迹分布和最终闭环行为随之变化；
- 变化方向由 reward 排名、centered 自蒸馏和 expert anchor 共同决定；
- 它不是在推理时从 32 条候选里选择最高 reward 的 Best-of-N；训练完成后仍由单个扩散 planner 正常生成规划轨迹；
- reward 只用于后训练，不会作为推理输入。

fixed200 上的 seed2026 结果曾显示：

| 模型 | score | route progress |
|---|---:|---:|
| B Epoch10 | 0.904567 | 0.912068 |
| RL Epoch2 seed2026 | 0.906065 | 0.917264 |
| 差值 | **+0.001498** | **+0.005196** |

这说明该 checkpoint 确实改变了策略行为，并在这 200 个固定场景上主要通过路线进度获得小幅收益。

但扩大到当时实际构建的 186 个 `test14-random` 场景后：

| 模型 | score | route progress | collision 指标 | TTC 指标 |
|---|---:|---:|---:|---:|
| B Epoch10 | 0.877131 | 0.941122 | 0.946237 | 0.919355 |
| RL Epoch2 seed2026 | 0.870830 | 0.946382 | 0.930108 | 0.892473 |
| 差值 | **-0.006301** | +0.005259 | -0.016129 | -0.026882 |

上述 186 场景结果是早期记录，不代表最终完整 Test14。它与后来的 261 场景集合不同，早期并行配置还需考虑后来发现的指标共享状态风险。

2026-09-02 完成的正式单 worker 结果为：Test14-hard 272 场景，B/RL score 为 0.705304/0.702422（−0.002882）；Test14-random 261 场景为 0.824270/0.831146（+0.006876）。533 场景按数量加权约提高 0.001896，仍未证明跨 benchmark 稳定正收益。详见 [完整 Test14 操作日志](Test14完整评测数据准备与三模型对比操作日志.md) 第 19 节。

## 20. 后来确认的主要限制

这些限制属于对历史训练的复盘，不是用后续代码替换历史事实：

1. **Buffer 覆盖不足**：rollout 10,000 个场景，但更新只保留最后 1,024 个；
2. **只有一次更新轮**：2 epoch 实际只有一个 rollout epoch 和一个 update epoch；
3. **候选区分度低**：`reward_std_mean` 只有约 0.00568；
4. **约 44.35% 的场景组没有安全合格候选**：这些组只能在不安全候选中相对排序；
5. **safety gate 没有过滤反向传播候选**：它只重排 reward；
6. **centered 权重存在负回归权重**：目标可能无下界；
7. **没有 B10 reference 约束**：expert anchor 约束真实轨迹，但不直接限制策略相对 B10 的输出漂移；
8. **reward 与 NuPlan 最终 score 未完全对齐**：旧的 collision、route、comfort 等量主要是诊断项。

这些是需要控制实验检验的不稳定因素，不能仅由历史均值差异确定每一项的因果贡献。最终完整 Test14 上 random 为正、hard 为负，说明当前证据仍不支持跨集合稳定提升。

## 21. 历史实现与当前代码的关系

本次 checkpoint 训练于 2026-08-22。最接近训练时工作区、并在次日提交的历史代码快照为：

```text
git commit b13841b
Consolidate HDP NuPlan RL and evaluation tooling
```

当前分支后来加入了：

- safety candidate 真正过滤；
- progress candidate 过滤；
- 正权重/positive-advantage 更新；
- reference-policy anchor；
- reference-relative reward；
- drivable-area gate；
- 更接近 NuPlan score 的 reward objective。

这些功能不属于本文 RL Epoch2 的历史训练。若要严格复盘，应同时查看该 checkpoint 的 `args.json`、原始日志和 `b13841b` 中的源码，而不能只按当前文件里的新增分支推断。

## 22. 复盘证据位置

### 22.1 完整参数

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/
hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/
2026-08-22-12:22:45/args.json
```

### 22.2 原始训练与 gate 日志

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/
rl_v2_train_and_gate59.log
```

### 22.3 外层多 seed 日志

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_v4_ttc1_anchor01_10k_multiseed.log
```

### 22.4 数据审计

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/sampling_report.json
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache_validation_report.json
```

### 22.5 fixed200 结果

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_v4_fixed200_eval_retry/b10_vs_rl_fixed200.json
```

### 22.6 test14-random 结果

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
rl_v4_random_val14_eval/b10_vs_rl_v4_test14_random.json
```

### 22.7 总操作日志对应章节

```text
HDP-nuplan/doc_hdp_nuplan/正式监督训练NPZ合并操作日志.md
```

重点查看 `16.42`、`16.43`、`16.58` 和后续 `test14-random 最终结果`。

## 23. 一句话总结

RL Epoch2 是从 B Epoch10 EMA 出发、冻结 Encoder，在 10,000 个缓存场景上先生成 320,000 条候选，再只保留最后 1,024 个场景组，用 `risk + follow + lane + progress guard` 奖励和 TTC 排序构造 centered reward-weighted diffusion loss，并结合 0.1 权重的专家监督，对 Decoder 执行 500 次小学习率更新后得到的 EMA checkpoint。
