# HDP-RL 论文方案迁移操作日志

## 1. 目标与边界

- 目标：参考 Zheng 等人的《Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving》，将 HDP-nuPlan 的 RL 训练逐步改造成论文的 RL-Hybrid 方案，并通过分阶段门禁判断是否获得正收益。
- 论文路径：`/home/yanjun/Zotero/storage/ICTSFYUX/Zheng 等 - 2026 - Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving.pdf`
- Python 环境：`/home/yanjun/NewDisk/conda_envs/diffusion_planner`
- 本文只记录本轮论文方案迁移。此前 mini 监督训练、reward v2、Best-of-N 等实验继续保留在原日志中。
- 正收益不能由代码修改直接保证；最终以独立验证集和 NuPlan 闭环指标相对同一监督 checkpoint 的变化为准。

## 2. 论文依据

2026-08-08 重新读取论文 Section 5、Appendix D.4、D.5、Table 6 和 Algorithm 2，得到以下明确约束：

1. RL 使用旧策略采样候选轨迹，以组内标准化奖励构造 `exp(beta * normalized_reward)` 权重。
2. IL 与 RL 使用同一套 hybrid trajectory loss，论文的 `omega=0.1`。
3. rollout 使用 6 个 diffusion sampling steps。
4. group size 为 32，`beta=1.0`。
5. 使用 EMA 更新策略；Table 6 记录 EMA 为 `0.05`，但正文没有明确它是 update rate 还是 decay。
6. multi-reward 为：

   ```text
   reward = 1.0 * risk + 3.0 * follow + 2.5 * lane
   ```

7. `risk` 对 TTC、THW、OCC 三个 `[0,1]` 分数在对象和时间上取最保守的最小值。
8. `follow` 平均 time gap、spacing、speed match、longitudinal comfort 四项。
9. `lane` 按到最近车道中心线的距离从 1 线性衰减到 0；专家离道或换道时屏蔽该项。
10. 非响应式 replay 中，后方车辆追尾只使用 `0.3` 碰撞代价，主动、迎面和侧向碰撞使用 `1.0`。
11. 论文没有报告人工轨迹平移增强，也没有在 RL 目标中加入额外 expert anchor。

论文未公开 TTC/THW/OCC shaping 阈值、ACC 目标间距和 lane-change 判定阈值。因此这些数值属于 NuPlan 工程适配，不得表述为论文原始超参数；代码中必须配置化并写入 `args.json`。

## 3. 修改前基线诊断

修改前默认设置：

```text
planning_hybrid_loss=0.01
rl_group_size=10
rl_rollout_steps=5
rl_sampling_noise_scale=0.2
rl_trajectory_augmentation_std=0.5
rl_trajectory_augmentation_epochs=5
rl_buffer_update_epoch=5
rl_expert_anchor_weight=0.1
EMA decay=0.999（代码硬编码）
```

修改前总奖励：

```text
progress
- 10 * collision_cost
- route_cost
- 0.01 * comfort_cost
- backward_cost
```

历史诊断中组内总奖励与 progress 的相关性中位数约为 `0.998`，说明候选排序几乎被进度控制；这与论文的 risk/follow/lane 多目标奖励不一致。

## 4. 分阶段执行计划

### 阶段 A：论文 multi-reward

- 实现 `[0,1]` 的 risk/follow/lane。
- 保留 progress、collision、route、comfort、backward 作为诊断指标，但不再加入训练 reward。
- 添加合成单元测试和真实 NPZ dry-run。

### 阶段 B：旧策略与 EMA policy iteration

- rollout 改为冻结的旧策略/EMA 策略。
- 一轮 rollout 后只进行短更新，再刷新 buffer。
- 将 EMA 参数配置化，并明确 `0.05` 的解释和对照实验。

### 阶段 C：论文一致的监督起点

- 用 `planning_hybrid_loss=0.1` 重新监督训练；不能直接把原 `omega=0.01` checkpoint 当成完全一致的论文起点。
- 数据从 mini 扩大到 NuPlan trainval 的平衡子集，先完成 100k 级别门禁。

### 阶段 D：RL 正收益门禁

- 先执行 reward sanity check 和自然候选 group diagnostics。
- 再执行 20-step train/val gate。
- 通过后进行完整 update epoch、固定场景闭环和扩大评测。

## 5. 阶段 A 实际操作

### 5.1 工作区保护

执行：

```bash
git status --short
```

结果：工作区已有大量已暂存、未暂存和未跟踪文件。本轮只修改以下范围，不清理或覆盖其他改动：

```text
HDP-nuplan/hdp_nuplan/rl/reward.py
HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py
HDP-nuplan/train_predictor_rl.py
HDP-nuplan/scripts/compare_checkpoint_behavior.py
HDP-nuplan/tests/test_rl_components.py
HDP-nuplan/doc_hdp_nuplan/HDP_RL_论文方案迁移操作日志.md
```

### 5.2 奖励实现

在 `reward.py` 中新增：

- `_ego_motion()`：从轨迹计算速度和纵向加速度。
- `_neighbor_geometry()`：构造 `[B,G,N,T]` 的邻车 OBB 间距、相对纵横向位置和速度。
- `_risk_reward()`：计算 TTC、THW、OCC 和碰撞安全分数，输出区间均为 `[0,1]`。
- `_following_reward()`：计算 time gap、spacing、speed match、纵向舒适度的平均分数；没有前车时返回中性值 1。
- `_lane_reward()`：使用 `lanes[...,0:2]` 中心线，以及 `lanes[...,4:8]` 左右边界偏移估计半车道宽；专家换道或离道场景对所有候选返回相同中性值。
- 新 `__call__()`：严格使用 `1*risk + 3*follow + 2.5*lane` 汇总训练奖励。

旧 reward v2 被保留为 `_legacy_reward()`，用于历史复盘；默认调用路径已经切换到论文 multi-reward。

### 5.3 默认参数对齐

`train_predictor_rl.py` 当前默认值调整为：

```text
planning_hybrid_loss=0.1
rl_group_size=32
rl_rollout_steps=6
rl_sampling_noise_scale=0.1
rl_trajectory_augmentation_std=0.0
rl_trajectory_augmentation_epochs=0
rl_buffer_update_epoch=2
rl_expert_anchor_weight=0.0
reward_risk_weight=1.0
reward_follow_weight=3.0
reward_lane_weight=2.5
```

注意：`planning_hybrid_loss=0.1` 只表示新实验默认配置已经对齐论文，不表示现有 `omega=0.01` 监督 checkpoint 自动变成论文一致 checkpoint。

### 5.4 合成单元测试

执行：

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
-m pytest -q HDP-nuplan/tests/test_rl_components.py
```

第一次结果：`1 failed, 10 passed`。

失败原因：旧测试要求“离开车道但不碰撞”的候选总奖励一定高于“在车道内但碰撞”的候选。论文 multi-reward 是加权和，不把 safety 作为硬约束，因此该断言不再成立。测试修改为只比较 `risk_reward` 的碰撞排序，并新增论文总奖励权重、奖励边界、跟车距离和车道中心线测试。

第二次结果：

```text
14 passed in 3.05s
```

### 5.5 真实 NuPlan NPZ dry-run

使用：

```text
HDP-nuplan/tmp/rl_smoke_cache/us-pa-pittsburgh-hazelwood_4dfe15e70158580b.npz
```

构造三条候选：专家轨迹、横向偏移 1 m、速度缩放为 0.7。真实输入包含 32 个邻车槽位、70 条 lane、25 条 route lane 和静态物体。

结果：所有 reward/details 均为有限数，无 NaN/Inf。三条候选总奖励为：

```text
[2.4767947, 1.9139037, 2.4767864]
```

横向偏移候选因 TTC 风险和 lane score 下降而得到更低奖励。该检查只证明张量路径和基本排序可运行，不代表 reward 阈值已完成数据集级标定。

## 6. 当前状态与下一步

阶段 A 的代码主体已完成，尚需：

1. 运行 HDP-nuPlan 全部测试，确认没有破坏其他模块。
2. 在 100/1000 个真实场景上统计 expert、扰动和自然 diffusion candidates 的 risk/follow/lane 分布，标定论文未公开阈值。
3. 完成阶段 B：EMA old-policy rollout 与短周期 buffer 刷新。
4. 在阶段 C 新监督 checkpoint 完成前，不启动正式 RL 长训练。

## 7. 阶段 B：EMA old-policy 策略迭代

### 7.1 Rollout 策略修正

修改前 `rollout_epoch()` 接收正在训练的 `model`。修改后主循环传入 `ema.ema`：

```text
EMA(pi_{k-1}) rollout
        -> replay buffer
        -> model update
        -> EMA update
        -> 下一轮重新 rollout
```

默认 `rl_buffer_update_epoch` 从 5 改为 2，因此 epoch 0/2/4 执行 rollout，epoch 1/3/5 各执行一次 update。每份 replay 不再连续复用四个 update epoch。

### 7.2 EMA 参数解释

`timm.ModelEma` 的实现是：

```text
ema = decay * ema + (1 - decay) * model
```

论文 Table 6 只写 `EMA=0.05`，没有注明语义。本实现将 `0.05` 明确解释为当前模型写入 EMA 的 update rate，因此：

```text
rl_ema_update_rate=0.05
decay=1-0.05=0.95
```

新增 `ema_decay_from_update_rate()` 和数值测试：旧 EMA 权重为 1、当前模型权重为 2，更新一次后必须得到 1.05。该测试已通过。若后续论文代码说明 0.05 实际是 decay，只需修改配置，不需要再改训练代码。

### 7.3 完整测试

执行：

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
-m pytest -q HDP-nuplan/tests
```

结果：

```text
31 passed, 1 warning in 3.84s
```

warning 来自 timm 的弃用导入提示，与本次逻辑无关。

## 8. 100 场景论文奖励门禁

### 8.1 新增可复现脚本

新增：

```text
HDP-nuplan/scripts/validate_paper_reward.py
```

候选轨迹包括 expert、stop、lateral、slow、jitter、collision；检查 risk/follow/lane 数值范围、碰撞排序、车道排序和停车 reward hacking。

第一次命令错误地把 smoke manifest 与 balanced10k cache 混用，出现 `FileNotFoundError`，没有生成报告。修正后执行：

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/validate_paper_reward.py \
  --cache-dir HDP-nuplan/tmp/mini_train_smoke_100_v2/cache \
  --manifest HDP-nuplan/tmp/mini_train_smoke_100_v2/diffusion_planner_training.json \
  --output HDP-nuplan/tmp/mini_train_smoke_100_v2/paper_reward_validation_100.json \
  --max-scenes 100 --minimum-pass-rate 0.8
```

结果：

| 检查 | 结果 |
|---|---:|
| 所有奖励有限 | 100/100 |
| 所有子奖励在 `[0,1]` | 100/100 |
| 横向偏移降低 lane reward | 63/65，96.9% |
| 人工碰撞降低 risk reward | 100/100 |
| 移动专家总奖励高于停车 | 46/66，69.7% |

因此总门禁为 `accepted=false`。前三项证明论文奖励主体工作正常；最后一项揭示停车 reward hacking 风险。当前不加入 progress 修补，因为本阶段目标是先忠实复现论文。论文依靠强 IL 先验、旧策略采样和 KL 正则避免生成任意停车 action；当前 mini10k 起点明显更弱，后续必须单独监测 progress/path length。

## 9. 自然 diffusion 候选诊断

为确认停车问题是否会出现在模型自然候选，而不仅是人工 stop 候选，使用同一个监督 epoch10 checkpoint 作为比较两侧，以相同随机种子生成 32 条自然候选：

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/compare_checkpoint_behavior.py \
  --args-file HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/args.json \
  --supervised-checkpoint HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --rl-checkpoint HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --data-dir HDP-nuplan/tmp/mini_train_smoke_100_v2/cache \
  --data-list HDP-nuplan/tmp/mini_train_smoke_100_v2/diffusion_planner_training.json \
  --batch-size 2 --num-workers 2 --repeats 1 \
  --num-samples 32 --diffusion-steps 6 \
  --sampling-noise-scale 0.1 --trajectory-augmentation-std 0 \
  --device cuda \
  --output HDP-nuplan/tmp/mini_train_smoke_100_v2/paper_reward_natural_g32_epoch10_100.json
```

关键结果：

| 指标 | 结果 |
|---|---:|
| group reward std，均值/中位数 | 0.01757 / 0.00153 |
| group reward range，中位数 | 0.00683 |
| group endpoint diversity，中位数 | 0.0820 m |
| 最优 reward 候选相对组均值 progress | +0.0030 |
| reward 与 progress 的候选级相关系数 | 0.309 |
| reward 与 follow reward 的候选级相关系数 | 0.858 |

与 reward v2 的 progress 相关性约 0.998 相比，新奖励不再被 progress 单项控制。但自然候选多样性仍然很弱，说明 mini10k 监督起点存在明显 mode collapse。

修改前 `rl_min_reward_std=0.01` 会跳过中位数仅 0.00153 的大部分自然候选组；论文只丢弃奖励完全相同的 action，因此默认值改为 `1e-6`，只吸收浮点误差并丢弃真正等价的组。

## 10. 更新后的下一步

1. 不使用现有 `omega=0.01` checkpoint 启动正式长 RL。
2. 先准备 `omega=0.1` 的论文一致监督起点；优先扩大至 trainval 平衡 100k。
3. 在等待大监督训练前，可以只做 100 场景、20 update-step 的工程 smoke，验证 EMA old-policy、replay 和新 reward 能完整跑通，但该结果不能作为正收益结论。
4. RL 验收必须同时报告 risk/follow/lane、progress、path length、stationary fraction 和 NuPlan 闭环分数。

## 11. 论文一致的 `omega=0.1` 监督起点

### 11.1 为什么重新训练

旧的 mini10k 监督 checkpoint 使用 `planning_hybrid_loss=0.01`，而论文 Table 6 给出的混合目标权重为 `omega=0.1`。如果直接从旧 checkpoint 做 RL，则监督起点、RL 正则项与论文并不一致，无法判断负收益究竟来自奖励、RL 更新还是目标权重不一致。因此先在相同 10,000 场景上只改变该权重，得到可对照的监督起点。

### 11.2 实际执行命令

工作目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
```

执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-paper-supervised-balanced10k-omega01 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 --num_workers 4 \
  --train_epochs 20 --warm_up_epoch 2 --save_utd 1 \
  --learning_rate 5e-4 --planning_hybrid_loss 0.1 \
  --encoder_pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 3 --use_ema true
```

完整参数快照由训练入口自动保存到：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json
```

### 11.3 训练结果

进程正常退出，exit code 为 0；20 个 epoch 均保存了 checkpoint。每轮包含 1,250 个 batch，即 10,000 场景除以 batch size 8。

| epoch | train loss | epoch | train loss |
|---:|---:|---:|---:|
| 1 | 3.7471 | 11 | 1.0773 |
| 2 | 2.3905 | 12 | 1.0371 |
| 3 | 1.3460 | 13 | 1.0009 |
| 4 | 1.9000 | 14 | 0.9782 |
| 5 | 1.5426 | 15 | 0.9792 |
| 6 | 1.3970 | 16 | 0.8950 |
| 7 | 1.3210 | 17 | 0.9259 |
| 8 | 1.2464 | 18 | 0.8773 |
| 9 | 1.2392 | 19 | 0.8756 |
| 10 | 1.1294 | 20 | 0.8478 |

epoch 4 是编码器冻结 3 轮后首次解冻，loss 从 1.3460 暂时升至 1.9000，随后恢复下降；最终 loss 为 0.8478。该结果只证明 `omega=0.1` 可以稳定训练，不能单凭训练 loss 认定 epoch 20 最好，也不能认定 RL 会产生正收益。

### 11.4 下一道门禁

使用独立的 1,000 场景 validation cache 对全部 20 个 EMA checkpoint 做一次同种子验证 loss 排名；再对候选 checkpoint 做重复验证和行为指标比较。监督起点通过后，才进行论文参数的 RL 更新。

## 12. 监督 checkpoint 选择

### 12.1 全部 checkpoint 的验证

先对 20 个 EMA checkpoint 在独立 validation-1000 上做一次粗筛，随后全部重复 3 次。执行入口为：

```text
evaluate_checkpoints.py
```

三次重复报告：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
paper_supervised_omega01_checkpoint_ranking_repeat3.json
```

前三名为：

| 排名 | epoch | train loss | 三次平均 val loss |
|---:|---:|---:|---:|
| 1 | 5 | 1.5426 | 0.782481 |
| 2 | 16 | 0.8950 | 0.790550 |
| 3 | 14 | 0.9782 | 0.795127 |

epoch 20 的 train loss 最低，但 val loss 为 0.854839，因此不能按最后一轮或最低训练 loss 选模型。

### 12.2 为什么最终用 epoch 14，而不是验证 loss 第一名 epoch 5

又把新 checkpoint 与旧 `omega=0.01` epoch 10 在相同 validation-1000、相同扩散噪声下做行为对照：

| 新 checkpoint | 论文 reward | progress | collision cost | comfort cost | ADE |
|---|---:|---:|---:|---:|---:|
| epoch 5 | -0.05033 | +0.06913 | +0.02993 | +1.78596 | -0.12424 m |
| epoch 14 | -0.01033 | +0.01084 | +0.01248 | +0.14630 | -0.10413 m |
| epoch 16 | -0.03746 | -0.07169 | +0.01582 | +0.02783 | +0.09064 m |

epoch 5 虽然 validation diffusion loss 最低，但安全和舒适代理明显更差；epoch 14 是三者中最接近旧模型、且 ADE 改善的折中点，因此后续论文参数实验固定从 epoch 14 开始。这里没有宣称新监督模型优于旧模型。

监督起点：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/
model_epoch_14_trainloss_0.9782.pth
```

## 13. 论文原式 RL 的失败结果

论文 Eq. (9) 经重新核对后确认：RL-Hybrid 不是额外增加专家 IL loss，而是用 `exp(beta*r)` 对同一个 hybrid diffusion regression loss 加权；因此 `rl_expert_anchor_weight=0` 才是论文原式。

从 epoch 14 做 group 32、6 sampling steps、beta 1、EMA update rate 0.05、20 update-step 的 100 场景 smoke。普通模型和 EMA 都能正常训练，但 validation-100 结果为：

| 权重来源 | reward | progress | collision cost | ADE |
|---|---:|---:|---:|---:|
| model | -0.02224 | -0.17996 | +0.02007 | +0.50027 m |
| EMA | -0.00811 | -0.07104 | +0.00349 | +0.18445 m |

因此工程链路已跑通，但论文原式在当前 mini10k 弱监督起点上仍是负收益。

## 14. 负收益根因定位

### 14.1 自然候选偏好

对 epoch 14 每场景生成 32 条自然候选，报告为：

```text
tmp/mini_val_pilot_100_seed3407_v1/
paper_reward_natural_g32_omega01_epoch14_val100.json
```

关键结果：

| 指标 | 数值 |
|---|---:|
| group reward std | 0.008325 |
| endpoint diversity | 0.2513 m |
| 最高奖励候选 progress 相对组均值 | -0.00665 |
| 最高奖励候选 path length 相对组均值 | -0.06725 m |
| 最高奖励候选 collision cost 相对组均值 | -0.00926 |

论文奖励能选出更安全候选，但平均略微偏向减速。论文依靠约 70M 帧 IL 起点维持驾驶先验；当前 10k 场景起点不足以提供同等约束。

### 14.2 有界 progress guard

新增两个参数：

```text
reward_progress_guard_weight
reward_progress_guard_stop_tolerance
```

默认 `weight=0`，保持论文 Eq. (25) 原样；只有 NuPlan 小数据适配实验显式开启。

设计：

- 移动场景：候选达到专家记录 progress 后得 1，超过专家不继续增益，避免鼓励超速；
- 停车场景：候选越接近专家的低 progress 越好；
- 输出始终在 `[0,1]`。

权重 5.0 的候选诊断：

| 指标 | 无 guard | guard 5.0 |
|---|---:|---:|
| 组内 reward-progress 相关性 | 0.010 | 0.210 |
| 全候选 reward-progress 相关性 | -0.076 | +0.218 |
| 最高奖励候选 progress delta | -0.00665 | +0.02813 |
| 最高奖励候选 path length delta | -0.067 m | +0.279 m |
| 最高奖励候选 collision cost delta | -0.00926 | -0.00924 |

guard 修正了候选排序，同时保留安全改善。

### 14.3 只改奖励仍然无效

即便 guard 5.0 已把候选方向改正，原始 `w*loss` 更新后仍减速。扩大至 1000 rollout 场景、500 update、学习率 `1e-7` 后，validation-1000 仍为：reward -0.02613、progress -0.04148、collision cost +0.00286、ADE +0.13843 m。

这说明问题不只是 reward 排序或 100 场景方差。

### 14.4 `beta=0` 控制实验

保持所有数据和更新参数不变，仅设置：

```text
rl_reward_temperature=0
```

此时所有有效候选权重均为 1，相当于无权自蒸馏。结果与 `beta=1` 几乎一致：

| 实验 | progress | ADE |
|---|---:|---:|
| beta=1 | -0.00758 | +0.01875 m |
| beta=0 | -0.00761 | +0.01886 m |

因此当前更新主要被权重中的常数 baseline `1` 支配，而不是被 reward-dependent 部分支配。有限 replay 上，无权自蒸馏梯度并不会像理论期望那样完全抵消。

### 14.5 deterministic update 消融

新增 `rl_deterministic_update`，默认 `true`：update 使用 `model.eval()` 关闭 Dropout/DropPath，但仍保留 autograd。该消融没有改变退化方向，因此随机层不是主因；保留该设置是为了使旧策略 target 与 update 前向条件一致。

## 15. 小数据 control-variate 修正

### 15.1 公式

新增参数：

```text
rl_center_reward_weights
```

默认 `false`，可复现论文原式；NuPlan 小数据实验显式设为 `true`。训练权重从：

```text
w = normalize(exp(beta * advantage))
```

改为：

```text
w_centered = w - 1
```

作用是消掉有限 replay 上 reward 无关的无权自蒸馏 baseline，只保留 reward 与 regression gradient 的协方差信号。该目标可能出现负 loss，这是 control-variate 的正常结果；它不是概率意义上的负 MSE。稳定性依靠小学习率、梯度裁剪和 EMA。

新增测试保证：当 `beta=0` 时，centered loss 和模型梯度都严格为 0。第一次运行暴露了 `[B]` 与 `[B,G]` 的广播错误；修复为 `active_groups[:, None]`，并把回归测试扩展到 `B=2`。失败运行没有生成可用 checkpoint。

完整测试：

```bash
PYTHONPATH=. \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q tests
```

结果：

```text
33 passed, 1 warning in 4.17s
```

warning 仍是 timm 弃用提示。

## 16. 正收益实验

### 16.1 100 场景初步门禁

centered weights、guard 5、学习率 `1e-6`、20 update 后，validation-1000 三次评估首次同时得到：reward +0.001591、progress +0.000060、collision cost -0.000018、comfort cost -0.000030、ADE -0.001125 m。方向为正但幅度很小。

把学习率提高到 `1e-5`、其他参数不变后：reward +0.014583、progress +0.000674、collision cost -0.000129、ADE -0.011252 m；comfort cost +0.001231，说明收益被放大但出现轻微舒适性副作用。

### 16.2 最终 1000 场景实验命令

工作目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
```

执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-paper-rl-progressguard5-centered-g32-lr4e7-gate500-from-omega01-epoch14 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/model_epoch_14_trainloss_0.9782.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --train_epochs 2 --batch_size 2 --learning_rate 4e-7 \
  --warm_up_epoch 1 --save_utd 1 --num_workers 2 \
  --planning_hybrid_loss 0.1 \
  --rl_group_size 32 --rl_rollout_steps 6 \
  --rl_sampling_noise_scale 0.1 \
  --rl_trajectory_augmentation_std 0 \
  --rl_trajectory_augmentation_epochs 0 \
  --rl_buffer_update_epoch 2 --rl_buffer_size 1024 \
  --rl_ema_update_rate 0.05 --rl_reward_temperature 1.0 \
  --rl_advantage_clip 5.0 --rl_min_reward_std 1e-6 \
  --rl_normalize_weights true --rl_center_reward_weights true \
  --rl_rollout_loss_weight 1.0 --rl_expert_anchor_weight 0.0 \
  --rl_max_update_steps_per_epoch 500 \
  --rl_detach_window_size 10 --rl_grad_clip 5.0 \
  --rl_freeze_encoder true --rl_deterministic_update true \
  --reward_progress_guard_weight 5.0 \
  --reward_progress_guard_stop_tolerance 0.2
```

训练结果：buffer size 1000、update steps 500、active group fraction 0.997、reward std mean 0.04165、regression weight mean `3.68e-9`，进程正常退出。

最终 EMA checkpoint：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-paper-rl-progressguard5-centered-g32-lr4e7-gate500-from-omega01-epoch14/
2026-08-08-19:47:02/model_epoch_2_trainloss_-0.0192.pth
```

### 16.3 最终评估命令与结果

评估使用 validation-1000、3 个种子、每次 6 步普通单轨迹推理；监督与 RL 两侧在每个 batch 前使用完全相同的随机种子，不使用 Best-of-N。

报告：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
behavior_paper_rl_progressguard5_centered_from_omega01_epoch14_
lr4e7_gate500_ema_val1k_repeat3.json
```

paired delta（RL - supervised）：

| 指标 | delta | 相对变化 |
|---|---:|---:|
| 总训练 reward | +0.032668 | +0.379% |
| progress | +0.036869 | +1.599% |
| path length | +0.373050 m | +1.599% |
| mean speed | +0.046631 m/s | +1.599% |
| low-speed fraction | -0.001333 | -0.258% |
| no-collision | +0.002667 | +0.287% |
| collision cost | -0.001261 | -1.308% |
| comfort cost | -0.002822 | -0.183% |
| ADE | -0.110914 m | -4.790% |
| FDE | -0.251088 m | -4.277% |
| heading error | -0.000295 rad | -0.864% |
| route cost | +0.000538 | +0.237% |
| stationary trajectory fraction | 0 | 0% |

论文原三项 reward 的净变化约为：

```text
delta_risk + 3 * delta_follow + 2.5 * delta_lane
= -0.000055 + 3 * 0.003555 + 2.5 * (-0.000377)
= +0.00967
```

所以正收益不只来自新增 progress guard；论文 risk/follow/lane 三项的加权和本身也提高，其中主要增益来自 follow reward。风险子奖励略降，但独立 OBB collision cost 和 no-collision 指标均改善，说明两个风险近似指标仍存在标定差异，后续闭环评估必须重点核查。

## 17. 当前结论与边界

1. 论文原式在当前 mini10k 监督起点上没有正收益，不能把工程可运行等同于算法有效。
2. 负收益的直接证据是 `beta=0` 与 `beta=1` 退化几乎相同，无权自蒸馏 baseline 支配有限 replay 更新。
3. bounded progress guard 修正了弱 IL 起点的候选减速偏好；centered weights 消除了 reward 无关梯度。两者均默认关闭并标为 NuPlan 小数据适配，不冒充论文原式。
4. 当前已在离线 pseudo-closed-loop 张量奖励和 open-loop paired behavior 上获得正收益；这还不是 NuPlan 官方闭环分数，更不是实车结论。
5. 下一道硬门禁应使用现有 mini-val closed-loop 场景对监督 epoch 14 与最终 RL EMA checkpoint 做同配置官方 NuPlan 闭环比较；route cost 的轻微退化必须单独检查。
6. 论文 `omega=0.1` 定义在物理 velocity 表示上，当前 NuPlan/HDP-NAVSIM 工程实际使用逐帧 displacement。两者线性相关但数值尺度并非严格相同，因此本日志称“论文参数一致”，不称“数学尺度完全等价”。

## 18. NuPlan 官方闭环 A/B 门禁（2026-08-09）

### 18.1 目的与受控变量

离线 pseudo-closed-loop reward 和 open-loop paired behavior 已经显示正收益，但二者都不能代替 NuPlan 官方仿真。因此本阶段固定：

- challenge：`closed_loop_nonreactive_agents`；
- simulation seed：`0`；
- worker：`sequential`；
- planner：`hyper_diffusion_planner`；
- 监督与 RL 使用相同的监督 `args.json`，保证模型结构、归一化和推理配置一致；
- 两侧只替换 EMA checkpoint；
- 先执行固定 3 场景工程 smoke，再执行固定 20 场景硬门禁。

公共 `args.json`：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json
```

监督 checkpoint：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/
model_epoch_14_trainloss_0.9782.pth
```

RL checkpoint：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-paper-rl-progressguard5-centered-g32-lr4e7-gate500-from-omega01-epoch14/
2026-08-08-19:47:02/model_epoch_2_trainloss_-0.0192.pth
```

### 18.2 预检

执行了路径、Python、NuPlan devkit、mini DB、shell 语法和固定场景测试检查：

```text
test_closed_loop_selection.py
test_closed_loop_summary.py
```

结果为 `2 passed in 0.82s`。GPU 为 RTX 4060 Laptop 8 GB，启动前仅使用 15 MiB。四个计划实验 UID 均没有历史结果目录。

### 18.3 第一次启动失败与修正

第一次监督 3 场景命令传入相对 `args.json`。脚本自身在项目目录预检时可以找到文件，但 Hydra 运行时改变工作目录，导致 planner 构建阶段报：

```text
FileNotFoundError: ... args.json
```

失败发生在 `Building simulations from 3 scenarios` 阶段，尚未启动任何仿真，也没有修改 checkpoint。失败目录保留为：

```text
tmp/closed_loop_eval/exp/simulation/closed_loop_nonreactive_agents/
hdp-paper-supervised-e14-cl3/
```

随后所有 `args.json` 和 checkpoint 均改用绝对路径，监督侧使用新 UID `hdp-paper-supervised-e14-cl3-retry1`，避免覆盖失败现场。

### 18.4 固定 3 场景 smoke

监督侧执行：

```bash
bash scripts/run_mini_closed_loop.sh \
  hdp-paper-supervised-e14-cl3-retry1 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/model_epoch_14_trainloss_0.9782.pth \
  mini-val-closed-loop-3
```

RL 侧执行相同命令，只替换 UID 与 checkpoint：

```bash
bash scripts/run_mini_closed_loop.sh \
  hdp-paper-rl-final-cl3 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-paper-rl-progressguard5-centered-g32-lr4e7-gate500-from-omega01-epoch14/2026-08-08-19:47:02/model_epoch_2_trainloss_-0.0192.pth \
  mini-val-closed-loop-3
```

两侧均为 `3/3` 成功、`0` 失败。监督耗时约 2 分 7 秒，RL 耗时约 2 分 13 秒。聚合结果：

| 指标 | supervised | RL | RL - supervised |
|---|---:|---:|---:|
| overall score | 0.942122 | 0.949534 | +0.007412 |
| expert-route progress | 0.814853 | 0.838573 | +0.023721 |
| no-at-fault collision | 1.000000 | 1.000000 | 0 |
| TTC within bound | 1.000000 | 1.000000 | 0 |
| drivable-area compliance | 1.000000 | 1.000000 | 0 |
| comfort | 1.000000 | 1.000000 | 0 |

满足“工程完整且关键指标不退化”的进入条件，因此继续 20 场景门禁。报告：

```text
tmp/closed_loop_eval/hdp-paper-e14-vs-rl-final-cl3.json
SHA-256: 42364fe861fcff719c2e0515b7103be1677899361bfab49b1e0dbd21685f4605
```

### 18.5 固定 20 场景硬门禁

监督侧执行：

```bash
bash scripts/run_mini_closed_loop.sh \
  hdp-paper-supervised-e14-cl20 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/model_epoch_14_trainloss_0.9782.pth \
  mini-val-closed-loop-20
```

RL 侧执行：

```bash
bash scripts/run_mini_closed_loop.sh \
  hdp-paper-rl-final-cl20 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-paper-supervised-balanced10k-omega01/2026-08-08-18:38:30/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-paper-rl-progressguard5-centered-g32-lr4e7-gate500-from-omega01-epoch14/2026-08-08-19:47:02/model_epoch_2_trainloss_-0.0192.pth \
  mini-val-closed-loop-20
```

监督与 RL 均为 `20/20` 成功、`0` 失败；仿真耗时分别约 14 分 21 秒和 14 分 40 秒。两边都在相同 3 个场景出现 `route_list empty` 警告，且所有 runner 与指标文件完整，因此继续把它记录为场景/route 指标提取现象，不归因于某一模型。

聚合结果：

| 指标 | supervised | RL | delta | 相对变化 |
|---|---:|---:|---:|---:|
| overall score | 0.755254 | 0.764288 | +0.009034 | +1.196% |
| expert-route progress | 0.656133 | 0.697180 | +0.041046 | +6.256% |
| no-at-fault collision | 1.000000 | 1.000000 | 0 | 0% |
| TTC within bound | 1.000000 | 1.000000 | 0 | 0% |
| drivable-area compliance | 0.950000 | 0.950000 | 0 | 0% |
| ego is making progress | 0.900000 | 0.900000 | 0 | 0% |
| driving-direction compliance | 0.975000 | 0.975000 | 0 | 0% |
| comfort | 1.000000 | 1.000000 | 0 | 0% |
| speed-limit compliance | 0.99998818 | 0.99998783 | -0.00000035 | -0.000035% |
| mean planner runtime | 0.226766 s | 0.232930 s | +0.006164 s | +2.718% |

逐场景配对统计：

- overall score：11 个提高、8 个相同、1 个降低；唯一降低为 `high_magnitude_speed` 场景 `134f95eed3775e22`，差值仅 `-1.77e-6`，来自 speed-limit compliance 的浮点级变化；
- expert-route progress：13 个提高、7 个相同、0 个降低；
- 最大 progress 增益为 `following_lane_without_lead` 场景 `1fb0bb88f9d35d59`，`+0.398825`；
- no-at-fault collision、TTC、drivable area、行驶方向和 comfort 的集合均与监督侧一致。

完整报告：

```text
tmp/closed_loop_eval/hdp-paper-e14-vs-rl-final-cl20.json
SHA-256: 4da0a2f800913174e12e7b73725c8715b053d943f53500a3d3e42b20a0db7c91
```

### 18.6 门禁结论与边界

本次固定 20 场景 NuPlan 官方 `closed_loop_nonreactive_agents` 门禁判定为 **通过**：RL 在安全、TTC、可行驶区域、行驶方向和舒适性均不退化的前提下，将路线进度提高 `6.256%`，使闭环总分提高 `1.196%`。这与离线阶段“恢复 progress 且不增加碰撞”的方向一致。

边界：20 个 mini-val 固定场景仍是小样本，且 challenge 中邻车为 non-reactive；该结果可以作为完整项目经历中的受控闭环正收益证据，但不能外推为完整 NuPlan benchmark、reactive closed-loop 或实车结论。下一步不应继续在这 20 个场景上调参，应扩大到独立且更多的闭环场景，防止对固定门禁集合过拟合。
