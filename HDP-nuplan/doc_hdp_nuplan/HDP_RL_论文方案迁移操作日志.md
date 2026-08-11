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

## 19. 在新 11,040 数据上验证论文 RL 正收益

时间：2026-08-11。

### 19.1 实验问题重新定义

本轮唯一主要问题改为：

```text
从 target-matched 11,040 新数据训练出的论文一致监督模型出发，
执行第 18 节已经获得正收益的 RL 方法后，闭环指标能否相对其自身监督起点提高？
```

因此，旧 `omega=0.01` balanced 10k epoch 10 只保留为历史排行榜参考，不再作为是否允许进入
RL 的直接门禁。新实验的因果比较必须是同一条模型链：

```text
target-matched 11,040 supervised（omega=0.1）
                    ↓ 从选中的监督 checkpoint 启动 RL
target-matched 11,040 RL（omega=0.1）
```

不能把既有 target-matched 11,040 `omega=0.01` checkpoint 直接当作论文一致起点，也不能把新
监督模型直接与旧论文 RL checkpoint 的绝对分数差解释为 RL 收益。

### 19.2 受控变量与判定标准

监督阶段复用第 11 节论文实验配置，只把训练集从 balanced 10k 换为 target-matched 11,040：

| 项目 | 本轮值 |
|---|---:|
| code commit | `720025446c0cb86bb236e4903454864db0b71153` |
| train samples | 11,040 |
| seed | 3407 |
| batch size | 8 |
| epochs | 20 |
| learning rate | `5e-4` |
| warm-up | 2 epochs |
| hybrid loss weight | `0.1` |
| encoder initialization | 发布版 `checkpoints/model.pth` |
| encoder frozen | 前 3 epochs |
| EMA | true |

数据与初始化证据：

```text
manifest SHA-256: 808d3b170d7067785e2120d7f4cffd81b427bdbae6f7fc62c9a634e5206e8bac
normalization SHA-256: c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a
encoder checkpoint SHA-256: 7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45
cache validation: passed；11,040 个 manifest 条目；11,040 个唯一 NPZ；train/val 日志重叠为 0
```

监督训练完成后，先用独立 val1k 对全部 checkpoint 排名并复测候选，再在固定 20 场景获取 RL
前基线。监督模型不要求先超过历史旧 10k；本轮要验证的是 RL 的配对增量。RL 正收益硬门禁为：

1. `RL overall score > paired supervised overall score`；
2. expert-route progress 应提高；
3. no-at-fault collision、TTC、drivable-area compliance 不得下降；
4. 两侧必须使用完全相同的场景 token、challenge、仿真环境和聚合脚本；
5. 若只提高训练 reward 或 open-loop loss，而固定 20 场景不提高，不判为 RL 正收益。

### 19.3 云端启动准备与第一次连接结果

计划使用实验名 `hdp-paper-supervised-targetmatch11040-omega0p1-e20`，避免与此前名字中含义容易
混淆的 `omega01` 写法重名。预定命令为：

```bash
screen -dmS hdp11040_paper_sup bash -lc '
  cd /root/autodl-tmp/workspace/Diffusion-Planner
  set -o pipefail
  CUDA_VISIBLE_DEVICES=0 \
  /root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
    --nnodes 1 \
    --nproc-per-node 1 \
    --standalone \
    HDP-nuplan/train_predictor.py \
    --name hdp-paper-supervised-targetmatch11040-omega0p1-e20 \
    --save_dir /root/autodl-tmp/experiments/hdp_targetmatch_11040_paper_omega0p1_e20 \
    --train_set /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache \
    --train_set_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/diffusion_planner_training.json \
    --normalization_file_path /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/normalization.json \
    --batch_size 8 \
    --train_epochs 20 \
    --save_utd 1 \
    --num_workers 4 \
    --learning_rate 0.0005 \
    --warm_up_epoch 2 \
    --planning_hybrid_loss 0.1 \
    --encoder_pretrained_model_path /root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth \
    --freeze_encoder_epochs 3 \
    --diffusion_model_type x_start \
    --diffusion_supervision_type x_start \
    --use_data_augment true \
    --augment_prob 0.5 \
    --use_ema true \
    --use_wandb false \
    --ddp true \
    --seed 3407 \
    --pin-mem \
    2>&1 | tee /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_e20.log
  rc=${PIPESTATUS[0]}
  printf "%s\\n" "$rc" | tee /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_e20.exit
  exit "$rc"
'
```

2026-08-11 首次连接 `connect.cqa1.seetacloud.com:11156` 返回 `Connection refused`，因此该命令
尚未在云端执行，也没有产生半启动实验。必须在实例恢复 SSH 后重新检查 GPU、11,040 缓存数量、
磁盘、现有训练进程、代码 commit 和目标目录不存在，全部通过后才能启动。

### 19.4 云端恢复、验收与正式启动

同日再次连接成功。启动前只读验收结果：

```text
host: autodl-container-b8e548a0d5-61312194
GPU: NVIDIA GeForce RTX 4090 24GB，启动前显存 1 MiB、利用率 0%
data disk: 170GB，总可用 111GB
Python: 3.9.25
NPZ: 11,040
已有训练/评测进程: 0
已有 screen: 0
目标实验目录: 不存在
```

云端工程是归档解压目录，没有 `.git`，所以不能用 `git rev-parse` 证明版本；改为比较训练入口、
loss、epoch 实现的 SHA-256。本地与云端三份文件逐字节一致：

```text
cc31d257b3ae89b6048626fcca0f90eaf8672c0bc8b9fcd0cbf89a0a2183a8cf  HDP-nuplan/train_predictor.py
eb94be2a052f8a8f9f6a5cd66cdeaf20e7c8601fbc63c93584185f349ba44d09  HDP-nuplan/hdp_nuplan/loss.py
41e04381a20dffd3da34425c0a3dfdedc3d0290f4abf54ba19d60e6b728fb10e  HDP-nuplan/hdp_nuplan/train_epoch.py
```

训练前执行全部测试：`56 passed, 15 warnings in 4.22s`，退出码 0；warning 均为第三方包的弃用
提示，不是测试失败。

随后执行第 19.3 节命令，正式启动成功：

```text
screen: hdp11040_paper_sup
启动时间: 2026-08-11 00:15:30 CST
run_dir: /root/autodl-tmp/experiments/hdp_targetmatch_11040_paper_omega0p1_e20/training_log/
         hdp-paper-supervised-targetmatch11040-omega0p1-e20/2026-08-11-00:15:33/
log: /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_e20.log
exit: /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_e20.exit
```

启动后日志确认 `Dataset Prepared: 11040`、encoder `151/151` 张量加载成功、epoch 1 encoder 为
冻结状态，并进入 `1380` batch 的训练循环。保存的 `args.json` 已逐项读取，确认
`planning_hybrid_loss=0.1`，而不是旧实验的 `0.01`。

另启动 `hdp11040_paper_eval` watcher。它在训练期间只每 30 秒检查退出码；训练成功后自动对全部
checkpoint 执行独立 val1k、3 repeats 排名：

```text
screen: hdp11040_paper_eval
evaluation output: /root/autodl-tmp/experiments/hdp_targetmatch_11040_paper_omega0p1_e20/evaluation/
                   mini_val_1000_checkpoint_ranking_repeat3.json
evaluation log: /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_val1k_repeat3.log
evaluation exit: /root/autodl-tmp/logs/hdp_targetmatch_11040_paper_omega0p1_val1k_repeat3.exit
```

watcher 不会自动启动 RL。val1k 排名完成后仍需检查候选，再用固定 20 场景建立配对监督基线。

首轮训练完成后的落盘验收：

```text
epoch 1 train loss: 3.8623
checkpoint: model_epoch_1_trainloss_3.8623.pth
checkpoint size: 67,417,699 bytes
next state: Epoch 2/20，training exit 文件尚未生成
```

首轮 loss 与第 11 节 balanced 10k 论文监督实验的 `3.7471` 同一量级；日志无 NaN、Traceback
或提前退出。这里只判断训练链路正常，不根据单轮训练 loss 判断最终模型优劣。

冻结/解冻检查随后通过：

```text
epoch 1: encoder trainable=False，train loss=3.8623
epoch 2: encoder trainable=False，train loss=2.6349
epoch 3: encoder trainable=False，train loss=1.4178
epoch 4: encoder trainable=True，显存由约 927 MiB 增至约 1,803 MiB
```

显存增长来自 encoder 梯度与优化器状态开始参与训练，证明 `freeze_encoder_epochs=3` 不是只写入
参数文件而未执行。前台验收到 `2026-08-11 00:24:53 CST` 时，5 个 checkpoint 已落盘并进入
epoch 6；训练与评测 watcher 均保持运行。随后停止的只是本地 SSH 前台轮询，不是云端两个 screen。

### 19.5 监督训练完成与 val1k 自动排名

2026-08-11 01:01 CST 复核：监督训练和自动 val1k 排名均已完成，两个退出码均为 0；20 个 epoch
对应的 20 个 checkpoint 全部落盘，GPU 已空闲。最后七轮训练 loss 为：

```text
epoch 14: 1.1029    epoch 15: 1.0895    epoch 16: 1.1276
epoch 17: 1.0685    epoch 18: 1.0702    epoch 19: 0.9356
epoch 20: 0.9882
```

独立 val1k、每个 checkpoint 3 repeats 的前五名为：

| 排名 | epoch | train loss | mean val total loss |
|---:|---:|---:|---:|
| 1 | 10 | 1.2008 | 0.613795605 |
| 2 | 14 | 1.1029 | 0.619970966 |
| 3 | 9 | 1.3176 | 0.631156356 |
| 4 | 15 | 1.0895 | 0.635913728 |
| 5 | 13 | 1.1211 | 0.640057264 |

epoch 10 的三次 total loss 为 `0.659980163、0.565663225、0.615743427`，均值为
`0.613795605`。因此当前第一候选是 epoch 10，而不是 train loss 最低的 epoch 19，也不是最后
epoch。证据哈希：

```text
0b179619534f9e26e2d498ac954c889587fd170cd019547c77fc8eedcb73f16c  model_epoch_10_trainloss_1.2008.pth
2a3b0349586ba64838b143fc1883992ce9025b3caf4886453dbab05d000691b9  args.json
79540a1a54d71e1e812d538816041d3fe34d26d3cc8113733ca1c6665a2295cc  mini_val_1000_checkpoint_ranking_repeat3.json
```

当前尚未启动 RL。下一步是把 epoch 10 与 `args.json` 下载到本地并校验哈希，在固定 20 场景上
获取这一个监督模型的 RL 前基线；之后才能从同一 epoch 10 checkpoint 启动论文 RL。

### 19.6 epoch 10 下载与配对监督闭环基线

下载到本地：

```text
tmp/targetmatch_11040_paper_omega0p1_eval/checkpoints/model_epoch_10_trainloss_1.2008.pth
tmp/targetmatch_11040_paper_omega0p1_eval/checkpoints/args.json
tmp/targetmatch_11040_paper_omega0p1_eval/open_loop/mini_val_1000_checkpoint_ranking_repeat3.json
```

三个文件的本地 SHA-256 与第 19.5 节云端值完全一致。随后在产生历史基线的本机环境，用
`mini-val-closed-loop-20`、`closed_loop_nonreactive_agents`、seed 0、sequential worker 运行：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_paper_omega0p1_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  hdp-paper-targetmatch11040-omega0p1-e10-cl20 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_paper_omega0p1_eval/checkpoints/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_paper_omega0p1_eval/checkpoints/model_epoch_10_trainloss_1.2008.pth \
  mini-val-closed-loop-20 \
  hdp
```

运行耗时约 16 分 12 秒，`20/20` 成功、`0` 失败，出现 3 次两侧共有的 `route_list empty`
warning。监督基线为：

| 指标 | supervised epoch 10 |
|---|---:|
| overall score | 0.703840943 |
| expert-route progress | 0.653433099 |
| no-at-fault collision | 0.900000 |
| TTC within bound | 0.800000 |
| drivable-area compliance | 0.950000 |
| ego is making progress | 0.900000 |
| driving-direction compliance | 1.000000 |
| comfort | 1.000000 |
| speed-limit compliance | 0.999991224 |

该模型低于历史旧 10k，但本轮检验的是从它自身出发的 RL 配对增量，因此仍是有效起点。

### 19.7 从新 11,040 固定抽取 RL rollout 1k

没有复用旧 `mini_train_pilot_1000`。从 target-matched 11,040 manifest 内按新训练集目标城市比例
无放回抽样，seed 固定为 3407：

| 城市 | 数量 |
|---|---:|
| Las Vegas | 700 |
| Singapore | 100 |
| Boston | 100 |
| Pittsburgh | 100 |

结果为 1,000 条唯一记录、0 缺失；其中 905 条属于旧 balanced 10k，95 条来自新增 1,040。
DataLoader 冒烟读取成功，返回 11 个场景张量和 1 个文件名元数据。只生成子集 manifest，不复制
1,000 个 NPZ；训练时 `train_set` 仍指向完整 11,040 cache，`train_set_list` 只允许读取这 1,000
条。证据：

```text
e50911179b71c327e6c9ba2081f9d94be1520b65f1529b7dd913bf1bee617255  diffusion_planner_training.json
2a5d4c6adeb368a35d337ba1f7c8435048ca9c209d14699d2786016a32575af7  selection_report.json
```

### 19.8 配对 RL 训练

云端确认 GPU 空闲、无旧 RL 进程、目标目录不存在，并确认 `train_predictor_rl.py`、
`train_epoch_rl.py`、RL loss 和 reward 四个核心文件与本地哈希一致。沿用第 16.2 节正收益参数，
只替换监督起点与 rollout manifest：

```text
supervised start: target-matched 11,040 omega=0.1 epoch 10
rollout set: 第 19.7 节固定 1,000 条
batch size: 2
learning rate: 4e-7
group size: 32
rollout steps: 6
buffer size: 1024
rollout/update epochs: 2
update steps: 500
center reward weights: true
expert anchor: 0
progress guard weight: 5
encoder frozen: true
deterministic update: true
```

训练于 `2026-08-11 01:42:48 CST` 启动并正常退出。结果：

```text
buffer size: 1000
update steps: 500
active group fraction: 0.999
reward std mean: 0.0587204
regression weight mean: -1.11e-9
final centered loss: -0.0299900
```

最终 EMA checkpoint：

```text
tmp/targetmatch_11040_paper_omega0p1_eval/checkpoints/model_epoch_2_trainloss_-0.0300.pth
SHA-256: 45207884470e1674b0fde7d5e2612598e7f800b2704e1232014978f0292e5df8
```

### 19.9 固定 20 场景 RL 前后最终配对

RL 侧只替换 checkpoint，继续使用监督侧同一个 `args.json`、20 个 token、seed、challenge、环境和
聚合脚本。RL 侧耗时约 16 分 16 秒，同样为 `20/20` 成功、`0` 失败，并在完全相同的 3 个场景
出现 route warning。

| 指标 | supervised | RL | RL - supervised | 相对变化 |
|---|---:|---:|---:|---:|
| overall score | 0.703840943 | 0.704483158 | +0.000642215 | +0.091244% |
| expert-route progress | 0.653433099 | 0.676960608 | +0.023527509 | +3.600599% |
| no-at-fault collision | 0.900000 | 0.900000 | 0 | 0% |
| TTC within bound | 0.800000 | 0.850000 | +0.050000 | +6.250000% |
| drivable-area compliance | 0.950000 | 0.950000 | 0 | 0% |
| ego is making progress | 0.900000 | 0.900000 | 0 | 0% |
| driving-direction compliance | 1.000000 | 0.975000 | -0.025000 | -2.500000% |
| comfort | 1.000000 | 1.000000 | 0 | 0% |
| speed-limit compliance | 0.999991224 | 0.999990815 | -0.000000409 | -0.000041% |

逐场景配对：route progress 为 15 个提高、5 个不变、0 个下降；overall score 为 10 个提高、9 个
不变、1 个下降。两个主要变化为：

1. `high_magnitude_speed` 场景 `134f95eed3775e22`：TTC 从 0 升至 1，score
   `0.650149 → 0.974248`；
2. `following_lane_without_lead` 场景 `1fb0bb88f9d35d59`：progress 仍提高
   `0.446678 → 0.466607`，但 driving direction 从 1 降至 0.5，使 score
   `0.827087 → 0.416657`。

最终结论：按第 19.2 节预先声明的三项安全门禁（collision、TTC、drivable area），本轮实现了
平均正收益；但 overall 仅提高 `0.091%`，且一个场景出现明显方向合规退化，因此只能称为
**带方向副作用的弱正收益**，不能称为全面或稳健提升。下一轮优先修复 driving-direction guard，
而不是继续放大 progress reward。

完整逐场景报告：

```text
tmp/targetmatch_11040_paper_omega0p1_eval/supervised_vs_rl_mini_val_20.json
SHA-256: 8ec8e1ee9fd99cf7f83996acddad0d9c7b45cf9fde43cb606a75243467178b50
```
