# HDP B Epoch10 训练过程详解

本文说明当前评测中所称的 **HDP B Epoch10** 是如何得到的。结论先说：它不是从随机权重独立训练十轮，而是先完成共同的 Epoch1～4，再从固定的 Epoch4 完整 checkpoint 分叉，以恒定 `5e-5` 学习率训练 Epoch5～10。

## 1. 模型身份

| 项目 | 内容 |
|---|---|
| 模型名称 | HDP B Epoch10 |
| `B` 的含义 | A/B 学习率对照中的低学习率分支，不代表另一种网络结构 |
| 模型类 | `hdp_nuplan.model.hyper_diffusion_planner.Hyper_Diffusion_Planner` |
| 训练入口 | `HDP-nuplan/train_predictor.py` |
| 单轮训练 | `HDP-nuplan/hdp_nuplan/train_epoch.py` |
| 监督损失 | `HDP-nuplan/hdp_nuplan/loss.py` |
| 参数量 | 5,092,996 |
| 最终 epoch | 10 |
| 最终 train loss | `0.009062370285391808` |
| 最终 checkpoint | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth` |
| checkpoint SHA256 | `22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce` |

最终 checkpoint 同时保存以下内容：

```text
epoch
model
ema_state_dict
optimizer
schedule
loss
wandb_id
```

闭环推理使用的是 checkpoint 中的 `ema_state_dict`，不是刚完成最后一次梯度更新的瞬时 `model` 权重。

## 2. 完整训练链路

```text
原始 Diffusion Planner checkpoint
        │
        │ 只加载兼容的 Encoder EMA 参数
        ▼
HDP 共同训练 Epoch1～3
Encoder 冻结，实际 LR=5e-5
        │
        ▼
HDP 共同训练 Epoch4
Encoder 解冻，实际 LR=5e-5
        │
        │ 固定复制 Epoch4 完整 checkpoint
        ▼
Experiment B：Epoch5～9
Encoder 可训练，恒定 LR=5e-5
        │
        ▼
Experiment B：Epoch10
Encoder 可训练，恒定 LR=5e-5
        │
        ▼
HDP B Epoch10 checkpoint
```

因此，B Epoch10 的有效权重历史是：

```text
Encoder warm-start + 4 个共同监督 epoch + 6 个 B 分支监督 epoch
```

## 3. 训练数据

训练使用本地完整 mini-train 缓存：

```text
NPZ 目录：
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache

训练 manifest：
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json
```

数据门禁结果为：

| 项目 | 数值 |
|---|---:|
| manifest 样本数 | 306,801 |
| manifest 去重后样本数 | 306,801 |
| NPZ 文件数 | 306,801 |
| 覆盖 mini-train DB 日志数 | 44 |
| 数据总大小 | 48,316,775,683 bytes，约 45.0 GiB |
| manifest SHA256 | `a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0` |

`DiffusionPlannerData` 根据 manifest 的 306,801 个文件名读取 NPZ。每个样本主要包含：

- 当前自车状态；
- 自车未来 80 个点；
- 最多 32 辆邻车的 21 帧历史；
- 最多 10 辆邻车的未来轨迹，监督训练中仅为数据增强和接口兼容保留；
- 最多 70 条 lane，每条 20 个点、每点 12 维；
- 最多 25 条 route lane，每条 20 个点、每点 12 维；
- lane/route lane 限速及有效标志；
- 最多 5 个静态物体。

`future_len=80`，采样间隔为 0.1 秒，对应未来 8 秒；`time_len=21` 对应包含当前帧在内的约 2 秒历史窗口。

## 4. 初始化：不是完整加载原始 DP

共同训练阶段以仓库原始 checkpoint 作为 Encoder warm-start：

```text
checkpoints/model.pth
SHA256：7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45
```

加载函数优先读取其中的 `ema_state_dict`，只选择名称和形状都与 HDP Encoder 匹配的参数。实际审计结果为：

| 项目 | 结果 |
|---|---:|
| HDP Encoder 目标张量数 | 151 |
| 成功加载张量数 | 151 |
| 成功加载参数数 | 1,799,040 |
| 缺失张量 | 0 |
| shape 不匹配 | 0 |
| 加载 Decoder 张量 | 0 |

这意味着：

- HDP Encoder 沿用了原始 Diffusion Planner 的场景理解能力；
- HDP Decoder 没有从原始 DP Decoder 加载参数，因为两者输出结构不同；
- HDP Decoder 在共同训练开始时使用自身初始化，然后在 Epoch1～4 中学习；
- B 分支开始时加载的是完整 HDP Epoch4 checkpoint，所以 B 的 Encoder、Decoder、EMA 和 AdamW 动量都有共同训练基础。

审计文件位于：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/
supervised_training_full_mini_omega001_nodetach/training_log/
hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/
encoder_warm_start_report.json
```

## 5. 共同阶段：Epoch1～4

### 5.1 启动参数

共同阶段最初由 `HDP-nuplan/scripts/finalize_full_mini_and_train.sh` 启动，核心参数为：

```bash
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 1 \
  --standalone \
  HDP-nuplan/train_predictor.py \
  --train_set HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache \
  --train_set_list HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path checkpoints/model.pth \
  --freeze_encoder_epochs 3 \
  --train_epochs 20 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --warm_up_epoch 2 \
  --save_utd 1 \
  --num_workers 4 \
  --planning_hybrid_loss 0.01 \
  --planning_detach_window_size 0 \
  --seed 3407 \
  --use_wandb false
```

虽然当时把总训练轮数写成 20，但后来在 Epoch4 分叉进行 A/B 对照。`train_epochs=20` 只是该进程的计划终点，不表示 B Epoch10 训练了 20 轮。

### 5.2 Encoder 冻结规则

训练循环使用零基 epoch 索引判断：

```python
encoder_trainable = epoch >= freeze_encoder_epochs
```

当 `freeze_encoder_epochs=3` 时：

| 人类所说 epoch | 程序中的 `epoch` | Encoder |
|---:|---:|---|
| 1 | 0 | 冻结 |
| 2 | 1 | 冻结 |
| 3 | 2 | 冻结 |
| 4 | 3 | 解冻 |

冻结只通过 `requires_grad=False` 阻止 Encoder 更新；Decoder 从 Epoch1 开始一直参与训练。Encoder 在 Epoch4 起与 Decoder 一起更新。

### 5.3 为什么实际 Epoch1～4 都是 `5e-5`

启动参数是基础学习率 `5e-4`、`warm_up_epoch=2`、warm-up 起始倍率 `0.1`，所以调度器首先把优化器学习率设为：

```text
5e-4 × 0.1 = 5e-5
```

训练代码当时按照下面的顺序工作：

```text
完成一个 epoch
→ 保存 checkpoint
→ scheduler.step()
```

共同阶段先后发生训练进程中断，并分别从最近一个完整 checkpoint 恢复。checkpoint 保存的是 `scheduler.step()` 之前的状态，因此每次恢复都再次回到 warm-up 起始状态。直接读取 Epoch1～4 checkpoint 后确认，四轮保存时的 optimizer LR 均为 `5e-5`。

这不是原计划的连续 warm-up 轨迹，而是当时中断与保存顺序共同造成的实际训练轨迹。B Epoch10 必须按这个事实描述，不能只根据命令中的 `--learning_rate 5e-4` 判断前四轮学习率。

### 5.4 中断和恢复

共同阶段发生过三次未完成 epoch 的重跑：

1. Epoch2 运行到约 26% 时 launcher 收到 `SIGTERM`，从完整 Epoch1 checkpoint 重跑 Epoch2；
2. Epoch3 运行到约 62% 时发生 CPU/RAM OOM，从完整 Epoch2 checkpoint 重跑 Epoch3，并把 `num_workers` 从 4 降为 0；
3. Epoch4 运行期间发现 `epoch_loss` 保存带计算图张量导致内存增长，修复为保存 `detach().item()` 标量后，从完整 Epoch3 checkpoint 重跑 Epoch4。

这些未完成 epoch 的更新没有进入最终 B 权重，因为恢复时重新加载了上一个完整 checkpoint 中的模型、EMA、optimizer 和 scheduler。`num_workers=4 → 0` 只改变数据加载并行度和墙钟速度，不改变样本集合、batch size 或损失定义。

### 5.5 共同阶段结果

| Epoch | Encoder | checkpoint optimizer LR | Train loss | checkpoint 时间 |
|---:|---|---:|---:|---|
| 1 | frozen | `5e-5` | `0.1196681112` | 2026-08-20 04:36:32 |
| 2 | frozen | `5e-5` | `0.0495298207` | 2026-08-20 10:33:06 |
| 3 | frozen | `5e-5` | `0.0348934792` | 2026-08-20 11:42:42 |
| 4 | trainable | `5e-5` | `0.0264890622` | 2026-08-20 12:44:54 |

## 6. 从 Epoch4 创建 Experiment B

为了比较“解冻 Encoder 后使用 `5e-4`”和“继续使用 `5e-5`”的差别，当时把共同 Epoch4 checkpoint 固定复制到 B 目录：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/
experiment_b_constant5e5_from_epoch4/
source_model_epoch_4_trainloss_0.0265.pth
```

该文件的 SHA256 为：

```text
b593a2d9671d84feeb20e247c3a16dc474fdd4ec5d35b0d97e467dda8fcac913
```

它是完整 checkpoint，包含：

- Epoch4 的即时模型权重；
- Epoch4 的 EMA 权重；
- AdamW 状态和动量；
- 学习率调度器状态；
- `epoch=4` 和 `loss=0.0264890622`。

B 不是把 Epoch4 的模型当作普通预训练权重重新开始，而是执行完整恢复。因此 `init_epoch=4`，训练循环从程序中的 epoch 索引 4 开始，也就是人类所说的 Epoch5。

## 7. B 分支：Epoch5～10

### 7.1 Epoch5～9

`HDP-nuplan/scripts/switch_epoch9_a_to_constant5e5_b.sh` 使用以下关键参数启动 B：

```bash
HDP-nuplan/train_predictor.py \
  --name hdp-full-mini-experiment-b-constant5e5-from-epoch4 \
  --save_dir HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4 \
  --train_set HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache \
  --train_set_list HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path HDP-nuplan/normalization.json \
  --resume_model_path HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4 \
  --freeze_encoder_epochs 3 \
  --train_epochs 9 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --warm_up_epoch 1 \
  --reset_lr_schedule_on_resume true \
  --save_utd 1 \
  --num_workers 0 \
  --planning_hybrid_loss 0.01 \
  --planning_detach_window_size 0 \
  --seed 3407 \
  --use_wandb false
```

这里最关键的是 `--reset_lr_schedule_on_resume true`：

- 保留 Epoch4 的模型权重；
- 保留 Epoch4 的 EMA 权重；
- 保留 AdamW 的一阶、二阶动量；
- 把 optimizer 的 `lr` 和 `initial_lr` 都重设为 `5e-5`；
- 丢弃旧学习率轨迹并用 `warm_up_epoch=1` 重建恒定倍率为 1 的 scheduler。

因此 Epoch5～9 的学习率始终是 `5e-5`，不会像实验 A 那样升到 `5e-4`。恢复后还执行 `train_sampler.set_epoch(4)`，使 Epoch5 的数据打乱序号与恢复位置对齐。

### 7.2 Epoch10

B 完成 Epoch9 后，由 `HDP-nuplan/scripts/train_ab_epoch10_then_closed_loop.sh` 从 B 目录中的 `latest.pth` 恢复：

```text
train_epochs=10
learning_rate=5e-5
warm_up_epoch=1
reset_lr_schedule_on_resume=true
```

所以只执行 Epoch10，没有重跑 Epoch1～9。Epoch10 从 2026-08-20 22:58:15 启动，约 1 小时 7 分钟后完成并保存 checkpoint。

保存完成后 `torchrun` 退出阶段收到 `SIGSEGV`，但 checkpoint 已完整写入并通过读取校验，包含模型、EMA、optimizer 和 scheduler，故不需要重新训练 Epoch10。

### 7.3 每轮结果

| Epoch | 所属阶段 | Encoder | LR | Train loss |
|---:|---|---|---:|---:|
| 1 | 共同阶段 | frozen | `5e-5` | `0.1196681112` |
| 2 | 共同阶段 | frozen | `5e-5` | `0.0495298207` |
| 3 | 共同阶段 | frozen | `5e-5` | `0.0348934792` |
| 4 | 共同阶段 | trainable | `5e-5` | `0.0264890622` |
| 5 | B 分支 | trainable | `5e-5` | `0.0186701473` |
| 6 | B 分支 | trainable | `5e-5` | `0.0147690745` |
| 7 | B 分支 | trainable | `5e-5` | `0.0127139408` |
| 8 | B 分支 | trainable | `5e-5` | `0.0110670766` |
| 9 | B 分支 | trainable | `5e-5` | `0.0099780047` |
| 10 | B 分支 | trainable | `5e-5` | `0.0090623703` |

最终有效链路共完成 `10 × 38,350 = 383,500` 个 optimizer update。由于 `drop_last=true`，每轮实际使用 `38,350 × 8 = 306,800` 个样本槽位，306,801 个样本中每轮有 1 个位于打乱后的尾部而被丢弃；不同 epoch 会重新打乱。

## 8. 一个 batch 内发生什么

### 8.1 数据加载和增强

单卡、单进程 DDP 下，全局 batch size 与单进程 batch size 都是 8。`DistributedSampler(shuffle=true)` 负责每轮确定性打乱，DataLoader 使用：

```text
batch_size=8
num_workers=0（稳定版本）
pin_memory=true
drop_last=true
```

训练开启 `StatePerturbation`，`augment_prob=0.5`。对满足速度条件的样本随机扰动当前自车的横向位置、航向、速度、加速度等，并用五次多项式重新连接前 2 秒未来轨迹；随后同步变换自车、邻车、lane、route 和静态物体坐标，保证输入与标签仍处于同一自车坐标系。

### 8.2 监督目标

自车未来原始标签为：

```text
[x, y, heading]，形状 [B, 80, 3]
```

首先把 heading 转换为 `cos(heading), sin(heading)`，然后对 x/y 沿时间做差分：

```text
[Δx, Δy, cos(heading), sin(heading)]，形状 [B, 80, 4]
```

第一个 x/y 增量相对于当前自车坐标系原点计算。这里的“增量”不是简单把四个属性全部差分：x/y 做差分，cos/sin 保留对应未来时刻的方向表示。

状态归一化参数为：

```text
mean = [0, 0, 0, 0]
std  = [0.5, 0.5, 1, 1]
```

因此主要把 x/y 位移缩放到更适合网络训练的数值范围。

### 8.3 扩散监督

每个样本独立采样：

```text
t ~ Uniform(0.001, 1)
z ~ Normal(0, I)
```

训练使用 `VPSDE_linear(beta_min=0.1, beta_max=20.0)` 计算给定 t 下的均值和标准差，再构造加噪状态：

```text
x_t = mean + std × z
```

模型的输出类型和监督类型都是 `x_start`，即直接预测未加噪的归一化自车运动序列。基础扩散损失是预测与真值在 4 个状态维度上的平方误差求和，再对 batch 和 80 个未来时间点取平均：

```text
L_increment = mean(sum((pred_x_start - gt_increment)^2, state_dim))
```

### 8.4 积分轨迹混合损失

模型预测的 x/y 是逐帧增量。将其逆归一化并沿时间累计，得到具体位置轨迹：

```text
pred_position = cumsum(pred_increment_xy, time)
```

然后与真实未来 x/y 计算位置均方误差：

```text
L_position = mean(sum((pred_position - gt_position)^2, xy_dim))
```

B Epoch10 使用：

```text
planning_hybrid_loss=0.01
planning_detach_window_size=0
```

最终损失为：

```text
L_total = L_increment + 0.01 × L_position
```

`planning_detach_window_size=0` 表示普通 `torch.cumsum`，位置损失的梯度可以沿积分关系传播到此前所有相关位移；B Epoch10 没有启用 detach。HDP 不计算原始 Diffusion Planner 的邻车预测损失。

### 8.5 参数更新

每个 batch 的更新顺序为：

1. `optimizer.zero_grad()`；
2. Encoder 和 Decoder 前向；
3. 计算扩散增量损失和积分位置损失；
4. `loss.backward()`；
5. 将全模型梯度范数裁剪到 5；
6. AdamW 更新可训练参数；
7. 以 `decay=0.999` 更新 EMA；
8. 把已 detach 的标量 loss 加入 epoch 统计。

AdamW 的实际参数为：

```text
lr=5e-5（B 分支）
betas=(0.9, 0.999)
eps=1e-8
weight_decay=0.01
amsgrad=false
```

## 9. 模型与训练参数汇总

下面是 B Epoch10 当时实际有效的配置，不采用后来 Epoch20 续训覆盖后的实验名称和总轮数。

| 类别 | 参数 | B Epoch10 值 |
|---|---|---|
| 数据 | `train_set` | `.../mini_train_full_306801_seed3407_v1/cache` |
| 数据 | `train_set_list` | `.../mini_train_full_306801_seed3407_v1/diffusion_planner_training.json` |
| 数据 | `future_len` | `80` |
| 数据 | `time_len` | `21` |
| 数据 | `agent_state_dim` / `agent_num` | `11 / 32` |
| 数据 | `static_objects_state_dim` / `static_objects_num` | `10 / 5` |
| 数据 | `lane_len` / `lane_state_dim` / `lane_num` | `20 / 12 / 70` |
| 数据 | `route_len` / `route_state_dim` / `route_num` | `20 / 12 / 25` |
| 数据 | `predicted_neighbor_num` | `10`，仅接口兼容，不参与 HDP 邻车预测损失 |
| 增强 | `use_data_augment` | `true` |
| 增强 | `augment_prob` | `0.5` |
| 加载 | `num_workers` | B 分支为 `0` |
| 加载 | `pin_mem` | `true` |
| 随机性 | `seed` | `3407` |
| 训练 | `batch_size` | `8` |
| 训练 | B 分支 `learning_rate` | `5e-5` |
| 训练 | B 分支 `warm_up_epoch` | `1`，实现为恒定学习率 |
| 训练 | `reset_lr_schedule_on_resume` | `true` |
| 训练 | `save_utd` | `1`，每个 epoch 保存 |
| 训练 | `freeze_encoder_epochs` | `3` |
| 训练 | `use_ema` / EMA decay | `true / 0.999` |
| 训练 | gradient clipping | 全模型 norm `5` |
| 损失 | `planning_hybrid_loss` | `0.01` |
| 损失 | `planning_detach_window_size` | `0`，无 detach |
| 扩散 | `diffusion_model_type` | `x_start` |
| 扩散 | `diffusion_supervision_type` | `x_start` |
| 扩散 | SDE | `VPSDE_linear(0.1, 20.0)` |
| 模型 | `hidden_dim` | `192` |
| 模型 | `encoder_depth` / `decoder_depth` | `3 / 3` |
| 模型 | `num_heads` | `6` |
| 模型 | `encoder_drop_path_rate` / `decoder_drop_path_rate` | `0.1 / 0.1` |
| 运行 | `device` | `cuda` |
| 运行 | DDP | `true`，但 `world_size=1` |
| 记录 | `use_wandb` | `false` |

归一化配置文件为：

```text
HDP-nuplan/normalization.json
SHA256：c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a
```

## 10. 为什么叫实验 B

共同 Epoch4 之后建立了两个分支：

| 条件 | 实验 A | 实验 B |
|---|---|---|
| 起点 | 同一个 Epoch4 checkpoint | 同一个 Epoch4 checkpoint |
| Epoch5～10 学习率 | `5e-4` | 恒定 `5e-5` |
| Encoder | trainable | trainable |
| 数据和 manifest | 相同 | 相同 |
| batch size | 8 | 8 |
| hybrid loss / detach | `0.01 / 0` | `0.01 / 0` |
| seed | 3407 | 3407 |

B 的目的就是检验：Encoder 解冻后，不把学习率提高十倍，而是继续用 `5e-5`，是否能得到更好的规划策略。

固定 20 场景闭环中，B Epoch10 的 NuPlan score 为 `0.876941`，实验 A Epoch10 为 `0.821742`；B 的优势主要来自沿专家路线前进率。但这是模型选择过程中的小规模固定场景结果，不等同于完整 Test14 的结论。

## 11. `args.json` 为什么看起来是 Epoch20

B 目录中的当前文件：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/
experiment_b_constant5e5_from_epoch4/args.json
```

目前包含：

```text
name=hdp-full-mini-experiment-b-epoch20-constant5e5
train_epochs=20
```

原因是 B Epoch10 后来又从 Epoch10 续训到了 Epoch20。`train_predictor.py` 在使用 `resume_model_path` 时，把恢复目录同时作为 `save_path`，并在每次启动训练时重新向同一个 `args.json` 写入当前参数。因此 Epoch20 的启动参数覆盖了先前 Epoch9、Epoch10 写入的内容。

这不会修改已经单独保存的 `model_epoch_10_trainloss_0.0091.pth`，但会使当前 `args.json` 不能单独证明 Epoch10 当时的 `train_epochs=10`。B Epoch10 的历史参数应联合以下文件还原：

- `HDP-nuplan/scripts/finalize_full_mini_and_train.sh`；
- `HDP-nuplan/scripts/resume_full_mini_training_numworkers0.sh`；
- `HDP-nuplan/scripts/switch_epoch9_a_to_constant5e5_b.sh`；
- `HDP-nuplan/scripts/train_ab_epoch10_then_closed_loop.sh`；
- Epoch1～10 checkpoint 内的 epoch、loss、optimizer 和 scheduler 状态；
- 对应训练日志。

## 12. 可复现性边界

当前 checkpoint 保存模型、EMA、optimizer 和 scheduler，但没有保存 Python、NumPy、PyTorch 与 CUDA 的 RNG 状态。因此：

- 数据 sampler 的 epoch 已在恢复时对齐，数据集合和每轮打乱编号可控；
- 每次恢复都会重新执行 `set_seed(3407)`；
- 扩散时间、扩散噪声和数据增强的随机流不等同于一次从 Epoch1 连续运行到 Epoch10 的随机流；
- 现有 B Epoch10 可以按 checkpoint 精确用于推理和评测，但若从零重新训练，只能复现训练协议，不能保证逐参数位级一致。

此外，共同阶段的数次未完成 epoch 虽然最后被 checkpoint 回滚，不进入最终模型，但确实消耗过额外计算时间；因此不能用十个完整 epoch 的理论耗时直接解释当时的总墙钟时间。

## 13. 核验依据

| 证据 | 路径 |
|---|---|
| 数据合并报告 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/merged_processing_report.json` |
| 数据门禁报告 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache_validation_report.json` |
| Encoder warm-start 报告 | `.../2026-08-20-04:08:42/encoder_warm_start_report.json` |
| 共同阶段首次日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_full_mini_omega001_nodetach_train.log` |
| 共同阶段恢复日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_full_mini_omega001_nodetach_resume_numworkers0.log` |
| A→B 切换日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/switch_epoch9_a_to_b.log` |
| B Epoch5～9 日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4.log` |
| B Epoch10 日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_epoch10_resume.log` |
| Epoch10 流水线日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/train_ab_epoch10_then_closed_loop.log` |
| 原始操作日志 | `HDP-nuplan/doc_hdp_nuplan/正式监督训练NPZ合并操作日志.md` 的第 16 节 |

最后核验日期：2026-09-07。本文以磁盘中的训练脚本、日志、数据报告和 checkpoint 元数据为准。
