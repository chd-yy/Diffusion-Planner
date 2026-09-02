# 原始 Diffusion Planner 与 B Epoch10 同数据对照实验

## 1. 实验目的

当前 fixed200 结果显示，仓库发布版原始 Diffusion Planner 的 score 为 `0.957971`，
B Epoch10 为 `0.904567`。为区分“训练数据/训练规模差异”和“模型结构及训练目标差异”，
使用 B Epoch10 完全相同的 mini-train 数据重新训练原始 Diffusion Planner，再使用同一
fixed200 场景集合进行闭环比较。

## 2. 对齐原则

本实验对齐：

- 训练数据目录和 manifest：完整 mini-train，`306,801` 个 NPZ；
- 随机种子：`3407`；
- batch size：`8`；
- 训练 epoch：`10`；
- 学习率：`5e-5`；
- warm-up：`1` epoch；
- 数据增强：启用，概率 `0.5`；
- DataLoader worker：`0`；
- EMA：启用；
- fixed200 闭环场景、NuPlan DB、metric 和 runner 配置。

不强行对齐模型结构和损失函数：原始 Diffusion Planner 与 HDP 的输出语义不同，分别
使用各自原生的邻车预测与自车规划目标。该实验因此用于评估“同数据训练后的工程性能
差异”，不构成严格的同架构消融实验。

## 3. 训练入口与产物

训练入口：

```text
train_predictor.py
```

执行脚本：

```text
HDP-nuplan/scripts/train_original_diffusion_same_data.sh
```

输出目录：

```text
HDP-nuplan/tmp/original_diffusion_retrain_306801_seed3407_epoch10/
```

训练期间每个 epoch 保存 checkpoint，以便异常中断后续训；训练完成后使用与 B Epoch10
相同的 `mini-val-fixed-200-rl-v4` 进行闭环评测和逐场景配对分析。

## 4. 当前状态

2026-08-26 已完成训练参数、数据 manifest、checkpoint 来源和磁盘空间核对，正式训练
已启动，当前处于 Epoch 1。该数据集每个 epoch 为 `38,350` 个 batch，单卡实测速度约
`11 batch/s`，预计每个 epoch 约 `55` 分钟，10 个 epoch 约 `9` 小时（速度会随数据读取
和系统负载变化）。

启动过程中发现并修复了两个仅影响入口兼容性的非算法问题：

1. `use_wandb=false` 时日志模块仍无条件导入 WandB；现改为关闭 WandB 时仅使用
   TensorBoard，不改变训练目标；
2. 训练入口未提供推理期可选的 `guidance_fn`，Decoder 直接访问导致初始化失败；现将
   未提供时处理为 `None`，保持无 guidance 的原始路径；
3. 单卡 DDP 检查到部分参数未参与当前 loss，已开启
   `find_unused_parameters=True`，仅修复梯度归约，不改变 loss 或优化目标。

首次正式运行在 Epoch 1 的 `26,656/38,350` batch（约 `69.5%`）处被系统 OOM killer
终止，未生成 checkpoint。内核日志确认当时 16GB RAM 和 16GB swap 接近耗尽，训练
进程常驻内存约 10GB；不是 CUDA OOM，也不是磁盘空间不足。

进一步定位发现，原始 `diffusion_planner/train_epoch.py` 将每个 batch 的 loss 张量直接
追加到 `epoch_loss`，导致计算图跨 batch 累积。现改为仅保存
`value.detach().item()` 标量；反向传播在保存前已经完成，因此该修复不改变 loss、梯度
或优化结果。同时将保存周期由 10 epoch 改为每个 epoch 保存，降低再次中断时的进度
损失。修复后已从 Epoch 1 重新启动；当前约为 `151/38,350` batch，训练进程常驻内存
约 `1.2GB`，未再出现随 batch 增长的内存占用。当前尚未生成 Epoch checkpoint，需等
待 Epoch 1 完成后确认保存结果。

## 5. 与 B Epoch10 的训练速度对比

实际日志表明两者单 epoch 速度接近，并不存在原始模型慢数倍的情况：

- B Epoch9：`38,350` batch，`53分29秒`；
- B Epoch10：`38,350` batch，`1小时07分25秒`；
- 当前原始 Diffusion Planner Epoch1：按运行中平均速度推算约 `53～55分钟`。

体感上当前训练更慢，主要因为首次运行到 Epoch1 的 `69.5%` 时发生系统内存 OOM，约
38 分钟计算没有形成 checkpoint，修复后只能从头重跑。另外，B Epoch10 实验阶段是从
已有 Epoch4 checkpoint 继续到 Epoch10，只需新增 6 个 epoch；当前对照实验为避免加载
官方预训练权重带来额外优势，从随机初始化完整训练 10 个 epoch。

原始模型单 batch 的理论计算量略高：参数量为 `6,042,628`，B 模型约为 `5,092,996`；
原始模型还联合预测自车和 10 个邻车未来轨迹，而 B 只优化自车轨迹。但当前训练同时受
NPZ 读取和 CPU 数据处理限制，因此这部分差异没有导致明显的单 epoch 时间差。

2026-08-26 23:37 状态记录：Epoch 1 已完成，epoch train loss 为 `0.9305`，并已保存
`model_epoch_1_trainloss_0.9305.pth`；Epoch 2 进行到约 `26,254/38,350`（68.5%），
当前速度约 `10.4 batch/s`。按当前速度和已观测的约 53～55 分钟/epoch 估计，剩余 8.3
个 epoch 约需 `7小时25分钟`，实际时间会受本机 CPU/磁盘负载影响。

## 6. 训练完成与 fixed200 评测

训练于 2026-08-27 06:59 完成。各 epoch loss 为：

```text
Epoch 1:  0.9305
Epoch 2:  0.3324
Epoch 3:  0.2390
Epoch 4:  0.1978
Epoch 5:  0.1727
Epoch 6:  0.1553
Epoch 7:  0.1420
Epoch 8:  0.1316
Epoch 9:  0.1229
Epoch 10: 0.1156
```

Epoch10 checkpoint：

```text
HDP-nuplan/tmp/original_diffusion_retrain_306801_seed3407_epoch10/training_log/
original-diffusion-retrain-306801-epoch10/2026-08-26-22:02:52/
model_epoch_10_trainloss_0.1156.pth
```

fixed200 使用脚本：

```text
HDP-nuplan/scripts/evaluate_original_diffusion_same_data_fixed200.sh
```

评测严格复用 `mini-val-fixed-200-rl-v4`、同一 mini DB、
`closed_loop_nonreactive_agents`、2-thread worker 和相同 NuPlan metrics；推理加载本次
Epoch10 checkpoint 的 EMA 权重。输出目录为：

```text
HDP-nuplan/tmp/original_diffusion_retrain_306801_seed3407_epoch10/fixed200_eval/
```

2026-08-27 10:52 已完成启动前严格加载检查：Epoch10 EMA 的 `6,042,628` 个参数全部
匹配，无 missing/unexpected key。随后启动 fixed200，日志确认成功构建 `200` 个场景，
并以 2-thread worker 进入 `closed_loop_nonreactive_agents` 仿真。最终指标待 200 个场景
全部完成后自动汇总。

### 6.1 fixed200 最终结果

评测于 2026-08-27 12:57 完成：`200/200` 成功、`0` 失败。配对分析确认两者场景集合
完全相同，以下差值均为“同数据原始 Diffusion Planner Epoch10 - B Epoch10”：

| 指标 | B Epoch10 | 同数据原始 DP Epoch10 | 差值 |
|---|---:|---:|---:|
| 综合 score | 0.904567 | 0.580725 | -0.323842 |
| 无自车责任碰撞 | 0.957500 | 0.870000 | -0.087500 |
| 可行驶区域合规 | 0.950000 | 0.710000 | -0.240000 |
| 自车保持进展 | 0.990000 | 0.915000 | -0.075000 |
| 行驶方向合规 | 1.000000 | 0.950000 | -0.050000 |
| 专家路线进度 | 0.912068 | 0.727933 | -0.184134 |
| TTC 合规 | 0.935000 | 0.765000 | -0.170000 |
| 限速合规 | 0.995431 | 0.999370 | +0.003939 |
| 舒适性 | 0.995000 | 0.920000 | -0.075000 |

逐场景 score 配对为：同数据原始 DP 胜 `7`、负 `140`、平 `53`。因此，控制训练数据、
seed、batch size、训练轮数和 fixed200 场景后，本次从随机初始化训练的原始 Diffusion
Planner 明显低于 B Epoch10。唯一均值提升项为限速合规，但不足以抵消路线进度、
可行驶区域、TTC 与碰撞指标的下降。

### 6.2 关于 Encoder 冻结策略的更正

复核 B 的原始训练日志后确认：B 的配置包含 `freeze_encoder_epochs=3`，初始监督训练的
Epoch 1～3 输出 `Encoder trainable: False`，从 Epoch 4 开始输出
`Encoder trainable: True`。B Epoch10 因此经历过“前 3 轮冻结、之后解冻”的训练过程。

本次原始 Diffusion Planner 对照的训练入口没有该参数，Encoder 从 Epoch 1 起始终参与
反向传播。因此，两者虽然使用相同训练数据、seed、batch、epoch 数和评测场景，但冻结
策略并不相同；此前将两者表述为冻结策略一致是不准确的。该差异应作为结果解释中的
一个额外变量，不能把本次结果完全归因于模型结构或损失函数。

结果文件：

```text
HDP-nuplan/tmp/original_diffusion_retrain_306801_seed3407_epoch10/fixed200_eval/
b10_vs_original_same_data_fixed200_analysis.md
```

## 7. 按 B Epoch10 训练协议重做原始 Diffusion Planner

### 7.1 重新对齐的原因

复核后确认，上一轮原始 Diffusion Planner 对照是从随机初始化开始，Encoder 没有复现
B Epoch10 的 warm-start 和前 3 个 epoch 冻结策略。因此上一轮结果不能作为严格公平的
最终对照。本次重新训练只修正这一项，原始 Diffusion Planner 的模型结构和原生 loss 不变。

### 7.2 代码改动

在仓库根目录 `train_predictor.py` 中新增：

- `--encoder_pretrained_model_path`：只加载 checkpoint 中同名且同形状的 Encoder 参数；
- `--freeze_encoder_epochs`：按 epoch 控制 Encoder 是否参与反向传播；
- warm-start 审计报告和 `Encoder trainable` 日志。

启动前已通过独立检查确认：从 `checkpoints/model.pth` 的 `ema_state_dict` 加载
`151/151` 个 Encoder 张量，共 `1,799,040` 个参数，Decoder 加载数为 `0`。

### 7.3 与 B Epoch10 的对齐项

| 项目 | 本次原始 Diffusion Planner | B Epoch10 |
|---|---|---|
| 训练数据 | 完整 mini-train，306,801 NPZ | 相同 |
| seed | 3407 | 3407 |
| batch size | 8 | 8 |
| learning rate | 阶段1为 5e-4，阶段2为 5e-5 | 阶段1为 5e-4，阶段2为 5e-5 |
| warm-up | 阶段1为 2 epoch，阶段2为 1 epoch | 阶段1为 2 epoch，阶段2为 1 epoch |
| 数据增强 | 概率 0.5 | 概率 0.5 |
| num_workers | 0 | 0 |
| EMA | 启用 | 启用 |
| Encoder 初始化 | `checkpoints/model.pth` 的 EMA Encoder | 相同来源 |
| Encoder 冻结 | Epoch1～3 冻结，Epoch4 起解冻 | 相同 |
| 模型结构与 loss | 原始 Diffusion Planner 原生实现 | HDP 原生实现 |

最后一项不能完全相同，否则就不再是原始 Diffusion Planner 对照，而会改变被比较的模型。

### 7.4 当前运行

启动脚本：

```text
HDP-nuplan/scripts/train_original_diffusion_same_data.sh
```

脚本分为两个连续阶段：

1. 阶段1训练到 Epoch4，使用 `5e-4` 学习率和 `2` epoch warm-up；
2. 阶段2恢复 Epoch4 checkpoint，重置为 `5e-5` 学习率和 `1` epoch warm-up，训练到总 Epoch10。

这样与 B Epoch10 的“Epoch4 起点分叉”学习率过程一致；原始 Diffusion Planner 在两个阶段
均使用自己的原生 loss。

输出目录：

```text
HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/
```

2026-08-27 首轮验收通过，日志已确认：

```text
Dataset Prepared: 306801 train data
Encoder warm-start: loaded=151/151 tensors, parameters=1799040, decoder_loaded=0
Epoch 1/10
Encoder trainable: False
```

训练已进入阶段1 Epoch1 正式计算，当前尚未完成 Epoch4 checkpoint、阶段2训练和 fixed200 评测。

### 7.5 Epoch5 暂停与恢复

阶段1已完成 Epoch1～4，训练损失依次为：

```text
Epoch1: 0.2311
Epoch2: 0.1595
Epoch3: 0.1002
Epoch4: 0.1295
```

阶段2完成 Epoch5 后按用户要求发送 `SIGTERM` 暂停。Epoch5 checkpoint 已完整保存，
元数据为 `epoch=5`、`loss=0.0831212923`。暂停后日志中出现的
`SignalException: signal 15` 是主动暂停的预期结果，不是训练故障。

2026-08-27 已从同一目录的 `latest.pth` 恢复训练。恢复验收结果：

```text
Model load done
Optimizer load done
Schedule load done
Step load done
ema load done
Learning-rate schedule reset after resume: lr=5e-05, warm_up_epoch=1
Epoch 6/10
Encoder trainable: True
```

恢复日志：

```text
HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/
phase2_resume_from_epoch5.log
```

### 7.6 Epoch10 中途暂停

恢复后已完整完成 Epoch6～9，对应训练损失为：

```text
Epoch6: 0.0718
Epoch7: 0.0678
Epoch8: 0.0654
Epoch9: 0.0635
```

2026-08-28 按用户要求在 Epoch10 约 5% 处立即暂停。Epoch10 尚未完成，因此该轮部分
进度不写入 checkpoint；当前 `latest.pth` 已核验为完整的 Epoch9 checkpoint：

```text
epoch=9
loss=0.06346216797828674
```

训练进程已全部退出，GPU 占用回落到空闲状态。后续恢复时将从 Epoch9 checkpoint
重新开始完整的 Epoch10。

### 7.7 恢复 Epoch10 并自动启动 fixed200 评测

2026-08-28 按用户要求恢复最后一轮，并将训练与评测串为同一后台流水线。新增脚本：

```text
HDP-nuplan/scripts/resume_original_diffusion_aligned_epoch10_and_eval.sh
```

流水线依次执行：

1. 从已核验的 Epoch9 `latest.pth` 恢复模型、优化器、调度器和 EMA；
2. 使用 `5e-5`、`warm_up_epoch=1` 重新训练完整 Epoch10；
3. 检查新 checkpoint 的元数据必须为 `epoch=10`；
4. 使用 `mini-val-fixed-200-rl-v4` 场景过滤器自动启动闭环评测；
5. 汇总指标，并与同一 fixed200 上的 B Epoch10 做逐场景配对分析。

为防止评测脚本误用旧模型，`evaluate_original_diffusion_same_data_fixed200.sh` 已支持通过
环境变量显式传入 `TRAIN_RUN`、`DP_ARGS`、`DP_CHECKPOINT`、`EVAL_OUT_ROOT` 和
`EXP_UID`。本次使用独立输出目录：

```text
HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/fixed200_eval/
```

恢复启动验收通过：

```text
Model/Optimizer/Schedule/EMA load done
Learning-rate schedule reset after resume: lr=5e-05, warm_up_epoch=1
Epoch 10/10
Encoder trainable: True
```

### 7.8 Epoch10 与 fixed200 最终结果

Epoch10 于 2026-08-28 04:57 完成，训练损失为 `0.0618`。checkpoint 已核验为
`epoch=10`，随后流水线自动执行 fixed200 闭环评测。评测耗时 `01:56:16`，结果为
`200/200` 成功、`0` 失败，且与 B Epoch10 的场景集合完全一致。

| 指标 | B Epoch10 | 对齐协议原始 DP Epoch10 | 原始 DP - B |
|---|---:|---:|---:|
| 综合 score | 0.904567 | 0.795445 | -0.109121 |
| 无自车责任碰撞 | 0.957500 | 0.857500 | -0.100000 |
| 可行驶区域合规 | 0.950000 | 0.915000 | -0.035000 |
| 自车保持进展 | 0.990000 | 0.995000 | +0.005000 |
| 行驶方向合规 | 1.000000 | 0.997500 | -0.002500 |
| 专家路线进度 | 0.912068 | 0.923151 | +0.011084 |
| TTC 合规 | 0.935000 | 0.840000 | -0.095000 |
| 限速合规 | 0.995431 | 0.995299 | -0.000132 |
| 舒适性 | 0.995000 | 0.915000 | -0.080000 |

逐场景综合 score 为原始 DP 胜 `63`、负 `64`、平 `73`。对齐 Encoder warm-start、
冻结和两阶段学习率后，原始 DP 的综合 score 相比此前随机初始化对照由 `0.580725`
提升至 `0.795445`（`+0.214721`），路线进度由 `0.727933` 提升至 `0.923151`
（`+0.195218`）。这证明此前训练协议差异是重要影响因素。

但对齐协议后，原始 DP 综合 score 仍低于 B Epoch10，主要差距来自碰撞、TTC 和舒适性；
原始 DP 的路线进度和保持进展则略高于 B。最终配对分析文件：

```text
HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/
fixed200_eval/b10_vs_original_same_data_fixed200_analysis.md
```
