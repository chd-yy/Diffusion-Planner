# HDP B Epoch10 共同阶段 Epoch1～4：`model_training()` 全过程

## 1. 文档范围

本文只解释 **HDP B Epoch10 的共同阶段 Epoch1～4**，并严格按照当前
`HDP-nuplan/train_predictor.py` 从第 457 行开始的实际执行顺序展开。

B Epoch10 不是从随机参数独立训练 10 个 epoch。它的训练关系是：

```text
原始 Diffusion Planner checkpoint
        │ 只加载兼容的 Encoder EMA 参数
        ▼
共同阶段 Epoch1～3：Encoder 冻结，Decoder 训练
        ▼
共同阶段 Epoch4：Encoder 解冻，Encoder 与 Decoder 联合训练
        ▼
固定 Epoch4 完整 checkpoint
        ├── Experiment A
        └── Experiment B → 最终的 HDP B Epoch10
```

因此，Epoch1～4 同时属于后来的 A、B 两个分支；本文不介绍分叉后的 Epoch5～10。

主要代码：

- 训练入口：`HDP-nuplan/train_predictor.py`
- 单 epoch 训练：`HDP-nuplan/hdp_nuplan/train_epoch.py`
- 监督损失：`HDP-nuplan/hdp_nuplan/loss.py`
- checkpoint 工具：`HDP-nuplan/hdp_nuplan/utils/train_utils.py`
- 初次启动脚本：`HDP-nuplan/scripts/finalize_full_mini_and_train.sh`
- 中断恢复脚本：`HDP-nuplan/scripts/resume_full_mini_training_numworkers0.sh`

## 2. Epoch1～4 的实际配置

共同阶段最初由 `finalize_full_mini_and_train.sh` 启动。训练参数中最关键的部分是：

```bash
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 1 \
  --standalone \
  HDP-nuplan/train_predictor.py \
  --name hdp-paper-supervised-full-mini-omega001-nodetach \
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

配置含义如下：

| 配置 | 共同阶段实际含义 |
|---|---|
| 训练样本 | 完整 mini-train，共 306,801 个 NPZ |
| DDP | 启用，但 `world_size=1`，即单进程、单 GPU |
| 全局/单卡 batch size | 都是 8 |
| 每个完整 epoch 的 batch 数 | `floor(306801 / 8) = 38,350` |
| 数据增强 | 默认开启，`augment_prob=0.5` |
| Encoder 初始化 | 从 `checkpoints/model.pth` 只加载兼容 Encoder 参数 |
| Encoder 冻结 | Epoch1～3 冻结，Epoch4 开始解冻 |
| 扩散输出和监督类型 | 都是默认的 `x_start` |
| 混合位置损失权重 | `0.01` |
| detach | `planning_detach_window_size=0`，不截断积分梯度 |
| EMA | 默认开启，衰减率 `0.999` |
| 命令声明的基础学习率 | `5e-4` |
| Epoch1～4 实际学习率 | 均为 `5e-5`，原因见第 12 节 |
| 随机种子 | 3407 |

`train_epochs=20` 是初始进程计划训练到的终点。后来在 Epoch4 固定 checkpoint 并建立 A/B 分支，所以它不表示 B Epoch10 已在共同进程中训练 20 轮。

## 3. 第 457 行：解析 `args`

程序直接运行 `train_predictor.py` 时，首先执行：

```python
args = get_args()
```

`get_args()` 使用 `argparse` 完成三件事：

1. 定义训练入口支持的全部命令行参数及默认值；
2. 通过 `parser.parse_args()` 把启动命令解析成 `argparse.Namespace`；
3. 根据 `normalization.json` 构造 `state_normalizer` 和 `observation_normalizer`，并加入 `args`。

参数来源的优先级是：

```text
启动命令显式传入的值
        >
get_args() 中对应参数的默认值
```

例如，命令显式传入的：

```bash
--batch_size 8
```

会得到：

```python
args.batch_size == 8
```

命令没有传入 `--future_len`，因此使用代码默认值：

```python
args.future_len == 80
```

随后读取 `normalization.json`，增加：

```python
args.state_normalizer
args.observation_normalizer
```

这里的 `args.json` 不是参数来源。它是进入 `model_training()` 后，把已经解析完成的 `args` 保存出来的运行快照。

## 4. 第 461 行：进入 `model_training(args)`

入口随后执行：

```python
model_training(args)
```

它负责整个训练生命周期：参数检查、DDP 初始化、数据集构造、模型初始化、Encoder warm-start、优化器和 EMA 创建、epoch 循环、日志记录与 checkpoint 保存。

整体顺序为：

```text
参数合法性检查
→ 初始化单进程 DDP
→ 建立输出目录并保存 args.json
→ 固定随机种子
→ 创建 Dataset / Sampler / DataLoader
→ 创建 HDP 模型
→ 加载原始 DP Encoder
→ 模型迁移到 GPU 并包装 DDP
→ 创建 EMA、AdamW、学习率调度器
→ 决定从新实验还是 checkpoint 恢复
→ 设置 sampler 和 logger
→ 进入 Epoch1～4
→ 每轮训练、记录、保存 checkpoint
```

## 5. 参数合法性检查

`model_training()` 首先检查：

- `resume_model_path` 与 `encoder_pretrained_model_path` 不能同时提供；
- 只有恢复训练时才能使用 `reset_lr_schedule_on_resume`；
- `freeze_encoder_epochs` 不能为负数；
- 如果要求冻结 Encoder，必须有 Encoder warm-start 或完整 checkpoint 作为权重来源。

共同阶段第一次启动时：

```text
encoder_pretrained_model_path = checkpoints/model.pth
resume_model_path             = None
freeze_encoder_epochs         = 3
```

所以第一次运行会“只加载 Encoder 后从 Epoch1 开始”。发生中断后则改为：

```text
encoder_pretrained_model_path = None
resume_model_path             = 共同训练输出目录
freeze_encoder_epochs         = 3
```

恢复时直接加载完整 HDP checkpoint，不会再次加载原始 DP Encoder。

## 6. 初始化 DDP 和输出目录

### 6.1 DDP

启动命令使用 `torch.distributed.run`，但配置是：

```text
nnodes=1
nproc-per-node=1
```

因此 `ddp_setup_universal()` 得到：

```text
global_rank = 0
local rank  = 0
world_size  = 1
```

程序仍会初始化 NCCL 进程组并包装 `DistributedDataParallel`，但没有第二张 GPU 与它同步。这里使用 DDP 主要是保持训练入口一致。

### 6.2 输出目录和 `args.json`

首次训练没有 `resume_model_path`，因此创建带时间戳的新目录：

```text
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/
supervised_training_full_mini_omega001_nodetach/training_log/
hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/
```

rank 0 将 `args` 转成普通字典，把两个 normalizer 转成可序列化字典，再写入 `args.json`。

恢复训练时：

```python
save_path = args.resume_model_path
```

因此继续使用原目录，并覆盖该目录中的 `args.json`。当前目录里的 `args.json` 后来又被 A 分支续训参数覆盖，不能单独作为 Epoch1 初次启动参数的证据；初次启动参数应以启动脚本、日志和 `encoder_warm_start_report.json` 联合还原。

## 7. 固定随机种子

程序执行：

```python
set_seed(args.seed + global_rank)
```

本次只有 rank 0，所以有效种子仍为 3407。函数同时设置 Python、NumPy 和 PyTorch 随机种子，并启用确定性 cuDNN、关闭 benchmark。

随机种子主要影响：

- `DistributedSampler` 的样本打乱顺序；
- `StatePerturbation` 的随机增强；
- 每个样本的扩散时间 `t`；
- 加入轨迹的高斯噪声 `z`；
- DropPath 等模型训练随机性。

## 8. 构造训练数据

### 8.1 数据增强器

`use_data_augment` 未在命令中关闭，使用默认值 `true`，因此创建：

```python
StatePerturbation(augment_prob=0.5, device="cuda")
```

它以给定概率扰动当前自车状态，并同步调整自车未来、邻车未来及场景坐标，保证增强后的输入和监督标签仍对应同一场景坐标系。

### 8.2 Dataset

`DiffusionPlannerData` 读取 manifest 中的 306,801 个 NPZ 文件名。一个原始样本的数据形状为：

| 字段 | 单样本形状 |
|---|---:|
| `ego_current_state` | `[10]` |
| `ego_agent_future` | `[80, 3]` |
| `neighbor_agents_past` | `[32, 21, 11]` |
| `neighbor_agents_future` | 原 NPZ 为 `[32, 80, 3]`，Dataset 截取前 10 辆后为 `[10, 80, 3]` |
| `static_objects` | `[5, 10]` |
| `lanes` | `[70, 20, 12]` |
| `lanes_speed_limit` | `[70, 1]` |
| `lanes_has_speed_limit` | `[70, 1]` |
| `route_lanes` | `[25, 20, 12]` |
| `route_lanes_speed_limit` | `[25, 1]` |
| `route_lanes_has_speed_limit` | `[25, 1]` |

HDP 不训练邻车未来预测；邻车未来在这里保留是为了兼容 Dataset、数据增强和 loss 调用接口。

### 8.3 Sampler 和 DataLoader

训练使用：

```python
DistributedSampler(..., world_size=1, rank=0, shuffle=True)
DataLoader(..., batch_size=8, pin_memory=True, drop_last=True)
```

所以每个完整 epoch 包含：

```text
306,801 // 8 = 38,350 batch
38,350 × 8 = 306,800 个样本槽位
```

每轮打乱后会有 1 个尾部样本因 `drop_last=True` 被丢弃；下一轮重新打乱，被丢弃的不一定还是同一个样本。

初次运行 `num_workers=4`。后续为避免内存问题，恢复运行改成 `num_workers=0`。这只改变 NPZ 数据加载方式和速度，不改变数据集合、batch size、损失函数或模型结构。

## 9. 创建模型并加载 Encoder

### 9.1 创建 HDP

程序实例化：

```python
diffusion_planner = Hyper_Diffusion_Planner(args)
```

模型总参数量为 5,092,996，主要数据流是：

```text
场景观测
→ HDP Encoder 生成场景 token
→ HDP Decoder 接收带噪自车运动序列、扩散时间、场景编码、route 和自车速度
→ 输出去噪目标
```

### 9.2 Encoder warm-start

初次启动提供：

```text
checkpoints/model.pth
```

`load_encoder_warm_start()` 按以下优先级选择源权重：

```text
ema_state_dict > model > checkpoint 根字典
```

本次从原始 checkpoint 的 `ema_state_dict` 中，只加载名称以 `encoder.` 开头、名称相同且 shape 相同的张量。实际结果为：

| 项目 | 结果 |
|---|---:|
| Encoder 目标张量 | 151 |
| 成功加载张量 | 151 |
| 加载参数量 | 1,799,040 |
| 缺失/shape 不匹配 | 0 |
| Decoder 加载张量 | 0 |

因此：

- Encoder 继承原始 Diffusion Planner 的 EMA 参数；
- HDP Decoder 没有继承原始 DP Decoder；
- HDP Decoder 使用自己的初始化，并从 Epoch1 开始训练。

加载报告保存为 `encoder_warm_start_report.json`。

### 9.3 GPU 和 DDP 包装

模型迁移到 GPU 0，然后包装为 DDP：

```python
DDP(model, device_ids=[0], find_unused_parameters=True)
```

`find_unused_parameters=True` 用于兼容 lane 限速 embedding 的数据依赖分支：某些 batch 可能不会使用其中一部分参数。

## 10. 创建 EMA、优化器和学习率调度器

### 10.1 EMA

程序创建模型的指数滑动平均副本：

```python
ModelEma(diffusion_planner, decay=0.999, device="cuda")
```

每完成一个 batch 的 AdamW 更新后，都会再更新一次 EMA：

```text
EMA_new = 0.999 × EMA_old + 0.001 × 当前模型参数
```

训练 checkpoint 同时保存即时 `model` 和 `ema_state_dict`；后续闭环推理优先使用 EMA 权重。

### 10.2 AdamW

所有模型参数都先加入同一个 AdamW 参数组，组学习率声明为 `5e-4`：

```python
optimizer = AdamW(model.parameters(), lr=5e-4)
```

Encoder 冻结不是把它从 optimizer 删除，而是把 Encoder 参数设置成 `requires_grad=False`。冻结时这些参数没有梯度，因此 AdamW 不会更新它们。

### 10.3 学习率调度器

代码调用：

```python
CosineAnnealingWarmUpRestarts(optimizer, train_epochs=20, warm_up_epoch=2)
```

虽然函数名称包含 `CosineAnnealing`，当前实现实际是：

```text
LinearLR warm-up
→ 固定学习率 MultiplicativeLR
```

没有执行余弦下降。`LinearLR(start_factor=0.1)` 在调度器创建时把当前学习率从 `5e-4` 设为：

```text
5e-4 × 0.1 = 5e-5
```

因此 Epoch1 从 `5e-5` 开始。

## 11. 新实验与恢复实验的两条路径

### 11.1 Epoch1：新实验路径

第一次启动时 `resume_model_path=None`，因此：

```python
init_epoch = 0
wandb_id = None
```

循环从程序索引 `epoch=0` 开始，即人类所说的 Epoch1。

### 11.2 Epoch2～4：完整恢复路径

共同阶段发生过中断，恢复命令使用输出目录作为 `resume_model_path`。`resume_model()` 实际读取：

```text
<resume_model_path>/latest.pth
```

并恢复：

- HDP 当前模型参数；
- AdamW 参数和一阶、二阶动量；
- scheduler 状态；
- checkpoint 中的 epoch，作为新的 `init_epoch`；
- 日志运行 ID；
- EMA 参数。

例如从 Epoch3 checkpoint 恢复时：

```text
checkpoint["epoch"] = 3
init_epoch          = 3
```

于是 `range(init_epoch, train_epochs)` 从索引 3 开始，也就是重跑/继续人类所说的 Epoch4。

`DistributedSampler` 的 epoch 不在 checkpoint 中，所以恢复后额外执行：

```python
train_sampler.set_epoch(init_epoch)
```

使恢复后的样本打乱序号与即将训练的 epoch 对齐。

未完成 epoch 中产生的更新不会保留到最终链路：恢复时会重新加载上一个完整 checkpoint 的模型、optimizer、scheduler 和 EMA。

## 12. 为什么共同阶段 Epoch1～4 实际都是 `5e-5`

这是理解 B Epoch10 训练史最重要的一点。

每个 epoch 末尾的代码顺序是：

```text
完成 train_epoch
→ 保存 checkpoint
→ scheduler.step()
```

因此 checkpoint 中保存的是本轮训练所使用、但尚未执行本轮末 `scheduler.step()` 的状态。

实际过程为：

```text
Epoch1 以 5e-5 完整训练并保存 checkpoint
→ scheduler.step() 后原进程开始后续训练
→ 后续 epoch 中途被终止
→ 从上一个完整 checkpoint 恢复
→ 恢复到 checkpoint 保存的 5e-5 和 step 前 scheduler 状态
→ 下一完整 epoch 又以 5e-5 训练
```

这个过程在 Epoch2、Epoch3、Epoch4 前后重复发生。因此，虽然启动参数写的是基础学习率 `5e-4`，四个最终有效 checkpoint 中保存的 optimizer 学习率均为 `5e-5`：

| Epoch | checkpoint 中的 LR | Encoder | Train loss |
|---:|---:|---|---:|
| 1 | `5e-5` | 冻结 | `0.1196681112` |
| 2 | `5e-5` | 冻结 | `0.0495298207` |
| 3 | `5e-5` | 冻结 | `0.0348934792` |
| 4 | `5e-5` | 解冻 | `0.0264890622` |

所以准确表述应为：

> 共同阶段声明了 `learning_rate=5e-4` 和 `warm_up_epoch=2`，但受中断恢复以及“先保存、后 scheduler.step()”顺序影响，最终进入 B 权重链路的 Epoch1～4 都是在 `5e-5` 下完成的。

## 13. Epoch1～4 的冻结和训练范围

每轮开始执行：

```python
encoder_trainable = epoch >= freeze_encoder_epochs
set_encoder_trainable(base_model, encoder_trainable)
```

`freeze_encoder_epochs=3`，而 `epoch` 从 0 开始，因此：

| 人类 epoch | 程序索引 | Encoder | Decoder |
|---:|---:|---|---|
| 1 | 0 | 冻结 | 训练 |
| 2 | 1 | 冻结 | 训练 |
| 3 | 2 | 冻结 | 训练 |
| 4 | 3 | 训练 | 训练 |

“冻结 Encoder”仅表示 `requires_grad=False`，并不把 Encoder 切换成 `eval()`。整个模型在 `train_epoch()` 开头执行 `model.train()`，因此 Encoder 的训练模式行为仍然开启，只是参数不接受梯度更新。

## 14. 一个 epoch 内的全过程

`model_training()` 每轮调用：

```python
train_loss, train_total_loss = train_epoch(...)
```

每个完整 epoch 遍历 38,350 个 batch。一个 batch 的执行过程如下。

### 14.1 整理输入并迁移到 GPU

DataLoader 把 8 个样本堆叠成 batch。`train_epoch()` 从 tuple 中取出：

- 当前自车状态；
- 自车未来真值；
- 邻车历史和未来；
- lane、route lane 及其限速信息；
- 静态物体。

模型观测被整理成 `inputs` 字典并移动到 CUDA；自车和邻车未来标签单独保留。

### 14.2 数据增强

如果当前 batch 中的样本命中 `augment_prob=0.5`，`StatePerturbation` 会同时改变场景输入与未来标签。增强不是只改自车当前位置而保留原标签，否则输入和标签会不一致。

### 14.3 航向角编码和邻车 mask

自车未来从：

```text
[x, y, heading]，形状 [8,80,3]
```

转换为：

```text
[x, y, cos(heading), sin(heading)]，形状 [8,80,4]
```

这样避免角度在 `-π/π` 边界不连续。

邻车未来也做相同转换。原始 x、y、heading 全零的位置被视为 padding，并在转换后重新整体置零。HDP 不用它计算邻车预测损失，但保留 mask 以兼容接口。

### 14.4 观测归一化

`ObservationNormalizer` 对当前自车、邻车历史、lane、route lane、限速和静态物体等输入按 `normalization.json` 归一化。全零 padding 在归一化后重新置零，避免 padding 被减均值后变成非零有效数据。

### 14.5 清空梯度并构造扩散监督

每个 batch 先执行：

```python
optimizer.zero_grad()
```

然后 `diffusion_loss_func()` 将自车未来位置转换成 HDP 的扩散目标：

1. 对 x/y 沿时间做差分，第一个点相对于自车坐标系原点；
2. `cos/sin` 不做差分，保留每个未来时刻的方向；
3. 得到 `[Δx, Δy, cos, sin]`，形状 `[8,80,4]`；
4. 使用 `StateNormalizer` 归一化，得到干净目标 `all_gt`；
5. 每个场景随机采样扩散时间 `t`，形状 `[8]`；
6. 采样高斯噪声 `z`，形状 `[8,80,4]`；
7. 使用 VP-SDE 构造带噪状态 `x_t = mean + std × z`。

### 14.6 HDP 前向传播

带噪轨迹和扩散时间加入 `inputs`：

```python
inputs["sampled_trajectories"] = x_t
inputs["diffusion_time"] = t
```

模型前向包括：

```text
Encoder：邻车历史 + 静态物体 + lane
       → 场景上下文 encoding

Decoder：x_t + t + encoding + route_lanes + 当前自车速度
       → model_pred，形状 [8,80,4]
```

本实验的：

```text
diffusion_model_type       = x_start
diffusion_supervision_type = x_start
```

所以模型直接学习预测去噪后的干净运动增量表示。

### 14.7 两项监督损失

第一项是运动增量空间中的扩散损失：

```text
ego_planning_loss
= mean(sum((model_pred - all_gt)², state_dim))
```

先对末维 `[Δx, Δy, cos, sin]` 求和，再对 batch 和未来时间求平均。

第二项是积分后位置轨迹损失：

1. 把预测从归一化空间恢复到真实尺度；
2. 取预测的 `Δx, Δy`；
3. 沿 80 个未来时间点做累计和，恢复 x/y 位置轨迹；
4. 与自车未来位置真值计算均方误差。

```text
ego_planning_hybrid_loss
= mean(sum((predicted_xy - ground_truth_xy)², xy_dim))
```

`planning_detach_window_size=0`，所以这里使用普通累计和，位置误差可以向此前全部位移预测传播梯度。

最终总损失为：

```text
loss = ego_planning_loss
     + 0.01 × ego_planning_hybrid_loss
```

### 14.8 反向传播和参数更新

每个 batch 依次执行：

```text
loss.backward()
→ DDP 梯度同步（本实验 world_size=1）
→ 全模型梯度范数裁剪到 5
→ optimizer.step()
→ ema.update(model)
```

在 Epoch1～3：

- Decoder 有梯度并由 AdamW 更新；
- Encoder 没有梯度，不被 AdamW 更新；
- EMA 在每个 batch 后更新。

在 Epoch4：

- Encoder 和 Decoder 都有梯度；
- 两部分都由 AdamW 更新；
- EMA 同样在每个 batch 后更新。

### 14.9 汇总 epoch loss

每个 batch 的三项标量都会脱离计算图后保存：

```text
ego_planning_loss
ego_planning_hybrid_loss
loss
```

epoch 结束后分别计算 38,350 个 batch 的平均值。DDP 再执行跨 rank 汇总；本实验只有一个 rank，所以数值不变。

## 15. 一个 epoch 完成后的操作

`train_epoch()` 返回后，rank 0 执行：

1. 记录各项平均 loss；
2. 记录当前 optimizer 学习率；
3. 因 `save_utd=1`，每个 epoch 都保存 checkpoint；
4. checkpoint 保存完成后调用 `scheduler.step()`；
5. 执行 `train_sampler.set_epoch(epoch + 1)`，为下一轮生成新的确定性打乱顺序。

checkpoint 文件名示例：

```text
model_epoch_4_trainloss_0.0265.pth
```

同时更新：

```text
latest.pth
```

每个 checkpoint 包含：

```text
epoch
model
ema_state_dict
optimizer
schedule
loss
wandb_id
```

它是“完整恢复点”，不只是模型权重文件。

## 16. Epoch1～4 的实际结果和分叉点

共同阶段最终有效结果为：

| Epoch | Encoder 状态 | 完整 batch 数 | 实际 LR | Train loss |
|---:|---|---:|---:|---:|
| 1 | 冻结 | 38,350 | `5e-5` | `0.1196681112` |
| 2 | 冻结 | 38,350 | `5e-5` | `0.0495298207` |
| 3 | 冻结 | 38,350 | `5e-5` | `0.0348934792` |
| 4 | 解冻 | 38,350 | `5e-5` | `0.0264890622` |

四轮合计执行：

```text
4 × 38,350 = 153,400 次 optimizer.step()
```

Epoch4 完整 checkpoint 随后被固定复制为 B 分支的共同起点。它包含当时的 Encoder、Decoder、EMA、AdamW 动量和 scheduler 状态，因此 B 分支不是重新初始化模型。

## 17. 最容易混淆的五点

1. **`args.json` 是输出快照，不是 `get_args()` 的输入。** 参数首先来自命令行和代码默认值。
2. **B Epoch10 的 Epoch1～4 与 A 分支完全相同。** A/B 差异从 Epoch4 checkpoint 之后才开始。
3. **共同阶段不是从零训练全部模型。** Encoder 来自原始 DP 的 EMA 权重；Decoder 没有迁移。
4. **冻结 Encoder 不等于 Encoder 不参与前向传播。** 它继续产生场景编码，只是不更新参数。
5. **命令中的 `5e-4` 不等于四轮实际 LR。** 四个有效 checkpoint 证明 Epoch1～4 最终都以 `5e-5` 完成。

## 18. 证据位置

| 内容 | 路径 |
|---|---|
| 初次启动参数 | `HDP-nuplan/scripts/finalize_full_mini_and_train.sh` |
| 恢复参数 | `HDP-nuplan/scripts/resume_full_mini_training_numworkers0.sh` |
| Epoch1 初始日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_full_mini_omega001_nodetach_train.log` |
| Epoch2～4 恢复日志 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_full_mini_omega001_nodetach_resume*.log` |
| Encoder 加载报告 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/encoder_warm_start_report.json` |
| Epoch1～4 checkpoint | 同一训练输出目录下的 `model_epoch_1...pth` 至 `model_epoch_4...pth` |
| B 分支固定的 Epoch4 源 | `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/source_model_epoch_4_trainloss_0.0265.pth` |
