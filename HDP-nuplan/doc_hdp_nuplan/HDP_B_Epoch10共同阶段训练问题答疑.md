# HDP B Epoch10 共同阶段训练问题答疑

本文回答 HDP B Epoch10 共同阶段 Epoch1～4 中与 batch、增强、归一化、随机种子、NPZ、数据加载、优化器、学习率调度和恢复训练有关的 11 个问题。

相关主文档：`HDP-nuplan/doc_hdp_nuplan/HDP_B_Epoch10共同阶段_Epoch1至4_model_training全过程.md`。

## 1. 全局/单卡 batch size 为什么设置为 8？

### 直接原因

这不是模型理论规定的数值，而是当时根据本地训练硬件作出的保守工程选择。项目操作日志明确记录：本地 RTX 4060 只有 8 GiB 显存，因此监督训练从全局 batch size 8 开始。

HDP 每个样本不只包含 `[80,4]` 的自车扩散轨迹，还包含：

- 32 辆邻车的 21 帧历史；
- 70 条 lane，每条 20 个点；
- 25 条 route lane，每条 20 个点；
- Encoder/Decoder 的中间 token、注意力激活；
- 反向传播需要保留的计算图；
- AdamW 的一阶和二阶动量；
- EMA 模型副本。

这些内容共同占用显存，所以当时先选择 8，降低 OOM 风险。没有证据表明 8 是经过系统超参数搜索得到的最优 batch size。

### 本次为什么“全局”和“单卡”都是 8？

代码把命令中的 `args.batch_size` 当作全局 batch size，然后计算：

```python
per_rank_batch_size = args.batch_size // world_size
```

共同阶段采用：

```text
world_size = 1
global batch size = 8
per-rank batch size = 8 // 1 = 8
```

因此每次 `optimizer.step()` 使用 8 个场景，没有额外的梯度累积。

如果用 4 个 DDP 进程且仍传 `batch_size=8`，每张卡只会得到 2 个样本，而不是每张卡 8 个。

### batch size 8 的影响

优点：显存需求较低，可以稳定运行。缺点：梯度噪声通常比大 batch 更大，而且 306,801 个样本每轮会产生 38,350 次更新，训练时间较长。

## 2. 为什么默认开启数据增强并设置 `augment_prob=0.5`？

### 目的

离线 NuPlan 数据主要来自人类驾驶轨迹，模型看到的大多是“车辆已经处于正常轨迹附近”的状态。但闭环推理时，模型前一次预测产生的小误差会使车辆偏离训练分布；如果模型从未见过偏移状态，误差可能继续累积。

`StatePerturbation` 的目的就是人为构造轻微偏离状态，并给它重新生成可行的恢复轨迹，使模型学习：

```text
正常状态 → 正常规划
轻微横向/航向/速度偏差 → 回到合理轨迹
```

这主要用于减轻模仿学习中的闭环分布偏移，提高对轻微定位、控制和历史预测误差的恢复能力。

设置为 0.5 的直观意图是保留一部分原始专家样本，同时让另一部分样本学习偏离恢复，避免训练数据全部被扰动。

### 当前代码中的实际概率

增强标志实际写成：

```python
aug_flag = (torch.rand(B) >= self._augment_prob) \
           & ~(abs(ego_current_state[:, 4]) < 2.0)
```

需要准确理解为：

1. 只有纵向速度绝对值不低于 2 m/s 的样本才允许增强；
2. 随机条件使用的是 `rand >= augment_prob`，所以一般情况下实际随机命中率是 `1-augment_prob`；
3. 本次 `augment_prob=0.5`，因此符合速度条件的样本仍恰好有约 50% 被增强；
4. 慢速和静止样本不增强，所以全数据集最终增强比例低于 50%。

变量名表达的是“增强概率”，但比较符号按通常写法更应该是 `rand < augment_prob`。本次参数恰好为 0.5，因此这处方向问题不改变本实验的 50% 随机命中率；如果以后改成 0.2 或 0.8，就必须注意实际概率会相反。

### 为什么不是全部增强？

原始无扰动数据仍然是专家分布的主体。全部增强可能让训练分布偏离真实数据，并削弱模型对标准状态的拟合，因此使用“原始样本 + 扰动样本”的混合更稳妥。

## 3. `HDP-nuplan/normalization.json` 是事先定好的吗？为什么这样设置？

### 是否事先确定

是。B Epoch10 训练启动前，这个 JSON 已经作为静态配置文件存在。训练过程中不会用 306,801 个 NPZ 重新统计均值和标准差，也不会自动更新该文件。

Git 历史显示，它最初由原 Diffusion Planner 的 `normalization.json` 复制而来，再针对 HDP 的目标表示进行修改：

```text
原 DP ego：位置轨迹 [x,y,cos,sin]
mean = [10,0,0,0]
std  = [20,20,1,1]

HDP ego：逐帧运动增量 [Δx,Δy,cos,sin]
mean = [0,0,0,0]
std  = [0.5,0.5,1,1]
```

其余观测项大体沿用原 Diffusion Planner 的固定尺度；`ego_current_state` 后来按 HDP 实际 10 维输入完成了维度对齐。

### 为什么要归一化

输入中同时存在米、米每秒、方向余弦、类别标志等不同量纲。如果把数值范围相差很大的特征直接送入网络，数值较大的特征容易主导梯度。归一化把它们缩放到相近数量级，使优化更稳定。

例如：

- 空间位置、速度等连续量通常用 20 左右的尺度缩放；
- `cos/sin` 本来就在 `[-1,1]`，所以标准差使用 1；
- lane/route 限速用 20 缩放；
- HDP 的相邻 0.1 秒位移通常远小于完整 8 秒位置，所以将 x/y 增量尺度改成 0.5。

`0.5 m / 0.1 s` 对应约 `5 m/s`，可把常见逐帧位移缩放到接近 1 的量级。这是按数据语义和典型数值范围设置的工程尺度，不是对当前 306,801 个样本计算得到的严格统计标准差。

### 哪些部分会使用它

`StateNormalizer` 读取 `ego`，用于归一化扩散监督目标 `[Δx,Δy,cos,sin]`。

`ObservationNormalizer` 读取 `ego_current_state`、`neighbor_agents_past`、`lanes`、`route_lanes`、限速和静态物体等配置，用于归一化模型观测。

JSON 中的 `neighbor` 是从原 DP 联合预测接口保留的兼容配置；当前 HDP 监督损失只构造自车目标，不使用它构造邻车扩散监督。

### 风险

由于这些不是根据完整 mini-train 重新拟合的统计量，它们只能保证大致数值尺度合理，不能保证零均值和单位方差。若以后改数据分布、采样间隔或状态定义，应重新统计并做对照实验，而不能无条件沿用。

## 4. rank 0、随机种子和确定性 cuDNN 是什么意思？

### 为什么有效种子仍是 3407

代码执行：

```python
set_seed(args.seed + global_rank)
```

共同阶段只有一个进程：

```text
args.seed = 3407
global_rank = 0
有效种子 = 3407 + 0 = 3407
```

多卡时各 rank 会使用 3407、3408、3409……，避免所有 GPU 产生完全相同的随机增强和扩散噪声。

### 设置三个随机种子是什么意思

程序中存在多个独立随机数生成器：

- Python `random`：普通 Python 随机操作；
- NumPy：数组随机操作；
- PyTorch：模型初始化、`torch.rand`、`torch.randn`、扩散噪声等。

只设置其中一个不能控制其他随机源，所以三者都设为同一个起点。相同代码、相同数据顺序和相同环境下，随机数序列会尽量重复。

### 确定性 cuDNN 和关闭 benchmark

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

含义是：

- `deterministic=True`：有多个 CUDA/cuDNN 实现可选时，尽量选择结果可重复的算法；
- `benchmark=False`：不在运行时反复测速并自动选择最快卷积算法，避免算法选择随运行变化。

代价是某些算子可能变慢。它也不等于绝对可复现：CUDA 版本、GPU 型号、并行调度、部分非确定性算子和依赖版本变化仍可能带来差异。

### 恢复训练的额外注意点

每次重启都会再次执行 `set_seed(3407)`。`train_sampler.set_epoch(init_epoch)` 能让数据索引顺序对齐到恢复 epoch，但 Python/NumPy/PyTorch 的完整随机状态没有保存在 checkpoint 中。因此数据增强和扩散噪声序列不一定与“一次不中断连续训练”完全相同。

## 5. manifest 中 306,801 个 NPZ 的单个文件是怎么生成的？

### 从 DB 到 NPZ 的关系

流程不是“一个 DB 生成一个 NPZ”，而是：

```text
一个 NuPlan DB 日志
→ NuPlanScenarioBuilder 根据有效场景锚点构造多个 Scenario
→ 每个被选中的 Scenario 生成一个 NPZ
```

共同阶段数据覆盖 44 个官方 mini-train 日志。`ScenarioBuilder` 共构造出 306,801 个去重 Scenario，本次目标数量也设为 306,801，因此全部进入最终缓存。

### 预处理入口

`HDP-nuplan/data_process.py` 依次执行：

1. 读取原始 DB 路径、地图路径和 44 个 train 日志名；
2. 创建 `NuPlanScenarioBuilder`；
3. 用 `ScenarioFilter` 获取这些日志中的 Scenario；
4. 去重并按日志组织/选择 Scenario；
5. 把 Scenario 列表交给 `DataProcessor.work()`；
6. 只把成功生成的 NPZ 文件名写入 manifest。

完整 mini-train 当时拆成 4 个预处理 shard 并行执行，最后合并、排序、去重和校验，形成 306,801 项总 manifest。

### 一个 Scenario 提取什么

`DataProcessor.process_scenario()` 以该 Scenario 的初始自车状态为锚点，提取：

- 当前自车状态；
- 过去 2 秒自车和邻车历史；
- 未来 8 秒、80 个自车轨迹点；
- 未来 8 秒邻车轨迹；
- 当前静态物体；
- 自车 100 米范围内的 lane、边界、route、限速和红绿灯状态；
- `map_name`、`log_name`、`scenario_type` 和 scenario token。

几何量转换到当前自车局部坐标系，再按固定最大数量和点数执行选择、补零或截断。

### 如何保存

输出文件名为：

```text
<map_name>_<scenario.token>.npz
```

程序先使用 `np.savez()` 写临时文件，执行 `flush` 和 `fsync`，最后用 `os.replace()` 原子替换成正式 NPZ。这样进程中断时不会把未写完的半文件误认为有效缓存。

因此：

```text
306,801 个唯一 Scenario
↔ 306,801 个唯一 NPZ
↔ manifest 中 306,801 个文件名
```

## 6. `DistributedSampler` 和 `DataLoader` 分别有什么作用？

### DistributedSampler

它决定“当前 rank 在这个 epoch 应该读取哪些 Dataset 索引，以及索引顺序是什么”。主要职责是：

- 每个 epoch 打乱样本索引；
- 多卡时把索引划分给不同 rank，避免所有 GPU 重复读取整套数据；
- `set_epoch(epoch)` 让不同 epoch 使用不同但可复现的打乱顺序。

本次 `world_size=1`，所以它不会真正切分数据，但仍负责每轮 shuffle。

### DataLoader

它负责把 sampler 给出的索引真正变成训练 batch：

1. 按索引调用 `DiffusionPlannerData.__getitem__()`；
2. 从磁盘打开对应 NPZ；
3. 读取并截取模型所需字段；
4. 把多个样本合并成一个 batch；
5. 根据 `num_workers` 决定由主进程还是 worker 进程加载；
6. 根据 `pin_memory` 使用页锁定内存；
7. 根据 `drop_last=True` 丢弃最后不足 8 个样本的 batch。

简化区分如下：

```text
DistributedSampler：决定读哪些索引、按什么顺序读
DataLoader：按这些索引加载文件，并组装成 batch
Dataset：定义一个索引具体返回哪些数据
```

## 7. B Epoch10 的 AdamW 和学习率调度器是什么？为什么这样设置？

### AdamW 是什么

训练代码使用：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
```

B Epoch4 checkpoint 中确认的其他参数是 PyTorch 默认值：

```text
betas       = (0.9, 0.999)
eps         = 1e-8
weight_decay= 0.01
amsgrad     = false
```

AdamW 为每个参数维护梯度的一阶、二阶滑动统计，并把 weight decay 与梯度更新解耦。它常用于 Transformer 和扩散模型，因为相较普通 SGD，对不同尺度、噪声较大的梯度通常更容易优化。

当前代码把全部参数放在一个参数组中，没有为 bias 或 LayerNorm 单独关闭 weight decay。这是代码沿用的简化配置，不代表已经证明是本任务最优设置。

### 共同阶段调度器

共同 Epoch1～4 创建：

```python
CosineAnnealingWarmUpRestarts(
    optimizer,
    train_epochs=20,
    warm_up_epoch=2,
)
```

虽然函数名包含 `CosineAnnealing`，当前实现实际上只有：

```text
LinearLR warm-up → 固定学习率
```

没有余弦退火。设计意图是先用基础学习率的 10% 启动，再升到基础学习率，减少训练初期随机 Decoder 梯度过大、破坏 warm-start Encoder 的风险。

初始参数为：

```text
基础学习率 5e-4
warm-up 起点 5e-4 × 0.1 = 5e-5
```

但由于第 9 问所述的中断恢复行为，最终有效的 Epoch1～4 都在 `5e-5` 完成。

### B 分支 Epoch5～10 调度器

从共同 Epoch4 分叉后，Experiment B 显式使用：

```text
learning_rate=5e-5
warm_up_epoch=1
reset_lr_schedule_on_resume=true
```

`warm_up_epoch<=1` 时调度函数返回倍率始终为 1 的 `MultiplicativeLR`，所以 Epoch5～10 保持恒定 `5e-5`。

完整恢复保留 Epoch4 的 AdamW 一阶、二阶动量；`reset_lr_schedule_on_resume=true` 只把学习率重设为 `5e-5` 并重建调度器，不会把 optimizer 动量清零。

### 为什么最后选择恒定 `5e-5`

这是 A/B 对照中的实验选择：A 解冻后升到较高学习率，训练 loss 明显上升；B 从同一个 Epoch4 checkpoint 继续使用低学习率，loss 更稳定。因此 B 的目标是验证“解冻后保持低学习率”是否更合适，而不是声称 `5e-5` 已通过完整搜索达到全局最优。

## 8. `wandb_id = None` 是什么？

`wandb_id` 是 Weights & Biases 实验运行的标识符，用于中断后把日志继续写入同一次 W&B run。

新实验没有旧运行可以恢复，所以代码先设置：

```python
wandb_id = None
```

含义只是：

```text
当前没有需要恢复的既有 W&B run ID
```

它与模型权重、训练随机种子和 GPU rank 都无关。

本次命令使用 `--use_wandb false`，实际 checkpoint 中的 `wandb_id` 也是 `None`。TensorBoard 日志仍会写入：

```text
<save_path>/tb/
```

如果启用并成功初始化 W&B，Logger 会得到新 ID，并在 checkpoint 中保存；恢复训练时再把该 ID 传给 W&B，实现日志续接。

## 9. Epoch1～4 每次恢复后又回到 `5e-5`，是预期内还是预期外？

### 结论

对原始训练计划来说，这是 **预期之外** 的结果。

原本的计划是：

```text
以 5e-5 开始 warm-up
→ 升到基础学习率 5e-4
→ 后续保持 5e-4
```

但是共同阶段多次在后续 epoch 中途被终止，而代码的 epoch 末顺序是：

```text
保存 checkpoint
→ scheduler.step()
```

因此完整 checkpoint 保存的是 `scheduler.step()` 之前的低学习率状态。恢复时模型、optimizer 和 scheduler 都回到这个保存点，使下一次完整训练的 epoch 再次使用 `5e-5`。

所以应区分：

- 从代码执行规律看：给定当前保存顺序和恢复方式，这个结果可以解释、可以复现；
- 从实验设计意图看：让 Epoch1～4 全部停留在 warm-up 起始学习率不是最初计划，属于中断恢复暴露出的调度状态问题。

后来发现低学习率训练更稳定，才有意识地建立 Experiment B，并让 Epoch5～10 恒定使用 `5e-5`。也就是说：共同阶段的低学习率轨迹最初是意外产生的；B 分支继续低学习率则是后续主动设计的对照实验。

更稳妥的长期修复方案应当是：保存 checkpoint 时记录完成 `scheduler.step()` 后的状态，或者恢复后根据 `init_epoch` 明确重建正确的调度位置，同时保存并恢复完整 RNG 状态。

## 10. DataLoader 把 8 个样本堆叠成 batch 是什么操作？在哪里进行？

### 什么是堆叠

Dataset 每次 `__getitem__(idx)` 返回一个场景的 11 个 NumPy 数组。例如一个样本的自车未来是：

```text
[80,3]
```

DataLoader 一次取 8 个索引后，把同一字段沿新建的第 0 维堆叠：

```text
8 个 [80,3]
→ 1 个 [8,80,3]
```

其他例子：

```text
8 个 ego_current_state [10]
→ batch[0] [8,10]

8 个 neighbor_agents_past [32,21,11]
→ batch[2] [8,32,21,11]

8 个 lanes [70,20,12]
→ batch[4] [8,70,20,12]
```

第 0 维就是 batch 维，表示 8 个彼此独立的场景。模型对它们并行计算，不会把不同场景的车辆或轨迹混在一起。

### 在哪里进行

`train_predictor.py` 创建 DataLoader 时没有传入自定义 `collate_fn`：

```python
train_loader = DataLoader(..., batch_size=8, ...)
```

因此 PyTorch 自动使用 `default_collate`。它会：

1. 收集 8 次 `Dataset.__getitem__()` 的返回值；
2. 按 tuple 的相同位置归组；
3. 把 NumPy 数组转成 PyTorch Tensor；
4. 用 `torch.stack(..., dim=0)` 增加 batch 维。

真正触发读取和堆叠是在 `train_epoch.py` 迭代 DataLoader 时：

```python
for batch in data_epoch:
```

不是在创建 DataLoader 对象的那一刻就把全部 306,801 个样本装入内存。DataLoader 每次只按需准备下一个 batch。

## 11. `train_epoch.py` 第 92 行是怎样进行数据增强的？

第 92 行是：

```python
inputs, ego_future, neighbors_future = aug(
    inputs, ego_future, neighbors_future
)
```

其中 `aug` 是训练入口创建的 `StatePerturbation` 对象。调用对象会进入它的 `__call__()`，过程分为三步。

### 第一步：决定哪些样本增强并扰动当前状态

对 batch 中每个场景单独生成 `aug_flag`。只有当前纵向速度绝对值不低于 2 m/s、且随机条件命中的样本才增强。

扰动范围为：

| 状态 | 扰动范围 |
|---|---:|
| x | 0，本次不直接扰动 |
| y | `[-0.75, 0.75] m` |
| yaw | `[-0.35, 0.35] rad` |
| vx | `[-1, 1] m/s` |
| vy | `[-0.5, 0.5] m/s` |
| ax | `[-0.2, 0.2] m/s²` |
| ay | `[-0.1, 0.1] m/s²` |

随后约束纵向速度不小于 0，并根据速度和 yaw rate 重新计算合理的转向角。

### 第二步：为扰动后的自车重新连接未来轨迹

如果只改变当前状态而仍使用原始未来标签，轨迹起点会发生跳变。代码使用五次多项式，根据扰动后的初始位置、方向、速度、加速度和原轨迹约 2 秒处的终端条件，重建前 20 个未来点。

输出仍然是 80 个点：

```text
重新生成的前 20 点
+ 原轨迹后续 60 点
= 80 点
```

只有 `aug_flag=True` 的样本会用新当前状态和新未来轨迹覆盖原数据。

### 第三步：重新建立自车中心坐标系

扰动后当前自车不再严格位于原点，所以 `centric_transform()` 以新的当前自车位姿重新平移和旋转：

- 当前自车位置、方向、速度、加速度；
- 自车未来轨迹；
- 邻车历史和未来；
- lane 和 route lane；
- 静态物体。

点坐标需要“减去新自车位置再旋转”；方向、速度、加速度只旋转，不平移。padding 项在变换后重新置零。

最终返回的三个对象仍保持原接口和形状：

```python
inputs, ego_future, neighbors_future
```

后续代码再把 heading 转成 `cos/sin`，执行观测归一化和扩散损失计算。

## 12. 总结

| 问题 | 核心结论 |
|---|---|
| batch size 8 | 由本地 8 GiB 显存约束选择，不是理论最优值 |
| 数据增强 | 用偏离状态学习闭环恢复；实际整体增强率低于 50% |
| normalization | 训练前静态设定，主要沿用 DP 并适配 HDP 增量目标，不是从 306,801 NPZ 自动统计 |
| seed 3407 | 控制多套 RNG；确定性设置提高复现性但可能降低速度，也不能保证绝对一致 |
| NPZ | 每个选中 NuPlan Scenario 生成一个，306,801 个 Scenario 对应 306,801 个 NPZ |
| Sampler/DataLoader | 前者决定索引和分片，后者读取并组 batch |
| AdamW/scheduler | AdamW 使用默认动量和 0.01 weight decay；共同阶段原计划 warm-up，B 分支恒定 `5e-5` |
| `wandb_id=None` | 新实验没有待恢复的 W&B run，本次主要使用 TensorBoard |
| 反复回到 `5e-5` | 对最初学习率计划属于意外，对当前代码恢复规律则可以解释 |
| 8 样本堆叠 | PyTorch `default_collate` 按字段新增 batch 维，不会混合场景语义 |
| 第 92 行增强 | 扰动当前状态、重建前 2 秒未来、再统一重建自车中心坐标系 |
