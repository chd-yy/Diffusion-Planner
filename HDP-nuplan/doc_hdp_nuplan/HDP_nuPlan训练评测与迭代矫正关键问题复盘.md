# HDP-nuPlan 训练、评测与迭代矫正关键问题复盘

本文复盘 HDP-nuPlan 从数据准备、监督训练、B Epoch10 选择、奖励加权后训练、固定场景验证，到 Test14 完整闭环评测过程中最值得关注的问题。重点不是罗列每次命令，而是说明：问题为什么发生、如何定位、采用了什么修正、修正是否真正有效，以及以后怎样避免重复踩坑。

本文所称：

- **B Epoch10**：完整 mini-train 监督训练中，从共同 Epoch4 checkpoint 分叉、以恒定 `5e-5` 继续到 Epoch10 的 HDP 模型；
- **RL Epoch2**：从 B Epoch10 出发，在平衡 10,000 NPZ 上训练的 `safetygate_ttc1_anchor01_10k_seed2026` checkpoint；
- **对齐原始 DP**：使用与 B Epoch10 尽量一致的数据、Encoder warm-start、冻结策略和学习率阶段重新训练的原始 Diffusion Planner Epoch10。

## 1. 最终能确认到什么程度

截至 2026-09-07，最可靠的正式结果来自 Test14-hard 272 场景和 Test14-random 261 场景的单 worker、固定 token、完整配对闭环评测。三个模型共完成：

```text
(272 + 261) × 3 = 1,599 次闭环仿真
```

结果为 `1599/1599` 成功，场景集合无重复、无遗漏。

| Benchmark | 对齐原始 DP | B Epoch10 | RL Epoch2 |
|---|---:|---:|---:|
| Test14-hard score | 0.602691 | **0.705304** | 0.702422 |
| Test14-random score | 0.689908 | 0.824270 | **0.831146** |
| 533 场景按数量加权 | 0.645400 | 0.763559 | **0.765456** |

由此可以严谨地说：

1. 在对齐训练协议的原始 DP 对照下，B Epoch10 在两个 Test14 benchmark 上都明显更强；
2. RL Epoch2 在两个 benchmark 上都提高了路线进度；
3. RL Epoch2 在 Test14-random 上获得 `+0.006876` score，在 Test14-hard 上为 `-0.002882`；
4. 533 场景按数量加权后，RL Epoch2 相对 B Epoch10 为 `+0.001896`，属于**总体小幅正收益**；
5. 由于困难集仍为负收益，且训练 seed 稳定性不足，不能表述为“RL 已在所有场景分布上稳定优于监督模型”。

533 场景加权值是便于总体观察的描述性汇总，不是 NuPlan 官方另行定义的单一 benchmark 指标。

## 2. 项目中最关键的经验

可以把整个过程浓缩为八条：

1. **数据量正确不等于数据身份正确**：必须用 manifest、日志集合、唯一 token 和文件哈希定义训练集与评测集；
2. **训练 loss 不能代替闭环指标**：B Epoch20 loss 更低，但闭环明显弱于 Epoch10；
3. **恢复训练本身就是实验变量**：scheduler、optimizer、sampler 和 RNG 任一项未恢复或被重置，都会改变训练轨迹；
4. **公平对照必须对齐训练协议**：只对齐 NPZ 和 epoch 不够，还要对齐初始化、Encoder 冻结、学习率阶段、EMA 和推理配置；
5. **Reward 改善不等于 NuPlan score 改善**：代理 reward 与官方碰撞、TTC、drivable area、progress 的定义不完全一致；
6. **RL 数据规模由 Replay Buffer 和 update 共同决定**：遍历 10,000 场景，不代表 10,000 场景都参与梯度更新；
7. **小评测只能做 smoke，不能做结论**：3、20、59、200 场景适合逐级排错，正式判断仍需更完整 benchmark；
8. **评测基础设施也会污染指标**：线程安全、分片聚合、YAML token 类型和断点恢复都必须设置硬门禁。

## 3. 数据准备：先定义“到底训练了什么”

### 3.1 早期问题

项目先后出现过 100、1,000、10,000、35,869、局部城市和完整 mini-train 等多个 NPZ 集合。如果只说“使用 mini 数据”或只统计目录下 NPZ 数，很容易把不同来源、不同采样策略的数据混为一谈。

此外，处理到 298,061 个 NPZ 时虽然已经接近完整 306,801，但缺失的 8,740 个样本集中在最后一个分片尾部。直接扫描现有文件生成 manifest 会产生排序和分片偏差，不能称为完整 mini-train。

### 3.2 解决方法

正式训练前建立四层门禁：

1. 每个预处理 shard 必须有 `status=complete`、`failed=0` 的报告；
2. 合并后的 manifest 必须为 306,801 项；
3. 检查 manifest 唯一性、NPZ 数量、缺失文件、shape 和有限数值；
4. 记录覆盖的 44 个 mini-train 日志、场景类型分布及 manifest SHA256。

最终 B Epoch10 使用：

```text
306,801 个完整 mini-train NPZ
44 个官方 mini-train 日志
manifest SHA256：a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0
```

RL Epoch2 使用的是从上述集合按日志平衡抽取的 10,000 NPZ，每个日志约 227～229 个样本，manifest SHA256 为：

```text
1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514
```

### 3.3 得到的经验

- 训练集身份应由 `cache + manifest + sampling report + hash` 共同定义；
- “目录里有多少 NPZ”只能说明物理文件数，不能说明训练入口实际读了哪些；
- 暂停或分片恢复后，不能把局部完成状态直接包装成完整数据；
- 训练 NPZ 与闭环评测 DB 是两套用途：NPZ 用于张量训练，NuPlan SQLite DB 和地图用于构建闭环场景。

## 4. 大规模预处理：并行化必须以不重复和可恢复为前提

### 4.1 问题表现

单进程生成 306,801 个 NPZ 速度逐渐下降，完整处理时间过长。直接增加进程又可能造成：

- 多个进程处理同一 Scenario；
- 输出文件名冲突；
- 多进程争用同一 SQLite DB；
- 某个 worker 中断后无法判断哪些样本已经可靠完成。

### 4.2 解决方法

- 按确定性场景列表切成四个互斥 shard；
- 所有 shard 写共享 cache，但输出名称全局唯一；
- 每个 shard 单独保存 manifest、日志和完成报告；
- 重启时先验证已有 NPZ，再越过完成点生成新增文件；
- 全部分片完成后统一合并和做全量验证。

### 4.3 经验

并行不是简单把 `workers` 调大。可靠并行需要同时定义：任务分片、幂等写入、完成标记、失败重试、最终集合校验。否则速度提高了，但数据身份无法证明。

## 5. B Epoch10 初始化：Encoder warm-start 不是加载整个原始模型

### 5.1 容易误解的地方

B Epoch10 确实使用了仓库原始 Diffusion Planner checkpoint，但只把其中与 HDP Encoder 名称和形状一致的 EMA 参数作为 warm-start：

```text
Encoder：151/151 个张量，1,799,040 个参数成功加载
Decoder：0 个参数加载
```

因此不能说 B Epoch10 是“在原始 Diffusion Planner 整体模型上继续训练”。准确说法是：

```text
沿用原始 DP Encoder 的场景理解参数，HDP Decoder 按自身结构学习
```

### 5.2 解决方法

训练入口增加 warm-start 审计报告，明确记录：

- 权重来源是 `model` 还是 `ema_state_dict`；
- 成功加载的张量和参数数量；
- missing、shape mismatch 和 unexpected keys；
- 是否误加载 Decoder。

### 5.3 经验

“使用预训练模型”必须拆成：加载了哪个 state dict、加载哪些模块、多少参数、是否 strict。否则模型优势来源无法解释，公平对照也无法建立。

## 6. 监督训练的内存问题：最危险的不是 GPU OOM

### 6.1 第一个误判：DataLoader workers

完整 mini-train 训练中，主进程和四个 DataLoader worker 使 15 GiB RAM 耗尽。最初将 `num_workers` 从 4 降到 0，内存暂时下降，因此一度认为问题已经解决。

但继续观察发现，即使 `num_workers=0`，RSS 仍会随 batch 持续增长到约 11.5 GiB。

### 6.2 真正根因：跨 batch 保留计算图

`train_epoch.py` 原实现把每个 batch 的 loss Tensor 字典直接追加到 `epoch_loss`。这些 Tensor 仍连接反向计算图，导致一个 epoch 已完成 batch 的图始终不释放。

小数据集上不明显，完整 mini-train 每轮 38,350 个 batch 时会形成持续内存泄漏式增长。

### 6.3 修复

记录 epoch 统计前改为：

```python
value.detach().item()
```

只保留 Python 标量，反向传播仍在此前完成，因此不改变 loss 和梯度定义。修复后训练内存约 1.8～3.3 GiB，不再随 batch 数线性上升。

### 6.4 经验

- 遇到 OOM 必须先区分 GPU 显存、主进程 RAM、DataLoader worker 和 page cache；
- 降低 `num_workers` 可以减轻内存，但不一定解决计算图滞留；
- 训练监控列表中不要长期保存带 grad graph 的 Tensor；
- 正式长跑前应做“RSS 随 batch 曲线”检查，而不只观察启动时内存。

## 7. 中断恢复与学习率：checkpoint 完整不代表轨迹完全连续

### 7.1 暴露的问题

B 的共同 Epoch1～4 多次中断恢复。原训练循环先保存 checkpoint，再执行 `scheduler.step()`，所以 checkpoint 中保存的是推进前的 scheduler 状态。恢复后低学习率阶段被重复，导致 Epoch1～4 的实际优化器学习率均为 `5e-5`，而不是最初计划的完整 warm-up 后切换。

Epoch5 首次切到 `5e-4` 后，loss 从 `0.0265` 跳到 `0.1459`。这个跳升与十倍学习率变化、Encoder 已解冻同时发生，不是训练数据突然改变。

### 7.2 处理方式

- 固定复制 Epoch4 完整 checkpoint，避免后续 `latest.pth` 覆盖起点；
- 创建 A/B 分支：A 使用 `5e-4`，B 重置为恒定 `5e-5`；
- 增加 `reset_lr_schedule_on_resume`，明确重设 optimizer 的 `lr/initial_lr` 并重建 scheduler；
- 对齐 `train_sampler.set_epoch(init_epoch)`，避免恢复后从 sampler epoch 0 重新打乱；
- 每次恢复必须检查 model、EMA、optimizer、scheduler 和 epoch 都成功加载。

### 7.3 尚未完全解决的边界

旧 checkpoint 没有保存 Python、NumPy、PyTorch 和 CUDA RNG 状态。因此 B Epoch10 可以精确用于推理，但从零按相同协议重训只能保证统计意义和流程一致，不能保证逐参数位级复现。

### 7.4 经验

以后 checkpoint 应额外保存：

```text
Python random state
NumPy random state
torch CPU RNG state
torch CUDA RNG states
sampler epoch
global optimizer step
当前实际 learning rate
完整不可变 args 快照
```

并统一规定 scheduler 在保存前还是保存后推进，不能依赖口头约定。

## 8. 为什么最终选择 B Epoch10，而不是 loss 更低的模型

### 8.1 关键反例

B 从 Epoch10 继续训练到 Epoch20 后：

```text
train loss：0.0091 → 0.0055
```

但固定 20 场景闭环 score：

```text
B Epoch10：0.876941
B Epoch20：0.810012
变化：     -0.066928
```

主要退化来自 drivable-area、舒适性和 TTC 的少数硬失败场景。

### 8.2 说明了什么

监督 loss 衡量的是模型对缓存专家轨迹和扩散目标的拟合，不直接等价于闭环中的：

- 误差递推；
- 与其他交通参与者的交互；
- drivable-area 合规；
- TTC 和碰撞；
- 车辆动力学和舒适性。

因此训练更久、loss 更低，仍可能让闭环能力下降。

### 8.3 解决思路

- 用开环指标做高频筛选；
- 用固定小场景闭环做阶段门禁；
- 对候选 checkpoint 做相同 token 的逐场景配对；
- 把安全硬指标退化设为否决项，而不是只看平均 score；
- 根据闭环结果选择 Epoch10，不能根据最低 train loss 自动选择 Epoch20。

## 9. 公平对比：原始 DP 基线曾经不公平

### 9.1 两种错误比较

第一种是直接拿仓库发布的成熟原始 Diffusion Planner 与本地 B Epoch10 比。发布模型可能使用了不同规模数据和训练过程，因此它适合作为工程参考，但不是同训练条件对照。

第二种是使用相同 306,801 NPZ 从随机初始化训练原始 DP，却没有复现 B 的 Encoder warm-start 和前三轮冻结。这一版本 fixed200 只有 `0.580725`，不能把巨大差距完全归因于模型结构。

### 9.2 修正后的对照

重新训练原始 DP 时对齐：

- 相同完整 mini-train；
- 相同 seed、batch size 和 epoch；
- 相同原始 DP Encoder EMA warm-start；
- 相同 Epoch1～3 冻结、Epoch4 起解冻；
- 相同两阶段学习率；
- 相同 EMA 和闭环场景。

修正后原始 DP fixed200 从 `0.580725` 提升到 `0.795445`，证明训练协议本身是重要变量；但仍低于 B Epoch10 的 `0.904567`。

### 9.3 仍要保留的边界

原始 DP 和 HDP 的网络结构及原生 loss 不同，因此即使数据和训练协议已对齐，也不是“只改一个神经网络模块”的严格消融。准确说法应是“同数据、同初始化与训练日程下的两种完整 planner 对照”。

## 10. `args.json`、脚本和 checkpoint 可能互相矛盾

### 10.1 实际问题

B Epoch10 所在目录后来继续训练到 Epoch20，目录级 `args.json` 的 `train_epochs` 和名称可能反映后续运行，而 checkpoint 本身仍然是 Epoch10。只读取当前 args 文件，会误解 B Epoch10 的形成过程。

另外，训练 args、planner YAML 和 Hydra 运行时 overrides 分别描述：

- checkpoint 如何训练；
- planner 如何构造；
- 本次闭环实际覆盖了哪些配置。

三者不能互相替代。

### 10.2 解决方法

- 用 checkpoint 内的 `epoch/loss/state_dict` 确认模型身份；
- 用训练日志重建实际学习率、冻结和恢复历史；
- 用同次运行的 `args.json` 确认参数；
- 用 Hydra `.hydra/config.yaml` 和 `overrides.yaml` 确认闭环实参；
- 对 checkpoint、args、manifest 记录 SHA256；
- 正式对比使用独立、不可覆盖的运行目录。

### 10.3 经验

模型名称不应成为证据。最可靠的模型身份是：

```text
checkpoint hash + checkpoint epoch + args hash + code commit + data manifest hash
```

## 11. RL Epoch2 的方法性质容易被说错

### 11.1 它不是真正的在线 RL

RL Epoch2 的训练没有在 simulator 中执行动作并获得环境下一状态，也没有 critic、value network、TD target 或 PPO。它在缓存 NPZ 场景中：

1. 为每个场景生成一组扩散候选；
2. 用固定的邻车未来、地图和路线计算代理 reward；
3. 用 reward 加权扩散回归；
4. 加入真实专家轨迹 anchor。

更严谨的名称是“奖励加权的扩散模型后训练”。项目中保留 RL 名称只是为了与历史脚本和 checkpoint 对应。

### 11.2 经验

对外描述方法时，要把“训练信号来自 reward”与“在线环境交互强化学习”区分开，避免方法定义被质疑。

## 12. Reward 的第一个问题：高分候选可能更保守、更慢

### 12.1 现象

最初只使用 risk、follow、lane 三类奖励时，自然候选中高 reward 轨迹虽然更安全，但平均更偏向减速。弱监督起点下，模型可以通过少走甚至停车减少风险。

### 12.2 定位方法

不是直接改权重，而是对每个场景生成 32 条候选，检查：

- reward 与 progress 的相关性；
- 最佳候选相对组均值的终点、路径长度和碰撞代价；
- `beta=0` 与 `beta=1` 更新差别。

诊断发现，无 progress guard 时最高奖励候选的路径长度平均比组均值短约 0.067 m。

### 12.3 修正

加入有界 `progress_guard_reward`：

- 移动场景达到专家 progress 后得满分，继续超出不额外奖励，避免鼓励超速；
- 停车场景则要求接近专家的低 progress；
- 输出限制在 `[0,1]`。

它把 reward-progress 相关性由负向修正为正向，同时没有直接奖励无限加速。

### 12.4 经验

Reward 设计不能只看公式是否合理，要先查看它在模型**实际可生成候选分布**上的排序结果。候选空间里不存在的行为，再合理的 reward 也学不到。

## 13. Reward 的第二个问题：无权自蒸馏 baseline 支配更新

### 13.1 关键控制实验

将 reward temperature 设为 0 后，所有有效候选权重都等于 1，相当于无权自蒸馏。结果与正常 reward 权重更新几乎一样地退化。

这说明当时的更新主要受权重中的常数 `1` 支配，而不是 reward 差异支配。有限 Replay Buffer 上，理论上可能抵消的无权自蒸馏梯度并没有实际抵消。

### 13.2 当时的修正

使用 centered weight：

```text
w_centered = w - 1
```

它消除了常数 baseline，使 `beta=0` 时梯度严格为 0，并曾在小规模开环与 fixed200 的 seed2026 上产生正向结果。

### 13.3 后来发现的新问题

当 `w<1` 时，`w-1<0`。低 reward 候选因此不是“少拟合”，而是被负 MSE 权重推动为**主动增大拟合误差**。该目标没有普通 MSE 的下界，可能把轨迹推向不可预测方向。

### 13.4 经验与后续方向

- `beta=0` 是识别 reward 是否真正起作用的重要控制实验；
- centered 方案证明了 reward-dependent 梯度存在，但不应作为长期稳定目标；
- 更稳妥的方向是只使用非负权重，例如 `softmax_positive` 或 `positive_advantage`；
- 但后续实验表明，换成正权重只解决数值目标问题，若 reward 与闭环指标错位，仍不会自动得到正收益。

## 14. RL 数据规模被误读：10,000 rollout 不等于 10,000 update

### 14.1 RL Epoch2 实际发生的事

Epoch1 遍历 10,000 场景，每场景生成 32 条候选，共 320,000 条轨迹。但：

```text
rl_buffer_size = 1024
```

Replay Buffer 使用 `deque(maxlen=1024)`，所以最终只留下打乱顺序末尾的 1,024 个场景组。Epoch2 从这 1,024 个场景中有放回抽样 500 步，每步 2 个场景。

因此本次模型不是“完整用 10,000 场景更新了一遍”，而是：

```text
完整 10k 用于 rollout 统计
最后 1,024 个场景用于 update 候选池
500 × 2 次有放回场景抽取用于梯度更新
```

### 14.2 为什么 seed 会影响很大

不同 seed 改变 DataLoader 的打乱顺序，从而改变最后留在 Buffer 的 1,024 个场景。seed42 和 seed2026 实际更新的数据子集不同，所以 fixed200 上一个略降、一个略升并不意外。

### 14.3 扩大 Buffer 后为什么仍可能变差

把 Buffer 扩到 10,000、增加到 6 epoch、提高 noise scale 后，确实提高了覆盖和候选差异，但也让错误的代理目标被更充分优化。某次实验仅 3 个 drivable-area 硬失败，就把其余 197 个场景的小幅正收益完全抵消。

### 14.4 经验

扩大数据、Buffer 和 update 次数只会更忠实地优化当前目标；如果目标有偏差，规模化会放大偏差。正确顺序应是：

```text
验证候选质量
→ 验证 reward 排序
→ 验证一次 update 方向
→ 验证固定场景闭环
→ 再扩大 Buffer 和轮数
```

## 15. `train_epochs=2` 的语义容易误解

在 `rl_buffer_update_epoch=2` 下：

```text
Epoch1 = rollout，0 次 optimizer step
Epoch2 = update，500 次 optimizer step
```

所以 RL Epoch2 只有一次 rollout-update 循环。增加到 4 或 6 epoch 才会产生 2 或 3 次循环。

经验是：RL 日志不能只报告 epoch 数，还应同时报告：

```text
rollout cycles
update epochs
optimizer steps
unique replay scenes
candidate trajectories
buffer replacement ratio
```

## 16. Safety gate 的名字比实际能力更“硬”

### 16.1 RL Epoch2 的 TTC gate

当时候选要求：

```text
risk_reward >= 0.3
min_ttc_seconds >= 1.0 s
```

但历史实现只是把不合格候选的 reward 压低，没有把 safety mask 传给 update loss。不安全候选仍然参与 centered 回归，而且可能获得负权重。

rollout 日志还显示：只有约 55.6% 的场景组至少存在一条合格候选，约 44.4% 的组没有安全候选。这些组只能在“不安全候选里选相对不坏的一个”。

### 16.2 后续修正与结果

后来把候选 mask 真正传入 loss，只允许安全候选成为正权重目标，并在整组无合格候选时只保留 expert anchor。这改善了部分碰撞退化，但可能偏向低进度的保守候选，仍未稳定提高 score。

### 16.3 drivable-area gate 为什么没有成功

NPZ 没有保存 NuPlan 原生 drivable-area polygon，只有有限范围的 lane 中心线和左右边界向量。用这些信息构造的代理硬门会在路口、换道和地图覆盖边缘把表示误差当成真实越界。

即使增加“专家轨迹也必须能被当前 lane corridor 覆盖才启用 gate”的保护，最终 fixed200 仍没有恢复正收益。因此该代理 gate 被关闭，没有冒充官方 drivable-area 约束。

### 16.4 经验

- “硬门”必须检查它在候选生成、Replay 保存和 loss 计算三个位置是否都真正生效；
- 代理地图表示不足时，不应声称等价于官方 polygon 指标；
- 整组无安全候选时，正确动作通常是跳过该组策略更新并保留专家约束，而不是强行选择一个危险候选。

## 17. 代理 Reward 与 NuPlan 官方 Score 不完全对齐

### 17.1 历史实现中的错位

RL Epoch2 的最终 reward 主要是：

```text
1 × risk + 3 × follow + 2.5 × lane + 5 × progress_guard
```

虽然 `args.json` 中还有 collision、route、comfort 等权重，但历史版本里这些量主要是诊断项，没有直接加入最终 reward。

NuPlan 最终 score 则受到碰撞、drivable area、TTC、progress、限速和舒适性等指标共同影响，其中部分还是乘法或近似硬门控。少数场景从 1 变成 0，就能压过大量场景中的微小 progress 收益。

### 17.2 后续尝试为什么仍不够

项目继续尝试了：

- NuPlan-aligned reward；
- score proxy v2/v3；
- positive advantage；
- B10 reference anchor；
- reference-relative reward；
- 多轮 update。

这些改造使代码链路更合理，也能改善某些训练统计，但多个 fixed200 实验仍为负收益。这证明“代理公式看起来更像官方指标”还不够，必须用逐场景闭环验证实际相关性。

### 17.3 解决思路

下一轮 reward 改造应先建立“代理—官方指标标定集”：

1. 固定一批场景和候选轨迹；
2. 同时计算训练代理 reward 与 NuPlan 闭环指标；
3. 按碰撞、TTC、越界、低进度和舒适性分组分析排序一致率；
4. 只有代理能稳定把官方更优轨迹排在前面，才进入参数更新；
5. 每次只改一个 reward 组成或门槛，保留完整消融。

## 18. Reference policy 不是“加一个权重就能保底”

后续实验加入冻结 B Epoch10 reference model，但曾出现 reference anchor 的加权损失只占 RL loss 约 `0.023%`。参数名看起来开启了约束，实际梯度贡献几乎可以忽略。

把 reference 权重增大后，训练链路稳定，但仍可能因为 proxy reward 与闭环目标错位而降低路线进度。reference-relative 多轮实验中，第一次 update 已朝错误方向，继续训练到 Epoch6 只是进一步放大退化。

经验是：正则项不能只看配置值，要在日志中检查：

```text
weighted_reference_loss / weighted_rl_loss
梯度范数比例
模型相对 reference 的输出漂移
闭环逐场景变化
```

Reference 只能限制偏离幅度，不能把错误的优化方向自动变正确。

## 19. 小场景门禁的价值和局限

### 19.1 为什么 3 场景结果波动很大

3 个场景中任意一个硬指标从 1 变 0，均值都会变化约 0.333；它适合检查能否运行、模型是否加载正确、指标是否生成，不适合比较千分位收益。

### 19.2 20、59、200 场景分别能做什么

- 20：快速发现明确 TTC、碰撞或越界回归；
- 59：合并不同小门禁，检查方向是否重复；
- 200：做逐场景配对、seed 对比和 bootstrap；
- 完整 Test14-hard/random：判断是否能跨场景分布泛化。

### 19.3 一个重要的“结果反转”

早期本地 test14-random 只有当前 64 个 mini DB 能构建出的 186 场景，RL seed2026 在该子集上为负收益。完整下载 test 数据后，官方 261 场景 Test14-random 上 RL Epoch2 反而为 `+0.006876`。

两者不是同一个评测集合，不能把 186 场景结果称为完整 Test14-random，也不能直接用两个均值判断代码前后矛盾。

### 19.4 经验

每个结果表必须同时写：

```text
benchmark 名称
实际 scenario_count
scenario token manifest/hash
DB 覆盖范围
challenge
worker 配置
checkpoint hash
```

## 20. 多随机种子：先用于诊断，后用于结论

fixed200 上：

```text
RL seed42：  -0.000070
RL seed2026：+0.001498
```

两个 seed 方向不同，说明当时训练结果对 Replay 子集和扩散噪声敏感。不能挑 seed2026 后声称算法稳定有效，但 seed2026 可以作为候选 checkpoint 继续扩大评测，前提是明确它是筛选出的实例。

合理流程是：

1. 单 seed 快速定位实现和目标问题；
2. 固定所有参数后增加至少 3 个训练 seed；
3. 每个 seed 使用同一评测 token；
4. 报告均值、方差、胜负场景和硬安全回归；
5. 在完整 benchmark 上确认方向，而不是只选最好 seed。

## 21. 闭环评测速度慢：瓶颈主要不在 GPU

NuPlan `closed_loop_nonreactive_agents` 需要逐步执行仿真、状态更新、指标计算和大量 Python/CPU 逻辑。模型本身参数量较小，GPU 显存约几百 MiB 属于正常现象。

因此：

- 4090 与更多 GPU 不会线性缩短单个场景仿真；
- 增加 simulation worker 主要提高 CPU 并发，而不是模型吞吐；
- 盲目增加 worker 会遇到 RAM 占用和指标线程安全问题。

最终正式 Test14 使用单 worker，牺牲速度换取指标可靠性。

## 22. 一次评测全部场景会导致 RAM 和桌面卡顿

一次构建 272 个 simulation 后，Python RSS 增长到约 9～10 GiB，本机 RAM 和 Swap 接近耗尽，桌面出现明显卡顿。

解决方法是按固定 token 切成每片约 40 个场景：

```text
Test14-hard：40×6 + 32
Test14-random：40×6 + 21
```

每片独立验证、归档 runner report 和 aggregator，最后按 token 集合统一聚合。单片运行时 RSS 约 2.3 GiB，支持中断后跳过完整分片。

经验是：闭环分片应以场景 token 为边界，而不是按“已经跑了多少个进度条”推测完成范围。

## 23. 并行评测曾产生指标线程安全问题

### 23.1 现象

两路仿真线程并行时，官方 `speed_limit_compliance` 指标出现 `time_stamps` 与 `violation_depths` 长度不一致。问题不在模型推理，而在同类场景复用了有状态 `MetricsEngine`，多个线程可能同时改写内部状态。

### 23.2 无效尝试

只把 `max_callback_workers` 改为 1 没有解决问题，因为当前 thread-pool 模式下并没有独立 callback pool，指标仍在多个 simulation 线程中同步执行。

### 23.3 最终处理

正式评测强制：

```text
worker.max_workers=1
worker.use_process_pool=false
max_callback_workers=1
disable_callback_parallelization=true
```

并把此前可能被线程竞争污染的结果移入备份，不通过“只补失败场景”继续使用，而是从零重测。

### 23.4 经验

没有报错不代表并行指标未被静默污染。只要发现共享有状态指标对象存在竞争，就应废弃整批可疑结果，而不是只修补显式失败样本。

## 24. 分片聚合也可能产生错误结果

NuPlan 每次启动会聚合当次可见的临时指标，并生成一个带时间戳的 aggregator parquet。多个恢复分片不会自动合成唯一总表。如果只取“最新 parquet”，可能只统计最后几场景。

解决方法：

- 每片归档自己的 runner 和 aggregator；
- 聚合前核对该片必需 token；
- 合并时要求总行数、唯一 token、目标集合完全一致；
- 发现重复或遗漏立即失败；
- 对只有 metric、没有 runner report 的中断历史明确标为 `metric_only_recovered`，不伪造耗时。

经验是：聚合正确性必须用 token 集合证明，不能用文件存在或进度条 100% 证明。

## 25. YAML token 也会改变场景集合

某些 16 位 token 含字母 `e`，未加引号写入 YAML 后可能被 OmegaConf 解释成科学计数法浮点数。场景本身并未缺失，但 filter 类型检查会失败。

修复后所有 token 强制双引号，并在启动前通过 `OmegaConf.load` 验证：

```text
类型必须是 str
长度必须是 16
数量、唯一性和目标集合必须一致
```

经验是：场景 ID 不只是字符串内容，还要验证经过最终配置解析器后的类型和值。

## 26. 长任务的暂停、恢复和托管

项目中出现过 SSH/会话断开、统一执行会话被回收、主动 `SIGTERM`、`SIGSTOP` 后系统重启，以及进程退出析构异常。

关键处理原则：

- 长训练每个 epoch 保存完整 checkpoint；
- 长评测每个 40 场景分片归档结果；
- 使用独立 systemd transient service 托管，不依赖前台终端寿命；
- 停止时针对明确 PID/PGID，避免误杀同进程组中的有效子任务；
- `SIGSTOP/SIGCONT` 只适用于机器不重启的原地暂停；重启后只能从磁盘断点恢复；
- 恢复前重新扫描已完成产物，不凭旧 PID 文件判断进度；
- 只有完整 checkpoint、runner 和 token 门禁通过后才跳过重跑。

## 27. 磁盘不足时，不应牺牲数据完整性

完整 NuPlan test ZIP 约 89.33 GiB，解压 DB 约 151.14 GiB，本机不能同时保存。项目采用 HTTP Range：

1. 逐 DB 临时下载并扫描场景索引；
2. 立即删除非目标 DB；
3. 确定 Test14-hard/random 最终 token；
4. 只保留 533 个场景涉及的 217 个 DB；
5. 验证 ZIP CRC、文件长度和 SQLite 完整性。

同时关闭 `simulation_log_callback` 以避免保存逐帧回放，但通过 smoke 确认 MetricCallback、aggregator 和 runner report 仍正常。

经验是：空间优化可以删除可重建的回放和非目标 DB，不能删除 manifest、checkpoint、正式指标或完整性证据。

## 28. 每次“修正”都必须重新证明有效

整个项目中有多次“代码更合理，但指标更差”的情况：

| 修正 | 工程上是否合理 | 闭环是否必然提升 |
|---|---|---|
| 扩大 Buffer | 是，提高数据覆盖 | 否，可能更充分优化错误 reward |
| 增加 update 轮数 | 是，提高学习强度 | 否，可能放大目标错位 |
| 增大采样噪声 | 是，提高候选多样性 | 否，可能增加越界候选 |
| drivable-area 代理 gate | 目的合理 | 否，lane 缓存不足会误判 |
| 正权重替代负权重 | 数值更稳定 | 否，安全但低进度候选仍可能主导 |
| reference anchor | 能限制漂移 | 否，不能纠正错误方向 |
| NuPlan score proxy | 更接近评测概念 | 否，代理与真实闭环仍有差距 |

最重要的经验是：**实现正确性、训练稳定性、代理指标改善和闭环正收益是四个不同门槛，必须逐个通过。**

## 29. 建议固化的标准实验流程

### 阶段 A：数据门禁

```text
固定 manifest
→ 验证数量/唯一性/shape/有限值
→ 记录日志和场景类型分布
→ 保存 manifest SHA256
```

### 阶段 B：训练前门禁

```text
固定代码 commit
→ 验证 checkpoint source 和 state_dict
→ 记录 trainable/frozen 参数量
→ 保存 args、normalizer 和启动命令
→ 运行单元测试和 1-batch smoke
```

### 阶段 C：训练监控

```text
记录实际 LR、Encoder 状态、loss 分项、RSS、GPU 显存
→ 每个 epoch 保存完整 checkpoint
→ 保存 RNG、sampler epoch 和 global step
→ checkpoint 写完后重新加载验证
```

### 阶段 D：RL 专用诊断

```text
候选多样性
→ reward std 分位数
→ 安全合格候选比例
→ 无合格候选组比例
→ Buffer 最终覆盖和唯一场景数
→ rollout/update 次数与 optimizer steps
→ 各损失项和梯度比例
```

### 阶段 E：逐级评测

```text
单场景启动 smoke
→ 固定 20 场景硬安全门
→ fixed200 逐场景配对
→ 多训练 seed
→ 完整 Test14-hard/random
→ 必要时完整 Val14
```

每一级都固定 checkpoint hash、scenario token、challenge、planner YAML、worker 和 metrics。上一级失败时先定位，不直接扩大数据或 epoch。

## 30. 当前仍未完全解决的问题

### 30.1 RL 跨 benchmark 稳定性

RL Epoch2 在 Test14-random 为正、Test14-hard 为负，说明更积极的路线进度改善还没有在困难场景中稳定兼顾碰撞和 drivable area。

### 30.2 Reward 与真实闭环因果关系

当前 reward 使用固定邻车未来，无法表示 reactive interaction。离线候选上安全，不代表闭环中仍安全。

### 30.3 候选覆盖

大量场景没有满足 safety gate 的候选。若扩散策略本身没有生成安全且高进度的候选，更新目标无从选择。

### 30.4 严格训练复现

早期 checkpoint 缺少 RNG 状态，B Epoch10 的共同阶段又经历多次恢复。现有 checkpoint 可以精确评测，但从零不能保证 bitwise 复现。

### 30.5 完整 Val14

当前最完整、可审计的正式结论来自 Test14-hard 272 和 Test14-random 261。早期仅基于本地部分 DB 构建的小型 val14 不能替代完整 Val14；若要补充，应复用当前的固定 token、分片、单 worker 和聚合门禁重新执行。

## 31. 下一步最有价值的改进顺序

1. **先固定现有三模型和 Test14 结果作为不可变基线**，不再覆盖目录或修改历史表；
2. **补齐训练可复现元数据**，包括 RNG、global step、代码 commit、manifest hash 和 args hash；
3. **构建代理 reward—闭环指标标定集**，先证明 reward 排序与官方指标一致；
4. **提高安全且高进度候选的覆盖率**，优先改候选生成，而不是继续放大 update；
5. **使用非负更新权重并在无合格候选时跳过策略项**，保留 expert/reference anchor；
6. **单变量消融**：Buffer、noise、reward、update 轮数一次只改一个；
7. **先在 hard 场景检查碰撞和越界**，再看总体平均；
8. **至少三个训练 seed**，最终报告均值、方差和逐场景胜负；
9. **如需完整项目结论，再补完整 Val14**，但不使用不完整 DB 子集冒充官方全集。

## 32. 可以沉淀成项目能力的部分

这段项目经历真正有价值的不只是“训练了一个模型”，而是形成了完整的问题闭环：

- 将 HDP 扩散规划器接入 NuPlan 数据、训练和官方闭环评测；
- 构建 306,801 场景的可验证监督训练缓存；
- 定位并修复大规模训练中的 CPU OOM、计算图滞留和恢复调度问题；
- 通过 A/B 学习率实验选出 B Epoch10，而不是盲目选择最低 loss；
- 构建奖励加权扩散后训练、Replay Buffer、EMA old policy 和专家 anchor；
- 通过 beta=0、候选排序、reward std、safety eligibility 等诊断定位更新问题；
- 构建固定 token、checkpoint hash、分片恢复、单 worker 的可审计闭环评测流水线；
- 在完整 Test14-hard/random 上完成 1,599 次无失败配对仿真；
- 得到“B Epoch10 明显优于对齐原始 DP，RL Epoch2 总体略升但困难集仍不稳定”的克制结论。

## 33. 证据文档索引

| 内容 | 文档 |
|---|---|
| B Epoch10 形成过程 | `HDP-nuplan/doc_hdp_nuplan/HDP_B_Epoch10训练过程详解.md` |
| RL Epoch2 形成过程 | `HDP-nuplan/doc_hdp_nuplan/HDP_RL_Epoch2训练过程详解.md` |
| 监督训练与 RL 总操作历史 | `HDP-nuplan/doc_hdp_nuplan/正式监督训练NPZ合并操作日志.md` |
| 论文方案与早期诊断结论 | 已合并在本文第 13～21 节；原始长日志可从 Git 历史查看 |
| 原始 DP 公平对照 | `HDP-nuplan/doc_hdp_nuplan/原始DiffusionPlanner同数据公平对照操作日志.md` |
| Test14 数据与正式评测 | `HDP-nuplan/doc_hdp_nuplan/Test14完整评测数据准备与三模型对比操作日志.md` |
| 三模型参数、YAML 与 hash | `HDP-nuplan/doc_hdp_nuplan/Test14三模型参数与配置完整清单.md` |
| TTC 退化案例 | `HDP-nuplan/doc_hdp_nuplan/RL_TTC退化场景分析_3713735b94cf5a7b.md` |
| 后续替代路线的拒绝依据 | 已合并在本文第 23～27 节；reference-relative、v7/v8 原始长日志可从 Git 历史查看 |

## 34. 最终总结

整个过程最核心的认识是：自动驾驶规划模型的“训练成功”至少有四层含义——代码能够运行、loss 能够下降、代理 reward 能够提高、官方闭环指标能够稳定改善。HDP-nuPlan 已完成前三级的大量验证，并在 533 个正式 Test14 场景的加权结果上获得 RL Epoch2 相对 B Epoch10 的小幅正收益；但困难集仍略有退化，因此最可信的结论不是“RL 已经稳定解决问题”，而是“已建立完整、可审计的训练与闭环评测链路，确认奖励加权后训练可以改变并改善部分规划行为，同时定位了其跨分布稳定性的关键瓶颈”。
