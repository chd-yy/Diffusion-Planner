# HDP-nuPlan 强化学习代码与公式完整报告

> 文档日期：2026-08-01<br>
> 分析对象：`/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan` 当前工作区<br>
> Git 基线：`7571a3baf54f182d9d980e9f559a6cfe9329085c`<br>
> 注意：当前工作区包含尚未提交的 RL、评估、测试和文档修改，因此本文描述的是**当前磁盘内容**，不只描述 Git 基线提交。

## 1. 结论先行

当前 HDP-nuPlan 已经实现了一条可以实际运行的强化学习式微调链路：

1. 从监督学习 checkpoint 初始化扩散规划器；
2. 对每个 NuPlan 缓存场景采样一组完整自车轨迹；
3. 用缓存中的真值邻车未来、路线和静态障碍物计算离线近似奖励；
4. 按场景保存候选轨迹和奖励到 replay buffer；
5. 在组内标准化奖励，使用指数优势权重重新拟合高奖励轨迹；
6. 更新模型、梯度裁剪并维护 EMA；
7. 使用 open-loop diffusion loss、固定 seed 离线 reward 和 NuPlan 官方 closed-loop 指标进行训练后评估。

不过，从算法性质上说，它不是 PPO、GRPO、SAC、DQN 或 Actor-Critic，而更准确地属于：

> **基于自生成候选轨迹和离线张量奖励的分组优势加权扩散微调**，也可以理解为 contextual-bandit 形式的 reward-weighted regression / self-imitation。

它借用了“同一场景内采样多个候选、组内标准化奖励”的思想，但没有标准 GRPO/PPO 中的策略概率比、旧策略、KL 约束和 clipped surrogate objective；也没有 critic、TD target、折扣回报或环境状态转移。

当前工程链路已经跑通，但策略质量仍处在 pilot 阶段：1000 场景短 RL 后，固定 seed 离线 reward 约提升 `0.11%`；3 个官方 non-reactive closed-loop 场景的平均 score 从 `0.297406` 变为 `0.297931`，责任碰撞合规比例仍为 `1/3`。因此现有结果证明的是“RL 软件链路可运行且没有明显崩溃”，不能证明已获得成熟的强化学习驾驶策略。

## 2. 阅读代码前需要了解的 Python / PyTorch 语法

### 2.1 `@dataclass`

`NuPlanRewardConfig` 使用 `@dataclass` 保存奖励权重和阈值。它自动生成构造函数，例如：

```python
config = NuPlanRewardConfig(
    collision_weight=10.0,
    comfort_weight=0.01,
)
```

未显式传入的字段继续使用类中的默认值。

### 2.2 张量维度与广播

RL 代码最重要的两个张量形状是：

- 候选轨迹：`[B, G, T, 4]`；
- 候选奖励：`[B, G]`。

其中 `B` 是场景 batch size，`G` 是同一场景的候选数，`T=80` 是未来时刻数，最后 4 维为 `[x, y, cos(yaw), sin(yaw)]`。

`reshape(B * G, T, 4)` 把场景维和候选维合并，便于把全部候选一次送入网络；`repeat_interleave(G, dim=0)` 则把每个场景条件连续复制 `G` 份，与展平后的候选一一对应。

### 2.3 `torch.no_grad()`、`detach()` 与 `.cpu()`

- `@torch.no_grad()`：rollout、奖励计算和 DPM-Solver 采样不构建反向传播图；
- `tensor.detach()`：切断张量和原计算图的联系；
- `.cpu()`：把 replay 中的候选和奖励移到主存，避免长期占用显存。

因此当前梯度只来自 update 阶段的扩散重建损失，不会穿过采样器或奖励函数直接反传。

### 2.4 `deque(maxlen=...)` 与有放回采样

Replay buffer 使用 `collections.deque(maxlen=max_size)`：容量满后再插入会自动淘汰最旧场景。`random.choices(..., k=batch_size)` 是有放回采样，同一场景在一个 update batch 中可能重复出现。

### 2.5 DDP、冻结参数与 EMA

- DDP 在多进程之间同步梯度；
- `requires_grad_(False)` 冻结顶层场景 encoder；
- `ModelEma` 保存参数的指数滑动平均，衰减率为 `0.999`。

冻结的是 `model.encoder`，不是所有名称中包含 `encoder` 的模块。Decoder 内部的 `route_encoder` 仍然属于 decoder，会正常更新。

## 3. 当前所有 RL 相关代码范围

### 3.1 直接实现 RL 的代码

| 路径 | 作用 |
|---|---|
| `train_predictor_rl.py` | RL 命令行入口、预训练加载、冻结、优化器、EMA、rollout/update 调度和 checkpoint |
| `hdp_nuplan/rl/reward.py` | 离线 NuPlan 张量奖励及各奖励分量 |
| `hdp_nuplan/rl/replay_buffer.py` | 按场景保存候选组和绝对奖励 |
| `hdp_nuplan/rl/loss.py` | 组内优势、指数权重、奖励加权扩散损失和 waypoint 混合损失 |
| `hdp_nuplan/rl/train_epoch_rl.py` | batch 适配、rollout epoch、replay 重载和 update epoch |
| `hdp_nuplan/rl/__init__.py` | 对外导出 replay buffer 和 reward scorer |
| `torch_run_rl.sh` | 单机 `torch.distributed.run` 启动模板 |

### 3.2 为 RL 提供采样、数据和扩散能力的代码

| 路径 | 与 RL 的关系 |
|---|---|
| `hdp_nuplan/model/hyper_diffusion_planner.py` | 提供显式、无梯度、可一次生成多候选的 `sample()` |
| `hdp_nuplan/model/module/decoder.py` | 复制条件、初始化随机噪声、调用 DPM-Solver、恢复物理轨迹 |
| `hdp_nuplan/model/diffusion_utils/sde.py` | VP-SDE 及 `score/x_start/noise/v` 参数化互转 |
| `hdp_nuplan/model/diffusion_utils/sampling.py` | 二阶 multistep DPM-Solver++ 采样 |
| `hdp_nuplan/utils/traj_kinematics.py` | waypoint 混合损失使用的积分函数 |
| `hdp_nuplan/utils/dataset.py` | rollout 返回场景文件名，update 按文件名重载条件 |
| `hdp_nuplan/utils/normalizer.py` | 动作和场景特征的归一化、反归一化 |
| `hdp_nuplan/loss.py` | 监督扩散损失；RL 损失沿用了相同的扩散参数化与 hybrid 设计 |

### 3.3 与 RL 实验验证直接相关的代码和产物

| 路径 | 作用 |
|---|---|
| `evaluate_predictor.py` | 固定 seed 计算监督/open-loop diffusion validation loss |
| `scripts/run_mini_closed_loop.sh` | 在固定 NuPlan 场景运行官方 closed-loop simulation |
| `scripts/summarize_closed_loop_metrics.py` | 汇总真实场景行、官方指标和运行耗时 |
| `hdp_nuplan/config/scenario_filter/mini-val-closed-loop-3.yaml` | 固定 3 个 mini-val 闭环场景 token |
| `tests/test_rl_components.py` | 碰撞奖励、replay round-trip、RL loss 梯度测试 |
| `tests/test_lr_schedule.py` | `warm_up_epoch=1` 时学习率不被错误缩小的回归测试 |
| `tests/test_closed_loop_summary.py` | 闭环聚合结果过滤和耗时合并测试 |
| `docs/mini_supervised_rl_operation_log.md` | 监督训练、RL pilot、前后测和闭环实验的原始操作记录 |

`data_processor.py` 还把 `log_name`、`scenario_type` 和 `token` 写入 NPZ。这些字段当前不进入 RL 模型或 replay item，但用于数据来源审计和从原始 DB 重建闭环场景。

## 4. 算法整体作用与项目位置

### 4.1 用 contextual bandit 表示当前问题

可把一个缓存场景记为上下文 \(c\)，把完整 8 秒自车轨迹记为动作

\[
\tau = \{(x_t,y_t,\cos\psi_t,\sin\psi_t)\}_{t=1}^{T}.
\]

扩散模型定义条件轨迹分布

\[
\tau \sim \pi_\theta(\tau\mid c).
\]

奖励函数给完整轨迹一个标量：

\[
R = R(c,\tau).
\]

每个场景一次性生成和评分完整轨迹，不执行

\[
s_t \rightarrow a_t \rightarrow r_t \rightarrow s_{t+1}
\]

形式的交互，所以当前训练更接近“一步 contextual bandit + 生成模型加权回归”，而不是多步 MDP 强化学习。

### 4.2 训练闭环

当前默认以 5 个 epoch 为一个周期：

- 周期第 1 个 epoch：清空 buffer，用当前在线模型 rollout；
- 后续 4 个 epoch：反复从同一 buffer 有放回采样并更新；
- 下一个周期重新 rollout，刷新候选与奖励。

代码判定为：

```python
is_rollout_epoch = epoch % args.rl_buffer_update_epoch == 0
```

默认 `rl_buffer_update_epoch=5`，因此以从 0 开始的 epoch 索引看，`0, 5, 10, ...` 是 rollout；其余是 update。

## 5. 数据与张量语义

### 5.1 Dataset 返回内容

`DiffusionPlannerData` 按固定顺序返回 11 个张量：

1. 当前自车状态；
2. 自车未来真值；
3. 邻车历史；
4. 邻车未来真值；
5. lane 特征；
6. lane 限速；
7. lane 是否有限速；
8. route lane 特征；
9. route 限速；
10. route 是否有限速；
11. 静态物体。

RL rollout 额外返回第 12 项 `file_name`。Replay buffer 不复制整套场景条件，只存这个文件名；update 时通过 `dataset.get_by_name(file_name)` 重新读取对应 NPZ。

### 5.2 模型输入与奖励输入分开

`prepare_nuplan_batch()` 同时返回：

- `model_inputs`：经过 `ObservationNormalizer` 的网络输入；
- `raw_inputs`：物理单位的原始张量，奖励函数使用其中的 route 和 static objects；
- `ego_future`、`neighbors_future`：把 yaw 角转换为 `cos/sin` 后的物理空间真值；
- `neighbor_mask`：未来前三维全零时视为 padding；
- `scene_names`：NPZ 文件名。

因此 reward 使用米、秒等物理尺度，网络条件使用归一化尺度，二者没有混用。

### 5.3 重要的 oracle future 边界

碰撞奖励使用缓存中的**邻车真值未来**，可选 imitation 奖励使用**自车真值未来**。这些未来信息只参与训练时的离线评分，不作为模型部署输入。

这是一种合法的离线监督/奖励构造方法，但必须明确：

- 它不是仿真中由其他交通参与者响应候选自车轨迹后产生的未来；
- 同一份邻车未来被用于评分同场景所有候选；
- 模型部署时只能看到历史、当前状态、地图和路线。

## 6. 分组扩散 rollout

### 6.1 条件只编码一次

顶层 `sample()` 先把模型临时切换到 eval 模式：

```python
was_training = self.training
self.eval()
encoder_outputs = self.encoder(inputs)
trajectories = self.decoder.decoder.sample(
    encoder_outputs,
    inputs,
    num_samples=G,
    diffusion_steps=K,
)
self.train(was_training)
```

场景 encoder 只计算一次 `[B, ...]` 条件表示，decoder 再对 encoding、route 和当前自车速度执行 `repeat_interleave(G)`。这比对同一场景完整运行 `G` 次 encoder 更省计算。

### 6.2 候选多样性的来源

复制后的条件完全相同，候选差异只来自独立初始噪声：

```python
xT = torch.randn(B * G, T, 4) * 0.1
```

即当前实际使用

\[
x_T \sim \mathcal N(0, 0.01I),
\]

然后调用 DPM-Solver++：

- `algorithm_type="dpmsolver++"`；
- 二阶 `order=2`；
- `logSNR` 跳步；
- `multistep`；
- `denoise_to_zero=True`；
- rollout 默认只使用 5 个求解步。

标准 VP 扩散的终端先验通常是 \(\mathcal N(0,I)\)，而这里显式乘了 `0.1`。这是当前代码的非标准选择，也会直接缩小候选多样性；它应当作为后续消融项，而不能默认认为与标准 DPM 采样完全等价。

### 6.3 从扩散动作恢复物理轨迹

Decoder 输出前两维是逐帧位移，反归一化后执行：

\[
x_t=\sum_{i=1}^{t}\Delta x_i,\qquad
y_t=\sum_{i=1}^{t}\Delta y_i.
\]

最后恢复为 `[B,G,T,4]`。后两维由模型直接输出为 `cos/sin` 表示，但代码没有再次投影到单位圆。

奖励函数只使用生成轨迹的 `x/y`，当前 rollout 奖励完全不检查生成的 `cos/sin` 是否归一化或航向是否合理；航向只在部署时通过 `atan2(sin, cos)` 转回角度。

## 7. 离线奖励的完整公式与代码语义

记第 \(b\) 个场景、第 \(g\) 个候选的二维轨迹为 \(p_{b,g,t}\)。默认奖励为：

\[
R_{b,g}
=w_p P_{b,g}
-w_c C^{\mathrm{collision}}_{b,g}
-w_r C^{\mathrm{route}}_{b,g}
-w_f C^{\mathrm{comfort}}_{b,g}
-w_b C^{\mathrm{backward}}_{b,g}
-w_i C^{\mathrm{imitation}}_{b,g}.
\]

默认权重为：

\[
\( (w_p,w_c,w_r,w_f,w_b,w_i)=(1,10,1,0.1,1,0) \)。
\]

1000 场景 pilot 把 comfort 权重改为 `0.01`。

### 7.1 Progress reward

NuPlan 缓存以当前自车为局部原点，局部 \(x\) 轴通常指向当前车头方向。代码使用：

\[
P_{b,g}=\frac{x_{b,g,T}}{10}.
\]

这不是沿 route arc-length 的真实进度，只是终点局部纵向坐标的缩放。直道上直观有效，但转弯、U-turn 或局部坐标与路线方向偏离时可能失真。

### 7.2 动态与静态碰撞代价

对有效邻车真值未来 \(q_{b,n,t}\)，计算时间对齐的中心点距离：

\[
d^{\mathrm{dyn}}_{b,g}
=\min_{n,t:\,\mathrm{valid}(n,t)}
\lVert p_{b,g,t}-q_{b,n,t}\rVert_2.
\]

对静态物体中心 \(o_{b,m}\)：

\[
d^{\mathrm{static}}_{b,g}
=\min_{m,t:\,\mathrm{valid}(m)}
\lVert p_{b,g,t}-o_{b,m}\rVert_2.
\]

取

\[
d^{\min}_{b,g}=\min(d^{\mathrm{dyn}}_{b,g},d^{\mathrm{static}}_{b,g}),
\]

再计算安全距离内的线性惩罚：

\[
C^{\mathrm{collision}}_{b,g}
=\frac{\max(0,d_{\mathrm{safe}}-d^{\min}_{b,g})}{d_{\mathrm{safe}}},
\qquad d_{\mathrm{safe}}=2.5\text{ m}.
\]

日志中的 `no_collision` 只是指标：

\[
I^{\mathrm{no\ collision}}_{b,g}
=\mathbf 1[d^{\min}_{b,g}\ge d_{\mathrm{safe}}],
\]

它没有作为独立项再次加入 reward。

实现限制：

- 使用点中心距离，不使用 ego/agent 的矩形包围盒和朝向；
- 不做连续时间 swept collision，只比较离散的 0.1 秒采样点；
- 动态邻车采用缓存真值未来，不会响应候选 ego 行为；
- 只取数据集切出的前 `predicted_neighbor_num=10` 个邻车未来；
- 整条轨迹只取一个最小距离，不能区分一次短暂接近和长时间危险接近；
- 全零 padding 判断会把坐标和 yaw 都恰好为零的条目视为无效。

### 7.3 Route deviation cost

每个场景把所有有效 route lane 点展平为集合 \(\mathcal R_b\)。对每个候选时刻求最近路线点距离：

\[
d^{\mathrm{route}}_{b,g,t}
=\min_{r\in\mathcal R_b}\lVert p_{b,g,t}-r\rVert_2.
\]

代价为：

\[
C^{\mathrm{route}}_{b,g}
=\frac{1}{T}\sum_{t=1}^{T}
\frac{\min(d^{\mathrm{route}}_{b,g,t},d_{\max})}{d_{\max}},
\qquad d_{\max}=5\text{ m}.
\]

它衡量“离 route 点近不近”，但不检查路线方向、lane width、边界、红绿灯、限速或 roadblock 拓扑顺序。

### 7.4 Comfort cost

使用有限差分，其中 \(\Delta t=0.1\) 秒：

\[
v_t=\frac{p_{t+1}-p_t}{\Delta t},\qquad
a_t=\frac{v_{t+1}-v_t}{\Delta t},\qquad
j_t=\frac{a_{t+1}-a_t}{\Delta t}.
\]

加速度和 jerk 惩罚分别是：

\[
C_a=\operatorname{mean}_t
\frac{\max(0,\lVert a_t\rVert_2-a_{\max})}{a_{\max}},
\quad a_{\max}=4\text{ m/s}^2,
\]

\[
C_j=\operatorname{mean}_t
\frac{\max(0,\lVert j_t\rVert_2-j_{\max})}{j_{\max}},
\quad j_{\max}=8\text{ m/s}^3,
\]

\[
C^{\mathrm{comfort}}=C_a+C_j.
\]

这里对二维加速度/jerk 取欧氏范数，没有区分纵向、横向和 yaw dynamics，也没有使用官方 NuPlan comfort metric 的完整动力学判定。

### 7.5 Backward cost

对相邻预测位置的局部 \(x\) 增量惩罚倒车：

\[
C^{\mathrm{backward}}_{b,g}
=\operatorname{mean}_{t=1}^{T-1}\max(0,-(x_{t+1}-x_t)).
\]

代码没有把原点 prepend 到位置序列，因此从当前原点到第一个预测点的倒退不进入该项。

### 7.6 Imitation cost

如果传入 expert ego future \(p^*_{b,t}\)：

\[
C^{\mathrm{imitation}}_{b,g}
=\operatorname{mean}_{t}\lVert p_{b,g,t}-p^*_{b,t}\rVert_2.
\]

默认 `imitation_weight=0`，所以当前默认 reward 不受它影响，但 scorer 仍计算并记录该指标。若权重大于零，训练就会同时包含显式专家模仿偏好。

### 7.7 1000 场景 pilot 的 reward 数值核对

1000 场景 rollout 使用 `comfort_weight=0.01`，平均分量为：

```text
progress       = 2.70126349
collision cost = 0.09572496
route cost     = 0.21621123
comfort cost   = 23.16818842
backward cost  = 0.00188407
```

代入公式：

\[
2.70126349
-10\times0.09572496
-0.21621123
-0.01\times23.16818842
-0.00188407
\approx1.29423668,
\]

与日志完全一致。这也说明 reward 日志中的各 cost 是未乘权重的原始分量。

## 8. Replay buffer 的真实语义

### 8.1 保存内容

每个 `NuPlanReplayItem` 保存：

```python
scene_name: str
trajectories: Tensor[G, T, 4]  # CPU
rewards: Tensor[G]             # CPU
```

它不保存 `(state, action, reward, next_state, done)`，所以和 DQN/SAC 中的 transition replay 不同。它保存的是一个场景条件下的整组完整轨迹及其标量回报。

### 8.2 采样和重载

Update 先对场景 item 做均匀有放回采样，再按 `scene_name` 重载场景条件。每个被采到的场景会带回完整的 `G` 个候选，候选本身不做 prioritized replay；候选偏好只由后面的 reward weight 实现。

### 8.3 buffer 生命周期

每个 rollout epoch 开始前调用 `clear()`。因此默认行为是：

- rollout 一次，得到当前策略下的候选；
- 连续 4 个 update epoch 重复使用这些固定候选和固定奖励；
- 随模型更新，buffer 相对于最新模型逐渐变旧；
- 第 5 个 epoch 后重新 rollout，清除旧数据。

Buffer 容量不足时，`deque` 保留遍历顺序中最后写入的场景，而不是随机保留全数据代表性子集。

## 9. 奖励加权扩散目标的完整推导

### 9.1 从位置轨迹转换为模型动作

Rollout 返回绝对局部位置轨迹。训练时先转换为：

\[
a_t=[\Delta x_t,\Delta y_t,\cos\psi_t,\sin\psi_t],
\]

其中

\[
\Delta p_1=p_1-0,\qquad
\Delta p_t=p_t-p_{t-1},\ t>1.
\]

代码等价于：

```python
displacement = torch.diff(
    torch.cat([zeros, trajectory_xy], dim=-2),
    dim=-2,
)
action = torch.cat([displacement, trajectory_heading_vector], dim=-1)
```

随后用 state normalizer：

\[
\bar a=\frac{a-\mu}{s},
\]

当前 ego 配置为：

\[
\mu=[0,0,0,0],\qquad
s=[0.5,0.5,1,1].
\]

### 9.2 VP-SDE 前向加噪

当前使用线性 VP-SDE：

\[
\mathrm dx=-\frac{1}{2}\beta(t)x\,\mathrm dt+\sqrt{\beta(t)}\,\mathrm dW_t,
\]

\[
\beta(t)=\beta_{\min}+(\beta_{\max}-\beta_{\min})t,
\quad \beta_{\min}=0.1,\ \beta_{\max}=20.
\]

其边缘分布可写为：

\[
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\]

\[
\alpha_t=
\exp\left[-\frac14(\beta_{\max}-\beta_{\min})t^2
-\frac12\beta_{\min}t\right],
\]

\[
\sigma_t=\sqrt{1-\alpha_t^2}.
\]

RL loss 对每个展平后的候选独立采样：

\[
t\sim U(10^{-3},1),\qquad
\epsilon\sim\mathcal N(0,I).
\]

### 9.3 四种扩散参数化

网络原始输出类型 `model_type` 和计算损失的 `supervision_type` 可以分别选择：

- `x_start`：预测干净动作 \(x_0\)；
- `noise`：预测噪声 \(\epsilon\)；
- `score`：预测 \(\nabla_{x_t}\log p_t(x_t)\)；
- `v`：预测 velocity parameterization。

代码先把任意源参数化转换成噪声：

\[
\epsilon_{\mathrm{score}}=-\sigma_t s_\theta,
\]

\[
\epsilon_{x_0}=\frac{x_t-\alpha_t\hat x_0}{\sigma_t},
\]

\[
\epsilon_v=\sigma_t x_t+\alpha_t v_\theta.
\]

再从噪声转换为目标参数化：

\[
s=-\frac{\epsilon}{\sigma_t},\qquad
x_0=\frac{x_t-\sigma_t\epsilon}{\alpha_t},
\]

\[
v=\frac{\epsilon-\sigma_t x_t}{\alpha_t}
=\alpha_t\epsilon-\sigma_t x_0.
\]

最后一个等式使用了 \(x_t=\alpha_t x_0+\sigma_t\epsilon\) 和 \(\alpha_t^2+\sigma_t^2=1\)。代码在分母加入 `1e-6` 保持数值稳定。

### 9.4 单候选 diffusion loss

对候选 \(j=(b,g)\)，四种监督空间对应：

\[
\ell^{\mathrm{score}}_j
=\frac1T\sum_t
\lVert \sigma_t\hat s_{j,t}+\epsilon_{j,t}\rVert_2^2,
\]

\[
\ell^{x_0}_j
=\frac1T\sum_t\lVert \hat x_{0,j,t}-x_{0,j,t}\rVert_2^2,
\]

\[
\ell^{\mathrm{noise}}_j
=\frac1T\sum_t\lVert \hat\epsilon_{j,t}-\epsilon_{j,t}\rVert_2^2,
\]

\[
\ell^v_j
=\frac1T\sum_t\lVert \hat v_{j,t}-v_{j,t}\rVert_2^2.
\]

最后一维 4 个状态分量先求和，时间维再求均值。

### 9.5 组内优势标准化

对同一场景的 \(G\) 个候选：

\[
\mu_b=\frac1G\sum_{g=1}^{G}R_{b,g},
\]

\[
\sigma_b=
\sqrt{\frac1G\sum_{g=1}^{G}(R_{b,g}-\mu_b)^2},
\]

这里使用 population standard deviation，即 PyTorch 的 `unbiased=False`。

优势为：

\[
A_{b,g}
=\operatorname{clip}\left(
\frac{R_{b,g}-\mu_b}{\sigma_b+10^{-6}},
-c,c
\right),
\quad c=5.
\]

指数权重为：

\[
w_{b,g}=\exp(\kappa A_{b,g}),
\quad \kappa=\texttt{rl\_reward\_temperature}.
\]

代码中的 `temperature` 实际是逆温度式系数：它越大，候选权重差距越大；这与某些文献使用 `exp(A / temperature)` 的命名相反。

### 9.6 最终 reward-weighted loss

扩散项为：

\[
\mathcal L_{\mathrm{diff}}
=\frac{1}{BG}\sum_{b=1}^{B}\sum_{g=1}^{G}
w_{b,g}\ell_{b,g}.
\]

注意代码没有除以 \(\sum w\)。令

\[
Z_b=\frac1G\sum_g w_{b,g},\qquad
\tilde w_{b,g}=\frac{w_{b,g}}{\sum_k w_{b,k}},
\]

则每个场景的目标可以改写为：

\[
\frac1G\sum_gw_{b,g}\ell_{b,g}
=Z_b\sum_g\tilde w_{b,g}\ell_{b,g}.
\]

也就是说，归一化权重决定场景内“更像谁”，而 \(Z_b\) 还会改变该场景的整体梯度尺度。1000 场景 pilot 最后一轮记录的 `weight_mean=1.5679`，说明非归一化权重确实放大了平均 loss/gradient 尺度。

在未裁剪且组内优势均值为零时，由 Jensen 不等式：

\[
\frac1G\sum_g e^{\kappa A_g}
\ge e^{\kappa\operatorname{mean}(A)}=1.
\]

所以 `weight_mean` 通常不小于 1，而不是天然保持单位尺度。

### 9.7 该目标如何偏向高奖励轨迹

高奖励候选有 \(A>0,w>1\)，其扩散去噪误差被放大；低奖励候选有 \(A<0,0<w<1\)，影响被削弱。所有权重始终为正，所以当前方法不会对低奖励轨迹施加“负梯度排斥”，只是减少对它们的模仿强度。

若把 diffusion loss 看作条件负对数似然的代理，当前目标近似：

\[
\min_\theta
\mathbb E_{c,\tau\sim q}
\left[e^{\kappa A(c,\tau)}
\left(-\log\pi_\theta(\tau\mid c)\right)\right],
\]

其中 \(q\) 是生成 replay 数据时的旧模型分布，\(\kappa\) 是代码中的 temperature 系数。它相当于把模型投影到一个“对高奖励样本指数倾斜”的经验分布上。

这与 REINFORCE/PPO 的关键差别是：代码没有计算当前轨迹的 `log_prob` 或新旧策略概率比，而是把生成轨迹重新当作扩散训练的 \(x_0\) 目标。

### 9.8 几个重要特殊情况

1. 若同组奖励完全相同，则 \(A=0,w=1\)，RL update 退化为对自身采样轨迹的普通扩散重建。
2. 对奖励整体加常数，组内标准化后权重基本不变。
3. 对奖励乘正数，若不考虑 `1e-6` 和裁剪，权重也基本不变。
4. 因此奖励各分量的主要作用是改变候选**相对排序和组内间距**，不是改变绝对 reward 均值。
5. `G=2` 虽然可计算优势，但方差估计和排序很不稳定；代码只检查 `G>=2`。

## 10. Waypoint hybrid loss 与积分梯度

### 10.1 设计意图

网络主要拟合逐帧位移。为了约束长期积分后的绝对位置，代码把预测统一转换到 `x_start`，反归一化并积分：

\[
\hat p_t=\sum_{i=1}^{t}\widehat{\Delta p_i}.
\]

候选 waypoint loss 为：

\[
\ell^{\mathrm{wp}}_{b,g}
=\frac{1}{2T}\sum_{t=1}^{T}
\lVert\hat p_{b,g,t}-p_{b,g,t}\rVert_2^2.
\]

同样使用 reward weight：

\[
\mathcal L_{\mathrm{wp}}
=\frac1{BG}\sum_{b,g}w_{b,g}\ell^{\mathrm{wp}}_{b,g}.
\]

总损失：

\[
\mathcal L_{\mathrm{RL}}
=\mathcal L_{\mathrm{diff}}
+\lambda_{\mathrm{wp}}\mathcal L_{\mathrm{wp}},
\qquad \lambda_{\mathrm{wp}}=0.01.
\]

### 10.2 注释描述的截断梯度意图

`detached_integral(u, W)` 的注释希望保持完整前向累计值，但仅让最近 \(W\) 个位移接收梯度：

\[
\frac{\partial p_t}{\partial u_i}
=
\begin{cases}
1,&\max(0,t-W+1)\le i\le t,\\
0,&\text{其他}.
\end{cases}
\]

这能在不改变轨迹数值的前提下截断长时域反向依赖。

### 10.3 当前代码的实际行为与注释不一致

当前实现对 `[N,T,D]` 张量使用：

```python
shifted[:, :, :detach_window_size] = 0
cum_detach_shifted[:, :, :detach_window_size] = 0
```

`torch.roll(..., dims=-2)` 滚动的是时间维，但上述切片清零的是最后一个特征维，正确表达“前 W 个时刻”应当切时间维。

RL waypoint 输入最后一维 `D=2`，默认 `W=10`，所以 `:10` 会覆盖两个坐标特征并把整个 `shifted` 清零。当前默认配置实际退化为：

\[
\hat p=\operatorname{cumsum}(u),
\]

并且最终 waypoint 对所有过去位移都有梯度，而不是只保留最近 10 步。

已用当前函数做自动求导复核：对 6 步二维输入、`W=3`，最后 waypoint 对全部 6 步的梯度均为 `[1,1]`，且输出与普通 `torch.cumsum` 完全相同。

因此当前报告必须区分：

- **算法设计意图**：截断窗口积分；
- **当前有效实现**：默认二维输入下的完整累计和、完整历史梯度。

本文只记录该问题，没有修改源代码。正式扩大训练前应修正切片维度并增加梯度窗口单元测试。

## 11. Rollout 和 update 的代码级流程

### 11.1 Rollout epoch

核心伪代码为：

```python
for batch in data_loader:
    model_inputs, raw_inputs, ego_gt, neighbor_gt, mask, names = prepare(batch)

    trajectories = model.sample(
        model_inputs,
        num_samples=group_size,
        diffusion_steps=rollout_steps,
    )

    rewards, details = scorer(
        trajectories,
        neighbor_gt,
        mask,
        raw_inputs["route_lanes"],
        raw_inputs["static_objects"],
        ego_gt,
    )

    for scene_name, group, group_rewards in zip(names, trajectories, rewards):
        replay_buffer.put(scene_name, group, group_rewards)
```

整个函数和 scorer 都在 `no_grad` 下运行。每个场景候选保持为一个组写入 buffer，组内关系不会被打散。

### 11.2 Update epoch

核心流程为：

```python
for loader_batch in data_loader:
    batch_size = loader_batch[0].shape[0]
    replay_items = replay_buffer.sample(batch_size)
    replay_batch, trajectories, rewards = reload_by_scene_name(replay_items)
    model_inputs = prepare(replay_batch)

    optimizer.zero_grad(set_to_none=True)
    loss, metrics = reward_weighted_diffusion_loss(...)
    loss.backward()
    clip_grad_norm_(trainable_parameters, rl_grad_clip)
    optimizer.step()
    ema.update(model)
```

原始 `loader_batch` 的场景数据没有参与更新，只用来决定：

- 一个 epoch 有多少次 optimizer step；
- 当前 step 的 batch size，特别是最后一个不满 batch。

随后代码又从 replay 重载另一批场景。因此 update 阶段存在不必要的 Dataset I/O，可以改为显式 `num_updates_per_epoch` 和 replay batch size。

## 12. 训练入口、优化器、EMA 与 checkpoint

### 12.1 预训练模型加载

加载优先级：

1. `ema_state_dict`；
2. `model`；
3. 把整个 checkpoint 当 state dict。

并去掉 `module.` 前缀。但 `_load_pretrained()` 使用 `strict=False`，只打印 missing/unexpected key 数量，不会在架构不兼容时停止。现有 pilot 另行做了 strict audit，结果为 0/0；通用入口本身仍有静默部分加载风险。

### 12.2 默认冻结 encoder

`rl_freeze_encoder=true` 时只训练 decoder。1000 场景 pilot 的 checkpoint 审计结果：

- `module.encoder.*` 共 151 个张量，变化数为 0；
- `module.decoder.*` 共 109 个张量，109 个均发生变化；
- decoder 最大绝对参数变化约 `0.00350105`。

这证明本次冻结逻辑按预期工作。

### 12.3 优化器和梯度

当前使用：

```python
optimizer = AdamW(trainable_parameters, lr=learning_rate)
clip_grad_norm_(trainable_parameters, max_norm=rl_grad_clip)
```

`AdamW` 未显式设置 `weight_decay`，因此使用 PyTorch 默认值 `0.01`。梯度裁剪默认最大范数为 `5.0`。

### 12.4 学习率调度

函数名为 `CosineAnnealingWarmUpRestarts`，但当前实现并没有 cosine annealing 或 restart：

- `warm_up_epoch>1`：线性 warm-up 后固定学习率；
- `warm_up_epoch<=1`：恒等 `MultiplicativeLR`，保持基础学习率。

Scheduler 只在 update epoch 调用 `step()`，但构造时的总 epoch 参数是 `train_epochs`，不是实际 update epoch 数。默认 `warm_up_epoch=5` 时，需要按“只在 update 时计数”的实际语义理解 warm-up。

### 12.5 EMA 与部署

EMA 衰减为：

\[
\theta^{\mathrm{EMA}}
\leftarrow0.999\theta^{\mathrm{EMA}}+0.001\theta.
\]

每个 update batch 后更新一次。Rollout 使用当前在线模型，不使用 EMA；新的 RL 或闭环评估默认优先加载 checkpoint 中的 `ema_state_dict`。

### 12.6 Checkpoint 保存条件

只有同时满足以下条件才保存：

```python
not is_rollout_epoch and (epoch + 1) % save_utd == 0
```

因此：

- rollout epoch 不保存；
- 若 `save_utd` 与 update epoch 排列不匹配，可能很久没有 checkpoint；
- 若误设 `rl_buffer_update_epoch=1`，每个 epoch 都是 rollout，将永远不 update、不保存；
- 当前入口只校验 `group_size>=2`，没有校验 buffer update 周期、buffer size、temperature、clip 等边界。

## 13. DDP 下的实际行为

多 GPU 时，每个 rank：

1. 由 `DistributedSampler` 获得不同场景子集；
2. 建立自己的进程内 replay buffer；
3. 只在本 rank 的 buffer 中采样；
4. 计算本地 reward-weighted loss；
5. DDP 在 backward 时平均各 rank 梯度。

当前没有：

- 在 rank 之间汇总或交换 replay item；
- 对 rollout/update 日志指标做 `all_reduce`；
- 把所有 rank 的候选组成全局组。

因此 rank 0 打印的 reward、buffer size 等是 rank 0 的本地统计，不是全局统计；`rl_buffer_size` 也是每 rank 容量。梯度仍会经 DDP 同步，所以优化使用各 rank 本地 batch 的联合平均。

## 14. 训练参数完整说明

### 14.1 RL 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `rl_group_size` | 8 | 每场景候选数 \(G\)，至少为 2 |
| `rl_rollout_steps` | 5 | DPM-Solver 采样步数 |
| `rl_buffer_update_epoch` | 5 | 每隔多少 epoch 重做 rollout |
| `rl_buffer_size` | 4096 | 每 rank 最大场景 item 数 |
| `rl_reward_temperature` | 1.0 | 指数优势系数，越大权重越尖锐 |
| `rl_advantage_clip` | 5.0 | 标准化优势裁剪范围 |
| `rl_detach_window_size` | 10 | hybrid 积分意图上的梯度窗口；当前默认实现未实际截断 |
| `rl_grad_clip` | 5.0 | 可训练参数的梯度范数上限 |
| `rl_freeze_encoder` | true | 是否冻结顶层场景 encoder |

### 14.2 可从 CLI 配置的 reward 参数

| 参数 | 默认值 |
|---|---:|
| `reward_progress_weight` | 1.0 |
| `reward_collision_weight` | 10.0 |
| `reward_route_weight` | 1.0 |
| `reward_comfort_weight` | 0.1 |
| `reward_backward_weight` | 1.0 |
| `reward_imitation_weight` | 0.0 |
| `reward_collision_distance` | 2.5 m |

### 14.3 RewardConfig 中存在但 CLI 当前不能修改的参数

| 参数 | 默认值 |
|---|---:|
| `dt` | 0.1 s |
| `max_route_distance` | 5.0 m |
| `acceleration_limit` | 4.0 m/s² |
| `jerk_limit` | 8.0 m/s³ |

若要调后三项，目前必须改代码或扩展命令行参数；它们不会从 `train_predictor_rl.py` 自动透传。

## 15. 当前方法与常见 RL 算法的区别

| 能力 | 当前 HDP-nuPlan | PPO / GRPO 类 | Actor-Critic / SAC 类 |
|---|---|---|---|
| 同场景多候选 | 有 | 常见 | 可有 |
| 组内奖励标准化 | 有 | GRPO 常见 | 非核心 |
| 策略 log-prob | 无 | 有 | 有 |
| 新旧策略概率比 | 无 | 有 | 通常不采用 PPO 比率 |
| clipped surrogate | 无 | PPO/GRPO 常见 | 无 |
| reference policy / KL | 无 | GRPO/RLHF 常见 | 通常无 |
| critic / value network | 无 | PPO 通常有，GRPO可无 | 有 |
| TD target / Bellman backup | 无 | 无 | 有 |
| 折扣因子 \(\gamma\) | 无 | 多步任务常有 | 有 |
| 环境闭环交互训练 | 无 | 通常有或基于轨迹数据 | 通常有 |
| 更新形式 | 奖励加权扩散回归 | policy-gradient surrogate | actor/critic objectives |

最准确的表述不是“实现了 GRPO”，而是：

> 使用 GRPO 风格的组内奖励标准化，构造指数优势权重，对扩散模型进行 reward-weighted behavior regression。

## 16. 训练部署之间的差别

训练 rollout 每个场景生成 `G` 个候选并全部评分，但 NuPlan planner 部署接口执行：

```python
"prediction": self.sample(..., num_samples=1)
```

即部署时：

- 只生成一个随机候选；
- 不调用训练用 tensor reward scorer；
- 不在多个候选中选择最高分轨迹；
- 每个 planning iteration 重新进行随机扩散采样。

因此 RL 的作用是改变整个条件生成分布，使单次抽样更偏向训练时的高奖励区域，而不是在部署时做 best-of-N reranking。

这也意味着训练中的 group size 提升不会自动转化成部署时的候选选择能力。如果希望 best-of-N，需要另行设计在线候选 scorer、确定性/随机性控制和实时计算预算。

## 17. 已完成实验与可得结论

### 17.1 100 场景 RL smoke

链路为：监督 checkpoint → group rollout → reward → buffer → update → RL checkpoint。

该实验通过了软件和梯度链路，但监督起点只训练 100 场景、2 epoch；rollout 仅 2 个采样步、4 个候选，reward 约为 `-14.52`，其中 comfort cost 约 `105.9`，在 `0.1` 权重下主导总奖励。因此它只有工程冒烟价值。

### 17.2 1000 场景 RL pilot

配置要点：

- 监督起点：固定 1000 场景、10 epoch 的 LR 修复版；
- group size：4；
- DPM steps：5；
- buffer：1024，可容纳 1000 场景；
- comfort weight：0.01；
- 学习率：真实 `1e-5`；
- encoder 冻结；
- epoch 1 rollout，epoch 2--5 update。

RL update loss：

| Epoch | loss |
|---:|---:|
| 2 | 0.01809396 |
| 3 | 0.00833132 |
| 4 | 0.00672711 |
| 5 | 0.00594997 |

Loss 下降只能说明模型越来越能重建 buffer 中按 reward 加权的自生成候选，不能单独证明 reward 或闭环性能上升。

### 17.3 独立 mini-val diffusion loss

在相同 100 个 val 场景和 3 个固定 seed 下：

| 指标 | RL 前 | RL 后 | 相对变化 |
|---|---:|---:|---:|
| ego planning loss | 0.238385 | 0.231356 | -2.95% |
| hybrid loss | 38.430509 | 36.227327 | -5.73% |
| 总损失 | 0.622690 | 0.593629 | -4.67% |

说明短 RL 没有造成监督目标灾难性退化，并略有改善；它不是闭环驾驶指标。

### 17.4 固定 seed 离线 post-rollout

| 指标 | RL 前 | RL 后 |
|---|---:|---:|
| reward | 1.29423668 | 1.29568970 |
| progress | 2.70126349 | 2.70512732 |
| no-collision | 0.749750 | 0.755000 |
| collision cost | 0.09572496 | 0.09562183 |
| route cost | 0.21621123 | 0.21645447 |
| comfort cost | 23.16818842 | 23.45995709 |
| backward cost | 0.00188407 | 0.00216523 |
| imitation cost | 3.90034562 | 3.84941584 |

完整离线平均 reward 增加 `0.00145303`，约 `0.11%`。这是单 seed 的微小变化，没有统计显著性结论。

### 17.5 NuPlan 官方 closed-loop 对照

统一使用：

- `closed_loop_nonreactive_agents`；
- 相同 3 个 mini-val token；
- sequential worker；
- simulation seed 0；
- 官方 metrics 和 weighted aggregator。

| 模型 | 官方 score | 无责任碰撞 | route progress | TTC 合规 | comfort |
|---|---:|---:|---:|---:|---:|
| 发布版 Diffusion-Planner | 0.995829 | 1.000000 | 0.986696 | 1.000000 | 1.000000 |
| HDP 监督 pilot | 0.297406 | 0.333333 | 0.755465 | 0.333333 | 1.000000 |
| HDP-RL pilot | 0.297931 | 0.333333 | 0.758802 | 0.333333 | 1.000000 |

HDP-RL 相对 HDP 监督：

- score 绝对增加约 `0.00052569`，相对约 `0.1768%`；
- route progress 增加约 `0.00333681`；
- 责任碰撞和 TTC 合规场景比例没有变化；
- stationary 和 high-speed 场景仍因责任碰撞得到 0 分；
- traffic-light 场景从 `0.892217` 微升至 `0.893794`。

发布版 Diffusion-Planner checkpoint 训练更充分，和仅 1000 场景的 HDP pilot 不是公平架构对比；它在这里主要用于证明数据库、地图、controller、指标和仿真环境正常。

3 个场景样本量也远不足以做统计结论，但已明确指出：当前主要问题是碰撞安全，不是软件链路或 diffusion validation loss。

## 18. 当前测试覆盖

### 18.1 已覆盖

1. 碰撞候选的 collision cost 高于横向偏移候选，reward 更低；
2. Replay item 可保存和有放回读出，形状保持 `[G,T,4]`；
3. Reward-weighted diffusion loss 为有限值，并能向 dummy model 参数传播有限梯度；
4. `warm_up_epoch=1` 不再把基础学习率永久乘以 0.1；
5. 闭环汇总器忽略 `final_score` 聚合行，只统计真实场景并合并 runtime。

### 18.2 尚未覆盖

- 每个 reward 分量的边界值和无有效障碍/路线情况；
- 组内优势在零方差、极大 temperature、clip 边界下的行为；
- 权重是否需要归一化的数值对照；
- `waypoint_to_model_action()` 与 decoder `cumsum()` 的严格逆变换；
- `detached_integral()` 的实际梯度窗口；
- 多 rank buffer/metrics 行为；
- checkpoint 保存周期的边界配置；
- 采样 `0.1` 噪声尺度和 group 多样性；
- 生成 `cos/sin` 的单位圆约束；
- reward 与官方 closed-loop metrics 的相关性。

## 19. 当前局限与风险，按优先级排序

### P0：当前监督起点和闭环安全性不足

3 个官方闭环场景中，HDP 监督和 RL 都只有 `1/3` 场景满足无责任碰撞和 TTC。继续在同一低质量起点上增加 RL epoch，可能只会更强地拟合当前有限候选，而不能解决模型分布覆盖和安全问题。

建议先把监督训练扩到更完整的数据和验证集，再以 closed-loop collision 指标作为进入下一轮 RL 的门槛。

### P0：训练 reward 与官方闭环指标不等价

当前 collision 是中心点距离，route 是最近点距离，comfort 是简单有限差分；缺少车辆外形、交互响应、交通规则、官方碰撞责任、TTC、drivable area、speed limit 和真实 route progress。

离线 no-collision 从 `0.74975` 变为 `0.755`，但 3 场景官方无责任碰撞比例仍是 `1/3`，已经说明两者不能互换。

### P0：`detached_integral` 没有按默认配置截断梯度

这是已用自动求导复核的实现问题。它不会改变前向轨迹数值，但会改变 hybrid loss 的反向传播范围、显存和梯度尺度。修复后训练动力学会变化，应重新验证监督和 RL checkpoint，不能直接假设旧结果等价。

### P1：候选多样性可能不足

候选只由初始随机噪声区分，且噪声标准差被缩小到 `0.1`，rollout 只有 5 个 DPM step。若组内 reward 很接近，标准化优势主要放大采样噪声和 scorer 误差。

应记录组内终点方差、轨迹 ADE pairwise distance、reward range、advantage/weight 分布，并消融 `xT` 噪声尺度、steps 和 group size。

### P1：指数权重未归一化

当前 `weight_mean` 会改变有效学习率，temperature 同时影响“偏好强度”和“整体梯度尺度”。建议至少对比：

\[
\frac{\sum_gw_g\ell_g}{\sum_gw_g}
\quad\text{与}\quad
\frac1G\sum_gw_g\ell_g,
\]

并监控每组 `weight_max/weight_min/effective_sample_size`。

### P1：同一 replay 数据连续更新且 reward 不刷新

默认 4 个 update epoch 使用相同候选和奖励，策略逐渐偏离生成这些轨迹的旧模型。当前没有 KL、trust region 或 reference model 约束来限制漂移。

可缩短 update-to-rollout 比例、增加 KL/行为保持项，或在每轮 update 后重新生成部分候选。

### P1：部署只采一个候选

训练依靠组间比较获得信号，但部署没有 best-of-N scorer。若训练只轻微移动生成分布，单样本收益可能很小，当前闭环结果正符合这种现象。

### P2：DDP 日志和 buffer 是 rank-local

多 GPU 时 rank 0 日志不能直接代表全局平均，buffer size 也不能理解为全局场景数。应对标量做 sample-weighted all-reduce，或明确日志键名为 `rank0/*`。

### P2：若干配置/工程边界

- `_load_pretrained(strict=False)` 可能部分加载不兼容 checkpoint；
- update 通过遍历 Dataset 只获得 step 数和 batch size，产生重复 I/O；
- scheduler 名称与实际算法不符；
- checkpoint 只在满足 cadence 的 update epoch 保存；
- buffer 容量不足时保留遍历末尾场景，可能产生采样偏差；
- reward 的 `dt/route distance/acceleration/jerk` 阈值不能从 CLI 修改；
- epoch 指标通常是 batch mean 的平均，最后小 batch 未按样本数加权；
- 生成 heading vector 不做单位圆约束，reward 又完全忽略 heading。

## 20. 推荐的改进顺序

1. 修正并测试 `detached_integral` 的时间维切片，重新跑监督 loss 和 RL 梯度审计；
2. 先扩大监督训练数据和独立 val 覆盖，使固定 closed-loop 安全门槛明显高于当前 `1/3`；
3. 扩大官方 closed-loop 场景数，不以 3 场景作为最终结论；
4. 量化 group diversity，消融初始噪声尺度、DPM steps、group size；
5. 校准或替换离线 reward，优先缩小 collision/TTC 与官方 metric 的定义差距；
6. 对比归一化与非归一化指数权重，并加入权重分布日志；
7. 评估更频繁 rollout、旧策略 KL 或 supervised anchor，控制 replay stale policy drift；
8. 修复 DDP 全局日志和 rank-local buffer 语义；
9. 最后再考虑部署时 best-of-N、在线 scorer 或真正 simulator-in-the-loop RL。

## 21. 运行方式

### 21.1 RL 训练模板

`torch_run_rl.sh` 中的三个路径当前留空，必须先填写：

```bash
TRAIN_SET_PATH=/path/to/cache
TRAIN_SET_LIST_PATH=/path/to/diffusion_planner_training.json
PRETRAINED_MODEL_PATH=/path/to/supervised/latest.pth
```

也可以直接运行：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python

$PY -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --train_set /path/to/cache \
  --train_set_list /path/to/diffusion_planner_training.json \
  --pretrained_model_path /path/to/supervised/latest.pth \
  --normalization_file_path ./normalization.json \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --rl_group_size 4 \
  --rl_rollout_steps 5 \
  --rl_buffer_update_epoch 5 \
  --rl_buffer_size 1024 \
  --rl_freeze_encoder true \
  --reward_comfort_weight 0.01
```

### 21.2 Open-loop validation

```bash
$PY evaluate_predictor.py \
  --args_file /path/to/args.json \
  --checkpoint /path/to/latest.pth \
  --data_dir /path/to/val/cache \
  --data_list /path/to/validation.json \
  --batch_size 8 \
  --repeats 3 \
  --seed 3407 \
  --device cuda \
  --output /path/to/eval.json
```

该入口严格加载 EMA，并在 eval 模式关闭 dropout。它计算的是 diffusion validation loss，不是 rollout reward 或闭环 score。

### 21.3 官方闭环评测

```bash
bash scripts/run_mini_closed_loop.sh \
  rl-mini-val-3 \
  /path/to/model_args.json \
  /path/to/rl/latest.pth \
  mini-val-closed-loop-3 \
  hdp
```

这会运行 NuPlan `closed_loop_nonreactive_agents`。当前脚本中的数据库、地图、devkit 和 Python 默认路径绑定本机目录，换机器时应通过环境变量覆盖。

## 22. 面试或项目评审中可能被问到的问题

### 22.1 为什么要按场景组内标准化 reward？

不同场景的绝对难度不同。组内标准化主要比较“同一场景下哪个候选更好”，减少容易场景和困难场景 reward 基线差异。但若组内候选过于相似，标准差很小，scorer 噪声会被放大。

### 22.2 为什么用指数优势权重而不是直接乘 advantage？

指数权重始终为正，适合把样本当作生成模型的回归目标；高奖励样本被强化，低奖励样本被弱化。缺点是容易产生尖锐权重和梯度尺度漂移，而且不会显式排斥坏样本。

### 22.3 这是不是 GRPO？

不是完整 GRPO。它只有组内标准化这一相似点，没有策略比、clipping、reference policy/KL 或 token/action log-prob。更准确的是 group-normalized reward-weighted diffusion regression。

### 22.4 为什么需要 replay buffer？

DPM rollout 成本较高，保存候选后可进行多轮 update，提高样本复用率。代价是候选和 reward 很快相对当前模型变旧，而当前实现没有 importance sampling 或 KL 修正。

### 22.5 为什么默认冻结 encoder？

小规模 RL reward 粗糙、样本少，冻结场景表征能降低灾难性遗忘和显存/计算成本，把更新集中在轨迹生成 decoder。缺点是如果新目标需要改变场景理解，encoder 无法适配。

### 22.6 为什么 reward 变好但闭环碰撞没改善？

训练 collision 只是缓存真值未来上的中心点最小距离，官方指标使用闭环仿真、车辆几何、责任判断和 TTC。两者定义及数据分布不同；加上 pilot 数据少、变化小，离线 improvement 不一定迁移到闭环。

### 22.7 当前最可能出错的位置在哪里？

优先检查：

1. 动作位移与位置积分是否互逆；
2. reward 输入是否为物理单位；
3. neighbor padding mask 是否误伤有效零值；
4. group reshape 和 condition repeat 顺序是否对应；
5. checkpoint 是否完整加载；
6. `detached_integral` 是否真的按时间窗口截断；
7. DDP 指标是否误当成全局指标；
8. rollout 候选是否有足够多样性；
9. reward 与官方 closed-loop 指标是否相关。

### 22.8 为什么 update loss 下降不能证明 RL 成功？

因为 loss 衡量的是对固定 replay 候选的加权重建能力。即使 reward 不提高，模型也可以通过记住这批候选使 loss 下降。必须使用新 rollout、独立场景和官方闭环指标评估。

## 23. 源码定位索引

- RL 入口：`train_predictor_rl.py`
- Batch、rollout、update：`hdp_nuplan/rl/train_epoch_rl.py`
- 奖励：`hdp_nuplan/rl/reward.py`
- Replay：`hdp_nuplan/rl/replay_buffer.py`
- 优势权重和 RL loss：`hdp_nuplan/rl/loss.py`
- 多候选采样：`hdp_nuplan/model/hyper_diffusion_planner.py`
- Decoder/DPM 调用：`hdp_nuplan/model/module/decoder.py`
- VP-SDE 转换：`hdp_nuplan/model/diffusion_utils/sde.py`
- DPM-Solver 包装：`hdp_nuplan/model/diffusion_utils/sampling.py`
- Hybrid 积分：`hdp_nuplan/utils/traj_kinematics.py`
- 场景重载：`hdp_nuplan/utils/dataset.py`
- Open-loop 评估：`evaluate_predictor.py`
- Closed-loop 执行：`scripts/run_mini_closed_loop.sh`
- Closed-loop 汇总：`scripts/summarize_closed_loop_metrics.py`
- 原始实验记录：`docs/mini_supervised_rl_operation_log.md`

## 24. 最终判断

当前 HDP-nuPlan 的强化学习部分已经形成了完整、可运行、可保存 checkpoint、可做独立前后测和官方闭环验证的工程闭环；其核心算法是“组内标准化奖励 + 指数优势加权的扩散回归微调”。

但训练本身仍是缓存张量 scorer 驱动的离线 self-training，而不是真正 simulator-in-the-loop RL。现阶段最重要的工作不是盲目增加 RL epoch，而是先修正 hybrid 积分梯度问题、提高监督策略的闭环安全性、扩大评估场景，并让训练 reward 与官方 collision/TTC/route 指标更一致。
