# HDP-nuPlan 与参考 HDP、Diffusion Planner 的功能和代码差异分析

## 1. 比较范围与结论口径

本文比较的是 2026-07-31 工作区中的实际文件，而不是只比较 Git 已提交版本。三个对象分别为：

1. 当前 HDP：`/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan`
2. 参考 HDP：`/home/yanjun/NewDisk/Hyper-Diffusion-Planner/HDP-nuplan`
3. 当前 Diffusion Planner：`/home/yanjun/NewDisk/Diffusion-Planner-main/diffusion_planner`

三个仓库均存在未提交修改，因此本文中的“当前”表示磁盘上的当前源码快照。对应 Git 基线分别为：

- 当前 HDP 所在仓库：`7571a3baf54f182d9d980e9f559a6cfe9329085c`
- 参考 HDP 所在仓库：`f37cb9ffc36986510691155126a1fe7a0ea89c32`
- Diffusion-Planner-main：`4882ad1945b25dd2452c320905955fab9ebe2d29`

比较重点是功能、数据语义、模型结构、训练目标和推理扩展。按要求不提供逐文件差异表、架构图或大段并排代码。

## 2. 阅读这些代码需要的核心 Python/PyTorch 语法

理解本项目差异主要需要以下语法概念：

- `class X(nn.Module)`：定义可训练的 PyTorch 模块；子模块赋给 `self.xxx` 后会自动注册参数。
- `forward()`：执行模型前向传播。调用 `model(inputs)` 时，PyTorch 会间接调用 `forward()`。
- `@property`：把方法包装成只读属性，例如通过 `model.sde` 访问扩散过程对象。
- `@torch.no_grad()`：关闭自动求导，适合 rollout 和正式推理，能够减少显存占用。
- 张量 `reshape/view/permute/repeat_interleave`：分别用于改变形状、交换维度和沿 batch 复制条件。本文最关键的是区分 batch、候选、智能体和时间四种维度。
- 布尔 mask：用全零状态识别 padding，并在注意力或损失中排除无效邻车。
- 字典展开 `{**inputs, ...}`：保留原输入并加入扩散时间和加噪轨迹。
- Hydra 的 `_target_`：按照字符串路径实例化 Python 类；YAML 中的每个普通字段通常都会作为构造参数传入。
- `try/except/finally`：当前 Diffusion Planner 的精修器大量采用失败回退，保证优化器异常时仍返回原始轨迹。

## 3. 整体作用与最重要的结论

三个实现都服务于 nuPlan 闭环规划，公共部分包括场景矢量化、Encoder、VP-SDE、DPM-Solver、训练数据加载和 nuPlan Planner 适配。但三者的核心目标已经不同。

| 能力 | 当前 HDP | 参考 HDP | 当前 Diffusion Planner |
|---|---|---|---|
| 生成目标 | 只生成自车轨迹 | 只生成自车轨迹 | 联合生成自车与邻车未来轨迹 |
| DiT token 语义 | 80 个未来时间步 | 80 个未来时间步 | 1 个自车 + 10 个邻车 |
| 轨迹表示 | 相邻时刻位移 + 航向 | 相邻时刻位移 + 航向 | 局部绝对位置 + 航向 |
| 扩散监督 | `x_start/noise/v/score` 可交叉转换 | 同左 | `x_start` 或 `score` |
| 额外训练目标 | 截断梯度的积分 waypoint loss | 同左 | 邻车预测 loss |
| 多候选显式采样 | 支持 `[B,G,T,4]` | 不支持 | 默认第二维是智能体，不是候选组 |
| 离线奖励微调 | 支持 | 不支持 | 不支持 |
| classifier collision guidance | 不支持 | 不支持 | 支持 |
| denoising 内轨迹修正 | 不支持 | 不支持 | 代码支持，默认配置关闭 |
| 末端 AL-iLQR/风险门控 | 不支持 | 不支持 | 支持多种可选配置 |

最核心的判断是：

1. 当前 HDP 与参考 HDP 的“论文主干”基本一致；当前 HDP 主要增加了多候选采样、离线 RL 微调、可复现预处理和训练健壮性修复。
2. HDP 相比 Diffusion Planner 的变化不是简单更换 loss，而是改变了扩散变量的组织方式：从“智能体 token + 整段轨迹特征”改成“时间 token + 单步运动特征”。
3. 当前 Diffusion-Planner-main 保留联合预测主干，并沿推理链增加 classifier guidance、denoising 修正 hook、轨迹后处理、FeasibilityNet、风险门控和 AL-iLQR；这些能力在两个 HDP 中都不存在。

## 4. 两个 HDP-nuPlan 之间的差异

### 4.1 没有变化的主干

两个 HDP 的以下核心机制一致：

- Encoder 的场景 token 构成和计算逻辑一致。
- Decoder 只生成自车未来 80 帧的四维状态。
- 位置训练目标先变成相邻帧位移，推理后通过 `cumsum` 恢复轨迹。
- 支持 `x_start`、`noise`、`v`、`score` 四种参数化转换。
- 使用“扩散损失 + 积分 waypoint 损失”的混合训练目标。
- DiT 以未来时间步为 token，并加入当前自车速度条件。
- 标准监督训练仍使用相同的数据张量顺序和 checkpoint 主体结构。

当前 HDP 中 `loss.py`、`sde.py`、`dit.py` 等多个文件虽然与参考目录文本不同，但不少差异只是新增中文解释，不是新的计算逻辑。

### 4.2 当前 HDP 修复了参考 HDP 的推理输出维度问题

参考 HDP 的 Decoder 推理输出实际为：

\[
\text{prediction}\in\mathbb{R}^{B\times T\times 4}.
\]

但其 nuPlan Planner 使用 `outputs['prediction'][0, 0]`，隐含期望第二维存在“轨迹组或智能体”维度。对 `[B,T,4]` 连续取两个索引后只剩一个四维状态，后续再按二维轨迹访问会发生维度错误。

当前 HDP 新增统一的 `sample()`，输出改为：

\[
\text{prediction}\in\mathbb{R}^{B\times G\times T\times 4}.
\]

普通推理令 `G=1`，RL rollout 可令 `G>1`。因此 Planner 中相同的 `[0,0]` 现在表示“第一个场景的第一条候选轨迹”，能够得到 `[T,4]`。

这一修改有两个效果：

- 修复了参考 HDP 标准推理与 Planner 的形状不一致。
- 为同一场景一次生成多条随机候选轨迹提供了正式接口。

新增接口不引入模型参数，所以当前 HDP 与参考 HDP 的神经网络权重结构仍兼容。

### 4.3 当前 HDP 新增离线奖励加权扩散微调

参考 HDP 只有监督训练。当前 HDP 新增 `hdp_nuplan/rl/`、`train_predictor_rl.py` 和 `torch_run_rl.sh`，形成以下训练闭环：

1. 对每个缓存场景采样 `G` 条 HDP 自车轨迹。
2. 使用缓存的邻车未来、路线点、静态目标和运动学代理计算奖励。
3. 将“场景名、候选轨迹组、奖励组”写入 replay buffer。
4. 后续 epoch 从 buffer 有放回采样，对候选轨迹执行奖励加权的扩散回归。

#### 4.3.1 奖励定义

当前默认奖励可概括为：

\[
R = w_pR_{progress}
-w_cC_{collision}
-w_rC_{route}
-w_fC_{comfort}
-w_bC_{backward}
-w_iC_{imitation}.
\]

默认权重为进度 `1.0`、碰撞 `10.0`、路线偏离 `1.0`、舒适性 `0.1`、倒车 `1.0`、模仿 `0.0`。其中：

- 进度使用末帧局部纵向位置 `x_T/10`。
- 碰撞代价使用自车点与邻车/静态物体点的最小欧氏距离，默认安全距离为 2.5 m。
- 路线代价使用候选轨迹点到有效 route 点的最近距离。
- 舒适性代价惩罚超过阈值的加速度与 jerk。
- 倒车代价惩罚局部 x 方向的负增量。
- 模仿项可与专家自车未来轨迹比较，但默认不进入总奖励。

这不是 nuPlan 官方闭环 metric，也没有用车辆矩形、真实碰撞判定、闭环反应式交通参与者或 PDM scorer。它是可运行的离线张量近似层。

#### 4.3.2 奖励加权目标

每个场景内部先标准化奖励：

\[
A_{b,g}=\operatorname{clip}\left(
\frac{R_{b,g}-\mu_b}{\sigma_b+10^{-6}},-c,c
\right),
\qquad
w_{b,g}=\exp(\tau A_{b,g}).
\]

然后用权重放大高奖励候选的扩散回归和 waypoint 回归：

\[
\mathcal L_{RL}
=\mathbb E_{b,g}\left[w_{b,g}\mathcal L_{diff}^{b,g}\right]
+\omega\mathbb E_{b,g}\left[w_{b,g}\mathcal L_{wp}^{b,g}\right].
\]

它本质上是 reward-weighted regression，不是 PPO，也不是对采样动作 log-probability 做策略梯度。权重没有再除以组内权重和，因此温度升高时不仅改变样本相对权重，也可能放大整体 loss 尺度。

#### 4.3.3 Replay Buffer 与训练调度

- Buffer 按场景保存一整组候选，轨迹和奖励转到 CPU。
- 更新时按场景有放回采样，并用场景文件名重新加载条件张量。
- 默认每隔 `rl_buffer_update_epoch` 进行一次 rollout；rollout 前清空旧 buffer，其余 epoch 做更新。
- 可冻结整个 Encoder，只更新 Decoder。
- RL checkpoint 与普通 HDP 模型结构相同，可由正常 Planner 加载。

当前实现的限制包括：

- DDP 下 buffer 是各 rank 本地 buffer，rollout 指标也没有做跨 rank 汇总。
- 奖励使用未来邻车真值，是离线训练期的特权信息，不能当作在线部署 scorer。
- 保存 checkpoint 的条件判断有一段重复的相同取模条件，虽然不改变结果，但应清理。
- 当前仓库日志已经验证小规模软件链路，但日志也明确说明 RL checkpoint 仅有冒烟测试意义，不代表有效驾驶策略。

### 4.4 当前 HDP 的数据预处理更可复现、元数据更完整

相对参考 HDP，当前 HDP 的顶层预处理入口增加了：

- 显式布尔解析，修复 `bool("False")` 仍为真的命令行陷阱。
- `--seed`，固定场景抽样随机性。
- `--log_names_json`，允许显式指定日志划分。
- `--output_list_path`，避免多个实验互相覆盖样本清单。
- 对 NPZ 文件名排序后再写入 JSON。
- 在 NPZ 中额外保存 `log_name` 和 `scenario_type`，便于审计数据来源。
- mini train/val/test 清单构建脚本和互斥性检查。

新增元数据不会进入 Dataset 返回的 11 个模型张量，因此不会改变监督训练的输入数值。

### 4.5 当前 HDP 修复了若干工程问题

当前 HDP 还包含以下有实际行为影响的修复：

- DDP 使用 `find_unused_parameters=True`，兼容车道限速编码器中的数据依赖分支。
- 未获得梯度的参数只在 `log_unused_parameters=true` 时打印，避免每个 batch 大量输出。
- warm-up 小于等于 1 时改用恒等学习率调度，避免零长度 `LinearLR` 异常缩放学习率。
- `wandb` 改成可选依赖；未安装且不启用时仍可使用 TensorBoard。
- `setup.py` 改用 `find_packages()`，非 editable 安装时也能包含 `model/data_process/rl` 等子包。
- ObservationNormalizer 避免用 `torch.tensor(existing_tensor)` 重复构造张量。
- `normalization.json` 中 `ego_current_state` 从错误的 16 维修正为 DataProcessor 实际输出的 10 维。参考 HDP 当前 JSON 仍为 16 维，进入观测归一化时存在尺寸不匹配风险。

现有测试在指定 Conda 环境下运行结果为 `4 passed`，覆盖 RL 基础组件和学习率调度修复。

## 5. HDP 与当前 Diffusion Planner 的核心模型差异

### 5.1 Encoder 基本相同，主要差异在 Decoder

对去除注释和模块名前缀后的语法结构检查表明，两者的场景 Encoder 计算逻辑基本一致：

- 32 个动态 agent 历史 token；
- 5 个静态物体 token；
- 70 个 lane token；
- Mixer 编码单个元素的时间/点序列；
- 位置特征投影；
- 多层自注意力融合场景上下文。

因此 HDP 并没有删除邻车感知。它删除的是“邻车未来轨迹作为扩散生成目标”，邻车历史仍是 Encoder 条件。

默认配置下实测参数量为：

| 模型 | 总参数 | Encoder | Decoder |
|---|---:|---:|---:|
| 当前 HDP | 5,092,996 | 1,799,040 | 3,293,956 |
| 当前 Diffusion Planner | 6,042,628 | 1,799,040 | 4,243,588 |

两者 Encoder 参数量完全相同；HDP 少 949,632 个参数，差异全部来自 Decoder。

### 5.2 扩散 token 和输出含义不同

Diffusion Planner 将一名智能体的“当前状态 + 80 帧未来状态”展平为一个大向量：

\[
\mathbf z_p\in\mathbb R^{(80+1)\times4}=\mathbb R^{324}.
\]

DiT 的 token 数为 `P=1+predicted_neighbor_num=11`，自注意力主要发生在智能体之间。最终训练输出为：

\[
[B,P,81,4].
\]

HDP 只保留自车，每个未来时刻是一个 token：

\[
\mathbf z_t\in\mathbb R^4,
\qquad t=1,\ldots,80.
\]

DiT 的自注意力发生在 80 个未来时间步之间，训练输出为：

\[
[B,T,4].
\]

这导致两者的归纳偏置不同：

- Diffusion Planner 更直接地联合建模“自车—邻车”的未来交互关系。
- HDP 更直接地建模自车轨迹的时间连续性，邻车只通过场景条件影响自车生成。

### 5.3 条件注入不同

Diffusion Planner 使用两类 agent embedding：一个表示自车，另一个共享给所有邻车；同时使用 `neighbor_current_mask` 屏蔽不存在的邻车 token。

HDP 的 embedding 数量等于未来长度，实质上是未来时间位置 embedding。它还把当前自车的 `(v_x,v_y)` 经过线性层后加到每个未来 token 上。

为使这一速度条件在闭环推理时有效，两个 HDP 目录的 DataProcessor 都会从 ego history 计算真实当前运动学量。当前 Diffusion Planner 的 observation adapter 则仍构造零速度状态，但它的 Decoder 只使用当前状态前四维，不依赖速度分量。

### 5.4 DiTBlock 内部结构不同

Diffusion Planner 的 block 使用 6 组 adaLN 参数调制 self-attention 和第一段 MLP；之后的 cross-attention 与第二段 MLP 没有相同的门控残差形式。

HDP 使用 9 组 adaLN 参数，分别调制：

1. self-attention；
2. 与场景编码的 cross-attention；
3. MLP。

三段都采用带 gate 的残差更新。换言之，HDP 不仅改变了 token 轴，也改变了条件交叉注意力在 block 中的融合方式。

### 5.5 Decoder 初始化存在重要差异

Diffusion Planner 在创建 Decoder 后会调用专门的 `initialize_weights()`，其中包括：

- Linear 的 Xavier 初始化；
- embedding 正态初始化；
- adaLN 调制层最后一层置零；
- 最终输出层置零。

两个 HDP 都定义了类似初始化函数，但构造函数没有调用它。因此 HDP Decoder 使用 PyTorch 子模块默认初始化，并不实际从完全置零的 adaLN/output 初始化开始。

这不会影响已训练 checkpoint 的推理，但会影响从头训练的初始优化动态，也是解释训练稳定性或复现实验时必须注意的差异。

## 6. 轨迹表示、扩散参数化和损失差异

### 6.1 Diffusion Planner 学绝对局部坐标，HDP 学逐帧位移

Diffusion Planner 对 `[x,y,cos\theta,sin\theta]` 直接归一化并加噪。其默认位置均值/标准差约为 `[10,0]/[20,20]`，适配未来局部坐标范围。

HDP 先计算：

\[
\Delta \mathbf p_t=\mathbf p_t-\mathbf p_{t-1},
\qquad \mathbf p_0=(0,0),
\]

航向仍保留 `cos/sin`，然后对位移使用更小的标准差 `0.5`。推理时恢复：

\[
\hat{\mathbf p}_t=\sum_{i=1}^{t}\Delta\hat{\mathbf p}_i.
\]

优点是每一步的数值尺度更一致，也更接近运动变化；缺点是小的单步偏差会随时间累积，因此 HDP 又加入 waypoint 混合损失。

### 6.2 参数化空间更灵活

两者都使用线性 VP-SDE：

\[
d\mathbf x=-\frac{1}{2}\beta(t)\mathbf x\,dt+\sqrt{\beta(t)}\,d\mathbf W_t,
\]

以及边缘加噪关系：

\[
\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon.
\]

Diffusion Planner 训练入口只允许模型直接输出 `x_start` 或 `score`，loss 与输出类型绑定。

HDP 增加统一变换接口，支持：

- 干净样本 `x_start`；
- 噪声 `noise`；
- score；
- velocity 参数化 `v`。

模型输出类型与监督类型可以独立设置。例如模型输出 noise，但训练前先转换到 v 空间计算 loss。代码中的 velocity 等价关系为：

\[
\mathbf v_t=\alpha_t\boldsymbol\epsilon-\sigma_t\mathbf x_0
=\frac{\boldsymbol\epsilon-\sigma_t\mathbf x_t}{\alpha_t}.
\]

这一设计便于实验不同 loss space，而不必重写 Decoder。

### 6.3 监督目标不同

Diffusion Planner 的总损失为：

\[
\mathcal L_{DP}
=\mathcal L_{neighbor}
+\alpha\mathcal L_{ego}.
\]

邻车有效点通过 mask 参与预测 loss，自车和邻车共享联合扩散过程。

HDP 不再计算邻车未来预测损失，总损失为：

\[
\mathcal L_{HDP}
=\mathcal L_{increment/diffusion}
+\omega\mathcal L_{waypoint}.
\]

waypoint loss 在积分后的物理位置空间计算。其 `detached_integral` 数值上仍使用完整历史增量，但对时刻 `t` 的位置误差，只允许最近一个窗口内的增量接收梯度：

\[
\frac{\partial \hat{\mathbf p}_t}{\partial\Delta\hat{\mathbf p}_i}=0,
\quad i<t-W+1.
\]

这是一种截断反向传播：保留完整轨迹数值，减少长时域累计梯度和显存压力，但较早动作无法从很远处 waypoint 误差获得梯度。

## 7. 推理和输出接口差异

### 7.1 第二维在两个模型中含义不同

当前 HDP 普通推理输出 `[B,1,T,4]`，第二维是候选组 `G`。

Diffusion Planner 输出 `[B,P,T,4]`，第二维是智能体，`P=1+邻车数`。

两者 Planner 都使用 `[0,0]` 取最终自车轨迹，但语义完全不同：

- HDP：第一个场景的第一条候选；
- Diffusion Planner：第一个场景的自车 agent。

如果后续扩展多候选筛选，不能因为索引形式相同就把两个维度当成同一种含义。

### 7.2 初始噪声与当前状态约束不同

- Diffusion Planner 把当前状态作为每个 agent 轨迹的第 0 帧，并在 DPM-Solver 每个修正步骤后强制恢复当前状态；未来随机噪声尺度为 `0.5`。
- HDP 不包含当前状态 token，直接从 80 帧四维噪声开始，噪声尺度为 `0.1`；自车当前速度通过条件投影输入。

两者默认都使用 10 步、二阶、多步、logSNR 时间离散和 `denoise_to_zero` 的 DPM-Solver。当前 HDP 的显式采样接口允许 rollout 单独指定采样步数。

### 7.3 checkpoint 兼容性

- 两个 HDP 的新增采样和 RL 代码没有增加神经网络参数，模型 checkpoint 主体兼容。
- Diffusion Planner 与 HDP 的 Encoder 可在谨慎筛选 key 后复用，但 Decoder 的 token 语义、输入输出维度和参数形状不同，不能严格加载同一完整 checkpoint。
- 当前 Diffusion Planner 的 guidance/refiner 大多是推理期对象，不改变基础神经网络 state dict，因此基础 DP checkpoint 仍可用于默认配置和可选后处理配置。

## 8. 当前 Diffusion-Planner-main 独有的推理扩展

### 8.1 联合邻车预测支持 classifier collision guidance

Diffusion Planner 的 `guidance_fn` 可在 DPM-Solver 中作为 classifier guidance 使用。当前 `GuidanceWrapper` 默认只组合碰撞能量：

- 将联合生成的自车和邻车轨迹恢复到物理空间；
- 根据 ego 与邻车长宽构造旋转矩形；
- 计算矩形有符号距离和碰撞惩罚；
- 对自车轨迹位置求梯度，并在采样阶段修改反向扩散方向。

HDP 不生成邻车未来，因此不能直接复用这套 joint-trajectory collision guidance。若迁移到 HDP，需要额外的邻车预测器、占用预测或基于当前观测的时空碰撞场。

### 8.2 真正的 denoising hook 与“denoising oracle”要区分

当前 Diffusion Planner 存在两类名称接近但调用位置不同的机制：

1. `DenoisingOptimizationGuidanceCorrector` 真正接在 DPM-Solver 的 `correcting_xt_fn` 上，可只在最后若干去噪步修正轨迹。
2. `DenoisingAlIlqrPathOracle` 实际由 Planner 当作 `trajectory_refiner` 调用，即模型完成全部采样后才调用 AL-iLQR，并不是在每个 denoising step 内调用完整 AL-iLQR。

前者支持两种轻量模式：

- `smooth_path_only`：对轨迹位置做弱局部平滑；
- `path_objective_opt`：对参考偏差、二阶差分和三阶差分目标做少量梯度下降。

修正后还会检查 jerk、横向加速度和 NaN/Inf；变差则拒绝修正。当前 `diffusion_planner_denoising_smooth_guidance.yaml` 中 `enabled: false`，所以代码能力存在，但该配置默认不会改变轨迹。

### 8.3 通用轨迹后处理接口

当前 Planner 新增可选 `trajectory_refiner`。基础模型输出 `[T,3]` 的局部 `x/y/heading` 后，可以：

- 原样返回；
- 做几何平滑；
- 只评估风险；
- 调用 AL-iLQR；
- 根据风险选择性调用 AL-iLQR；
- 记录调试 polyline 和指标。

默认 `diffusion_planner.yaml` 不传 refiner，仍保持基础 DP 行为。另一个全局变化是：未提供 checkpoint 时当前 Planner 会直接报错，而不是用随机模型继续仿真。

### 8.4 几何平滑基线

`GeometricTrajectorySmoother` 对位置和 unwrap 后的 heading 做滑动平均，再与原轨迹按 `blend_alpha` 融合。它不使用车辆动力学、地图、障碍物或约束，主要用于判断 AL-iLQR 的收益是否超过普通平滑。

### 8.5 Post-DP FeasibilityNet

`PostDpFeasibilityEvaluator` 从轨迹计算路径长度、速度、加速度、jerk、曲率、曲率变化率和横向加速度等代理特征，经一个小型 MLP 输出 `[0,1]` 风险分数。

单独使用时它是只读 refiner：记录风险后原样返回 DP 轨迹。带阈值的 shadow 配置也只记录“是否本应触发 AL-iLQR”，不会真正调用优化器。

### 8.6 Terminal AL-iLQR

`TerminalAlIlqrRefiner` 在扩散轨迹生成完成后：

1. 估计原始 DP 轨迹的曲率、横向加速度、加速度和 jerk。
2. 把局部 DP 轨迹转成全局 guide/reference line。
3. 结合 route、道路边界、障碍物、前车和红绿灯构造优化场景。
4. 通过 `AL_iLQR_Planning` 的绑定调用底层 AL-iLQR。
5. 将优化路径转换回局部 80 帧轨迹。
6. 选择 `al_ilqr`、`dp_timing`、`selective` 或 `selective_v2` 时间参数化。
7. 与原 DP 轨迹按 `blend_alpha` 融合；任何异常都回退原轨迹。

该功能虽从 `diffusion_planner/` 入口调用，但实际依赖仓库中的兄弟目录 `AL_iLQR_Planning`、C++/Python binding、路线参考线和纵向场景构建代码，不能把 `diffusion_planner/` 单独复制出去就认为 AL-iLQR 功能完整可用。

### 8.7 风险门控的稀疏 AL-iLQR

`RiskGatedAlIlqrRefiner` 先调用 FeasibilityNet：

- 风险低于阈值：直接返回 DP 轨迹；
- 风险高于阈值：调用 Terminal AL-iLQR；
- 评估或求解异常：默认回退 DP 轨迹。

这与当前 HDP 的 RL 完全不同：HDP RL 改变的是训练权重；风险门控 AL-iLQR 改变的是部署时单帧推理后的轨迹。

## 9. 数据、训练和工程能力的横向差异

### 9.1 数据主张量大体相同

三个实现的预处理主张量仍包括 ego 当前状态、ego 未来、邻车历史/未来、lane、route lane、限速和静态物体。除 HDP 闭环速度状态外，大部分数据处理函数的有效 Python 语法结构一致，许多体积差异来自 Diffusion-Planner-main 中的大量中文注释。

当前 Diffusion Planner 的顶层预处理入口仍更接近参考 HDP：`shuffle_scenarios` 使用 `type=bool`，日志列表和生成的训练清单路径也是固定值；它没有当前 HDP 新增的显式 seed、可选 split JSON、独立输出清单路径、排序清单以及 `log_name/scenario_type` 元数据。因此当前 HDP 的数据划分审计和实验复现能力更完整。

### 9.2 训练目标和部署增强走了两条不同路线

- 当前 HDP：重点改训练表示、参数化空间和奖励加权微调；部署 Planner 较简单。
- 参考 HDP：只包含监督版 HDP 主干。
- 当前 Diffusion Planner：监督训练主干基本保持原始联合预测目标，主要在部署推理阶段增加可选优化和门控。

因此，如果研究目标是“奖励如何塑造扩散策略”，当前 HDP 更接近；如果目标是“学习规划器与显式约束优化如何组合”，当前 Diffusion-Planner-main 更完整。

### 9.3 安装和日志健壮性

当前 HDP 已使用 `find_packages()`，参考 HDP 和当前 Diffusion Planner 的 `setup.py` 只列顶层包。在 editable 安装下通常仍能从源码目录导入，但普通 wheel/非 editable 安装可能遗漏子包。

当前 HDP 允许没有 `wandb` 时继续使用 TensorBoard；另外两个实现仍依赖其原有日志环境。

当前 Diffusion Planner 的监督训练仍使用普通 DDP 包装，且学习率调度器没有当前 HDP 针对 `warm_up_epoch<=1` 的保护。因此当前 HDP 的 DDP 条件分支兼容性和小 epoch 冒烟训练能力也更强；这些属于训练工程修复，不是 HDP 方法本身的理论差异。

## 10. 当前代码中值得优先注意的问题

### 10.1 两个 HDP 共享的 Hydra 配置接口不一致

`hyper_diffusion_planner.yaml` 的 Config 节点仍包含：

```yaml
guidance_fn: null
```

但 `hdp_nuplan.utils.config.Config.__init__()` 只接收 `args_file`。按 Hydra 的正常实例化规则，这个字段会作为额外关键字参数传入，从而产生 `unexpected keyword argument 'guidance_fn'`。代码注释声称 HDP 不接收 guidance，但 YAML 字段尚未删除，两者需要统一。

### 10.2 参考 HDP 的闭环推理存在两个直接风险

- Decoder 输出缺少候选/智能体维，但 Planner 按四维输出索引；当前 HDP 已修复。
- `ego_current_state` 的归一化统计为 16 维，而 DataProcessor 实际状态为 10 维；当前 HDP 已修复。

因此若目标是直接运行当前源码，当前 HDP 明显比参考 HDP 完整。

### 10.3 当前 HDP 的 docstring 有部分旧形状描述

HDP Decoder 已经只处理 `[B,T,4]`，但函数 docstring 中仍残留 `[B,P,1+T,4]` 等来自 Diffusion Planner 的说明。阅读代码和调试时应以实际张量操作为准，后续应更新文档以防误用。

### 10.4 当前 Diffusion Planner 的扩展验证覆盖不均

本次运行 `tests/test_al_ilqr_refiner_geometry.py` 得到 `2 passed`，说明几何辅助逻辑通过现有测试；但 denoising correction、完整 AL-iLQR binding、FeasibilityNet checkpoint 和风险门控闭环并没有被这一测试完整覆盖。它们还依赖外部动态库、checkpoint、地图和 nuPlan 仿真上下文。

## 11. 如何选择和复用

### 11.1 需要纯 HDP 监督训练

应优先使用当前 HDP，而不是参考 HDP，因为模型主干相同，但当前版本修复了推理形状、归一化、DDP、warm-up、日志和安装问题。

### 11.2 需要多候选或奖励微调

只能直接使用当前 HDP。建议在正式实验前：

- 用高质量监督 checkpoint 替换当前 smoke checkpoint；
- 提高 rollout 的 DPM 步数并检查候选多样性；
- 校准 comfort 和 collision 奖励尺度；
- 用官方 nuPlan 闭环指标验证，而不是把离线奖励当最终指标；
- 若使用多 GPU，补充跨 rank 的 rollout 指标聚合和 buffer 设计说明。

### 11.3 需要显式碰撞引导或优化器约束

当前 Diffusion Planner 更合适，因为联合邻车预测、classifier guidance 和 AL-iLQR 插件链已经存在。若要迁移到 HDP，需要先解决“没有邻车未来生成”的条件信息缺口，并决定优化发生在 denoising 内还是最终轨迹后处理阶段。

### 11.4 想组合两条路线

较自然的组合不是直接复制 Decoder，而是保留清晰接口：

1. HDP 负责生成 `G` 条自车候选。
2. 使用更可靠的闭环/可行性 scorer 对候选排序或做训练奖励。
3. 仅对高风险或最终选中的候选调用 AL-iLQR。
4. 分别评估“RL 改善生成分布”和“AL-iLQR 修复尾部失败”的独立贡献。

直接把 DP 的 collision guidance 搬进 HDP 不够，因为它依赖联合生成的邻车未来轨迹；直接对所有 HDP 候选调用 AL-iLQR 又可能造成不可接受的运行时开销。

## 12. 面试或项目评审可能追问的问题

### 12.1 为什么 HDP 要预测位移而不是绝对位置？

位移数值范围更平稳，能够减轻远期位置尺度差异；但会带来累计误差，所以增加积分 waypoint loss。还需要说明截断积分梯度是在稳定性和长期 credit assignment 之间做折中。

### 12.2 为什么删除邻车未来预测仍能规划？

因为邻车历史仍进入 Encoder，HDP 是条件式自车生成，不是忽略邻车。但没有显式邻车未来输出后，联合一致性监督和原 DP 的 collision guidance 都不能直接保留。

### 12.3 HDP 的 RL 算法是不是策略梯度？

不是。当前实现对采样候选做组内奖励标准化，再执行指数权重的监督扩散回归。没有计算策略 log-probability、importance ratio 或 PPO clipping。

### 12.4 为什么当前 HDP 参数更少？

Encoder 完全相同。Decoder 不再把 81 帧四维状态展平成 324 维，也不再同时处理 11 个 agent 的完整轨迹，因此输入/输出投影更小，总计减少约 95 万参数。

### 12.5 为什么 Diffusion Planner 的优化扩展不等于 HDP 的 hybrid loss？

HDP hybrid loss 是训练期可微监督；AL-iLQR 是推理期显式约束优化。前者改变模型参数，后者在不改 checkpoint 的情况下修正单次输出。

### 12.6 最容易出错的地方在哪里？

- `[B,G,T,4]` 与 `[B,P,T,4]` 的第二维语义混淆；
- 位移空间、归一化空间和物理 waypoint 空间混淆；
- Hydra YAML 与构造函数参数不一致；
- 使用错误维度的 normalization JSON；
- 把离线近似奖励当作官方闭环指标；
- 将名字含 denoising 的 planner-level oracle 误认为逐步 denoising 引导；
- 忽略 AL-iLQR 对兄弟目录、动态库、地图上下文和额外 checkpoint 的依赖。

## 13. 本次验证记录

本次完成了以下只读/轻量验证：

- 对三个目录进行了源码结构、当前 Git 状态和公共模块语义比较。
- 用相同默认规模实例化两个模型并统计参数量。
- 当前 HDP 测试：`4 passed in 3.11s`。
- 当前 Diffusion Planner 几何精修测试：`2 passed in 1.42s`，另有第三方 pyparsing 弃用警告。

未执行完整 nuPlan 闭环仿真、完整 AL-iLQR 动态库集成测试或大规模训练，因此本文对这些部分的判断来自当前代码路径、配置和已有仓库日志，不代表重新获得了闭环性能结论。
