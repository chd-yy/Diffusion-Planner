# HDP-NAVSIM 强化学习代码与公式完整报告

> 文档日期：2026-08-01<br>
> 分析对象：`/home/yanjun/NewDisk/Diffusion-Planner/HDP-navsim` 当前磁盘代码<br>
> 文档位置：`/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan`<br>
> Git 基线：`7571a3baf54f182d9d980e9f559a6cfe9329085c`（2026-07-30）<br>
> 范围说明：本文只梳理 HDP-NAVSIM 中与强化学习训练、奖励、扩散采样、缓存、验证和部署相关的不同/关键部分，不做逐文件差异表，也不画架构图。

## 1. 结论先行

当前 HDP-NAVSIM 实现的是一条“自生成候选轨迹—PDM 打分—优势加权扩散回归”的强化学习式微调链路：

1. 从监督训练的 Dp-VLA checkpoint 初始化扩散轨迹 decoder；
2. 对同一 NAVSIM 场景随机生成 $G$ 条完整 4 秒轨迹；
3. 用 NAVSIM 的 PDM simulator 与 PDM scorer 对每条轨迹打分；
4. 将同场景候选组及其绝对 PDM 分数写入 replay buffer；
5. 在同一场景内部标准化 PDM 分数，得到 group-relative advantage；
6. 用 \(\exp(A)\) 加权每条 rollout 轨迹的扩散去噪 MSE；
7. 只更新 Dp-VLA 的完整 diffusion decoder，视觉语言 encoder 不参与 RL 反向传播。

从算法性质看，它不是 PPO、GRPO、SAC、DQN 或 Actor-Critic。更准确的名称是：

> **基于 PDM 奖励的分组优势加权扩散微调**，或 contextual-bandit 形式的 reward-weighted regression / self-imitation。

它与 PPO/GRPO 的共同点只有“同一上下文采样一组候选并做组内相对比较”。当前实现没有：

- 动作或轨迹的策略 log-probability；
- old policy / reference policy；
- probability ratio 与 clip；
- KL 正则；
- value/critic 网络；
- TD target、Bellman backup 或折扣累计回报；
- 多步环境交互产生的状态转移。

还必须区分“代码注释/配置声称的行为”和“当前实际执行的行为”。本次审计发现：

- YAML 中 `progress_weight=30`、`ttc_weight=5`、`comfortable_weight=2` 没有传给 `PDMScorerConfig`，实际 reward 仍采用 NAVSIM v1.1 默认权重 `5/5/2/0`；
- `bc_data=true` 没有被代码读取，ground-truth trajectory 虽写入 replay buffer，但 update 时被丢弃；
- 模型包含正/负 LoRA adapter 和 classifier-free guidance 支持，但 RL 初始化没有调用 `init_lora_adapter()`，实际是完整 decoder fine-tuning；
- rollout 有组索引重复问题：默认 `G=10` 时，一个有效场景组通常会被重复写入 buffer 10 次；
- PDM 失败返回 `None` 后，reward 聚合会先访问 `None.__dict__`，使原本设计的 NaN 过滤/重试无法工作；
- validation 把 `PDMResults` dataclass 直接交给 `pd.concat`，按 NAVSIM v1.1 API 会进入异常分支并返回空指标；
- 当前目录未包含 RL checkpoint、TensorBoard 日志、CSV 或实验报告，且本机尚未配置/安装 `navsim`，所以能够做静态审计和局部公式验证，不能据此宣称 RL 训练已端到端跑通或取得性能提升。

因此，本文会同时给出：算法意图、数学推导、当前实际代码路径、未生效配置和已确认的实现风险。

## 2. 阅读代码前需要了解的 Python / PyTorch 语法

### 2.1 类、继承与 Hydra 实例化

`DpVlaRlAgent(AbstractAgent)` 继承 NAVSIM 的 agent 接口。Hydra 根据 YAML 中的 `_target_` 动态构造类，例如：

```yaml
_target_: hdp_navsim.agent.dp_vla.dp_vla_rl_agent.DpVlaRlAgent
```

`_get(node, key, default)` 同时兼容字典、OmegaConf 和普通对象，使一套代码可以读取多种配置对象。

### 2.2 装饰器 `@torch.no_grad()`

`_rl_rollout()`、`DpVlaModel.generate()` 都用 `@torch.no_grad()`，表示：

- rollout 采样不保存反向传播图；
- PDM reward 不能对模型参数直接求导；
- 梯度只来自后续 update 阶段重新加噪、重新 decode 得到的扩散监督损失。

这正是“黑盒奖励 + 回归更新”的实现基础。

### 2.3 张量维度、展平与复制

最关键的形状记号为：

- $B$：场景 batch size；
- $G$：每个场景的候选轨迹数，默认 10；
- $K$：每条 replay 轨迹重复采样扩散噪声的次数，默认 1；
- $T=8$：轨迹点数；
- $D=4$：每点动作维度 `[x, y, cos(yaw), sin(yaw)]`；
- $M$：Florence-2 encoder token 数；
- $C=1024$：encoder/decoder hidden size。

主要张量如下：

| 张量 | 形状 | 含义 |
|---|---:|---|
| `encoder_output` | `[B, M, 1024]` | 缓存的视觉语言场景条件 |
| `proprio` | `[B, 12]` | 4 帧 × 3 维自车历史状态 |
| `rollout_action` | `[B*G, 8, 4]` | 展平后的候选轨迹 |
| `rewards` | `[B, G]` | 同场景候选 PDM 分数 |
| `advantage` | `[B, G]` | 组内标准化奖励 |
| `action_with_noise` | `[B*G*K, 8, 4]` | 前向扩散后的轨迹 |

`repeat_interleave(G, 0)` 会把每个场景连续复制 $G$ 次；`flatten(0, 1)` 把 `[B,G,...]` 合成 `[B*G,...]`；`unflatten(0, (-1,G))` 再恢复组结构。

### 2.4 `deque` 与有放回采样

`ReplayBuffer` 用 `deque(maxlen=maxsize)` 存储场景组。当前构造为 `ReplayBuffer()`，即没有容量上限。更新时使用：

```python
random.choices(tuple(self.sample_queue), k=size)
```

这是有放回采样，同一个 buffer item 可以在同一 update batch 中被多次抽到。

### 2.5 Ray 远程任务与对象引用

`@ray.remote` 把每条候选轨迹的 PDM 评分分发到 Ray worker。大对象如 metric cache、simulator、scorer 先通过 `ray.put()` 放入对象存储，任务之间共享，避免为每条候选反复序列化。

### 2.6 `detach()`、`.cpu()` 与显存所有权

- `detach()` 切断计算图，但不改变设备；
- `.cpu()` 才会把数据从 GPU 移到主存；
- 当前 replay 中的 rollout/reward 由 `no_grad` 产生，却没有 `.cpu()`，因此仍会长期占用 GPU 显存。

### 2.7 Lightning hook

训练包装器在每个 epoch 开始调用：

```python
self.agent.on_train_epoch_start(self.current_epoch)
```

`compute_loss()` 返回含 `loss` 的字典时，Lightning 才执行反向传播；rollout epoch 不返回 `loss`，所以 `training_step()` 返回 `None`，该 batch 不更新参数。

## 3. 当前所有 RL 相关代码范围

### 3.1 直接实现 RL 的文件

| 路径 | 作用 |
|---|---|
| `hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py` | RL agent 主体：初始化、rollout、PDM reward、replay update、loss、优化器、checkpoint、validation |
| `hdp_navsim/agent/dp_vla/scoring.py` | Ray 并行 PDM 评分、候选排序、早期轨迹扰动 |
| `hdp_navsim/agent/dp_vla/model/rl_utils.py` | `ReplayBuffer` 与按 token 重载 RL feature cache |
| `hdp_navsim/config/agent/dp_vla_rl_agent.yaml` | RL 周期、group size、rollout steps、cache 路径等配置 |
| `scripts/training/run_training_rl.sh` | RL 训练启动入口 |

### 3.2 为 RL 提供数据的文件

| 路径 | 作用 |
|---|---|
| `preprocessing/dp_vla_rl_feature_builder.py` | 用监督 checkpoint 预计算 Florence-2 encoder hidden states |
| `preprocessing/dp_vla_rl_target_builder.py` | 构造 RL target builder 名称，目标逻辑继承基础 builder |
| `preprocessing/dp_vla_target_builder.py` | 生成 8×4 future waypoint target |
| `training/training_utils/dataset.py` | cache 写入、JSON 数据列表、token 返回、训练时加载 |
| `training/run_cache_training.py` | 单节点 cache 入口 |
| `training/run_cache_training_multi_node.py` | RL encoder feature 多 GPU cache 入口 |
| `scripts/training/run_cache_training.sh` | cache shell 入口 |

### 3.3 为 RL 提供扩散模型和采样的文件

| 路径 | 作用 |
|---|---|
| `model/modeling_dp_vla.py` | Florence-2 encode、DiT decode、DPM-Solver 生成、LoRA/CFG 支持 |
| `model/decoder.py` | 12 层 CustomDiT action decoder |
| `model/diffusion_utils/diffusion_sde.py` | VP 前向扩散、训练 target、DPM-Solver++ 封装 |
| `model/diffusion_utils/dpm_solver_pytorch.py` | VP noise schedule、model wrapper、DPM-Solver++ |
| `dp_vla_agent.py` | waypoint/difference 转换、prediction 参数化互转、可选 hybrid loss |
| `config/agent/_shared/model.yaml` | decoder 维度、动作维度、LoRA 参数 |
| `config/agent/_shared/diffusion_sde.yaml` | VP-SDE 与采样器参数 |
| `config/agent/_shared/trajectory_sampling.yaml` | target 和 PDM 的两个时间分辨率 |

### 3.4 为 reward 与评估提供支持的文件

| 路径 | 作用 |
|---|---|
| `agent/dp_vla/utils.py` | 构造 `PDMSimulator`、`PDMScorer`、`TrajectorySampling` |
| `scripts/evaluation/run_metric_caching.sh` | 生成每个 token 的 PDM metric cache |
| `scripts/evaluation/run_pdm_score.sh` | 调用上游 NAVSIM `run_pdm_score.py` 做离线 PDMS 评估 |
| `training/agent_lightning_module.py` | RL/监督共用 Lightning 生命周期与日志 |
| `training/training_utils/hf_export.py` | Lightning checkpoint 之外的 HF 格式导出 |

### 3.5 不属于 RL 算法、但容易被误认为 RL 的代码

- `DpVlaAgent.compute_loss()` 是普通监督扩散训练；
- Florence-2 encoder 和 CustomDiT 是策略模型结构，不等同于 RL；
- LoRA positive/negative adapter 是模型能力，但当前 RL 路径没有启用；
- PDM metric caching 是 reward 的前置数据准备，不更新策略；
- `compute_batch_trajectory()` 只做多样本推理，不做候选选择或学习。

## 4. 算法整体作用与项目位置

### 4.1 把当前问题写成 contextual bandit

记场景条件为：

\[
c=(h_{\text{VLM}}, y),
\]

其中 $h_{\text{VLM}}$ 是 Florence-2 编码后的多视角图像与语言 token，$y$ 是 4 帧自车状态。完整 4 秒轨迹记为：

\[
\tau=\{(x_t,y_t,\cos\psi_t,\sin\psi_t)\}_{t=1}^{8}.
\]

扩散策略隐式定义条件分布：

\[
\tau\sim\pi_\theta(\tau\mid c).
\]

PDM 对整条轨迹返回单个标量：

\[
R=R_{\mathrm{PDM}}(c,\tau).
\]

训练中没有显式的逐时刻动作 $a_t$、环境反馈 $r_t$ 和新状态 $s_{t+1}$。因此当前形式是“一次观察场景、一次生成完整轨迹、一次得到整体 reward”的 contextual bandit，而不是标准多步 MDP。

### 4.2 默认 epoch 周期

代码判定：

```python
if current_epoch % replay_buffer_update_epoch == 0:
    rollout
else:
    update
```

默认 `replay_buffer_update_epoch=10`，所以：

- epoch `0, 10, 20, ...`：清空旧 buffer，并用当前模型 rollout；
- 每个 rollout epoch 后的 9 个 epoch：有放回抽取同一个 buffer 更新 decoder；
- 下一轮 rollout 前 buffer 被完全清空。

这是一种周期性刷新数据的近似 on-policy 自训练。rollout 当下来自当前模型，但后续 9 个 epoch 会重复使用逐渐变旧的轨迹，所以更新阶段又带有 replay/off-policy 特征。

### 4.3 一次训练 batch 的实际分支

`compute_loss()` 先读取：

```python
encoder_output = features["encoder_output"]
proprio = features["history"]
actions = targets["ego_future_trajectory"]
```

然后：

- rollout epoch：当前 batch 条件真的用于生成轨迹与 PDM 评分；
- update epoch：当前 batch 的 `encoder_output` 和 `proprio` 只用于取得 $B$ 与 device，随后被 replay 抽到的 token 对应 cache 覆盖。

也就是说，update 数据来源是 replay，而不是当前 DataLoader batch 的场景。

## 5. RL 数据、条件与动作语义

### 5.1 RL feature cache

缓存阶段构造带 encoder 的监督模型：

```python
self.initialize_pretrain_model(with_encoder=True)
```

随后 `DpVlaRlFeatureBuilder`：

1. 拼接多相机、多历史帧图像；
2. 根据自车状态构造语言 prompt；
3. 调用 Florence-2 vision tower 与 BART encoder；
4. 保存 `encoder_output.detach().cpu()`；
5. 同时保存 `meta_status`。

训练阶段模型改为：

```python
self.initialize_pretrain_model(with_encoder=False)
```

因此模型对象中只保留 decoder，encoder 完全不参与 RL 训练和梯度同步。历史状态由：

```python
feature["meta_status"][:, :3].reshape(-1)
```

得到。默认 4 帧、每帧取 3 维，所以 proprio 为 12 维。

### 5.2 注意力 mask 的边界

RL feature cache 只存 `last_hidden_state`，没有存 Florence-2 原始 `attention_mask`。update 阶段创建全 1 mask：

```python
attention_mask = torch.ones(encoder_output.shape[:2], dtype=torch.bool)
```

这相当于假设缓存的全部 encoder token 都有效。如果不同样本存在 padding token，当前 update 会让 decoder 也 attend 到这些位置；该行为与在线 encode 时使用真实 mask 不完全一致。

### 5.3 target 轨迹

target sampling 配置为：

```yaml
time_horizon: 4
interval_length: 0.5
```

得到 8 个 future pose。yaw 被变为连续表示：

\[
(x_t,y_t,\psi_t)
\rightarrow
(x_t,y_t,\cos\psi_t,\sin\psi_t).
\]

默认 `kinematic_type=waypoint`，所以模型直接生成绝对局部 waypoint。代码也支持 `diff`：

\[
\Delta p_t=p_t-p_{t-1},\qquad p_0=(0,0),
\]

推理时再通过累加恢复：

\[
p_t=\sum_{i=1}^{t}\Delta p_i.
\]

但当前 RL YAML 使用 `waypoint`，不会进入 difference integration 和 hybrid waypoint loss。

### 5.4 PDM 使用更密的时间分辨率

PDM proposal sampling 为 4 秒、0.1 秒。模型只输出 0.5 秒间隔的 8 个点；NAVSIM `Trajectory` 将其解释为 4 秒/0.5 秒轨迹，随后 PDM 的 `get_trajectory_as_array()` 按 0.1 秒插值，并送入 LQR/自行车模型 simulator。

所以 reward 不是只在 8 个离散点上做几何判断，而是基于 0.1 秒仿真状态评分。

## 6. 扩散模型与训练公式

### 6.1 线性 VP 噪声日程

当前 `NoiseScheduleVP(schedule="linear")` 使用默认：

\[
\beta_0=0.1,\qquad \beta_1=20.
\]

代码中的 log mean coefficient 为：

\[
\log\alpha_t
=-\frac14(\beta_1-\beta_0)t^2
-\frac12\beta_0t.
\]

因此：

\[
\alpha_t=
\exp\left[-\frac14(\beta_1-\beta_0)t^2
-\frac12\beta_0t\right],
\]

\[
\sigma_t=\sqrt{1-\alpha_t^2}.
\]

扩散时间均匀采样：

\[
t\sim\mathcal U(10^{-3},1).
\]

YAML 中虽然还保留 `alpha: 1.0`、`beta: 1.5` 注释，`TimeSampler` 当前只支持 uniform，这两个 beta distribution 参数没有被使用。

### 6.2 前向加噪

给 replay 中的干净轨迹 $x_0$ 采样高斯噪声：

\[
\epsilon\sim\mathcal N(0,I),
\]

构造：

\[
x_t=\alpha_t x_0+\sigma_t\epsilon.
\]

`DiffusionSDE.sample()` 同时准备三种可选监督 target：

\[
y_{\text{noise}}=\epsilon,
\]

\[
y_{\text{score}}=-\frac{\epsilon}{\sigma_t},
\]

\[
y_{x_0}=x_0.
\]

当前 `DpVlaConfig` 默认 `model_type="noise"`，共享 model YAML 没有覆盖它，因此实际 RL target 是 $\epsilon$。

### 6.3 Decoder 条件

CustomDiT 的单步预测可写为：

\[
\hat\epsilon_\theta
=f_\theta(x_t,t,y,h_{\text{VLM}},m),
\]

其中：

- $x_t\in\mathbb R^{8\times4}$：带噪轨迹；
- $t$：扩散时间 embedding；
- $y\in\mathbb R^{12}$：自车历史 proprio；
- $h_{\text{VLM}}\in\mathbb R^{M\times1024}$：缓存的场景 token；
- $m$：attention mask。

默认 decoder 是 12 层、hidden size 1024、16 heads 的 DiT。动作先经 MLP 投影，时间和 proprio 合并为 adaLN 条件，VLM token 作为 cross-attention 条件。

### 6.4 rollout 的反向扩散

生成时先采样：

\[
x_{t=1}^{(0)}\sim\mathcal N(0,0.5^2I).
\]

这里 `sample_temperature=0.5` 是 `generate()` 默认值；rollout 没有显式覆盖它。随后使用：

- DPM-Solver++；
- order 2；
- `logSNR` 时间步；
- multistep；
- `denoise_to_zero=true`；
- rollout 实际 steps = 5。

需要注意三个不同的 step 值：

| 来源 | 值 | 当前用途 |
|---|---:|---|
| `diffusion_sde.yaml: sample_steps` | 25 | 没有被 `DpVlaModel.generate()` 自动读取 |
| `generate()` 默认 `steps` | 10 | 普通单轨迹/验证推理 |
| `rl_config.rollout_steps` | 5 | RL rollout 明确传入 |

因此不能只看 `sample_steps: 25` 就认为 RL 用 25 步采样。

### 6.5 参数化互转

如果模型预测 noise：

\[
\hat x_0=\frac{x_t-\sigma_t\hat\epsilon}{\alpha_t}.
\]

如果预测 score $s_\theta$：

\[
\hat\epsilon=-\sigma_t s_\theta,
\qquad
\hat x_0=\frac{x_t+\sigma_t^2s_\theta}{\alpha_t}.
\]

代码也支持直接预测 $x_0$。当前 RL update 直接根据 `model_type` 选择 target，没有像监督 `DpVlaAgent` 那样读取独立的 `supervision_type`，所以 RL 的模型输出参数化和 loss target 参数化总是相同。

## 7. Rollout：同场景分组生成

### 7.1 条件复制

对每个场景复制 $G$ 份条件：

```python
encoder_rep = encoder_output[:, None].repeat_interleave(G, 1)
proprio_rep = proprio[:, None].repeat_interleave(G, 1)
```

再展平为 `[B*G,...]`，一次调用 DPM-Solver。候选差异主要来自初始化高斯噪声。

### 7.2 早期轨迹增强

当 `current_epoch < 5` 时，每条候选采样：

\[
a,b\sim\mathcal N(0,0.5^2).
\]

令轨迹原 heading 为 $\psi_t$，位置扰动为：

\[
x'_t=x_t+a\cos\psi_t-b\sin\psi_t,
\]

\[
y'_t=y_t+a\sin\psi_t+b\cos\psi_t.
\]

`cos(yaw)`、`sin(yaw)` 本身不变。因此同一条轨迹上的所有点共享一个局部纵向/横向平移量，不是逐点抖动。

默认 rollout 周期为 10，满足 `epoch < 5` 的 rollout 只有 epoch 0。epoch 1–4 是 update，不会执行增强；epoch 10 的下一次 rollout 已不再增强。

### 7.3 送入 PDM 前恢复 heading

模型动作 `[x,y,cos,sin]` 转为：

\[
\psi=\operatorname{atan2}(\sin\psi,\cos\psi),
\]

最终每条候选以 `[x,y,heading]` 构造 NAVSIM `Trajectory`。

### 7.4 rollout 不产生梯度

该阶段只返回 reward/metric 字典，没有 `loss` 键。Lightning 因而不执行 optimizer step。`self.eval()` 只切换模型模式；当前 decoder 没有训练期随机 dropout，主要作用是保持推理语义一致。

## 8. PDM reward 的代码与公式推导

### 8.1 reward 不是手写近似，而是调用 NAVSIM PDM

`compute_pdm_score_single()` 直接调用：

```python
pdm_result = pdm_score(
    metric_cache=metric_cache,
    model_trajectory=trajectory,
    future_sampling=proposal_sampling,
    simulator=simulator,
    scorer=train_scorer,
)
reward = pdm_result.score
```

这与在 tensor 上手写碰撞、舒适度等启发式项不同。其过程是：

1. 把模型局部轨迹转换到全局坐标；
2. 从 metric cache 读取 PDM baseline trajectory、初始 ego、中心线、路线、drivable map 和 observation；
3. 将 baseline proposal 与 model proposal 拼成两个候选；
4. 用 PDM simulator/LQR 自行车模型生成 0.1 秒状态；
5. 用 PDM scorer 计算各子指标；
6. 返回 model proposal 的 `PDMResults.score`。

它属于基于缓存交通观测的 pseudo-closed-loop/non-reactive 评分：自车轨迹经过动力学仿真，但其他交通参与者不是一个会针对当前候选轨迹实时响应的交互式策略。

### 8.2 当前实际 PDM 权重

`build_pdm_components()` 使用：

```python
scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
```

没有把 RL YAML 的 reward weight 传进去。按当前代码所依赖的 NAVSIM v1.1 API，默认值为：

\[
w_{EP}=5,
\quad w_{TTC}=5,
\quad w_C=2,
\quad w_{DDC}=0.
\]

### 8.3 乘法门控项

记：

- $NC$：no-at-fault-collision；
- $DAC$：drivable-area-compliance。

乘法门控为：

\[
M=NC\cdot DAC.
\]

一旦门控为 0，最终 PDMS 也为 0。

### 8.4 Progress 的相对归一化

PDM 每次评分同时包含 cache 中的 baseline proposal 和当前 model proposal。先计算原始 progress $p_i$，再乘门控：

\[
\tilde p_i=p_iM_i.
\]

若两个 proposal 的最大有效 progress 大于 5 米：

\[
EP_i=\frac{\tilde p_i}{\max_j\tilde p_j}.
\]

否则：

\[
EP_i=
\begin{cases}
1,&M_i>0,\\
0,&M_i=0.
\end{cases}
\]

因此 EP 不是固定尺度的绝对行驶距离，而是当前 model candidate 与 cache baseline 在这次二候选评分中的相对量。

### 8.5 最终 reward

记：

- $EP$：normalized progress；
- $TTC$：time-to-collision within bound；
- $C$：comfort；
- $DDC$：driving-direction compliance。

一般公式为：

\[
R_{PDM}
=M\cdot
\frac{
w_{EP}EP+w_{TTC}TTC+w_C C+w_{DDC}DDC
}{w_{EP}+w_{TTC}+w_C+w_{DDC}}.
\]

代入当前实际默认权重：

\[
\boxed{
R_{PDM}
=NC\cdot DAC\cdot
\frac{5EP+5TTC+2C}{12}
}
\]

虽然 DDC 被计算并记录，但权重为 0，不影响最终 score。

### 8.6 YAML 的 `30/5/2` 为什么没生效

YAML 声明：

```yaml
progress_weight: 30.0
ttc_weight: 5.0
comfortable_weight: 2.0
```

但 `_init_rl()` 只读取周期、group、repeat、rollout、data/cache path 和 proposal sampling；没有读取这三个字段，也没有构造自定义 `PDMScorerConfig`。全仓搜索也只在 YAML 找到它们。

因此当前实际 reward 不是：

\[
M\frac{30EP+5TTC+2C}{37},
\]

而仍是上一节的默认 $5/5/2$ 公式。README 把 “PDM reward weights” 列为可控 RL 参数，与当前代码不一致。

### 8.7 Ray 并行语义

每条 `[8,3]` 轨迹启动一个 Ray remote task。完成顺序可能不同，但返回 `(input_index, score, result)`，driver 最后按 index 排序，所以正常情况下 score 与原候选顺序一致。

`compute_pdm_score_single()` 捕获 PDM 内部异常，设计上返回：

```python
(i, np.nan, None)
```

但后续 `_reward_fn()` 当前没有安全处理 `None`，详见第 14 节。

### 8.8 上游版本边界

本仓库的 `setup.py` 没有固定 NAVSIM 版本。本文 PDM 公式依据当前 import 路径、`PDMResults` 字段和 NAVSIM 官方 v1.1 源码；若实际环境安装其他版本，必须重新核对 scorer 权重和结果结构。

官方对应源码：

- [NAVSIM v1.1 `pdm_score.py`](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/evaluate/pdm_score.py)
- [NAVSIM v1.1 `pdm_scorer.py`](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py)
- [NAVSIM v1.1 `PDMResults`](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/common/dataclasses.py)

## 9. Replay buffer 的实际数据流

### 9.1 设计上的 item

每个 replay item 设计为：

```text
(
  scene_token,
  rollout_group [G,8,4],
  gt_group      [G,8,4],
  reward_group  [G]
)
```

只保存 token，不重复保存大型 VLM feature。update 时通过 token 查 JSON 中的 `log/token` 路径，再从磁盘读取 encoder output 和 proprio。

### 9.2 ground truth 当前不参与 update

取 replay 时：

```python
scene_tokens, rollout_actions, _, reward_abs = replay_buffer.get(B)
```

第三项 ground truth 被 `_` 丢弃。YAML 的 `bc_data: true` 也没有被读取。因此当前目标只拟合模型自己的 rollout，不包含：

- behavior cloning loss；
- 与 expert trajectory 的 imitation loss；
- expert/sample 混合 batch。

### 9.3 buffer 生命周期

buffer 在每个 rollout epoch 开始被清空，没有跨 rollout 周期积累。当前没有 `maxsize`，但一个 rollout epoch 会遍历整个训练 DataLoader，因此仍可能很大。

### 9.4 update 有放回抽样

每个 update batch 抽 $B$ 个 scene group，之后将其展平为 $BG$ 条轨迹。因为有放回，同一 scene group 可以重复出现。

### 9.5 当前显存风险

rollout/group/reward 都留在当前 device，buffer `put()` 没有 `.cpu()`。即使它们没有 autograd graph，也仍是 GPU tensors。再叠加当前组索引重复写入，buffer 显存消耗大致会被放大到设计值的 $G$ 倍。

### 9.6 CacheExtractor 的 token 映射

JSON 每项是 `log_name/token`，但映射构造为：

```python
{p.split("/")[-1]: p for p in data_list}
```

即只用 token 作为 key。如果不同 log 中 token 不全局唯一，后出现者会静默覆盖前者。NAVSIM token 通常预期唯一，但代码没有显式校验。

## 10. Group-relative advantage

### 10.1 公式

对场景 $b$ 的 $G$ 个 PDM reward：

\[
R_{b,1},\ldots,R_{b,G},
\]

计算组均值：

\[
\mu_b=\frac1G\sum_{g=1}^{G}R_{b,g}.
\]

PyTorch `std()` 默认使用 Bessel correction，即样本标准差：

\[
s_b=
\sqrt{
\frac{1}{G-1}
\sum_{g=1}^{G}(R_{b,g}-\mu_b)^2
}.
\]

优势为：

\[
A_{b,g}
=\frac{R_{b,g}-\mu_b}{s_b+10^{-6}}.
\]

这一步消除了不同场景 reward 标度差异，只比较同一场景中哪条轨迹更好。

### 10.2 与 GRPO 的关系和区别

“组内标准化奖励”与 GRPO 的 group-relative advantage 思想相似。但 GRPO/PPO 通常还需要：

\[
r_\theta=
\frac{\pi_\theta(\tau\mid c)}
{\pi_{\theta_{old}}(\tau\mid c)},
\]

再构造 clipped objective 和 KL。当前代码没有计算扩散轨迹 log-probability，也没有 old/reference model，因此不能称为完整 GRPO。

### 10.3 `G=1` 的 NaN 问题

样本标准差分母为 $G-1$。当 `group_size=1` 时，PyTorch 返回 NaN；`NaN + 1e-6` 仍是 NaN，整个 loss 随之污染。配置中没有断言 `group_size > 1`。

### 10.4 指数权重

代码使用：

\[
w_{b,g}=\exp(A_{b,g}).
\]

没有 temperature、clip 或归一化。由于是组内样本标准化，对 $G=10$，单个标准化值理论最大约为：

\[
\frac{G-1}{\sqrt G}
=\frac9{\sqrt{10}}
\approx2.846,
\]

对应最大指数权重约：

\[
e^{2.846}\approx17.2.
\]

低 reward 轨迹的权重仍严格大于 0，所以它们不是被完全拒绝，只是贡献较小。

另一个细节是权重没有除以组内总和。即使 $\sum_g A_{b,g}\approx0$，由 Jensen 不等式通常有：

\[
\frac1G\sum_g e^{A_{b,g}}\ge1.
\]

因此 reward 分散程度也会改变整体 loss/gradient 尺度。

## 11. 奖励加权扩散损失推导

### 11.1 单条轨迹的去噪误差

对 replay 轨迹 $x_0^{b,g}$，重复 $K$ 次独立采样 $t$ 和噪声 $\epsilon$。当前默认 noise prediction 的逐样本误差为：

\[
\ell_{b,g,k}
=\frac1{TD}
\left\|
\epsilon-\epsilon_\theta(x_t,t,c_b)
\right\|_2^2.
\]

其中 $T=8,D=4$。

### 11.2 当前总 loss

\[
\boxed{
\mathcal L_{RL}
=\frac1{BGK}
\sum_{b=1}^{B}
\sum_{g=1}^{G}
\sum_{k=1}^{K}
e^{A_{b,g}}\ell_{b,g,k}
}
\]

默认 $K=1$。增大 `diffusion_repeat_size` 只会对同一条 replay 轨迹重复采不同的噪声与时间，降低 Monte Carlo 方差，不会生成新的 PDM candidate，也不会重新计算 reward。

### 11.3 为什么它会偏向高奖励轨迹

设 rollout 来自旧模型分布 $p_{old}(\tau\mid c)$。加权回归相当于用经验目标分布：

\[
q(\tau\mid c)
\propto
p_{old}(\tau\mid c)\exp(A(c,\tau)).
\]

高 advantage 轨迹在扩散 denoising objective 中出现更大权重，新模型会更努力拟合这些轨迹附近的数据分布。这是 exponential tilting / reward-weighted regression 的直观解释。

它并不直接最大化：

\[
\mathbb E_{\tau\sim\pi_\theta}[R(\tau)],
\]

也没有使用 REINFORCE 的：

\[
R(\tau)\nabla_\theta\log\pi_\theta(\tau\mid c).
\]

当前方法的优势是无需求扩散采样全过程的精确 log-probability，代价是缺少标准 policy-update 的 trust region、KL 和重要性比控制。

### 11.4 可选 hybrid waypoint loss

代码还保留：

\[
\mathcal L
=\mathcal L_{diff}
+\lambda_{wp}\mathcal L_{wp}.
\]

只有同时满足：

```text
kinematic_type == "diff"
hybrid_loss_weight > 0
```

才计算。`detached_integral()` 在累积 displacement 时，只让最近 $W$ 个差分动作向当前位置误差反传，避免很长的累积梯度链。

当前 RL YAML：

- `kinematic_type=waypoint`；
- 没有 `hybrid_loss_weight`，代码默认 0。

因此实际：

\[
\mathcal L_{wp}=0,
\qquad
\mathcal L=\mathcal L_{RL}.
\]

## 12. 模型哪些参数实际更新

### 12.1 完整 decoder fine-tuning

RL optimizer 参数为：

```python
{"params": list(self.model.decoder.parameters()), "lr": lr}
```

所以 12 层 CustomDiT 的全部 decoder 参数都更新。默认 shell 把学习率覆盖为：

\[
3\times10^{-4}.
\]

YAML 的 Lightning 默认是 $10^{-3}$，但 `run_training_rl.sh` 显式传入 `DP_VLA_LR`，默认 $3\times10^{-4}$。

### 12.2 encoder 冻结方式

训练模型根本不构造 encoder，而不是仅仅将 encoder `requires_grad=False`。encoder hidden state 已离线缓存，所以：

- encoder 没有 optimizer state；
- encoder 不参与 DDP gradient all-reduce；
- 训练速度和显存明显低于端到端 VLM fine-tuning；
- RL 无法适配视觉/语言特征，只能调整 action decoder。

### 12.3 LoRA 当前未启用

`DpVlaModel` 支持：

- positive LoRA adapter；
- negative LoRA adapter；
- 采样时：

\[
f_{cfg}=(1+s)f_{pos}-sf_{neg}.
\]

但 `DpVlaRlAgent.initialize_training()` 没有调用 `model.init_lora_adapter()`，optimizer 也指向原始 `decoder.parameters()`，HF callback 使用 `mode="full"`。所以当前：

- `lora_r=64`、`lora_alpha=16`、`lora_dropout=0` 不影响 RL；
- `cfg_scale=1.0` 不影响 RL rollout；
- positive/negative adapter 不存在；
- 文档/模型注释中的 LoRA RL 能力属于预留路径，不是当前执行路径。

## 13. 训练工程、DDP、优化器与 checkpoint

### 13.1 优化器与 scheduler

- 优化器：AdamW；
- 参数：完整 decoder；
- scheduler：`LambdaLR(lambda _: 1.0)`，即常数学习率；
- YAML 的 `warmup_epochs=10` 会由 Lightning 计算成 `warmup_steps`，但当前 scheduler 完全不使用它；
- gradient clipping：global norm 1.0；
- precision：16-mixed。

### 13.2 DDP 语义

默认 Lightning `strategy=ddp`。每个 rank：

- 拥有自己的 replay buffer；
- 对自己的 DataLoader shard rollout；
- 启动/连接节点本地 Ray；
- 记录 rank-local train metric，因为 `sync_dist=False`；
- update 时 decoder gradient 仍由 DDP 同步。

因此 TensorBoard 的训练 reward/buffer size 不一定代表所有 rank 的全局均值。

### 13.3 Ray CPU 资源

每个节点 local rank 0 启动 Ray，CPU 数量为：

\[
\max(1,\text{os.cpu_count()}-4).
\]

其余 local rank 通过 `address="auto"` 连接，并用 distributed barrier 协调。`ray_reserved_cpus=4` 是为 DataLoader/主进程保留 CPU。

### 13.4 EMA 当前关闭，打开会报错

默认 `use_ema=false`，所以没有影响。但 Lightning `setup()` 中真正创建 `self.agent_ema` 的代码已被注释，而 `on_train_batch_end()`、`on_save_checkpoint()` 在 `use_ema=true` 时仍访问它。

因此仅把配置改成 `use_ema=true` 会触发 `AttributeError`，不是可直接启用的功能。

### 13.5 checkpoint

每 10 epoch 保存：

- Lightning `.ckpt`：含模型和 optimizer state，可恢复训练；
- HF `config.json + model.safetensors`：`mode="full"`。

RL 训练模型以 `with_encoder=False` 构造，所以 HF 导出主要包含 decoder 权重。评估时可重新从 Florence checkpoint 构造 encoder，再以 `strict=False` 覆盖 RL decoder。

checkpoint loader 支持：

- Lightning 文件；
- HF directory；
- 可选 LoRA directory。

但默认 `strict=False`，missing/unexpected keys 只写 warning，不会阻止运行；迁移 checkpoint 时必须检查日志。

## 14. 当前实现中需要重点注意的问题

以下不是理论上的猜测，而是按当前控制流逐段核对得到的实际行为。

### 14.1 高优先级：有效组筛选把场景重复写入 $G$ 次

当前代码：

```python
filter_mask = (rewards > -1.) & ~rewards.isnan().any(dim=1)[:, None]
filter_idx = torch.where(filter_mask)[0]
```

`filter_mask` 形状是 `[B,G]`。如果场景 0 的 10 个 reward 都有效，`torch.where(mask)[0]` 返回：

```text
[0,0,0,0,0,0,0,0,0,0]
```

而不是 `[0]`。后面的 zip 会把同一个完整 `[G,8,4]` group 写入 replay 10 次。

设计意图更接近 scene-level mask：

```python
valid_group = (rewards > -1.).all(dim=1) & ~rewards.isnan().any(dim=1)
filter_idx = torch.where(valid_group)[0]
```

当前后果：

- buffer size 约膨胀 $G$ 倍；
- GPU replay 显存约膨胀 $G$ 倍；
- `Metric/Buffer_Size` 失真；
- 数据统计并没有增加新信息，只是机械复制同一 group。

### 14.2 高优先级：PDM 失败会在 NaN 过滤之前崩溃

remote task 对异常返回 `details=None`。但 `_reward_fn()` 紧接着执行：

```python
details_dicts = [d.__dict__ for d in details]
```

任何一个 `None` 都会触发 `AttributeError`，所以后面的 `rewards.isnan()`、`max_rollout_iter` 和 retry 逻辑根本没有机会处理该失败。

另外，`cache_dict[token]` 在 remote function 的 `try` 之前读取；token 缺失产生的 KeyError 也不会转换为 NaN。

### 14.3 高优先级：retry 控制流基本不能处理“部分失败”

初始 `remaining_token` 长度就是 $B$，while 条件还要求：

```python
len(remaining_token) >= B
```

一旦部分场景成功并从 remaining 中移除，长度便小于 $B$，循环停止，不会重试剩余失败场景。只有“一个场景都没成功”时才可能继续。

默认 `max_rollout_iter=1` 又使 retry 完全只运行一次。

此外，`filter_idx` 是当前 remaining batch 的局部 row index，后续与原始 absolute index 混用；进入第二轮后，删除和回填也可能错位。

### 14.4 高优先级：reward details 被重复拼接两次

先用：

```python
result_details = {k: [d[k] for d in details_dicts] ...}
```

已经收集了一遍，随后 for-loop 又逐条 append 一遍。结果每个 metric 长度从 $BG$ 变为 $2BG$。

均值碰巧通常不变，但：

- 维度语义错误；
- 求和会翻倍；
- `avg_ep` 被重复累计；
- retry 回填索引不再对应场景组。

### 14.5 高优先级：validation 按 NAVSIM v1.1 会失败

`parallel_pdm_scores()` 返回 `List[PDMResults]` dataclass。validation 却执行：

```python
pd.concat(results, ignore_index=True)
```

`pd.concat` 需要 Series/DataFrame，不能直接拼 dataclass。随后还尝试删除 `weighted_metrics`、`weighted_metrics_array`、`pdm_score`，这些也不是 v1.1 `PDMResults` 字段。

外层 broad `except` 会记录 warning 并返回 `{}`，所以不会让训练崩溃，但验证指标为空。默认 `check_val_every_n_epoch=10000000`，常规训练几乎不会暴露此问题。

### 14.6 中高优先级：replay 无容量且驻留 GPU

当前 `ReplayBuffer()` 没有 `maxsize`，也不把 tensor 转 CPU。长数据集、较大 $B/G$ 或修复前述重复写入问题前，都可能造成显存快速增长。

`ReplayBuffer.full()` 在 `maxlen=None` 时执行 `len >= None` 会报 TypeError，不过当前代码没有调用 `full()`。

### 14.7 中优先级：`bc_data` 和 ground truth 是死路径

配置写了 `bc_data: true`，buffer 也存了 GT，但 update 完全不使用。若实验设计认为当前 loss 包含 BC 正则，这一理解是错误的。

### 14.8 中优先级：奖励权重配置是死路径

`30/5/2` 没被读取。调 YAML 不会改变 reward，除非显式把参数接入 `PDMScorerConfig`。

### 14.9 中优先级：`only_ep` 不改变任何训练行为

epoch 开始会计算：

```python
self.only_ep = self.avg_ep / max(self.total_num, 1) < 0.95
```

但之后没有代码读取 `self.only_ep`。而且 `avg_ep` 因 details 重复、候选展平而不是真正的“每场景平均 EP”。这段状态目前对 reward、rollout、loss 都没有作用。

### 14.10 中优先级：指数权重没有温度、clip、归一化

当前最大权重随 group size 增大，且不同 batch 的 loss 总尺度会随 advantage 分布变化。虽然 $G=10$ 的标准化值存在有限上界，但没有明确的 trust-region 控制。

### 14.11 中优先级：LoRA/CFG 配置看似有效，实际未接入

仅修改 `lora_r` 或 `cfg_scale` 不会改变当前 RL，因为 adapter 从未初始化。实验记录若把此训练称为 LoRA fine-tuning 会与代码不符。

### 14.12 中优先级：README 与 evaluation shell 不一致

README 示例使用：

```text
DP_VLA_RL_CKPT
DP_VLA_RL_HPARAMS
```

但 `run_pdm_score.sh` 实际要求：

```text
DP_VLA_CKPT
```

且脚本固定 `agent=dp_vla_agent_base`，没有读取 `DP_VLA_RL_HPARAMS`。脚本中还硬编码了 `/home/tanty/huggingface`。换机器运行前必须调整。

### 14.13 中优先级：启动脚本关键路径默认为空

`run_training_rl.sh` 中：

```bash
PRETRAINED_CKPT_FILE=
CACHE_DATA_LIST_PATH=
```

需要通过 Hydra overrides 覆盖。`env.sh` 的 `NAVSIM_DEVKIT_ROOT` 和 `OPENSCENE_DATA_ROOT` 也是 `/path/to/...` 模板。本机当前 `import navsim` 失败，说明尚未形成可直接复现的运行环境。

### 14.14 低优先级：重复 import 和配置注释漂移

RL agent 顶部重复导入了两次 RL feature/target builder，不影响功能，但反映代码清理不完整。若继续扩展配置，建议用“配置字段是否被读取”的测试防止注释和实际逻辑漂移。

### 14.15 低优先级：ReplayBuffer.get() 的类型分支永远不会进入

代码遍历 `zip(*item_list)` 时，变量 `row` 一定是 tuple，却检查：

```python
isinstance(row, torch.Tensor)
```

所以 `torch.cat()` 分支不会执行，函数总是返回按字段组织的 Python list。当前调用方随后自行 `cat/stack`，因此暂时能工作，但该条件与函数表面意图不一致，后续复用时容易误判返回类型。

## 15. 配置项：声明值与实际效果

| 配置 | 默认值 | 当前是否生效 | 实际含义 |
|---|---:|:---:|---|
| `replay_buffer_update_epoch` | 10 | 是 | 每 10 epoch 清 buffer 并重新 rollout |
| `group_size` | 10 | 是 | 同场景候选数；必须大于 1 才能安全算样本 std |
| `diffusion_repeat_size` | 1 | 是 | update 对同轨迹重复采 $t,\epsilon$ 的次数 |
| `max_rollout_iter` | 1 | 部分 | 控制 while 上限，但默认无重试，部分失败逻辑也有缺陷 |
| `rollout_steps` | 5 | 是 | RL rollout 的 DPM-Solver 步数 |
| `bc_data` | true | 否 | 没有任何 Python 读取 |
| `progress_weight` | 30 | 否 | 未传入 PDM scorer |
| `ttc_weight` | 5 | 否 | 未传入 PDM scorer |
| `comfortable_weight` | 2 | 否 | 未传入 PDM scorer |
| `proposal_sampling` | 4 s / 0.1 s | 是 | PDM simulator/scorer 时间分辨率 |
| `sample_steps` | 25 | 否（RL） | `generate()` 不自动使用；RL 显式传 5 |
| `sample_order` | 2 | 是 | DPM-Solver order |
| `sample_skip_type` | logSNR | 是 | 反向时间步选择 |
| `sample_method` | multistep | 是 | solver 方法 |
| `denoise_to_zero` | true | 是 | 最后去噪到 $t=0$ |
| `time_sampler.sample_method` | uniform | 是 | update 的训练时间均匀采样 |
| `time_sampler.alpha/beta` | 1/1.5 | 否 | uniform sampler 不使用 |
| `model_type` | noise（类默认） | 是 | decoder 预测噪声 |
| `kinematic_type` | waypoint | 是 | 直接生成绝对 waypoint |
| `cfg_scale` | 1.0 | 否（当前 RL） | LoRA adapters 未初始化 |
| `lora_r/alpha/dropout` | 64/16/0 | 否（当前 RL） | LoRA 未初始化 |
| `hybrid_loss_weight` | 缺省→0 | 是但关闭 | waypoint hybrid loss 为 0 |
| `lr` | shell 默认 3e-4 | 是 | 完整 decoder AdamW 学习率 |
| `warmup_epochs` | 10 | 否 | scheduler 恒为 1 |
| `use_ema` | false | 关闭 | 改 true 还会因 `agent_ema` 未创建而报错 |
| `gradient_clip_val` | 1.0 | 是 | Lightning norm clip |
| `precision` | 16-mixed | 是 | 混合精度训练 |

## 16. 训练、缓存与评估的实际操作链

### 16.1 前置依赖

至少需要：

1. 可导入的 NAVSIM 与 nuPlan devkit；
2. NAVSIM/OpenScene 数据；
3. 地图；
4. Florence-2 encoder；
5. 监督 Dp-VLA checkpoint；
6. RL feature cache；
7. 与数据列表逐 token 对齐的 PDM metric cache。

### 16.2 RL feature cache

概念上执行：

```bash
DP_VLA_NPROC=1 ./scripts/training/run_cache_training.sh \
  dp_vla_rl_agent navtrain \
  agent.config.pretrain_config.checkpoint_path=/path/to/pretrained
```

缓存目录每个 `log/token` 至少需要：

- `dp_vla_rl_feature.gz`；
- `dp_vla_rl_target.gz`。

feature 中保存 encoder output 和 meta status；target 中保存 future trajectory。

### 16.3 PDM metric cache

```bash
./scripts/evaluation/run_metric_caching.sh navtrain
```

`navtrain` 自带 `frame_interval: 1`，会覆盖所列 token。训练开始时每个 rank 对 batch 的 unique token 打开 `.lz` 文件并解压，随后放入 Ray object store。

### 16.4 RL 训练

概念上执行：

```bash
DP_VLA_SPLIT=navtrain \
DP_VLA_NPROC=1 \
./scripts/training/run_training_rl.sh \
  agent.config.pretrain_config.checkpoint_path=/path/to/pretrained \
  agent.config.rl_config.data_list_path=/path/to/navtrain.json
```

必须确保以下三者 token 集合一致：

- DataLoader 使用的 RL feature cache；
- `rl_config.data_list_path`；
- `metric_cache_path`。

### 16.5 评估边界

仓库提供的 `run_pdm_score.sh` 调用上游 NAVSIM evaluator，但当前 README 环境变量与脚本不一致。实际运行前应以脚本为准或先统一入口。

RL agent 自身的 `validation_step()` 当前存在 dataclass 聚合问题，不能把其空结果当作“指标为零”，而应当视为 validation 失败被捕获。

## 17. 本次静态验证与证据边界

### 17.1 已执行的检查

- `dp_vla_rl_agent.py`、`scoring.py`、`rl_utils.py`、`diffusion_sde.py` 通过 `py_compile`；
- 四个训练/缓存/评估 shell 通过 `bash -n`；
- 用独立 PyTorch 片段验证：`torch.where([B,G] mask)[0]` 会重复 row index；
- 用独立 PyTorch 片段验证：`group_size=1` 时默认 `torch.std()` 为 NaN；
- 核对 NAVSIM v1.1 官方 PDM scorer 和 `PDMResults` 源码；
- 全仓搜索确认 reward weight、`bc_data`、`only_ep`、LoRA 初始化的实际引用情况；
- 当前 `HDP-navsim` 没有发现 checkpoint、TensorBoard event、CSV 或训练日志产物。

### 17.2 未能执行的检查

当前环境：

```text
navsim import: FAIL ModuleNotFoundError
```

且 `env.sh` 仍是：

```text
NAVSIM_DEVKIT_ROOT=/path/to/navsim
OPENSCENE_DATA_ROOT=/path/to/navsim-dataset
```

因此没有执行：

- Hydra agent 实例化；
- Florence-2 checkpoint 加载；
- RL cache round-trip；
- Ray + PDM 真实 reward；
- 一步 Lightning backward；
- 完整 rollout/update 周期；
- NAVSIM PDMS 前后对比。

本文对代码语义和公式的结论有直接源码依据，但对“可运行性”和“性能提升”不作未经实验支持的结论。

## 18. 分块理解主代码

### 18.1 初始化块

`initialize_training()`：

1. 用 `with_encoder=False` 构造 decoder-only 模型；
2. 加载监督 checkpoint；
3. 创建无限容量 replay；
4. 读取 JSON data list；
5. 创建 feature cache 与 metric cache loader；
6. 创建 4 秒/0.1 秒 PDM simulator/scorer；
7. 初始化节点本地 Ray；
8. optimizer 参数绑定完整 decoder。

关键点：`strict=False` 允许从带 encoder 的监督 checkpoint 中只加载 decoder 对应权重。

### 18.2 rollout 块

1. 复制场景条件 $G$ 份；
2. 加载 batch unique token 的 metric cache；
3. 5 步 DPM-Solver 生成 $BG$ 条候选；
4. epoch 0 添加局部纵/横向偏移；
5. 转 `[x,y,heading]`；
6. Ray 并行 PDM；
7. 尝试过滤失败组；
8. 保存 token、rollout group、GT group、reward group；
9. 返回 reward/metric，不返回 loss。

关键风险集中在步骤 7–8 的 scene index 与 candidate index 混用。

### 18.3 update 块

1. 从 replay 有放回抽 $B$ 个 group；
2. 按 token 从磁盘重载 encoder/proprio；
3. 对 reward 做场景内样本标准化；
4. 展平为 $BG$，再重复 $K$ 次；
5. 对 replay rollout 重新采 $t,\epsilon$；
6. decoder 预测 noise；
7. 每条轨迹计算 MSE；
8. 乘 $e^A$ 后求均值；
9. 可选 difference trajectory hybrid loss；
10. Lightning backward、clip、AdamW、DDP all-reduce。

关键点：PDM reward 在 update 中只作为常数权重，不参与梯度图。

### 18.4 生命周期块

rollout epoch 开始：

```python
replay_buffer.clear()
only_ep = avg_ep / total_num < 0.95
avg_ep = 0
total_num = 0
```

`only_ep` 当前没有消费者，因此生命周期实际有效操作只有清空 buffer 和重置无效统计。

### 18.5 validation 块

设计意图是：单轨迹生成 → PDM → 每个子指标均值。但当前 `pd.concat(dataclass list)` 不匹配 v1.1 返回类型，异常被吞成 `{}`。训练默认又几乎关闭 validation，所以需要独立评估入口验证 checkpoint。

## 19. 面试或项目评审中可能被问到的问题

### 19.1 为什么这不算 PPO 或完整 GRPO？

因为没有 policy log-prob、old/reference policy、ratio、clip 和 KL。它只借用了组内相对奖励，优化的是 reward-weighted diffusion denoising regression。

### 19.2 为什么用组内标准化，不直接用绝对 PDMS？

不同场景难度差异很大。组内标准化强调同一场景下相对更优的候选，减少简单场景高绝对分对梯度的支配。但它会丢失跨场景绝对质量信息。

### 19.3 为什么使用指数权重？

$e^A$ 保证权重为正，并形成 exponential tilting，使高优势轨迹被更强拟合。缺点是梯度尺度敏感，通常需要 temperature、clip 或归一化。

### 19.4 为什么不直接对 PDM reward 反向传播？

PDM 包含几何、碰撞规则、地图查询、LQR simulation 和离散逻辑，不是可微计算图；代码还通过 NumPy、Ray 和 dataclass 运行。当前通过黑盒打分后加权可微的 diffusion loss 绕开这一问题。

### 19.5 PDM reward 是闭环 reward 吗？

它会对自车 proposal 做动力学仿真，并应用类似闭环的安全/舒适/进度指标，但其他交通参与者来自缓存 observation，不会对候选自车实时反应。因此更准确是 pseudo-closed-loop、non-reactive PDM reward。

### 19.6 为什么 encoder 离线缓存？

Florence-2 编码成本高。缓存 hidden states 可把 RL 计算集中在 decoder 和大量 PDM rollout 上，显著降低 GPU 成本。缺点是 encoder 无法通过 reward 适配，且必须处理 cache/checkpoint/mask 一致性。

### 19.7 当前是 on-policy 还是 off-policy？

每 10 epoch 的 rollout 是当前模型产生的，具有 on-policy 起点；随后 9 epoch 反复训练同一 buffer，模型改变后这些轨迹逐渐变成 stale/off-policy。它是周期性刷新 replay 的混合形式。

### 19.8 GT trajectory 为什么存在却没有作用？

buffer item 保留了 GT，YAML 也有 `bc_data`，说明可能计划加入 BC 稳定项。但当前 update 用 `_` 丢弃 GT，属于未完成/未接入功能。

### 19.9 为什么 `group_size=1` 不安全？

当前使用样本标准差，分母为 $G-1$。$G=1$ 时标准差是 NaN。至少要断言 $G>1$，或显式使用 `correction=0` 并设计单候选情形。

### 19.10 当前 reward weight 到底是多少？

实际是 NAVSIM v1.1 `PDMScorerConfig()` 默认 `EP/TTC/Comfort/DDC = 5/5/2/0`；YAML `30/5/2` 未传入，当前不生效。

### 19.11 为什么 `sample_steps: 25` 和 rollout 不一致？

`DiffusionSDE` 虽保存该配置，但 `DpVlaModel.generate()` 要求调用方传 `steps`，没有读取 `diffusion_sde.sample_steps`。RL 明确传 5，普通 generate 默认 10。

### 19.12 最大的工程风险在哪里？

首先是 group filter 重复写入和 GPU replay 无上限，其次是 PDM failure/validation 聚合错误，再次是配置字段未生效导致实验记录与真实算法不一致。

### 19.13 如何证明 RL 真正有效？

至少需要：

1. 修复/回归测试 rollout、reward、replay、validation；
2. 固定监督 checkpoint、数据 split、seed 和 NAVSIM 版本；
3. 记录 rollout reward 分布而非只看加权 loss；
4. 在独立 navtest 上用官方 PDMS 评估；
5. 多 seed 报告均值和方差；
6. 做 `G`、rollout steps、refresh period、weight temperature、BC regularizer 消融；
7. 比较监督基线、无 reward 普通 self-training、绝对 reward 加权和 group-relative 加权。

## 20. 建议的最小验证清单

若后续要让该 RL 链路成为可复现实验，建议按优先级验证：

1. scene-level valid mask 确保每场景只写入一个 group；
2. PDM `None/NaN` 可被过滤并真正重试；
3. `result_details` 长度严格等于 $BG$，不重复；
4. replay item 全部 detach 后转 CPU，并设置容量；
5. `group_size=1` 明确报配置错误；
6. 自定义 reward weight 真正传给 `PDMScorerConfig`，或删除误导字段；
7. `bc_data` 要么实现，要么删除；
8. validation 用 `dataclasses.asdict()` 转成 DataFrame/字典；
9. 测试 5/10/25 sampling steps 的实际调用值；
10. 确认 cache encoder checkpoint 与 RL 初始化 checkpoint 一致；
11. 保存并核对 attention mask；
12. 加一轮 smoke test：1 个 batch rollout + 1 个 batch update + checkpoint reload + PDM eval；
13. 固定 NAVSIM tag/commit，避免 scorer 公式随上游变化；
14. 统一 README、训练 shell 和评估 shell 的环境变量。

## 21. 最终总结

当前 HDP-NAVSIM 的 RL 核心可以压缩为三条公式。

第一，PDM reward：

\[
R_{b,g}
=NC_{b,g}DAC_{b,g}
\frac{5EP_{b,g}+5TTC_{b,g}+2C_{b,g}}{12}.
\]

第二，组内优势：

\[
A_{b,g}
=\frac{R_{b,g}-\mu_b}{s_b+10^{-6}}.
\]

第三，奖励加权扩散回归：

\[
\mathcal L
=\frac1{BGK}
\sum_{b,g,k}
e^{A_{b,g}}
\left\|
\epsilon-\epsilon_\theta(x_t,t,c_b)
\right\|_2^2.
\]

这条路线的价值在于：不需要让 PDM 可微，也不需要计算扩散轨迹精确 log-probability，就能把黑盒规划指标反馈到 diffusion decoder。它的本质是对当前策略样本进行 reward-guided distribution fitting，而不是标准 policy-gradient RL。

当前代码已经具备模型、分组采样、PDM、replay、加权 loss、DDP 和 checkpoint 等主要模块，但关键控制流仍存在足以影响训练正确性与资源占用的问题；同时没有本地端到端实验产物可证明现有状态已跑通。因此对该实现最准确的判断是：

> **算法骨架完整，PDM 接入真实，但当前版本仍属于需要修复并补齐回归测试的实验性 RL 微调实现。**
