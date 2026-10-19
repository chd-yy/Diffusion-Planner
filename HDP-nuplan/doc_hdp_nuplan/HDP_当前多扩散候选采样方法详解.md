# HDP 当前多扩散候选采样方法详解

> 整理时间：10月07日19时  
> 审计对象：当前 `codex/phd-thesis-research` 工作树中的实现，以及已经完成的 306,801 场景全量后训练实际配置  
> 当前方法简称：**Guided Safe-Set 88→32 多候选采样**  
> 文档性质：代码与实验事实审计；“代码支持的能力”和“本次实际启用的配置”分开陈述

配套可编辑流程图：[HDP_多扩散候选采样方法流程图.drawio](figures/HDP_多扩散候选采样方法流程图.drawio)

## 1. 一句话结论

当前 RL 后训练并不是对同一场景简单地以 `noise_scale=0.1` 重复采样 32 次，而是先为每个场景生成一个由四个分支组成的 **88 条 proposal pool**：

1. 冻结参考策略自然采样 16 条；
2. 当前 EMA 策略自然采样 16 条；
3. 当前 EMA 策略以中等噪声采样 24 条；
4. 当前 EMA 策略进行场景自适应纵向 guided denoising，采样 32 条。

随后，系统对 88 条候选统一计算 reward、碰撞与可行驶区域资格以及五维质量向量，再按“自然分布保留 + 中噪声高分 + 中噪声质量多样性 + 全局高分”的规则压缩为固定的 32 条 replay 候选。最后，只有同时满足硬安全资格、参考策略任务保持约束、组内质量离散度门和参考相对优势门的候选，才会成为 RL 回归目标。

因此，当前方法应准确描述为：

> **多源、中等尺度探索与训练期纵向引导相结合的候选池生成，经硬安全资格和质量多样性压缩后，构造参考策略约束下的扩散后训练目标。**

## 2. “当前方法”的依据与适用范围

本文档所说的“当前”，特指下列已经完成的全量后训练运行：

- 预训练起点：`model_epoch_10_trainloss_0.0091.pth`，即 B10；
- 场景数：306,801；
- 随机种子：2026；
- rollout 阶段：Epoch 1；
- update 阶段：Epoch 2；
- replay backend：`disk_sharded`；
- 实际配置：`rl_guided_safe_set_protocol=true`；
- 配置文件：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/training_outputs/EXP_CH4_guided_safeset_full306801_seed2026_v3_1006/training_log/chapter4-safety_reference-noise0p1-std1em06-steps0-seed2026-guided-safeset88-full306801/2026-10-06-00-13-19/args.json`；
- 运行日志：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/training_outputs/EXP_CH4_guided_safeset_full306801_seed2026_v3_1006/train.log`。

需要区分三种不同概念：

| 概念 | 数量 | 含义 |
|---|---:|---|
| proposal pool | 88 | reward 评分与安全筛选前生成的原始候选池 |
| replay group | 32 | 从 88 条候选压缩后写入 replay 的固定形状候选组 |
| update targets | 每场景可变 | 32 条中最终允许参与 RL 回归的候选子集 |

“每个场景 32 个候选”描述的是 replay group，不等于最初只生成 32 条，也不等于 32 条全部参与更新。

## 3. 完整数据流

```text
单个 nuPlan 训练场景
        │
        ├── 场景编码器：只需构造共享条件
        │
        ├── 冻结 B10 reference，自然采样 16 条，noise=0.1
        ├── 当前 EMA policy，自然采样 16 条，noise=0.1
        ├── 当前 EMA policy，中噪声采样 24 条，noise=0.5
        └── 当前 EMA policy，纵向引导采样 32 条，noise=0.1
                         │
                         ▼
                  88 条 proposal pool
                         │
                         ├── NuPlan 对齐 proxy reward
                         ├── collision / DA 硬安全资格
                         └── 安全、进度、路线、舒适、跟车五维质量
                         │
                         ▼
               source-aware safe-set compression
                         │
        ┌────────────────┼────────────────┬────────────────┐
        ▼                ▼                ▼                ▼
   自然候选槽位      中噪声高分槽位    中噪声多样槽位     全局高分槽位
      8 条              8 条              8 条              8 条
        └────────────────┴────────────────┴────────────────┘
                         │
                         ▼
                    32 条 replay group
                         │
                         ├── 硬安全 mask
                         ├── 相对 reference 的路线/进度/reward 约束
                         ├── 五维质量离散度门
                         └── reference-relative advantage ≥ 0.1
                         │
                         ▼
                  可变数量的 update targets
                         │
                         ▼
          RL rollout regression + expert anchor + reference anchor
```

## 4. 底层扩散候选是怎样生成的

### 4.1 输入、输出与候选维

统一采样入口位于：

- `hdp_nuplan/model/module/decoder.py` 的 `Decoder.sample()`；
- DPM-Solver 实现位于 `hdp_nuplan/model/diffusion_utils/sampling.py`。

采样器输出：

$$
\mathbf{X}\in\mathbb{R}^{B\times G\times T\times 4},
$$

其中：

- $B$：batch 内场景数；
- $G$：同一场景候选数；
- $T=80$：未来轨迹时间点数；
- 最后一维为 $[x,y,\cos\psi,\sin\psi]$。

场景编码、route lanes 和自车速度条件会按候选数 $G$ 复制，再将场景维和候选维展平为 $B\times G$，一次送入扩散采样器。

### 4.2 每条候选都使用独立高斯噪声

底层噪声张量直接按 `[B,G,T,4]` 生成：

$$
\epsilon_{b,g}\sim\mathcal{N}(0,I),\qquad
x_T^{b,g}=s_g\epsilon_{b,g}.
$$

不同候选 $g$ 的噪声是独立抽样的，不存在把同一条噪声复制 32 次的实现错误。`noise_scale` 只改变初始终端噪声的幅度。

当前 rollout 使用：

- DPM-Solver++；
- 二阶求解；
- `logSNR` 时间步；
- multistep；
- `denoise_to_zero=true`；
- 每条候选 6 个去噪步。

### 4.3 轨迹表示恢复

当前 B10 使用 increment trajectory representation。模型产生标准化的逐帧增量后，采样器依次执行：

1. normalizer inverse，恢复真实量纲；
2. increment representation decode；
3. 累积得到 ego-local absolute waypoints。

reward 与后续安全判断使用的是解码后的实际轨迹，而不是归一化空间中的噪声或增量。

## 5. 当前 88 条候选池的四个分支

| 分支 | 模型 | 数量 | 噪声尺度 | Guidance | 主要作用 |
|---|---|---:|---:|---|---|
| reference natural | 冻结参考策略 | 16 | 0.1 | 无 | 提供 B10 原有策略分布与参考基线 |
| policy natural | 当前 EMA 策略 | 16 | 0.1 | 无 | 表示当前策略的自然低噪声支持集 |
| policy medium | 当前 EMA 策略 | 24 | 0.5 | 无 | 直接扩大同场景几何支持范围 |
| guided | 当前 EMA 策略 | 32 | 0.1 | 场景自适应纵向 guidance | 定向产生减速、保持或加速候选 |
| 合计 | — | **88** | — | — | 评分和压缩前的 proposal pool |

代码入口为 `hdp_nuplan/rl/train_epoch_rl.py` 中的 `_sample_guided_safe_set_pool()`。

### 5.1 reference 与 current policy 在本次全量运行中的真实关系

冻结 reference 是在载入 B10 后、DDP 包装前对模型进行深拷贝得到的；rollout policy 使用 EMA 模型。

本次全量实验只执行了一轮 rollout，然后执行一轮 update。因此，在这一轮 306,801 场景候选生成时：

- 冻结 reference 的权重来自 B10；
- 当前 EMA policy 的权重也尚未经过本轮 update，仍来自 B10；
- 两个自然分支的模型权重实际相同，但高斯噪声各自独立采样；
- 两个分支的“reference/current”语义为后续多轮策略迭代预留，本次不是通过两套不同权重制造多样性。

如果以后进行“更新后再次 rollout”的多周期训练，reference 仍冻结在 B10，而 current EMA 才会逐步偏离 reference。

### 5.2 为什么只使用 0.1 和 0.5

早期单尺度 `noise=0.1` 的 32 候选支持范围过窄；旧 Repair 实现曾使用最高到 2.0 的多尺度池，但高噪声容易产生远离基础驾驶分布的低质量轨迹。

当前方案的取舍是：

- 保留 `0.1`，维护 B10 的高概率自然轨迹分布；
- 加入 `0.5`，以有限幅度增加几何多样性；
- 不加入 1.0、1.5、2.0 的高噪声探索；
- 使用 guided branch 代替无方向的高噪声搜索，定向改变纵向行为。

## 6. Guided denoising 的具体机制

实现位于：

- `hdp_nuplan/model/guidance/longitudinal.py`；
- `hdp_nuplan/rl/train_epoch_rl.py` 中的 `_temporal_forward_conflict_mask()` 与 `_scene_adaptive_guidance_deviations()`。

### 6.1 纵向引导能量

专家未来轨迹给出每个时间点的 Frenet 切向 $\mathbf{t}_t^{E}$ 和专家速度 $v_t^{E}$。候选轨迹在该切向上的速度为：

$$
v_{\parallel,t}=\mathbf{v}_t\cdot\mathbf{t}_t^{E}.
$$

对第 $g$ 条 guided 候选分配速度偏差 $\delta_g$，目标纵向速度为：

$$
\hat v_{t,g}=(1+\delta_g)v_t^{E}.
$$

引导能量为：

$$
E_g=\gamma_s\cdot\frac{1}{T}
\sum_{t=1}^{T}
\left(v_{\parallel,t,g}-(1+\delta_g)v_t^{E}\right)^2,
$$

其中 $\gamma_s\in\{0,1\}$ 是场景级 guidance gate。DPM-Solver 以 classifier guidance 方式使用该能量，当前 `guidance_scale=0.01`。

切向随专家路径逐时刻变化，而不是用当前自车朝向固定整个 8 秒时域，因此弯道中的横向转弯不会被误当成纵向速度误差。

### 6.2 三类场景使用不同的偏差梯子

基础偏差集合为：

$$
\Delta=\{-0.5,-0.25,-0.125,0,0.125,0.25,0.5\}.
$$

其含义例如：

- $\delta=-0.25$：目标约为专家纵向速度的 75%；
- $\delta=0$：目标与专家速度一致；
- $\delta=+0.25$：目标约为专家纵向速度的 125%。

32 条 guided proposal 按场景分为：

| 场景类型 | 判定 | 使用的偏差集合 | 目的 |
|---|---|---|---|
| 低运动场景 | 专家归一化路线进度不超过 5.358 | gate=0，偏差全为 0 | 禁止在停车/低运动场景强行加减速 |
| 存在前向冲突 | 运动场景且未来任一时刻走廊内有前车/切入目标 | `{-0.5,-0.25,-0.125,0}` | 只探索减速或保持，不探索加速 |
| 无前向冲突 | 运动场景且无上述冲突 | 完整对称集合 $\Delta$ | 同时探索减速、保持和加速 |

注意：阈值 `5.358` 使用的是 reward scorer 中归一化后的 route-aligned progress，不是 5.358 m。默认 `progress_normalization=10` 时，它约对应 53.58 m 的累计路线方向位移。

### 6.3 前向冲突怎样判定

以专家未来轨迹的逐时刻切向构造 Frenet 走廊。只要某个有效邻车在任一重叠未来时刻满足：

$$
|d_{lat}|<3.5\ \mathrm{m},\qquad
0\le d_{lon}<15\ \mathrm{m},
$$

该场景就判为 forward conflict。该判断覆盖完整未来时域，因此晚些时候发生的 cut-in 也会禁止正向加速 guidance。

### 6.4 Guidance 的信息边界

当前 guidance 使用了缓存中的：

- 专家未来轨迹；
- 邻车未来轨迹。

这些属于训练阶段可用的 privileged information。因此：

- 它只用于构造 RL 训练候选；
- 普通 planner 推理不会传入 `guidance_fn`；
- 它不是部署时需要在线运行的模块；
- 论文中应称为“训练期 proposal guidance”，不能表述为在线闭环规划器直接读取未来真值。

## 7. 88 条候选如何进行安全资格判定

资格判定实现位于 `hdp_nuplan/rl/candidate_safe_set.py`。

对候选 $i$，当前硬安全资格为：

$$
H_i=mathrm{Finite}_i
\land \mathrm{NoCollision}_i
\land(\neg \mathrm{DAActive}\lor \mathrm{DACompliant}_i).
$$

含义为：

1. 轨迹数值必须有限；
2. 不允许发生 proxy collision；
3. 当该场景的 drivable-area proxy 有效时，候选必须满足 DA compliance；
4. 当专家本身无法可靠激活 DA proxy 时，不强行使用错误地图代理淘汰所有候选。

当前**没有**作为硬资格门的项目包括：

- TTC：配置的硬阈值为 0，因此未形成 TTC 硬淘汰；
- speeding：缓存中没有独立 speeding 事件标签；
- comfort：作为 reward/质量维度，而非硬排除；
- wrong direction：通过路线/进度相关约束处理，而非独立事件硬门。

不合格候选可能为了保持 tensor 固定形状而被写入 replay，但对应 mask 为 false，不能成为 RL 回归目标。

## 8. Reward 与五维质量表征

当前使用 `reward_objective_mode=nuplan_aligned`。总 reward 是五维质量的加权几何平均：

$$
R_i=
q_{safe,i}^{0.45}
q_{progress,i}^{0.25}
q_{route,i}^{0.15}
q_{comfort,i}^{0.10}
q_{follow,i}^{0.05}.
$$

其中：

| 质量维度 | 当前实现 |
|---|---|
| safety | `risk_reward × no_collision` |
| progress | `progress_guard_reward` |
| route | `lane_reward` |
| comfort | `exp(-comfort_cost)` |
| following | `effective_follow_quality` |

这五个量同时组成候选质量向量：

$$
\mathbf{q}_i=[q_{safe},q_{progress},q_{route},q_{comfort},q_{follow}]_i,
$$

供后续的质量多样性选择与 component dispersion gate 使用。

需要准确表述：该 reward 是与 NuPlan 指标方向对齐的离线 proxy，不等于官方闭环 score，也不能单独保证闭环碰撞率不退化。

## 9. 88→32 的 source-aware safe-set 压缩

压缩实现为 `safe_set_compression_indices()`。四类槽位各 8 条，总计 32 条。

### 9.1 第一阶段：保留 8 条安全自然候选

从 reference natural 与 policy natural 中只取 hard-safe 候选，并在两种来源之间交替选择，累计选到 8 条。

目的：保留基础 B10/当前策略的自然高概率分布，避免 32 条全部被探索分支占据。

### 9.2 第二阶段：保留 8 条中噪声 reward elite

在 hard-safe 的 `noise=0.5` 候选中按 reward 从高到低排序，将总选择数补到 16 条。

目的：保留中等探索范围内最有任务价值的候选。

### 9.3 第三阶段：保留 8 条中噪声质量多样候选

仍然只在 hard-safe 的中噪声候选中，基于五维质量向量执行 farthest-point selection：每次选择距离已有候选集合最近距离最大的候选，将总选择数补到 24 条。

形式上，第 $k$ 次选择：

$$
i_k=\arg\max_{i\in\mathcal{C}_{medium}}
\min_{j\in\mathcal{S}_{k-1}}
\lVert\mathbf{q}_i-\mathbf{q}_j\rVert_2.
$$

目的：这里追求的是 safety/progress/route/comfort/follow 的行为质量差异，不只是末端坐标距离。

### 9.4 第四阶段：保留 8 条全局安全高分候选

从所有来源的 hard-safe 候选中按 reward 排序，将选择结果补到 32 条。guided 候选主要通过这一全局槽位进入最终集合，也可以在安全候选回填时进入。

目的：在保留自然支持和中噪声多样性的基础上，使最终集合仍包含全局最优轨迹。

### 9.5 槽位是累计目标，不是不可跨越的刚性配额

“8+8+8+8”描述的是各阶段的累计选择上限。若某类来源没有足够 hard-safe 候选，后续阶段会用其他 hard-safe 候选回填。因此最终来源计数不保证机械地等于 8/8/8/8。

安全优先级为：

```text
hard-safe 未重复候选
    > finite 但不安全候选（只作存储填充）
    > 重复索引 padding（mask=false）
```

若一个场景连有限候选都没有，系统以零值存储并把该组变成 anchor-only inactive group，避免 NaN 污染训练。

## 10. 32 条候选中哪些才是训练目标

### 10.1 冻结参考策略的场景级基线

在最初 16 条 reference natural proposal 中，选择 hard-safe 且 aligned reward 最高的一条，得到该场景的：

- 参考 reward $R_{ref}$；
- 参考 route cost $C_{route}^{ref}$；
- 参考 progress guard $Q_{prog}^{ref}$。

若 reference 分支没有安全候选，该场景不能产生 rollout 更新目标。

### 10.2 任务保持约束

压缩后的候选 $i$ 必须同时满足：

$$
H_i=1,
$$

$$
C_{route,i}\le C_{route}^{ref}+10^{-4},
$$

$$
Q_{prog,i}\ge Q_{prog}^{ref}-0.005,
$$

$$
R_i\ge R_{ref}.
$$

这一步排除“reward 看似较高但路线或进度相对参考策略退化”的候选。

### 10.3 场景级五维质量离散度门

只有候选组内至少有 2 条 hard-safe 候选，且五维质量加权离散度不小于 0.01，该场景才被判为 active group。

该门的目的不是要求几何距离必须很大，而是要求同组候选在实际优化的质量维度上存在足够差异；否则组内排序基本没有可利用信息。

### 10.4 Reference-relative advantage 门

当前 advantage 以参考策略为中心：

$$
A_i=\frac{R_i-R_{ref}}{\sigma_{safe}+10^{-6}},
$$

其中 $\sigma_{safe}$ 是场景内安全候选 reward 的标准差。最终回归目标还要求：

$$
A_i\ge0.1.
$$

当前使用 `constraint_uniform_positive`：通过全部约束的正向目标按统一正权重参与 rollout regression，而不是让低分或不安全候选产生反向模仿梯度。

## 11. 当前实际参数汇总

| 参数 | 当前值 |
|---|---:|
| rollout diffusion steps | 6 |
| proposal pool size | 88 |
| reference natural | 16 @ noise 0.1 |
| policy natural | 16 @ noise 0.1 |
| policy medium | 24 @ noise 0.5 |
| guided | 32 @ noise 0.1 |
| guidance scale | 0.01 |
| guidance deviations | `-0.5,-0.25,-0.125,0,0.125,0.25,0.5` |
| guidance progress threshold | 5.358（归一化 progress） |
| conflict corridor half-width | 3.5 m |
| conflict corridor longitudinal range | 15 m |
| compression | 8 natural + 8 medium elite + 8 medium diverse + 8 global elite |
| replay group size | 32 |
| route tolerance | $10^{-4}$ |
| progress tolerance | 0.005 |
| reference reward margin | 0 |
| positive advantage margin | 0.1 |
| component dispersion threshold | 0.01 |
| reward std epsilon | $10^{-6}$ |
| weighting mode | `constraint_uniform_positive` |
| rollout loss weight | 1.0 |
| expert anchor weight | 0.1 |
| reference anchor weight | 0.1 |
| replay target sampling fraction | 0.5 |
| minimum target-bearing group gate | 30,000 |

## 12. 306,801 场景全量 rollout 的实际统计

下表来自该次正式运行日志，不是设计目标或估计值。

| 指标 | 实际结果 | 含义 |
|---|---:|---|
| proposal pool size | 88 | 每个场景评分前生成 88 条 |
| guidance gate fraction | 26.92% | guided branch 真正开启引导的场景比例 |
| free symmetric fraction | 14.57% | 可对称加减速引导 |
| conflict deceleration fraction | 12.35% | 只允许非正偏差引导 |
| low-motion natural fraction | 73.08% | guided branch 关闭引导，退化为低噪声自然采样 |
| final safe-set fraction | 92.67% | 压缩后 32 条中的 hard-safe 比例 |
| update-candidate fraction | 24.35% | 通过参考任务保持初筛的候选比例 |
| component-active groups | 37,145 / 306,801 = 12.11% | 五维质量差异达到门槛的组 |
| target-bearing groups | 30,159 | 最终至少有一条训练目标的组 |
| active→target retention | 81.19% | active group 中保留为目标组的比例 |
| targets per target-bearing group | 5.56 | rollout 数据中每个目标组的平均目标数 |
| target group quality gate | 30,159 ≥ 30,000 | 达到正式训练最低门槛 |

update 阶段对全部 30,159 个 target-bearing group 做了覆盖式采样，共执行 30,159 个 optimizer steps；每个目标组至少被覆盖一次。

这些结果支持以下有限结论：

1. 当前候选构造已经能够在完整 306,801 场景中产生超过 30,000 个可训练目标组；
2. 它解决了“几乎没有可用学习目标、无法进行全量后训练”的工程与信号覆盖问题；
3. 它不等于已经证明后训练策略在闭环安全性上优于 B10。

补充的validation-1000冻结B10正式复核表明，当前`88→32`方法相对原始`0.1×32`将终点与整轨迹两两距离P50分别提高到2.64倍和2.22倍，候选平均ADE改善0.161%，Oracle ADE改善0.101478 m；但未达到旧Repair预注册的0.25 m、0.10 m和40%有效reward组等强多样性门槛。完整结果见[validation-1000当前多扩散候选采样对照实验](phd_midterm/stage_results/EXP_CH4_guided_safe_set_candidate_validation/validation1000多扩散候选采样对照实验.md)。

## 13. 与历史候选方法的区别

| 方法 | 候选构造 | 当前是否启用 | 说明 |
|---|---|---|---|
| 原始单尺度组采样 | 32 条，全部 `noise=0.1` | 否 | 候选几何差异和 reward 排序信号不足 |
| 旧 Repair 多尺度池 | 200 条：`0.1×40+0.5×64+1.0×32+1.5×32+2.0×32`，再压缩到 32 | 代码保留，当前关闭 | `rl_candidate_pool_profile=""`、`rl_candidate_selection_profile=""` |
| 当前 Guided Safe-Set | 四分支 88 条，再安全/质量压缩到 32 | **是** | 本次 306,801 全量后训练实际方案 |
| trajectory augmentation | 对候选再做轨迹扰动 | 代码支持，当前关闭 | 本次 augmentation std/epochs 为 0 |
| 高噪声低概率模式探索 | noise 1.0–2.0 分支 | **无** | 当前方案明确不包含该部分 |
| 推理时 best-of-N 选优 | 在线生成多条后再选最好 | 否 | 当前 planner 闭环推理仍使用常规单候选 |
| 推理时 guided denoising | 在线 guidance | 否 | guidance 只服务训练期 proposal collection |

因此，不能再用“当前方法是 200 条多尺度候选压缩为 32 条”来描述正式全量后训练，也不能把 88 条 proposal 误写成部署时的在线规划开销。

## 14. 为什么这样设计

当前设计针对原始 `noise=0.1, G=32` 的两个主要问题：

### 14.1 原问题一：候选几何支持过窄

历史诊断中，B10 在 validation-1000、32 候选、`noise=0.1` 下：

- 终点两两距离 P50 仅 0.055530 m；
- 整轨迹两两距离 P50 仅 0.025200 m；
- 只有 12.6% 场景达到 `reward std≥0.01`。

只增加 group size 会在同一个狭窄分布里产生更多相似样本，不会自动产生有意义的行为差异。

### 14.2 原问题二：无方向增加噪声会牺牲候选质量

直接把噪声提高到 1.0–2.0 可以增加几何距离，但更容易产生离路、碰撞或明显偏离基础策略的候选。当前方法选择：

- 用 `noise=0.5` 提供有限的非定向扩展；
- 用 longitudinal guidance 提供任务相关的定向扩展；
- 用 hard-safe gate 先保证基本安全资格；
- 用五维质量多样性而非单纯坐标距离选择训练支持集；
- 用 reference-relative constraints 防止把“变化大”误当作“变化正确”。

这正对应论文第四章的科学问题：候选不仅要“不同”，还必须在安全集合内形成能够支持策略改进的、方向正确的优势信号。

## 15. 当前方法已经解决了什么，尚未解决什么

### 15.1 已解决或已有直接证据支持的部分

- 每条候选独立采样，不存在 group 维复制错误；
- 候选池不再局限于单一 `noise=0.1`；
- 可以生成中等尺度与方向性纵向变化；
- 安全候选在 88→32 压缩中始终优先；
- 不安全 padding 不会进入 RL 回归目标；
- 全量数据上形成 30,159 个 target-bearing groups，满足 30,000 组的训练门槛；
- 全量 replay、覆盖式目标组采样和完整 update 已实际跑通。

### 15.2 尚未解决的部分

正式 Test14 结果表明，新模型相对 B10：

- hard 与 random 的总体 score、route progress 有改善；
- collision 指标仍有退化；
- 因此新 checkpoint 未被接受为替代 B10 的正式模型。

这意味着：

> 当前多候选方法已经让后训练“有信号、能更新、能改变策略”，但还没有证明这些变化在官方闭环安全指标上始终方向正确。

可能的剩余原因包括：

1. 离线 collision/DA proxy 与官方闭环碰撞指标仍存在错配；
2. 73.08% 场景关闭 guidance，guided 分支在大多数场景中仍是额外的低噪声自然样本；
3. 当前 guidance 只调纵向速度，不直接构造横向避障或路权决策差异；
4. 训练目标门保证候选优于离线 reference proxy，但不能给出闭环状态分布下的单调改进保证；
5. 本次只有一次 rollout-update，没有验证多轮重新采样后 current policy 与 frozen reference 分离时的稳定性。

## 16. 论文中建议采用的准确表述

可以写成：

> 针对低噪声重复采样导致同场景候选支持范围狭窄、组内优势信号不足的问题，本文构造了由冻结参考策略自然候选、当前策略自然候选、中等噪声候选和场景自适应纵向引导候选四个分支组成的候选池。各分支分别承担基础策略支持保持、当前策略分布刻画、有限尺度几何探索和任务相关纵向行为探索。随后，在碰撞与可行驶区域硬资格约束下，综合自然支持、中噪声奖励精英、五维质量最远点样本与全局奖励精英，将 88 条候选压缩为 32 条固定训练候选。最终，仅将同时满足参考策略路线保持、进度保持、奖励非退化、质量离散度及正参考相对优势条件的候选作为扩散后训练回归目标。

不建议写成：

- “随机生成 32 条轨迹后选择最好的一条”；
- “使用 200 条多尺度候选进行当前训练”；
- “Guidance 在闭环推理时在线生效”；
- “安全门已经保证官方闭环绝不碰撞”；
- “当前参考分支与策略分支由两个不同 checkpoint 生成”。

## 17. 关键代码与实验文件索引

### 17.1 采样与 guidance

- `hdp_nuplan/model/module/decoder.py`：多候选独立噪声、DPM 采样与轨迹解码；
- `hdp_nuplan/model/diffusion_utils/candidate_noise.py`：候选噪声尺度校验；
- `hdp_nuplan/model/diffusion_utils/sampling.py`：DPM-Solver++ 参数；
- `hdp_nuplan/model/guidance/longitudinal.py`：纵向 guidance energy；
- `hdp_nuplan/rl/train_epoch_rl.py`：四分支池、场景自适应 guidance、reference bound 与 update masks。

### 17.2 安全、压缩与 reward

- `hdp_nuplan/rl/candidate_safe_set.py`：hard-safe 资格和 88→32 压缩；
- `hdp_nuplan/rl/reward.py`：NuPlan 对齐 reward 与五维质量；
- `hdp_nuplan/rl/loss.py`：reference-relative advantage 和候选加权；
- `hdp_nuplan/rl/safety_update.py`：约束正目标更新配置检查；
- `train_predictor_rl.py`：B10 载入、reference 深拷贝、EMA rollout 与阶段调度。

### 17.3 配置、测试与结果

- `scripts/run_midterm_posttraining.py`：guided safe-set 协议固定参数；
- `tests/test_candidate_noise_profiles.py`；
- `tests/test_candidate_selection.py`；
- `tests/test_guided_safe_set_protocol.py`；
- `tests/test_candidate_generation_profiles.py`；
- `doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_full306801_posttraining/Test14全量闭环评测结果与分析.md`：训练后 Test14 官方闭环结果；
- `doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_summary/第四章阶段性结论与成果.md`：第四章阶段总结。

本次文档整理时，使用项目指定 Python 环境重新执行上述四个直接相关测试文件，结果为：

```text
35 passed, 1 warning
```

该结果验证了当前候选噪声、候选选择、guided safe-set protocol 和候选生成 profile 的单元测试；它不是闭环性能证明。

## 18. 最终结论

当前多扩散候选采样方法的核心不是“把候选数变多”，而是同时处理三个问题：

1. **支持范围**：用低噪声、中等噪声和定向 guidance 扩大候选行为覆盖；
2. **候选质量**：用碰撞/DA 硬资格与 source-aware compression 阻止低质量探索样本主导训练；
3. **学习方向**：用 frozen reference 的路线、进度、reward 和相对优势约束定义真正的更新目标。

它已经把原来“低噪声候选高度相似、有效组过少”的问题推进为“全量数据上具有足够目标组、可以完成策略更新”。但 Test14 碰撞退化说明，下一阶段的关键不再只是继续扩大候选几何差异，而是进一步缩小离线安全 proxy 与官方闭环安全反馈之间的错配。
