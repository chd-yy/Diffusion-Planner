# HDP RL 后训练流程与 Reward 设计全解

整理时间：09月16日 11:46（北京时间）。正文时间不写年份；文件路径中的原始时间戳保持不变，便于定位证据。

核验完成时间：09月16日 11:54（北京时间）。

评测状态更正：09月16日 12:26（北京时间），见第 11.2 节；奖励与训练配置说明未改变。

本文回答：当前 RL 后训练究竟怎样运行，各项奖励为什么设计、怎样计算、哪些真正启用，以及奖励怎样最终变成参数更新。

**范围声明：**以当前工作区实际执行代码为准，以 **09月15日几何修复版 RL Epoch2** 的落盘 `args.json` 为“当前运行配置”。另外单独区分历史 RL Epoch2、通用训练入口默认值和未启用的实验能力。不能用“代码里存在某个参数”推断“本次训练使用了该项奖励”。下文 motivation 是依据实现解释的工程目的，不代表经过消融实验验证的最优设计，也不宣称已经逐式复现论文。源码中“论文 D.5”等注释不是本文独立核验的论文证据。

## 1. 先给结论

当前方法是：**从 B Epoch10 出发，用冻结的 EMA 策略离线采样一组轨迹，根据缓存环境计算奖励，过滤不合格候选，再用正优势加权的扩散回归更新 decoder，同时保留专家监督锚点。**

它不是在 nuPlan simulator 中边交互边训练的 PPO，也没有 critic、价值网络、GAE、PPO ratio/clipping 或可学习 reward model。`rl_advantage_clip` 只是裁剪标准化奖励优势，不是 PPO 的策略概率比裁剪。

当前基础奖励为：

```text
R_base = 1 × risk + 3 × follow + 2.5 × lane + 5 × progress_guard
```

四项取值均在 `[0,1]`，因此基础奖励在 `[0,11.5]`；**经过安全门控后的 reward 不一定仍在此范围**。随后用于学习的资格是：

```text
eligible = (risk >= 0.3) AND (min_ttc_seconds >= 1.0)
candidate_mask = eligible AND no_collision
```

对于满足资格的候选，只拟合**奖励高于本场景合格候选均值**的轨迹；不合格候选的 rollout 回归权重为零。专家监督锚点不受该候选 mask 限制。

最需要区分的状态如下。

| 项目 | 当前是否真正启用 | 作用 |
|---|---|---|
| risk | 是，权重 1 | TTC、THW、静态占用距离、重叠风险的最差项 |
| follow | 是，权重 3 | 跟车时距、间距、速度匹配、纵向舒适性 |
| lane | 是，权重 2.5 | 车道居中；疑似专家换道时屏蔽该项辨别力 |
| progress guard | 是，权重 5 | 相对专家轨迹的前进/停车约束 |
| risk/TTC reward 门控 | 是，阈值 0.3 / 1 秒 | 不合格候选不能靠其他加分越过合格候选 |
| 安全候选过滤 | 是 | mask 真正传到 loss |
| 独立无碰撞过滤 | 是 | 与 safety mask 取交集 |
| progress guard 候选硬过滤 | 否 | `0.9` 阈值存在，但当前不执行 |
| 可行驶区域候选门控 | 否 | 代理值仍计算，仅诊断 |
| 专家监督 anchor | 是，loss 权重 0.1 | 防止单纯自蒸馏偏离专家行为；它不是 reward |
| B10 reference anchor | 否，权重 0 | 可选冻结参考模型输出约束 |
| B10 reference-relative advantage | 否 | 当前不是“必须超过 B10 才学习” |
| 六项旧式 reward 权重 | 不参与当前总奖励 | 参数保留不等于执行，见第 8 节 |
| 三种 aligned/proxy 目标 | 否 | 实现存在，但当前 `objective_mode=legacy` |
| rollout 轨迹后处理扰动 | 否 | std=0、epochs=0 |
| Detached Integral 截断反传 | 否 | window=0，实际普通累积积分 |

## 2. “当前”对应哪个模型和哪份配置

### 2.1 核心证据入口

| 内容 | 证据 |
|---|---|
| 当前实际参数 | [几何修复版 args.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/training_log/hdp-rl-safety-repair-controlled/2026-09-15-21:12:50/args.json) |
| 完整启动命令、输入 hash、运行状态 | [run.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/run.json) |
| 实际 rollout/update 指标 | [train.log](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/train.log) |
| 显式安全修复训练入口 | [train_rl_safety_repair.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/train_rl_safety_repair.py) |
| 通用训练入口、默认参数和 epoch 调度 | [train_predictor_rl.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/train_predictor_rl.py) |
| 奖励真实调用链 | [reward.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/reward.py) |
| 之前的排查、修复过程 | [RL Epoch2 安全退化修复记录](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan/RL_Epoch2安全退化修复记录.md) |

起点是 [B Epoch10 checkpoint](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth)。当前产物为 [几何修复版 RL Epoch2 checkpoint](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/training_log/hdp-rl-safety-repair-controlled/2026-09-15-21:12:50/model_epoch_2_trainloss_0.0009.pth)。

```text
B10 SHA256:
22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce

当前几何修复版 RL Epoch2 SHA256:
3a2348a12eb88f32018921514ffd4bce46720392370af9928aa0392ac4e56948
```

修改 reward 源码不会自动改变已经生成的历史 checkpoint。新训练产物也不能冒充历史 Test14 日志中那个 RL Epoch2。

### 2.2 历史版本、默认值与当前运行的区别

历史参数证据：[原 RL Epoch2 args.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json)。历史缺失的新参数不能解释成历史已启用。

| 配置/行为 | 通用 CLI 当前默认 | 历史 RL Epoch2 | 当前几何修复版 |
|---|---|---|---|
| weighting | exp_advantage | 指数优势并中心化 | positive_advantage |
| center_reward_weights | false | true | false |
| safety / collision 过滤 | false / false | 没有当前显式过滤链 | true / true |
| 真实 Pacifica 几何开关 | false | 旧尺寸、后轴当中心 | true |
| progress guard 权重 | 0 | 5 | 5 |
| risk 门阈值 | 0 | 0.3 | 0.3 |
| TTC 门阈值 | 1 秒 | 1 秒 | 1 秒 |
| expert anchor | 0 | 0.1 | 0.1 |
| replay 大小 | 4096 组 | 1024 组 | 1024 组 |
| 单 update epoch 抽样上限 | 0，即不限 | 500 | 500 |
| detach window | 10 | 0 | 0 |

还有一层容易遗漏：直接实例化 `NuPlanRewardConfig()` 的 TTC 门默认是 **0 秒**，而通用 CLI 默认是 **1 秒**。直接调用 `group_advantage_weights()` 的 `min_reward_std` 默认是 `0.01`，训练入口则传入 `1e-6`。看函数默认值、看 CLI 默认值、看实际 args 是三件不同的事。

## 3. 端到端流程与数据

```text
B10 的 EMA 权重
  ↓ 初始化当前策略 + EMA；冻结 encoder
10k mini-train NPZ 场景
  ↓ Epoch 1：EMA 固定、no_grad，每场景采样 32 条 8 秒轨迹
物理空间候选 + 原始场景张量
  ↓ 计算四项 reward → 安全门控 → safety ∩ no_collision mask
CPU FIFO replay：最多 1024 个场景组
  ↓ Epoch 2：有放回抽样，每批 2 组，最多抽样 500 批
合格组内优势 → 正权重 → 加权 diffusion + waypoint loss
  ＋ 0.1 × 专家监督 loss
  ↓ 有效目标检查 → backward → 梯度裁剪 → AdamW → EMA
保存 model / EMA / optimizer / scheduler
  ↓ 单独执行 nuPlan 官方闭环评测；不是训练 reward 的一部分
```

### 3.1 场景来源与张量口径

训练缓存来自原有 10k mini-train manifest：[训练清单](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json)。安全修复脚本沿用该清单，不把 Test14 故障场景当训练样本，并检查 6 个诊断 token 未混入。

当前主要维度：历史 21 帧、未来 80 帧、`dt=0.1 s`，即未来 8 秒；输入邻居最多 32 个，但 dataset 提供给未来监督/奖励的邻居数为 10；静态物体最多 5 个；普通 lanes 为 `70×20×12`，route lanes 为 `25×20×12`。超过这些截取范围的目标不自动进入 reward。

候选形状为 `[B,G,T,4]`，本次每批为 `[2,32,80,4]`，末维：

```text
[x, y, cos(yaw), sin(yaw)]
```

`x,y` 是以当前自车后轴为局部原点的**未来后轴位置**；邻车轨迹和静态物体位置是物体中心。奖励使用未经 observation normalization 的物理张量，网络输入使用归一化张量。专家未来和邻车未来来自记录数据，不是模型奖励打分器在线预测出来的交互响应。

邻车无效未来点由原始未来状态前三维全零识别并屏蔽；缺失对象不应当作位于原点的障碍物。缺失场景几何则有中性回退，详见各奖励边界条件。

### 3.2 初始化与可训练部分

`_load_pretrained()` 优先读取 checkpoint 中的 `ema_state_dict`，不是一律取 `model`。这里只热启动模型权重，重新创建 RL optimizer/scheduler，不恢复 B10 的 optimizer 动量、训练进度或 replay。

当前 encoder 和 decoder 深度均为 3，hidden dim=192、heads=6。encoder 参数冻结，仅优化可训练 decoder 参数。可选 reference policy 是从初始模型深拷贝出的冻结副本，并非随着 EMA 更新；本次两个 reference 开关均关闭，因此不创建该副本。

### 3.3 Rollout 具体怎样采样

入口是 `Hyper_Diffusion_Planner.sample()`，实际采样在 [decoder.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/model/module/decoder.py) 的 `sample()`。

1. 对同场景编码一次，再复制条件到 32 个候选。
2. 在**归一化 action 空间**采样 `xT = 0.1 × N(0,I)`；0.1 不是米，也不是每个 waypoint 独立加 0.1 米噪声。
3. 使用 [sampling.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/model/diffusion_utils/sampling.py) 的 DPM-Solver++，配置 steps=6、order=2、logSNR、multistep、denoise_to_zero=true。steps=6 不应简单等同于严格只有 6 次网络求值。
4. 对输出 action 反归一化；前两维是逐帧位移 `Δx,Δy`，通过 `cumsum` 恢复后轴位置。
5. scorer 在 `no_grad` 下给每条完整轨迹一个标量 reward 和若干诊断值。

EMA 在一轮 rollout 中不更新，因此这一批 replay 来自固定旧策略；没有对 DPM 采样链或 reward 求导。下一轮 rollout 才使用此前 update 后的 EMA。

存在独立的 [trajectory_augmentation.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/trajectory_augmentation.py)：每条轨迹抽一对纵横向偏移，沿整条轨迹共享，在各点朝向坐标中变换，保持 yaw 不变。本次 std=0、epochs=0，**没有启用**，不能和初始扩散噪声混淆。

### 3.4 Replay 与 Epoch2 的真实含义

[replay_buffer.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/replay_buffer.py) 用有容量上限的 `deque`，每条记录是一个场景组，保存 scene name、32 条物理轨迹、reward、bool mask，以及可选 reference reward；张量 detach 后存 CPU。场景条件在 update 时按 scene name 重新读 NPZ。

当前 `buffer_update_epoch=2`，代码 epoch 从 0 起算：

| 日志名称 | 实际代码 epoch | 行为 |
|---|---:|---|
| Epoch 1 | 0 | 清空 replay，遍历训练集进行 rollout，无梯度更新 |
| Epoch 2 | 1 | 从 replay 抽样更新模型，保存 checkpoint |

因此 `train_epochs=2` 是 **一次 rollout + 一次 update 阶段**。若设为 6，则默认是三个这样的周期，并非每一轮都同时采样和更新。

单卡、10k 场景、每场景 32 候选，总共生成约 **320,000 条**；但 replay 容量只有 **1024 个场景组，即 32,768 条候选**。更新使用遍历末尾保留下来的组，不是从全部 10k 组做 reservoir sampling。末尾顺序来自打乱的 sampler，因此不是原清单简单末尾，却仍有明显覆盖限制。

`sample()` 使用 `random.choices` **有放回**抽样，500 批×每批 2 组=1000 次组抽取，既不是恰好训练 1000 个不同场景，也不是 replay 的完整一遍。update 中 DataLoader 提供外层迭代长度/批大小，真正训练场景以 replay 抽样结果为准。

随机种子本次为 2026；DDP rank 使用 `seed+rank`。`DistributedSampler` 构造时未显式传 `args.seed`，使用其默认 seed，并通过 `set_epoch(epoch)` 改变顺序；不要把“训练 seed=2026”写成“所有随机组件均独立显式设为 2026”。当前 torchrun 只有一个进程。

## 4. 先理解安全奖励的几何基础

### 4.1 为什么必须将后轴转换到车身中心

当前 `reward_use_nuplan_vehicle_geometry=true`，从本地 nuPlan devkit 的 `get_pacifica_parameters()` 取得：

```text
ego length = 5.176 m
ego width  = 2.297 m
rear axle → center = 1.461 m
box_center = rear_axle_xy + 1.461 × normalized_heading
```

旧默认 `length=4.8,width=2.0,offset=0` 把后轴当车身中心，旧车头仅在后轴前 2.4 米，真实车头则是 `1.461+5.176/2=4.049` 米，漏掉约 **1.649 米前向覆盖**。这不是微小超参数差别，而会改变碰撞和安全排序。

邻车尺寸从过去状态 width/length 提取，静态物体也使用自身尺寸；缺失尺寸回退 `2.0×4.8 m`。自车中心速度还补入旋转引起的偏移速度：

```text
v_center(t) = v_rear(t) + offset × [heading(t)-heading(t-1)] / dt
```

否则转弯时 TTC 的位置与速度参考点仍不一致。速度比、THW 等使用的 ego speed 则仍由后轴轨迹差分得到，不能说所有速度量都改成车身中心速度。

### 4.2 OBB separation 与 no_collision

`_rectangle_signed_separation()` 使用二维有向矩形的四条分离轴：自车纵/横轴和目标纵/横轴。对每条轴，计算中心投影间距减去双方投影半径，再取四轴最大值 `s`。

```text
s > 0：存在分离轴，矩形分离
s = 0：边界接触
s < 0：矩形重叠
```

`s` 是 SAT signed separation，不是严格的二维欧氏最短距离。离散 0.1 秒点之间也没有连续 swept-volume 检测。

`_collision_cost()` 在有效动态/静态对象及时间上取最小 separation，得到：

```text
collision_cost = clip((0.5 - min_s) / 0.5, 0, 1)
no_collision   = (min_s >= 0)
```

当前 `collision_cost` 只作诊断；**no_collision 用于候选硬过滤**。碰撞过滤并不要求 0.5 米安全余量。risk 对 `s<=0` 视作重叠，和这里 `s>=0` 视作 no_collision 在恰好接触边界的口径略有区别；当前还叠加 risk/TTC 门。

静态物体在未来 80 步保持原位；邻车使用记录的未来轨迹。不区分官方责任碰撞，也没有完整的场景交互预测。

## 5. 当前四项 Reward：动机、公式、实现

以下令 `clip01(z)=min(max(z,0),1)`，速度单位 m/s、距离 m、时间 s。所有奖励都针对每个场景的每条候选计算。源码主要对应 `_risk_reward()`、`_following_reward()`、`_lane_reward()`、`_progress_guard_reward()`。

### 5.1 Risk：不要用大部分安全时刻掩盖单帧危险

**动机：**单一“撞/没撞”过于稀疏，碰撞前的接近、跟车过近、贴近静态障碍物也应提供信号；采用最差值，避免 79 个安全点平均掉 1 个危险点。

```text
risk = min(ttc_reward, thw_reward, occupancy_reward, safety_reward)
```

四项并不是再按独立权重相加，当前外部只对聚合后的 risk 乘权重 1。

#### A. TTC：接近时间

对有效邻居 j、时刻 t，令 `u` 为自车中心指向邻居中心的单位向量：

```text
closing_speed = -(v_neighbor - v_ego_center) · u
ttc_raw = max(s, 0) / max(closing_speed, 0.001)
speed_ratio = clip01(v_ego_rear / 15)
safe_ttc = 2 + 2 × speed_ratio
```

无重叠且 closing_speed≤0.001 时，物理 `ttc_seconds=∞`、TTC reward=1；重叠时物理 TTC=0。其余情况下：

```text
ttc_score = clip01(ttc_raw / safe_ttc)
```

对重叠另有 shaping：位于后方的目标得 `1-0.3=0.7`，其他位置重叠得 0。再在所有有效邻居和时间上取最小值。

后方判定仅是 `longitudinal<0`，不是 nuPlan 的责任归属判断。这一减弱处罚的工程解释是减少 nonreactive 记录回放中后车无法响应自车所带来的惩罚，但不能由此证明追尾一定无责。

**硬门用的 `min_ttc_seconds` 是上述动态邻居 TTC 的最小值，静态列表不参与这个最小 TTC。** 静态安全由 OCC、安全重叠项及 no_collision 共同处理。即使后方重叠获得软分 0.7，TTC=0 仍通不过当前 1 秒门。

这只是 `SAT 间隔/视线接近速度` 近似，并非官方 TTC metric 的完整预测与筛选实现。

#### B. THW：跟车时间间隔

只考虑同向前方车辆代理集合：有效、车辆类型、纵向位置>0，横向距离不超过 `ego_width/2+neighbor_width/2+0.5`。这不是基于车道拓扑严格确定同车道。

```text
bumper_gap = max(longitudinal - ego_length/2 - neighbor_length/2, 0)
THW = bumper_gap / max(v_ego, 0.1)
safe_THW = 1 + speed_ratio
thw_score = clip01((THW - 0.5) / (safe_THW - 0.5))
```

ego speed≤0.1 时直接给 1；没有前车也给 1；否则跨有效前车及时间取最小值。bumper_gap 用长度近似，并不是旋转矩形的精确前后保险杠距离。

**动机：**即便相对速度很小、TTC 很大，极短跟车间距仍危险，因此 THW 补足 TTC 的盲区。速度越高要求的安全时距越大。

#### C. OCC：静态障碍物安全距离

```text
safe_distance = min(0.5 + 0.2 × v_ego, 3.0)
occupancy_score = clip01(max(s_static,0) / safe_distance)
occupancy_reward = 对有效静态对象、时间取最小值
```

无有效静态物体时为 1。**动机：**静态障碍物没有跟车语义，也不能只等真正碰撞才扣分；速度高时增大净空需求。它不是可行驶区域奖励，也不是地图 occupancy grid。

#### D. Safety：重叠兜底

动态重叠处罚：后方 0.3、其他位置 1；跨对象/时间取最大处罚，`safety_reward=1-max_penalty`。任意有效静态重叠则置 0。

**动机：**防止低相对速度、静止或其他 soft shaping 特例把已经发生的重叠视为高安全性。尽管有该项，仍保留独立 no_collision 过滤，让“软分非零”不能自动成为“可模仿候选”。

#### Risk 的已知倾向

取 min 会突出极端危险点，但也容易被离散噪声或一个对象的几何误差主导；无对象/低速等情形给中性高分，容易对停车过于友好。因此 risk 不能单独作为完整驾驶目标，需要 progress guard 等约束。

### 5.2 Follow：安全之外，还应像正常车辆一样跟车

**动机：**risk 只要求别太危险，不能鼓励合适距离、跟随前车速度和避免突加减速。follow 提供更密集的行为质量信号。

使用与 THW 相近的前方车辆集合，每个时刻选 bumper gap 最小的前车，定义：

```text
ideal_gap_time = 1 + speed_ratio
actual_gap_time = bumper_gap / max(v_ego,0.1)
target_spacing = 2 + ideal_gap_time × v_ego
speed_tolerance = max(2, 0.25 × v_ego)
```

四个子分数为：

```text
gap_score = clip01(1 - |actual_gap_time - ideal_gap_time| / ideal_gap_time)
spacing_score = clip01(1 - |bumper_gap - target_spacing| / max(target_spacing,1))
speed_score = clip01(1 - |v_ego - v_leader| / speed_tolerance)

acc_violation = relu(a_long-2)/2 + relu(-3-a_long)/3
comfort_score = clip01(1-acc_violation)

per_step_follow = (gap_score+spacing_score+speed_score+comfort_score)/4
```

前车速度使用其速度在自车朝向上的投影并截为非负；纵向加速度由自车运动差分并投影得到。只对**有前车的时刻**求平均；整条轨迹没有前车则 follow=1，同时 leader_fraction=0。

区别要点：THW 只关注过近，follow 同时惩罚离理想时距/间距过远；follow 是时间平均而非风险最小值。gap 和 spacing 两项相关但不完全相同，后者包含 2 米静止间距。follow 中的舒适性正在生效，但只在有前车时起作用；它不是全场景 jerk comfort，也不是限速合规奖励。

### 5.3 Lane：鼓励居中，但不要强迫专家换道轨迹一直贴某条中心线

**动机：**仅有安全与跟车容易横向漂移；但是朴素车道居中奖励会惩罚合理换道，因此用专家轨迹判断是否屏蔽该项。

当前调用传入普通 `lanes`，不是只用 `route_lanes`。对候选每个后轴轨迹点找最近离散车道中心点：

```text
half_width = (|left_boundary_vector|+|right_boundary_vector|)/2
point_lane_score = clip01(1 - distance_to_nearest_center_point / half_width)
lane = 各时间点 point_lane_score 的均值
```

half_width≤0.5 时使用 1.75 米回退，并设最小宽度保护。这里距离是到**离散中心点的二维欧氏距离**，不是到连续车道中心线的精确垂距。

若专家未来满足以下任一条件，整组 lane reward 统一置 1、lane_reward_mask=0：

1. 任意专家点到最近中心点的距离超过相应 half_width；
2. 专家首尾相对最近车道的横向偏移变化，大于 `0.5×min(首尾 half_width)`。

没有有效车道时也回退为 1、mask=0。统一为 1 表示取消该项候选间的辨别力，而非对所有候选扣分。

这是专家驱动的**启发式换道/偏离屏蔽**，不是官方 lane-change 标签；可能漏判换道，也可能因稀疏地图屏蔽整组。lane 用后轴点，不能保证车身四角位于官方可行驶区域，也不能保证选对 route 分支。

### 5.4 Progress guard：防止安全奖励把模型推向不必要停车

**动机：**risk、跟车、居中都可能在低速/停车时较高；直接无限奖励前进距离又会激励乱冲。因此以专家的路线前进量作为有限目标，同时在专家停车场景要求停得相近。

先用 `_route_motion_metrics()` 得到候选归一化前进量 P：

```text
step_progress(t) = [p(t)-p(t-1)] · nearest_route_tangent(t)
P = sum_t step_progress(t) / 10
```

首步以局部原点为 p(0)；最近 route 点方向归一化后作投影。无有效 route tangent 时回退末端 x/10。P 不是 clip 到 `[0,1]` 的官方 route progress，可以为负或大于 1。对专家做同样计算得到 `P_expert`。

当前 stop tolerance `δ=0.2`，对应该归一化下约 2 米：

```text
若 P_expert > 0.2：
    progress_guard = clip01(P / P_expert)
否则：
    progress_guard = clip01(1 - |P-P_expert|/0.2)
```

没有专家轨迹则回退 1。

例如：专家前进 20 米，候选前进 16 米，则 guard=0.8、贡献 4 分；达到或超过专家进度则最多贡献 5 分。专家基本停车而候选多前进约 2 米，可将 guard 降到 0。

**边界：**这是相对专家的路线投影量，不是相对 B10 的闭环进度；接近专家不代表优于 B10，也不意味着在安全前提下进度最优。当前该项是 soft reward，guard≥0.9 的候选硬过滤尚未启用。

### 5.5 为什么权重是 1、3、2.5、5

实现上这组系数控制合格候选间的偏好：follow 提供行为引导，lane 控制横向，progress guard 用较大权重避免停滞。risk 权重虽然只有 1，但还具有独立资格门，不是单靠加法权重竞争。

这只能解释设计意图，不能证明这些数值最优。四项存在交叠，例如 TTC/THW 与 follow、risk 与无碰撞过滤；不同场景的可变项也不同。没有前车时 follow 恒为 1，关闭 lane 区分时 lane 恒为 1，相关项只增加共同常数，并不驱动该组候选之间的选择。

## 6. 安全门控和候选过滤：两个不同环节

### 6.1 Reward 门控改变排序

`_apply_safety_gate()` 当前判断 `risk>=0.3` 且 `min_ttc_seconds>=1`。可选 drivable 判断当前关闭。门控函数本身没有独立 `no_collision` 参数。

若一组至少有一个合格候选：

```text
合格候选：R_gate = R_base
不合格候选：
R_gate = min(合格候选的 R_base) - 1
         - relu(0.3-risk)
         - relu(1-min_ttc_seconds)/1
```

这样即使不合格轨迹的跟车、车道或前进分数很高，也排在所有合格候选之后。

若整组无合格候选，门控返回 `risk+0.001×R_base` 作风险优先排序。这是兼容性的 reward 输出，**不代表当前配置会模仿整组中“相对没那么坏”的轨迹**：下一步过滤会让它们全部权重为零。

### 6.2 Mask 决定谁能成为 rollout 回归目标

[safety_update.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/safety_update.py) 的 `build_candidate_mask()` 从 safety gate 资格出发，按开启项继续求交集。本次：

```text
M = safety_gate_eligible AND (no_collision >= 1)
```

若开启 progress filter，还会交上 `progress_guard>=0.9`。注意实现语义：任何一个候选过滤开关开启，都会启用 mask 传递，而 mask 的基底始终是 safety gate 资格；不是三个完全独立任选的 loss mask。

当前修复保证：mask 从 rollout → replay → update → loss 全程保留；显式过滤却缺失 mask 时直接报错；mask 必须为 bool；过滤不能与 centered 负回归权重并存；非有限轨迹/reward/reference/资格会拒绝，而不是静默当安全。

**动机：**“惩罚较少”与“值得模仿”不同。只有门控排序但仍给低分候选正拟合权重，会继续学习不安全轨迹；负权重 MSE 则可能通过无限增大重建误差降低 loss，不能当成受限策略优化。

### 6.3 已实现但关闭的 drivable area 门

`_drivable_area_compliance()` 是车道走廊代理：降采样 lane 中心点，每个轨迹点选附近最多 8 点，用车身在车道横向上的投影半宽及 0.1 米 margin，估计左右净空，取附近候选走廊的最大净空，要求全时域净空非负。

这比后轴点居中更接近 footprint，但没有使用官方完整道路多边形并集。缺少 lanes/有效边界时回退合规；若开启该门而专家轨迹也不通过，代码会对该场景绕过此门，避免缓存走廊不完整把整组都拒绝。因此它不是缺地图即拒绝的 fail-closed 地图安全检查。

当前关闭的原因是：弯道、交叉口、换道等场景存在走廊代理误拒风险，尚不能把它等价为官方 drivable-area 约束。相关值仍会写诊断。尤其 `drivable_area_gate_active` 在开关关闭时初始化为 1，**不能看到该指标为 1 就认定门已经开启**；必须检查 args。

## 7. Reward 怎样变成训练损失

实现入口：[loss.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/loss.py)，调度与专家项：[train_epoch_rl.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py)。

### 7.1 只在合格候选中计算组统计

设组内合格数 n，μ 为合格 reward 均值，σ 为总体标准差（方差除 n，不是 n−1）。

```text
active_group = (n >= 2) AND (σ >= 1e-6)
A_i = clip((R_i-μ)/(σ+1e-6), -5, 5)
w_raw_i = max(A_i,0)                 # 当前 positive_advantage
w_i = w_raw_i / max(sum(w_raw)/n, 1e-12)
```

不合格候选及 inactive group 权重统一为 0。只有一条合格轨迹也会跳过 rollout 回归；所有奖励相同时同样没有组内改进信号。

归一化的分母是**所有合格候选的数量 n**，并非仅正优势候选数；随后 loss 对完整 `B×G` 求均值，不是按正权重数量重新平均。因此合格比例少的组，总贡献也随之减小。

例：三个合格候选奖励为 4、5、6，其余不合格。均值为 5，只有 6 的候选得到正权重；归一化后该权重为 3，而不是“所有合格轨迹都以权重 1 拟合”。

`rl_reward_temperature=1` 虽写入本次参数，但 **positive_advantage 分支并不使用 temperature**。不能把它描述成当前生效的探索温度。

### 7.2 其余权重模式

| 模式 | 未归一化/初始权重逻辑 | 当前状态 |
|---|---|---|
| exp_advantage | `exp(temperature×A)`，低于均值仍有正拟合权重 | 未使用；CLI 默认 |
| softmax_positive | `softmax(A/temperature)×n`，合格候选均正权重 | 未使用 |
| positive_advantage | `relu(A)`，只拟合高于 baseline 的候选 | 使用 |

temperature 在前两种模式中一个作乘法、一个作除法，不能跨模式照搬同一种“温度越大越尖锐”的解释。

旧 `center_reward_weights=true` 在 exp 模式把 w 改成 w−1，产生负回归权重。当前显式安全过滤禁止该组合。本次 center=false。

### 7.3 候选轨迹变成 diffusion target

轨迹 waypoint 先转成逐帧位移：

```text
u_t = [x_t-x_(t-1), y_t-y_(t-1), cos(yaw_t), sin(yaw_t)]
x_0=y_0=0
```

这里 `Δx,Δy` **没有除以 dt，不是物理速度**。经过 state normalizer 得到 diffusion 的干净目标 z0；随机采样扩散时间 `τ∈[0.001,1)` 与标准高斯 ε，用 SDE marginal 得到 `zτ=α(τ)z0+σ(τ)ε`。

每组条件复制 32 次，当前网络预测 `x_start`，监督类型也为 `x_start`。每条候选的 diffusion error 为 4 个状态分量平方误差之和，再对 80 步平均：

```text
L_diff_i = mean_t sum_d (z0_pred[t,d]-z0_target[t,d])²
L_diff = mean_(B,G) [w_i × L_diff_i]
```

这里名字 `x_start` 指扩散过程的干净 action 表示，不是“仅预测轨迹第一个点”。其他 score/noise/v 监督分支存在，但本次不使用。

### 7.4 Waypoint hybrid loss 与 Detached Integral

预测 action 反归一化后，对其前两维积分回位置，与采样候选的 XY 做 MSE（时间和 XY 维一起平均）：

```text
L_waypoint = mean_(B,G) [w_i × mean_(t,xy)(p_pred-p_candidate)²]
L_rollout = L_diff + 0.01 × L_waypoint
```

当前 `rl_detach_window_size=0`，调用 [detached_integral()](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/utils/traj_kinematics.py) 时走普通 cumsum；后面位置误差可以向此前所有位移反传。若 window>0，则前向积分结果不变，反向仅保留最近窗口的累积梯度；**函数存在不代表当前开启截断反传**。

### 7.5 专家 anchor：reward 之外的监督约束

当前通过原监督训练 `diffusion_loss_func()` 对 replay 对应场景的专家 ego future 计算：

```text
L_expert = ego_planning_loss + 0.01 × ego_planning_hybrid_loss
L_total = 1 × L_rollout + 0.1 × L_expert + 0 × L_reference
```

虽然函数还接收邻车 GT，当前 RL 组合只取上述 ego 项，不把邻车监督 loss 加入总损失。

**动机：**reward 代理不完备、候选多样性不足、自蒸馏分布漂移时，专家监督保留基本驾驶能力。它约束专家行为，而不是直接约束 B10 的输出；专家也不是所有闭环状态下的最优轨迹。

`reward_imitation_weight=0` 和 `rl_expert_anchor_weight=0.1` 不矛盾：前者是旧奖励接口参数，后者是正在执行的监督损失权重。

### 7.6 Reference 两种能力均已实现，但本次关闭

1. **Reference anchor**：冻结初始化模型，对与当前模型完全相同的带噪 action、扩散时间和场景条件输出预测，在相同 supervision 空间做全张量 MSE。它不是 KL，也不是直接约束最终闭环轨迹。该 MSE 不使用候选 reward 权重过滤。
2. **Reference-relative advantage**：每场景用冻结参考模型额外采样 1 条轨迹并评分，将优势分子从 `R_i-μ` 改为 `R_i-R_reference`。标准差仍取合格候选组 σ；仍需 n≥2 且 σ 达阈值。搭配 positive_advantage 才使“超过参考奖励”的候选得到正权重。

本次 `rl_reference_anchor_weight=0`、`rl_relative_to_reference=false`；`rl_reference_noise_scale=0` 仅为闲置参数。即使启用相对参考 reward，也不是“B10 闭环 progress 不下降”硬保证，因为比较的是离线综合代理 reward；而参考轨迹单独以 G=1 评分，组内安全门的排序上下文也不同，使用前应核查其可比性。

## 8. 哪些 reward/cost 只是保留、诊断或备选

### 8.1 最大命名陷阱：legacy 模式并不调用 `_legacy_reward()`

`NuPlanTensorRewardScorer.__call__()` 在 `objective_mode="legacy"` 时使用的是第 1 节的四项相加。源码中另一个函数 `_legacy_reward()` 才是下面的旧式六项公式：

```text
R_old = w_progress×progress
        - w_collision×collision_cost
        - w_route×route_cost
        - w_comfort×comfort_cost
        - w_backward×backward_cost
        - w_imitation×imitation_cost
```

当前 scorer 主调用链没有调用它，四种可选 objective 也没有通过这六个 weight 来组合奖励。因此下面这些配置值**不是本次生效的六个 reward 权重**。

| 参数/量 | 当前记录值 | 当前真实用途 |
|---|---:|---|
| reward_progress_weight | 1 | 不参与当前总分；progress 本身用于 guard 和诊断 |
| reward_collision_weight | 10 | 不参与当前总分；no_collision 过滤另行生效 |
| reward_route_weight | 1 | 不参与当前总分；route_cost 诊断 |
| reward_comfort_weight | 0.01 | 不参与当前总分；comfort_cost 诊断；不能与 hybrid=0.01 混淆 |
| reward_backward_weight | 1 | 不参与当前总分；backward_cost 诊断 |
| reward_imitation_weight | 0 | 旧 imitation reward 未执行，当前主调用链不返回 imitation_cost |

### 8.2 各个诊断 cost 怎样算

**route_cost**：后轴轨迹各点到有效 route 中心采样点的最近距离，逐点上限 5 米，再时间平均并除以 5，范围 `[0,1]`。无 route 点回退 0。它不同于当前 lane reward：前者针对 route、固定 5 米归一化，后者针对普通 lanes、按车道半宽归一化并有专家屏蔽。

**comfort_cost**：用当前 ego velocity/acceleration 补齐差分边界，计算二维速度、加速度、jerk：

```text
acc_violation = relu(||a||-4)/4
jerk_violation = relu(||jerk||-8)/8
每项超限值最多截为 5
comfort_cost = mean(acc_violation)+mean(jerk_violation)
```

当前 legacy 总奖励不使用该 cost；follow 内的加速度舒适分仍使用。此 cost 也不是官方舒适性 metric 的完整横纵向、yaw 等规则。

**backward_cost**：存在 route tangent 时，对逐帧 route 投影位移的负部分取均值；无有效 route 时使用相邻 x 位移负部分均值。它不是当前的独立倒车处罚项，不过倒退可能通过 guard、lane、risk 间接影响分数。

**collision_cost**：见第 4 节，0.5 米 SAT 距离 margin 代理；当前直接过滤用的是另一个 bool `no_collision`。

**imitation_cost**：仅旧函数内以候选和专家 XY 平均距离衡量模仿偏差；当前不执行。专家 anchor 在 diffusion loss 层执行，不能拿该 cost 代替其公式。

源码还保留 `_collision_cost_center_distance_legacy()`、`_comfort_cost_finite_difference_legacy()`，不应把这些旧实现当成当前 scorer 的几何/舒适性逻辑。

### 8.3 三个未启用的复合目标

它们都使用加权几何平均，而不是当前四项加法：

```text
R = exp(sum_k normalized_exponent_k × log(max(Q_k,1e-4)))
exponents = safety:0.45, progress:0.25, route:0.15, comfort:0.10, follow:0.05
```

指数会归一化；log 前的 floor 导致某质量为 0 时总分不必严格为 0。随后仍经过同一个安全门。

| 模式 | safety quality | progress quality | route quality | comfort / follow |
|---|---|---|---|---|
| nuplan_aligned | risk×no_collision | progress_guard | **lane reward** | exp(-comfort_cost) / follow |
| nuplan_score_proxy_v2 | risk×no_collision | progress_guard | clip01(1-route_cost) | 同上 |
| nuplan_score_proxy_v3 | risk×(1-collision_cost)×no_collision | progress_guard | clip01(1-route_cost) | 同上 |

v2 的 `collision_cost` 和 `backward_cost` 虽被传入并记录，但没有进入其最终公式；v3 加入 collision_cost，backward_cost 仍只是记录。不能按函数注释中“纳入某指标”的表述推断它真的改变 reward。

**设计动机：**用几何平均减少“某一低质量项被其他高分完全补偿”，并尝试把路线、舒适等因素显式纳入。但这些仍是离线代理，不是官方 nuPlan aggregate score；当前主修复模型没有使用它们，已有实验脚本也不代表已选为最终方案。

## 9. 优化器、EMA、保存与异常行为

当前参数：AdamW，学习率 `4e-7`，梯度范数裁剪 5，冻结 encoder。没有显式覆盖 AdamW 的 betas、eps、weight_decay；使用安装版本默认值。`rl_deterministic_update=true` 将模型设为 eval 模式以关闭训练态随机层，**不关闭 autograd**；扩散时间、训练噪声、replay 抽样仍随机，不等于完全确定性训练。

EMA 更新：

```text
EMA ← 0.95×EMA + 0.05×current_model
```

只在实际 optimizer step 后更新，不是在 rollout 每条候选后更新。

当前 [lr_schedule.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/utils/lr_schedule.py) 虽函数名叫 `CosineAnnealingWarmUpRestarts`，**实际是线性 warm-up 后恒定学习率，不是 cosine restart**。本次 warm_up_epoch=1 走恒定分支，保持 `4e-7`。不要把仓库另一套 `diffusion_planner` 的同名文件或历史实现混入当前 HDP 结论。scheduler 只在存在实际更新的 update epoch 后 step，rollout 不推进。

更新保护：

- 无 rollout 回归目标且 expert/reference anchor 都为 0：跳过 backward、optimizer、EMA，防止 AdamW 动量/衰减仍移动参数。
- 当前 expert anchor=0.1，所以即便某批全无安全候选，仍可只按专家项更新；不应误报为“无候选却继续模仿危险轨迹”。
- 非有限 loss/gradient 会报错；DDP 统一是否更新的决定，任一 rank 有目标时所有 rank 参与 backward。
- `rl_max_update_steps_per_epoch=500` 限制的是抽样 batch 数；另记实际 update_steps 与 skipped_update_steps。不能在任意配置下把“上限 500”直接写成“必定 500 次 optimizer step”。

保存发生在 update epoch，checkpoint 包含 `model`、`ema_state_dict`、optimizer、schedule、epoch、loss 等，并写 latest。文件名的 trainloss 是训练损失，不是 nuPlan score，也不能据此把不同 reward/mask 配置的 checkpoint 直接排序为好坏。

### 9.1 后训练完成后，闭环推理怎样使用模型

[planner.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/planner/planner.py) 默认 `enable_ema=true`，当前完整评测未覆盖此选项，因此使用 RL checkpoint 的 EMA 权重。每个重规划时刻重新构建观测、归一化，模型输出未来轨迹，再将局部后轴预测转换为 nuPlan 状态轨迹。

这里没有专家 future 可供在线比较，也没有运行上述离线 reward scorer 来做 32 选 1。decoder 普通 forward 在没有带噪训练输入时调用 `sample(num_samples=1)`，使用其默认 **10 个 diffusion steps、noise_scale=0.1**；RL rollout 才显式指定 **32 候选、6 steps**。`rl_rollout_steps=6` 写在 args 中，不代表闭环推理也自动变为 6 steps。

因此，当前的 safety mask 是**训练目标选择约束，不是部署时的安全盾或碰撞拦截器**。实际 Test14 使用 `closed_loop_nonreactive_agents`：自车执行、状态随闭环演化，其他车辆沿记录行为回放；官方指标在仿真轨迹上另行计算。训练 reward、训练 loss 和官方评测 score 必须分别报告。

## 10. 当前配置速查表

下表与第 5、8 节一起覆盖当前 reward/RL 主要参数。`reward_` 前缀在 reward 阈值行省略，完整字段可在实际 args 中查询。

| 类别 | 当前值 | 解释 |
|---|---|---|
| 训练 | epochs=2，batch=2，seed=2026 | 一轮 rollout、一轮 update；单卡 |
| 优化 | lr=4e-7，warmup=1，grad_clip=5 | 当前无学习率缩小的 warm-up 区间 |
| DataLoader | workers=2，pin_mem=true | manifest=10k |
| Rollout | group=32，steps=6，noise_scale=0.1 | 候选多样性主要由初始扩散噪声产生 |
| 数据增强 | std=0，epochs=0 | 关闭 |
| Replay | buffer_size=1024，update_epoch=2 | 以场景组计数，FIFO |
| Update | max_batches=500，freeze_encoder=true，deterministic_update=true | 仅 eval 模式确定性，不代表无随机采样 |
| EMA | update_rate=0.05 | 等价 decay=0.95 |
| 权重 | mode=positive_advantage，normalize=true，center=false | 正优势拟合 |
| 优势 | clip=5，min_std=1e-6，temperature=1 | temperature 对当前分支无作用 |
| Loss | rollout=1，expert=0.1，reference=0 | hybrid=0.01，detach_window=0 |
| Reference | relative=false，noise_scale=0 | 均未使用 |
| 候选过滤 | safety=true，collision=true，progress=false | progress_min=0.9 闲置 |
| Reward objective | legacy | 四项加法，非旧六项函数 |
| Reward 权重 | risk=1，follow=3，lane=2.5，guard=5 | 当前基础总分 |
| Safety gate | risk=0.3，TTC=1，margin=1 | 排序与 mask 结合 |
| Drivable gate | require=false，margin=0.1 m | 只算代理诊断 |
| Geometry | Pacifica=true | L=5.176，W=2.297，offset=1.461 m |
| Collision | collision_distance=0.5 m | cost margin，不是过滤净空 |
| 风险速度参照 | risk_speed_reference=15 m/s | 速度比裁剪至 0～1 |
| TTC soft | low=2 s，high=4 s | 随速度比线性变化 |
| THW | critical=0.5 s，safe_low=1 s，safe_high=2 s | 仅前方车辆集合 |
| OCC | safe_min=0.5 m，safe_max=3 m，headway=0.2 s | 静态目标 |
| 后方重叠 | rear_end_collision_penalty=0.3 | 不是官方责任判定 |
| Follow | time_gap_low=1 s，high=2 s，min_spacing=2 m | 时距与间距子分数 |
| Follow speed | tolerance=2 m/s | 实际取 max(2,0.25×speed) |
| Follow comfort | acceleration=2，deceleration=3 m/s² | 有前车时的纵向舒适 |
| Leader corridor | lateral_margin=0.5 m | 几何近似，非拓扑同车道 |
| Lane | half_width_fallback=1.75 m，change_ratio=0.5 | 专家屏蔽启发式 |
| Progress guard | stop_tolerance=0.2 | P 归一化尺度下约 2 m |
| Config 内部常量 | dt=0.1，progress_normalization=10 | 非本次 CLI 独立参数 |
| Config 诊断常量 | max_route_distance=5，acc_limit=4，jerk_limit=8，comfort_clip=5 | 主要用于诊断 cost / 备选目标 |
| 对象尺寸回退 | width=2，length=4.8 m | 不替代本次真实 ego 尺寸 |

## 11. 本次实际训练信号和评测状态

### 11.1 训练已完成，但 loss 不能替代驾驶质量

几何修复训练于 **09月15日 21:12** 启动、**21:23** 完成。落盘日志的 Epoch2：

| 指标 | 实际值 | 正确解释 |
|---|---:|---|
| sampled_batches / update_steps / skipped | 500 / 500 / 0 | 本次确实更新 500 步；expert anchor 始终存在 |
| buffer_size | 1024 | 场景组数 |
| eligible_candidate_fraction | 0.55575 | 被抽到的候选中约 55.6% 符合 mask |
| no_eligible_group_fraction | 0.443 | 约 44.3% 被抽样场景组没有合格候选 |
| active_group_fraction | 0.543 | 满足数量和 reward 方差要求的组比例 |
| has_regression_targets | 0.786 | 按 batch 平均，约 78.6% 的批次有非零 rollout 权重 |
| L_rollout | 0.00060733 | 加权 diffusion + 0.01×waypoint |
| L_expert（未乘 0.1） | 0.00340872 | 独立监督锚点 |
| 总 loss | 0.00094820 | 约 0.00060733+0.1×0.00340872 |

这些是训练采样批次的汇总，不是全部 Test14 场景安全通过率。日志里的 reward_mean 包含被过滤候选，不等于“真正进入 loss 的样本平均质量”；reference_reward_mean=0 在本次表示功能未启用，不表示 B10 奖励恰好为 0。p10/p50/p90 也是每批统计后聚合，不应擅自解释成整个训练集的全局分位数。

### 11.2 闭环验证必须独立报告

此前针对 6 个已知故障场景、seed=0 的重放，几何修复版平均 score 约 0.6940，B10 约 0.5919，历史 RL 约 0.1146；但修复版 progress 约 0.7564，低于 B10 约 0.7686，因此不能只报安全改善就宣称所有 gate 通过。

这 6 场是依据历史故障挑选的诊断集，已经参与问题定位，**不是独立盲测，也不是完整 Test14 的收益估计**。

**09月16日 12:26 状态更正：**11:43 初次整理仅依据日志中的 Starting，误判评测仍在运行；12:24 同时核查进程与日志后，确认旧进程已退出、日志停在 11:27 左右，退出原因未明。已独立验证 hard 完成 2 个 40 场 chunk、random 完成 3 个 40 场 chunk，runner 均成功且 aggregator 覆盖完整；12:26 保留这些结果，从 hard chunk 2、random chunk 3 恢复，确认新进程存活。恢复前后 10 个归档 parquet 的 SHA256 不变，尚无最终 full_comparison。

完整目标为 hard 272 场、random 261 场，两组交叠 11 场，共 533 条分组记录、522 个唯一场景。这里应称全分布复核，不称完全未见过的盲测。

运行入口：[evaluate_rl_safety_geometry_test14_seed0.sh](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/evaluate_rl_safety_geometry_test14_seed0.sh)，进度证据：[evaluate.log](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_test14_seed0/evaluate.log)。本节是带时间戳的快照，不承诺之后状态不变。首次文档整理未修改或重启评测；12:26 的恢复是后续自动巡检发现中断后按既定授权执行，过程另记于安全退化修复记录。

## 12. 当前仍未解决什么

1. **离线代理与闭环安全之间仍有差距。** 缓存起始状态的 8 秒候选评分，不会覆盖执行后状态分布变化；最多 10 个未来邻居、5 个静态物体也限制障碍覆盖。修正 ego 几何非常必要，但不是完整环境安全建模。
2. **当前缺少可靠的地图可行驶区域约束。** lane reward 不等于 footprint 合规，drivable proxy 关闭；不能声称已经全面解决越界。
3. **相对组均值改善不等于优于 B10。** 当前 reference 两项均关闭；即使启用离线 reference reward，也不能直接保证官方闭环 progress/safety 不下降。B10-reference progress 硬约束尚未在本次配置中实现/启用。
4. **训练覆盖与计算使用效率有限。** 为 10k 场景采样，却仅保留最后 1024 组；约 44.3% 抽到的组又没有合格候选。修改 buffer 或采样策略需要受控验证，不能靠扩容就直接宣称收益。
5. **目前不是完整交通规则 reward。** 没有当前生效的显式速度限制、红灯、路权、完整官方责任判定等奖励。网络输入包含地图/交通相关特征，也不等于 reward 对这些规则有显式约束；同样不能据此反推模型完全没有从模仿学习学到相关行为。
6. **评价要看安全与行驶效率共同变化。** reward/loss 变好可能只是代理被优化；必须固定模型、args、场景集合、推理 seed、worker/chunk 协议，报告完整闭环指标和失败场景，再考虑多 seed 复核。

当前合理的后续次序是：完成现有完整评测并核实尾部失败 → 判断安全/进度退化是否稳定 → 再选择参考约束、地图代理或覆盖扩展等单变量修正。上述是后续决策依据，**不是本文已经实施的新训练改动**。

## 13. 如何复现与检查

### 13.1 复现当前显式修复训练配置

下面只给命令，不在整理文档时再次启动训练。`--output` 必须指向一个全新的目录，脚本拒绝覆盖已有目录；执行前确认 GPU 资源，避免干扰正在运行的闭环评测。

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/train_rl_safety_repair.py \
  --output /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_reproduce_new_run
```

脚本继承历史 args，并显式覆盖正优势、非中心化、安全/碰撞过滤和真实几何；校验 B10 SHA256、10k 清单数量、诊断 token 隔离，保存完整实际命令与输入 hash。`--legacy-geometry` 仅供保留旧几何的受控消融，不是推荐修复配置。

输入路径、依赖版本、GPU 算法及当前源码也会影响复现，固定 seed 不保证跨版本逐位一致。通用 RL 入口没有在此流程中恢复完整 replay/RNG/optimizer 的续训路径；仅把另一个 checkpoint 传入 pretrained 参数不是严格断点续训。共享 reward.py 的后续变化也需要留代码快照或 diff，不能只保留 args。

### 13.2 对照源码时的阅读顺序

1. [train_rl_safety_repair.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/train_rl_safety_repair.py)：本次实际覆盖了什么。
2. [train_predictor_rl.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/train_predictor_rl.py)：参数、模型、scorer 配置和交替 epoch。
3. [train_epoch_rl.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py)：raw/normalized 数据、rollout、replay、专家监督。
4. [reward.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/reward.py)：先读 `__call__()`，再沿它实际调用的函数阅读，不从旧函数名猜总公式。
5. [safety_update.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/safety_update.py) 与 [loss.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/loss.py)：资格怎样真正约束梯度更新。
6. [replay_buffer.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/replay_buffer.py)、[decoder.py](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/model/module/decoder.py)：保存单位与扩散采样表示。

### 13.3 文档核验范围

本文依据当前函数体、实际 args、run.json 和训练日志逐项核对，刻意区分“代码实现”“本次开启”“设计解释”和“闭环已证实”。本次只新增说明文档，不变更训练逻辑、模型权重或 reward 参数。

09月16日 11:54 核验记录：以下四个测试文件合计 **54 passed**，仅有一条已有 timm 导入弃用警告。本次测试显式隐藏 CUDA，在 CPU 上执行，不另启 GPU 训练。

```bash
PYTHONPATH=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan:/home/yanjun/NewDisk/Diffusion-Planner:/home/yanjun/NewDisk/nuplan-devkit \
CUDA_VISIBLE_DEVICES='' \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests/test_rl_components.py \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests/test_rl_safety_update.py \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests/test_rl_vehicle_geometry.py \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests/test_lr_schedule.py
```

测试覆盖组件和回归契约，不等于对所有 reward 设计动机、论文一致性或闭环性能的证明。另已检查文中本地文件链接存在、代码围栏闭合；截至本文核验仍不能用未完成的完整评测作最终收益结论。

**一句话总结：当前 RL 不是直接最大化官方 nuPlan 得分，而是在有限缓存环境中，用风险/跟车/车道/专家进度代理筛选 EMA 生成的轨迹，再通过安全过滤的正优势扩散回归和专家监督微调 B10。**
