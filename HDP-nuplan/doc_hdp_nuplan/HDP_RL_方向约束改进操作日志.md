# HDP-RL 方向约束改进操作日志

日期：2026-08-11  
状态：第一阶段代码改造和本地验证完成，尚未启动新一轮 RL 训练。

## 1. 改造背景

target-matched 11,040 监督模型经过论文参数 RL 微调后，固定 20 场景闭环结果为：

| 指标 | supervised | RL | 变化 |
|---|---:|---:|---:|
| overall score | 0.703840943 | 0.704483158 | +0.000642215 |
| expert-route progress | 0.653433099 | 0.676960608 | +0.023527509 |
| TTC within bound | 0.800000 | 0.850000 | +0.050000 |
| driving-direction compliance | 1.000000 | 0.975000 | -0.025000 |

逐场景中，`following_lane_without_lead` 场景 `1fb0bb88f9d35d59` 的 progress 提高，
但 driving direction 从 1 降到 0.5，使总分下降约 0.41043。说明现有 RL 已能改善进度和 TTC，
但代理 reward 缺少与 NuPlan driving-direction 指标对应的约束。

## 2. 原代码问题定位

论文模式 reward 原来是：

```text
risk_weight × risk_reward
+ follow_weight × follow_reward
+ lane_weight × lane_reward
+ progress_guard_weight × progress_guard_reward
```

`backward_cost` 虽然已经计算，但只写入 `details` 用于诊断，没有进入论文 reward。
`lane_reward` 只衡量轨迹点到车道中心线的距离，不能区分以下两种轨迹：

1. 靠近中心线并沿路线正向运动；
2. 靠近中心线但运动方向或车头方向与路线相反。

因此，继续增大 progress reward 或学习率可能同时放大方向退化，不能从根因解决问题。

## 3. 方法依据与本阶段选择

安全强化学习通常将安全要求表示为 cost/constraint，而不是只依赖任务 reward 的人工加权。
本阶段先实现可独立诊断、可配置开启的 `direction_cost`，作为完整约束优化前的最小验证版本。

设计原则：

1. 默认 `direction_guard_weight=0`，保证历史命令和旧实验数值不变；
2. 显式设置正权重后，从 multi-reward 中扣除方向代价；
3. 同时检查轨迹位移方向和车头朝向，避免只检查其中一个产生漏洞；
4. 停车场景的位移小于阈值时不统计运动方向，避免数值噪声造成误判；
5. 无有效 route 时回退到 ego 局部坐标系正 x 方向，保证输出有限。

本阶段是固定权重 guard，不宣称等同于 CPO 或 PID-Lagrangian。只有固定 guard 的有效性通过
闭环实验后，才考虑加入自动更新拉格朗日乘子的第二阶段。

## 4. direction cost 定义

对候选轨迹相邻点计算位移：

```text
delta_p[t] = p[t] - p[t-1]
```

根据候选轨迹点找到最近 route 点，并读取该 route 点的单位切向 `u[t]`。

运动方向余弦：

```text
motion_cosine[t] = normalize(delta_p[t]) dot u[t]
```

航向方向余弦：

```text
heading_cosine[t] = [cos(yaw[t]), sin(yaw[t])] dot u[t]
```

余弦低于 margin 的部分产生代价，并归一化到 `[0,1]`：

```text
component_cost = relu(margin - cosine) / (margin + 1)
```

最终：

```text
direction_cost =
    (motion_weight × motion_cost + heading_weight × heading_cost)
    / (motion_weight + heading_weight)
```

本轮默认 margin 为 0，即只惩罚夹角超过 90 度的明确反向运动或朝向。

新的可选 reward 为：

```text
reward_new = reward_old - direction_guard_weight × direction_cost
```

## 5. 代码改动

### 5.1 `hdp_nuplan/rl/reward.py`

新增配置：

```text
direction_guard_weight
direction_motion_cosine_margin
direction_heading_cosine_margin
direction_motion_weight
direction_heading_weight
direction_min_displacement
```

新增 `_direction_metrics()`，输出形状均为 `[B,G]`：

```text
direction_cost
motion_alignment
heading_alignment
reverse_fraction
```

四项全部写入 `details`，因此 rollout 日志会自动记录对应均值。

### 5.2 `train_predictor_rl.py`

新增六个对应命令行参数、参数范围检查，并传入 `NuPlanRewardConfig`。所有参数会随训练写入
`args.json`，保证实验可复现。

### 5.3 `scripts/compare_checkpoint_behavior.py`

恢复新方向配置，并在监督/RL checkpoint 行为对比中加入：

- 越低越好：`direction_cost`、`reverse_fraction`；
- 越高越好：`motion_alignment`、`heading_alignment`。

### 5.4 `scripts/validate_reward_v2.py`

真实 NPZ sanity check 新增方向指标有限性、范围和反向轨迹排序检查；新增 progress/direction
guard 权重参数，使检查配置与实际训练配置一致。

### 5.5 `tests/test_rl_components.py`

新增两个测试：

1. 正向、反向和停车轨迹的方向代价是否符合预期；
2. guard 权重为 0 时 reward 是否保持不变，启用后是否精确扣除方向代价。

## 6. 执行记录

### 6.1 定向单元测试

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q \
  HDP-nuplan/tests/test_rl_components.py \
  -k 'direction or paper_rewards_are_bounded'
```

结果：

```text
3 passed, 16 deselected
```

### 6.2 第一次真实 100 场景 sanity check

第一次沿用 reward 默认值，即 progress guard 为 0。方向相关检查全部通过，但历史检查
`moving_expert_beats_stop` 只有 `29/87` 通过，因此整体 strict check 失败。

这个失败不是新增方向指标造成的，而是检查配置没有复现当前 RL 实验使用的
`progress_guard_weight=5`。因此没有删除或放宽失败项，而是让脚本显式接收 guard 权重后重跑。

### 6.3 按当前 RL 配置重新检查

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/validate_reward_v2.py \
  --cache-dir HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache \
  --manifest HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/diffusion_planner_training.json \
  --output HDP-nuplan/tmp/targetmatch_11040_paper_omega0p1_eval/direction_guard_reward_sanity_100.json \
  --max-scenes 100 \
  --minimum-pass-rate 0.8 \
  --progress-guard-weight 5 \
  --direction-guard-weight 1 \
  --strict
```

结果：`accepted=true`。

| 检查 | 结果 |
|---|---:|
| direction metrics 有限且有界 | 100/100 |
| reverse 增大 direction cost | 87/87 |
| reverse 降低 motion alignment | 87/87 |
| reverse 降低 heading alignment | 87/87 |
| moving expert 优于 stop | 87/87 |
| collision 增大 collision cost | 84/84 |

100 场景候选均值：

| 候选 | direction cost | motion alignment | heading alignment |
|---|---:|---:|---:|
| expert | 0.02387 | 0.80321 | 0.99202 |
| stop | 0.00000 | 0.00000 | 0.99808 |
| reverse | 0.82978 | -0.71281 | -0.89995 |

报告 SHA-256：

```text
cfa3a073df34861da715cfca3e06c35b6f2eca1e6b42150693881d801c8214a5
```

### 6.4 相关测试

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q \
  HDP-nuplan/tests/test_rl_components.py \
  HDP-nuplan/tests/test_checkpoint_behavior.py
```

结果：`23 passed`。

### 6.5 全部 HDP-nuPlan 测试

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q \
  HDP-nuplan/tests
```

结果：`63 passed`，只有已有第三方库弃用警告。

尝试执行 `python -m black --check`，但当前环境没有安装 `black`；这不是代码失败，Python 编译和
全部 pytest 均已通过。

## 7. 当前结论

第一阶段已证明：

1. 新方向指标能在真实 NuPlan route 几何下稳定识别人工反向轨迹；
2. 正常 expert 的方向代价接近 0，反向轨迹接近 1，信号区分度足够；
3. 停车轨迹不会因为无位移被误判为逆向；
4. 默认权重为 0，旧实验兼容；
5. 训练、数据处理和评测测试没有回归。

尚未证明：加入该代价训练后一定提高 NuPlan 闭环分数。这个结论必须通过相同监督起点、相同
rollout manifest、相同随机种子下的 RL A/B 实验获得。

## 8. 下一步实验

只改一个变量进行配对实验：

```text
A：历史基线，direction_guard_weight=0
B：方向 guard，direction_guard_weight=1
```

其余参数保持第 19.8 节不变：group size 32、rollout steps 6、learning rate 4e-7、500 update、
progress guard 5、centered weights、expert anchor 0、encoder frozen。

先比较：

1. rollout 中 direction cost 与 reward 的组内方差；
2. validation-1000 的 direction cost、reverse fraction、progress、TTC 和 ADE；
3. 固定闭环场景的 overall、driving direction、TTC 和 progress；
4. 原退化场景是否恢复，同时原高收益场景是否保留收益。

只有 B 在不损害 progress/TTC 的前提下消除方向退化，才进入行为类型均衡采样和 expert anchor
消融阶段。

## 9. 第一阶段 direction guard=1 云端训练与评测

### 9.1 云端同步和训练

同步前先把云端五个目标文件与本地 Git `HEAD` 哈希比较，确认云端仍是本轮修改前的基线版本，
然后只同步 reward、RL 入口、两个评测脚本和对应测试。云端测试结果为 `58 passed`；数量少于
本地 63 是因为本地另有 5 个与本轮无关、尚未同步的新增测试。

云端配置：RTX 4090 单卡，训练时 GPU 空闲，数据盘剩余约 109GB。实验只在原参数上新增：

```text
reward_direction_guard_weight=1.0
```

其余设置保持：同一监督 epoch 10、同一 1000 场景 manifest、seed 3407、group 32、rollout 6、
learning rate 4e-7、500 update、progress guard 5、centered weights、expert anchor 0、冻结 encoder。

训练正常退出：

```text
buffer size: 1000
update steps: 500
active group fraction: 1.0
reward std mean: 0.0587238
final loss: -0.0301711
```

checkpoint：

```text
model_epoch_2_directionguard1_trainloss_-0.0302.pth
SHA-256: 1755d488220fd9342ba2d97f838fae320847ca0baf3e3688bb17c41c794c89f8
```

### 9.2 validation-1000 开环配对

同一监督模型、同一 validation-1000、3 repeats、每次 6 步单轨迹推理：

| 指标 | direction guard RL - supervised |
|---|---:|
| reward | +0.039660 |
| progress | +0.047299（+2.176%） |
| direction cost | -0.000540（-0.995%） |
| reverse fraction | -0.001763（-1.269%） |
| motion alignment | +0.003520 |
| collision cost | -0.004235（-4.545%） |
| comfort cost | -0.007293 |
| ADE | -0.169092 m（-6.297%） |

随后用新版评测器重跑旧 RL（guard=0）。新 guard=1 相对旧 RL 的额外变化仅为：

```text
direction cost: -0.0000365
reverse fraction: -0.0000973
progress: +0.0001994
ADE: -0.0005947 m
```

说明大部分增益来自原 RL，guard=1 产生的额外更新很小。

报告：

```text
supervised_vs_directionguard1_val1k_repeat3.json
SHA-256: 7cd278d25f93f88441b26d4f0c769679c99b73e39b4fc58ad10008509688ad57

supervised_vs_oldrl_guard0_val1k_repeat3_newmetrics.json
SHA-256: 64fe0597b45ae19b2628a251aeae762276c08b4af4a8f365452568700cd58f1e
```

### 9.3 固定 20 场景官方闭环

云端只有 3 个 mini DB，因此把新 checkpoint 下载回本机，在与历史监督/旧 RL 完全相同的环境
运行 `mini-val-closed-loop-20`。本机没有 `screen`，第一次 `nohup` 进程未进入 Python且没有
生成实验目录；随后改用持续 PTY 会话运行。最终耗时 16 分 10 秒，20/20 成功、0 失败，出现
3 次与历史两侧相同的 route warning。

| 指标 | supervised | old RL | guard=1 RL |
|---|---:|---:|---:|
| overall | 0.703840943 | 0.704483158 | 0.704511306 |
| progress | 0.653433099 | 0.676960608 | 0.677086150 |
| collision | 0.900000 | 0.900000 | 0.900000 |
| TTC | 0.800000 | 0.850000 | 0.850000 |
| drivable area | 0.950000 | 0.950000 | 0.950000 |
| driving direction | 1.000000 | 0.975000 | 0.975000 |

失败场景 `1fb0bb88f9d35d59`：

| 模型 | overall | direction | 官方 1 秒最小进度 |
|---|---:|---:|---:|
| supervised | 0.827087 | 1.0 | 0.000 m |
| old RL | 0.416657 | 0.5 | -3.557869 m |
| guard=1 RL | 0.416682 | 0.5 | -3.559009 m |

所以第一阶段 guard=1 **没有修复方向退化**。其 overall 相对旧 RL 的 `+0.0000281` 只是大量
微小轨迹变化，不能当作方向约束成功。

三方闭环报告：

```text
supervised_oldrl_directionguard1_mini_val20.json
SHA-256: 2f784e53a28e4e6b6eb6e72ee36c6cee3cafe2f25fd0d460b321d5ca96cd2829
```

## 10. 第一阶段失败原因：代理定义与官方指标不一致

检查 NuPlan devkit 官方 `DrivingDirectionComplianceStatistics` 后确认：官方指标不是平均方向余弦，
而是：

1. 根据 ego 所在 lane/lane connector 的 baseline 计算逐帧有符号进度；
2. 对过去 1 秒进度做滑动累计；
3. 最大反向累计小于 2 m 得 1 分；
4. 2～6 m 得 0.5 分；
5. 大于等于 6 m 得 0 分。

第一阶段 `direction_cost` 对 80 个规划点的余弦违规取平均。局部 1 秒严重逆行会被其余正常
时间步稀释，而且没有显式表达 2 m/6 m 阈值。因此它与官方指标相关，但不够对齐。

## 11. 第二阶段：对齐官方 1 秒滑窗指标

### 11.1 新增计算

在已有 route tangent 上计算每步有符号位移：

```text
signed_motion[t] = delta_p[t] dot route_tangent[t]
```

按照 `dt=0.1s`、官方 `time_horizon=1s` 构造滑动累计，得到：

```text
min_progress_in_1s
max_negative_progress = relu(-min_progress_in_1s)
```

连续方向代价为：

```text
progress_window_cost = clamp(max_negative_progress / 2m, 0, 1)
```

它从出现反向运动开始连续增大，在达到官方 2m 部分违规阈值时饱和为 1。同时按官方 2m/6m
阈值输出 `direction_compliance_score_approx ∈ {1, 0.5, 0}` 作为诊断。

最终 motion cost 取：

```text
max(旧余弦方向代价, progress_window_cost)
```

这样保留局部方向检查，同时防止 1 秒累计逆行被长时间平均稀释。

新增可复现参数：

```text
reward_direction_time_horizon=1.0
reward_direction_compliance_threshold=2.0
reward_direction_violation_threshold=6.0
```

### 11.2 本地验证

定向测试：`3 passed`。  
全部测试：`63 passed`。

真实 100 场景 sanity check：

| 检查 | 结果 |
|---|---:|
| direction metrics finite/bounded | 100/100 |
| reverse 降低 1 秒最小进度 | 87/87 |
| reverse 增大 direction cost | 87/87 |
| reverse 降低近似官方方向分 | 84/87（96.6%） |

候选均值：

| 候选 | direction cost | min progress in 1s | 近似官方方向分 |
|---|---:|---:|---:|
| expert | 0.02783 | +0.28895 m | 0.99 |
| stop | 0.00000 | 0.00000 m | 1.00 |
| reverse | 0.89455 | -5.39054 m | 0.37 |

第二阶段计划使用 `direction_guard_weight=5`，使方向安全项与当前 progress guard 处于相同量级。
这仍是固定权重消融，不冒充 CPO/PID-Lagrangian。

### 11.3 当前云端状态

准备同步第二阶段代码时，AutoDL SSH 端口 `11156` 连续返回 `Connection refused`。因此：

1. 第二阶段文件尚未上传；
2. 第二阶段训练尚未启动；
3. 第一阶段云端 checkpoint 和本地所有报告均已保存；
4. 云端恢复后应从“同步五个文件并运行云端测试”继续，不能误认为第二阶段已经训练。
