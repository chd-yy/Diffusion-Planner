# HDP-NuPlan RL v8：proxy v3 + positive advantage 操作日志

## 1. 目标与固定边界

本实验针对 v7 fixed200 负收益进行修正，目标是验证 RL 是否能在相同训练集和相同 B Epoch 10 起点上改善闭环结果。

训练集固定为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache
```

要求严格满足：

- NPZ 数量：10,000；
- manifest：10,000 个唯一条目；
- 监督起点：B Epoch 10；
- `rl_detach_window_size=0`；
- 不下载、不生成、不替换训练 NPZ。

## 2. v7 失败原因复盘

v7 的 fixed200 结果为 `0.90168001`，B Epoch 10 为 `0.90456672`，差值 `-0.00288672`。其中路线进度下降 `-0.01028475`，而碰撞、TTC、可行驶区域和舒适性没有变化。

审计确认：

1. v2 proxy 虽然计算了 `collision_cost`，但它只进入 `details`，没有进入最终 reward；
2. v7 关闭硬筛选后，32 条候选全部参与正权重拟合，产生全候选自蒸馏；
3. v7 的 reward 标准差中位数只有 `0.0022155`，但 `rl_min_reward_std=1e-6` 使 `97.3%` 的组都参与更新；
4. v7 的 reference anchor 加权损失为 `3.53e-7`，只约占 RL loss 的 `0.023%`，约束过弱。

## 3. 代码修正

### 3.1 `nuplan_score_proxy_v3`

保留 v2 以保证历史实验可复现，新增 v3。v3 使用：

```text
collision_quality = 1 - collision_cost
safety_quality = risk_reward × collision_quality × no_collision
```

因此连续 `collision_cost` 真正影响候选排序；实际碰撞仍由 `no_collision=0` 保持安全优先。route、progress、comfort 和 follow 继续使用有界质量分和加权几何平均。

### 3.2 `positive_advantage`

新增 rollout 权重模式：

```text
weight = max(advantage, 0)
```

低于或等于组内平均 reward 的候选权重为 0，只有高优势候选参与拟合。它与 `softmax_positive` 的区别是：不会给所有候选保留一个恒定的自蒸馏基线。

### 3.3 reward 区分度门槛

v8 显式使用：

```text
rl_min_reward_std = 0.002
```

低于该门槛的候选组整体跳过 rollout 更新，避免把采样噪声标准化成强 RL 信号。

### 3.4 reference anchor 校准

v7 实测 raw reference anchor loss 为 `7.05e-6`，RL loss 为 `1.50e-3`。若希望 reference 项约占 RL loss 的 10%，所需权重约为：

```text
0.10 × 1.50e-3 / 7.05e-6 ≈ 21
```

因此 v8 使用：

```text
rl_reference_anchor_weight = 20.0
```

最终仍以训练日志中的 `weighted_reference_anchor_loss / weighted_rl_loss` 检查实际比例，不能只看命令行参数。

## 4. 验证入口

代码单元测试：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q HDP-nuplan/tests
```

新增测试覆盖：

- v3 连续 collision cost 会改变 reward 排序；
- positive advantage 会去掉低优势候选的权重；
- reference anchor 会进入总 update loss。

v8 训练入口：

```bash
bash HDP-nuplan/scripts/run_rl_v8_proxy_v3_positive_advantage_10k_seed2026.sh
```

训练前必须先运行 reward sanity check，并确认 `accepted=true`。训练完成后只使用同一个 fixed200 场景集合评测，复用已经完成的 B Epoch 10 基线，不重复评测基线。

## 5. 当前状态

截至本日志创建时，代码检查结果为：

```text
58 passed
```

v8 尚未启动训练；需先完成 v3 reward sanity check 和参数审计。

## 6. 实际执行记录

v3 sanity check 已通过：`accepted=true`，20 个场景上的路线偏离、舒适性、倒退、停止和碰撞排序检查均通过。

固定数据审计通过：

```text
cache NPZ = 10000
manifest = 10000
manifest unique = 10000
```

2026-08-25 已启动 v8 训练，后台启动入口为：

```bash
bash HDP-nuplan/scripts/run_rl_v8_proxy_v3_positive_advantage_10k_seed2026.sh
```

训练目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10
```

训练结束后由以下脚本自动启动固定 200 场景评测：

```bash
bash HDP-nuplan/scripts/evaluate_rl_v8_fixed200_parallel2.sh
```

评测脚本复用已完成的 B Epoch 10 fixed200 汇总，不重复运行基线，只运行 v8，并使用双线程并行闭环。

## 7. v8 训练结果

训练于 2026-08-25 05:06 前完成，checkpoint 为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10/training_log/hdp-rl-proxy_v3_positive_advantage_ref20_10k_seed2026-from-full-mini-b10/2026-08-25-04:55:36/model_epoch_2_trainloss_0.0006.pth
```

训练统计：

| 指标 | 数值 |
|---|---:|
| total loss | 0.0005951 |
| reward-weighted diffusion loss | 0.0001551 |
| reward-weighted waypoint loss | 0.0087564 |
| reward mean | 0.5189256 |
| reward std p10 / p50 / p90 | 0.0006293 / 0.0022190 / 0.0038088 |
| active group fraction | 0.099 |
| weighted RL loss | 0.0002426 |
| weighted expert anchor loss | 0.0003470 |
| raw reference anchor loss | 2.7410e-7 |
| weighted reference anchor loss | 5.4820e-6 |
| update steps | 500 |

`active_group_fraction=0.099` 表明 `rl_min_reward_std=0.002` 确实过滤掉了低区分度组；这与 v7 的 `0.973` 有明显区别。reference 项占 weighted RL loss 约 `2.26%`，虽然还未达到预设的 10%，但已经比 v7 的约 `0.023%` 提高约两个数量级，且没有出现 loss 爆炸。

训练期间发现 loss 诊断字段 `positive_weighting_mode` 仍沿用旧模式判断，已修正为同时识别 `softmax_positive` 和 `positive_advantage`；该修正只影响后续日志字段，不影响已完成 checkpoint。

## 8. 自动 fixed200 评测状态

训练完成后已自动启动：

```bash
bash HDP-nuplan/scripts/evaluate_rl_v8_fixed200_parallel2.sh
```

当前使用双线程并行，只评测 v8，复用历史 B Epoch 10 fixed200 结果。最终 score 和逐场景差异待评测完成后补充。

## 9. fixed200 评测恢复与最终结果

首次 v8 fixed200 运行于 `172/200` 附近被外部中断，未发现 NuPlan 场景异常；随后检查指标目录确认已有 `174/200` 个场景写出临时指标文件，缺少 26 个 token。为避免重复运行已完成场景，新增过滤配置：

```text
HDP-nuplan/hdp_nuplan/config/scenario_filter/mini-val-fixed-200-rl-v8-missing-26.yaml
```

恢复入口为：

```bash
bash HDP-nuplan/scripts/resume_rl_v8_fixed200_missing26.sh
```

恢复过程使用与原评测完全相同的 checkpoint、NuPlan mini DB、planner 配置、`worker.max_workers=2` 和同一 `experiment_uid`，只将 `scenario_filter` 换成缺失的 26 个 token。NuPlan 在恢复运行结束时自动完成了指标整合，因此临时 `pickle.temp` 文件被正常清理；恢复脚本原先按临时文件数量判断完成，曾误报 `found 0`，不代表评测失败。

最终确认：

```text
补跑：26/26 successful，0 failed
指标聚合：200 个真实场景
同场景配对：200/200
```

最终结果文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10/fixed200_eval_parallel2/rl_v8_fixed200_only.json
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_v8_fixed200.json
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_v8_fixed200_analysis.json
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_proxy_v3_positive_advantage_ref20_10k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_v8_fixed200_analysis.md
```

### 9.1 B Epoch 10 与 RL v8

| 指标 | B Epoch 10 | RL v8 | 平均变化 |
|---|---:|---:|---:|
| score | 0.9045667 | 0.8996747 | -0.0048920 |
| route progress | 0.9120677 | 0.9122254 | +0.0001577 |
| no at-fault collision | 0.9575 | 0.9575 | 0 |
| drivable area | 0.9500 | 0.9500 | 0 |
| making progress | 0.9900 | 0.9850 | -0.0050 |
| TTC compliance | 0.9350 | 0.9350 | 0 |
| speed-limit compliance | 0.9954311 | 0.9908772 | -0.0045540 |
| comfort | 0.9950 | 0.9950 | 0 |

score 的逐场景配对统计为：`41 wins / 53 losses / 106 ties`。因此本次 v8 改造没有带来正收益，主要退化来自 making-progress 和 speed-limit compliance；route progress 只有极小的正变化，安全指标没有改善。

### 9.2 结果解释

本次实验验证了“positive advantage + v3 proxy reward + reference anchor”代码链路能够稳定运行，但不能据此宣称 RL 已学到新能力。原因是：

1. 训练中 `active_group_fraction=0.099`，只有约 9.9% 的候选组真正参与 RL 更新，信号覆盖率很低；
2. fixed200 上碰撞、可行驶区域和 TTC 指标与 B Epoch 10 完全持平，说明 RL 没有改善安全行为；
3. reference anchor 的实际权重只占 weighted RL loss 约 2.26%，仍不足以有效约束策略漂移；
4. 正优势加权只保留高于组内 baseline 的候选，配合低 active fraction，容易让更新集中在少量噪声较大的候选上；
5. 因此下一轮应先提高有效候选覆盖率、校准 reference anchor 比例并增加训练轮次，再重新做固定场景配对评测，不能直接扩大到全 mini-val。
