# RL reference-relative 多轮更新操作日志

## 1. 本次目的

100k RL 实验相对 B Epoch10 的固定 200 场景结果为 `-0.000453`。分析显示，当前
`positive_advantage` 只把候选与同组均值比较，可能学习“比同组平均好、但比 B Epoch10
差”的轨迹；同时 2 epoch 只有一次 rollout-update，且 `rl_min_reward_std=0.002`
使只有约 9.3% 的组参与 RL 更新。

本次改造验证三个问题：

1. advantage 改为候选 reward 相对冻结 B Epoch10 reward；
2. 将 `rl_min_reward_std` 降为 `1e-6`，先观察低方差组是否提供有效信号；
3. 训练 6 epoch，按 `rl_buffer_update_epoch=2` 形成 3 次 rollout-update。

## 2. 代码改造

### 2.1 reference-relative reward

rollout 阶段在每个场景额外使用冻结的 B Epoch10 reference model 生成一条确定性候选，
计算 `reference_reward` 并写入 Replay Buffer。update 阶段使用：

```text
advantage = (candidate_reward - reference_reward) / (group_reward_std + eps)
```

`positive_advantage` 模式下，只有超过 B Epoch10 reference reward 的候选获得正权重。
旧 replay 条目不包含 reference reward 时仍兼容组内均值 baseline。

涉及文件：

- `HDP-nuplan/hdp_nuplan/rl/replay_buffer.py`
- `HDP-nuplan/hdp_nuplan/rl/loss.py`
- `HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py`
- `HDP-nuplan/train_predictor_rl.py`

新增参数：

```text
--rl_relative_to_reference true
--rl_reference_noise_scale 0.0
```

### 2.2 多轮训练脚本

新增脚本：

```text
HDP-nuplan/scripts/run_rl_reference_relative_10k_6ep_seed2026.sh
```

主要参数：

```text
数据：10,000 NPZ
起点：B Epoch10
train_epochs：6
rl_buffer_update_epoch：2
rollout-update：3 次
rl_min_reward_std：1e-6
rl_group_size：32
rl_reference_anchor_weight：20
rl_expert_anchor_weight：0.1
rl_detach_window_size：0
seed：2026
```

旧实验脚本和旧 checkpoint 均未覆盖。

## 3. 验证记录

代码改造后执行：

```bash
python -m py_compile \
  HDP-nuplan/hdp_nuplan/rl/replay_buffer.py \
  HDP-nuplan/hdp_nuplan/rl/loss.py \
  HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py \
  HDP-nuplan/train_predictor_rl.py

/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果：`59 passed`。

## 4. 实验运行

启动命令：

```bash
bash HDP-nuplan/scripts/run_rl_reference_relative_10k_6ep_seed2026.sh
```

后台日志：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
  rl_reference_relative_v1_10k_6ep_seed2026_logs/launcher.out
```

训练输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
  rl_reference_relative_v1_10k_6ep_seed2026_from_b10/
```

启动后确认第 1 次 rollout 正常运行，数据集为 10,000 NPZ，DataLoader 为 5,000 个
batch；因为每个 batch 还需要计算一条 B Epoch10 参考轨迹，速度约为 7～9 batch/s。

训练于 2026-08-26 完成，Epoch2、Epoch4、Epoch6 checkpoint 均已保存。Epoch6 最终
update 指标如下：

| 指标 | 数值 |
|---|---:|
| total loss | 0.0016985 |
| reward-weighted diffusion loss | 0.0006930 |
| reward-weighted waypoint loss | 0.0540465 |
| reward mean | 0.5321430 |
| reference reward mean | 0.5349475 |
| reference positive fraction | 0.604 |
| reward std mean | 0.0034178 |
| active group fraction | 0.977 |
| weighted expert anchor loss | 0.0003260 |
| weighted reference anchor loss | 0.0001391 |
| update steps | 500 |

这说明 reference-relative 改造确实提高了有效训练组比例，并且候选 reward 中有相当
一部分超过参考策略；是否转化为真实闭环收益，必须以固定场景评测为准。

## 5. 判断标准

训练完成后先检查：

1. `reference_positive_fraction` 是否大于 0，确认确实存在超过 B Epoch10 的候选；
2. `active_group_fraction` 是否明显高于旧实验的 `0.093`；
3. 三次 update 的 route/progress 相关指标是否持续改善；
4. 固定 200 场景闭环中 score 和 route progress 是否相对 B Epoch10 为正；
5. collision、TTC、drivable area、comfort 不得明显退化。

只有固定场景闭环结果为正，且上述安全指标不退化，才继续扩大到 100k 数据。

## 6. 固定 200 场景闭环评测

训练完成后使用 Epoch6 checkpoint 启动固定 200 场景评测，评测脚本为：

```text
HDP-nuplan/scripts/evaluate_rl_reference_relative_fixed200.sh
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
  rl_reference_relative_v1_10k_6ep_seed2026_from_b10/fixed200_eval/
```

评测于 2026-08-26 03:21 完整结束，200/200 场景集合与 B Epoch10 基线一致。

| 指标 | B Epoch10 | RL reference-relative Epoch6 | 差值 | 胜/负/平 |
|---|---:|---:|---:|---:|
| score | 0.90456672 | 0.89830025 | -0.00626647 | 7/89/104 |
| no at-fault collision | 0.95750000 | 0.95750000 | 0.00000000 | 0/0/200 |
| drivable area | 0.95000000 | 0.95000000 | 0.00000000 | 0/0/200 |
| making progress | 0.99000000 | 0.98500000 | -0.00500000 | 0/1/199 |
| route progress | 0.91206773 | 0.90737836 | -0.00468937 | 0/98/102 |
| TTC | 0.93500000 | 0.93500000 | 0.00000000 | 0/0/200 |
| speed compliance | 0.99543114 | 0.99114009 | -0.00429105 | 13/1/186 |
| comfort | 0.99500000 | 0.99500000 | 0.00000000 | 0/0/200 |

结论：本次改造把 active group fraction 从旧 100k 实验的约 `0.093` 提升到
`0.977`，并得到 `0.604` 的 reference-positive candidate fraction，但最终闭环仍为
负收益。训练 proxy reward 相对 B Epoch10 的提升与闭环 NuPlan score 不一致；多轮更新
放大了这类错位，主要表现为路线进度和限速合规下降，而安全指标没有获得收益。

结果文件：

```text
fixed200_eval/b10_vs_reference_relative_fixed200_analysis.json
fixed200_eval/b10_vs_reference_relative_fixed200_analysis.md
```

## 7. Epoch2 / Epoch4 退化曲线评测

为了区分“第一次更新方向就错”与“多轮更新过度”，保留训练集、场景集合、仿真器、
随机种子和 B Epoch10 基线不变，并行启动 Epoch2 与 Epoch4 的 fixed200 评测：

```text
HDP-nuplan/scripts/evaluate_rl_reference_relative_epoch2_epoch4_fixed200.sh
```

两组各使用 1 个 simulation worker，总并发为 2，与 Epoch6 评测的总 worker 数相同。
输出目录：

```text
rl_reference_relative_v1_10k_6ep_seed2026_from_b10/
  fixed200_epoch2_epoch4_eval/
```

启动时间：2026-08-26 04:25 CST。最终结果待评测完成后补录。

评测中途记录：两组 simulation 均在约 90 分钟后停止，Epoch2 已完成 120/200，
Epoch4 已完成 121/200。日志没有 Python traceback、CUDA error 或系统 OOM 记录，
且已有 `.pickle.temp` 指标文件仍然保留，因此判定为外部进程生命周期/后台任务中断，
不是场景评测逻辑失败。

已生成断点恢复配置并启动缺失场景补跑：

```text
Epoch2：已完成 120，补跑 80
Epoch4：已完成 121，补跑 79
```

恢复配置由 `scripts/prepare_fixed200_resume_filter.py` 根据已有指标文件自动生成，
恢复命令由 `scripts/resume_rl_reference_relative_epoch2_epoch4_fixed200.sh` 执行，
不会重新计算已完成场景。

## 8. 断点恢复后的再次续跑

2026-08-26 期间，第一次断点恢复运行到 Epoch2 `55/80`、Epoch4 `54/79` 时，
两个后台评测进程被暂停；当时各自累计完成 `175/200`。恢复前重新执行
`scripts/prepare_fixed200_resume_filter.py`，根据当前 `metrics/*.pickle.temp` 文件
重新生成缺失场景过滤器，确认两组都只剩 `25` 个场景。

随后使用与原评测完全相同的 checkpoint、NuPlan 闭环配置、固定 200 场景过滤器和
单 worker 参数，从剩余 25 个场景并行续跑：

```text
Epoch2 PID：25073
Epoch4 PID：25074
```

完成后应分别达到 `200/200`，再自动生成 Epoch2/Epoch4 汇总及与 B Epoch10 的
配对比较。已有 175 个指标文件不删除、不重复计算。

## 9. Epoch2 / Epoch4 fixed200 最终结果

剩余场景补跑于 2026-08-26 12:58 CST 完成。Epoch2 和 Epoch4 均无仿真失败，
最终聚合结果都包含相同的 200 个 fixed200 场景。由于断点续跑，最终一次
`runner_report.parquet` 只记录新补跑的 25 个场景；分析脚本同时读取前 175 个已保留
场景指标，因此配对分析中的实际场景数仍为 `200`，且
`same_scenario_set=true`、`evaluation_complete=true`。

| 模型 | score | 相对 B10 | route progress | 相对 B10 | speed compliance | 相对 B10 |
|---|---:|---:|---:|---:|---:|---:|
| B Epoch10 | 0.90456672 | 0 | 0.91206773 | 0 | 0.99543114 | 0 |
| RL Epoch2 | 0.90343762 | -0.00112910 | 0.90799076 | -0.00407697 | 0.99565243 | +0.00022129 |
| RL Epoch4 | 0.90325512 | -0.00131160 | 0.90730290 | -0.00476483 | 0.99568556 | +0.00025442 |
| RL Epoch6 | 0.89830025 | -0.00626647 | 0.90737836 | -0.00468937 | 0.99114009 | -0.00429105 |

配对场景统计：

```text
Epoch2 score：6 胜 / 90 负 / 104 平
Epoch4 score：8 胜 / 88 负 / 104 平

Epoch2 route progress：0 胜 / 98 负 / 102 平
Epoch4 route progress：0 胜 / 98 负 / 102 平
```

collision、drivable area、making progress、driving direction、TTC 和 comfort 在
Epoch2/Epoch4 上均与 B Epoch10 相同。Epoch2/Epoch4 仅带来约 `+0.00022` 至
`+0.00025` 的限速合规提升，但路线进度下降约 `0.0041` 至 `0.0048`，无法抵消路线
退化，所以总分为负。

结论：第一次 RL 更新（Epoch2）已经朝错误的闭环目标方向移动；继续更新到 Epoch4
没有修复该问题，Epoch6 则进一步放大总分退化。因此问题不是单纯“训练轮数过多”，
而是训练 reward / reference-relative proxy 与 NuPlan 闭环路线进度目标仍不对齐。

结果文件：

```text
fixed200_epoch2_epoch4_eval/b10_vs_epoch2_analysis.md
fixed200_epoch2_epoch4_eval/b10_vs_epoch4_analysis.md
fixed200_epoch2_epoch4_eval/b10_vs_epoch2_epoch4_fixed200.json
```

## 10. 原始 Diffusion Planner fixed200 对照评测

为回答“B Epoch10 是否超过原始 Diffusion Planner”，新增同场景对照评测脚本：

```text
HDP-nuplan/scripts/evaluate_original_diffusion_fixed200.sh
```

该脚本使用仓库发布的：

```text
checkpoints/args.json
checkpoints/model.pth
```

并严格复用 B Epoch10 的 `mini-val-fixed-200-rl-v4`、NuPlan mini DB、
`closed_loop_nonreactive_agents`、2 个 thread worker、0.5 GPU/仿真和 metric 配置。
评测于 2026-08-26 13:12 CST 启动，输出目录为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/original_diffusion_fixed200_eval/
```

评测完成后自动生成原始 Diffusion Planner 的汇总，以及与 B Epoch10 的逐场景配对结果。

评测于 2026-08-26 完成，原始 Diffusion Planner 的 200 个场景均成功，且与 B Epoch10
使用完全相同的场景集合。配对分析结果如下：

| 指标 | B Epoch10 | 原始 Diffusion Planner | 差值（原始 DP - B10） | 胜/负/平 |
|---|---:|---:|---:|---:|
| score | 0.90456672 | 0.95797087 | +0.05340415 | 88/33/79 |
| no at-fault collision | 0.95750000 | 0.98000000 | +0.02250000 | 10/1/189 |
| drivable area | 0.95000000 | 0.99000000 | +0.04000000 | 9/1/190 |
| making progress | 0.99000000 | 1.00000000 | +0.01000000 | 2/0/198 |
| route progress | 0.91206773 | 0.96030783 | +0.04824009 | 89/22/89 |
| TTC | 0.93500000 | 0.97500000 | +0.04000000 | 9/1/190 |
| speed compliance | 0.99543114 | 0.99433434 | -0.00109679 | 5/12/183 |
| comfort | 0.99500000 | 0.97000000 | -0.02500000 | 1/6/193 |

因此，在同一 fixed200 场景上，B Epoch10 的 score 比仓库发布版原始 Diffusion Planner
低 `0.05340415`，即低约 `5.57%`（以原始 Diffusion Planner 为分母）。主要差距来自
expert-route progress，B Epoch10 低 `0.04824009`；原始模型同时在碰撞、可行驶区域、
making progress 和 TTC 上更好，但舒适性略低、限速合规略低。

最终对比文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
  original_diffusion_fixed200_eval/b10_vs_original_diffusion_fixed200_analysis.md
```
