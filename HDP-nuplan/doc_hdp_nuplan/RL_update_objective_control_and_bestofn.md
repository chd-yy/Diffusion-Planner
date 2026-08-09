# RL update 目标控制实验与 Best-of-N 门禁

> 当前状态：Best-of-N 已按用户决策从训练代码移除。本文件仅保留历史实验、结果和拒绝依据；
> 当前主线继续使用 HDP-NAVSIM 的 exponential 全候选奖励加权。

## 结论

三种候选协议的20步更新都产生约 `0.026 m` 的路径缩短后，本轮增加显式 loss 权重并完成
expert-anchor-only 控制实验。结果表明：额外监督更新只造成约 `0.004 m` 缩短，全候选 rollout
自蒸馏贡献了其余约 `0.022 m`。随后实现保守 Best-of-N；它比全候选加权稳定，但 train/val reward
仍未转正，也比 anchor-only 更差。因此当前 rollout 自蒸馏目标被正式拒绝，不继续调大训练规模。

## 控制开关

总损失改为显式形式：

\[
L=\lambda_{rollout}L_{rollout}+\lambda_{expert}L_{expert}.
\]

新增：

- `rl_rollout_loss_weight`，默认 1；
- `rl_candidate_weighting`，可选 `exponential` 或 `best_of_n`；
- 日志中的 `weighted_rl_loss` 与 `weighted_expert_anchor_loss`。

默认值保持此前行为。两个权重不能同时为0。

## Anchor-only 控制实验

固定上一轮 NAVSIM 候选协议和随机种子，只设置：

```text
rl_rollout_loss_weight=0
rl_expert_anchor_weight=1
```

日志确认 `weighted_rl_loss=0`、`weighted_expert_anchor_loss=0.068052`，实际执行20步。正常推理结果：

| 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 |
|---|---:|---:|---:|---:|
| train1k | -0.000834 | -0.000388 | -0.004016 m | +0.001818 m |
| val1k | -0.000743 | -0.000385 | -0.003994 m | +0.001697 m |

与全候选 NAVSIM 协议比较，train/val 路径退化分别减少约 `84.8%` 和 `85.0%`。这说明大部分漂移
来自 rollout 自蒸馏；anchor 自身的小幅退化符合“从已经选中的 epoch10 继续训练”的控制预期。

## Best-of-N 改进分支

每个 reward 方差达到阈值的场景只保留最高 reward 候选。胜者权重设为组大小 `G`，使对 `B*G`
求均值后等价于对场景胜者 loss 求均值；低方差组仍为零权重。本轮使用：

```text
group_size=10
sampling_noise_scale=0.2
trajectory_augmentation_std=0
candidate_weighting=best_of_n
rollout_loss_weight=0.25
expert_anchor_weight=1
max_update_steps=20
```

关闭0.5 m增强是为了避免让模型学习整条轨迹平移造成的首点位置跳变。update 中有效组比例为
`0.55625`，`weighted_rl_loss=0.010049`，`weighted_expert_anchor_loss=0.057369`。

正常推理门禁：

| 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 |
|---|---:|---:|---:|---:|
| train1k | -0.002097 | -0.000916 | -0.009667 m | +0.003684 m |
| val1k | -0.001501 | -0.000933 | -0.009743 m | +0.003572 m |

Best-of-N 比全候选加权稳定，但比 anchor-only 更差；新增的 rollout 分量在 train/val 上仍是负贡献。
因此该方案未通过门禁，不进入完整 epoch、10k 或官方闭环。

## 最终决策

不继续在 `rollout_loss_weight` 上做负区间插值，也不扩大当前离线 RL。项目最终模型保持监督
epoch10。已有实验已经区分：低候选多样性、全候选权重放大、专家监督漂移和 Best-of-N 选择，足以
形成完整、可复盘的技术结论。若未来重启 RL，前提应是引入与 NuPlan 官方闭环指标一致的 simulator
reward 或真正的策略优化目标，而不是继续拟合当前静态轨迹候选。
