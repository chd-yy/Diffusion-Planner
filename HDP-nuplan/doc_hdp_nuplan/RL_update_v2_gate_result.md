# HDP-nuPlan RL update v2 短步门禁结果

## 结论

本轮实现了低 reward 方差组跳过、有效组权重均值归一化、专家轨迹监督 anchor、候选噪声尺度可调
以及 update 步数上限。20-step 门禁显示，更新后的退化幅度远小于旧 RL，但 train1k 和 val1k 的
reward 仍未转正，因此两个新 checkpoint 都被拒绝，不进入完整 epoch 或官方闭环评测。当前接受模型
仍是监督 epoch 10。

## 代码改动

- `hdp_nuplan/rl/loss.py`：当组内 reward 标准差小于 `rl_min_reward_std` 时，将该组 rollout 权重
  置零；有效组权重归一化到均值 1。
- `hdp_nuplan/rl/train_epoch_rl.py`：RL loss 外加入真实 `ego_future` 的原监督扩散 loss；支持
  `rl_max_update_steps_per_epoch`；日志记录有效组比例和两个 loss 分量。
- `hdp_nuplan/model/module/decoder.py`：多候选初始噪声尺度可配置。普通 planner 默认仍为 0.1，
  RL 默认使用 0.2。
- `train_predictor_rl.py`：公开上述配置并检查参数范围。
- `scripts/compare_checkpoint_behavior.py`：行为对比支持显式指定相同的采样噪声尺度。

## 门禁协议

- 起点：已接受的监督 epoch 10。
- 数据：train1k 更新，另在独立 val1k 验证。
- 更新：只执行 20 个 optimizer step；`group_size=4`、`diffusion_steps=5`、学习率 `1e-5`。
- 稳定配置：`min_reward_std=0.01`、有效组权重均值 1、专家 anchor 权重 0.1、冻结 encoder。
- 比较：监督和 RL checkpoint 使用相同随机噪声，重复 3 次，共比较每个数据集 12000 条轨迹。
- 验收：train reward 必须先提高；若没有提高，立即停止，不扩大训练和闭环评测。

## 结果

| 初始噪声 | 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 0.1 | train1k | -0.006054 | -0.002522 | -0.025493 m | +0.008956 m | 拒绝 |
| 0.1 | val1k | -0.004725 | -0.002534 | -0.025700 m | +0.007946 m | 拒绝 |
| 0.2 | train1k | -0.008393 | -0.002555 | -0.025663 m | +0.008876 m | 拒绝 |
| 0.2 | val1k | -0.004312 | -0.002563 | -0.025924 m | +0.007775 m | 拒绝 |

旧 reward-v2 RL 在 train1k 上 reward 下降 `0.162830`、路径缩短 `0.548756 m`；v2 更新将漂移
压低了一个数量级，但严格门禁关心的是是否产生正收益，而不只是是否“退化得更少”。

将噪声从 0.1 增至 0.2 后，监督候选的 reward 标准差中位数从 `0.007111` 增至 `0.014335`，候选
终点两两距离中位数从 `0.098388 m` 增至 `0.197904 m`，说明候选多样性确实提高；但 20-step
更新仍没有提高模型行为，因此低多样性不是剩余问题的唯一原因。

## 产物

噪声 0.2 的训练 checkpoint：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-update-v2-noise02-gate20-from-10k-epoch10/2026-08-08-14:59:35/
model_epoch_2_trainloss_0.0457.pth
SHA-256: 1ea298958724eb63692944b63b554be4519a55a826875af867452fe221137a25
```

行为报告：

```text
tmp/mini_train_pilot_1000_seed3407_v1/behavior_update_v2_gate20_train1k_group4_step5_repeat3.json
tmp/mini_val_balanced_1000_seed3407_v1/behavior_update_v2_gate20_val1k_group4_step5_repeat3.json
tmp/mini_train_pilot_1000_seed3407_v1/behavior_update_v2_noise02_gate20_train1k_repeat3.json
tmp/mini_val_balanced_1000_seed3407_v1/behavior_update_v2_noise02_gate20_val1k_repeat3.json
tmp/mini_train_pilot_1000_seed3407_v1/supervised_group_diagnostics_noise0.2_repeat1.json
```

## 下一步边界

后续已优先完成忠实 HDP-NAVSIM 候选协议消融：group size 改为 10，并在早期 rollout 加 0.5 m
局部轨迹扰动。候选多样性显著提高，但 train/val 的 20-step reward 仍未转正，详见
`HDP_NAVSIM_RL_protocol_migration_gate.md`。因此不继续调大 epoch、数据量或采样噪声。若仍做额外
算法实验，应改变 update 目标本身。后续 anchor-only 控制和 Best-of-N 也已完成：Best-of-N 比全候选
加权稳定，但仍不如 anchor-only，详见 `RL_update_objective_control_and_bestofn.md`。
