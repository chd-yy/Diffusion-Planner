# HDP-NAVSIM RL 协议迁移与门禁

## 结论

HDP-NAVSIM 不使用 Best-of-N、PPO 或 GRPO。它为每个场景生成 10 条候选，在前 5 个 epoch 对
轨迹施加局部纵向/横向平移，用 NAVSIM PDM scorer 评分，再以组内标准化 reward 的指数作为
diffusion MSE 权重。NuPlan 已迁移候选数和轨迹扰动；候选多样性显著提高，但 20-step 更新后
train1k、val1k 仍同时轻微退化，因此该 checkpoint 被拒绝。

## 原实现核对

核对文件：

```text
HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py
HDP-navsim/hdp_navsim/agent/dp_vla/scoring.py
HDP-navsim/hdp_navsim/agent/dp_vla/model/rl_utils.py
HDP-navsim/hdp_navsim/config/agent/dp_vla_rl_agent.yaml
```

它们与 `/home/yanjun/NewDisk/Hyper-Diffusion-Planner/HDP-navsim` 中的对应核心 agent/config 文件
无差异。原协议为：

- `group_size=10`，`rollout_steps=5`；
- `current_epoch < 5` 时调用 `augment_trajectory_batch()`；
- 每条轨迹只采样一对标准差 0.5 m 的纵向/横向偏移，沿时间共享，航向不变；
- reward 为 `pdm_result.score`；
- update 使用 `exp(group_zscore(reward)) * diffusion_mse`；
- ground truth 被 Replay Buffer 保存但 update 时丢弃，没有监督 anchor；
- 只优化 decoder。

配置中的 `progress_weight`、`ttc_weight`、`comfortable_weight`、`bc_data` 以及运行时计算的
`only_ep` 在当前核心 reward/loss 路径中没有实际生效。

## NuPlan 迁移

新增 `hdp_nuplan/rl/trajectory_augmentation.py`，接受 `[B,G,T,4]` 的
`[x,y,cos_yaw,sin_yaw]`：

\[
x'=x+a\cos\theta-b\sin\theta,\qquad
y'=y+a\sin\theta+b\cos\theta,
\]

其中 `a,b ~ N(0,std²)`，每条候选只采样一次并广播到全部未来点。新增参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `rl_group_size` | 10 | 对齐 NAVSIM 的每场景候选数 |
| `rl_trajectory_augmentation_std` | 0.5 | 局部平移标准差，单位 m；0 关闭 |
| `rl_trajectory_augmentation_epochs` | 5 | 零基 `epoch < 5` 时启用 |

行为对比工具新增 `--trajectory-augmentation-std`，默认 0，确保正常推理评估不会误加训练增强。

## 20-step 门禁结果

训练配置：监督 epoch 10 起点、train1k、batch 8、group 10、5-step diffusion、noise 0.2、
augmentation 0.5 m、学习率 `1e-5`、冻结 encoder、专家 anchor 0.1、最多 20 次 update。

rollout：reward=`0.331166`、progress=`2.311024`、collision cost=`0.175123`、route cost=
`0.194760`、comfort cost=`3.057588`。20-step update：total loss=`0.078735`、RL loss=
`0.072015`、anchor loss=`0.067193`、reward std mean=`1.062877`、active group fraction=`1.0`。

门禁比较关闭轨迹扰动，以共同随机噪声对监督与 RL checkpoint 各生成正常推理轨迹：

| 数据 | reward 变化 | progress 变化 | path length 变化 | collision cost 变化 | ADE 变化 |
|---|---:|---:|---:|---:|---:|
| train1k | -0.007442 | -0.002649 | -0.026392 m | +0.000481 | +0.009038 m |
| val1k | -0.004216 | -0.002656 | -0.026712 m | +0.000158 | +0.007910 m |

候选诊断表明迁移确实生效：

| group 指标 | 中位数 |
|---|---:|
| reward std | 0.110409 |
| reward range | 0.366213 |
| progress std | 0.053150 |
| endpoint diversity | 0.967435 m |
| reward-progress correlation | 0.703297 |
| best reward 候选的 progress 增量 | 0.071681 |
| best reward 候选的 path length 增量 | 0.439728 m |

对照此前仅 noise=0.2、group=4、无增强的结果，reward std 中位数为 `0.014335`、endpoint
diversity 中位数为 `0.197904 m`。因此 NAVSIM 协议已把低多样性问题显著缓解；更新后仍变短，
不能继续把退化归因于“候选几乎重合”。

## 决策

该 checkpoint 不进入完整 epoch、10k 或官方闭环评测。当前接受模型仍为监督 epoch 10。
从项目交付角度，忠实 NAVSIM 协议迁移已经完成并得到明确负结果；继续工作应转向总结交付。
Best-of-N 曾作为额外改进分支完成短步负实验，随后按用户决策从当前代码移除；当前主线继续保留
HDP-NAVSIM 的 exponential 全候选加权。
