# HDP-nuPlan RL checkpoint 行为归因

## 结论

reward v2 的排序方向基本正确，但当前 RL 更新没有把生成分布推向高奖励候选。主要问题是候选
多样性过低、组内 z-score 放大微小差异，以及更新阶段缺少专家数据锚点。EMA 只能缓解退化，不能
修复更新方向。因此暂时不应扩大数据量或接入 simulator-in-the-loop，应先修正 RL update。

## 对比协议

- 监督模型：train10k 选出的 epoch 10。
- RL 模型：reward v2 pilot 的 epoch 2。
- 数据：train1k 和独立 val1k。
- 每个 batch 在两个模型采样前重置同一随机种子，保证初始扩散噪声相同。
- 协议 A：4 candidates、5 diffusion steps，复现 RL rollout。
- 协议 B：1 trajectory、10 diffusion steps，复现闭环 planner 推理。
- checkpoint 默认严格加载 `ema_state_dict`，另行比较 RL 的即时 `model` 权重。

## 主要结果

| 数据与协议 | 指标 | 监督 | RL EMA | 变化 |
|---|---|---:|---:|---:|
| train1k，4×5 | reward | 0.643565 | 0.480735 | -0.162830 |
| train1k，4×5 | progress | 2.294749 | 2.239211 | -0.055538 |
| train1k，4×5 | path length/m | 23.248677 | 22.699921 | -0.548756 |
| val1k，4×5 | reward | 1.245572 | 1.141913 | -0.103659 |
| val1k，4×5 | progress | 2.311822 | 2.256419 | -0.055403 |
| val1k，4×5 | path length/m | 23.456154 | 22.902174 | -0.553980 |
| val1k，1×10 | reward | 1.313272 | 1.191847 | -0.121425 |
| val1k，1×10 | progress | 2.358922 | 2.292237 | -0.066685 |
| val1k，1×10 | path length/m | 23.921527 | 23.252980 | -0.668547 |

train 和 val 的 reward 同时下降，排除了单纯过拟合。5 步和 10 步趋势一致，排除了训练/推理
采样步数不一致是主要原因。10 步协议中 RL 仅在 0.1% 的配对轨迹上提高 progress，仅在 1.77%
的配对轨迹上提高 reward，退化具有一致性。

即时 RL `model` 在 train1k 上比 EMA 更差：reward=`-0.804320`、path length=`18.067617 m`；
监督模型分别为 `0.645065` 和 `23.246511 m`。因此 EMA 只是在平滑更新造成的漂移。

## 组内候选诊断

监督模型在 train1k 的 4 候选组中：

| 组内统计 | 中位数 |
|---|---:|
| reward 标准差 | 0.007111 |
| reward 极差 | 0.018100 |
| progress 标准差 | 0.006437 |
| 终点两两距离 | 0.098388 m |

候选几乎重合，但 `group_advantage_weights()` 会除以每组自身标准差。因此 reward 差异无论是
`0.007` 还是 `0.7`，标准化后都可产生相近的指数权重；实际训练日志的平均权重约为 `1.59`。

同时，组内 reward-progress 相关系数中位数为 `0.998305`；每组最高 reward 候选相对组均值平均：

- progress 增加 `0.017117`，约等于 0.171 m route 投影进度；中位数增加约 0.065 m；
- path length 增加 `0.167490 m`，中位数增加 `0.063770 m`；
- collision cost 平均降低 `0.003948`。

这说明 reward 排序通常同时偏好更高进度和更低碰撞代价，模型最终变短并不是 reward 直接选择了
更短候选，而是奖励加权自蒸馏没有稳定复现这种非常微弱的组内偏好。

## 建议修复顺序

1. 对 reward 标准差过小的组跳过奖励加权，避免把数值噪声当作强优势。
2. 将指数权重按组归一化为均值 1，固定有效学习率。
3. 在 RL update 中加入真实 `ego_future` 的监督扩散 anchor，防止只拟合旧策略样本造成自蒸馏漂移。
4. 提高候选有效多样性后，再重新评估 group size 和初始噪声尺度。
5. 每 10–20 个 update step 运行小型同噪声门禁；train reward 没有提高就立即停止，不再等完整 epoch。

在上述门禁通过前，不扩大到 10k RL，也不接入昂贵的 simulator-in-the-loop。

## 工具与证据

归因工具：`scripts/compare_checkpoint_behavior.py`。

主要报告：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
behavior_supervised_vs_rl_reward_v2_rollout5_group4_repeat3.json
behavior_supervised_vs_rl_reward_v2_inference10_single_repeat3.json

tmp/mini_train_pilot_1000_seed3407_v1/
behavior_train1k_supervised_vs_rl_reward_v2_rollout5_group4_repeat3.json
behavior_train1k_group_preference_diagnostics_repeat1.json
behavior_train1k_supervised_ema_vs_rl_model_group4_step5_repeat1.json
```
