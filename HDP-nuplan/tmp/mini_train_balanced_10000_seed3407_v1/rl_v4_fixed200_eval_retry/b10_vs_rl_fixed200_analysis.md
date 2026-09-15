# fixed200 配对闭环自动分析

- 评测状态：`mixed`
- 基线：`hdp_b_epoch10`
- 配对规则：同一 `scenario` 逐场景相减，候选值减基线值；均值指标越大越好。

## 总体均值差值

| 指标 | RL seed42 - B10 | RL seed2026 - B10 |
|---|---:|---:|
| `score` | -0.00006979 | 0.00149828 |
| `no_ego_at_fault_collisions` | 0.00000000 | 0.00000000 |
| `drivable_area_compliance` | 0.00000000 | 0.00000000 |
| `ego_is_making_progress` | 0.00000000 | 0.00000000 |
| `driving_direction_compliance` | 0.00000000 | 0.00000000 |
| `ego_progress_along_expert_route` | -0.00042174 | 0.00519588 |
| `time_to_collision_within_bound` | 0.00000000 | 0.00000000 |
| `speed_limit_compliance` | 0.00012573 | -0.00019598 |
| `ego_is_comfortable` | 0.00000000 | 0.00000000 |

## 逐场景胜负统计

| 指标 | seed42 胜/负/平 | seed2026 胜/负/平 |
|---|---:|---:|
| `score` | 33/60/107 | 71/23/106 |
| `no_ego_at_fault_collisions` | 0/0/200 | 0/0/200 |
| `drivable_area_compliance` | 0/0/200 | 0/0/200 |
| `ego_is_making_progress` | 0/0/200 | 0/0/200 |
| `driving_direction_compliance` | 0/0/200 | 0/0/200 |
| `ego_progress_along_expert_route` | 26/69/105 | 78/18/104 |
| `time_to_collision_within_bound` | 0/0/200 | 0/0/200 |
| `speed_limit_compliance` | 14/0/186 | 2/12/186 |
| `ego_is_comfortable` | 0/0/200 | 0/0/200 |

## 自动判断

- `consistent_positive`：两个 seed 的 score 均值都提升，且无责任碰撞指标不下降。
- `mixed`：两个 seed 的结果方向不一致，不能宣称 RL 稳定带来正收益。
- `not_positive`：两个 seed 均未显示 score 正收益。
- 若状态为 `evaluation_incomplete`，不能作任何收益结论。
