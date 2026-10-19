# RL Following Reward 适用性解耦修复记录

修改时间：09月21日 16:38（北京时间）。

## 1. 问题定义

历史 `_following_reward()` 的 `leader_valid` 由每条 diffusion candidate 的位置、航向和横向 corridor 自行决定。候选可以通过横向偏移让 leader 失效，从低跟车分突变为“无 leader”的 1 分。

这不是“无前车应该给 0 还是 1”的标量选择，而是：

> 被评价策略不应该自己决定某个困难的奖励项是否适用。

本次将该问题定义为 **candidate-dependent applicability 导致的 objective-avoidance reward hacking**。

## 2. 修复方法

### 2.1 候选无关的 leader applicability

使用数据中已有的专家未来轨迹 `ego_future` 计算：

```text
reference_leader_valid = valid
                       AND vehicle
                       AND reference_longitudinal > 0
                       AND |reference_lateral| <= corridor
```

每个时刻选参考轨迹前方 gap 最小的车辆为 `reference_leader_index`，再 broadcast 给同场景的全部 `G` 条候选。

### 2.2 参考坐标系中的候选质量

候选仍使用自身位置、速度和加速度评分，但纵向量统一投影到专家轨迹朝向：

```text
gap      = (p_leader - p_candidate) dot forward_reference - half_lengths
v_ego    = v_candidate dot forward_reference
v_leader = v_leader dot forward_reference
a_long   = a_candidate dot forward_reference
```

因此候选无法通过偏移或旋转自己的航向删除 leader 或操纵纵向投影。

### 2.3 无 leader、短 leader 和换道

- 无 reference leader：`follow_reward=1` 只作诊断中性值，`follow_contribution=0`。
- 有效 leader 不足 8 帧：`follow_confidence=valid_steps/8`。
- 有效 leader 至少 8 帧：`follow_confidence=1`。
- 专家换道或当前 lane cache 不能覆盖专家：整组候选的 following 适用性同时关闭。

Legacy 加法目标使用：

```text
follow_contribution = follow_confidence * reference_follow_reward
R_base = risk_weight * risk
       + follow_weight * follow_contribution
       + lane_weight * lane
       + progress_guard_weight * progress_guard
```

几何组合目标使用：

```text
effective_follow_quality = 1 - follow_confidence * (1 - follow_reward)
```

不适用时 quality 等于乘法中性元素 1。

## 3. 兼容与配置

```text
reward_follow_assignment_mode=candidate         # 历史精确复现
reward_follow_assignment_mode=expert_reference  # 新训练
reward_follow_reference_full_weight_steps=8
```

- `NuPlanRewardConfig()` 保留 `candidate` 默认，避免直接调用者被静默改写。
- 通用 RL CLI 默认 `expert_reference`。
- 历史 `train_rl_safety_repair.py` 显式锁定 `candidate`。
- 后续 `run_midterm_posttraining.py` 显式锁定 `expert_reference`。
- 旧 `args.json` 缺少字段时，checkpoint 审计按 `candidate` 解释。

源码修改不会改写已有 B10、RL Epoch2 或 M2/M3/M4 checkpoint 的方法身份。

## 4. 测试

09月21日 16:34 定向测试：`39 passed`。覆盖：

- 历史 candidate 模式仍可复现；
- 横向偏移 4 m 后无法通过丢失 leader 获得 1 分；
- 无参考 leader 时 following 总贡献为 0；
- 专家换道时候选无关地关闭 following；
- 4/8 有效帧得到 `follow_confidence=0.5`。

09月21日 16:44 执行全部 `HDP-nuplan/tests`：`130 passed`，只有已知的第三方库 deprecation warnings。

## 5. 100 场景真实 NPZ 影子审计

数据为 10k mini-train manifest 的前 100 个真实 NPZ。对每个场景构造相同的 expert/stop/lateral/reverse/jitter/collision 候选，仅替换 following 定义，不训练模型。

原始结果：

- [candidate_100.json](../tmp/follow_reward_audit_0921/candidate_100.json)，SHA256 `d583cb038f58125d476b363f1f3e8513967a416439c100fe3005b9d49b40251d`；
- [expert_reference_confidence_100.json](../tmp/follow_reward_audit_0921/expert_reference_confidence_100.json)，SHA256 `91fd4a0c2e8bbfb9c2022d879049004e6a8bbc093bdeec9b707ead9881c32b75`。

| 指标 | 结果 |
|---|---:|
| reference following 适用场景 | 19 / 100 |
| lateral 候选发生 leader escape 的场景 | 12 / 100 |
| legacy 中 lateral 获得正 follow 差的场景 | 4 / 100（其中 3 场同时发生 leader escape） |
| legacy lateral-expert 最大 follow 增益 | +0.721056 |
| 修复后 lateral-expert 最大 follow 差 | +0.029506 |
| 修复后 reference leader fraction（全部场景） | 0.057625 |
| 修复后 follow confidence（全部场景） | 0.163750 |
| 适用场景内平均 follow confidence | 0.861842 |
| expert leader escape | 0 |

这证明漏洞在真实缓存上确实存在，且修复将最大横向逃逸增益从 `0.721056` 压到 `0.029506`。

### 不能夸大的结论

这份通用 reward sanity 报告的总 `accepted=false`，因为“运动专家优于 stop”仅从 `25/88` 变为 `26/88`。这说明：

- following applicability 漏洞已被定向修复；
- 整体 reward 仍存在 stop/progress 相对偏好问题；
- 不能根据本次影子审计宣称闭环安全或策略已提升。

## 6. 新日志指标

`reward.py` 现在输出：

- `legacy_follow_reward`；
- `reference_follow_reward`；
- `candidate_leader_fraction`；
- `reference_leader_fraction`；
- `follow_applicable`；
- `follow_confidence`；
- `follow_contribution`；
- `effective_follow_quality`；
- `leader_escape_fraction`；
- `reference_leader_switch_fraction`。

这些量由现有 rollout 日志汇总链自动记录，可用于后续 reward ablation 和失败场景分析。
