# 全量 306,801 场景安全约束后训练：Test14 全量闭环评测结果与分析

> 完成时间：10月06日 22:23  
> 评测状态：`test14-hard` 272/272、`test14-random` 261/261 全部成功，无失败仿真。  
> 当前决策：**新模型不替换 B10；保留为已证明具有任务改进能力、但尚未满足安全接受条件的实验 checkpoint。**

## 1. 本次评测回答什么问题

本实验检验全量 306,801 场景后训练模型相对监督学习基线 B10，是否在未参与训练的 Test14 闭环场景中实现了：

1. 可测且方向正确的策略变化；
2. 闭环任务得分和路线进度提升；
3. 责任碰撞、可行驶区域、TTC、舒适性等安全指标不退化；
4. 足以支持用新模型替换 B10 的整体改进。

其中，第 2 项回答“后训练是否有效”，第 3、4 项回答“当前安全约束是否足够”。因此不能只根据总分上升判断模型已经通过安全验收。

## 2. 实验协议与身份固定

| 项目 | 设置 |
|---|---|
| 基线 | 历史 HDP B10，EMA 权重 |
| 候选 | 全量 306,801 场景安全约束后训练 Epoch 2，EMA 权重 |
| 仿真模式 | nuPlan `closed_loop_nonreactive_agents` |
| 场景 | Test14-hard 272；Test14-random 261 |
| 推理随机种子 | `seed=0` |
| worker | 单 worker，避免共享 metric engine 的并发污染 |
| 分片 | 每片 40 场景；hard/random 各 7 片 |
| B10 checkpoint SHA256 | `22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce` |
| 后训练 checkpoint SHA256 | `41cf43c5a7b7a70663409b9cb1260e6d549814b128b2b91cc0e94af93cf58763` |
| Test14 清单 SHA256 | `a6b495893945d719055d9841a91e1e637063da7f3d6b439c5e82077d5964230d` |

按照已确定的节省算力方案，本次只运行新模型，复用历史 B10。复用前对 B10 的 checkpoint、args、Test14 清单及逐场输出进行了重新校验：hard 为 272/272 成功，random 为 261/261 成功，且场景集合与当前冻结清单完全一致。普通闭环推理未启用 guided denoising 或多尺度候选池；当前新增接口在默认参数下仍为单候选、`noise_scale=0.1`、无 guidance 的历史推理语义。

需要明确：这是“严格身份校验后的历史结果复用”，不是同一时刻重新运行 B10。因而运行耗时不能作为两模型计算效率差异的正式证据。

## 3. 评测完整性

| 数据集 | 期望场景 | 新模型成功 | 失败 | runner 是否完整 | 聚合指标是否完整 |
|---|---:|---:|---:|---|---|
| Test14-hard | 272 | 272 | 0 | 是 | 是 |
| Test14-random | 261 | 261 | 0 | 是 | 是 |

两套集合共有 533 条评测记录，但存在 11 个重叠场景，因此只有 522 个不同 token。后文以 hard 和 random 分别报告为主，不把 533 条记录当作相互独立样本进行显著性推断。

## 4. Test14-hard 结果

| 指标（越大越好） | 历史 B10 | 全量后训练 | 差值（后训练-B10） | 逐场胜/负/平 |
|---|---:|---:|---:|---:|
| Overall score | 0.705304 | **0.714113** | **+0.008809** | 115/56/101 |
| No ego at-fault collision | **0.867647** | 0.862132 | **-0.005515** | 5/6/261 |
| Drivable-area compliance | **0.944853** | 0.941176 | **-0.003676** | 1/2/269 |
| Ego is making progress | 0.948529 | 0.948529 | 0.000000 | 0/0/272 |
| Driving-direction compliance | 0.985294 | **0.988971** | +0.003676 | 1/0/271 |
| Progress along expert route | 0.819141 | **0.832731** | **+0.013590** | 144/47/81 |
| TTC within bound | 0.761029 | **0.775735** | +0.014706 | 6/2/264 |
| Speed-limit compliance | **0.974670** | 0.972686 | **-0.001984** | 22/37/213 |
| Ego is comfortable | 0.977941 | **0.981618** | +0.003676 | 1/0/271 |

hard 总分提高 0.008809，即 0.88 个百分点，相对 B10 提高约 1.25%。最稳定的改善来自路线进度：均值提高 0.013590，144 个场景改善、47 个场景退化。与此同时，责任碰撞和可行驶区域均出现净退化，因此总分提高不能解释为安全提升。

## 5. Test14-random 结果

| 指标（越大越好） | 历史 B10 | 全量后训练 | 差值（后训练-B10） | 逐场胜/负/平 |
|---|---:|---:|---:|---:|
| Overall score | 0.824270 | **0.843211** | **+0.018941** | 95/54/112 |
| No ego at-fault collision | **0.908046** | 0.898467 | **-0.009579** | 3/5/253 |
| Drivable-area compliance | 0.965517 | **0.977011** | +0.011494 | 3/0/258 |
| Ego is making progress | 0.980843 | **0.996169** | +0.015326 | 4/0/257 |
| Driving-direction compliance | 0.998084 | 0.998084 | 0.000000 | 1/1/259 |
| Progress along expert route | 0.910270 | **0.918674** | **+0.008404** | 97/39/125 |
| TTC within bound | 0.869732 | **0.873563** | +0.003831 | 4/3/254 |
| Speed-limit compliance | **0.966454** | 0.965072 | **-0.001383** | 20/40/201 |
| Ego is comfortable | 0.977011 | **0.984674** | +0.007663 | 2/0/259 |

random 总分提高 0.018941，即 1.89 个百分点，相对 B10 提高约 2.30%。可行驶区域、是否产生有效进度、路线进度和舒适性均提高；但责任碰撞下降 0.009579，且限速合规同样小幅退化。两套集合在碰撞和限速方向上给出了相同警告。

## 6. 配对分布与日志聚类区间

为避免把同一驾驶日志中的多个相关场景错误地当作完全独立样本，按 `log_name` 整组有放回重采样 100,000 次，报告场景加权均值差的 95% 百分位区间。该区间只描述已观测日志间的变异，不覆盖训练 seed、额外推理 seed 或测试集选择不确定性。

| 集合/指标 | 均值差 | 日志聚类 95% 区间 | 解释 |
|---|---:|---:|---|
| hard / Overall score | +0.008809 | [-0.011610, +0.029606] | 点估计提高，但区间跨 0 |
| random / Overall score | +0.018941 | [-0.005577, +0.048602] | 点估计提高，但区间跨 0 |
| hard / Route progress | +0.013590 | **[+0.010188, +0.017436]** | 稳定正向 |
| random / Route progress | +0.008404 | **[+0.005026, +0.012623]** | 稳定正向 |
| hard / No collision | -0.005515 | [-0.027491, +0.014652] | 点估计退化，事件稀疏、区间较宽 |
| random / No collision | -0.009579 | [-0.031120, +0.010034] | 点估计退化，事件稀疏、区间较宽 |
| hard / Speed compliance | -0.001984 | **[-0.003169, -0.000985]** | 一致的小幅退化 |
| random / Speed compliance | -0.001383 | **[-0.002383, -0.000518]** | 一致的小幅退化 |

因此，当前最扎实的正结论不是“总体闭环分数已经高置信提升”，而是：**后训练稳定提高了路线进度，并使更多发生变化的场景获得正向分数变化。** 总分均值受少量离散安全事件影响很大，尚未形成高置信的整体优势。

## 7. 安全事件交换与尾部风险

### 7.1 责任碰撞并非单向改善

| 集合 | 碰撞指标改善 | 碰撞指标退化 | 净变化 |
|---|---:|---:|---:|
| hard | 5 场（其中 1 场为 0→0.5） | 6 场（均为 1→0） | -1.5 个满分场景等价值 |
| random | 3 场（其中 1 场为 0.5→1） | 5 场（均为 1→0） | -2.5 个满分场景等价值 |

hard 中共有 7 个零分场景被恢复，但同时产生 7 个新的零分场景，零分总数仍为 60。random 中恢复 8 个零分场景、产生 4 个新零分场景，零分总数由 34 降至 30。这说明新策略不是简单地整体平移，而是在不同场景之间发生了明显的成功—失败交换。

### 7.2 新出现的责任碰撞场景

| 集合 | token | 场景类型 | B10→新模型碰撞指标 | B10→新模型总分 |
|---|---|---|---:|---:|
| hard | `295432d96052578e` | high_lateral_acceleration | 1→0 | 0→0 |
| hard | `2b34e2a1fc6b543c` | near_multiple_vehicles | 1→0 | 0.636523→0 |
| hard | `2e56997063c057a0` | stationary | 1→0 | 0.839850→0 |
| hard | `63352660f02f5e3d` | traversing_intersection | 1→0 | 0.613096→0 |
| hard | `be2efbeede795568` | stationary | 1→0 | 0.644335→0 |
| hard | `cbdc3e2a73e15687` | traversing_traffic_light_intersection | 1→0 | 0.599477→0 |
| random | `2e56997063c057a0` | stationary | 1→0 | 0.839850→0 |
| random | `78c5b673402b5861` | starting_right_turn | 1→0 | 0.687500→0 |
| random | `7f50df6eea5c54e9` | high_lateral_acceleration | 1→0 | 0.998074→0 |
| random | `b7485487d49853ca` | high_magnitude_speed | 1→0 | 0→0 |
| random | `fa10175cddfa52fa` | near_multiple_vehicles | 1→0 | 0.635357→0 |

`2e56997063c057a0` 同时属于 hard 和 random，是两套集合的重叠场景，不能重复计为两个独立失败。两套集合合计出现 10 个不同 token 的新责任碰撞退化。

### 7.3 少数灾难性场景主导负向尾部

- hard 中，最差 4 个场景占全部总分下降量的 56.15%；
- random 中，最差 4 个场景占全部总分下降量的 96.17%；
- 在碰撞、可行驶区域、进度、方向、TTC、舒适性等离散指标均未改变的场景中，总分仍小幅提高：hard 为 +0.003127，random 为 +0.001513。

这表明多数普通场景的连续性能变化较温和，最终能否安全替换 B10，主要由少量闭环灾难性事件决定。仅继续提高平均 reward 或平均 progress 不能自动消除该问题。

## 8. 与独立 validation-1000 的一致与不一致

### 一致之处

1. validation-1000 中 Reward 和 progress 提升；Test14 两套闭环集合中路线进度同样提高。
2. validation-1000 中 DA 代理总体提高；Test14-random 的官方 DA 指标提高。
3. 两种评价都确认模型已经发生可测策略变化，而不是早期短训练阶段的近似不变。

### 不一致之处

1. validation-1000 的平均 collision cost 小幅改善，但 no-collision 代理少 1/3000；Test14 的官方责任碰撞在 hard 和 random 上都出现净退化。
2. validation-1000 的 comfort cost 退化，而 Test14 官方二值舒适性提高。这两个指标定义和统计方式不同，不能相互替代。
3. validation-1000 的 FDE 尾部退化没有阻止闭环路线进度提高，说明模仿误差不是闭环任务性能的充分代理。

因此，离线 reward/安全代理适合用于训练和初筛，但不足以单独授权安全策略替换；官方闭环事件仍是最终验收的必要证据。

## 9. 当前可支持的科学结论

### 结论 A：全量后训练已经产生有效的闭环策略改进信号

路线进度在 hard 和 random 上分别提高 0.013590 和 0.008404，且日志聚类区间均为正；变化场景中正向总分变化也明显多于负向变化。由此可以排除“全量后训练仍然没有改变策略”的判断。

### 结论 B：改善集中在任务效率与部分可行性指标，不等于安全单调改进

总分、路线进度、TTC 点估计和舒适性总体改善，但责任碰撞点估计在两套数据上都下降，限速合规也一致小幅退化。当前方案实现的是**带有安全机制的 reward-driven improvement**，尚未实现可验证的 **safety-preserving improvement**。

### 结论 C：现有安全约束仍无法阻止少数闭环灾难性退化

训练期安全候选筛选和 reference preservation 能改善总体方向，但它们依赖 rollout reward 与近似安全标签，不能保证闭环执行中每个高风险场景均不退化。现有结果不能证明 reward 函数“错误”，但能够证明：**当前 reward、候选安全门和更新约束的组合尚不足以提供事件级碰撞保持保证。**

### 结论 D：当前 checkpoint 不满足替换 B10 的安全接受条件

对于以“安全约束后训练”为主题的研究，责任碰撞不能被总分和 progress 的平均增益抵消。因此 B10 继续作为已接受基线；全量后训练模型作为后续失效分析和安全机制改进的候选模型保存。

## 10. 对第四章的价值

本实验形成了一个完整且可用于论文的闭环证据链：

```text
候选支持与优势信号不足
        ↓
修复候选生成、有效组采样和 Replay 覆盖
        ↓
完成 306,801 场景 rollout 与 30,159 个有效目标组更新
        ↓
独立 validation 证明策略发生可测变化
        ↓
Test14 全量闭环证明任务进度和平均得分提高
        ↓
同时发现少数新责任碰撞与限速退化
        ↓
得到“有效改进但尚非安全单调改进”的阶段结论
```

这既是正结果，也是明确的负结果：它证明扩大训练覆盖和修复优势信号能够解决“几乎学不动”的问题，但不能单独解决“学动以后如何保证安全方向正确”的问题。该结论自然引出下一阶段的事件级安全约束、失效样本回灌和高置信策略接受机制。

## 11. 最优先的下一步

在重新训练前，优先对 10 个不同的新责任碰撞 token 及 hard 中 2 个新 DA 退化 token 做闭环失效审计：

1. 定位首次风险出现时刻、碰撞对象、相对速度、TTC 和规划轨迹变化；
2. 回查这些场景在 rollout/replay 中是否出现、是否被判为合格候选、对应 reward 与 advantage 是否方向错误；
3. 区分“训练数据未覆盖”“reward/安全标签漏检”“闭环分布偏移”和“reference 约束不足”；
4. 只针对确认的机制缺口设计修复，不直接用更长训练或简单增大 reward 权重掩盖问题；
5. 修复后先建立这 10 个 token 的零新增碰撞回归门，再进入更大集合验证。

成功标准应至少包括：新责任碰撞为 0、DA 不产生新退化、路线进度增益仍为正；否则不能宣称安全约束后训练已经完成。

## 12. 证据边界

1. 当前只有一个训练 seed 和一个闭环推理 seed；结论不覆盖 seed 稳定性。
2. hard/random 有 11 个重叠 token，不能将两者简单合并进行独立样本推断。
3. 使用的是 non-reactive agents 闭环，不代表对交互式对手行为的完整验证。
4. 总分的日志聚类区间跨 0，不能表述为“总体分数已高置信优于 B10”。
5. 历史 B10 经身份和逐场结果校验后复用，但没有在本次会话中重跑；运行耗时不作方法对比。
6. 分场景类型均值容易被单个碰撞事件主导，目前仅用于定位失效模式，不作为类别级普遍结论。

## 13. 原始证据与复现入口

- 运行脚本：`scripts/evaluate_full306801_test14_single_arm.sh`
- 总运行日志：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/evaluate_full306801_test14_single_arm.log`
- hard 原始配对摘要：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/test14-hard_historical_B10_vs_full306801.json`
- hard 摘要 SHA256：`8208b2b2109b86072a7cbaf28bd0d072e523db1d10fb2a7f928e8974dac584b7`
- random 原始配对摘要：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/test14-random_historical_B10_vs_full306801.json`
- random 摘要 SHA256：`ff1f99f92985849570690a572b35cab0c45f66445d9cbcd612d43a966b7b62af`
- hard 配对结果：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/test14-hard_paired_analysis.json`
- random 配对结果：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/test14-random_paired_analysis.json`
- 日志聚类扩展分析：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/extended_paired_analysis.json`
- 扩展分析 SHA256：`bc6a75cb5d6e2fc39dd8e968b6950312be23e4ccbbd79ca4370ec79919a6faad`
- 扩展分析脚本：`scripts/analyze_full306801_test14.py`
- 新模型 checkpoint：`/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/training_outputs/EXP_CH4_guided_safeset_full306801_seed2026_v3_1006/training_log/chapter4-safety_reference-noise0p1-std1em06-steps0-seed2026-guided-safeset88-full306801/2026-10-06-00-13-19/model_epoch_2_trainloss_0.0005.pth`
