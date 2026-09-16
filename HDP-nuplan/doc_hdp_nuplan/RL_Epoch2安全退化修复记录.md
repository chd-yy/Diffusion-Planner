# RL Epoch2 安全退化修复记录

记录时间：09月15日 20:44（北京时间）。本文时间均采用“月日 时:分”，不含年份。

**09月15日 21:14 更新后的核心判断：最优先的可确认实现问题，是安全奖励的自车几何坐标基准错误——将后轴轨迹当车身中心，并使用偏小车身尺寸。真实碰撞帧已经证实它会漏判重叠。前面排查的更新约束缺口仍需修，但不能替代这一几何修复。以下按发现时间保留完整过程。**

## 09月15日 20:44｜目标与证据

用户要求优先修复 RL Epoch2 的最大问题，并记录修复思路、过程和验证。

正式 Test14-hard 中 RL 相对 B10 有116场综合得分上升、48场下降，但均值下降0.002882。最严重的4场占全部下降量97.17%：

| token | B10 → RL 得分 | 指标退化 |
|---|---:|---|
| `2e56997063c057a0` | 0.83985 → 0 | 碰撞、TTC |
| `6eb38f317f2251f2` | 0.76594 → 0 | 可行驶区域 |
| `33187fb09d0e52f8` | 0.67545 → 0 | 碰撞 |
| `75d56d4c7b6f5013` | 1 → 0.6875 | TTC |

另外纳入 random 新增越界的 `9e507708119c5596`、`a874cefaca8e5b69`，共6个去重诊断场景。它们只用于故障诊断，不能进入训练或被当作独立模型选择验证集。hard/random 共11个重复token，533条评测记录对应522个唯一场景。

证据：`tmp/test14_full_remote_subset/three_model_eval/test14-{hard,random}_three_models.json`，以及《Test14完整评测数据准备与三模型对比操作日志》第19节。

核心目标是减少后训练新增的严重安全失败，同时保住进度。单纯扩大Buffer、增加epoch或放大reward权重，不足以保证这个目标。

## 09月15日 20:46｜建立可复查的闭环诊断

新增 `scripts/diagnose_rl_safety_regressions.py`，核验两个历史checkpoint SHA256，固定6个token，使用相同推理seed、单worker、官方nonreactive闭环，保留simulation_log逐帧轨迹；逐模型核对runner、场景集合和trace数量。新建独立输出目录，拒绝覆盖现有目录。

已启动：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/diagnose_rl_safety_regressions.py \
  --output HDP-nuplan/tmp/rl_safety_repair_0915_2048 --seeds 0
```

原始checkpoint不因代码修复自动改变。此重放用于观察历史策略的实际退化；修复后的训练配置与模型需另行验证。

## 09月15日 20:48｜已定位的更新约束缺口与修复

当前源码已有后续实验加入的过滤和正权重功能，不能把“再次开启已有功能”当成新修复。本轮定位并修改其执行漏洞：

1. **progress过滤绕过**：rollout生成了progress与safety交集，但update仅在safety开关开启时传mask。改为任一候选过滤开启即传mask。
2. **未知资格被视为安全**：旧Replay条目没有mask时原代码用全True补齐。显式过滤模式改为立即报错，只有未过滤历史模式保留兼容。
3. **安全过滤与负回归权重冲突**：新增入口、update和直接loss调用的约束，过滤模式禁止centered权重。历史未过滤运行仍可显式复现旧目标。
4. **risk合格不等于无碰撞**：新增独立 `rl_filter_collision_candidates`，把代理 `no_collision` 资格与safety mask相交，避免risk shaping的非零碰撞得分被视为合格。此项不宣称等于官方责任碰撞或保证真实闭环安全。
5. **零目标仍发生参数漂移**：全组无有效回归权重、且没有expert/reference anchor时，原实现仍执行AdamW和EMA，动量与weight decay可能移动模型。改为真正跳过optimizer/EMA；多卡时同步“任一rank有目标”的决定。保留anchor时仍正常学习。
6. **缺失reference/非有限数据**：显式reference-relative模式缺失baseline时立即报错；拒绝非有限reward、Replay轨迹、资格以及loss/梯度，避免静默异常更新。
7. **计数与调度**：分别记录抽样batch数、实际optimizer步数和跳过步数；零optimizer步的update epoch不推进scheduler。

修改位置：`hdp_nuplan/rl/safety_update.py`、`train_epoch_rl.py`、`replay_buffer.py`、`loss.py`、`train_predictor_rl.py`。

这些改动修复可确认的执行缺口；是否能减少上述真实闭环失败，需要新训练与独立验证，不能由单元测试或历史重放直接断言。

## 09月15日 21:05｜闭环重放与逐帧证据

两个历史模型各6场、各149帧均运行完整，退出码均为0。相同场景、seed=0、单worker下，B10平均分0.591913，原RL平均分0.114583。此处是特意选择的故障子集，不是Test14整体成绩，也不与原整批评测声称逐小数位一致。

新增 `scripts/analyze_rl_safety_traces.py`，从本轮可信的simulation_log重新执行官方lane-change、责任碰撞和drivable metric，逐场核对与aggregator指标一致；可行驶区域使用官方0.3米容差，而不是lane-center代理。

| 场景token | 原RL的首个关键事件（相对仿真起点） | 对照B10 |
|---|---|---|
| `2e56997063c057a0` | 11.40秒撞静止车辆，责任碰撞；自车速度0.221米/秒 | 无碰撞 |
| `33187fb09d0e52f8` | 9.20秒与行人发生前向责任碰撞；自车速度1.543米/秒 | 无碰撞 |
| `6eb38f317f2251f2` | 14.20秒首次越出官方可行驶区域容差 | 无越界 |
| `9e507708119c5596` | 12.40秒首次越界 | 无越界 |
| `a874cefaca8e5b69` | 11.50秒首次越界；12.50秒撞静止行人 | 也存在行人责任碰撞，但无越界 |
| `75d56d4c7b6f5013` | 9.20秒与车辆发生侧向碰撞；官方判定不属于自车责任，TTC=0 | 无碰撞，但本轮TTC也为0 |

证据保存于 `tmp/rl_safety_repair_0915_2048/diagnosis.json`、`trace_analysis.json`及其逐帧trace路径。不能将“非责任碰撞”写成“没有任何碰撞”，也不能把B10已有的行人碰撞算成RL新增碰撞。

这进一步限定了本次结论：严重退化既有纵向低速碰撞，也有横向越界，并非单一高速激进问题。训练reward发生在缓存起始状态、有限邻居和有限预测时域；闭环轨迹不断改变后续状态，因此代理安全资格不能替代实际执行安全。当前Dataset还只取前10个邻居future，完整地图可行驶多边形不在此NPZ奖励接口内；这些是尚存的覆盖限制，不是本轮已解决的能力。

## 09月15日 21:06｜修复验证与受控后训练

测试先通过47项，再增加真实双进程Gloo测试，覆盖“只有一张卡有目标→所有卡无目标→重新更新”的同步与参数一致性，48项通过。随后运行整个 `HDP-nuplan/tests`，72项通过，只有已有依赖弃用警告。该数字对应新增诊断gate测试之前的执行记录。

关键回归测试覆盖：progress-only/collision-only过滤能进入真实diffusion loss、缺失mask/reference拒绝、过滤与负权重冲突、非有限数据拒绝、全无资格时AdamW动量/衰减/EMA不动、仍有anchor时继续学习，以及实际更新计数。

新增 `scripts/train_rl_safety_repair.py` 作为显式修复配置入口：复制原RL args中的结构与训练配置，核验B10 SHA256及10k manifest，拒绝6个诊断token进入训练，记录完整命令和文件hash，输出目录必须不存在。旧命令的历史默认值没有暗中改变；**要使用本轮安全更新配置，应使用此新入口，不能原样重跑原RL命令并声称已修复。**

已启动的训练命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/train_rl_safety_repair.py \
  --output HDP-nuplan/tmp/rl_safety_repair_training_0915_2103
```

实验保持：B10 EMA起点、seed2026、原10k mini-train数据、group32、6步采样、noise0.1、buffer1024、2阶段epoch、500次update、batch2、lr4e-7、冻结encoder、expert anchor0.1。没有同时扩大buffer或换训练数据，以减少混杂因素。

明确变化：开启safety与独立collision候选交集；关闭centered负回归；使用已有 `positive_advantage` 模式，只拟合合格组内高于平均reward的候选。progress额外过滤和lane近似drivable门本轮不启用，reference anchor为0，reward仍为legacy。多项相关配置共同构成一个修复方案，**不能从此单次对照归因每一个开关的独立贡献**。只有组内一个合格候选或低方差组时，不构造相对优势，仍保留expert anchor。

之所以不直接把lane走廊当成硬drivable门：已有后续实验复盘指出其在路口、换道和地图边界存在误判。本轮只修复能确认的更新约束，不将近似车道范围冒充官方可行驶区域。

## 09月15日 21:06｜防止“均分提升掩盖安全失败”的诊断门

诊断脚本新增 `--models safety_repair --candidate-args ... --candidate-checkpoint ... --baseline-diagnosis ...`，允许新模型复用同seed、同token的B10重放结果，避免重复运行历史模型。

诊断门逐场比较碰撞、可行驶区域、making_progress、方向和TTC是否退化，同时要求平均score、进度和限速合规不下降（数值容差1e-6）。缺失、重复、非有限指标或仿真不完整直接拒绝；即使均分上升，也不能覆盖新增碰撞。通过只表示该诊断子集无退化，输出中始终 `promotion_authorized=false`，不会替换原模型。

正式接受新模型仍需另做未用于排错/调参的独立验证、多seed成对评测，核查新的严重失败和进度，而非只看训练loss或6个已知故障的平均分。本次不会修改历史checkpoint，也不会把本轮试验自动登记为正式Test14成绩。

## 09月15日 21:14｜更高优先级发现：后轴与车身中心混用

### 证据链

1. `data_process/ego_process.py` 的future轨迹来自 `state.rear_axle`，相对当前后轴变换；`agent_process.py` 的邻居/静态目标来自 `agent.center`。
2. 原 `rl/reward.py::_rectangle_signed_separation` 直接使用轨迹XY作为自车矩形中心，并使用 `ego_length=4.8`、`ego_width=2.0`。
3. 本地nuPlan的 `get_pacifica_parameters()` 给出长度5.176米、宽度2.297米、后轴到车身中心偏移1.461米；官方碰撞用 `CarFootprint.build_from_rear_axle` 构造车身。
4. 直行时，真实车头距离后轴4.049米，旧奖励只覆盖到2.4米，车头范围少了1.649米；与此同时，旧几何向车后多伸出了1.273米。这不是简单增大collision权重可以修复的问题。

新增 `scripts/audit_rl_collision_geometry.py`，在已保存的真实碰撞帧上用同一个目标的位置、尺寸和航向，对照旧奖励OBB、修正OBB与官方车身多边形。以下是原RL的结果，正值代表分离、负值代表重叠；数值是分离轴余量，不等于任意方向的欧氏距离。

| token | 旧几何有符号余量/米 | 修正后余量/米 | 官方判定 |
|---|---:|---:|---|
| `2e56997063c057a0` | +1.65099 | -0.00444 | 已碰撞 |
| `33187fb09d0e52f8` | +1.46860 | -0.13879 | 已碰撞 |
| `a874cefaca8e5b69` | +1.49722 | -0.15178 | 已碰撞 |
| `75d56d4c7b6f5013` | +0.13152 | -0.01187 | 已碰撞，但非自车责任 |

B10已有的 `a874...` 行人碰撞帧也会被旧几何漏判。这说明问题存在于奖励度量本身，不应归因于某一个checkpoint文件损坏。结果见 `tmp/rl_safety_repair_0915_2048/collision_geometry_audit.json`。

严格说，这验证的是“真实执行到碰撞状态时，旧几何仍漏判重叠”，**不是**已经证明这些Test14状态出现在原10k训练中，也不是证明全部越界和行为退化都由这一单因子引起。

### 实现修复

- 新增 `reward_use_nuplan_vehicle_geometry` 显式配置，通过本地devkit读取真实Pacifica尺寸及偏移，不另写一套近似常量。普通训练入口默认false，保持历史实验可复现；本轮新修复启动脚本默认开启，`--legacy-geometry` 只用于旧几何消融。
- 奖励内部统一使用 `center = rear_axle + offset * heading_direction`；动态/静态碰撞、跟车纵向距离、TTC视线以及lane包络代理使用相同车身基准。碰撞矩形函数的输入仍约定为后轴XY，避免重复偏移。
- 转弯时TTC所用车身中心速度加入偏移随航向变化产生的速度；进度、轨迹监督和舒适性仍保留原后轴轨迹含义，不能全局平移模型输出。
- 没有修改planner输出坐标、模型结构、历史checkpoint或官方评测定义。

补充说明：在原v4的TTC>=1秒配置下，已被旧几何检测到的动态重叠通常早已因TTC=0被排除；所以独立collision开关更多是防御性约束，不是此历史模型失败的充分解释。**几何本身漏判时，再多安全开关也可能在错误输入上给出“合格”判断。**

### 验证与实验分层

新增5项车辆几何测试，包括：车头行人碰撞漏判复现、车后误判消除、200组随机平移/旋转/尺寸的OBB与官方多边形碰撞一致性、动态risk/跟车坐标一致性、静态碰撞一致性。全套测试此时84项通过。

第一轮 `tmp/rl_safety_repair_training_0915_2103` 已于21:11完成，保留为“仅修更新约束、旧几何”的消融，不把它称为完整修复模型。其评测输出为 `tmp/rl_safety_constraint_only_eval_0915_2112`。

第二轮包含几何修复的训练已启动，仍从同一个B10重新开始，不接第一轮权重：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/train_rl_safety_repair.py \
  --output HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112
```

此时该脚本默认开启修正几何；除几何开关之外，使用相同10k数据、采样及更新配置。第一轮启动早于几何缺陷确认，所以其原始命令没有此开关；今后重现第一轮需显式附加 `--legacy-geometry`。新一轮必须完成训练和闭环检查，才能报告行为层面是否改善。

## 09月15日 21:17｜对照实验有效性与第一轮训练结果

第一轮旧几何rollout的全部共同统计项与历史日志逐值一致，包括reward均值5.613861610594019、risk均值0.5438065205355175、无碰撞比例0.97725、安全候选比例0.554209375、有安全候选的组比例0.5565。新增的drivable代理诊断项不属于历史日志。这为相同B10起点、数据顺序和候选生成提供了额外证据，而不只依赖命令行看起来相同。

第一轮update共500个batch、500次实际更新，loss为0.0010041464；44.6%的抽样场景没有合格候选，53.9%的组形成有效相对优势。虽然部分batch没有rollout回归目标，expert anchor=0.1仍使其正常更新；因此本次真实训练的skipped_update_steps=0符合设计，不能误解为“跳过机制没有生效”。零anchor时的真正跳过由回归测试验证。

历史loss0.0003094包含可为负的centered回归项，本轮改成非负加权，**loss变大不意味着驾驶变差，变小也不构成安全改善证据**。两者需要各自看闭环行为。

注意旧reward日志中的 `drivable_area_gate_active=1` 并不单独证明该门已开启，应检查实际 `reward_safety_gate_require_drivable_area=false` 参数；本轮没有依赖这项旧命名指标做准入判断。

## 09月15日 21:20｜仅修更新约束的闭环消融结果

| 指标（6场均值） | 本轮B10对照 | 原RL Epoch2 | 更新约束修复、旧几何 |
|---|---:|---:|---:|
| 综合score | 0.591913 | 0.114583 | 0.641437 |
| 无责任碰撞 | 5/6 | 3/6 | 5/6 |
| 可行驶区域合规 | 6/6 | 3/6 | 6/6 |
| TTC合规 | 2/6 | 1/6 | 3/6 |
| 路线进度 | 0.768616 | 0.786639 | 0.754440 |
| 限速合规 | 0.914646 | 0.905381 | 0.924062 |

原RL新增的 `2e569...`、`33187...` 两处责任碰撞，以及三个越界，在本轮约束修复消融中不再出现；`a874...` 原来B10就有的责任碰撞仍存在。相对B10，score提高0.049524、TTC改善1场、无新增上述硬安全退化，但进度下降0.014176，所以预先定义的严格诊断门仍判为失败，未授权替换正式模型。

重要复现边界：正式整批Test14中 `75d56...` 的B10 TTC=1，本次6场单worker重放中B10 TTC=0。随机采样、场景分组/随机数消耗与整批运行不完全等价，不能称为所有原始指标逐位复现。此次实验均使用本轮固定6场、相同seed与执行配置的B10作配对对照；发现这项差异后没有改seed或重挑子集。

第一轮评测与第二轮训练曾同时运行，运行耗时受资源竞争影响，不用来比较模型推理效率。几何修复版的闭环结果尚待完成，不能把第一轮改善归功于当时尚未启用的几何修复。

## 09月15日 21:24｜几何修复版训练完成，进入闭环验证

第二轮于21:23正常退出，500个抽样batch、500次实际更新，loss=0.000948204，无非有限loss/梯度异常。保存的checkpoint为 `tmp/rl_safety_geometry_training_0915_2112/training_log/hdp-rl-safety-repair-controlled/` 下该次时间目录的 `model_epoch_2_trainloss_0.0009.pth`；精确路径与原始命令见该训练根目录的 `run.json`。

几何修正后的10k rollout安全候选比例0.546656，有合格候选的组比例0.5496。`no_collision`代理比例从旧几何0.97725变为0.991128，同时前向真实碰撞漏判被修正；这并不矛盾，因为原几何还会产生车后误报。**这是同批候选的计分几何变化，不是rollout阶段模型已经变强。**

本轮update中有效组比例0.543、无资格组比例0.443，expert anchor仍为0.1。进度、route、comfort等不依赖此次几何变换的rollout统计与第一轮相同。

已启动完整6场闭环验证，输出目录 `tmp/rl_safety_geometry_eval_0915_2123`，复用 `tmp/rl_safety_repair_0915_2048/diagnosis.json` 中同seed的B10对照。模型EMA、参数文件hash、完整仿真命令和安全gate结果均由诊断脚本写入新目录的 `diagnosis.json`。待评测结束后再决定是否通过诊断，当前不提升为正式模型。

## 09月15日 22:11｜几何修复版闭环结果与最终判定

6场仿真全部成功、trace完整，使用与B10相同的seed=0、token集合和单worker配置。几何修复版指标如下：

| 指标（6场均值） | B10对照 | 原RL Epoch2 | 更新约束修复、旧几何 | 更新约束+车辆几何修复 |
|---|---:|---:|---:|---:|
| 综合score | 0.591913 | 0.114583 | 0.641437 | **0.694002** |
| 无责任碰撞 | 5/6 | 3/6 | 5/6 | **5/6** |
| 可行驶区域合规 | 6/6 | 3/6 | 6/6 | **6/6** |
| TTC合规 | 2/6 | 1/6 | 3/6 | **4/6** |
| 路线进度 | 0.768616 | 0.786639 | 0.754440 | 0.756441 |
| 限速合规 | 0.914646 | 0.905381 | 0.924062 | **0.923438** |

相对同seed B10，几何修复版score +0.102089、TTC +0.333333、限速 +0.008792；碰撞和可行驶区域没有新增退化。原RL新增的两处责任碰撞（`2e569...`、`33187...`）和三个越界（`6eb38...`、`9e507...`、`a874...`）均未在修复版重放中重现；B10原有的 `a874...` 行人责任碰撞仍然存在，因此修复没有宣称“零碰撞”。`75d56...` 的TTC在本轮恢复为1，但该场景仍需注意官方责任分类与TTC指标并非同一件事。

严格诊断gate结果：`diagnostic_gate_passed=false`，原因不是安全指标回退，而是相对B10的平均路线进度 `-0.012175`，超过当前零回退阈值；`promotion_authorized=false`。因此本轮可以确认“最大安全退化显著缓解”，不能确认“已经达到正式模型替换标准”。

几何修复版checkpoint SHA256：

```text
3a2348a12eb88f32018921514ffd4bce46720392370af9928aa0392ac4e56948
```

关键输出：

- `tmp/rl_safety_geometry_training_0915_2112/run.json`
- `tmp/rl_safety_geometry_eval_0915_2123/diagnosis.json`
- `tmp/rl_safety_repair_0915_2048/trace_analysis.json`
- `tmp/rl_safety_repair_0915_2048/collision_geometry_audit.json`

### 当前修复结论

本轮最值得保留的代码修复是：

1. 用 nuPlan Pacifica真实车身尺寸和“后轴→车身中心”偏移统一奖励几何，消除车头漏判与车后误判；
2. 过滤模式禁止负 centered回归权重，保证不安全候选不会靠负权重反向影响策略；
3. 过滤资格、collision资格、缺失mask、非有限数据和零目标optimizer/EMA更新全部采用 fail-closed 约束；
4. 通过逐场官方指标gate，避免均分提升掩盖碰撞或越界新增。

下一优先级应是处理安全改善带来的进度损失：在独立未调参验证集上进行至少多seed成对评测，并分别消融几何修复、collision过滤、正权重和expert anchor；不应直接放宽gate或重新打开centered负权重来换取进度。当前正式Test14三模型对比和历史checkpoint均未被覆盖。

## 09月16日 09:57｜按优先级启动完整Test14确认评测

先验证完整分布，再决定是否修改进度目标。新增 `scripts/evaluate_rl_safety_geometry_test14_seed0.sh` 与 `scripts/assemble_test14_safety_geometry_comparison.py`，对原Test14-hard 272场和Test14-random 261场运行修复版checkpoint，并与历史评测中相同benchmark、seed=0的B10逐场配对。

两个集合共有11个重复token，所以这里严格报告为533条benchmark记录、522个唯一场景；不会错误称作533个独立场景。由于6个故障token曾参与本轮诊断，这次属于“完整分布确认评测”，其中其余场景是未用于本轮修复选择的样本，但整个Test14不能再称为完全未触碰的最终盲测集。

执行策略：hard与random使用两个独立Python进程并行，每个进程内部保持单worker、单指标引擎，避免曾发现的共享stateful MetricsEngine线程污染。每40场一个chunk，逐chunk核对runner与aggregator token、失败数和重复项，支持从已验证chunk断点继续；禁用simulation log以限制磁盘占用。两个进程共享GPU只影响耗时，不共享评测指标对象。

启动命令：

```bash
bash HDP-nuplan/scripts/evaluate_rl_safety_geometry_test14_seed0.sh
```

输出根目录：`tmp/rl_safety_geometry_test14_seed0`。完成seed=0全量筛查后，才根据安全硬指标与进度结果决定多seed复核及reference进度约束；不会在完整结果返回前继续调参。

## 09月16日 12:26｜发现评测中断，保留已验证分块恢复

12:24 定时检查发现 launcher 与两个 `run_simulation.py` 均已退出。hard 日志最后更新为 11:27:53，停在 chunk 2 的 18/40；random 最后更新为 11:27:38，停在 chunk 3 的 7/40。没有最终 full_comparison，不能把日志中的 Starting 当成进程仍存活。

退出原因暂未确定：末尾没有仿真异常、CUDA OOM 或正常结束记录；可见 kernel journal 在对应时段无记录，原工具会话也已无法取回退出码。日志前面存在的 FileNotFoundError 来自脚本探测尚未完成的 run/chunk，随后均正常启动仿真，不是这次中断的直接证据。不能据此虚构为模型崩溃或系统 OOM。

恢复前独立读取归档 parquet，逐个核验 runner 与 aggregator 的场景集合和行数：

| benchmark | 已验证 chunk（从 0 编号） | 完整场景记录 | runner 失败数 |
|---|---|---:|---:|
| test14-hard | 0、1 | 80 | 0 |
| test14-random | 0、1、2 | 120 | 0 |

这里的“失败数 0”仅指程序执行成功，不等于无碰撞或其他驾驶指标全部通过。中断 chunk 的局部进度没有完整 runner/aggregator 归档，因此不作为有效完整 chunk 合并；重跑整个中断 chunk，保持原场景次序与 seed=0 协议。

12:26 使用现有可恢复脚本，以独立会话、忽略终端挂断方式重新启动；没有修改模型、reward、安全 gate 或评测 seed：

```bash
nohup setsid bash /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/evaluate_rl_safety_geometry_test14_seed0.sh \
  >> /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_test14_seed0/recovery_0916_1226.launcher.log \
  2>&1 < /dev/null &
```

恢复日志确认保留 hard 0/1、random 0/1/2，从 hard 2、random 3 继续；恢复前后 10 个已验证归档文件（5 份 runner、5 份 aggregator）的 SHA256 全部一致。已确认 launcher PID 157834 及两个新仿真进程存活；独立会话只是降低终端挂断影响，不代表已经查明旧进程退出原因。

同时纠正《HDP RL 后训练流程与 Reward 设计全解》中 11:43 仅依据日志误判“仍在运行”的表述。今后健康检查必须同时核对进程存活与日志推进。完整 seed=0 结果仍未齐，不提前启动多 seed 或实现 B10 progress 约束。

## 09月16日 14:55｜完整 Test14 seed=0 结果：未通过

恢复后的评测于 14:30 完成。重新调用管理器验证两个完整 run：hard 272/272、random 261/261，runner 与 aggregator 场景集合逐一一致，两个模型的失败仿真数均为0，无重复场景记录。最终成对结果保存于 `tmp/rl_safety_geometry_test14_seed0/full_comparison.json`，SHA256 为 `6eaec01bb7b7333678dd975fc3e7c91c0f4364f2aac7e77997fdfdb9d01d3967`。

| benchmark | B10 score | 修复版 score | score差 | B10进度 | 修复版进度 | 进度差 | 诊断gate |
|---|---:|---:|---:|---:|---:|---:|---|
| hard 272 | 0.705304 | 0.704216 | -0.001088 | 0.819141 | 0.807828 | -0.011313 | 未通过 |
| random 261 | 0.824270 | 0.830803 | +0.006533 | 0.910270 | 0.902460 | -0.007810 | 未通过 |

random 上责任碰撞均值提高0.011494、TTC提高0.003831、可行驶区域提高0.003831，且没有逐场新增五项硬安全退化；但进度下降，因此仍不能通过。hard 上 TTC均值提高0.011029、可行驶区域提高0.003676，但责任碰撞均值下降0.003676，并出现3个明确的逐场安全退化：

| token | 新退化 | B10 → 修复版 score | B10 → 修复版进度 |
|---|---|---:|---:|
| `40e5fc0034035139` | 责任碰撞 1→0 | 0.616020→0 | 0.771263→0.746460 |
| `63352660f02f5e3d` | 责任碰撞 1→0 | 0.613096→0 | 0.761908→0.749146 |
| `f7700356e6d85410` | TTC 1→0 | 0.959379→0.640214 | 0.870011→0.848684 |

进度损失不是少数极端场景的偶然平均：hard 中191场下降、76场不变、5场上升；random 中140场下降、120场不变、1场上升（阈值 `1e-6`）。这表明当前正优势、安全过滤、专家anchor和几何修复的组合虽修复了已知故障子集并改善部分总体安全指标，仍把策略系统性推向较保守进度，而且没有消除全部新碰撞。

因此当前checkpoint明确不具备推广资格，也不应因random平均分提高而覆盖B10。完整Test14包含6个曾用于诊断的token，所以结论仍是完整分布确认，不是最终盲测；但新增hard退化token不属于那6个原始诊断token，必须严肃处理。

## 09月16日 14:55｜下一优先级：固定模型的完整多seed成对复核

在改训练目标前，先确认上述碰撞和系统性进度损失是否随推理随机噪声稳定存在。新增 `scripts/evaluate_rl_safety_geometry_multiseed.py`，仅评测固定的B10与几何修复版，不训练模型，预注册 seed=1、2；每个seed均覆盖 hard 272和random 261，并对相同benchmark/seed的两模型逐场成对比较。

协议约束：

1. 每个仿真进程单worker、独立指标引擎；最多并行两个进程；
2. 固定原40场chunk边界、manifest顺序、两个checkpoint SHA256和模型args；
3. 输出目录独立于seed=0，已有完整归档只读保留；部分/损坏归档拒绝静默覆盖；
4. 每个chunk前复核模型、当前HDP源码、nuPlan devkit、配置快照与manifest hash，防止长评测中途代码变化造成混合结果；
5. 每个chunk验证runner成功、aggregator覆盖后才归档，支持安全断点恢复；
6. gate继续要求无逐场新增责任碰撞、可行驶区域、making progress、方向和TTC退化，且score、路线进度、限速均值不下降；不因均值提升放宽硬安全gate。

新增 `tests/test_rl_multiseed_evaluation.py`，覆盖完整272/261协议、重复/乱序/错误chunk拒绝、相同seed与单worker命令、只读完整归档、部分归档拒绝、失败runner拒绝和源码漂移拒绝。连同RL、安全诊断、汇总、几何及学习率测试共 **71 passed**，仅有已有timm弃用警告。

只有seed=1、2结果确认进度损失稳定存在后，才进入冻结B10参考的逐场安全优先、进度不退化约束实现；当前不改reward、不重训、不放宽安全gate。

14:56 已启动：

```bash
nohup setsid /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/evaluate_rl_safety_geometry_multiseed.py \
  --output /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_multiseed_0916_1455 \
  --seeds 1 2 \
  > /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_multiseed_0916_1455/launcher.log 2>&1 < /dev/null &
```

启动后主进程PID 170041，B10与修复版均从 `test14-hard` seed=1 chunk 0 开始，命令中的seed、场景filter、单worker和checkpoint身份符合协议。`protocol.json` 已固定966个输入文件hash；后续PID可能随chunk变化，不作为模型身份依据。定时巡检已切换到该输出目录，并被要求在评测期间不修改哈希锁定源码。
