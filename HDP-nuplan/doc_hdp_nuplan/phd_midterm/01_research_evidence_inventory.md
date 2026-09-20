# 01｜当前研究事实审计与 Evidence Inventory

审计时间：09月16日 22:46（北京时间，不含年份）。研究快照：codex/phd-thesis-research，HEAD=d52d49c。

## 1. 审计结论先行

已经建立了可以支撑博士中期检查的实质性研究基础：完整 mini-train 数据体系、监督扩散规划模型、奖励加权后训练、完整 Test14 闭环比较、逐帧失效诊断和安全修复。但现有材料尚不能证明已经形成三项独立、经过充分验证的原创算法贡献。

目前最强的算法性能证据是 B10 在两个完整 Test14 集合上优于本地重训 DP；最强的机制证据是奖励几何存在可复现的坐标基准错误；最重要的负结果是修复并未带来跨集合、兼顾安全和进度的稳定改进。

**本次新增的关键审计发现：所谓“对齐原始 DP”并未与 B10 对齐实际学习率。** 因而不能将模型间差距解释为 hybrid loss、增量表示或某个结构模块的独立贡献。详见第 5 节。

## 2. 分析计划、材料清单及读取边界

本轮按以下顺序实施，前两步完成后才创建本文件：

1. 固定分支、commit、未提交改动及模型身份；检查目录、历史、训练和评测产物。
2. 全文阅读 16 个文献 Markdown；沿数据→表示→监督→采样→reward→Replay→更新→推理→闭环汇总核对主要代码，并用实际 args、日志、TensorBoard、checkpoint 校验。
3. 建立本事实清单；再生成文献矩阵、缺口与负结果、三套候选技术路线。
4. 仅交付 01～04，停止；不提前编写 05～10，不启动训练/闭环测评。

### 2.1 实际材料

| 材料 | 实际找到的内容 | 本次读取/核验方式及边界 |
|---|---|---|
| 主代码 | [HDP-nuplan][ROOT]，包括 model/module、diffusion_utils、loss、rl、data_process、utils、planner、config、scripts、tests | 核心模型、监督和 RL 调用链按执行代码阅读；数据、评测及相关测试交叉检查。不是仅阅读 README；也不声称逐行阅读所有第三方求解器和重复注释 |
| 原始基线 | [diffusion_planner][DP]、根目录 train_predictor.py、normalization.json | 比较表示、Decoder、损失、guidance、训练调度；Encoder/Mixer/augmentation 经统一包名前缀后 AST 相同 |
| 旁支材料 | HDP-navsim、RiskDiffuser-main、Experiment_Preparation | HDP-navsim 是参考实现，README 中 PDMS 不算本地成果；RiskDiffuser-main 当前只见 checkpoints 子目录，不是已落地的风险条件训练工程；未发现可证明本地 AL-iLQR 实验完成的证据 |
| 研究文档 | doc_hdp_nuplan 中 16 个 Markdown；另有 docs/mini_supervised_rl_operation_log.md | 精读 B10、RL Epoch2、Test14、修复记录、同数据对照；核查 reward 总解与历史操作记录中的相关论断。操作日志不是高于源数据的证据 |
| 文献 | [带前导空格的实际目录][LIT]中全部 16 个 Markdown | **16/16 全文阅读完成**，不是关键词片段；逐篇清单、行数、指纹见 02。它们是中文解读笔记，不是原论文全文；不能宣称已核验原论文全部实验 |
| 数据 | 306,801 全量 mini-train NPZ、10,000 balanced_logs 子集、217 个 Test14 DB | 阅读 manifest/validation/coverage 报告并核查数量；没有重新扫描或反序列化所有大文件 |
| 模型 | 本地重训 DP、B10、旧 RL Epoch2、几何修复版 | 四个实际 checkpoint 重新计算 SHA256；读取 DP/B10 checkpoint 的 epoch、loss、optimizer、scheduler；核对训练 args |
| 实验 | 三模型完整 Test14、修复版完整 seed0、6 场诊断、fixed200、A/B 与 B10/B20、小规模旧实验 | 读取原始 JSON 的逐场景和均值结构；正式结论优先采用完整单 worker Test14；旧并行结果降级为历史探索 |
| 回归测试 | HDP-nuplan/tests 14 个测试文件 | 本轮 CPU 执行全部：93 passed，18 warnings，4.43 秒；这不是 93 次驾驶实验 |
| Git | 7571a3b→7200254→b13841b→a38e5b0→4bf2839→d08c0d7→d52d49c | 用于实现演化与历史定位，提交作者/新增行数不能证明思想原创 |

工作区原有两处代码改动是注释变更：dpm_solver_pytorch.py、train_predictor.py；文献目录原为 untracked。本轮保留它们，没有改动研究代码、历史指标或 checkpoint。当前闭环评测继续保持暂停；seed1 状态文件里的 running 是中断前的持久化记录，不是完成证据。

### 2.2 证据等级和状态

- A：代码、可复用的运行协议/模型/数据身份和定量实验产物齐备。这里是归档可复查/可重跑，不承诺从头训练位级一致，也不自动证明显著性或创新性。
- B：代码加初步实验，样本规模、对照或历史评测条件尚不充分。
- C：实现存在，但缺少系统实验。
- D：仅有规划或本轮提出的假设。

“已完成且有充分实验”只针对该行的窄任务，例如完整评测工程已完成；不意味着整项科学问题已解决。completed / implemented / preliminary / proposed 分别对应已经完成的限定事实、实现、初步证据、拟议工作。

## 3. 科学研究相关工作清单

| 工作项 | 科学问题 | 具体方法 | 对应代码路径 | 是否已有实验 | 实验数据/指标 | 完成度 | 证据强度 | 是否可能成为博士论文贡献 | 缺失内容 |
|---|---|---|---|---|---|---|---|---|---|
| S01 监督 HDP 基础模型 | 如何由场景生成自车轨迹 | 时间 token Decoder；增量 XY 与位置重建混合监督；DP Encoder warm-start | [模型][C01]、[监督损失][C02]、[训练入口][C03] | 是 | 306,801 样本；B10 loss=0.00906237；完整 hard/random score=0.705304/0.824270 | 已实现但实验不足（原创归因不足） | A：模型运行证据；C：单模块贡献 | 可作为基础模型与方法载体；不能把文献 L02 的 hybrid 当原创 | 公平基线、表示/损失分离消融、训练重复、生成轨迹质量 |
| S02 多种 diffusion 预测/监督空间 | 哪种监督更适合累积轨迹误差 | SDE 在 x_start/noise/score/v 间变换 | [SDE][C04]、[损失][C02] | 正式模型只确认 x_start→x_start | 未见本地完整预测×监督交叉矩阵 | 已实现但实验不足 | C | 单纯比较已被 L02 做过；需新机制才可成为贡献 | 控制表示、归一化和 loss 尺度的机制实验 |
| S03 增量积分与 detach | 长时域梯度和位置误差如何耦合 | 增量积分、可选窗口截断梯度 | [轨迹工具][C05] | B10/修复 RL 都有训练，但 detach=0 | hybrid=0.01，无独立有/无 hybrid、detach 消融 | 已实现但实验不足 | B（组合）；C（独立机制） | 目前属于 L02 方法迁移 | 物理速度与位移区分、长时域误差、动力学一致性、梯度分析 |
| S04 reward-weighted 后训练 | 监督分布能否按驾驶目标改进 | G=32 候选、代理 reward、组优势、回归自蒸馏、专家 anchor | [RL 入口][C06]、[RL loss][C07]、[循环][C08] | 是 | 旧 Epoch2 hard Δscore=-0.002882；random +0.006876 | 已实现但实验不足 | A：运行与负结果；非稳定增益证明 | 可以形成问题发现与改进研究基础；基本范式来自 L02 | 真正闭环目标关联、归因消融、重复训练、独立最终测试 |
| S05 多目标代理奖励 | 局部候选分数是否预测闭环质量 | risk/follow/lane/progress guard，TTC/THW/静态 OCC | [reward][C09] | 是 | 当前四项权重 1/3/2.5/5；已发现几何漏判和闭环退化 | 已实现但实验不足 | B | 若进一步形成可检验的指标一致性方法，可能；手工拼 reward 不自动创新 | 与官方指标的混淆矩阵、排序一致性、未知状态处理、部署信息限制 |
| S06 自车几何校正 | 奖励是否在正确坐标基准上计算碰撞 | 后轴→车身中心；Pacifica 尺寸；共用 OBB 几何 | [reward][C09]、[几何测试][C10] | 是，真实碰撞帧+测试+修复训练 | 见第 7 节：旧正净空→修复负净空 | 已完成且有充分实验（几何错误这一窄问题） | A | 当前是必要纠错/机制案例，不是独立理论创新 | 不能由这些帧断定旧训练退化全由此造成；需系统误判率和因果对照 |
| S07 安全资格进入训练目标 | 排名惩罚能否阻止危险样本被拟合 | 显式 mask、非负权重、缺失资格拒绝、无目标跳过 optimizer/EMA | [safety_update][C11]、[Replay][C12]、[RL loss][C07] | 是 | 更新约束版6场 score=0.641437；几何版0.694002；完整结果仍未过诊断门 | 已实现但实验不足 | B | 可支撑安全约束后训练研究，但开关/修 bug 不足以独立成章 | 因子消融、所有候选不合格时的策略、进度保持、训练分布覆盖 |
| S08 冻结参考策略约束 | 优于组均值是否等于优于 B10 | reference reward baseline、同噪声输出 anchor | [RL loss][C07]、[RL 入口][C06] | 有历史工具/探索记录；当前修复版未启用 | relative=false，reference weight=0 | 已实现但实验不足 | C（当前贡献证据） | 有潜力，但“加 reference”本身很常规 | 确认历史产物身份，独立对照和漂移—性能关系，不可把输出 MSE 当 KL 保证 |
| S09 DA gate / aligned reward v2/v3 | 更接近评测公式是否更可靠 | lane corridor DA proxy、几何乘性综合目标 | [reward][C09] | 有 sanity 文件，不是完整有效性验证 | 当前 geometry run 仍是 legacy，DA gate=false | 已实现但实验不足 | C | 仅命名 nuplan_aligned 不证明对齐；需新证据 | 真正道路多边形、边界/交叉口鲁棒性、统计校准、独立闭环 |
| S10 候选增广 | 候选缺乏安全解时如何探索 | 整条轨迹共享纵/横偏移 | [augmentation][C13] | 当前运行关闭 | std=0，epochs=0 | 原型阶段 | C | 简单扰动已有文献，不宜直接宣称贡献 | 安全候选覆盖、模式多样性、动力学可行性及等算力比较 |
| S11 优化/推理 guidance | 训练后残留危险是否需执行期校正 | 原始 DP 有碰撞 energy guidance；HDP 正式推理没有接入 | [DP guidance][C14]、[HDP planner][C15] | 无本地 HDP guidance 系统结果 | 当前 HDP 采样未启用 guidance | 原型阶段（外部参考）；HDP 方案仅有设想 | C / D | 可能是后续条件分支，但 L09/L11/L12/L14 已有充分先例 | 可部署邻车预测、可微 cost、约束可行性、延迟和 correction displacement |

## 4. 工程研究基础清单

| 工作项 | 科学问题/研究支撑 | 具体方法 | 对应代码路径 | 是否已有实验 | 实验数据/指标 | 完成度 | 证据强度 | 是否可能成为博士论文贡献 | 缺失内容 |
|---|---|---|---|---|---|---|---|---|---|
| E01 完整 mini-train 预处理 | 保证训练材料可追溯 | 分片、manifest、去重、校验、恢复 | [data_process][C16]、[缓存报告][E01] | 是 | 44 logs、306,801 唯一 NPZ、48,316,775,683 bytes | 已完成且有充分实验（数据工程） | A | 平台建设，不自动理论创新 | 按 log/时间相关性说明有效样本量，不把 NPZ 当独立驾驶案例 |
| E02 balanced_logs 子集 | 小规模实验覆盖更多日志 | 44 log 基础配额+补齐 | [sampling][C17]、[子集报告][E02] | 是 | 10,000 NPZ，每 log 227～229 | 已完成且有充分实验（抽样实现） | A | 属于实验支撑；**不是 scenario-category balanced** | 类型分布、对照抽样及实际 replay 覆盖 |
| E03 warm-start/冻结与 A/B | 有限数据训练稳定性 | 151/151 Encoder tensors，前三轮冻结；Epoch4 后分叉 | [训练入口][C03]、[历史详解][D01] | 是，小样本 | A/B 20场 score 0.821742/0.876941 | 已实现但实验不足 | B | 初始化和 LR 是基础协议，不宜独立包装创新 | 旧并行评测复核、真实 LR、训练多 seed |
| E04 checkpoint 选择 | loss 排名能否代表驾驶质量 | 验证 loss 排名+闭环比较 | [evaluate_checkpoints][C18]、[B10/B20][E03] | 是 | 20场 B10/B20 score=0.876941/0.810012 | 已实现但实验不足 | B | 可支持“loss 非闭环质量”问题；门禁工程本身不是算法贡献 | 独立 selection set；正式 open-loop ADE/FDE；更多点的相关分析 |
| E05 完整 Test14 数据准备 | 有限磁盘获取真实 benchmark | 远程 ZIP Range、索引、token 对齐、CRC/SQLite 门禁 | [准备脚本][C19]、[coverage][E04] | 是 | 1,349 DB扫描；保留217 DB；272 hard+261 random | 已完成且有充分实验（平台） | A | 可靠评测平台和工作量 | 533条 benchmark记录仅522唯一token；不是全量 nuPlan test 所有场景 |
| E06 正式闭环评测 | 可比较的模型效果 | 单 worker、固定分片、runner/metric 双验证、hash | [评测脚本][C20]、[hard][E05]、[random][E06] | 是 | 三模型1,599次仿真成功、失败0 | 已完成且有充分实验（指定协议） | A | 实验平台，不是模型原创 | 固定seed结果仍缺重复和区间；zero failed simulations 不等于 zero collisions |
| E07 逐帧失败诊断 | 平均得分掩盖了什么 | 6个预先暴露退化案例、trace、官方碰撞分类 | [诊断脚本][C21]、[trace][E07] | 是 | 149帧/场；几何audit含5条事件 | 已完成且有充分实验（定点诊断） | A | 有价值的机制证据，不是独立泛化试验 | 未选择案例上的检验；候选轨迹与执行轨迹分开 |
| E08 成对多seed协议 | 改变是否稳定 | 固定token/顺序/分片，分别跑B10与修复模型 | [multiseed][C22]、[protocol][E08] | 未完整完成 | seed1仅部分分片，seed2未形成完整汇总 | 已实现但实验不足 | C | 稳健实验设计基础 | 用户允许恢复后完成；这是推理seed重复，不替代训练seed重复 |
| E09 测试与可复现性约束 | 防止静默失效 | 资格、几何、训练、数据、汇总测试 | [tests][C23] | 本轮验证 | 93 passed/18 warnings | 已完成且有充分实验（当前单测） | A | 质量保障，不是驾驶性能成果 | 动态场景、分布外、GPU/不同环境验证 |

## 5. 必须纠正的“公平基线”表述

### 5.1 直接证据

读取 DP 的 [TensorBoard 目录][E09]，lr/lr 在 epoch 对应值如下；B 的 [TensorBoard][E10] 与 checkpoint 一致：

| 有效 epoch | HDP B10 | 现有 aligned-original-DP | 证据性质 |
|---|---:|---:|---|
| 1 | 5e-5 | 5e-5 | B共同阶段历史元数据；DP scalar |
| 2～4 | 5e-5 | 5e-4 | B经历中断恢复；DP实际scalar，不是命令行值 |
| 5～10 | 5e-5 | 5e-6 | 两端TB；两个Epoch10 checkpoint optimizer再次确认 |

DP checkpoint 的 schedule 包含 LinearLR(total_iters=0, start_factor=0.1, base_lrs=[5e-5], _last_lr=[5e-6])。原始 [DP scheduler][C24] 未处理 warm_up_epoch<=1；[HDP scheduler][C25] 有单独分支保持 LR 不变。根目录 DP 的现有实现仍存在这个差异。

因此：[同数据对照日志][D02]中“阶段2实际5e-5”“学习率过程一致”的描述不成立。这里只记录审计纠正，不修改旧日志或代码，不启动重训。

### 5.2 可以和不可以得出的结论

- 可以：在已归档训练轨迹下，B10 在本地两个完整 Test14 的 score、安全、舒适指标较该 DP 更好。
- 不可以：控制了所有训练变量；仅 hybrid loss 导致约0.10～0.13的score优势；证明 HDP 方法普遍优于原始 DP。
- 即使修好 LR，仍有 Decoder 结构、token 方式、ego-only vs joint prediction、状态归一化与初始化等变化。需要同架构消融而不只是系统对照。
- 官方发布 checkpoint 是额外数据/预训练资源下的另一参照，不可与同数据重训结果混称“DP baseline”。fixed200 旧记录还有线程安全风险，不能作为正式超越官方模型的证据。

## 6. 当前模型到底实现了什么

### 6.1 监督模型

输出是 [Δx, Δy, cosθ, sinθ]，未来80点，dt=0.1s。XY 是逐帧位移，不是除以 dt 后的物理速度，也不是加速度/转角控制。diffusion-v 是加噪参数化术语，与上述物理速度不同。

监督主式为增量空间误差加0.01位置误差，位置由累计求和恢复；B10 detach=0。积分是表示一致性，不意味着满足非完整运动学、曲率/加速度约束。RL waypoint 项和监督 waypoint 项的维度归约需分开报告，不能只见同一个0.01就认定有效尺度完全相等。[C02][C05][C07]

HDP 没有邻车预测监督头；训练 reward 用最多10个缓存邻车未来，不能等同完整环境的所有参与者。正常部署从模型生成轨迹交给 NuPlan 跟踪，当前没有额外安全投影、iLQR 或 reward Best-of-N。[C08][C15]

### 6.2 当前后训练配置，不是 CLI 默认值

实际使用 [几何修复 args][E11]：从B10 EMA加载，冻结Encoder；2个epoch=1次rollout+1次update；G32、6采样步、噪声0.1；batch2，lr4e-7；500次实际更新；expert anchor0.1；EMA update rate0.05。

| 功能 | 修复版真实状态 | 解释 |
|---|---|---|
| risk/follow/lane/progress guard | 开，权重1/3/2.5/5 | 当前 reward 基础四项 |
| safety gate | 开，risk>=0.3、TTC>=1s | 代理资格，不是官方闭环保证 |
| safety/collision候选过滤 | 开/开 | mask从rollout到loss，独立碰撞排除 |
| positive_advantage | 开 | 只拟合高于合格组均值者；不是保证高于B10 |
| centered w−1 | 关 | 历史RL开，新模型禁止过滤与负权同时用 |
| 真实Pacifica几何 | 开 | L5.176m、W2.297m、后轴到中心1.461m |
| expert anchor | 开，0.1 | 与 reward imitation_weight=0不矛盾 |
| reference anchor / relative reward | 关/关 | 代码存在，不属于该checkpoint训练 |
| progress候选过滤 / DA gate | 关/关 | guard奖励存在不等于progress硬门开 |
| nuplan_aligned / proxy_v2 / proxy_v3 | 关 | 当前 objective_mode=legacy |
| trajectory augmentation / detach | 关/关 | std0；detach_window0 |
| 独立progress/collision/route/comfort/backward加权和 | 未作为当前总reward执行 | args中有参数不等于生效；诊断量仍可被计算；follow含局部纵向舒适项 |
| reward temperature | positive_advantage分支不使用 | 不能在当前模式下把temperature消融当有效算法变量 |

Replay是FIFO deque(maxlen=1024)，不是长期经验库或终身学习记忆。10k×32=320,000条候选被评分，最终仅保留1,024组；500×2=1,000次有放回组抽样。唯一更新场景数当时未记录；均匀抽样期望约638只是数学推算，不是实验观测。[C12][E12]

## 7. 可以直接展示的真实数字

### 7.1 完整 Test14，nonreactive闭环

以下均为0～1标度；NC=无自车责任碰撞指标，DA=可行驶区域合规，TTC=官方TTC合规。不是“训练代理reward”。[E05][E06][E13]

| 集合 | 模型 | N | score | route progress | NC | DA | TTC | comfort |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hard | 本地重训DP（LR未完全对齐） | 272 | .602691 | .809441 | .779412 | .886029 | .713235 | .886029 |
| hard | B10 | 272 | .705304 | .819141 | .867647 | .944853 | .761029 | .977941 |
| hard | 旧RL Epoch2 | 272 | .702422 | .827959 | .860294 | .941176 | .761029 | .974265 |
| hard | 几何/更新修复版 seed0 | 272 | .704216 | .807828 | .863971 | .948529 | .772059 | .970588 |
| random | 本地重训DP（LR未完全对齐） | 261 | .689908 | .900808 | .800766 | .892720 | .766284 | .869732 |
| random | B10 | 261 | .824270 | .910270 | .908046 | .965517 | .869732 | .977011 |
| random | 旧RL Epoch2 | 261 | .831146 | .915213 | .911877 | .957854 | .877395 | .977011 |
| random | 几何/更新修复版 seed0 | 261 | .830803 | .902460 | .919540 | .969349 | .873563 | .977011 |

不能将hard/random拼接后按533个相互独立样本做显著性推断：两集合重叠11个token，并且同log场景存在相关性。所有模型完成仿真不表示驾驶安全通过。当前没有正式 reactive Test14 对照可据此展示。

### 7.2 几何错误的直接证据

来自 [collision_geometry_audit.json][E14]。下列为旧RL实际闭环碰撞帧；正值表示分离，负值表示重叠。是同一帧不同几何计算，不是新模型实验得分。

| token | 旧几何 separation/m | 校正后/m | 官方重叠 |
|---|---:|---:|---|
| 2e56997063c057a0 | +1.650986 | -0.004441 | 是 |
| 33187fb09d0e52f8 | +1.468604 | -0.138791 | 是 |
| a874cefaca8e5b69 | +1.497225 | -0.151775 | 是 |
| 75d56d4c7b6f5013 | +0.131522 | -0.011873 | 是，但官方非自车责任 |

坐标修复同时会消除车尾的假碰撞，不能把reward no_collision均值上升简单解释成策略更安全。风险、碰撞、跟车关系都受同一坐标语义影响。

### 7.3 6场诊断不是独立测试

| 模型 | score | NC | DA | TTC | progress |
|---|---:|---:|---:|---:|---:|
| B10 | .591913 | 5/6 | 6/6 | 2/6 | .768616 |
| 旧RL | .114583 | 3/6 | 3/6 | 1/6 | .786639 |
| 更新约束修复、旧几何 | .641437 | 5/6 | 6/6 | 3/6 | .754440 |
| 更新约束+几何修复 | .694002 | 5/6 | 6/6 | 4/6 | .756441 |

来源：[旧模型诊断][E15]、[约束版][E16]、[几何版][E17]。案例由旧失败选择；同seed重新编排场景也可能改变随机流，不能把该表和原完整集合对应行混搭。它展示修复方向，不能证明全分布改进。

## 8. 模型身份与历史来源

| 证据对象 | SHA256 | 关联历史 |
|---|---|---|
| B10 | 22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce | 共同4轮+B分支6轮；当前args被后续Epoch20续训覆盖 |
| 原始DP本地重训E10 | 37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786 | 4bf2839归档对齐训练/评测；本轮实际LR纠正 |
| 旧RLE2 | 8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c | b13841b最接近历史训练快照；不等于逐字相同的训练工作区 |
| 几何修复E2 | 3a2348a12eb88f32018921514ffd4bce46720392370af9928aa0392ac4e56948 | [run.json][E18]记录完整命令与输入hash；代码修复归入d52d49c |

7571a3b引入HDP两个环境，7200254引入nuPlan RL/数据管线，a38e5b0加入reference工具，d08c0d7整理分析，d52d49c安全修复。算法思想先验归属以02文献矩阵为准，不用git history代替文献溯源。

## 9. 中期检查的可辩护表述

可以说：“已完成扩散规划与奖励后训练实验平台，并在完整Test14上发现后训练安全—效率收益不稳定；已定位并修复一种奖励几何语义错误，完成受控训练和闭环验证，进一步识别公平基线与代理目标对齐缺口。”

不应说：“已提出并充分验证全新HDP算法”“已实现在线闭环RL”“已保证安全”“已完成三章原创成果”“修复版稳定优于B10”“已证明混合损失优于DP原方法”。

**当前最稳妥定位：基础模型与平台已成形，负结果和机制证据充实，原创方法和因果实验仍需收敛。** 下一步路线比较见04；详细不足见03。本轮到04停止。

[ROOT]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
[DP]: /home/yanjun/NewDisk/Diffusion-Planner/diffusion_planner
[LIT]: </home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/ Compiled_articles>
[C01]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/model/hyper_diffusion_planner.py
[C02]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/loss.py
[C03]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/train_predictor.py
[C04]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/model/diffusion_utils/sde.py
[C05]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/utils/traj_kinematics.py
[C06]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/train_predictor_rl.py
[C07]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/loss.py
[C08]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py
[C09]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/reward.py
[C10]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests/test_rl_vehicle_geometry.py
[C11]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/safety_update.py
[C12]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/replay_buffer.py
[C13]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/rl/trajectory_augmentation.py
[C14]: /home/yanjun/NewDisk/Diffusion-Planner/diffusion_planner/model/guidance/collision.py
[C15]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/planner/planner.py
[C16]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/data_process
[C17]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/data_process/sampling.py
[C18]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/evaluate_checkpoints.py
[C19]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/prepare_test14_remote_subset.py
[C20]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/evaluate_full_test14_three_models_chunked.sh
[C21]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/diagnose_rl_safety_regressions.py
[C22]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/evaluate_rl_safety_geometry_multiseed.py
[C23]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests
[C24]: /home/yanjun/NewDisk/Diffusion-Planner/diffusion_planner/utils/lr_schedule.py
[C25]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/hdp_nuplan/utils/lr_schedule.py
[D01]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan/HDP_B_Epoch10训练过程详解.md
[D02]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan/原始DiffusionPlanner同数据公平对照操作日志.md
[E01]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache_validation_report.json
[E02]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache_validation_report.json
[E03]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20/closed_loop_b_epoch10_vs20.json
[E04]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/coverage_validation.json
[E05]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-hard_three_models.json
[E06]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-random_three_models.json
[E07]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_repair_0915_2048/trace_analysis.json
[E08]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_multiseed_0916_1455/protocol.json
[E09]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/tb
[E10]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/tb
[E11]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/training_log/hdp-rl-safety-repair-controlled/2026-09-15-21:12:50/args.json
[E12]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/train.log
[E13]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_test14_seed0/full_comparison.json
[E14]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_repair_0915_2048/collision_geometry_audit.json
[E15]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_repair_0915_2048/diagnosis.json
[E16]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_constraint_only_eval_0915_2112/diagnosis.json
[E17]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_eval_0915_2123/diagnosis.json
[E18]: /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_training_0915_2112/run.json
