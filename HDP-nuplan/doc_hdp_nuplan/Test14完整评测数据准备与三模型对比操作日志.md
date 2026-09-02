# Test14 完整评测数据准备与三模型对比操作日志

## 1. 目标

在完整 NuPlan test split 上，对以下三个模型使用完全相同的场景、闭环模式和指标配置进行比较：

1. 对齐训练的原始 Diffusion Planner Epoch 10；
2. HDP B Epoch 10；
3. 从 B Epoch 10 后训练得到的 RL Epoch 2（seed 2026）。

第一阶段只评测：

- `test14-hard`：272 个场景；
- `test14-random`：261 个场景。

## 2. 2026-08-30 数据与空间核验

官方 NuPlan v1.1 test ZIP：

```text
https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip
```

通过 HTTP HEAD 与 ZIP64 中央目录核验得到：

| 项目 | 数值 |
|---|---:|
| ZIP Content-Length | 95,919,476,643 bytes |
| ZIP 压缩体积 | 89.33 GiB |
| SQLite DB 数量 | 1,349 |
| DB 解压总体积 | 151.14 GiB |

本机 `/home/yanjun/NewDisk` 清理后可用约 72 GiB，不能容纳完整 ZIP，更不能同时容纳压缩包与解压结果。其余单个分区也没有足够且安全的连续空间。因此不直接下载整个 ZIP。

## 3. 磁盘受限方案

新增脚本：

```text
HDP-nuplan/scripts/prepare_test14_remote_subset.py
```

流程分为三个可恢复阶段：

1. `scan`：利用 HTTP Range 从远程 ZIP 逐个提取 DB；每次只保留当前 DB，使用 NuPlan devkit 原生查询读取有效场景后立即删除 DB，并把小型索引写入 SQLite；
2. `select`：严格按照 `NuPlanScenarioBuilder` 的排序、每类等距采样和 15 秒时间过滤复现 `test14-random`，同时定位 `test14-hard` 的显式 token；
3. `extract`：只重新下载最终 533 个场景实际涉及的 DB。

每个 ZIP 成员均验证：

- 解压后字节数；
- ZIP 中央目录 CRC32；
- 永久保留 DB 的 SQLite `PRAGMA quick_check`。

扫描支持断点恢复。并行 worker 的完成顺序不会改变采样结果，因为最终选择始终按照 NuPlan 使用的 DB 文件名字典序和 DB 内 timestamp 顺序进行。

## 4. 空间判断修正

早期按 OpenScene test metadata 的体积判断本地空间足够并不适用于本任务。OpenScene metadata 不是当前 NuPlan devkit 闭环评测直接使用的 1,349 个 SQLite DB。实际应以官方 NuPlan test ZIP 的 ZIP64 中央目录统计为准。

## 5. 当前状态

- 已确认官方 ZIP 可通过 HTTP Range 访问；
- 已确认 ZIP 中 1,349 个 DB 的压缩与解压总体积；
- 已设计并实现远程逐 DB 扫描、断点恢复、选择和按需提取脚本；
- 已完成 2 个 DB 的真实远程 smoke：两个 DB 均通过解压长度、CRC32 和 NuPlan SQLite 查询，临时 DB 已删除，断点索引显示 `2/1349`；
- smoke 双 worker 累计速度约 4.16 MiB/s，按该速度完整第一遍扫描约需 6 小时；
- 正式扫描采用 4 个 worker 和有界在途任务队列，网络失败会重试，最终失败时可快速停止并从已完成 DB 继续。

## 6. 固定模型身份

| 模型 | checkpoint SHA256 |
|---|---|
| 对齐训练原始 DP Epoch 10 | `37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786` |
| HDP B Epoch 10 | `22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce` |
| HDP RL v4 seed2026 Epoch 2 | `8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c` |

统一评测脚本在运行前同时验证三个 checkpoint 和三个 `args.json` 的 SHA256，防止实验期间路径对应文件被替换。

## 7. 覆盖门禁与评测脚本

新增：

```text
HDP-nuplan/scripts/validate_test14_full_coverage.py
HDP-nuplan/scripts/evaluate_full_test14_three_models.sh
```

覆盖检查直接调用 `NuPlanScenarioBuilder`，不加载模型，要求：

```text
test14-hard=272
test14-random=261
```

统一评测使用相同 DB、场景 token、`closed_loop_nonreactive_agents`、metric 配置和双线程调度。通过 Hydra 删除 `simulation_log_callback`，只取消逐帧回放文件，不取消 `MetricCallback`、聚合指标和 runner report，以控制磁盘占用。

## 8. 选择算法交叉验证

使用当前本地 64 个 mini DB 独立运行远程索引方案中的选择逻辑，并与此前由原始 `NuPlanScenarioBuilder` 生成、已完成 B Epoch 10 闭环评测的 `test14-random` 汇总逐 token 对比：

| 项目 | 数量 |
|---|---:|
| 索引算法选择 | 186 |
| 历史 ScenarioBuilder 汇总 | 186 |
| token 交集 | 186 |
| 索引算法缺失 | 0 |
| 索引算法额外 | 0 |

因此，远程扫描完成后用同一逻辑得到的 261 个完整 test split token，能够保持与 NuPlan 当前过滤流程一致，而不是另行随机抽样。

## 9. 低占用闭环 smoke

使用 B Epoch 10 和 mini DB 中一个真实 `test14-hard` 场景，执行与正式评测相同的 `closed_loop_nonreactive_agents`，并删除 `simulation_log_callback`：

| 检查项 | 结果 |
|---|---:|
| 场景成功/失败 | 1/0 |
| metrics parquet | 正常生成 |
| 聚合指标 | 正常生成 |
| runner report | 正常生成 |
| `simulation_log` 目录 | 0 |
| 单场景结果占用 | 约 1.2 MB |

证明关闭逐帧回放不会关闭 NuPlan 指标计算。按单场景体积估算，三个模型共 1,599 次闭环的结果约为 2 GB。

## 10. 自动续跑与磁盘门禁

新增：

```text
HDP-nuplan/scripts/continue_full_test14_pipeline.sh
```

该脚本等待完整扫描结束，然后自动执行：

1. 生成 Test14-hard/random 选择清单；
2. 检查所需 DB 解压体积；
3. 只有在提取后仍能保留至少 15 GiB 空间时才继续；
4. 提取所需 DB；
5. 执行 `272/261` 场景覆盖门禁；
6. 启动三模型统一闭环评测。

任一数量、SHA256、SQLite 完整性或空间门禁失败都会停止流水线。

考虑本地网络曾出现 TLS EOF，续跑器还会在扫描进程异常结束且 DB 未达到 `1349/1349` 时，从 SQLite 断点自动重启，最多尝试 10 次。已完成 DB 不会重复扫描。

## 11. 完整扫描结果、空间释放与提取启动（2026-08-30）

远程扫描最终完成：

| 项目 | 结果 |
|---|---:|
| 已扫描 test DB | `1349/1349` |
| 索引中的候选场景行 | `757,844` |
| Test14-hard 最终选择 | `272/272` |
| Test14-random 最终选择 | `261` |
| 两个集合涉及的去重 DB | `217` |
| 所需 DB 解压体积 | `70.79 GiB` |

第一次通过空间门禁时，`NewDisk` 只有约 `72 GiB` 可用；提取所需 DB 后无法继续保留 15 GiB 安全余量，因此流水线按设计停止，没有冒险写满磁盘。

为避免删除正式 NPZ、NuPlan 数据和 checkpoint，使用 `/home` 的独立分区保存以下原始目录，并在原路径建立软链接：

```text
/home/yanjun/NewDisk/CARLA_0.9.16
  -> /home/yanjun/storage_offload_from_NewDisk_20260830/CARLA_0.9.16

/home/yanjun/NewDisk/temp
  -> /home/yanjun/storage_offload_from_NewDisk_20260830/temp
```

迁移前后分别检查了字节数和文件数，并执行 `rsync -aHn --delete --itemize-changes`，两组目录的差异均为 0。迁移释放约 `22 GiB`，`NewDisk` 可用空间从约 `72 GiB` 增加到约 `94 GiB`；原路径读取检查通过。

随后重新启动 `continue_full_test14_pipeline.sh`。流水线跳过已完成扫描，重新生成相同选择结果，通过空间门禁，并于 `2026-08-30 19:04` 开始按清单提取 217 个必需 test DB。提取完成后将自动执行 `272/261` 覆盖验证和三个模型的统一闭环评测。

## 12. 闭环评测内存问题与分片恢复（2026-08-30）

完整 DB 提取于 `20:22` 完成，覆盖门禁得到：

```text
test14-hard=272
test14-random=261
db_count=217
```

随后首先启动 `aligned-original-dp-test14-hard`。原评测一次构建并提交全部 272 个仿真；运行约 200 个场景后，Python RSS 增长到约 9～10 GiB，本机 15 GiB 物理内存和 15 GiB Swap 接近耗尽，造成明显桌面卡顿和换页抖动。GPU 利用率很低，因此瓶颈不是 GPU，而是一次驻留过多 simulation/scenario 对象。

决定让后续任务改成每片 40 个场景：

| Benchmark | 总场景 | 分片数量 | 分片大小 |
|---|---:|---:|---|
| Test14-hard | 272 | 7 | `40×6 + 32` |
| Test14-random | 261 | 7 | `40×6 + 21` |

切换编排时，先冻结旧评测 Bash 进程，仅允许当前 Python 子进程继续。之后终止旧流水线会话主进程时，同一终端进程组收到了挂断信号，当前 Python 在 `227/272` 提前退出。这是进程层级处理失误，不是模型、场景或 NuPlan 异常。

中断后检查：

- 已留下 227 个 `*.pickle.temp` 场景指标文件；
- 逐文件执行 pickle 反序列化，`227/227` 均有效；
- 每个文件中的 `planner_name=diffusion_planner`；
- 227 个 scenario token 均属于完整 Test14-hard，且无重复；
- 缺失 token 精确为 45 个，恢复分为 `40 + 5` 两片。

已完成的 227 个场景不重跑。它们保留完整 NuPlan 官方指标，但被中断进程尚未来得及写出 `runner_report.parquet`，因此这些场景的仿真耗时记录不可恢复；得分、安全、路线进度和舒适性指标不受影响。补跑的 45 个场景保留完整 runner report，最终聚合器读取累计 272 个指标文件并生成完整官方指标表。

新增工具：

```text
HDP-nuplan/scripts/prepare_test14_eval_chunks.py
HDP-nuplan/scripts/prepare_test14_metric_recovery_chunks.py
HDP-nuplan/scripts/manage_test14_chunk_results.py
HDP-nuplan/scripts/evaluate_full_test14_three_models_chunked.sh
HDP-nuplan/scripts/continue_test14_after_current_run_chunked.sh
```

分片方案具备以下门禁：

1. 从 `coverage_validation.json` 生成固定 token 分片；
2. 使用真实 217 个 DB 逐片调用 `NuPlanScenarioBuilder`，验证 272/261 全覆盖、无遗漏、无重复；
3. 每片结束后校验 runner report 的 token 集合、行数和 `succeeded=true`；
4. 每片单独保存 runner report，支持断点跳过；
5. 最终合并 runner report，并再次校验聚合指标中的完整 token 集合；
6. 三个 checkpoint 和 args 继续执行原 SHA256 门禁。

脚本已通过 Python 编译、Bash 语法、`git diff --check`、分片集合检查，以及合成 parquet 的分片校验/合并测试。新流水线于 `23:21` 启动，首先补跑原始 DP 缺失的 45 个 Test14-hard 场景。首个 40 场景分片启动后 Python RSS 约 2.3 GiB，可用物理内存约 9.7 GiB，显著低于原 272 场景整批运行时约 10 GiB RSS；Swap 不再持续写入。后续五组评测将全部使用相同的 40 场景分片策略。

## 13. 分片聚合门禁修复与正式续跑（2026-08-31）

原始 DP 的两个恢复分片分别于 `2026-08-30 23:42:58` 和 `23:45:29` 完成，结果为 `40/40`、`5/5` 成功。随后严格聚合门禁发现 `aggregator_metric` 下有两个 parquet，因此停止，没有启动 B Epoch10。

原因是 NuPlan 每次运行会：

1. 将本次可见的 `*.pickle.temp` 集成为 metric parquet；
2. 删除这些临时文件；
3. 生成带本次启动时间的 aggregator parquet。

因此，同一 experiment UID 的多个分片不会自动产生一个唯一 aggregator：第一个恢复分片包含中断前 227 个有效临时指标和新补的 40 个，共 267 个场景；第二个分片单独包含剩余 5 个场景。两个文件分别覆盖 267 和 5 个不重复 token，合计正好 272。

修复 `manage_test14_chunk_results.py` 与分片评测脚本：

- 每片同时归档 runner report 和对应 aggregator parquet；
- 聚合前逐片验证必需 token；
- 合并时要求总行数、唯一 token 和目标集合完全一致，发现重复或遗漏立即失败；
- 原始分片 aggregator 移入 `aggregator_metric/chunk_originals/` 保留；
- 顶层只生成一个 `chunk_merged_scenario_metrics.parquet`，供统一汇总脚本读取；
- 对中断恢复运行明确记录 `runner_report_count=45`、`metric_only_recovered_scenarios=227`，不伪造不可恢复的耗时记录。

修复通过正常分片和“已有指标 + 缺失补跑”的合成 parquet 测试。真实结果门禁随后通过：

```text
aligned-original-dp-test14-hard metrics = 272/272
runner_report_count = 45
metric_only_recovered_scenarios = 227
duplicate metric tokens = 0
```

`2026-08-31 10:01` 从断点重新启动，脚本跳过已完成的原始 DP Test14-hard，开始 `hdp-b-epoch10-test14-hard` 的第 0 片（40 个场景）。该进程 RSS 约 2.3 GiB，可用物理内存约 7.3 GiB，未再出现整批评测的内存耗尽。

## 14. OmegaConf token 类型门禁与第二次断点恢复（2026-08-31）

B Epoch10 / Test14-hard 前三片分别于 `10:24`、`11:02`、`11:27` 完成，均为 `40/40` 成功，累计 `120/272`。第 3 片启动时，NuPlan 的 `build_scenario_filter` 类型门禁报错并终止：某些 16 位 token 含字母 `e`，在未引用的 YAML 标量中可能被 OmegaConf 解释为科学计数法浮点数，而不是字符串。

这不是场景缺失或模型失败。生成器此前通过 PyYAML/ScenarioBuilder 验证，但未覆盖 OmegaConf 对未引用标量的额外类型解析。修复如下：

- `prepare_test14_eval_chunks.py` 和恢复配置生成器对每个 token 强制使用双引号；
- 使用 `OmegaConf.load` 逐配置确认所有 token 均为长度 16 的 `str`；
- 再次用真实 217 个 DB 验证 14 个分片，结果仍为 Test14-hard `272/272`、Test14-random `261/261`，无重复或遗漏；
- Python 编译、Bash 语法和 `git diff --check` 均通过。

`2026-08-31 11:30` 再次从断点启动：原始 DP 的完整 272 指标被跳过，B Epoch10 已完成的前三片也通过 runner/aggregator 双门禁后被跳过，流水线直接从第 3 片继续。

## 15. 指标回调线程安全问题与一致性重测（2026-08-31）

Test14-hard 的 B Epoch10 于 `12:51` 完成 `272/272`；RL Epoch2 在最后一片完成仿真后报告 `31/32` 成功，失败 token 为 `edc971e9dc7f5165`。失败发生在 NuPlan 官方 `speed_limit_compliance` 指标中：`time_stamps` 与 `violation_depths` 长度不一致，并非模型推理或仿真本身崩溃。

代码核查确认：NuPlan 的 `build_metrics_engines()` 按 `scenario_type` 复用同一个有状态 `MetricsEngine`，其中 `EgoLaneChangeStatistics.ego_driven_route` 等字段会在每场景计算时改写。此前同时设置了两路仿真和两路指标回调；当同类场景的指标并发计算时，共享状态可能互相覆盖。本次长度断言是该竞争条件的显式证据；未触发异常的旧结果也不能排除静默污染，因此不能仅补跑一个失败场景后将旧结果作为正式对比。

初次修复尝试只将 `max_callback_workers` 从 2 改为 1。继续核查
`build_callbacks_worker()` 后确认：当前使用的是 `single_machine_thread_pool`，不是
`Sequential` worker；在这种模式下函数直接返回 `None`，所以该参数不会创建独立的
callback worker pool，指标仍在两个仿真线程中同步执行。该尝试在首片尚未完成时立即终止，
其产物不进入正式结果。

最终处理原则：

1. 三个模型、两个 benchmark、场景 token、checkpoint、推理参数和闭环模式均保持不变；
2. 将 `worker.max_workers` 固定为 1，使仿真及其同步指标回调都在唯一工作线程中执行；
3. 脚本增加单 worker 启动门禁，若未来误改为大于 1，则在启动任何场景前直接退出；
4. 不修改 NuPlan 指标公式，不跳过失败指标，不手工填补结果；
5. 已生成的 Test14-hard 结果完整移动到
   `three_model_eval/unsafe_parallel_metrics_backup_20260831_160747/`，作为问题复盘证据，不删除；
6. 从零重测三个模型的 Test14-hard，之后按同一安全配置继续 Test14-random。

最终修复通过 Bash 语法检查和 533 个固定场景的真实 DB 覆盖校验。单 worker
配置从机制上消除了同进程内两个场景同时进入共享指标引擎的可能；这只解决已识别的
线程竞争，不等同于承诺评测不会遇到其他数据或系统异常。

无效的 callback-limit 尝试产物已移动到
`three_model_eval/ineffective_callback_limit_attempt_20260831_1624/`。正式单 worker
重测于 `2026-08-31 16:26` 启动，实际进程参数已核验为：

```text
worker.max_workers=1
worker.use_process_pool=false
number_of_gpus_allocated_per_simulation=1
max_callback_workers=1
disable_callback_parallelization=true
```

## 16. 用户要求暂停（2026-08-31 22:58）

正式单 worker 重测在 B Epoch10 / Test14-hard 最后一片运行到 `31/32` 时，按用户要求
对完整进程组 `PGID=179573` 发送 `SIGSTOP`。Bash 编排器和 NuPlan Python 子进程均已
进入 `T`（stopped）状态，没有终止进程，也没有删除或改写已归档分片。恢复时对同一
进程组发送 `SIGCONT`，可从当前内存状态继续最后一个场景。

## 17. 系统重启后的断点恢复（2026-09-01 23:24）

用户要求继续时，系统 uptime 仅约 3 分钟，原暂停进程组已随系统重启消失，无法执行
`SIGCONT` 原地恢复。检查确认：原始 DP / Test14-hard 的 272 场景完整结果仍通过门禁；
B Epoch10 的前六片（240 场景）runner/aggregator 均已正式归档；最后一片留下 31 个
临时指标文件，但没有完整 runner report，因此不将其冒充完整分片。

流水线于 `2026-09-01 23:24` 按相同单 worker 配置重新启动，自动跳过原始 DP 完整结果
和 B Epoch10 已归档的前六片，仅重新运行 B Epoch10 / Test14-hard 最后一片 32 个场景。

## 18. Test14-hard 完成与 Test14-random 退出恢复（2026-09-02）

Test14-hard 三个模型均完成 272 场景并通过 runner/aggregator 完整集合门禁，汇总及
B Epoch10 对 RL Epoch2 的配对分析于 `2026-09-02 02:27` 生成。随后完成原始 DP 的
Test14-random `261/261`，并完成 B Epoch10 前六片 `240/261`。

B Epoch10 / Test14-random 第六片在 `40/40` 仿真成功、runner 与 aggregator 均完成
归档后，进程退出析构阶段输出 `terminate called without an active exception`，使 Bash
在写入 `Completed chunk` 日志前停止。该异常发生在正式结果归档之后；重启时分片双门禁
验证通过，因此不重跑这 40 个场景。

流水线于 `2026-09-02 09:36` 恢复，自动跳过所有完整运行及 B Epoch10 已归档的前六片，
仅运行 B Epoch10 / Test14-random 最后一片 21 个场景，随后继续 RL Epoch2 的完整
Test14-random 261 场景。

## 19. 三模型完整结果（2026-09-02 12:26）

三模型在 Test14-hard 272 场景和 Test14-random 261 场景上的正式单 worker 闭环评测
全部完成，共 `1599/1599` 次仿真成功，失败数为 0。两个 benchmark 的三组结果均通过
场景集合一致、runner report 完整和 aggregator token 无重复/无遗漏门禁。

| Benchmark | 模型 | 场景数 | 综合得分 | 路线进度 | 无自车责任碰撞 | 可行驶区域 | TTC | 舒适性 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Test14-hard | 对齐原始 DP | 272 | 0.602691 | 0.809441 | 0.779412 | 0.886029 | 0.713235 | 0.886029 |
| Test14-hard | B Epoch10 | 272 | 0.705304 | 0.819141 | 0.867647 | 0.944853 | 0.761029 | 0.977941 |
| Test14-hard | RL Epoch2 | 272 | 0.702422 | 0.827959 | 0.860294 | 0.941176 | 0.761029 | 0.974265 |
| Test14-random | 对齐原始 DP | 261 | 0.689908 | 0.900808 | 0.800766 | 0.892720 | 0.766284 | 0.869732 |
| Test14-random | B Epoch10 | 261 | 0.824270 | 0.910270 | 0.908046 | 0.965517 | 0.869732 | 0.977011 |
| Test14-random | RL Epoch2 | 261 | 0.831146 | 0.915213 | 0.911877 | 0.957854 | 0.877395 | 0.977011 |

RL Epoch2 相对 B Epoch10：

| Benchmark | 综合得分差值 | 路线进度差值 | 碰撞指标差值 | 可行驶区域差值 | TTC 差值 |
|---|---:|---:|---:|---:|---:|
| Test14-hard | -0.002882 | +0.008818 | -0.007353 | -0.003676 | 0.000000 |
| Test14-random | +0.006876 | +0.004944 | +0.003831 | -0.007663 | +0.007663 |
| 533 场景加权合并 | +0.001896 | +0.006921 | — | — | — |

结论：RL Epoch2 在两个评测集上都提高了路线进度；综合得分在 Test14-random 为正收益，
在 Test14-hard 为小幅负收益。533 个场景按样本数加权后，综合得分由 B Epoch10 的
`0.763559` 提升到 `0.765456`（`+0.001896`），说明总体是小幅正收益，但尚未做到跨
benchmark 稳定正收益。

## 16. 四小时执行会话回收与持久化恢复（2026-08-31）

单 worker 重测首先完成了原始 Diffusion Planner / Test14-hard `272/272`，并通过
7 个分片的 runner/aggregator 合并门禁。随后 B Epoch10 完成前两片 `80/272`，
第 3 片运行至 `33/40` 时，承载前台任务的统一执行会话在约 4 小时处被环境回收。
日志中没有仿真失败、指标异常或 OOM 记录；Bash 和 Python 进程同时消失，停止时间与
会话寿命一致，因此判定为任务托管方式问题，不是模型问题。

为避免再次受到交互会话寿命影响，评测改由用户级 systemd transient service 托管：

```text
unit: hdp-test14-eval.service
working directory: /home/yanjun/NewDisk/Diffusion-Planner
active since: 2026-08-31 21:00:51 CST
```

恢复脚本已自动跳过完整的原始 DP `272/272` 和 B Epoch10 已归档的前两片；未完成的
第 3 片从头重跑，以免采用没有 runner report 和正式聚合结果的半片数据。`21:00:58`
恢复运行 B Epoch10 / Test14-hard chunk 2，实际参数仍为单 worker 安全配置。
