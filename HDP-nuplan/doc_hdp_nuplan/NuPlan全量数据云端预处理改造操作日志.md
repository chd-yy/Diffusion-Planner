# NuPlan 全量数据云端预处理改造操作日志

日期：2026-08-09

运行环境：`/home/yanjun/NewDisk/conda_envs/diffusion_planner`

工程目录：`/home/yanjun/NewDisk/Diffusion-Planner`

## 1. 改造目标

本地磁盘无法长期保存完整 NuPlan 原始数据，因此预处理必须能在云端按小批日志执行，并把处理结果分批上传到对象存储。改造目标如下：

1. 原始日志按 50～100 个 log 切成一个 shard，避免一次构造 13,180 个 train log 的全部场景。
2. 已完成 NPZ 可以跳过，任务中断后能够续跑。
3. 单个坏 Scenario 只记录失败，不中断同一 shard 的后续 Scenario。
4. NPZ 和 JSON 使用原子写入，进程中断时不留下伪装成完整文件的半文件。
5. 每个 shard 保存 manifest、抽样报告、处理报告和 SHA256。
6. 多个 shard 可以合并为训练可直接读取的总 manifest。
7. 合并前检查 shard 重复，训练/验证集使用前检查 log 和 Scenario 泄漏。

本次只改造数据工程链路，不改变 NPZ 中的模型特征、张量形状、训练损失或 RL 算法。

## 2. 原流程的问题

原入口为 `HDP-nuplan/data_process.py`，主要流程是：

```text
完整 log_names.json
  -> NuPlanScenarioBuilder.get_scenarios()
  -> 一次得到全部候选 Scenario
  -> DataProcessor.work() 串行逐场景处理
  -> 直接 np.savez(目标文件)
  -> 最后生成 manifest
```

存在四个全量数据风险：

- `get_scenarios()` 在均衡抽样前先构造清单中所有日志的候选场景，日志清单过大时占用大量内存和时间。
- 任意一个 Scenario 抛异常都会终止整个任务。
- `np.savez()` 直接覆盖目标文件，异常退出可能留下不完整文件。
- 重跑时没有跳过已完成文件，会重复计算。

## 3. 本次新增或修改的文件

| 文件 | 作用 |
|---|---|
| `data_process.py` | 分片级预处理入口；写 manifest、报告和校验和；任务结束后统一判断是否失败 |
| `hdp_nuplan/data_process/data_processor.py` | 单场景失败隔离、断点续跑、损坏文件重算、NPZ 原子写入 |
| `hdp_nuplan/data_process/run_utils.py` | JSON 原子写入、SHA256、日志名校验与去重 |
| `hdp_nuplan/data_process/sampling.py` | 抽样前按最终 NPZ 名稳定去重 |
| `scripts/build_preprocessing_plan.py` | 把完整日志清单切分为 shard，并精确分配总场景目标 |
| `scripts/download_nuplan_log_subset.py` | 从 NuPlan v1.1 城市 ZIP 中按日志做 HTTP Range 下载，并校验 SQLite 与 SHA256 |
| `scripts/run_preprocessing_shard.py` | 根据计划执行指定 shard；适合云平台 array job |
| `scripts/run_preprocessing_range.py` | 多 worker 互斥滚动执行下载、预处理、强校验、报告归档和 raw 回收，并保存恢复状态 |
| `scripts/merge_preprocessing_shards.py` | 合并成功 shard，检查重复并生成总 manifest |
| `scripts/audit_dataset_splits.py` | 检查 train/val 是否共享 log 或 NPZ |
| `scripts/validate_processed_cache.py` | 增加对 `shard_x/cache/*.npz` 嵌套 manifest 的验证 |
| `tests/test_preprocessing_pipeline.py` | 覆盖失败隔离、续跑、分片、合并和泄漏检查 |
| `tests/test_nuplan_subset_download.py` | 覆盖带点日志名、索引和下载报告等按日志下载行为 |
| `tests/test_preprocessing_range.py` | 覆盖 worker 分工、范围选择、报告先归档及精确 raw 清理 |
| `requirements_data_download.txt` | 固定按 ZIP 成员下载所需的 `remotezip` 版本 |

## 4. 改造后的执行流程

```text
完整官方 log 清单
  -> build_preprocessing_plan.py
  -> shard_00000_logs.json ... shard_00131_logs.json
  -> 云端 array job：每个任务只读取一个 shard 的日志
  -> data_process.py
       -> 场景处理成功：原子写 NPZ
       -> 已有可读 NPZ：跳过
       -> 已有损坏 NPZ：重算并原子替换
       -> 单场景失败：记录 traceback，继续下一个
  -> shard manifest / sampling report / processing report / checksums
  -> 上传该 shard 到对象存储
  -> 删除该 shard 对应的临时原始 DB
  -> merge_preprocessing_shards.py
  -> audit_dataset_splits.py
  -> validate_processed_cache.py
  -> 监督训练
```

## 5. 关键行为说明

### 5.1 断点续跑

`--skip_existing true` 是默认值。目标 NPZ 已存在时先用 `numpy.load(..., allow_pickle=False)` 打开：

- 能正常打开且至少有一个字段：计入 `skipped_existing`。
- 无法打开：计入 `reprocessed_invalid`，重新处理并替换。

原子写入能够保证新流程产生的目标 NPZ 要么完整存在，要么不存在。可读性检查主要用于兼容旧缓存、传输损坏或人工中断遗留文件。

### 5.2 单场景失败隔离

每个 Scenario 单独执行 `try/except`。错误记录包含：

- `log_name`、`map_name`、`scenario_type`、`token`；
- 异常类型和信息；
- 完整 Python traceback；
- 对应的预期 NPZ 文件名。

程序会处理完 shard 中其余场景，写出成功 manifest 和所有报告。默认 `--fail_on_error true`，因此报告写完后仍返回非零退出码，避免云调度系统把部分成功误判为完整成功。修复数据或代码后用同一命令重跑，成功文件会被跳过。

### 5.3 原子写入

NPZ 和 JSON 都先写到目标目录下的临时文件，刷新到磁盘后调用 `os.replace()`。同一文件系统内替换是原子的，训练进程不会看到写了一半的目标文件。

### 5.4 校验和

- `--checksum_mode manifest`：只计算 manifest、sampling report 和 processing report 的 SHA256，速度快。
- `--checksum_mode files`：另外计算每个 NPZ 的 SHA256，适合上传对象存储前做强完整性校验，但会多读取一遍所有 NPZ。

建议试运行和最终归档使用 `files`，大规模频繁重跑阶段使用 `manifest`。

### 5.5 总 manifest 的路径

每个 shard 的 manifest 保存简单文件名，例如：

```text
us-nv-las-vegas-strip_xxx.npz
```

合并后的 manifest 保存相对总数据根目录的路径，例如：

```text
shard_00000/cache/us-nv-las-vegas-strip_xxx.npz
```

训练时把 `data_dir` 设置为所有 `shard_*` 的共同根目录即可。现有 `DiffusionPlannerData` 使用 `os.path.join(data_dir, file_name)`，可以直接读取这种嵌套路径，不需要把百万个 NPZ 摊平到一个目录。

### 5.6 官方日志可能没有可用 Scenario

真实 train 数据验证发现，合法且 SQLite 完整的 DB 也可能因为 NuPlan 官方 scene 边界条件而生成 0 个 Scenario。默认仍使用严格模式，防止错误路径或缺失数据被静默跳过；正式处理已核验的官方 split 时需显式传入：

```text
--allow_empty_logs
```

此时抽样器会把空日志记录到 `empty_logs`，并在其余有效日志间重新分配 shard 的完整目标数。报告中的字段含义为：

```text
log_count            请求并参与 split 审计的全部日志数
selected_log_count   实际贡献 NPZ 的日志数
empty_log_count      ScenarioBuilder 返回 0 个 Scenario 的日志数
```

空日志仍留在完整 `log_names` 中用于 train/val 泄漏审计，但不会伪造进 `selected_per_log`。

## 6. 本地计划生成验证

执行：

```bash
PLAN_ROOT=/tmp/hdp-plan-test
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python

$PY HDP-nuplan/scripts/build_preprocessing_plan.py \
  --log_names_json HDP-nuplan/nuplan_train.json \
  --output_dir "$PLAN_ROOT" \
  --logs_per_shard 100 \
  --total_scenarios 100000 \
  --seed 3407
```

实际结果：

- 官方 train 清单有 13,180 个唯一日志。
- 生成 132 个 shard。
- 前 131 个 shard 最多 100 个日志，最后一个 shard 有 80 个日志。
- 所有 shard 的场景目标合计严格等于 100,000。
- `shard_00000` 的目标为 800 个场景。
- `shard_00131` 的目标为 560 个场景。

场景目标不是简单地给每个 shard 相同数字，而是先对每个 log 分配 `floor(100000 / 13180)`，再把余数按稳定顺序分配，因此总数不会因向上取整而超出 100,000。

正式云端首轮建议先使用 `--logs_per_shard 50`，观察单 shard 的 Builder 内存峰值和处理时间；确认有余量后再提高到 100。

## 7. 真实 mini 数据冒烟测试

### 7.1 失败尝试及原因

第一次使用了：

```text
HDP-nuplan/config/smoke_logs.json
```

其中的日志不在：

```text
/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini
```

NuPlan Builder 返回 `No log files found`，均衡抽样随后报告该日志缺失。此时尚未进入 NPZ 处理，原因是日志清单与原始 DB 目录不匹配，不是特征处理错误。

### 7.2 修正后的命令

改用官方 mini-train 的 44 日志清单，并以 `random + shuffle=false` 选前 2 个 Scenario：

```bash
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path /tmp/hdp-preprocess-smoke/cache \
  --log_names_json HDP-nuplan/config/mini_splits/mini_train_logs.json \
  --total_scenarios 2 \
  --sampling_strategy random \
  --shuffle_scenarios false \
  --seed 3407 \
  --shard_id smoke_real_00000 \
  --output_list_path /tmp/hdp-preprocess-smoke/manifest.json \
  --sampling_report_path /tmp/hdp-preprocess-smoke/sampling_report.json \
  --processing_report_path /tmp/hdp-preprocess-smoke/processing_report.json \
  --checksum_path /tmp/hdp-preprocess-smoke/checksums.json \
  --checksum_mode files \
  --skip_existing true \
  --fail_on_error true
```

第一次结果：

```text
selected_scenarios=2
processed=2
skipped_existing=0
failed=0
manifest_count=2
status=complete
```

不删除缓存，原命令再次执行：

```text
selected_scenarios=2
processed=0
skipped_existing=2
failed=0
manifest_count=2
status=complete
```

`checksums.json` 中包含 2 个 NPZ 的 SHA256。该结果验证了真实 NuPlan Builder、特征提取、原子写入、manifest、逐文件校验和与断点续跑的完整链路。

测试目录位于 `/tmp`，验证结束后已经删除，没有占用项目磁盘。

## 8. 自动化测试

执行：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner
PYTHONPATH=HDP-nuplan \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果：

```text
38 passed, 15 warnings in 5.77s
```

警告来自现有 `timm` 和 `matplotlib/pyparsing` 依赖的弃用提示，不是本次预处理测试失败。

新增测试覆盖：

1. 一个 Scenario 失败后其余 Scenario 继续处理。
2. 第二次运行跳过可读 NPZ。
3. 损坏 NPZ 被识别并重新生成。
4. 原子写入后不残留临时文件。
5. 日志分片无遗漏，场景目标精确守恒。
6. shard 合并后生成 Dataset 可直接读取的相对路径。
7. train/val 的 log 泄漏和 NPZ 泄漏可以被识别。
8. 缓存验证器可以读取嵌套 shard 路径。

## 9. 云端正式执行命令

### 9.1 生成 train 计划

建议先生成 100,000 场景计划：

```bash
PY=/path/to/conda/env/bin/python
PROJECT=/workspace/Diffusion-Planner
PLAN_ROOT=/workspace/plans/nuplan_train_100k

$PY "$PROJECT/HDP-nuplan/scripts/build_preprocessing_plan.py" \
  --log_names_json "$PROJECT/HDP-nuplan/nuplan_train.json" \
  --output_dir "$PLAN_ROOT" \
  --logs_per_shard 50 \
  --total_scenarios 100000 \
  --seed 3407
```

### 9.2 执行单个 shard

```bash
$PY "$PROJECT/HDP-nuplan/scripts/run_preprocessing_shard.py" \
  --plan "$PLAN_ROOT/preprocessing_plan.json" \
  --shard_index 0 \
  --data_path /local-nvme/nuplan-v1.1/splits/trainval \
  --map_path /local-nvme/nuplan-maps-v1.0 \
  --output_root /local-nvme/processed/nuplan_train_100k \
  --checksum_mode manifest \
  --allow_empty_logs
```

云平台 array job 中只需要把 `--shard_index 0` 替换为平台提供的数组任务编号。`launch.json` 会保存该 shard 的实际命令，便于复盘。

### 9.3 重试

同一个 shard 使用完全相同的命令重跑即可。默认开启：

```text
--skip_existing
--fail_on_error
```

成功文件跳过，失败或损坏文件重算。不要在重试前删除整个 shard。

### 9.4 合并 shard

所有 shard 下载到训练机本地 NVMe 后执行：

```bash
$PY "$PROJECT/HDP-nuplan/scripts/merge_preprocessing_shards.py" \
  --shards_root /local-nvme/processed/nuplan_train_100k \
  --plan /local-nvme/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --output_manifest /local-nvme/processed/nuplan_train_100k/train_manifest.json \
  --output_report /local-nvme/processed/nuplan_train_100k/train_merged_report.json
```

默认要求：

- shard 目录集合与计划精确一致，不能缺失或混入计划外 shard；
- 每个 shard 的 `status=complete` 且 `failed=0`；
- shard 日志、场景目标和 manifest 数量与原计划精确一致；
- shard 之间没有重复日志；
- shard 之间没有重复 NPZ；
- cache 与 manifest 的 NPZ 集合精确一致；
- 每个 shard 的强校验报告存在且哈希仍匹配；
- 默认重新核对 `checksums.json` 中所有元数据和 NPZ 的 SHA256；
- 最终 shard、日志和 NPZ 总数分别等于计划中的 264、13,180 和 100,000。

正式训练机合并不要使用 `--no_verify_files` 或 `--no_verify_checksums`。合并失败时不会生成新的正式 manifest，应先修复具体 shard，再重新执行。

### 9.5 train/val 泄漏审计

```bash
$PY "$PROJECT/HDP-nuplan/scripts/audit_dataset_splits.py" \
  --train_manifest /local-nvme/processed/train/train_manifest.json \
  --train_report /local-nvme/processed/train/train_merged_report.json \
  --val_manifest /local-nvme/processed/val/val_manifest.json \
  --val_report /local-nvme/processed/val/val_merged_report.json \
  --output /local-nvme/processed/train_val_leakage_report.json
```

只要共享一个 log 或一个 NPZ，脚本就写出详细重叠清单并返回非零退出码。

### 9.6 最终缓存验证

合并报告包含 `requested_scenarios` 和 `selected_per_log`，可以作为验证器的 sampling report：

```bash
$PY "$PROJECT/HDP-nuplan/scripts/validate_processed_cache.py" \
  --cache_dir /local-nvme/processed/nuplan_train_100k \
  --manifest /local-nvme/processed/nuplan_train_100k/train_manifest.json \
  --sampling_report /local-nvme/processed/nuplan_train_100k/train_merged_report.json \
  --expected_count 100000 \
  --expected_log_count <train_merged_report中的selected_log_count> \
  --output /local-nvme/processed/nuplan_train_100k/cache_validation_report.json
```

这里不能再硬编码 `13180`。`--expected_log_count` 验证的是实际出现在 NPZ 中的日志数；官方清单中可能存在合法空日志。完整 split 日志数仍应检查 `train_merged_report.json` 的 `log_count=13180`，二者用途不同。

## 10. 云端存储顺序

推荐每个 shard 执行：

1. 从官方数据源把该 shard 需要的原始 DB 下载到临时 NVMe。
2. 保留公共地图目录，不下载当前模型不使用的 camera/LiDAR sensor blobs。
3. 执行 shard 预处理。
4. 检查 `processing_report.json` 的 `status=complete`。
5. 上传完整 `shard_xxxxx/` 到对象存储。
6. 在对象存储端确认对象数量或校验和。
7. 删除该 shard 的临时原始 DB，释放 NVMe。
8. 执行下一个 shard。

训练前再把处理后的 shard 批量同步到 GPU 实例本地 NVMe。不要让 PyTorch DataLoader 直接通过 S3/FUSE 随机读取几十万或上百万个小 NPZ，否则请求延迟会成为主要瓶颈。

## 11. 首个官方 50-log 云端门禁实测

云端环境：AutoDL RTX 4090，数据盘 `/root/autodl-tmp`。

计划配置：

```text
train logs：13,180
logs_per_shard：50
shards：264
total scenarios：100,000
seed：3407
```

`shard_00000` 的 50 个 DB 通过 NuPlan 官方公开的 `motional-nuplan` S3 城市 ZIP 按成员下载，执行 SQLite `quick_check`，结果：

```text
下载：50/50 DB
下载数据：8,341,491,712 bytes，约 7.77 GiB
下载耗时：2,307.856 秒
损坏或缺失：0
```

第一次预处理发现其中两个 DB 各只有 4 个 scene。NuPlan 官方查询要求有效 scene 满足 `row_num >= 3 AND row_num < n.cnt - 1`，因此二者返回 0 个 Scenario；这不是下载损坏。启用显式 `--allow_empty_logs` 后结果为：

```text
请求日志：50
有效日志：48
空日志：2
目标/成功 NPZ：400/400
失败：0
wall time：3 分 11.78 秒
峰值 RSS：976,036 KB
NPZ 总大小：62,995,484 bytes
```

原命令第二次执行：

```text
processed=0
skipped=400
failed=0
wall time=7.76 秒
```

单 shard 和合并根目录两次缓存验证均通过：400 个 NPZ 字段齐全、shape 正确、没有非有限值，manifest 无重复、无缺失、无额外文件。48 个有效日志各贡献 8～11 个场景；合并报告仍保存 50 个请求日志和 2 个空日志。

随后用合并 manifest 做了 400 场景、batch size 8、1 epoch 的监督训练门禁：

```text
50/50 batch 完成
encoder warm-start：151/151 tensors
epoch train loss：12.1730
checkpoint 可读取且包含 EMA、optimizer 和 scheduler
退出码：0
```

本地与云端完整测试分别为：

```text
44 passed, 15 warnings
```

完整命令、绝对路径、故障过程和 checkpoint 位置见：

```text
HDP-nuplan/doc_hdp_nuplan/AutoDL全量训练部署操作日志.md
```

### 11.1 跨城市 4-shard 补充门禁

首个 Las Vegas shard 通过后，选择纯 Pittsburgh `shard_00152`、纯 Singapore `shard_00175`、纯 Boston `shard_00197`，避免相邻 shard 仍全部来自 `vegas_1` 而漏掉跨城市问题。

四 shard 合计结果：

```text
请求日志：200
有效日志：190
空日志：10
计划/实际 NPZ：1,500/1,500
处理失败：0
NPZ 总大小：236,211,544 bytes
合并缓存校验：passed
```

分城市空日志：Las Vegas 2、Pittsburgh 1、Singapore 0、Boston 7。这验证了 `--allow_empty_logs` 必须显式记录和重新分配目标，不能假设所有官方 DB 都能生成 Scenario。

Boston 下载中出现一次 HTTP Range 连接数分钟无 I/O 且不退出。保留 17 个完整 DB、删除唯一 67,108,864-byte 未完成临时文件后，从断点成功续跑。为此下载器新增：

```text
连接超时：10 秒
socket read 空闲超时：60 秒
member 重试：3 次
重试间隔：5 秒
失败 member 重开 RemoteZip/HTTP 连接
报告 retry_count/download_attempts
```

新增超时重试回归测试后，本地与云端完整测试均为 `45 passed`。合并后的 1,500 场景还完成 1 epoch、187 batch 的监督训练门禁，loss `9.4630`，checkpoint 可读取。

因此 4-shard 工程门禁已通过，可以进入 100k 滚动预处理；完整命令、日志路径、每 shard 耗时、峰值内存与故障处理见 `AutoDL全量训练部署操作日志.md` 第 21～22 节。

四份 `download_report.json` 复制到对应 processed shard 并逐份确认 SHA256 一致后，已删除四个明确的临时 `trainval` 目录。raw 占用降为 112K，数据盘可用空间从 146G 增加到 159G；1,500 个 NPZ、全部报告、日志和 checkpoint 均保留。由此“处理并校验后删除可恢复 raw DB”的空间回收步骤也已实跑通过。

### 11.2 100k 滚动任务已启动

新增 `run_preprocessing_range.py`，按 `shard_index % 3` 将 264 个 shard 均分给三个互斥 worker，每个 88 个。执行器实时强校验已有 shard，只有下载、预处理、缓存验证、下载报告归档及 SHA256 核对全部通过后才删除该 shard 的 `trainval`；新下载前还要求数据盘至少剩余 20 GiB。

真实 `shard_00000` 跳过测试通过，完整项目测试更新为本地/云端分别 `49 passed`。随后启动三个 `nohup` worker，PID 为 17454、17455、17456。首次状态显示 worker 0 已跳过 shard 00000 并进入 shard 00003，worker 1/2 分别进入 shard 00001/00002，三个状态均为 running。完整命令、state/log 路径和恢复语义见 `AutoDL全量训练部署操作日志.md` 第 25 节。

## 12. 尚未完成的事项与边界

本次已完成本地代码改造、mini 实跑和首个官方 50-log 云端门禁，但没有把以下工作描述为已完成：

- 尚未下载和预处理官方完整 train 数据。
- 尚未验证对象存储上传命令和 AutoDL 之外平台的 array job 环境变量。
- 已实测四城市 4 shard 的并行下载、串并行重叠预处理与磁盘峰值；尚未实测连续 264 shard 的无人值守任务。
- `DataProcessor.work()` 仍为单进程逐 Scenario 特征提取；先用多 shard/多任务并行扩展。只有在云端基准确认 CPU 利用率不足后，再考虑进程池化单 shard，避免 NuPlan map/DB 对象的进程序列化风险。

因此，下一步不是立即启动 13,180 日志全量任务，而是在当前 AutoDL 实例继续执行以下门槛实验：

1. 四城市 200 log、1,500 Scenario 门禁已完成，不重复执行。
2. 归档各 shard 下载报告后删除已验证的临时 raw DB，验证空间回收。
3. 按 shard 滚动执行剩余 train 计划，持续合并和验证到 100k。
4. 单独生成官方 val 数据并执行 train/val 日志与 NPZ 泄漏审计。
5. 100k 监督训练和评测通过后，再决定是否扩大到 500k。

这条顺序能够先验证云端基础设施和数据正确性，再投入大规模租赁费用。
