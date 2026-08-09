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
| `scripts/run_preprocessing_shard.py` | 根据计划执行指定 shard；适合云平台 array job |
| `scripts/merge_preprocessing_shards.py` | 合并成功 shard，检查重复并生成总 manifest |
| `scripts/audit_dataset_splits.py` | 检查 train/val 是否共享 log 或 NPZ |
| `scripts/validate_processed_cache.py` | 增加对 `shard_x/cache/*.npz` 嵌套 manifest 的验证 |
| `tests/test_preprocessing_pipeline.py` | 覆盖失败隔离、续跑、分片、合并和泄漏检查 |

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
  --checksum_mode manifest
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
  --output_manifest /local-nvme/processed/nuplan_train_100k/train_manifest.json \
  --output_report /local-nvme/processed/nuplan_train_100k/train_merged_report.json
```

默认要求：

- 每个 shard 的 `status=complete`；
- manifest 数量与 processing report 一致；
- shard 之间没有重复日志；
- shard 之间没有重复 NPZ；
- manifest 引用的本地文件真实存在。

只在对象存储控制节点没有下载 NPZ 时，才使用 `--no_verify_files`。正式训练机合并不要关闭文件验证。

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
  --expected_log_count 13180 \
  --output /local-nvme/processed/nuplan_train_100k/cache_validation_report.json
```

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

## 11. 尚未完成的事项与边界

本次已完成本地代码改造和 mini 实跑，但没有在本地伪装完成以下工作：

- 尚未下载和预处理官方完整 train 数据。
- 尚未验证特定云厂商的对象存储命令、array job 环境变量和临时盘挂载路径。
- 尚未测量 50/100 个完整 train log 在云 CPU 实例上的峰值内存和吞吐。
- `DataProcessor.work()` 仍为单进程逐 Scenario 特征提取；先用多 shard/多任务并行扩展。只有在云端基准确认 CPU 利用率不足后，再考虑进程池化单 shard，避免 NuPlan map/DB 对象的进程序列化风险。

因此，下一步不是立即启动 13,180 日志全量任务，而是租一台 CPU 实例执行以下门槛实验：

1. 50 log、约 400 Scenario 的一个 shard。
2. 同一 shard 中断后重跑，确认跳过计数和对象存储上传。
3. 4 个 shard 并行，观察内存、NVMe、DB 读取和地图加载。
4. 合并并验证约 1,600 Scenario。
5. 通过后再执行 100k；100k 监督训练和评测通过后，再扩大到 500k。

这条顺序能够先验证云端基础设施和数据正确性，再投入大规模租赁费用。
