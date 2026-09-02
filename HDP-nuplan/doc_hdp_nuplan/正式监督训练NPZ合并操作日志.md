# 正式监督训练 NPZ 合并操作日志

日期：2026-08-13

## 1. 目的

将本地两个分支已生成且可用于监督训练的 NuPlan NPZ 合并为一个确定、去重、无 validation 泄漏的训练集。原 cache 不修改，合并目录使用 hardlink，避免复制约 1.9 GB 数据。

当前分支：`diffusion_planner_and_rl`。

对比分支：`hdp-nuplan-scalable-preprocessing-direction-rl`。

两个分支的 `train_predictor.py`、Dataset、监督 train epoch、监督 loss 和 Encoder 文件的 Git 对象完全相同，因此监督训练所需 NPZ schema 相同。direction 分支新增内容不改变监督 Dataset 字段。

## 2. 输入选择

按以下优先级合并：

1. `mini_train_targetmatch_11040_seed3407_v1/cache`；
2. `mini_train_pilot_1000_seed3407_v1/cache`；
3. `mini_train_smoke_100_v2/cache`；
4. `balanced_sampling_smoke_44/cache`。

未纳入 `mini_train_smoke_100_v1`：它是缺少 `log_name/token` 元数据的早期格式，无法完成正式的 train/val 日志审计；其 100 个文件名与 v2 完全一致，所以舍弃 v1 不损失场景。

也没有单独加入 `mini_train_balanced_10000_seed3407_v1`：它的全部 10,000 个文件已经包含在 targetmatch-11,040 中，后者等于原 10,000 加新增 1,040。

validation 排除基准：

```text
tmp/mini_val_balanced_1000_seed3407_v1/cache
```

## 3. 合并规则

- 按 NPZ 文件名和内部 `token` 双重去重；
- 同名文件必须具有相同 token 和 SHA-256，否则立即失败；
- 排除与 validation-1000 共享文件名、token 或 `log_name` 的训练候选；
- 同名场景按输入优先级保留第一份；
- 输出 manifest 排序，保证训练清单可复现；
- 使用 hardlink，原文件和输出文件共享磁盘数据块；
- 输出目录若已存在则拒绝覆盖。

合并工具：

```text
scripts/merge_training_caches.py
```

## 4. 执行命令

工作目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
```

执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/merge_training_caches.py \
  --source targetmatch11040=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache \
  --source pilot1000=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --source smoke100v2=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v2/cache \
  --source smoke44=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/balanced_sampling_smoke_44/cache \
  --validation-cache /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --output-dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_formal_union_12109_seed3407_v1
```

## 5. 合并结果

| 来源 | 输入 | 重复 | 新增到并集 |
|---|---:|---:|---:|
| targetmatch-11,040 | 11,040 | 0 | 11,040 |
| pilot-1,000 | 1,000 | 29 | 971 |
| smoke-v2-100 | 100 | 2 | 98 |
| smoke-44 | 44 | 44 | 0 |
| 合计 | 12,184 | 75 | 12,109 |

75 个跨 cache 同名文件均为字节级一致，没有内容冲突。

最终城市分布：

| 地图 | 场景数 |
|---|---:|
| Las Vegas | 8,609 |
| Pittsburgh | 1,190 |
| Boston | 1,161 |
| Singapore | 1,149 |
| 合计 | 12,109 |

最终训练日志数为 473。

输出目录：

```text
tmp/mini_train_formal_union_12109_seed3407_v1/
├── cache/                              # 12,109 个 hardlink NPZ
├── diffusion_planner_training.json     # 监督训练 manifest
├── sampling_report.json                # 日志/地图/场景类型统计
├── merge_report.json                   # 来源、去重和泄漏排除报告
├── cache_validation_report.json        # 全量 NPZ schema 验收
└── train_val_split_audit.json           # train/validation 防泄漏审计
```

## 6. 全量验收

缓存验收命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/validate_processed_cache.py \
  --cache_dir tmp/mini_train_formal_union_12109_seed3407_v1/cache \
  --manifest tmp/mini_train_formal_union_12109_seed3407_v1/diffusion_planner_training.json \
  --sampling_report tmp/mini_train_formal_union_12109_seed3407_v1/sampling_report.json \
  --expected_count 12109 \
  --expected_log_count 473 \
  --output tmp/mini_train_formal_union_12109_seed3407_v1/cache_validation_report.json
```

结果：

```text
status: passed
manifest: 12,109 unique / sorted
NPZ: 12,109
logs: 473
required keys/shapes: passed
finite values: passed
```

train/validation审计命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/audit_dataset_splits.py \
  --train_manifest tmp/mini_train_formal_union_12109_seed3407_v1/diffusion_planner_training.json \
  --train_report tmp/mini_train_formal_union_12109_seed3407_v1/sampling_report.json \
  --val_manifest tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --val_report tmp/mini_val_balanced_1000_seed3407_v1/sampling_report.json \
  --output tmp/mini_train_formal_union_12109_seed3407_v1/train_val_split_audit.json
```

结果：

```text
train logs: 473
validation logs: 10
overlapping logs: 0
overlapping NPZ: 0
status: passed
```

修复了旧 `audit_dataset_splits.py` 只读取 `log_names` 的兼容问题：早期 sampling report 只提供 `selected_per_log`，现在取两者并集，避免错误显示 validation log count 为 0。

Dataset 实读首、中、尾三个样本均成功：Dataset 长度 12,109，每个样本返回监督入口需要的 11 个数组；自车未来形状 `(80, 3)`，邻车历史原始形状 `(32, 21, 11)`。

相关测试：

```text
7 passed in 2.14s
```

## 7. 文件哈希

```text
diffusion_planner_training.json
ed2a66f8e3acdfb637d1f6defe82c9386d96c3e6ba2663dcbe3eb9e81400aebf

sampling_report.json
0552c06ead04a8ed10c3863130108de36f79e1eafd6128612e0ae412abf8d9f3

merge_report.json
b56c8ef54046cef9a321ef971ac42a0e5dfbdcce0fffa5649c49721f77f3c4bf

cache_validation_report.json
68193ddec2c45c0c8875a5579606241e8accc1c60c9db9326557c0c037b36364

train_val_split_audit.json
e59bbe3b580f04d4a28a76a05542e6f3136bffa7d624c48c2f8e9a570c22eb8d
```

## 8. 正式监督训练参数

训练时使用：

```bash
--train_set \
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_formal_union_12109_seed3407_v1/cache \
--train_set_list \
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_formal_union_12109_seed3407_v1/diffusion_planner_training.json
```

不要把 validation-1000 合入训练。不要再把原 10,000、11,040 或 pilot-1,000 与本目录叠加计算；12,109 已经是上述可审计训练缓存的去重并集。

## 9. 复用本地已生成的官方训练NPZ（2026-08-13）

继续下载前发现以下官方train来源已经完成预处理，但 targetmatch-11,040 只选择了其中一部分：

```text
Singapore processed: 9,450 NPZ
Pittsburgh processed: 1,400 NPZ
Boston processed: 350 NPZ
Boston partial processed: 100 NPZ
```

因此先扩展合并工具：`--source` 目录改为递归查找 `*.npz`，既支持单一cache，也支持 `shard_*/cache` 根目录。以12,109集为基线重新去重合并：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/merge_training_caches.py \
  --source formal12109=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_formal_union_12109_seed3407_v1/cache \
  --source singapore_processed=/home/yanjun/NewDisk/processed/nuplan_train_100k_gate_singapore_local \
  --source pittsburgh_processed=/home/yanjun/NewDisk/processed/nuplan_train_100k_gate_pittsburgh_local \
  --source boston_processed=/home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_source \
  --source boston_partial12_processed=/home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source \
  --validation-cache /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --output-dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_existing_v2
```

结果：

| 来源 | 输入 | 已在12,109中 | 新增 |
|---|---:|---:|---:|
| 12,109基线 | 12,109 | 0 | 12,109 |
| Singapore | 9,450 | 196 | 9,254 |
| Pittsburgh | 1,400 | 422 | 978 |
| Boston | 350 | 350 | 0 |
| Boston partial | 100 | 72 | 28 |
| 最终并集 | 23,409次输入 | 1,040 | **22,369** |

22,369集城市分布：

```text
Singapore: 10,403
Las Vegas: 8,609
Pittsburgh: 2,168
Boston: 1,189
```

全量验收：

```text
manifest/NPZ: 22,369/22,369
unique logs: 1,559
validation logs: 10
overlapping logs: 0
overlapping NPZ: 0
schema/shape/finite values: passed
```

关键哈希：

```text
diffusion_planner_training.json
d7a4326cda011c9ae4f8f4736a6c617624a88b0b319f79e3ed594f894a4517d9

cache_validation_report.json
9f263ab9ce11d259bafe8dc6250774c502840f4a376ec5182f5703734606b75e

train_val_split_audit.json
97bf5cba4cbda59fc0c2556ae8a2dfcf01f6a8ddd67858d736632de07043fdcb
```

正式训练应优先把第8节路径替换为：

```bash
--train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_existing_v2/cache \
--train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_existing_v2/diffusion_planner_training.json
```

## 10. 官方DB滚动下载与增量NPZ流水线（2026-08-13）

### 10.1 设计

采用 `hdp-nuplan-scalable-preprocessing-direction-rl` 提供的滚动工具。为了不触碰当前分支未提交修改，建立detached worktree：

```text
/home/yanjun/NewDisk/Diffusion-Planner-scalable-pipeline
commit: 989a53f
```

定向测试：`22 passed in 2.26s`。

滚动顺序：

```text
一个shard的官方DB下载
→ SQLite/大小/CRC校验
→ 生成NPZ
→ manifest和全部NPZ强校验
→ 归档下载报告
→ 删除该shard原始DB
→ 下一个shard
```

它不会下载完整35.5 GB Boston ZIP，而是利用HTTP Range只下载计划日志对应的ZIP member。磁盘低于20 GiB时会在下一shard前停止。

### 10.2 下载前去重

选择Boston shard 258–263：

```text
shard 258: 50 logs
shard 259: 50 logs
shard 260: 50 logs
shard 261: 50 logs
shard 262: 50 logs
shard 263: 30 logs
合计: 280 logs
```

直接用22,369集的1,559个 `log_name` 审计，以上280个日志与现有训练日志交集均为0；这些shard在100k计划中也互斥。因此下载和生成阶段即避免重复日志，最终合并仍会再次按文件名/token双重去重。

### 10.3 第一次启动失败

第一次没有传 `--only_archive boston`，下载器发现Boston-only索引与默认全城市归档定义不一致，在下载任何DB前报错退出。未生成NPZ，也没有可用DB半成品。失败state和日志保留：

```text
/home/yanjun/NewDisk/nuplan/logs/incremental_boston_258_263_state.json
/home/yanjun/NewDisk/nuplan/logs/incremental_boston_258_263.log
```

### 10.4 修正后的运行命令

```bash
env PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/Diffusion-Planner-scalable-pipeline/HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /home/yanjun/NewDisk/nuplan/indexes/archive_index_train_boston_remote_targetmatch.json \
  --raw_root /home/yanjun/NewDisk/nuplan/raw_incremental_boston \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root /home/yanjun/NewDisk/processed/nuplan_train_incremental_boston \
  --state_path /home/yanjun/NewDisk/nuplan/logs/incremental_boston_258_263_retry1_state.json \
  --shard_indices 258 259 260 261 262 263 \
  --worker_index 0 --worker_count 1 \
  --checksum_mode files --cleanup_raw --resume \
  --connect_timeout_seconds 10 --read_timeout_seconds 60 \
  --max_member_retries 5 --retry_delay_seconds 5 \
  --download_backend curl \
  --curl_low_speed_limit_bps 131072 \
  --curl_low_speed_time_seconds 20 \
  --only_archive boston \
  --min_free_gib 20
```

官方URL：

```text
https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip
```

服务器返回 `Accept-Ranges: bytes` 和 `Content-Length: 38161149300`。修正后已进入 shard 258，第1个DB完整落盘并通过检查，正在下载第2/50个DB。运行日志：

```text
/home/yanjun/NewDisk/nuplan/logs/incremental_boston_258_263_retry1.log
```

流水线完成后，不能直接把“22,369 + 新生成数量”当作最终数量；仍需调用 `merge_training_caches.py`，以22,369为第一来源、incremental Boston为第二来源，再执行全量validation审计后生成新的正式manifest。

## 11. 基于当前 35,869 个正式 NPZ 重新训练无 detach 监督模型（2026-08-14）

### 11.1 目标和无 detach 边界

本次重新训练使用截至 `shard_00032` 暂停时全部正式可用 NPZ，并明确不启用监督混合轨迹
损失中的 Detached Integral：

```text
planning_detach_window_size=0
前向：predicted_xy = torch.cumsum(predicted_displacement)
反向：每个位置损失可向此前全部 displacement 传播梯度
```

这里只关闭积分轨迹 loss 的 stop-gradient。数据加载、optimizer 清梯度、checkpoint 保存以及日志
张量转标量等正常训练机制不属于该 detach，保持不变。

代码调整：

- `hdp_nuplan/loss.py`：去掉写死的 `detach_window_size=10`，增加可配置参数，默认 `0`；
- `hdp_nuplan/train_epoch.py`：把监督入口的 `args.planning_detach_window_size` 传入 loss；
- `train_predictor.py`：新增 `--planning_detach_window_size`，默认 `0`，负数拒绝运行；
- `hdp_nuplan/rl/train_epoch_rl.py`：expert anchor 显式沿用
  `args.rl_detach_window_size`，避免 RL 主 loss 与 anchor 的窗口不一致。

回归测试：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q tests
```

结果：`40 passed, 15 warnings in 5.75s`；warning 均为既有第三方依赖弃用提示。

### 11.2 合并 35,869 个正式训练 NPZ

来源：

| 来源 | NPZ |
|---|---:|
| 已有正式并集 | 22,369 |
| Boston 新完成 shard 00258～00259 | 700 |
| Las Vegas 新完成 shard 00001～00032 | 12,800 |
| 合计 | 35,869 |

执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/merge_training_caches.py \
  --source formal22369=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_existing_v2/cache \
  --source boston700=/home/yanjun/NewDisk/processed/nuplan_train_incremental_boston \
  --source vegas12800=/home/yanjun/NewDisk/processed/nuplan_train_incremental_vegas1 \
  --validation-cache /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --output-dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1
```

合并采用 hardlink，不复制 NPZ 内容。结果：

```text
selected=35,869
unique filename=35,869
unique token=35,869
duplicate=0
excluded validation=0
unique train logs=3,148
```

城市分布：

```text
Las Vegas: 21,409
Singapore: 10,403
Pittsburgh: 2,168
Boston: 1,889
```

### 11.3 全量训练前门禁

`validate_processed_cache.py` 全量打开并检查 35,869 个 NPZ。结果：

```text
status=passed
manifest/unique manifest/NPZ=35,869/35,869/35,869
log count=3,148
schema/shape/finite values=passed
total logical bytes=5,648,579,059
```

与固定 validation-1000 审计：

```text
train logs=3,148
validation logs=10
overlapping logs=0
overlapping NPZ=0
status=passed
```

Dataset 首项、中间项、末项均成功读取 11 个数组；`ego_future=(80,3)`，
`neighbors_past=(32,21,11)`。

关键文件 SHA-256：

```text
diffusion_planner_training.json  73dc2a359be7dbe39b24da23f9c2304736107b6d3b7d3b421ef36fb67b9352b4
sampling_report.json             ed52e4bb68e85c819cfa82d4a4cd3bde228b87333c2174742cb364371192c490
merge_report.json                2242fad383812e5956a121198ba0db9f213d6e04e72e1776ff543902bb4a7895
cache_validation_report.json     754d1730f9dd26a2bff3018b8877040a2d0bb46907026e69137e8d17c258325f
train_val_split_audit.json       b279687f3a140af05d4a672ccb1a7aab70c068cb483cc8f9eddb0ef014972210
```

### 11.4 正式监督训练启动

本机资源为 RTX 4060 Laptop 8 GB。沿用已经验证的 HDP 监督配方：`omega=0.1`、20 epochs、
学习率 `5e-4`、encoder warm-start、前 3 epochs 冻结 encoder；本次显式加入无 detach 参数。

系统没有安装 `screen`，因此使用 `setsid` 后台运行，避免当前终端结束时终止训练：

```bash
train_log=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/supervised_formal35869_nodetach_train.log
setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-paper-supervised-formal35869-omega01-nodetach \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 3 \
  --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
  --warm_up_epoch 2 --save_utd 1 --num_workers 4 \
  --planning_hybrid_loss 0.1 \
  --planning_detach_window_size 0 \
  --seed 3407 --use_wandb false \
  > "$train_log" 2>&1 < /dev/null &
```

启动证据：

```text
launcher PID=161515
Dataset Prepared=35,869
batch size=8
batches/epoch=4,483
encoder warm-start=151/151 tensors, 1,799,040 parameters
decoder_loaded=0
model parameters=5,092,996
epoch 1 encoder_trainable=False
args.json planning_detach_window_size=0
首批无 OOM/NaN，约 17～18 batch/s
```

运行目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/training_log/
hdp-paper-supervised-formal35869-omega01-nodetach/2026-08-14-16:17:33/
```

每个 epoch 保存 checkpoint。后台日志：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/
supervised_formal35869_nodetach_train.log
```

### 11.5 第一个 epoch 和可恢复性验收

第 1 epoch 完整遍历 `4,483` 个 batch 后正常结束：

```text
epoch 1 mean total loss=2.6595
checkpoint=model_epoch_1_trainloss_2.6595.pth
checkpoint size=67,417,699 bytes
```

使用 CPU 独立加载 checkpoint，确认包含：

```text
model：260 项
ema_state_dict：260 项
optimizer：存在
schedule：存在
epoch=1
loss=2.6595
```

因此该 checkpoint 不只是推理权重，也具备完整恢复训练所需的模型、EMA、optimizer 和学习率
调度状态。检查完成时已进入 epoch 2，约 `293/4483`，launcher PID `161515` 仍存活。正式训练
继续在后台执行，计划完成 20 epochs。

### 11.6 正式监督训练完成

完成时间：2026-08-14 18:32:54 CST。

```text
epochs=20/20
batches per epoch=4,483
checkpoint count=20
final epoch loss=1.0615
最低训练 loss=1.0458（epoch 18）
NaN/OOM/Traceback=0
后台训练进程=已正常退出
```

最终 checkpoint：

```text
training_log/hdp-paper-supervised-formal35869-omega01-nodetach/
2026-08-14-16:17:33/model_epoch_20_trainloss_1.0615.pth
```

按训练 loss 得到的候选 checkpoint：

```text
training_log/hdp-paper-supervised-formal35869-omega01-nodetach/
2026-08-14-16:17:33/model_epoch_18_trainloss_1.0458.pth
```

训练 loss 最低不等于规划指标最好。下一步应至少比较 epoch 18、19、20 在固定 validation 集上的
开环指标，再对候选执行相同场景、相同 seed 的 NuPlan 闭环评测后确定监督基线。

## 12. 35,869 NPZ 无 detach 监督模型评测（2026-08-14）

### 12.1 评测原则

本次不根据训练 loss 直接选 epoch，而是按以下顺序评测：

1. 在固定 `mini_val_balanced_1000_seed3407_v1` 上对 20 个 EMA checkpoint 各评测 1 次；
2. 对 validation loss 前 3 名分别用 seed `3407/3408/3409` 重复 3 次；
3. 前 3 名先跑相同的固定 3 场景闭环，排除安全性明显失败的候选；
4. 对通过门禁的候选跑固定 20 场景闭环，与旧 10k epoch 10 和原
   Diffusion-Planner 使用完全相同的场景比较。

加载的是 checkpoint 中的 `ema_state_dict`。开环数值越低越好，闭环数值越高越好。

### 12.2 validation-1000 全 checkpoint 扫描

核心命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python evaluate_checkpoints.py \
  --args_file tmp/nuplan_train_formal_union_35869_seed3407_v1/training_log/hdp-paper-supervised-formal35869-omega01-nodetach/2026-08-14-16:17:33/args.json \
  --checkpoint_dir tmp/nuplan_train_formal_union_35869_seed3407_v1/training_log/hdp-paper-supervised-formal35869-omega01-nodetach/2026-08-14-16:17:33 \
  --data_dir tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_training.json \
  --batch_size 16 --num_workers 2 --repeats 1 --seed 3407 --device cuda \
  --output tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_ranking_formal35869_nodetach_repeat1.json
```

20 个 epoch 的单次 total validation loss：

| Epoch | Loss | Epoch | Loss | Epoch | Loss | Epoch | Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.032699 | 6 | 0.588699 | 11 | 0.521764 | 16 | 0.464141 |
| 2 | 0.586128 | 7 | 0.539132 | 12 | 0.488965 | 17 | 0.486745 |
| 3 | 0.487400 | 8 | 0.479246 | 13 | 0.484495 | 18 | 0.460410 |
| 4 | 0.645883 | 9 | 0.482824 | 14 | 0.486511 | 19 | 0.462515 |
| 5 | 0.564559 | 10 | 0.566800 | 15 | **0.455088** | 20 | 0.465074 |

前 3 名重复 3 次后：

| Checkpoint | 三次 total loss | 均值 | planning loss | hybrid loss |
|---|---|---:|---:|---:|
| epoch 15 | 0.455088 / 0.460234 / 0.537979 | **0.484433** | 0.029991 | 4.544429 |
| epoch 18 | 0.460410 / 0.460034 / 0.554384 | 0.491609 | 0.030121 | 4.614881 |
| epoch 19 | 0.462515 / 0.461255 / 0.560259 | 0.494676 | 0.029703 | 4.649734 |

因此 epoch 15 是开环第一，epoch 18 和 epoch 19 作为备选。第 3 次都明显高于前两次，
说明扩散采样仍有随机方差，不应仅依赖一次开环结果选模型。

### 12.3 固定 3 场景闭环门禁

通过 `scripts/run_mini_closed_loop.sh` 分别运行 epoch 15、18、19：

```bash
bash scripts/run_mini_closed_loop.sh \
  hdp-formal35869-nodetach-epoch18-mini-val-3 \
  <training-run>/args.json \
  <training-run>/model_epoch_18_trainloss_1.0458.pth \
  mini-val-closed-loop-3 hdp
```

| 模型 | NuPlan score | Collision | TTC | Progress | 结论 |
|---|---:|---:|---:|---:|---|
| 旧 10k epoch 10 | 0.947107 | 1.000000 | 1.000000 | 0.830792 | 参考基线 |
| 35,869 epoch 15 | 0.589614 | 0.666667 | 0.666667 | 0.694111 | 安全门禁失败 |
| 35,869 epoch 18 | **0.873345** | 1.000000 | 1.000000 | 0.594762 | 通过 |
| 35,869 epoch 19 | 0.869441 | 1.000000 | 1.000000 | 0.582265 | 通过 |

epoch 15 虽然开环最佳，但场景 `134f95eed3775e22` 同时发生 collision/TTC 失败，
因此被排除。epoch 18 在通过安全门禁的候选中得分略高于 epoch 19，送入 20 场景。

### 12.4 固定 20 场景闭环结果

epoch 18 完成 `20/20` 个 simulation，`failed_simulations=0`，总耗时约 15 分 24 秒。
运行期间 3 次 `All route_list elements are empty` 是 NuPlan route extractor 警告，对应场景仍正常完成，
不是 runner 失败。

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC | Direction | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diffusion-Planner | **0.881446** | 0.900000 | 0.950000 | 1.000000 | **0.960774** | 0.900000 | 1.000000 | 0.950000 |
| 旧 10k epoch 10 | 0.787373 | 0.950000 | **1.000000** | 0.900000 | 0.668771 | 0.900000 | 0.975000 | **1.000000** |
| 35,869 epoch 18 无 detach | 0.715602 | 0.950000 | 0.950000 | 0.900000 | 0.588029 | **0.950000** | **1.000000** | **1.000000** |

35,869 epoch 18 相对旧 10k epoch 10：

```text
score:          -0.071771（约 -9.1%）
route progress: -0.080741
TTC:            +0.050000
direction:      +0.025000
drivable:       -0.050000
collision:       0
comfort:         0
场景级得分：2 胜 / 11 负 / 7 平
```

新模型的 4 个零分/主要失败场景：

| Token | 类型 | 主要原因 |
|---|---|---|
| `08dd79a302275f09` | traversing_intersection | drivable area=0 |
| `2da316803b1d561d` | traversing_traffic_light_intersection | progress 过低（0.03525） |
| `53dcab93303a5a9d` | on_pickup_dropoff | progress 过低（0.10939） |
| `c016e450b2e55365` | high_magnitude_speed | collision=0、TTC=0 |

### 12.5 结论与边界

本次 35,869 NPZ、`omega=0.1`、无 detach 监督实验已经完整训练和评测，但未产生优于旧
10k epoch 10 的监督基线。闭环 TTC 有改善，但不足以抵消 route progress 和 drivable area 的退化；
因此暂不将 epoch 18 晋升为项目的主基线，旧 10k epoch 10 仍保留为当前闭环基线。

该对比同时变更了训练数据数量/分布、hybrid loss 权重和按 step 计算的优化日程，所以
不能仅根据当前结果归因为“NPZ 变多”或任意单一因素。而且 20 场景只适合做候选筛选，
不是具有统计稳健性的官方大规模结论。

后续如果要定位原因，应在相同 35,869 NPZ 上只做单变量对照：先保持无 detach，把
`planning_hybrid_loss` 调回旧 10k 配方的 `0.01`，其他设置不变；如仍退化，再单独比较数据采样分布。

### 12.6 产物与 SHA-256

```text
checkpoint_ranking_formal35869_nodetach_repeat1.json
546b972587a14f173e628a6b9135187341d0a1ba2ce5bf2ef0c1881db565ee2f

checkpoint_evaluation_formal35869_nodetach_epoch15_repeat3.json
57e6ac4e54e2bb730aaa8a52ce1fd7fd4ff9ec03cb6a7177183b7e27745942b7

checkpoint_evaluation_formal35869_nodetach_epoch18_repeat3.json
ffb7bb889c328793b404ff99f8651c7206e082f05548c7e73fd32695d8b3cf2d

checkpoint_evaluation_formal35869_nodetach_epoch19_repeat3.json
32556c6ff6ff9d4b8c1c95cd3de6c6b5d954ee510be15f10c7f7e2b26b55869c

mini_val_3_hdp10k_vs_formal35869_top3.json
dcef974091c64920ecb020d20a47c229427b10cb2e9ddfa218d3b98382861a13

mini_val_20_diffusion_hdp10k_vs_formal35869_nodetach.json
953c0baca460ea3826d0f2ac591d7784dee0ab71ee796b1ac5513953572d17c0
```

### 12.7 旧 10k epoch 10 闭环表现较好的原因分析

首先需要纠正：新模型不是开环轨迹拟合更差。在同一 validation-1000 上：

| 模型 | Diffusion loss | Hybrid trajectory loss | 训练时 hybrid 权重 | 加权 total loss |
|---|---:|---:|---:|---:|
| 旧 10k epoch 10 | 0.041313 | 7.794738 | 0.01 | 0.119260 |
| 35,869 epoch 18 | **0.030121** | **4.614881** | 0.1 | 0.491609 |

新模型的两个未加权子项分别下降约 `27.1%` 和 `40.8%`。它的 total loss 更大主要是因为
hybrid 权重扩大了 10 倍，因此 `0.119260` 和 `0.491609` 不能直接当作同一标尺比较。

旧 10k 的优势出现在闭环行为，目前有以下已确认差异：

1. **训练分布更接近当前 mini-val。** 旧 10k 从 44 个 mini-train 日志中每日志均衡抽取
   `227～229` 条；新 35,869 是 3,148 个日志的去重并集，每日志 `7～268` 条，不是日志均衡抽样。
2. **城市比例更匹配。** val1k 是 Singapore/Boston/Las Vegas/Pittsburgh =
   `10%/10%/70%/10%`；旧 10k 是 `9.08%/6.82%/77.28%/6.82%`，新 35,869 是
   `29.00%/5.27%/59.69%/6.04%`。以城市比例的 total variation distance 衡量，旧数据为 `7.28%`，
   新数据为 `19.00%`。
3. **Hybrid loss 权重不同。** 旧 10k 使用 `planning_hybrid_loss=0.01`，新模型使用
   `0.1`，因此新配方对累积位置误差的约束强 10 倍。
4. **两次监督训练在功能上都没有有效 detach。** 旧 loss 确实硬编码传入
   `detach_window_size=10`，但当时 `detached_integral()` 使用
   `shifted[:, :, :detach_window_size] = 0`。对实际输入 `[B,T,2]` 来说，该切片清零的是
   最后一个特征维，且窗口 10 大于特征数 2，所以两个 shifted 张量被整体清零，
   返回值及梯度都退化为普通完整 `torch.cumsum`。该问题于 2026-08-08 审计发现，
   2026-08-09 提交修复；旧 10k checkpoint 在 2026-08-01 已经训练完成，所以修复不会追溯改变它。
   新 35,869 训练则显式使用 `detach_window_size=0`。因此 detach 不是两个 checkpoint 性能差异的有效变量。
5. **相同 epoch 不等于相同优化步数。** batch size 都是 8，但旧 10k 的 20 epochs 约为 `25,000`
   个 optimizer step，新 35,869 约为 `89,660` 个 step。学习率和 warm-up 仍按相同 epoch 设置，
   实际的按 step 优化日程并不相同。
6. **旧 10k 也不是全面最优。** 在 20 场景中它的 score 为 `0.787373`，仍低于发布版
   Diffusion-Planner 的 `0.881446`，且 TTC 为 `0.9`。当前只能说它在这组固定的小规模
   mini-val 场景上比新模型更好，不能推广为对官方全量 validation 也必然更好。

基于已有证据，最可能的解释是：旧 10k 对当前 mini-val 分布的匹配度更高，同时较弱的
hybrid 约束可能对闭环进度更友好。新模型虽然开环轨迹误差更小，但在闭环中的
route progress 从 `0.668771` 降到 `0.588029`，这才是总分下降的主要表现。

上述“哪一项是主因”目前仍不能确定，因为两次实验同时改变了数据、loss 权重和优化步数。
需要在 35,869 数据上先做 `omega=0.01, detach=0` 的单变量对照；这不是在模仿一个有效的
旧 detach 窗口，而是在保持两者都为完整 cumsum 梯度的前提下，只对齐 hybrid 权重。

## 13. 35,869 NPZ：omega=0.01 无 detach 单变量对照（2026-08-14）

### 13.1 实验目的和唯一变量

目标是判断前一轮 35,869 NPZ 模型的闭环退化是否主要来自过强的 hybrid trajectory loss。
与第 11 节的 `omega=0.1` 训练相比，只修改：

```text
planning_hybrid_loss: 0.1 -> 0.01
```

以下配置保持不变：

```text
train NPZ=35,869
planning_detach_window_size=0
seed=3407
train_epochs=20
batch_size=8
learning_rate=5e-4
warm_up_epoch=2
encoder_pretrained_model_path=checkpoints/model.pth
freeze_encoder_epochs=3
normalization=normalization.json
```

这里的 `detach=0` 是显式的完整 `torch.cumsum` 梯度。旧 10k checkpoint 虽然调用时传入
`detach_window_size=10`，但旧实现的维度切片错误使其实际也退化为完整 `cumsum` 梯度。
因此本轮是在保持无有效 detach 的前提下，只对齐 hybrid 权重。

### 13.2 启动前门禁

```text
GPU=RTX 4060 Laptop 8 GB
GPU memory used=15 MiB，无冲突训练进程
NewDisk available=52 GB
manifest count=35,869
cache/manifest 首末样本=存在
同名实验目录=不存在
```

### 13.3 实际启动命令

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-paper-supervised-formal35869-omega001-nodetach \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 3 \
  --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
  --warm_up_epoch 2 --save_utd 1 --num_workers 4 \
  --planning_hybrid_loss 0.01 \
  --planning_detach_window_size 0 \
  --seed 3407 --use_wandb false \
  > /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/supervised_formal35869_omega001_nodetach_train.log \
  2>&1 < /dev/null &
```

### 13.4 启动验收

启动时间为 2026-08-14 21:58:11 CST，后台 launcher PID 为 `191669`。运行目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/
training_log/hdp-paper-supervised-formal35869-omega001-nodetach/2026-08-14-21:58:11/
```

`args.json` 二次核对结果：

```text
Dataset Prepared=35,869
planning_hybrid_loss=0.01
planning_detach_window_size=0
encoder warm-start=151/151 tensors，1,799,040 parameters
decoder_loaded=0
model parameters=5,092,996
epoch 1 encoder_trainable=False
```

首次验收时 epoch 1 已运行约 `769/4483` batch，速度约 `23 batch/s`，未发现 NaN、OOM、
Traceback 或进程异常。后台日志：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/
supervised_formal35869_omega001_nodetach_train.log
```

训练完成后不直接选最后一轮；仍将执行 20 checkpoint 的固定 val1k 扫描、前 3 名三次复测、
3 场景闭环门禁和 20 场景闭环对比。跨 `omega` 实验不直接比较加权 total loss，而比较
未加权的 diffusion/hybrid 子项和相同场景的闭环指标。

### 13.5 训练完成结果

训练于 2026-08-14 23:28:06 CST 正常完成，后台进程已自然退出。启动至最后一个 checkpoint
落盘约用时 1 小时 30 分钟。

```text
epochs=20/20
batches per epoch=4,483
checkpoint count=20
NaN/OOM/Traceback/Killed=0
最低训练 total loss=0.1442（epoch 18）
最终训练 total loss=0.1453（epoch 20）
```

每个 epoch 的训练 total loss：

| Epoch | Loss | Epoch | Loss | Epoch | Loss | Epoch | Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3979 | 6 | 0.1991 | 11 | 0.1653 | 16 | 0.1499 |
| 2 | 0.2709 | 7 | 0.1904 | 12 | 0.1613 | 17 | 0.1486 |
| 3 | 0.1767 | 8 | 0.1849 | 13 | 0.1607 | 18 | **0.1442** |
| 4 | 0.2448 | 9 | 0.1801 | 14 | 0.1569 | 19 | 0.1450 |
| 5 | 0.2181 | 10 | 0.1719 | 15 | 0.1533 | 20 | 0.1453 |

epoch 18 和 epoch 20 checkpoint 都通过 CPU 独立加载检查：

```text
model items=260
ema_state_dict items=260
optimizer=存在
schedule=存在
epoch/loss 元数据=与文件名一致
```

关键产物 SHA-256：

```text
args.json
b99a3fe68e9d83f160ac573bd1992096749008f51b80b5a38ab41ef98686e33c

model_epoch_18_trainloss_0.1442.pth
1a617fc64036fdbfc81101aefc646c3a6e233e9784884e6731292328b109015c

model_epoch_20_trainloss_0.1453.pth
f55c9c4a34e1337cac14dac0507cf6fd27a9e0d1828f31431029af4201f454a9
```

这些是训练集上的加权 total loss，仅能用于本次 `omega=0.01` 运行内部观察收敛，不能与
`omega=0.1` 运行的 `1.0458` 直接判定优劣，因为 total loss 中 hybrid 项的权重相差 10 倍。
训练 loss 最低的 epoch 18 只是候选，正式 checkpoint 仍需通过固定 val1k 与闭环评测决定。

### 13.6 固定 val1k 全 checkpoint 扫描（2026-08-16）

使用与前两轮完全相同的 validation-1000、batch size 16、seed 3407 和 EMA 加载规则，
先对 20 个 checkpoint 各评测 1 次：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python evaluate_checkpoints.py \
  --args_file tmp/nuplan_train_formal_union_35869_seed3407_v1/training_log/hdp-paper-supervised-formal35869-omega001-nodetach/2026-08-14-21:58:11/args.json \
  --checkpoint_dir tmp/nuplan_train_formal_union_35869_seed3407_v1/training_log/hdp-paper-supervised-formal35869-omega001-nodetach/2026-08-14-21:58:11 \
  --pattern 'model_epoch_*.pth' \
  --data_dir tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 16 --num_workers 2 --repeats 1 --seed 3407 --device cuda \
  --output tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_ranking_formal35869_omega001_nodetach_repeat1.json
```

单次 val total loss：

| Epoch | Loss | Epoch | Loss | Epoch | Loss | Epoch | Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.151561 | 6 | 0.070690 | 11 | 0.065225 | 16 | 0.066771 |
| 2 | 0.082007 | 7 | 0.075660 | 12 | 0.065544 | 17 | 0.062848 |
| 3 | 0.069118 | 8 | 0.069105 | 13 | 0.070231 | 18 | **0.057157** |
| 4 | 0.104267 | 9 | 0.068328 | 14 | 0.066636 | 19 | 0.064650 |
| 5 | 0.079104 | 10 | 0.066954 | 15 | 0.069041 | 20 | 0.067597 |

前 3 名是 epoch 18、17、19。用 seed `3407/3408/3409` 各重复 3 次后：

| Epoch | 三次 total loss | 均值 | Diffusion loss | Hybrid loss |
|---:|---|---:|---:|---:|
| 18 | 0.057157 / 0.067317 / 0.073985 | **0.066153** | **0.023637** | **4.251637** |
| 17 | 0.062848 / 0.064518 / 0.073754 | 0.067040 | 0.024279 | 4.276047 |
| 19 | 0.064650 / 0.065166 / 0.075347 | 0.068388 | 0.024841 | 4.354716 |

epoch 18 在三次均值和两个未加权子项上都是第一，因此是稳定的开环候选。相对同数据
`omega=0.1` epoch 18，它的 Diffusion loss 下降 `21.53%`，Hybrid loss 下降 `7.87%`。
相对同为 `omega=0.01` 的旧 10k epoch 10，total val loss 从 `0.119260` 降至 `0.066153`，
下降 `44.53%`。这证明新模型的开环拟合更好，但不能代替闭环验证。

### 13.7 固定 3 场景闭环门禁

第一次 epoch 17 命令把 `args.json` 和 checkpoint 传为相对路径。Shell 的启动前文件检查通过，
但 Hydra 进入运行目录后 `Config` 无法解析该相对路径，在构建 planner 时报 `FileNotFoundError`。
该尝试没有启动 simulation，也没有修改 checkpoint。之后改用绝对路径，并使用
`epoch17-mini-val-3-retry1` 新实验名保留失败尝试供审计。

三个候选都完成 3/3 simulation、0 runner failure、Collision/TTC/Drivable/Direction/Comfort 均为 1：

| 模型 | Score | Route progress | 门禁结论 |
|---|---:|---:|---|
| 旧 10k epoch 10 | 0.947107 | 0.830792 | 参考 |
| 35,869 omega=0.1 epoch 18 | 0.873345 | 0.594762 | 参考 |
| 35,869 omega=0.01 epoch 17 | 0.893137 | 0.658094 | 通过 |
| 35,869 omega=0.01 epoch 18 | **0.912050** | **0.718615** | 通过，送入 20 场景 |
| 35,869 omega=0.01 epoch 19 | 0.595117 | 0.461327 | 通过安全项，但进度差 |

epoch 18 相对同数据 `omega=0.1` 在 3 场景上得分提升约 `4.43%`，route progress 提高
`0.123853`。但 3 场景只是候选门禁，不能作为最终结论。

### 13.8 固定 20 场景闭环最终结果

epoch 18 使用 `mini-val-closed-loop-20` 完成 20/20 simulation，0 runner failure，主体仿真用时
15 分 01 秒。中途出现 3 次已知 `All route_list elements are empty` 警告，相应场景均正常完成；
警告数与先前的同场景评测一致。

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC | Direction | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diffusion-Planner | **0.881446** | 0.900000 | 0.950000 | **1.000000** | **0.960774** | 0.900000 | 1.000000 | 0.950000 |
| 旧 10k epoch 10 | 0.787373 | **0.950000** | **1.000000** | 0.900000 | 0.668771 | 0.900000 | 0.975000 | **1.000000** |
| 35,869 omega=0.1 epoch 18 | 0.715602 | **0.950000** | 0.950000 | 0.900000 | 0.588029 | **0.950000** | 1.000000 | **1.000000** |
| 35,869 omega=0.01 epoch 18 | 0.593706 | 0.850000 | 0.850000 | 0.850000 | 0.645880 | 0.750000 | 1.000000 | **1.000000** |

`omega=0.01` 相对同数据 `omega=0.1`：

```text
score:          -0.121896（-17.03%）
route progress: +0.057851（+9.84%）
collision:      -0.100000
TTC:            -0.200000
drivable:       -0.100000
progress gate:  -0.050000
场景级得分：5 胜 / 6 负 / 9 平
```

相对旧 10k epoch 10，`omega=0.01` 新模型为 `2 胜 / 11 负 / 7 平`，平均得分下降
`0.193667`。相对 Diffusion-Planner 为 `0 胜 / 13 负 / 7 平`。

新模型的主要失败场景：

| Token | 类型 | 主要失败 |
|---|---|---|
| `08dd79a302275f09` | traversing_intersection | drivable=0、TTC=0 |
| `1fb0bb88f9d35d59` | following_lane_without_lead | drivable=0 |
| `2da316803b1d561d` | traversing_traffic_light_intersection | progress gate=0 |
| `36b0447fd8235d0c` | traversing_crosswalk | drivable=0、progress gate=0 |
| `53dcab93303a5a9d` | on_pickup_dropoff | collision=0、TTC=0、progress gate=0 |
| `b2c0d1c5589e540f` | following_lane_with_slow_lead | collision=0、TTC=0 |
| `c016e450b2e55365` | high_magnitude_speed | collision=0、TTC=0 |
| `e88fc19066bd5a43` | traversing_traffic_light_intersection | TTC=0 |

### 13.9 单变量实验结论

`planning_hybrid_loss=0.01` **没有为 35,869 NPZ 模型带来闭环正收益**。它确实提高了平均
route progress，也进一步降低了开环误差，但约束过弱导致 collision、TTC 和 drivable area 同时
退化，最终 score 比 `omega=0.1` 低 `17.03%`。因此不晋升该 checkpoint，当前 35,869 数据
的较安全基线仍是 `omega=0.1` epoch 18，项目总基线仍是旧 10k epoch 10。

这次结果还说明：

1. val1k 开环 loss 显著更低不保证闭环更安全；
2. 3 场景门禁可排除明显失败，但会漏掉低频安全问题；
3. `omega=0.01` 和 `0.1` 之间存在进度/安全折中，如果继续调权重，应优先尝试中间值，
   但必须把 20 场景安全指标作为门禁，不能再只根据开环 loss 或 3 场景决策。

本轮产物 SHA-256：

```text
checkpoint_ranking_formal35869_omega001_nodetach_repeat1.json
e4f5ce8a62d99367bd65035d2ba07d048eeadb6a99d05ee76eaf9dbc29ce1e9e

checkpoint_evaluation_formal35869_omega001_nodetach_epoch17_repeat3.json
cc70f14b660745ab0c24744e4534e04b387702fd5c21e3ea3cf683ca2e18017b

checkpoint_evaluation_formal35869_omega001_nodetach_epoch18_repeat3.json
388b1833541df7683f258e0b6ac9d02021dd34e5389daa7352153539d9c4fdf0

checkpoint_evaluation_formal35869_omega001_nodetach_epoch19_repeat3.json
6757b336249d3bd5b840b059527af69ec6334d54612c65211496193d96e78251

mini_val_3_old10k_omega01_vs_omega001_top3.json
596a940aa4589559fa1476b2173be90e987e8380a56dd713edb5d4e940362393

mini_val_20_diffusion_old10k_omega01_vs_omega001.json
f9d2e26c1b6367bd3bc58e1b23bd59f2d1480ab9adee72ea078ed1b33494e38c
```

### 13.10 为什么旧 10k epoch 10 在当前闭环评测中更好

本节区分“已验证事实”和“待消融推断”，避免把相关性直接当成因果。

#### 已验证事实 1：旧 10k 的地域分布更贴近这 20 个场景

通过 mini DB 的 `log.location/map_version` 字段核对，固定 20 场景来自 8 个日志：

```text
Las Vegas=14/20=70%
Boston=3/20=15%
Singapore=3/20=15%
Pittsburgh=0/20=0%
```

训练集地域比例：

| 数据集 | Las Vegas | Boston | Singapore | Pittsburgh | 相对闭环20的 TV distance |
|---|---:|---:|---:|---:|---:|
| 旧 10k | 77.28% | 6.82% | 9.08% | 6.82% | **14.10%** |
| 新 35,869 | 59.69% | 5.27% | 29.00% | 6.04% | 20.04% |

旧 10k 还对 44 个 mini-train 日志做了 `balanced_logs`，每日志 `227～229` 条；新 35,869
是 3,148 个官方 train 日志的去重并集，每日志 `7～268` 条。因此旧 10k 是更强的
mini 域内专用数据，新数据则覆盖更广、但与当前小评测集的分布匹配更差。

这不是 train/val 泄漏：旧 10k 使用官方 44 个 mini-train 日志，val1k/闭环使用官方
mini-val 日志，日志列表相互隔离。

#### 已验证事实 2：开环误差与 NuPlan 闭环安全不同目标

新 35,869、`omega=0.01` epoch 18 在同一 val1k 上的 total loss 为 `0.066153`，比旧 10k
epoch 10 的 `0.119260` 低 `44.53%`，但 20 场景闭环分数反而从 `0.787373` 降到
`0.593706`。

原因是监督 loss 只比较数据集中专家未来轨迹的运动增量和积分位置，不直接监督：

```text
无责碰撞
TTC
drivable area
闭环 route progress
车辆执行后的分布偏移
```

模型在离线专家状态上的平均误差更小，不代表在逐步重规划后遇到自己诱导出的偏离状态时仍安全。

#### 已验证事实 3：NuPlan 乘法门禁会放大少数安全失败

`closed_loop_nonreactive_agents_weighted_average` 先对 route progress、TTC、speed limit、comfort 等做加权
平均，再乘以以下门禁：

```text
no_ego_at_fault_collisions
drivable_area_compliance
ego_is_making_progress
driving_direction_compliance
```

某个布尔门禁为 0 时，即使其他指标很高，整个场景 score 也会变成 0。固定 20 场景中：

| 模型 | 零分场景 | score<0.5 场景 |
|---|---:|---:|
| Diffusion-Planner | 2 | 2 |
| 旧 10k epoch 10 | 2 | 4 |
| 35,869 omega=0.1 epoch 18 | 4 | 4 |
| 35,869 omega=0.01 epoch 18 | 7 | 7 |

因此旧 10k 的优势不是每个场景都领先，而是它更少触发会把整个场景归零的硬门禁。

#### 已验证事实 4：不能将优势归因于 detach

旧 10k 虽然调用时硬编码 `detach_window_size=10`，但旧切片 bug 使其实际退化为完整
`torch.cumsum` 梯度。新实验显式使用 `detach_window_size=0`，功能上也是完整 `cumsum`。
因此 detach 不是旧 10k 更好的解释。

#### 目前最合理、但仍需消融的推断

1. **目标域专用化。** 旧 10k 对 mini 日志和 Las Vegas 占比更集中，在当前 mini-val 小域上形成
   了更有利的归纳偏置；新 35,869 需要同时拟合更多城市、地图和行为分布。
2. **数据平衡方式。** 新数据是去重并集而不是按城市/日志/场景类型分层平衡采样，有效目标分布
   可能被大量容易或不匹配样本稀释。
3. **优化日程。** 同为 20 epochs 时，新训练约有 3.6 倍 optimizer step，但沿用同一按 epoch 的
   warm-up 和 cosine 日程。这是待对照变量，不是已证明主因。
4. **小样本方差。** 20 场景足以发现当前安全退化，但不足以证明旧 10k 在官方全量 validation
   上普遍更强；它的 `0.787373` 仍低于 Diffusion-Planner 的 `0.881446`。

综合而言，旧 10k epoch 10 在当前评测中更好，最直接的原因是**数据分布更贴近评测域，
并且更少触发 NuPlan 乘法安全门禁**。“更多 NPZ 必然更好”在目标分布不匹配、监督目标与
闭环指标不一致时不成立。

### 13.11 旧 10k 与新 35,869 的训练/推理流程是否只差数据集

结论分为两部分：

```text
监督训练：不是逐代码版本、逐优化步完全一样，不能说只差训练集。
闭环推理评测：本次对比使用同一份当前推理代码和相同协议，是受控的。
```

#### 训练配置对比

两份 `args.json` 共有关键训练参数均一致：

```text
train_epochs=20
batch_size=8
learning_rate=5e-4
warm_up_epoch=2
freeze_encoder_epochs=3
encoder_pretrained_model_path=checkpoints/model.pth
planning_hybrid_loss=0.01
seed=3407
use_ema=true
diffusion_model_type=x_start
diffusion_supervision_type=x_start
future_len=80
agent_num=32
predicted_neighbor_num=10
num_workers=4
```

两份 encoder warm-start report 也完全一致：都从发布版 checkpoint 的 `ema_state_dict` 加载
`151/151` 个 encoder 张量、`1,799,040` 个参数，且 `decoder_loaded=0`。

`args.json` 的实际差异只有：

```text
name/save_dir/train_set/train_set_list 路径
旧实验没有 planning_detach_window_size 字段
新实验显式 planning_detach_window_size=0
```

最后一项在功能上不构成差异：旧实现虽硬编码传 10，但切片 bug 使梯度退化为完整
`cumsum`；新实验传 0 也是完整 `cumsum`。

#### 模型结构与 checkpoint 对比

对旧 epoch 10 和新 epoch 18 的 `model`/`ema_state_dict` 逐键检查：

```text
张量项：260 vs 260
键集合：完全相同
张量 shape：0 个不同
参数量：5,092,996 vs 5,092,996
checkpoint 顶层字段：都有 model/ema_state_dict/optimizer/schedule/epoch/loss/wandb_id
```

因此模型架构和 checkpoint 格式是一样的。

#### 实际优化过程不一样

虽然两个完整 run 都配置 20 epochs，但最终比较的 checkpoint 是旧 epoch 10 和新 epoch 18：

| Checkpoint | 每 epoch batch | 已训 epoch | 约 optimizer step | 样本暴露次数 |
|---|---:|---:|---:|---:|
| 旧 10k epoch 10 | 1,250 | 10 | 12,500 | 100,000 |
| 新 35,869 epoch 18 | 4,483 | 18 | 80,694 | 645,642 |

新候选的参数更新次数是旧候选的约 `6.46` 倍，encoder 解冻后的训练 epoch 也分别为 15 和 7。
因此从“实际优化轨迹”看，它们不是只替换了数据内容的等计算量对照。

#### 训练时代码版本不是同一快照

旧 checkpoint 于 2026-08-01 训练，新 checkpoint 于 2026-08-14 训练；中间在 2026-08-08/
08-09 审计并提交了代码修正。已知与监督路径相关的差异包括：

1. `detached_integral` 时间维切片修复；但本次两个 checkpoint 在功能上都是无 detach，不构成主差异。
2. `VPSDE_linear.transform(src==tgt)` 新增直接返回，避免 `x_start->x_start` 绕行转换的
   `1e-6` 数值误差。随机张量复现中旧绕行实现的平均绝对偏差约 `2.0e-5`、最大约
   `8.6e-4`，是实际但很小的数值差异。
3. Decoder 由 `self.training` 判断分支改为根据是否传入扩散状态判断，并新增 RL 多候选 `sample()`
   接口。常规监督训练中都传入扩散状态，所以两者均走 score 分支；新接口主要服务 RL。
4. Dataset 新增可选 metadata/get-by-name 路径；监督训练仍使用默认 `return_metadata=False`，返回张量协议不变。

旧 run 的 `args.json` 没有保存 git SHA 或全部源码 hash，而且旧训练当时包含后来才于 08-09 提交的
encoder warm-start 参数，说明当时使用过未提交工作树。因此无法对旧训练源码做逐字节完整还原，
也不应声称两次训练代码一模一样。

#### 闭环推理对比是同协议的

旧、新 checkpoint 的本次 20 场景结果都是之后使用当前同一份 HDP planner/decoder/DPM-Solver 代码重测，
而不是直接拿两个时期不同代码产生的历史分数对比。已核对：

```text
20 个 scenario token 及顺序完全相同
simulation=closed_loop_nonreactive_agents
global seed=0
planner 都加载 checkpoint.ema_state_dict
num_samples=1
diffusion_steps=10
noise_scale=0.1
worker=sequential
```

两份 `args.json` 中不同的数据路径、实验名、`planning_hybrid_loss`/detach 字段不参与常规闭环
前向采样。两个 checkpoint 的模型键和 shape 又完全相同，所以**当前闭环分数差异可归因于两个
checkpoint 学到的参数不同，而不是推理协议不一致**。

## 14. Las Vegas-only 21,409 NPZ 监督训练（2026-08-19）

### 14.1 实验目的与控制变量

为检查旧 10k 在 mini-val 上较强是否与 Las Vegas 样本占比更高有关，从正式 35,869
训练并集中只筛选 Las Vegas 样本重新训练。模型结构、编码器预训练起点和主要监督训练参数沿用
35,869 的 `planning_hybrid_loss=0.01`、无 detach 实验；本次只替换训练 manifest。

需要注意：该实验可以直接检验 Las Vegas-only 数据分布在当前代码下的效果，但旧 35,869
实验没有保存完整源码快照，因此不能将它描述成逐字节源码一致的严格单变量复现。本次额外保存
Git commit 和关键源码 SHA256，供后续复盘。

### 14.2 Las Vegas 数量与筛选依据

输入 manifest：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/
nuplan_train_formal_union_35869_seed3407_v1/diffusion_planner_training.json
```

按 NPZ 文件名前缀 `us-nv-las-vegas-strip_` 筛选。结果：

```text
35,869 中 Las Vegas NPZ = 21,409（59.69%）
manifest 条目 = 21,409
唯一文件名 = 21,409
缓存缺失文件 = 0
```

输出 manifest：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/
nuplan_train_las_vegas_21409_seed3407_v1/diffusion_planner_training.json
```

manifest SHA256：

```text
86ff8fbdca6fafc569cbd835fa84a8473971f0d4dbc58215f3b11fe140ffe95f
```

这里只生成新的文件清单，不复制 21,409 个 NPZ；`train_set` 继续指向原 35,869 cache，
DataLoader 只读取新 manifest 列出的 Las Vegas 文件。

### 14.3 启动前门禁

```text
GPU = NVIDIA GeForce RTX 4060 Laptop GPU 8 GB
磁盘可用空间 = 50 GB
训练冲突进程 = 0
测试 = 40 passed, 15 warnings in 5.82s
```

warning 均为既有第三方依赖弃用提示。

本次源码记录：

```text
Git commit: 720025446c0cb86bb236e4903454864db0b71153
train_predictor.py: 9ab0d211aad9ab60f152baee1b922b3e942656e9b375ebd242be13889dfbaca4
train_epoch.py: b81781b780c0fa356496677711f7b38ed234b9151adfe9628d9847272b1512cc
loss.py: 3280cf779aece5d0dc8f3ae64824d7da4444f8649e6588d01500b256632eb5e3
hyper_diffusion_planner.py: 08c85b5447785ed1bab34f41d01bd0b169df7497acb843511ff1e1c1bdcabe89
```

由于工作区存在未提交修改，Git commit 不能单独代表源码状态，以上关键文件 SHA256 才是本次
训练实现的精确标识。

### 14.4 实际启动命令

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-paper-supervised-las-vegas21409-omega001-nodetach \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_las_vegas_21409_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_las_vegas_21409_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 3 \
  --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
  --warm_up_epoch 2 --save_utd 1 --num_workers 4 \
  --planning_hybrid_loss 0.01 \
  --planning_detach_window_size 0 \
  --seed 3407 --use_wandb false \
  > /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/nuplan_train_las_vegas_21409_seed3407_v1/supervised_las_vegas21409_omega001_nodetach_train.log \
  2>&1 < /dev/null &
```

### 14.5 启动验收

```text
启动时间 = 2026-08-19 01:49:11 CST
launcher PID = 33194
Dataset Prepared = 21,409
每 epoch batch 数 = ceil(21,409 / 8) = 2,676
模型参数量 = 5,092,996
编码器 warm-start = 151/151 张量，1,799,040 参数
前三个 epoch 冻结编码器
epoch 1 已正常产生有限 loss
启动后训练速度约 23 batch/s，无 OOM
epoch 1 完成：train loss = 0.4188
epoch 1 checkpoint = model_epoch_1_trainloss_0.4188.pth（67,417,699 bytes）
epoch 2 已自动开始，训练进程持续运行
```

训练输出目录：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/
nuplan_train_las_vegas_21409_seed3407_v1/training_log/
hdp-paper-supervised-las-vegas21409-omega001-nodetach/2026-08-19-01:49:11/
```

### 14.6 训练完成结果

训练于 `2026-08-19 02:47:25 CST` 正常完成，耗时约 58 分钟；20 个 epoch checkpoint
和 `latest.pth` 均已保存，日志中没有 traceback、OOM、NaN 或 Inf。

| Epoch | Train loss | Encoder |
|---:|---:|---|
| 1 | 0.4188 | frozen |
| 2 | 0.2484 | frozen |
| 3 | 0.1512 | frozen |
| 4 | 0.2251 | trainable |
| 5 | 0.1914 | trainable |
| 6 | 0.1733 | trainable |
| 7 | 0.1593 | trainable |
| 8 | 0.1590 | trainable |
| 9 | 0.1488 | trainable |
| 10 | 0.1413 | trainable |
| 11 | 0.1361 | trainable |
| 12 | 0.1332 | trainable |
| 13 | 0.1312 | trainable |
| 14 | 0.1270 | trainable |
| 15 | 0.1240 | trainable |
| 16 | 0.1244 | trainable |
| 17 | 0.1202 | trainable |
| 18 | 0.1192 | trainable |
| 19 | 0.1166 | trainable |
| 20 | 0.1156 | trainable |

训练 loss 总体稳定下降；epoch 4 的短暂上升对应编码器由冻结切换为可训练。最低训练 loss
来自 epoch 20，但不同训练集上的 train loss 不可直接用于判断规划能力，也不能据此认定 epoch 20
是最佳 checkpoint。下一步应在固定 mini-val-1000 上对后期 checkpoint 排名，再使用完全相同的
mini-val 闭环场景评测入选 checkpoint。

### 14.7 固定 mini-val-1000 checkpoint 排名

20 个 checkpoint 使用固定 1,000 个验证样本、batch size 16、seed 3407、EMA 权重扫描。
单次扫描前三名为 epoch 13、epoch 7、epoch 10；随后用 seed 3407/3408/3409 重复3次：

| Checkpoint | 单次 loss | 三次均值 | planning loss | hybrid loss |
|---:|---:|---:|---:|---:|
| epoch 10 | 0.102283 | **0.109284** | 0.038394 | 7.089020 |
| epoch 13 | **0.101282** | 0.111002 | 0.040374 | 7.062717 |
| epoch 7 | 0.101925 | 0.113676 | 0.040947 | 7.272959 |

重复评测后选择 epoch 10 作为开环均值候选，但它需要经过闭环安全门禁；epoch 13 和 epoch 7
作为备选候选保留。

### 14.8 固定3场景闭环门禁

三个候选均完成 3/3 simulation，runner failure=0：

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧10k epoch10 | 0.947107 | 1.000000 | 1.000000 | 1.000000 | 0.830792 | 1.000000 | 参考 |
| 完整35,869 epoch18 | 0.912050 | 1.000000 | 1.000000 | 1.000000 | 0.718615 | 1.000000 | 参考 |
| Las Vegas epoch10 | 0.591368 | 0.666667 | 1.000000 | 1.000000 | 0.668410 | 0.666667 | 安全门禁失败 |
| Las Vegas epoch13 | **0.875785** | 1.000000 | 1.000000 | 1.000000 | 0.602560 | 1.000000 | 通过 |
| Las Vegas epoch7 | 0.577267 | 0.666667 | 1.000000 | 1.000000 | 0.631649 | 0.666667 | 安全门禁失败 |

epoch 13 是唯一通过安全门禁的 Las Vegas 候选，因此送入20场景闭环。

### 14.9 固定20场景闭环结果

Las Vegas epoch 13 完成 20/20 simulation，runner failure=0。使用同一份
`mini-val-closed-loop-20`、同一顺序worker和当前统一推理协议：

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC | Direction | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧10k epoch10 | **0.787373** | 0.950000 | 1.000000 | 0.900000 | **0.668771** | 0.900000 | 0.975000 | 1.000000 |
| 完整35,869 epoch18 | 0.593706 | 0.850000 | 0.850000 | 0.850000 | 0.645880 | 0.750000 | 1.000000 | 1.000000 |
| Las Vegas epoch13 | 0.616348 | 0.950000 | 0.900000 | 0.850000 | 0.562268 | 0.900000 | 1.000000 | 1.000000 |

相对完整 35,869，Las Vegas-only 模型的 score 提高 `0.022642`（约 `+3.81%`），collision
和 TTC 恢复到旧10k水平，但 route progress 下降 `0.083612`。相对旧10k，score 仍下降
`0.171025`（约 `-21.71%`），route progress 下降 `0.106503`，因此没有带来项目目标所需的
整体正收益。

本轮结论：Las Vegas 数据专门化改善了完整35,869模型的部分安全指标，但牺牲了路线进度，且
没有超过旧10k基线。Las Vegas-only 不晋升为主监督模型；它作为“城市分布消融实验”保留。

新增评测结果：

```text
固定3场景汇总：
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/closed_loop_eval/mini_val_3_old10k_full35869_vs_las_vegas21409_top3.json
SHA256=72f7d4b50ee46147621f743f57e2d624ff51069f8ab278d55777a3d35077722c

固定20场景汇总：
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/closed_loop_eval/mini_val_20_old10k_full35869_vs_las_vegas21409_epoch13.json
SHA256=c22191d0e41eb978dd3db361582a7af9a3fd2837113e32f682bcbbf8726cca5f
```

## 15. 旧10k训练复现（2026-08-19）

### 15.1 复现边界

用户要求按照原旧10k训练集和代码重新训练，并检查闭环指标是否一致。

已确认原始训练集和配置：

```text
训练 cache：/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache
训练 manifest：/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json
manifest 条目：10,000
manifest SHA256：1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514
来源：44个mini-train日志的balanced_logs采样
```

原始 `args.json` 保存在旧实验目录：

```text
.../training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/args.json
```

原始配置为：20 epochs、batch size 8、learning rate 5e-4、warm-up 2 epochs、编码器冻结3
epochs、`planning_hybrid_loss=0.01`、seed 3407、EMA开启、同一 `checkpoints/model.pth` 编码器
预训练起点。

原旧10k运行没有保存完整源码快照或Git SHA，因此无法逐字节恢复 2026-08-01 当时的工作区。
本次是“原始10k数据+原始训练参数+当前监督训练代码”的可审计重现；会在结果中单独标注这一
限制，不能把它称为源码完全相同的复现。旧参数中没有 `planning_detach_window_size`，当前命令
显式使用 `0`；根据旧版切片行为审计，旧训练实际也是完整 cumsum 梯度。

### 15.2 新复现实验输出目录

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/
mini_train_balanced_10000_reproduction_currentcode_20260819/
```

### 15.3 实际训练命令与结果

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-reproduce-old10k-currentcode \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_reproduction_currentcode_20260819 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 3 \
  --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
  --warm_up_epoch 2 --save_utd 1 --num_workers 4 \
  --planning_hybrid_loss 0.01 \
  --planning_detach_window_size 0 \
  --seed 3407 --use_wandb false
```

训练目录：

```text
.../training_log/hdp-reproduce-old10k-currentcode/2026-08-19-10:30:54/
```

20/20 epochs 正常完成，无 traceback、OOM、NaN 或 Inf。训练 loss 对比：

| Epoch | 原始旧10k | 当前代码重现 | 差值（重现-原始） |
|---:|---:|---:|---:|
| 1 | 0.5834 | 0.5833 | -0.0001 |
| 2 | 0.3564 | 0.3659 | +0.0095 |
| 3 | 0.1868 | 0.1836 | -0.0032 |
| 4 | 0.2484 | 0.2519 | +0.0035 |
| 5 | 0.1995 | 0.1968 | -0.0027 |
| 6 | 0.1803 | 0.1826 | +0.0023 |
| 7 | 0.1709 | 0.1647 | -0.0062 |
| 8 | 0.1553 | 0.1648 | +0.0095 |
| 9 | 0.1438 | 0.1477 | +0.0039 |
| 10 | 0.1380 | 0.1347 | -0.0033 |
| 11 | 0.1396 | 0.1420 | +0.0024 |
| 12 | 0.1253 | 0.1272 | +0.0019 |
| 13 | 0.1283 | 0.1279 | -0.0004 |
| 14 | 0.1193 | 0.1217 | +0.0024 |
| 15 | 0.1210 | 0.1207 | -0.0003 |
| 16 | 0.1106 | 0.1095 | -0.0011 |
| 17 | 0.1119 | 0.1181 | +0.0062 |
| 18 | 0.1083 | 0.1091 | +0.0008 |
| 19 | 0.1027 | 0.1021 | -0.0006 |
| 20 | 0.1046 | 0.1020 | -0.0026 |

loss 曲线形态和量级相近，但不是逐 epoch 完全一致。epoch 10 checkpoint：

```text
原始 SHA256=6902eb677c94a33287fc496878b1783d719fe187c7c45abdfe5d9a5ff7269258
重现 SHA256=d25286edf1fd3111b538dfd945ee6e92f235fe401715b4f1b29731d3ad3230de
```

### 15.4 固定 val1k 重复3次

两者都使用当前同一评测代码、固定 val1k、batch size 16、seed 3407/3408/3409 和 checkpoint
中的 EMA 权重：

| 模型 | Total loss | Diffusion loss | Hybrid loss |
|---|---:|---:|---:|
| 原始旧10k epoch10 | 0.119260 | 0.041313 | 7.794776 |
| 当前代码重现 epoch10 | **0.114473** | **0.039331** | **7.514191** |

重现 checkpoint 的开放环指标略好，但这不能直接推出闭环得分更高。

### 15.5 固定3场景闭环

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC |
|---|---:|---:|---:|---:|---:|---:|
| 原始旧10k epoch10 | **0.947107** | 1.000000 | 1.000000 | 1.000000 | 0.830792 | 1.000000 |
| 当前代码重现 epoch10 | 0.622202 | 0.666667 | 1.000000 | 1.000000 | **0.836102** | 0.666667 |

重现模型未通过常规3场景安全门禁。为直接回答“与原指标是否一致”，仍继续执行固定20场景，
而不是按正常模型筛选流程提前停止。

### 15.6 固定20场景最终对比

两者均完成 20/20 simulation，runner failure=0：

| 模型 | Score | Collision | Drivable | Progress gate | Route progress | TTC | Direction | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始旧10k epoch10 | **0.787373** | 0.950000 | 1.000000 | 0.900000 | **0.668771** | 0.900000 | 0.975000 | 1.000000 |
| 当前代码重现 epoch10 | 0.750168 | 0.950000 | 1.000000 | 0.900000 | 0.660921 | **0.950000** | 0.975000 | 1.000000 |

差值（重现-原始）：

```text
score=-0.037205（-4.73%）
collision=0
drivable=0
progress gate=0
route progress=-0.007850
TTC=+0.050000
direction=0
comfort=0
场景级：重现 6胜 / 8负 / 6平
```

结论：结果**接近但不完全一致**。安全和合规指标中除 TTC 外都相同，TTC 反而改善，但 route
progress 略降且场景级得分重新分配，最终总分低 `4.73%`。这说明固定 seed 不足以让当前代码
重建出原 checkpoint；原始源码快照缺失、当前代码差异以及 CUDA/DataLoader 的随机执行细节都
可能导致优化轨迹不同。现有证据不能把差异归因到单一原因。

结果文件：

```text
val1k原始：tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_evaluation_old10k_original_epoch10_repeat3.json
val1k重现：tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_evaluation_old10k_reproduction_currentcode_epoch10_repeat3.json
3场景：tmp/closed_loop_eval/mini_val_3_old10k_original_vs_reproduction_epoch10.json
20场景：tmp/closed_loop_eval/mini_val_20_old10k_original_vs_reproduction_epoch10.json
20场景 SHA256=16cb78a58c44e7318a1b11fe2898dde7051e633259662abb832a1980d5e9a3ed
```

### 15.7 为什么3场景门禁差异很大

逐场景拆分后，差异集中在唯一一个场景，而不是三个场景整体退化：

| Token | 类型 | 原始旧10k score | 重现 score | 重现主要变化 |
|---|---|---:|---:|---|
| `08764235932a5530` | stationary_in_traffic | 0.915351 | 0.940923 | 略好 |
| `134f95eed3775e22` | high_magnitude_speed | 0.980777 | **0** | collision=0、TTC=0 |
| `263da852dba35cac` | traffic-light intersection | 0.945191 | 0.925683 | 略低 |

在 `134f95eed3775e22` 中，重现模型仍有 `route progress=0.935164`，但发生1次自车有责的
vehicle collision；记录的碰撞能量代理值为 `3.600928 m/s`，最小 TTC 为 `0`。NuPlan 的安全
乘法门禁因此把该场景总分直接变为0。

3场景均分随之从：

```text
原始：(0.915351 + 0.980777 + 0.945191) / 3 = 0.947107
重现：(0.940923 + 0 + 0.925683) / 3 = 0.622202
```

也就是说，1个场景失败在3场景集合中占 `1/3`，会把 collision/TTC 均值直接从1降到
`2/3`；这就是分数看起来差异极大的直接原因。相同失败在20场景中只占 `1/20`，并且重现模型
在其他场景还有改善，所以20场景总分仅下降 `4.73%`，TTC均值反而从0.90升到0.95。

该高速场景说明闭环安全指标存在阈值效应：两个 checkpoint 的小参数差异经过闭环反馈累积后，
可能使轨迹从“刚好安全”跨到“发生碰撞”；一旦跨过阈值，指标不是小幅下降，而是整场归零。

## 16. 完整 mini-train 306,801 NPZ 生成

### 16.1 启动配置（2026-08-19）

完整 mini-train 候选数来自既有 10k 采样报告：44 个官方 mini-train 日志合计包含
`306,801` 个去重后 Scenario。`HDP-nuplan/nuplan_train.json` 是完整 nuPlan train 日志清单，
包含 13,180 项，不能直接当作 mini-train 的 44 日志清单。因此从已验证报告
`tmp/mini_train_balanced_10000_seed3407_v1/sampling_report.json` 的
`selected_per_log` 键生成：

```text
tmp/mini_train_full_306801_seed3407_v1/mini_train_log_names.json
```

生成任务使用本地 `diffusion_planner` 环境、`balanced_logs`、种子 3407，并启用
`--skip_existing true` 和 `--fail_on_error true`。输出目录为：

```text
tmp/mini_train_full_306801_seed3407_v1/cache
```

后台进程 PID 为 `91472`，日志为：

```text
tmp/mini_train_full_306801_seed3407_v1/data_process_full_mini_train.log
```

### 16.2 速度下降诊断（2026-08-19 16:12 CST）

当缓存达到约 53,900 个 NPZ 时，滑动速度为：

| 时间窗口 | 新增 NPZ | 平均速度 |
|---|---:|---:|
| 最近 10 分钟 | 869 | 1.448/s |
| 最近 30 分钟 | 2,752 | 1.529/s |
| 最近 60 分钟 | 6,967 | 1.935/s |

诊断结论：

1. `DataProcessor.work()` 使用 `for scenario in tqdm(scenarios)` 逐场景串行提取 NPZ；
   `SingleMachineParallelExecutor` 只用于前面的 Scenario 构建，并未并行化 NPZ 提取阶段。
2. 场景按 `scenario_output_name` 排序，当前连续处理
   `us-nv-las-vegas-strip`。不同地图区域的车道、边界、路线和邻车数量不同，因此每个 Scenario
   的地图查询与数组构造成本并不恒定。
3. 实测系统仍有约 75%～79% CPU 空闲，而主进程约使用 2.2～2.4 个 CPU 核，说明主要限制是
   当前实现的串行/有限线程并发，不是 CPU 总算力不足。
4. NVMe 即时利用率约 0.6%、I/O wait 约 4%，不是磁盘吞吐瓶颈；虽然系统已有约 5.4 GiB
   swap 占用，但观测窗口内没有持续换入换出，因此也不是本次持续降速的直接原因。

前期使用全程累计平均速度计算的 ETA 会被最开始的快速场景拉高。当前阶段应优先参考滑动速度；
若后续保持 1.45～1.94 NPZ/s，剩余任务约需 36～48 小时，但切换地图区域后速度仍可能变化。

### 16.3 从单进程切换为四分片并行（2026-08-19）

用户确认并行切换后，先校验 `launcher.pid` 对应命令确实是本次完整 mini-train
`data_process.py`，再发送 `SIGTERM`。停止记录为：

```text
停止时间：2026-08-19 16:25:24 CST
单进程 PID：91472
停止前 NPZ：54,996
停止后 NPZ：54,996
```

这说明停止过程没有丢失已原子写入的 NPZ。停点记录保存在：

```text
tmp/mini_train_full_306801_seed3407_v1/single_process_stop_record.txt
```

并行化没有把 NuPlan `Scenario` 对象直接提交给新的 `ProcessPool`。采用的是更稳妥的进程级
日志分片：每个进程独立构建数据库与地图对象，四个分片的日志集合互不重叠，并共同使用原 cache。
由于原完整采样报告满足：

```text
selected_per_log == available_per_log
sum(selected_per_log) == 306,801
duplicate_output_names_removed == 0
```

所以每个日志的真实场景数可以直接作为分片目标；不同日志不会生成相同输出名，已存在文件则由
`--skip_existing true` 安全跳过。

代码改造：

1. `scripts/build_preprocessing_plan.py`
   - 新增 `build_weighted_plan()`；
   - 使用 Largest-Processing-Time-First 贪心分配，把大日志优先放入当前负载最小的分片；
   - 新增 `--sampling_report` 与 `--num_shards`，直接读取审计过的 `selected_per_log`。
2. `scripts/run_preprocessing_shard.py`
   - 新增 `--shared_cache_path`，允许互斥日志分片复用原 cache；
   - 新增显式 `--scenario_builder_workers` 参数并传给 `data_process.py`。
3. `data_process.py`
   - 新增 `--scenario_builder_workers`；
   - 将其传给 `SingleMachineParallelExecutor(max_workers=...)`，避免四个分片启动时各自占满
     全部 16 个 CPU。
4. `scripts/merge_preprocessing_shards.py`
   - 新增共享 cache 的文件验证和 manifest 前缀支持；由于训练入口的 `--train_set` 指向
     cache 本身，正式训练清单使用裸文件名 `<NPZ>`。
5. `tests/test_preprocessing_pipeline.py`
   - 增加真实数量均衡、日志不重叠、共享 cache runner 命令和共享 cache 合并测试。

分片计划生成命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/build_preprocessing_plan.py \
  --log_names_json HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/mini_train_log_names.json \
  --output_dir HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/parallel_plan_4 \
  --total_scenarios 306801 \
  --sampling_report HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/sampling_report.json \
  --num_shards 4 \
  --seed 3407
```

计划门禁结果：

| 分片 | 日志数 | 目标 Scenario |
|---|---:|---:|
| `shard_00000` | 11 | 76,002 |
| `shard_00001` | 11 | 79,600 |
| `shard_00002` | 11 | 76,400 |
| `shard_00003` | 11 | 74,799 |
| 合计 | 44 | 306,801 |

四个分片的日志并集为原 44 个日志，交集为空。测试结果：

```text
7 passed in 2.98s
```

### 16.4 四路后台任务与启动验收

每个分片使用下列等价命令启动，其中 `INDEX` 为 0～3：

```bash
setsid env PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/run_preprocessing_shard.py \
  --plan HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/parallel_plan_4/preprocessing_plan.json \
  --shard_index INDEX \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/parallel_workers_4 \
  --shared_cache_path HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache \
  --scenario_builder_workers 4
```

首次启动前参数门禁发现 runner 的旧 `argparse.REMAINDER` 不能直接接收未知的
`--scenario_builder_workers`；该尝试在解析参数时退出，没有启动任何分片，也没有修改 cache。
随后把参数加入 runner 的显式参数表，测试通过后正式启动。

正式 runner PID：

```text
shard_00000: 116918
shard_00001: 116975
shard_00002: 117024
shard_00003: 117069
```

每个分片的 `runner.pid`、`runner.log`、`launch.json` 和最终报告均位于：

```text
tmp/mini_train_full_306801_seed3407_v1/parallel_workers_4/shard_*/
```

启动后最初每秒跳过数千个 Scenario，是四个分片在检查已有的 54,996 个可读 NPZ，并非新增
文件速度。进入新增阶段后的实测结果：

| 观测窗口 | 新增 NPZ | 聚合速度 | 相对单进程近期 1.45/s |
|---|---:|---:|---:|
| 45 秒 | 207 | 4.60/s | 3.17 倍 |
| 稳定期 60 秒 | 321 | 5.35/s | 3.69 倍 |

稳定期四个数据进程各约使用 1.7 个 CPU 核、约 0.9 GiB RSS；四个日志均无错误。16:58 CST
缓存达到 55,954 个 NPZ，磁盘剩余约 98 GiB。按 5.35 NPZ/s 的短时稳定速度估算，剩余约
13 小时；不同地图阶段仍会使 ETA 波动。

全部分片完成后的合并命令预定为：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/merge_preprocessing_shards.py \
  --shards_root HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/parallel_workers_4 \
  --shared_cache HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache \
  --output_manifest HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json \
  --output_report HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/merged_processing_report.json
```

该合并操作必须等四个 `processing_report.json` 的 `status` 都为 `complete` 后执行；脚本会拒绝
不完整分片、重复日志、重复 NPZ、manifest 数量不匹配或缺失文件。

### 16.5 与旧 10k 的 NPZ 生成方法核对

旧 10k 的 `sampling_report.json` 显示：

```text
strategy=balanced_logs
seed=3407
requested_log_count=44
requested_scenarios=10,000
raw_scenarios=unique_scenarios=306,801
base_quota_per_log=227
```

当前完整 mini-train 同样使用 `balanced_logs`、种子 3407 和相同 44 个日志，但目标为全部
306,801 个唯一 Scenario。四分片运行时种子为 3407～3410；因为每个分片的目标等于其日志的
全部可用 Scenario，随机打乱只影响处理顺序，不影响成员集合和 NPZ 内容。

旧缓存生成于 2026-08-01，而当前任务增加了分片、共享 cache、断点跳过、原子写入、失败审计
和 manifest/checksum 报告。因此若把“生成方法”理解为整个任务编排，两次并非逐项完全相同。
但场景特征提取仍调用同一套 `DataProcessor`、地图处理、邻车处理和默认张量规格。

为了验证模型实际读取的数据是否变化，在当前已生成文件中找到 5,510 个与旧 10k 同名的
Scenario，并按地图分层抽取 60 个：Singapore One North、Boston、Las Vegas 各 20 个。逐项
比较两侧 NPZ 的 key、shape、dtype 和数组值，结果为：

```text
old_only_keys=空
new_only_keys=空
common_array_mismatch_count=0
```

因此当前证据支持：两次任务的采样规模和运行调度不同，但相同 Scenario 产出的训练特征完全
一致；当前完整数据可视为用同一特征定义把旧 10k 扩展到完整 306,801 个 Scenario。并行化本身
不会改变模型输入数值。

### 16.6 四分片暂停点（2026-08-20 03:20 CST）

用户要求暂停时，四个分片中前三个已自然完成：

| 分片 | 状态 | manifest | processed | skipped | failed |
|---|---|---:|---:|---:|---:|
| `shard_00000` | complete | 76,002 | 59,690 | 16,312 | 0 |
| `shard_00001` | complete | 79,600 | 60,746 | 18,854 | 0 |
| `shard_00002` | complete | 76,400 | 62,100 | 14,300 | 0 |
| `shard_00003` | 暂停 | 尚未生成最终报告 | 运行到 66,059/74,799 | - | 日志中无场景失败 |

只对仍活动的 `shard_00003` 进程组发送 `SIGTERM`，其他三个已完成分片没有重启或修改。
暂停后门禁结果：

```text
暂停时间：2026-08-20 03:20:09 CST
完整 cache：298,061 / 306,801（97.15%）
剩余：8,740
隐藏临时文件：0
残留预处理进程：0
cache 占用：45 GiB
磁盘剩余：62 GiB
```

暂停前最近 5、10、30 分钟的聚合速度约为 2.05、1.95、2.35 NPZ/s；当时只剩最后一个分片
工作，因此恢复后预计还需约 1～1.5 小时。

恢复时只启动索引 3：

```bash
out=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1
shard_dir="$out/parallel_workers_4/shard_00003"

setsid env PYTHONUNBUFFERED=1 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/scripts/run_preprocessing_shard.py \
  --plan "$out/parallel_plan_4/preprocessing_plan.json" \
  --shard_index 3 \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root "$out/parallel_workers_4" \
  --shared_cache_path "$out/cache" \
  --scenario_builder_workers 4 \
  > "$shard_dir/runner_resume.log" 2>&1 < /dev/null &
```

`--skip_existing` 默认为 true，因此恢复后会先快速检查并跳过 `shard_00003` 已有文件，然后只
生成剩余 NPZ。完成后再执行 16.4 节记录的四分片合并命令。

### 16.7 暂停点的正式训练门禁

2026-08-20 03:21 CST 检查结果：cache 中有 298,061 个 NPZ，但最终
`diffusion_planner_training.json`、`merged_processing_report.json` 和全量校验结果尚不存在；
`shard_00003` 也没有完成态 `processing_report.json`。因此当前不能把该目录作为“完整
mini-train”正式训练输入。

技术上可以临时扫描现有 cache 生成 298,061 项 manifest，但缺失的 8,740 项集中在最后一个
分片的处理尾段，并不是对完整数据的重新均匀抽样；这样会引入分片/排序偏差，也会使实验无法
声明使用完整 mini-train。因此不采用该方案。

正式开训前必须依次满足：恢复并完成 `shard_00003`、四分片合并得到 306,801 项 manifest、
验证 manifest 数量/重复项/缺失文件/NPZ 可读性均通过，然后把该 manifest 传给
`train_predictor.py --train_set_list`。

### 16.8 恢复最后一个分片（2026-08-20 03:22 CST）

恢复前门禁确认：没有残留预处理进程，cache 为 298,061 个 NPZ；前三个分片均为
`status=complete, failed=0`，`shard_00003` 尚无完成报告。只按 16.6 节命令重启索引 3，
新 runner PID 为 `172461`，运行日志为：

```text
tmp/mini_train_full_306801_seed3407_v1/parallel_workers_4/shard_00003/runner_resume.log
```

进程先高速遍历并验证已有文件，约 40 秒后越过原暂停位置 66,059/74,799，开始生成新 NPZ。
进入纯新增阶段后，46 秒新增 88 个，实测速度约 1.913 NPZ/s。03:24 CST 总缓存达到
298,218 个，剩余 8,583 个，按该速度预计约 1.25 小时；进程错误数为 0。

### 16.9 自动收尾与正式监督训练授权（2026-08-20 03:28 CST）

用户确认：最后分片完成后自动执行剩余步骤并开始训练。为避免人工轮询期间遗漏完成事件，新增
并启动：

```text
脚本：scripts/finalize_full_mini_and_train.sh
watcher PID：173555
watcher 日志：tmp/mini_train_full_306801_seed3407_v1/finalize_and_train.log
```

该脚本按以下门禁顺序执行：

1. 等待四个 `processing_report.json` 都为 `status=complete` 且 `failed=0`；
2. 使用共享 cache 合并四个 manifest，输出裸 NPZ 文件名的
   `diffusion_planner_training.json`；
3. 使用 `validate_processed_cache.py` 检查 306,801 项 manifest、NPZ 数量、重复/缺失文件、
   所有默认 shape、有限数值、44 个日志计数和采样报告；
4. 验证报告必须同时满足 `status=passed`、`manifest_count=306801`、`npz_count=306801`，
   否则脚本退出且不启动训练；
5. 验证通过后启动完整 mini-train 监督训练。

自动训练配置沿用最近正式监督实验：

```text
planning_hybrid_loss=0.01
planning_detach_window_size=0（无 detach）
train_epochs=20
batch_size=8
learning_rate=5e-4
warm_up_epoch=2
freeze_encoder_epochs=3
encoder_pretrained_model_path=checkpoints/model.pth
num_workers=4
seed=3407
```

训练输出目录预定为：

```text
tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/
```

训练日志预定为：

```text
tmp/mini_train_full_306801_seed3407_v1/supervised_full_mini_omega001_nodetach_train.log
```

03:28 CST 启动自动收尾器时 cache 已达到 298,787 个 NPZ，最后分片仍在运行。

### 16.10 全量数据完成与首次训练中断（2026-08-20 04:48 CST）

自动收尾流程已完成全部数据门禁：最终 manifest 包含 306,801 项、cache 包含
306,801 个 NPZ、覆盖 44 个 mini-train 日志；`merged_processing_report.json` 为
`status=complete`，`cache_validation_report.json` 为 `status=passed`，没有发现缺失或
重复项。正式监督训练于 04:08:36 CST 自动启动。

训练第 1 个 epoch 已完整结束，训练损失为 `0.1197`，并于 04:36:32 保存完整状态：

```text
tmp/mini_train_full_306801_seed3407_v1/
  supervised_training_full_mini_omega001_nodetach/training_log/
  hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/latest.pth
```

checkpoint 中包含 `epoch=1`、模型参数、EMA 参数、优化器状态、学习率调度器状态和
W&B ID，可通过将上述运行目录传给 `--resume_model_path` 从第 2 个 epoch 重新开始。

第 2 个 epoch 执行到 `9987/38350` batch，约为该 epoch 的 26.0% 时，训练 launcher
在 04:48:16 CST 收到外部信号 15（`SIGTERM`）并退出。日志中没有 OOM、CUDA 或模型
计算异常；当前也没有残留训练进程。由于 checkpoint 只在完整 epoch 结束时保存，第 2 个
epoch 已完成的 9,987 个 batch 不在 checkpoint 中，恢复时会重新执行第 2 个 epoch；第 1 个
epoch 的模型、优化器、调度器和 EMA 状态不会丢失。

### 16.11 从 Epoch 1 checkpoint 恢复训练（2026-08-20）

用户确认继续训练。启动前发现参数校验原先只允许
`freeze_encoder_epochs > 0` 与 `encoder_pretrained_model_path` 配套使用，会误拦截完整
checkpoint 恢复。完整 checkpoint 已包含 encoder，恢复时仍必须保留
`freeze_encoder_epochs=3`，否则 encoder 会在 Epoch 2 提前解冻，改变实验条件。因此将校验
修正为：encoder-only 初始化和完整 checkpoint 恢复二者满足其一即可。修正后执行完整测试：

```text
43 passed, 15 warnings in 5.86s
```

续训使用与首次训练相同的数据、20 epoch 总轮数、batch size、学习率、warm-up、冻结策略和
loss 配置，只将 `encoder_pretrained_model_path` 替换为：

```text
--resume_model_path tmp/mini_train_full_306801_seed3407_v1/
  supervised_training_full_mini_omega001_nodetach/training_log/
  hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42
```

后台 launcher PID 为 `187338`，PID 文件和续训日志分别为：

```text
tmp/mini_train_full_306801_seed3407_v1/supervised_training_resume.pid
tmp/mini_train_full_306801_seed3407_v1/
  supervised_full_mini_omega001_nodetach_resume.log
```

启动后日志确认 `Model load done`、`Optimizer load done`、`Schedule load done` 和
`Step load done`，随后进入 `Epoch 2/20`，且 `Encoder trainable: False`。首次核验时已运行
到 `400/38350` batch，速度约 `23.16 batch/s`，训练进程和 GPU 计算均正常。

### 16.12 调整为 Epoch 10 阶段性评测（2026-08-20 10:28 CST）

用户决定先训练到 Epoch 10 查看效果，再决定是否继续到 Epoch 20。正在运行的 Python 进程已将
`train_epochs=20` 读入内存，直接修改启动参数或配置文件不会改变该进程；立即重启为
`train_epochs=10` 又会丢失当前 Epoch 2 的进度。因此保留当前训练进程，并启动安全停止门禁：

```text
脚本：scripts/stop_full_mini_training_at_epoch10.sh
monitor PID：191791
PID 文件：tmp/mini_train_full_306801_seed3407_v1/stop_at_epoch10_monitor.pid
监控日志：tmp/mini_train_full_306801_seed3407_v1/stop_at_epoch10_monitor.log
```

门禁每 15 秒检查一次 Epoch 10 checkpoint。检测到文件后先使用 PyTorch 完整读取，确认
`epoch=10`，并确认 checkpoint 同时包含模型、EMA、优化器和调度器状态；只有验证通过后才向
launcher 发送 `SIGTERM`。因此最多可能开始少量 Epoch 11 batch，但不会在 Epoch 10
checkpoint 写入过程中停止，也不会损坏后续恢复所需状态。

设置门禁时训练位于 Epoch 2 的 86%，按冻结阶段和解冻阶段历史实测速度，预计约 5～6 小时
到达 Epoch 10。停止后先评测 Epoch 10；若决定继续，可从该运行目录的 `latest.pth` 恢复到
Epoch 11，并保持原优化器、学习率调度器和 EMA 状态。

为使“训练到 Epoch 10 后查看效果”能够自动闭环，另行启动评测等待器：

```text
脚本：scripts/evaluate_full_mini_epoch10_after_stop.sh
watcher PID：191982
PID 文件：tmp/mini_train_full_306801_seed3407_v1/evaluate_epoch10_after_stop_monitor.pid
等待日志：tmp/mini_train_full_306801_seed3407_v1/evaluate_epoch10_after_stop_monitor.log
```

它只在安全停止门禁生成 `stopped_after_epoch10` 后启动评测，避免与监督训练争用 GPU。评测固定
使用 `mini_val_balanced_1000_seed3407_v1`、EMA 权重、seed 3407～3409 共 3 次重复，输出为：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
  checkpoint_evaluation_full_mini306801_epoch10_repeat3.json
```

该阶段先使用固定 val1k 开环指标判断 Epoch 10 的基本效果；闭环仿真成本更高，待开环结果生成
并与旧 10k、35,869 等基线比较后再决定是否执行，避免对明显不合格 checkpoint 浪费时间。

### 16.13 Epoch 10 门禁进度复核（2026-08-20 10:41 CST）

Epoch 2 已完成并保存，训练损失为 `0.0495`；当前进入 Epoch 3，进度为
`10809/38350`（约 28%），实时速度约 `15.89 batch/s`。Epoch 3 仍处于 encoder 冻结阶段；
Epoch 4～10 将进入 encoder 解冻阶段，预计剩余约 5～6 小时。Epoch 10 停止门禁和后续
val1k 评测等待器均仍在运行。

### 16.14 RAM OOM 定位与第二次恢复（2026-08-20 11:07 CST）

Epoch 3 在 `23783/38350`（约 62%）提前中断，Epoch 10 门禁没有触发。内核日志给出明确
根因：10:57:06 CST 发生全局 OOM，训练主 worker `PID 187359` 被内核杀死；该进程当时
匿名 RSS 约 7.2 GiB，4 个 DataLoader worker 的 RSS 各约 4.2 GiB。机器物理内存只有
15 GiB，因而这是 CPU/RAM OOM，不是 GPU 显存 OOM。torch elastic 在子进程被杀后向
launcher 发送 `SIGTERM`，所以训练日志末尾显示 signal 15。

Epoch 2 checkpoint 已于 10:33 完整保存，包含 `epoch=2`、模型、EMA、优化器和调度器状态；
Epoch 3 的部分进度没有 checkpoint，需要从 Epoch 3 开头重跑。恢复时仅将
`num_workers` 从 4 调整为 0，训练集、manifest、batch size 8、模型、loss、学习率、冻结
策略、优化器和调度器均保持不变。此修改取消 DataLoader 子进程复制和并行预取，主要影响
墙钟速度，不改变训练样本或梯度定义；实测速度约从 19～25 batch/s 降至 16～17 batch/s。

为把训练从编辑器/Codex 的 process scope 中隔离，三个长期任务改由用户级 systemd transient
service 托管：

```text
训练：hdp-full-mini-train-epoch10.service（MainPID 198467）
停止门禁：hdp-full-mini-stop-epoch10.service（MainPID 198646）
评测等待器：hdp-full-mini-eval-epoch10.service（MainPID 198652）
恢复脚本：scripts/resume_full_mini_training_numworkers0.sh
训练日志：tmp/mini_train_full_306801_seed3407_v1/
  supervised_full_mini_omega001_nodetach_resume_numworkers0.log
```

恢复后日志确认模型、优化器、调度器和 step 均加载成功，进入 `Epoch 3/20` 且
`Encoder trainable: False`。运行 90 秒时 systemd 统计训练 service 内存约 3.30 GiB，系统
available RAM 约 8.4 GiB，只有 launcher 和单个训练 worker，没有 4 个 DataLoader worker；
进度为 `1445/38350`，GPU 正常计算。Epoch 10 checkpoint 完整保存后仍会自动停止并启动固定
val1k 三次评测。

### 16.15 修复跨 batch 计算图滞留并从 Epoch 4 恢复（2026-08-20 11:53 CST）

继续监测发现：即使 `num_workers=0`，Epoch 4 运行到 17% 时训练 service 内存仍升到约
11.5 GiB、系统 available RAM 降到约 1.7 GiB。这说明 DataLoader worker 复制不是唯一原因。
代码检查定位到 `hdp_nuplan/train_epoch.py`：原实现把每个 batch 的整个 `loss` 张量字典追加
到 `epoch_loss`，使已完成 batch 的反向计算图一直保留到 epoch 结束。小数据训练不明显，但
38,350 batch 会导致内存随进度持续增长。

Epoch 3 已完整保存（`loss=0.0349`），因此主动停止尚未保存的 Epoch 4。修改为追加 loss 前
对每个张量执行 `detach().item()`，仅保存用于求 epoch 平均值的 Python 标量。该修改不会改变
反向传播、优化器更新或最终 loss 均值，只释放已经完成 batch 的计算图。修改后完整测试结果：

```text
43 passed, 15 warnings in 5.85s
```

随后从 `epoch=3` 的完整 checkpoint 再次恢复 Epoch 4，并重新启动三个 systemd 服务。
恢复日志确认模型、优化器、调度器和 step 均成功加载，`Encoder trainable: True`。运行约
54 秒时进度为 `611/38350`，速度约 10.9 batch/s，训练 service 内存约 1.80 GiB，系统
available RAM 约 11 GiB；相比修复前同一 epoch 的 11.5 GiB 已显著下降。按当前解冻 encoder
后的实测速度，预计约 6.5～7.5 小时到达 Epoch 10，之后自动停止并执行 val1k 三次评测。

### 16.16 Epoch 8 进度复核（2026-08-20 15:44 CST）

Epoch 4～7 均已完整保存，对应训练损失依次为 `0.0265`、`0.1459`、`0.0923`、
`0.0737`。当前位于 Epoch 8 的 `15608/38350`（约 41%），实时速度约
`9.54 batch/s`。系统 available RAM 约 11 GiB，说明 detached loss 修复后不再因反向计算图
滞留耗尽物理内存；训练、Epoch 10 停止门禁和评测等待器三个 systemd service 均为
`active/running`。按当前速度预计约 2.5 小时到达 Epoch 10。

### 16.17 Epoch 5 loss 跳升原因核查（2026-08-20）

逐个读取 Epoch 1～7 checkpoint 中的 optimizer 和 scheduler 状态后，确认 loss 跳升与实际
学习率变化严格对齐：Epoch 1～4 使用 `5e-5`，Epoch 5 起使用 `5e-4`，即学习率一次提高
10 倍；同时 encoder 从 Epoch 4 起已解冻。最终保存结果为：

| Epoch | Encoder | 实际学习率 | Train loss |
|---:|---|---:|---:|
| 1 | frozen | 5e-5 | 0.1197 |
| 2 | frozen | 5e-5 | 0.0495 |
| 3 | frozen | 5e-5 | 0.0349 |
| 4 | trainable | 5e-5 | 0.0265 |
| 5 | trainable | 5e-4 | 0.1459 |
| 6 | trainable | 5e-4 | 0.0923 |
| 7 | trainable | 5e-4 | 0.0737 |

`learning_rate=5e-4`、`warm_up_epoch=2` 和 warm-up 起始倍率 0.1 本来会让训练先从
`5e-5` 起步，再升到 `5e-4`。但当前训练循环在 epoch 末尾先执行 `save_model()`，之后才
执行 `scheduler.step()`；checkpoint 因此保存的是调度器更新前的状态。前三次中断都从这样
的 checkpoint 恢复，使低学习率阶段被重复，最终延长到 Epoch 4。Epoch 4 完整结束且没有
再次中断后，调度器才把 Epoch 5 的学习率推进到 `5e-4`。

因此 Epoch 5 跳升的主要原因是学习率提高 10 倍，并叠加 encoder 已解冻、全部 encoder 参数
共同更新；不是训练样本突然变化，也不是 loss 公式改变。之后 loss 从 `0.1459` 连续下降到
`0.0923`、`0.0737`，说明模型正在适应新的学习率，暂时没有出现持续发散。不过，这也说明
本次因多次恢复导致的学习率轨迹不等同于原计划的连续训练轨迹；后续正式可复现实验应把
`scheduler.step()` 放到 checkpoint 保存之前，或在恢复时显式补执行尚未保存的 scheduler
step，并重新验证学习率序列。

### 16.18 Epoch 9 进度复核（2026-08-20 16:32 CST）

Epoch 8 已完整保存，训练损失为 `0.0616`；当前位于 Epoch 9 的 `8439/38350`（约 22%），
实时速度约 `10.24 batch/s`。训练、Epoch 10 停止门禁和评测等待器均为
`active/running`，系统 available RAM 约 9.1 GiB。按当前速度预计约 1 小时 45 分钟完成
Epoch 9 和 Epoch 10，随后门禁验证 Epoch 10 checkpoint、停止训练并启动 val1k 三次评测。

### 16.19 改为 Epoch 9 A/B 恒定低学习率分叉（2026-08-20 16:43 CST）

用户决定当前实验 A 完成 Epoch 9 后暂停，不再继续 Epoch 10；随后从同一个 Epoch 4
checkpoint 启动实验 B，所有参数在 Epoch 5～9 恒定使用 `5e-5`。因此停止原 Epoch 10
门禁和评测等待器，改由以下 systemd service 自动完成 A→B 切换：

```text
service：hdp-full-mini-switch-a9-to-b.service
MainPID：233644
脚本：scripts/switch_epoch9_a_to_constant5e5_b.sh
日志：tmp/mini_train_full_306801_seed3407_v1/switch_epoch9_a_to_b.log
```

为避免 A 运行目录中的 `latest.pth` 后续被 Epoch 9 覆盖，已提前把 A 的
`model_epoch_4_trainloss_0.0265.pth` 固定复制到：

```text
tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/
  source_model_epoch_4_trainloss_0.0265.pth
tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/latest.pth
```

源文件与两个副本的 SHA-256 均为
`b593a2d9671d84feeb20e247c3a16dc474fdd4ec5d35b0d97e467dda8fcac913`；checkpoint
元数据为 `epoch=4`、`loss=0.0264890622`、optimizer LR=`5e-5`，并包含模型、EMA、
optimizer 和 scheduler 状态。

为保证 B 不会加载旧 scheduler 后再次升到 `5e-4`，训练入口新增
`--reset_lr_schedule_on_resume`：保留模型、EMA 和 AdamW 动量，但把所有 optimizer param
group 的 `lr/initial_lr` 重设为命令行学习率，并重建 scheduler。B 使用
`learning_rate=5e-5`、`warm_up_epoch=1`，当前 scheduler 实现因此恒定返回 `5e-5`。
同时恢复后执行 `train_sampler.set_epoch(init_epoch)`，使 B 从 Epoch 5 对应的 sampler epoch
4 开始，而不是错误回到 sampler epoch 0。修改后测试为 `43 passed`。

A/B 固定条件如下：

| 条件 | 实验 A | 实验 B |
|---|---|---|
| 共同起点 | Epoch 4 checkpoint | 同一个 Epoch 4 checkpoint |
| Epoch 5～9 LR | `5e-4` | 恒定 `5e-5` |
| Encoder | trainable | trainable |
| 数据/manifest | 306,801 full mini-train | 相同 |
| batch size / workers | 8 / 0 | 8 / 0 |
| hybrid loss / detach | 0.01 / 0 | 相同 |
| seed | 3407 | 3407 |

切换脚本会先完整读取并验证 A Epoch 9 checkpoint，再向 A launcher 发送 SIGTERM；随后 B
自然训练到 Epoch 9。B 完成后自动在固定 val1k、seed 3407～3409 上分别评测 A/B 的 EMA
checkpoint，并生成：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
  checkpoint_evaluation_full_mini306801_experiment_a_epoch9_repeat3.json
  checkpoint_evaluation_full_mini306801_experiment_b_constant5e5_epoch9_repeat3.json
  checkpoint_comparison_full_mini306801_a_vs_b_epoch9_repeat3.json
```

### 16.23 A/B 补训 Epoch 10 并进行固定闭环（2026-08-20 21:58 CST）

用户决定将 A/B 都补训到 Epoch 10，再进行同一组固定闭环场景评测。已启动用户级 systemd
服务：

```text
service：hdp-ab-epoch10-closed-loop.service
脚本：scripts/train_ab_epoch10_then_closed_loop.sh
```

该脚本按顺序执行：

1. 从 A Epoch 9 checkpoint 继续使用 `5e-4` 训练 Epoch 10；
2. 从 B Epoch 9 checkpoint 继续使用恒定 `5e-5` 训练 Epoch 10；
3. 校验 A/B 的 Epoch 10 checkpoint 均包含模型、EMA、optimizer 和 scheduler；
4. 使用官方 `closed_loop_nonreactive_agents`、`mini-val-closed-loop-20.yaml`、
   sequential worker、相同 mini DB 和相同 GPU 配置，顺序运行 A/B；
5. 使用 `summarize_closed_loop_metrics.py` 生成 A/B 闭环汇总。

固定闭环输出根目录：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/
```

对应闭环汇总预定为：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/
  closed_loop_a_vs_b_epoch10.json
```

实验边界：旧 checkpoint 没有保存 Python/NumPy/PyTorch RNG 状态，因此 B 虽使用相同 seed
和 sampler epoch，无法复现 A 当时连续经过 Epoch 4 后的完全相同 diffusion 噪声与增强随机
流。本轮属于同起点、同数据顺序、同 seed 的单次学习率对照；三次固定验证可降低评测随机性，
但若需要严格 paired training randomness，应在后续让 A/B 都从 Epoch 4 重新运行并采用逐 epoch
显式种子。

### 16.20 A Epoch 9 完成、B 已启动（2026-08-20 17:35 CST）

切换门禁已验证并暂停实验 A：

```text
A checkpoint：model_epoch_9_trainloss_0.0551.pth
验证时间：2026-08-20 17:18:40 CST
```

随后实验 B 已从固定 Epoch 4 checkpoint 启动，日志确认：

```text
Learning-rate schedule reset after resume: lr=5e-05, warm_up_epoch=1
Epoch 5/9
Encoder trainable: True
```

当前 B 位于 Epoch 5 的 `11757/38350`（约 31%），速度约 `9.5 batch/s`。B 尚未生成新的
epoch checkpoint；当前系统 available RAM 约 5.7 GiB，训练仍在正常运行。按当前速度预计 B
约 4～5 小时完成 Epoch 5～9，随后自动执行 A/B 的 val1k 三次评测。

### 16.21 实验 B Epoch 7 进度复核（2026-08-20 19:58 CST）

B 的 Epoch 5、6 已完整保存，训练损失分别为 `0.0187`、`0.0148`；当前 Epoch 7 为
`36239/38350`（约 94%），速度约 `9.69 batch/s`。systemd 切换服务仍为
`active/running`，系统 available RAM 约 5.0 GiB，GPU 正常计算。预计约 2 小时完成
Epoch 7～9，之后自动执行 A/B Epoch 9 的 val1k 三次评测。

### 16.22 A/B Epoch 9 训练与 val1k 评测完成（2026-08-20 21:52 CST）

实验 B 已完成 Epoch 5～9，最终训练损失为 `0.0100`；切换 service 正常结束。随后 A/B
Epoch 9 EMA checkpoint 在固定 `mini_val_balanced_1000_seed3407_v1` 上分别以 seed
3407～3409 完成三次开环评测。结果如下（validation loss 越低越好）：

| 实验 | 训练阶段 | Train loss | Val loss | Planning loss | Hybrid loss |
|---|---|---:|---:|---:|---:|
| A | Epoch 5～9，LR=`5e-4` | 0.0551 | **0.1730** | **0.0579** | **11.5124** |
| B | Epoch 5～9，恒定 LR=`5e-5` | **0.0100** | 0.2297 | 0.0723 | 15.7336 |

B 的训练 loss 更低，但 val loss 比 A 高 `0.0567`，约高 32.8%；说明“训练 loss 更低”
在本次对照中没有转化为更好的泛化效果，恒定 `5e-5` 反而出现了更明显的训练/验证差距。
目前只能据此判断：在本次同起点、同数据和单次学习率分叉中，A 的 `5e-4` 对 val1k 更有利；
最终规划结论仍需对 A/B 做相同固定场景的闭环评测。

结果文件：

```text
tmp/mini_val_balanced_1000_seed3407_v1/
  checkpoint_evaluation_full_mini306801_experiment_a_epoch9_repeat3.json
  checkpoint_evaluation_full_mini306801_experiment_b_constant5e5_epoch9_repeat3.json
  checkpoint_comparison_full_mini306801_a_vs_b_epoch9_repeat3.json
```

### 16.23 A/B Epoch 10 训练完成，闭环流水线在退出阶段中断（2026-08-21 00:05 CST）

A/B 都已遍历完 Epoch 10 的 `38350` 个 batch，并成功保存 checkpoint：

| 实验 | Epoch 10 train loss | checkpoint 优化器学习率 |
|---|---:|---:|
| A：基础学习率 `5e-4` | `0.0502208732` | `5e-4` |
| B：恒定学习率 `5e-5` | `0.0090623703` | `5e-5` |

B 在 checkpoint 写入完成后的 `torchrun` 进程退出阶段收到 `SIGSEGV`（exit code
`-11`）。系统日志没有 OOM 记录；A/B checkpoint 均能由 PyTorch 完整读取，且都含
`model`、`ema_state_dict`、`optimizer`、`schedule`，`epoch=10`。因此这是保存后的
退出异常，不需要重训；但原脚本使用 `set -e`，所以未继续启动闭环评测。

### 16.24 仅恢复 A/B Epoch 10 固定闭环评测（2026-08-21 CST）

为避免重复训练，新增恢复脚本：

```text
HDP-nuplan/scripts/run_ab_epoch10_closed_loop_only.sh
```

脚本会先重新校验两个 Epoch 10 checkpoint，然后使用完全相同的评测条件顺序运行
A 和 B：

```text
simulation: closed_loop_nonreactive_agents
scenario_filter: mini-val-closed-loop-20
worker: sequential
DB: nuplan-v1.1_mini/data/cache/mini
```

运行日志与最终汇总分别保存为：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/
  closed_loop_only.log
  closed_loop_a_vs_b_epoch10.json
```

NuPlan 本次实际的单次仿真目录位于
`closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/`；已在恢复脚本中按
该实际路径设置汇总输入。

恢复 service 于 `2026-08-21 00:28:40 CST` 启动，两个 checkpoint 通过校验后，A
于 `00:28:56 CST` 开始顺序仿真 20 个固定场景。

### 16.25 A/B Epoch 10 固定 20 场景闭环结果（2026-08-21 00:58 CST）

A 和 B 均成功运行完全部 20 个场景，失败数均为 `0`；两次运行的 scenario ID 完全一致，因此可作为配对比较。

| 指标 | A：LR `5e-4` | B：恒定 LR `5e-5` | B - A |
|---|---:|---:|---:|
| NuPlan score | 0.821742 | **0.876941** | **+0.055198** |
| 无责任碰撞率 | 0.950000 | 0.950000 | 0 |
| 可行驶区域合规率 | 1.000000 | 1.000000 | 0 |
| 正在前进率 | 0.950000 | 0.950000 | 0 |
| 沿专家路线前进率 | 0.758318 | **0.875490** | **+0.117171** |
| 行驶方向合规率 | 0.975000 | **1.000000** | +0.025000 |
| 速度限制合规率 | **0.998772** | 0.994825 | -0.003948 |
| 舒适性 | 1.000000 | 1.000000 | 0 |

本次结果显示：虽然 B 的 train loss 更低，但在同一组固定闭环场景上，B 的 NuPlan score 比 A 高
`0.055198`，主要来自沿专家路线前进指标的改善；碰撞、可行驶区域、正常前进和舒适性指标与 A 相同。这是固定 20 场景的对照结果，不代表完整 mini-val 的统计显著性。

结果文件：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/closed_loop_a_vs_b_epoch10.json
```

### 16.26 实验 B 从 Epoch 10 继续至 Epoch 20（2026-08-21 CST）

根据闭环结果，先只对 B 进行延长训练，不重新训练 A。训练从以下 Epoch 10 checkpoint 恢复：

```text
tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/
  model_epoch_10_trainloss_0.0091.pth
```

新增可复现脚本：

```text
HDP-nuplan/scripts/resume_experiment_b_epoch10_to20.sh
```

参数保持不变：完整 mini-train `306801` 个 NPZ、`batch_size=8`、
`learning_rate=5e-5`、`warm_up_epoch=1`、`planning_hybrid_loss=0.01`、
`planning_detach_window_size=0`、`num_workers=0`、`seed=3407`。将 `train_epochs` 设为 `20`，
因此只执行 Epoch 11–20，保留 Epoch 10 的模型、EMA 和 AdamW 状态。

运行日志：

```text
tmp/mini_train_full_306801_seed3407_v1/experiment_b_epoch10_to20.log
```

该任务已于 `2026-08-21 01:23:34 CST` 通过 user systemd 后台启动，启动日志已确认进入 `Epoch 11/20`，没有重新训练 Epoch 1–10。

### 16.27 实验 B Epoch 20 完成并启动固定闭环评测（2026-08-21 CST）

B 于 `2026-08-21 09:57:14 CST` 正常完成 Epoch 11–20，每轮 checkpoint 均已保存。
Epoch 20 结果为：

```text
checkpoint: model_epoch_20_trainloss_0.0055.pth
exact epoch: 20
train loss: 0.0055
optimizer lr: 5e-5
```

该 checkpoint 已通过完整性检查，包含 `model`、`ema_state_dict`、`optimizer`、
`schedule`，可以开始评测。新增评测脚本：

```text
HDP-nuplan/scripts/evaluate_experiment_b_epoch20_closed_loop.sh
```

评测继续使用与 B Epoch 10 相同的 `closed_loop_nonreactive_agents`、
`mini-val-closed-loop-20`、sequential worker 和 mini DB。完成后会与已保存的 B Epoch 10
结果生成配对汇总：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20/
  closed_loop_b20.log
  closed_loop_b_epoch10_vs20.json
```

评测 service 已于 `2026-08-21 10:57:41 CST` 启动，checkpoint 通过校验后已开始构建和顺序执行 20 个场景。

### 16.28 B Epoch 10 与 Epoch 20 闭环对比结果（2026-08-21 11:10 CST）

B Epoch 20 评测成功完成 `20/20`，且与 B Epoch 10 使用相同的 20 个 scenario ID。但继续训练并未带来更高的闭环得分：

| 指标 | B Epoch 10 | B Epoch 20 | Epoch 20 - Epoch 10 |
|---|---:|---:|---:|
| NuPlan score | **0.876941** | 0.810012 | **-0.066928** |
| 无责任碰撞率 | 0.950000 | 0.950000 | 0 |
| 可行驶区域合规率 | **1.000000** | 0.950000 | -0.050000 |
| 正在前进率 | 0.950000 | 0.950000 | 0 |
| 沿专家路线前进率 | 0.875490 | **0.875867** | +0.000377 |
| 行驶方向合规率 | 1.000000 | 1.000000 | 0 |
| 速度限制合规率 | 0.994825 | **0.995257** | +0.000432 |
| 舒适性 | **1.000000** | 0.950000 | -0.050000 |

分场景查看显示，主要退化来自三个场景：

- `1fb0bb88f9d35d59`：可驶区域合规从 `1` 变为 `0`，场景分从 `0.897` 降至 `0`；
- `2da316803b1d561d`：舒适性从 `1` 变为 `0`，场景分从 `1.000` 降至 `0.875`；
- `3713735b94cf5a7b`：时间到碰撞阈值从 `1` 变为 `0`，场景分从 `0.910` 降至 `0.599`。

因此，Epoch 20 的训练 loss 虽降至 `0.0055`，但固定 20 场景闭环指标反而下降；本实验不支持“继续训练一定改善闭环”的假设。

对比文件：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20/closed_loop_b_epoch10_vs20.json
```

### 16.29 加入原始 Diffusion Planner 同场景闭环基线（2026-08-21 CST）

为让 B Epoch 10/20 的结果有原始模型参考，加入仓库原始 Diffusion Planner checkpoint：

```text
checkpoint: /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth
args:       /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/args.json
```

该 checkpoint 包含 `model` 和 `ema_state_dict`，args 包含原始 planner 所需的
`future_len=80` 、`agent_num=32` 、`predicted_neighbor_num=10`，已通过启动前校验。
将使用与 B Epoch 10/20 完全相同的 `mini-val-closed-loop-20`、mini DB、
`closed_loop_nonreactive_agents` 和 sequential worker。

评测脚本：

```text
HDP-nuplan/scripts/evaluate_original_diffusion_vs_b_closed_loop.sh
```

运行后会将原始 Diffusion Planner、B Epoch 10和 B Epoch 20 汇总为：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20_vs_diffusion/
  closed_loop_diffusion_vs_b_epoch10_epoch20.json
```

原始 Diffusion Planner 评测 service 已于 `2026-08-21 11:23:55 CST` 启动，已成功构建 `20` 个场景，当前正在按 sequential worker 执行。

### 16.30 原始 Diffusion Planner 与 B Epoch 10/20 三模型闭环结果（2026-08-21 11:36 CST）

三个模型均成功运行 `20/20`，失败数均为 `0`，且三次运行使用相同的 20 个 scenario ID。

| 指标 | 原始 Diffusion Planner | HDP B Epoch 10 | HDP B Epoch 20 |
|---|---:|---:|---:|
| NuPlan score | **0.881446** | 0.876941 | 0.810012 |
| 无责任碰撞率 | 0.900000 | **0.950000** | **0.950000** |
| 可行驶区域合规率 | 0.950000 | **1.000000** | 0.950000 |
| 正在前进率 | **1.000000** | 0.950000 | 0.950000 |
| 沿专家路线前进率 | **0.960774** | 0.875490 | 0.875867 |
| 行驶方向合规率 | 1.000000 | 1.000000 | 1.000000 |
| 时间到碰撞合规率 | 0.900000 | **0.950000** | 0.900000 |
| 速度限制合规率 | 0.984866 | **0.994825** | **0.995257** |
| 舒适性 | 0.950000 | **1.000000** | 0.950000 |
| 单次规划耗时 | **0.196970 s** | 0.202807 s | **0.189651 s** |

解读：在这组固定 20 场景上，原始 Diffusion Planner 只比 B Epoch 10 高 `0.004505`，两者得分非常接近；原始 DP 的主要优势是沿专家路线前进率更高，B Epoch 10 在碰撞、可行驶区域、TTC 和舒适性指标上反而更好。B Epoch 20 的 score 低于两者，与之前的 B10/B20 对比一致。

该原始 DP checkpoint 是仓库自带的成熟模型，与 B 使用的训练数据、训练轮数和模型架构不完全相同；因此这是工程参考基线，不是严格的同训练条件架构比较。此外，旧有 3 场景评测中原始 DP 的 `0.995829` 不应与本次 20 场景均值混用。

三模型汇总文件：

```text
tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20_vs_diffusion/
  closed_loop_diffusion_vs_b_epoch10_epoch20.json
```

### 16.31 从 HDP B Epoch 10 开始 1,000 NPZ RL pilot（2026-08-21 CST）

用户确认先进行小规模验证：`1,000` 个 NPZ、`2` 个 RL epoch，训练后在同一组 20 场景上进行闭环评测。

训练起点为 HDP B Epoch 10：

```text
tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/
  model_epoch_10_trainloss_0.0091.pth
```

使用的关键参数：

```text
train_set                         mini_train_pilot_1000_seed3407_v1/cache
batch_size                        2
learning_rate                     4e-7
planning_hybrid_loss              0.01
rl_group_size                     32
rl_rollout_steps                  6
rl_buffer_update_epoch            2
rl_buffer_size                    1024
rl_max_update_steps_per_epoch    500
rl_detach_window_size             0
rl_freeze_encoder                 true
rl_expert_anchor_weight           0
rl_center_reward_weights          true
reward_progress_guard_weight      5.0
```

`planning_hybrid_loss=0.01` 与 B Epoch 10 保持一致，`rl_detach_window_size=0` 表示 RL 微调也不使用 detach。脚本会在 RL 训练完成后自动运行 NuPlan 固定 20 场景闭环，并与原始 Diffusion Planner 和 B Epoch 10 汇总。

可复现脚本：

```text
HDP-nuplan/scripts/run_rl_from_full_mini_b10_pilot1000.sh
```

训练和评测日志：

```text
HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/rl_from_full_mini_b10_pilot1000.log
```

首次启动在数据训练前停止：原校验脚本误将 JSON 文件的文本行数 `1002` 当作 manifest 条目数，实际解析后的数组长度为 `1000`。已改为按 JSON 解析后计数，未修改数据。

RL 组件回归测试为 `19 passed`，任务于 `2026-08-21 12:05:22 CST` 后台启动。启动日志已确认：`Loaded pretrained model: missing=0, unexpected=0`，并已进入 `NuPlan RL rollout` 阶段。

RL 训练已于 `2026-08-21 12:07:25 CST` 成功退出，Epoch 2 checkpoint 完整可读：

```text
training_log/hdp-rl-from-full-mini-b10-pilot1000-omega001-nodetach/2026-08-21-12:05:28/
  model_epoch_2_trainloss_-0.0000.pth
```

原自动串联脚本因 `--save_dir` 与 checkpoint 实际保存目录不一致而未找到 checkpoint，仅使评测阶段退出；训练本身没有失败，未重新训练。已修正主脚本的输出根目录，并使用新增的评测专用脚本从该 checkpoint 继续：

```text
HDP-nuplan/scripts/evaluate_rl_from_full_mini_b10_pilot1000_closed_loop.sh
```

该评测任务于 `2026-08-21 12:19:03 CST` 启动，已成功读取 RL Epoch 2 checkpoint，并建立 `20` 个固定闭环场景；当前正在 sequential 执行，尚未生成最终汇总文件。

### 16.32 B Epoch 10 的 1,000 NPZ RL pilot 闭环结果（2026-08-21 12:32 CST）

原始 Diffusion Planner、B Epoch 10 和 RL pilot 均完成 `20/20`，失败数均为 `0`，三者 scenario ID 完全一致。

| 指标 | 原始 Diffusion Planner | HDP B Epoch 10 | B10 + RL pilot | RL - B10 |
|---|---:|---:|---:|---:|
| NuPlan score | **0.881446** | 0.876941 | 0.862734 | **-0.014206** |
| 无责任碰撞率 | 0.900000 | 0.950000 | 0.950000 | 0 |
| 可行驶区域合规率 | 0.950000 | 1.000000 | 1.000000 | 0 |
| 正在前进率 | 1.000000 | 0.950000 | 0.950000 | 0 |
| 沿专家路线前进率 | **0.960774** | 0.875490 | 0.881342 | **+0.005852** |
| 行驶方向合规率 | 1.000000 | 1.000000 | 1.000000 | 0 |
| TTC 合规率 | 0.900000 | **0.950000** | 0.900000 | **-0.050000** |
| 速度限制合规率 | 0.984866 | 0.994825 | 0.994206 | -0.000618 |
| 舒适性 | 0.950000 | 1.000000 | 1.000000 | 0 |

RL 平均路线进度改善 `+0.005852`，多个场景得分有小幅上升；但
`3713735b94cf5a7b / traversing_pickup_dropoff` 的 TTC 从 `1` 变为 `0`，场景分从
`0.909614` 降为 `0.603598`。该单场景的 `-0.306016` 足以抵消其他场景的进度改善，使平均 NuPlan score 比 B Epoch 10 低 `0.014206`（约 `-1.62%`）。

因此本次 RL pilot **未通过闭环门禁**：它提升了 progress，但没有带来总分正收益，并引入一个 TTC 退化场景。在修复或加强 TTC/risk 约束前，不应直接扩大到 10,000 或 306,801 NPZ。

三模型汇总：

```text
tmp/mini_train_pilot_1000_seed3407_v1/
rl_from_full_mini_b_epoch10_omega001_nodetach_closed_loop20/
  closed_loop_dp_b10_rlpilot1000.json
```

### 16.33 扩大 RL 闭环评测至 100 个 mini-val 场景（2026-08-21 CST）

20 场景评测中单个 TTC 退化场景对均值影响较大，因此新增 100 场景配对评测。新 filter 使用 official mini-val 的 10 个 log，包含有效 mission goal，不使用 train log，并固定 `limit_total_scenarios=100` 和 `shuffle=false`。

```text
filter: HDP-nuplan/hdp_nuplan/config/scenario_filter/mini-val-closed-loop-100.yaml
protocol: closed_loop_nonreactive_agents + sequential
models: HDP B Epoch 10 vs B Epoch 10 + RL pilot
```

可复现脚本：

```text
HDP-nuplan/scripts/evaluate_rl_vs_b10_closed_loop100.sh
```

输出与日志：

```text
tmp/mini_train_pilot_1000_seed3407_v1/rl_vs_b10_closed_loop100/
  closed_loop100.log
  b10_vs_rl_closed_loop100.json
```

实际启动时 NuPlan 在该 filter 下构建出 `39` 个有效场景，而非配置上限 `100`；原因是 10 个 mini-val log、指定的 14 类 scenario type 和 `remove_invalid_goals=true` 共同过滤后仅剩 39 个。B Epoch 10 已成功开始执行这 39 个场景，完成后将用完全相同的 39 个场景运行 RL。

### 16.34 B Epoch 10 与 RL pilot 的 39 场景扩大评测结果（2026-08-21 14:20 CST）

B Epoch 10 和 RL pilot 均完成 `39/39`，失败数均为 `0`，两次运行的 scenario ID 集合完全相同。

| 指标 | HDP B Epoch 10 | B10 + RL pilot | RL - B10 |
|---|---:|---:|---:|
| NuPlan score | 0.930621 | **0.932507** | **+0.001886** |
| 无责任碰撞率 | 0.961538 | 0.961538 | 0 |
| 可行驶区域合规率 | 0.948718 | 0.948718 | 0 |
| 正在前进率 | 1.000000 | 1.000000 | 0 |
| 沿专家路线前进率 | 0.942435 | **0.949383** | **+0.006948** |
| 行驶方向合规率 | 1.000000 | 1.000000 | 0 |
| TTC 合规率 | 0.948718 | 0.948718 | 0 |
| 速度限制合规率 | **0.991494** | 0.990756 | -0.000738 |
| 舒适性 | 1.000000 | 1.000000 | 0 |
| 单次规划耗时 | 0.210021 s | **0.207994 s** | -0.002026 s |

逐场景配对统计：

- NuPlan score：RL `14` 胜、`0` 负、`25` 平；最大单场景改善为 `+0.010156`。
- 沿专家路线前进率：`15` 胜、`0` 负、`24` 平。
- 速度限制合规率：`0` 胜、`5` 负、`34` 平，但降幅较小。
- 碰撞、可行驶区域、TTC 和舒适性的逐场景值都没有发生好坏互换。

对 39 个配对 score 差值使用固定随机种子 `3407` 进行 `200,000` 次 paired bootstrap，平均改善的 95% percentile 区间为 `[+0.000998, +0.002877]`。该区间只描述当前选中的 39 个场景，不能直接外推到全部 NuPlan 场景。

结论：RL 在这组 39 场景上带来了**弱正收益**，主要是更高的路线进度，且未观察到安全硬指标回退。但这组 39 场景不包含 20 场景评测中 TTC 从 `1` 退化为 `0` 的 `3713735b94cf5a7b`，而且平均提升仅为约 `0.20%`。因此可以证明 RL 已改变策略并能改善部分 progress，但尚不足以证明它获得了稳定、可泛化的新能力。

最终汇总文件：

```text
tmp/mini_train_pilot_1000_seed3407_v1/rl_vs_b10_closed_loop100/
  b10_vs_rl_closed_loop100.json
```

### 16.35 RL v2 安全门改造与固定 59 场景实验（2026-08-21 CST）

在继续扩大 RL 训练前，先将两次评测的 scenario ID 取并集。核对结果为：20 场景与 39 场景的交集为 `0`，合计 `59` 个独立场景。合并已有配对结果后，旧 RL pilot 相对 B Epoch 10 的结果为：

```text
NuPlan score mean delta                 -0.003569
ego_progress_along_expert_route delta  +0.006577
score win / loss / tie                  20 / 3 / 36
TTC win / loss / tie                     0 / 1 / 58
```

因此 39 场景的弱正收益不足以通过总门禁；这不是继续增加 NPZ 数量可以直接解决的问题。

对 TTC 失败场景 `3713735b94cf5a7b / traversing_pickup_dropoff` 读取 NuPlan 指标时序和 simulation history：

- B Epoch 10 最小 TTC 为 `1.3 s`，RL pilot 为 `0.9 s`。
- 约在场景第 10.7–13.3 秒，RL 速度比 B Epoch 10 高约 `0.15–0.4 m/s`。
- RL 自车在该阶段沿路线多前进约 `1.5–1.7 m`，这与 progress 改善和 TTC 退化同时出现一致。

旧 rollout 的平均奖励组成显示 `progress_guard_weight=5`、`risk_weight=1`，且平均 `risk_reward≈0.55`。在普通加权和中，低 TTC 候选可能被 progress/follow/lane 收益抵消。

为此实现可选 RL v2 组内安全门：

```text
reward_safety_gate_threshold = 0.4
reward_safety_gate_margin    = 1.0
rl_expert_anchor_weight      = 0.1
```

安全门的规则是：

1. 如果同一场景的 32 个 diffusion 候选中存在 `risk_reward >= 0.4` 的候选，任何不达标候选都不能靠 progress/follow/lane 取得更高排名。
2. 如果整组都不达标，先按 risk 排名，原始多目标 reward 只作 `1e-3` 量级的 tie-breaker。
3. 阈值为 `0` 时保持原论文奖励行为，因此旧实验仍可复现。
4. `expert_anchor_weight=0.1` 使用同批真实 ego future 提供弱监督锚点，抑制小 replay buffer 上的策略漂移。

修改文件：

```text
HDP-nuplan/hdp_nuplan/rl/reward.py
HDP-nuplan/train_predictor_rl.py
HDP-nuplan/tests/test_rl_components.py
HDP-nuplan/scripts/merge_paired_closed_loop_summaries.py
HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh
```

回归测试结果为 `46 passed`。新增测试分别覆盖：安全候选压过高 progress 的不安全候选、整组都不安全时优先更高 risk，以及关闭安全门时结果与旧 reward 一致。

实验保持与旧 pilot 相同的 B Epoch 10 起点、`1,000 NPZ`、`2 RL epoch`、`learning_rate=4e-7`、`group_size=32`、无 detach 和随机种子 `3407`；只改变安全门和 expert anchor。训练后将顺序评测原 20 场景与新 39 场景，最终生成去重的 59 场景配对结果。

### 16.36 RL v2 固定 59 场景验证结果（2026-08-21 16:56 CST）

RL v2 已完成：

```text
training:  1,000 NPZ, 2 epochs
checkpoint: .../rl_v2_safetygate04_anchor01_from_b10/training_log/
            hdp-rl-v2-safetygate04-anchor01-from-full-mini-b10/2026-08-21-16:05:53/
            model_epoch_2_trainloss_0.0003.pth
closed-loop: 20/20 + 39/39, all successful
```

同一批 59 场景上与 B Epoch 10 的对比：

| 指标 | B Epoch 10 | RL v2 | RL v2 - B10 |
|---|---:|---:|---:|
| NuPlan score | 0.912424 | **0.913063** | **+0.000639** |
| 无责任碰撞率 | 0.957627 | 0.957627 | 0 |
| 可行驶区域合规率 | 0.966102 | 0.966102 | 0 |
| 正在前进率 | 0.983051 | 0.983051 | 0 |
| 沿专家路线前进率 | 0.919741 | **0.921449** | **+0.001708** |
| 行驶方向合规率 | 1.000000 | 1.000000 | 0 |
| TTC 合规率 | 0.949153 | 0.949153 | 0 |
| 速度限制合规率 | 0.992623 | **0.992799** | **+0.000176** |
| 舒适性 | 1.000000 | 1.000000 | 0 |
| 单次规划耗时 | 0.207575 s | 0.258642 s | +0.051067 s |

配对统计：

- score：`15` 胜、`9` 负、`35` 平。
- 路线进度：`10` 胜、`14` 负、`35` 平，平均仍提升 `+0.001708`。
- 碰撞、可行驶区域、TTC 和舒适性：59 个场景全部与 B10 相同，没有新的安全硬失败。
- 之前失败的 `3713735b94cf5a7b`：B10 与 RL v2 的 TTC 都为 `1`；RL v2 场景分 `0.907138`，略低于 B10 的 `0.909614`，但没有再出现 TTC `1→0` 的硬退化。

与旧无安全门 RL pilot 在同一批 59 场景直接比较：RL v2 的 score 提高 `+0.004207`，TTC 合规率从 `0.932203` 恢复到 `0.949153`。这说明改造确实解决了之前的安全退化。

对 RL v2 相对 B10 的 59 个 score 差值进行 `200,000` 次 paired bootstrap，95% percentile 区间为 `[-0.000101, +0.001561]`，包含 0。因此这次可以得出两个分开结论：

1. **安全目标通过**：没有新增 TTC/碰撞硬失败，原失败场景已修复。
2. **稳定正收益尚未证明**：平均 score 只提高 `+0.000639`，bootstrap 区间包含 0，不应直接扩大到全部 NuPlan 数据。此外，单次规划耗时增加约 `24.6%`，需在扩大训练前单独确认。

最终评测文件：

```text
tmp/mini_train_pilot_1000_seed3407_v1/rl_v2_safetygate04_anchor01_from_b10/
  b10_vs_rl_v2_fixed59.json
  rl_v2_train_and_gate59.log
```

### 16.39 扩大到 10,000 NPZ 的 RL v3 实验启动（2026-08-21 CST）

用户确认直接扩大到 `10,000 NPZ`。为了保持对比公平，仅扩大 train cache 与 manifest，其余训练起点和 RL v3 参数全部不变：

```text
cache:       HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache
manifest:    HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json
NPZ count:   10,000
pretrained:  HDP B Epoch 10
RL epochs:   2
threshold:   0.3
anchor:      0.05
detach:      0
seed:        3407
```

10k 训练与 1k RL v3 的主要差别仅是 train cache 规模；评测仍然使用相同的固定 59 场景，以便区分“数据量改变”与“训练逻辑改变”。可复现脚本为：

```bash
RL_PILOT_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1 \
RL_BASELINE_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
RL_EXPECTED_NPZ=10000 \
RL_VARIANT_TAG=safetygate03_anchor005_10k \
RL_SAFETY_GATE_THRESHOLD=0.3 \
RL_EXPERT_ANCHOR_WEIGHT=0.05 \
bash HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh
```

### 16.37 RL v3 中间强度约束实验启动（2026-08-21 CST）

RL v2 已经修复 TTC 硬退化，但进度收益被压弱。因此启动中间强度版本：

```text
variant                    safety_gate_threshold    expert_anchor_weight
safetygate03_anchor005     0.3                      0.05
```

保持不变：B Epoch 10 起点、`1,000 NPZ`、`2 RL epoch`、`4e-7` 学习率、`group_size=32`、无 detach、随机种子 `3407`以及固定 59 场景评测。只放松安全候选阈值并减小 expert anchor，用于分离“约束过强”对进度收益的影响。

可复现脚本使用环境变量参数化：

```bash
RL_VARIANT_TAG=safetygate03_anchor005 \
RL_SAFETY_GATE_THRESHOLD=0.3 \
RL_EXPERT_ANCHOR_WEIGHT=0.05 \
bash HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh
```

### 16.38 RL v3 固定 59 场景结果（2026-08-21 18:26 CST）

RL v3 训练和评测已完成。首次评测脚本在汇总时错用了 v2 的固定目录名，已修复并重用已完成的 20 场景结果，只重跑 39 场景；不重复训练。最终 20+39 均为全部成功。

| 指标 | B Epoch 10 | RL v2（0.4 / 0.1） | RL v3（0.3 / 0.05） |
|---|---:|---:|---:|
| NuPlan score | 0.912424 | 0.913063（+0.000639） | **0.913259（+0.000834）** |
| 沿专家路线前进率 | 0.919741 | 0.921449（+0.001708） | **0.922241（+0.002500）** |
| TTC 合规率 | 0.949153 | 0.949153 | 0.949153 |
| 无责任碰撞率 | 0.957627 | 0.957627 | 0.957627 |
| 可行驶区域合规率 | 0.966102 | 0.966102 | 0.966102 |
| 舒适性 | 1.000000 | 1.000000 | 1.000000 |
| 单次规划耗时 | 0.207575 s | 0.258642 s | 0.261848 s |

RL v3 相对 B Epoch 10 的逐场景统计：

- score：`16` 胜、`8` 负、`35` 平。
- 路线进度：`14` 胜、`10` 负、`35` 平。
- TTC、碰撞、可行驶区域和舒适性：`0` 个场景发生好坏互换。
- 原 TTC 失败场景 `3713735b94cf5a7b`的 TTC 仍为 `1`，场景分为 `0.908087`，已不再出现安全硬退化。

对 RL v3 相对 B10 的 59 个 score 差值进行 `200,000` 次 paired bootstrap，95% percentile 区间为 `[+0.000064, +0.001794]`，不包含 0；但该统计仅基于当前 59 个固定场景，还需多随机种子重复验证。

与 RL v2 直接比较，RL v3 的 score 另外提高 `+0.000196`，路线进度另外提高 `+0.000792`，而 TTC 保持不变。这支持了“约束不宜过强”的判断：`0.3 / 0.05` 比 `0.4 / 0.1` 更适合当前小规模训练。

结论：RL v3 同时通过了当前安全门和小幅正收益门禁，可以进入“多随机种子重复”阶段，但不建议仅凭 seed 3407 就直接扩大到全部 NuPlan。单次规划耗时比 B10 增加约 `26.1%`，也需在大规模训练前单独核查。

结果文件：

```text
tmp/mini_train_pilot_1000_seed3407_v1/rl_safetygate03_anchor005_from_b10/
  b10_vs_rl_safetygate03_anchor005_fixed59.json
  rl_v2_train_and_gate59.log
```

### 16.40 扩大到 10,000 NPZ 的 RL v3 结果（2026-08-21 19:48 CST）

#### 实验目的

在保持 RL 算法、起点模型、超参数和评测场景不变的情况下，将 RL 训练数据从 `1,000 NPZ` 扩大到 `10,000 NPZ`，验证数据规模是否能够保持或增强正收益。

#### 实际配置

```text
train cache:                  tmp/mini_train_balanced_10000_seed3407_v1/cache
manifest:                     tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json
NPZ 数量:                     10,000
监督模型起点:                 HDP B Epoch 10
RL epoch:                     2
learning rate:                4e-7
planning_hybrid_loss:         0.01
rl_detach_window_size:        0
rl_freeze_encoder:            true
rl_group_size:                32
rl_rollout_steps:             6
rl_buffer_update_epoch:       2
rl_buffer_size:               1024
rl_max_update_steps_per_epoch: 500
reward_safety_gate_threshold: 0.3
expert_anchor_weight:         0.05
seed:                         3407
```

训练实际完成 `500` 个 update step；最终训练损失约为 `0.000151`，replay buffer 达到 `1024`。最终 checkpoint：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate03_anchor005_10k_from_b10/
  training_log/hdp-rl-safetygate03-anchor005_10k-from-full-mini-b10/2026-08-21-18:47:58/
  model_epoch_2_trainloss_0.0002.pth
```

第一次启动时，脚本的 NPZ 参数索引存在错误，导致训练前置检查直接失败；该错误已修复，随后重新启动。实际训练和闭环评测均使用修复后的脚本完成，没有使用失败启动产生的 checkpoint。

#### 固定 59 场景闭环结果

gate20 和 gate39 均成功完成，合并后的比较文件为：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate03_anchor005_10k_from_b10/
  b10_vs_rl_safetygate03_anchor005_10k_fixed59.json
```

| 指标 | B Epoch 10 | 10k RL v3 | 差值（RL - B10） |
|---|---:|---:|---:|
| NuPlan score | 0.912424 | **0.913292** | **+0.000868** |
| 沿专家路线前进率 | 0.919741 | **0.922486** | **+0.002744** |
| TTC 合规率 | 0.949153 | 0.949153 | 0 |
| 无责任碰撞率 | 0.957627 | 0.957627 | 0 |
| 可行驶区域合规率 | 0.966102 | 0.966102 | 0 |
| 速度限制合规率 | 0.992623 | 0.992570 | -0.000053 |
| 舒适性 | 1.000000 | 1.000000 | 0 |
| 单次规划耗时 | 0.207575 s | 0.264552 s | +0.056976 s |

逐场景配对统计：

- score：`15` 胜、`9` 负、`35` 平。
- 沿专家路线进度：`15` 胜、`8` 负、`36` 平。
- TTC、无责任碰撞、可行驶区域和舒适性：均为 `0` 个场景发生好坏互换。
- 固定 59 场景上 score 差值的 `200,000` 次 paired bootstrap 95% percentile 区间为 `[+0.000183, +0.001720]`，不包含 0；但这仍不是完整 NuPlan 测试结论。

#### 与 1,000 NPZ RL v3 的对比

两者使用同一批固定 59 场景、同一 B Epoch 10 起点和同一 RL v3 配置。10k RL 相对 1k RL：

- score 平均再提高 `+0.000034`，逐场景 `14` 胜、`10` 负、`35` 平。
- 沿专家路线进度平均再提高 `+0.000245`。
- TTC、碰撞和可行驶区域指标保持完全一致。
- 速度限制合规率下降约 `0.000100`，属于很小的波动。

#### 阶段结论

扩大到 `10,000 NPZ` 后，RL v3 仍然在相同 59 场景上获得平均正收益，并且相较 1k RL 的收益没有消失，路线进度还有轻微提升；同时没有引入新的 TTC、碰撞或越界硬失败。因此，10k 结果支持继续做多随机种子和更大评测范围验证，但还不能仅凭这一次 seed 3407 的固定 59 场景结果宣称 RL 已经稳定有效。另一个需要后续处理的问题是单次规划耗时增加约 `27.4%`。

### 16.41 10,000 NPZ 多随机种子复现实验启动（2026-08-21 CST）

为验证 seed 3407 的正收益是否可重复，固定 10k RL v3 的全部训练设置，仅增加 `seed=42` 和 `seed=2026` 两次独立训练。两次训练均从同一个 B Epoch 10 checkpoint 开始，不从 seed 3407 的 RL checkpoint 继续训练。

本次对执行脚本做了两项不改变算法的参数化改造：

1. `run_rl_v2_safety_gate59_from_b10_pilot1000.sh` 支持通过 `RL_SEED` 传入随机种子，默认值仍为 `3407`，因此旧命令行为不变。
2. 增加 `RL_GATE20_ONLY=1`。每个 seed 完成训练后先停止在固定 20 场景，并逐场景检查碰撞、TTC、可行驶区域三个硬安全指标；没有新增硬退化才复用同一 checkpoint 继续 gate39，最终合并为固定 59 场景结果。

新增的可复现入口：

```bash
bash HDP-nuplan/scripts/run_rl_v3_10k_multiseed.sh
```

固定参数如下：

```text
NPZ:                          10,000
seeds:                        42, 2026
pretrained:                   HDP B Epoch 10
RL epochs:                    2
learning rate:                4e-7
group size:                   32
rollout steps:                6
buffer size:                  1024
max update steps per epoch:   500
detach window size:           0
safety gate threshold:        0.3
expert anchor weight:         0.05
```

GPU 检查结果为单张 RTX 4060 Laptop 8GB，因此两个 seed 串行执行，避免同时训练导致显存不足，也避免并发闭环仿真干扰耗时和指标。

实际启动时间为 `2026-08-21 20:04:27 CST`，后台 PID 为 `393685`。启动后确认 seed 42 已进入 `NuPlan RL rollout`，GPU 利用率约为 `79%`，没有出现初始化或显存错误。运行状态与主调度日志分别保存在：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_v3_10k_multiseed.pid
tmp/mini_train_balanced_10000_seed3407_v1/rl_v3_10k_multiseed.log
tmp/mini_train_balanced_10000_seed3407_v1/rl_v3_10k_multiseed_launcher.out
```

各 seed 的训练和评测细节分别写入：

```text
tmp/mini_train_balanced_10000_seed3407_v1/
  rl_safetygate03_anchor005_10k_seed42_from_b10/rl_v2_train_and_gate59.log
  rl_safetygate03_anchor005_10k_seed2026_from_b10/rl_v2_train_and_gate59.log
```

#### seed 42 中间结果（2026-08-21 20:50 CST）

seed 42 已完成 2 epoch RL 训练、gate20 和 gate39，最终固定 59 场景全部成功。gate20 检查显示没有碰撞、TTC 或可行驶区域硬安全退化，因此继续完成 gate39。

| 指标 | B Epoch 10 | seed 42 RL | 差值 |
|---|---:|---:|---:|
| NuPlan score | 0.912424 | **0.912794** | **+0.000370** |
| 沿专家路线进度 | 0.919741 | **0.920659** | **+0.000918** |
| TTC 合规率 | 0.949153 | 0.949153 | 0 |
| 无责任碰撞率 | 0.957627 | 0.957627 | 0 |
| 可行驶区域合规率 | 0.966102 | 0.966102 | 0 |
| 速度限制合规率 | 0.992623 | 0.992771 | +0.000148 |
| 舒适性 | 1.000000 | 1.000000 | 0 |
| 单次规划耗时 | 0.207575 s | 0.201901 s | -0.005674 s |

seed 42 的 score 逐场景统计为 `11` 胜、`13` 负、`35` 平；路线进度为 `8` 胜、`16` 负、`35` 平。也就是说，平均 score 和路线进度仍为正，但逐场景胜负并不占优势，说明收益较弱，不能只依据 seed 42 宣称稳定提升。

seed 42 结果文件：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate03_anchor005_10k_seed42_from_b10/
  b10_vs_rl_safetygate03_anchor005_10k_seed42_fixed59.json
```

截至本记录时，seed 2026 已完成 RL 训练并进入 gate20，尚未生成最终固定 59 场景结果。

#### 多 seed 复现实验最终结果（2026-08-21 21:11 CST）

seed 2026 已完成 2 epoch RL 训练和 gate20，但安全门禁失败，因此按预设流程没有继续 gate39。seed 42 的完整 59 场景结果保留，seed 2026 仅有 gate20 结果。

| seed | gate20 score 差值 | gate20 路线进度差值 | TTC 变化 | 处理结果 |
|---:|---:|---:|---:|---|
| 42 | -0.000115 | -0.000697 | 0 | 通过安全门禁，完成固定59 |
| 2026 | -0.014699 | +0.003845 | **-0.050000** | **安全门禁失败，跳过gate39** |

seed 2026 的硬退化来自场景 `3713735b94cf5a7b`：

```text
TTC 合规率：B Epoch 10 = 1.0，seed 2026 RL = 0.0
场景 score：B Epoch 10 = 0.909614，seed 2026 RL = 0.598826
```

seed 2026 的 gate20 其余情况为：无责任碰撞率没有下降、可行驶区域合规率没有下降，路线进度平均提高 `+0.003845`，但不能抵消 TTC 硬退化。结果文件：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate03_anchor005_10k_seed2026_from_b10/
  closed_loop20/b10_vs_rl_safetygate03_anchor005_10k_seed2026_gate20.json
```

多 seed 结论：seed 3407 和 seed 42 在固定 59 场景上得到小幅平均正收益，但 seed 2026 在仅 20 个场景的预检中出现 TTC 退化。因此当前 `10k + 2 epoch + 当前 RL 超参数` 不能被认定为稳定有效；下一步应优先分析 `3713735b94cf5a7b` 的动作、奖励和采样轨迹，定位安全门没有阻止该退化的原因，再调整 RL 更新强度或安全约束后重跑，而不是直接扩大到完整 NuPlan。

详细诊断见：`HDP-nuplan/doc_hdp_nuplan/RL_TTC退化场景分析_3713735b94cf5a7b.md`。诊断确认该场景没有实际碰撞，退化来自 NuPlan 恒速投影 TTC 从 `1.3 s` 降至 `0.9 s`；当前 `risk_reward` 的归一化阈值 `0.3` 与官方 `least_min_ttc=0.95 s` 并不等价。

### 16.42 按 TTC 分析结果实现硬安全门（2026-08-22 CST）

根据上述退化分析，新增物理单位的候选级 `min_ttc_seconds`：

```text
min_ttc_seconds = 候选轨迹与有效动态目标在整个预测时域内的最小预计 TTC
```

实现规则：

- OBB 已重叠时记为 `0 s`；
- 有效目标但没有 closing speed 时记为 `inf`；
- 无有效动态目标时记为 `inf`；
- 当前 RL 脚本要求 `min_ttc_seconds >= 1.0 s` 才能通过 safety gate；
- 原有归一化 `risk_reward` 仍保留，用于连续奖励和旧的风险资格条件。

涉及文件：

```text
HDP-nuplan/hdp_nuplan/rl/reward.py
HDP-nuplan/train_predictor_rl.py
HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh
HDP-nuplan/scripts/compare_checkpoint_behavior.py
HDP-nuplan/tests/test_rl_components.py
```

当前启动脚本显式传入：

```bash
--reward_safety_gate_min_ttc_seconds 1.0
```

新增日志指标包括 `min_ttc_seconds` 和 `safety_gate_ttc_eligible`，便于确认候选是因为物理 TTC 还是归一化 risk 未通过门禁。

验证结果：

```text
组件测试：24 passed
完整 HDP-nuplan 测试：48 passed，15 warnings
bash -n：通过
git diff --check：通过
```

这次只完成了代码实现和单元测试，还没有用新硬门重新训练 seed 42/2026；下一步应保持 10,000 NPZ 和其他参数不变，重新执行两个 seed 的训练与 gate20/gate39 对比。

### 16.43 按硬 TTC 门禁和更强 expert anchor 重跑（2026-08-22 CST）

用户确认执行上一节提出的方案。本次实验保持 `10,000 NPZ`、B Epoch 10 起点、2 个 RL epoch、训练/评测数据和两个随机种子不变，只做两项针对性修改：

1. 在 candidate safety gate 中加入物理单位门槛：`min_ttc_seconds >= 1.0 s`；
2. 将 `rl_expert_anchor_weight` 从 `0.05` 提高到 `0.10`，抑制少量随机更新把策略推向危险区域。

这里的 `min_ttc_seconds` 是基于候选轨迹与有效动态邻车的 OBB 几何关系和 closing speed 估计得到的整个预测时域最小 TTC。它不是评测器的闭环 TTC 替代品，而是训练阶段的候选过滤条件；最终仍必须通过固定场景闭环评测确认。

实际启动命令为：

```bash
RL_MULTI_LOG_TAG='rl_v4_ttc1_anchor01_10k_multiseed' \
RL_MULTI_VARIANT_PREFIX='safetygate_ttc1_anchor01_10k' \
RL_MULTI_EXPERT_ANCHOR_WEIGHT=0.1 \
bash HDP-nuplan/scripts/run_rl_v3_10k_multiseed.sh
```

本次使用的两个 seed 为 `42` 和 `2026`。为避免覆盖旧结果，输出目录使用新前缀：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed42_from_b10/
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/
```

启动信息：

```text
launcher PID: 16203
launcher log: tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_ttc1_anchor01_10k_multiseed_launcher.out
master log:   tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_ttc1_anchor01_10k_multiseed.log
```

启动后首次检查时，seed 42 正在 rollout 阶段，约处理到 `1429/5000` 个 batch，GPU 利用率约 `75%`，进程正常运行，无 traceback 或 ERROR。待两个 seed 完成后，将分别比较 score、路线进度、TTC、碰撞、可行驶区域合规率，并重点检查场景 `3713735b94cf5a7b` 是否仍出现 TTC 退化。

#### 最终结果（2026-08-22 13:10 CST）

两个 seed 均完成 2 epoch RL 训练、gate20 和 gate39，最终都得到固定 59 场景结果；全部仿真成功，没有 failed simulation。

| seed | B10 score | RL score | score 差值 | 路线进度差值 | TTC | 碰撞 | 可行驶区域 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.912424 | 0.912580 | **+0.000156** | **+0.000115** | 0 | 0 | 0 |
| 2026 | 0.912424 | 0.913835 | **+0.001411** | **+0.004842** | 0 | 0 | 0 |
| 两 seed 平均 | 0.912424 | 0.913208 | **+0.000783** | **+0.002478** | 0 | 0 | 0 |

表中 TTC、碰撞和可行驶区域列表示相对 B10 的指标差值。三项均为 `0`，即新 RL checkpoint 没有降低固定 59 场景上的硬安全合规率。

逐场景 score 胜/负/平：

```text
seed 42:   10 / 14 / 35
seed 2026: 19 /  5 / 35
```

退化回归场景 `3713735b94cf5a7b` 的 NuPlan 官方闭环 TTC 结果：

| 模型 | 最小 TTC | TTC 指标 | 场景 score |
|---|---:|---:|---:|
| B Epoch 10 | 1.3 s | 1.0 | 0.909614 |
| 新 RL seed 42 | **1.6 s** | 1.0 | 0.906827 |
| 新 RL seed 2026 | **1.1 s** | 1.0 | 0.910097 |

原设置下 seed 2026 在该场景的最小 TTC 为 `0.9 s`、TTC 指标为 `0`；新设置将其恢复到 `1.1 s`、TTC 指标为 `1`。因此，本次改造确实消除了已知的 TTC 回归，并让两个 seed 都获得平均正 score，而不只是让安全门脚本放行。

需要保留的限制：seed 42 的平均提升很小，而且逐场景仍是负多于胜；当前证据只覆盖两个 seed 和固定 59 个 mini-val 场景。可以得出的结论是“硬 TTC 门禁加更强 expert anchor 明显改善了随机种子稳定性，并在当前评测集上保持正收益”，还不能外推为完整 NuPlan 上稳定正收益。

最终结果文件：

```text
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed42_from_b10/
  b10_vs_rl_safetygate_ttc1_anchor01_10k_seed42_fixed59.json
tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/
  b10_vs_rl_safetygate_ttc1_anchor01_10k_seed2026_fixed59.json
```

### 16.44 扩大到固定 200 个 mini-val 场景评测（2026-08-22 CST）

用户确认继续扩大评测范围。本次不重新训练 RL，而是冻结已经完成的两个 checkpoint，在完全相同的 200 个 mini-val 场景上重新评测：

```text
B Epoch 10
RL seed 42：safetygate_ttc1_anchor01_10k
RL seed 2026：safetygate_ttc1_anchor01_10k
```

场景清单构建规则：

- 来源为官方 mini-val 的 10 个 validation log 配置；本机当前可用并通过 NuPlan goal 校验的场景池共 `32,407` 个；
- 原固定 59 个场景全部保留，避免新旧实验失去可比性；
- 从剩余场景中按场景类型和日志数量做确定性贪心均衡，补足到 200 个；
- 评测 filter 使用 200 个显式 scenario token，`shuffle=false`，不会因为数据库顺序或随机种子改变评测集合。

生成文件：

```text
HDP-nuplan/hdp_nuplan/config/scenario_filter/mini-val-fixed-200-rl-v4.yaml
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/mini_val_fixed_200_rl_v4_manifest.json
```

评测脚本：

```text
HDP-nuplan/scripts/run_fixed_200_eval_rl_v4.sh
```

脚本会依次运行 B Epoch 10、RL seed42、RL seed2026，并使用同一个 `closed_loop_nonreactive_agents` 和同一个 200 场景 filter，最后生成配对汇总 JSON。当前已完成脚本语法检查和 filter preflight，正式闭环评测随后启动；结果需要同时满足平均 score、路线进度和 TTC/碰撞/可行驶区域安全指标要求后才能判断扩大评测是否支持 RL 正收益结论。

### 16.45 200 场景评测首次启动被 GPU 环境阻断（2026-08-22 19:06 CST）

评测脚本已经成功读取自定义 filter，并确认 NuPlan 构建出 `200 scenarios`；但在实例化第一个 HDP planner 时退出：

```text
AssertionError: cuda is not available
nvidia-smi: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

因此本次没有产生有效评测结果，也没有修改任何 checkpoint。原因是当前本地会话的 NVIDIA 驱动/GPU 不可用，而 `HyperDiffusionPlanner` 明确要求 CUDA；不能用 CPU 冒充完成该闭环评测。评测输入、200 场景 filter 和启动脚本均已保留，GPU 恢复后直接重新运行即可。

### 16.46 GPU 恢复后重新启动 200 场景评测（2026-08-22 21:29 CST）

GPU 已恢复并完成验证：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Driver: 580.173.02
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
```

已使用新的输出目录重新启动评测，避免与前一次 GPU 不可用的残留目录混淆：

```text
PID: 36078
输出目录: tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_fixed200_eval_retry/
```

当前 B Epoch 10 已成功构建 200 个 simulation，正在顺序执行 200 个闭环场景；启动后 GPU 显存约 `280 MiB`，进程正常，无错误。完成 B10 后脚本会继续评测 seed42、seed2026，最后生成 `b10_vs_rl_fixed200.json`。

### 16.47 200 场景评测显存占用说明（2026-08-22 CST）

评测过程中 `nvidia-smi` 显示 RTX 4060 Laptop GPU 总显存约 `8188 MiB`，实际占用约 `332 MiB`；计算进程为评测 Python 进程，说明 CUDA planner 确实正在使用 GPU。

显存没有被填满是正常现象：当前闭环配置是 sequential worker，每次只处理一个场景；HDP 推理不保存反向传播图，也没有训练 batch，因此只需要保存模型权重、当前场景张量和扩散采样的中间张量。8 GB 是显存容量上限，不是程序必须占用的数量。当前评测速度主要受 NuPlan CPU 仿真、场景读取和 sequential 配置影响，而不是显存容量不足。

### 16.48 固定 200 场景评测中间进度（2026-08-23 00:13 CST）

B Epoch 10 已完成全部 200 个场景，`200/200` simulation 成功、0 失败。中间基线指标为：

| 指标 | B Epoch 10 fixed200 |
|---|---:|
| score | 0.904567 |
| 路线进度 | 0.912068 |
| TTC 合规率 | 0.935000 |
| 无责任碰撞率 | 0.957500 |
| 可行驶区域合规率 | 0.950000 |
| 舒适性 | 0.995000 |

RL seed42 随后启动，当前约完成 `7/200`；seed2026 尚未开始。最终配对 JSON 尚未生成，因此现在只能确认 B10 基线，不能提前判断 RL 的 200 场景收益。

### 16.49 重新核算 200 场景评测耗时（2026-08-23 CST）

“约 5 小时”不是固定理论值，而是根据实际运行日志估算：

```text
B Epoch 10：21:29:47 开始提交 200 个 simulation
B Epoch 10：00:06:32 完成 200 个 simulation
耗时约 2 小时 37 分钟，平均约 47 秒/场景
```

随后 RL seed42 的单场景耗时约 `50～60 秒`，因此一个 RL checkpoint 预计约 `2.8～3.3 小时`。seed42 完成后还要顺序评测 seed2026，所以从 B10 已完成时点起，两个 RL checkpoint 合计约 `5～6.5 小时`。

耗时较长的直接原因是评测命令明确使用：

```text
worker=sequential
number_of_cpus_allocated_per_simulation=1
number_of_gpus_allocated_per_simulation=1
```

这意味着 200 个闭环场景逐个运行；每个场景包含多次 NuPlan 状态推进、HDP 扩散推理、地图/路线处理和指标计算。GPU 显存并不是瓶颈，主要耗时来自 CPU 仿真和顺序执行。当前实验不改变并行配置，以保证 B10、seed42、seed2026 的配对结果和可复现性；后续可单独做并行评测速度消融。

### 16.50 关于并行评测的判断（2026-08-23 CST）

NuPlan 并不是不能并行；它提供了 `single_machine_thread_pool`，可以同时提交多个 simulation。单张 8 GB GPU 上建议先从 2 个并发任务做小规模速度和一致性验证。

并行的代价与风险：

- 多个任务会同时在同一张 GPU 上执行，模型和中间张量显存会叠加；
- 多线程 CUDA 推理可能产生资源竞争，不能只看显存，还要检查仿真失败数和结果一致性；
- 评测输出、metric callback 和随机数调用需要确认不会发生竞争；
- 同一 filter 下场景集合仍然可以保持一致，但执行顺序可能改变。

因此当前已启动的正式 fixed200 对比不在中途切换；等本次结果完成后，再用固定 10～20 个场景比较 sequential 与 2-worker thread pool，确认 `0 failed simulation` 且指标一致，再决定是否用并行重跑全量。

### 16.51 AutoDL 代码、checkpoint 与 fixed200 评测同步（2026-08-23 CST）

目标：在 AutoDL 主机上使用与本地相同的 HDP 代码、NuPlan devkit、B Epoch 10/RL seed2026 checkpoint、200 场景筛选器和 mini DB，启动 RL seed2026 的 200 场景闭环评测。

#### 同步目标

- SSH：`root@connect.cqa1.seetacloud.com:11156`，实例主机 `autodl-container-b8e548a0d5-61312194`。
- 远端项目：`/root/autodl-tmp/workspace/Diffusion-Planner`。
- 远端 Python：`/root/autodl-tmp/conda_envs/diffusion_planner/bin/python`，Python `3.9.25`，RTX 4090 可见。
- 远端 NuPlan 数据：`/root/autodl-tmp/nuplan/dataset`；地图：`/root/autodl-tmp/nuplan/maps`。
- 代码提交：`b13841bd57586849bcd9a62e591f0f4a7da0c69b`。
- NuPlan devkit 提交：`e9241677997dd86bfc0bcd44817ab04fe631405b`。
- fixed200 filter：`mini-val-fixed-200-rl-v4`；manifest 中场景数为 `200`。

#### 一致性校验

以下关键文件已在本地与远端逐一计算 SHA256，结果一致：

- `HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py`
- `HDP-nuplan/hdp_nuplan/rl/loss.py`
- `HDP-nuplan/hdp_nuplan/model/hyper_diffusion_planner.py`
- `HDP-nuplan/scripts/run_mini_closed_loop.sh`
- `mini-val-fixed-200-rl-v4.yaml`
- `mini_val_fixed_200_rl_v4_manifest.json`
- RL seed2026 的 `args.json`
- RL seed2026 的 `model_epoch_2_trainloss_0.0003.pth`

已同步的关键 checkpoint SHA256：

```text
B Epoch 10: 22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce
RL seed2026: 8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c
```

#### 发现并处理的问题

1. 第一次源码整体归档包含约 `1.68 GiB` 已跟踪历史数据，已停止该低效传输；改为只同步评测依赖的代码、配置、脚本和测试。
2. 第一次远端启动使用了错误的相对 NuPlan 数据路径，启动前置检查立即退出，没有执行场景；该日志保留在 `rl_v4_autodl_seed2026_fixed200_eval/remote_eval.log`。
3. 第二次启动改用正确绝对路径，但远端 mini DB 正在断点同步，SQLite 报 `database disk image is malformed`。该次也在场景构建阶段退出，没有产生有效评测结果。
4. 当前保留原有 mini DB 断点同步任务，不删除或覆盖其数据；待目标 DB 完整后先执行 SQLite `PRAGMA integrity_check`，再重新启动评测。

#### 当前状态

- 本地 RL seed42 fixed200：约 `144/200`，仍在运行；不停止。
- 本地 B10 fixed200：已完成 `200/200`，结果目录约 `3.9G`。
- AutoDL：GPU 空闲，远端 RL seed2026 当前未运行，等待 DB 完整性检查通过后重启。
- B10 完整 simulation_log 的上传因 mini DB 断点同步占用链路而暂缓，checkpoint、代码、filter、manifest 和 metric 配置已同步。

补充：远端原有完整 mini DB 同步随后被暂时暂停，保留 `--append-verify` 可续传；fixed200 所需 4 个缺失 DB 改用 gzip 流式传输。已部署远端等待脚本 `HDP-nuplan/tmp/remote_fixed200_watch.sh`，其行为是：等待 4 个 DB 字节数与本地一致，执行 SQLite `PRAGMA integrity_check`，通过后自动启动 `rl-seed2026-closed-loop200-autodl-v3`。截至记录时，第一个目标 DB 已完整到达，第二个正在传输，评测尚未启动。

### 16.52 当前状态快照（2026-08-23 02:15 CST）

- 本地 RL seed42 fixed200：`158/200`，仍在运行。
- AutoDL required DB：第 1 个已完成；第 2 个约 `96/282 MB`；第 3、4 个等待 gzip 流继续写入。
- AutoDL watcher：PID `4574`，持续等待并在完整性检查通过后自动启动 seed2026 fixed200。
- AutoDL 当前尚未产生有效 RL seed2026 评测结果；GPU 仍未进入正式评测阶段。

### 16.53 AutoDL fixed200 评测失败诊断（2026-08-23 CST）

4 个缺失 DB 完成传输后，4 个 DB 的 `PRAGMA integrity_check` 均为 `ok`，并且 8 个 fixed200 相关 DB 与本地逐个 SHA256 一致。为隔离其他尚未同步完成的 mini DB，远端建立了 `/root/autodl-tmp/nuplan/mini_fixed200_db`，其中仅包含实际覆盖 200 个 manifest token 的 8 个 DB 硬链接。

随后启动了 AutoDL RL seed2026 fixed200 v4。场景构建成功，日志明确显示 `Building simulations from 200 scenarios`，但闭环阶段累计 `201` 条失败记录，主要错误为：

```text
AssertionError: angle is not finite
numpy.linalg.LinAlgError: SVD did not converge
```

这表明该次不能产生有效指标，已停止，不作为 RL 结果使用。随后用同一远端环境、同一批 3 场景和 seed42 checkpoint 做最小复现，GPU 与 CPU 两种模式均为 `4` 个场景失败；因此暂时不能把问题归因于 seed2026 checkpoint 或 4090 GPU。当前需要继续检查远端运行时的 planner 输入/模型输出，以及与本地已成功闭环的场景 filter 选择是否完全一致。

### 16.54 fixed200 场景集合直接比对（2026-08-23 CST）

为确认“远端成功构建 200 个场景”是否与本地完全相同，将远端 v4 的 `runner_report.parquet` 拉回本地，并按 `log_name + scenario_name` 排序后逐条比较。

结果：

- 本地报告：200 行，200 个成功、0 个失败；
- 远端报告：200 行，0 个成功、200 个失败；
- 两份报告的场景键集合完全相同；
- 两份报告的场景键 SHA256 均为：

```text
02007777c7d4d54f6f194ccc6787b9d1e3e76a840088e584da159c738ed3f28c
```

结论：远端 v4 与本地使用的是完全相同的 200 个场景；差异不在场景选择，而在远端闭环执行阶段全部失败。因此远端 v4 不能用于比较规划指标，只能证明场景集合和同步输入一致。

### 16.55 暂停 AutoDL 评测（2026-08-23 CST）

用户已关闭 AutoDL 实例，当前放弃远端 fixed200 评测，不再继续启动或排查远端运行环境。

当前结论：

- 云端与本地的 200 个场景集合已确认完全一致；
- 云端 v4 评测的 200 个场景均在闭环执行阶段失败，未产生可用规划指标；
- 后续实验暂时全部回到本地环境进行；
- 云端已同步的代码、checkpoint、配置和 200 场景相关 DB 保留在云端数据盘中，之后如需恢复可继续使用。

### 16.56 本地 fixed200 评测进度更新（2026-08-23 CST）

- B Epoch 10：`200/200` 成功；
- RL seed42：`200/200` 成功，`0` 个失败；
- RL seed2026：已开始执行，当前约 `19/200`，尚未生成最终报告；
- 三组评测使用同一个 `mini-val-fixed-200-rl-v4` 场景集合，完成后再进行统一指标汇总。

### 16.57 设置评测完成后的自动分析（2026-08-23 CST）

为避免只查看均值而忽略逐场景差异，在 fixed200 三组评测完成后自动执行：

1. 检查 B Epoch 10、RL seed42、RL seed2026 是否均为 200 个成功场景；
2. 以 B Epoch 10 为基线，按相同 `scenario` 计算两个 RL seed 的逐场景指标差值；
3. 统计每项指标的均值差、 中位数差和胜/负/平场景数；
4. 根据两个 seed 的 `score` 和责任碰撞指标，给出 `consistent_positive`、`mixed` 或 `not_positive` 状态。

本次评测因启动时旧脚本已经在运行，另启动后台等待器 `auto_analysis_watcher.pid`；它等待三组评测汇总文件生成后运行同一分析程序。后续新启动的 `run_fixed_200_eval_rl_v4.sh` 已直接内置该自动分析步骤。

### 16.58 fixed200 三组最终结果（2026-08-23 CST）

三组评测均完成 `200/200`，失败场景均为 `0`：

| 模型 | mean score | 相对 B10 | score 胜/负/平 | mean progress | mean speed compliance |
|---|---:|---:|---:|---:|---:|
| B Epoch 10 | 0.90456672 | 基线 | - | 0.91206773 | 0.99543114 |
| RL seed42 | 0.90449693 | -0.00006979 | 33/60/107 | 0.91164599 | 0.99555687 |
| RL seed2026 | 0.90606500 | +0.00149828 | 71/23/106 | 0.91726361 | 0.99523515 |

安全和基本合规指标在三组间保持一致：责任碰撞 `0.9575`、drivable area `0.95`、making progress `0.99`、行驶方向 `1.0`、TTC `0.935`、舒适性 `0.995`。

自动判断为 `mixed`：seed42 的总体得分几乎持平但略降，seed2026 小幅提升，两个随机种子方向不一致。当前证据表明 RL 确实改变了部分场景行为，但不足以宣称 RL 在 fixed200 上稳定带来正收益；seed2026 的主要收益来自路线进度提升，同时限速合规有极小下降。

### 16.59 RL 不稳定原因与改进优先级（2026-08-23 CST）

根据两次训练日志、checkpoint 参数差异和 fixed200 配对结果，当前不稳定主要不是“RL 天生无效”，而是实现配置造成了高方差、低有效信号：

1. `rl_buffer_size=1024`，但 rollout 数据集有约 `10000` 个场景。`NuPlanReplayBuffer` 使用 `deque(maxlen=1024)`，因此一次 rollout 后只保留最后 1024 个场景；不同 seed 的 `DistributedSampler` 顺序不同，导致两个 seed 实际更新的数据子集不同。
2. 只训练 2 个 epoch，按 `rl_buffer_update_epoch=2` 实际只有 1 次 rollout 和 1 次 update，统计平均不足。
3. 两次 update 的 `reward_std_mean` 只有约 `0.0043/0.0057`，而 `rl_min_reward_std=1e-6` 仍把几乎所有组判为有效；组内极小差异被标准化后可能放大数值/采样噪声。
4. RL loss 约 `-2e-5`，expert anchor 的加权项约 `3.3e-4`，后者约大一个数量级；checkpoint 相对 B10 的参数变化也只有约 `1.2e-4` 的相对 L2，更新过于保守。
5. reward 的 `base_rewards` 实际由 risk、follow、lane 和 progress guard 组成；collision、route、comfort 等量在当前实现中主要作为诊断详情，未直接进入总 reward。因此优化目标与 NuPlan 的 speed-limit、舒适性等闭环指标并不完全一致。

建议顺序：先将 buffer 扩大到覆盖完整 rollout 数据集，并保持其他参数不变做控制实验；随后增加到至少 4～6 个 epoch，再根据 reward 标准差分布调整 `rl_min_reward_std` 和候选多样性。不要同时修改 reward 权重、学习率和训练轮数，否则无法判断收益来源。

### 16.60 启动 buffer 覆盖范围控制实验（2026-08-23 CST）

已启动独立入口：`HDP-nuplan/scripts/run_rl_buffer10k_control_multiseed.sh`。

实验只改变：

```text
rl_buffer_size: 1024 -> 10000
```

其余参数保持上一轮 safetygate/anchor 配置不变：2 epoch、`learning_rate=4e-7`、`rl_expert_anchor_weight=0.1`、`rl_detach_window_size=0`、冻结 encoder、`rl_group_size=32`、采样噪声 `0.1`。

原先误启动了多 seed 总控；发现流程顺序不符合当前阶段目标后，已停止 seed42 的未完成进程，不使用其未完成结果。当前只运行之前约定的单 seed `2026`，输出目录为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_anchor01_2ep_seed{seed}_from_b10
```

当前 seed2026 已开始 rollout，进度约 `47/5000` batch；GPU 训练进程正常运行。seed42 的部分日志保留在对应目录，但没有生成可用 checkpoint。当前单 seed PID 保存在 `rl_buffer10k_control_single_seed2026.pid`。

训练结果更新：单 seed2026 已完成 2 epoch 控制实验。Epoch 1 rollout 的 `buffer_size=10000`，Epoch 2 完成 `500` 个 update step，并生成 checkpoint：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_anchor01_2ep_seed2026_from_b10/training_log/hdp-rl-buffer10k_anchor01_2ep-seed2026-from-full-mini-b10/2026-08-23-11:47:11/model_epoch_2_trainloss_0.0003.pth
```

本次 update 指标：`loss=0.0003151`、`rl_loss=-0.00002145`、加权 expert anchor loss `0.00033655`、`reward_std_mean=0.00635`、`buffer_size=10000`。当前尚未启动 fixed200 闭环评测。

### 16.61 启动 6 epoch 与 reward std 分位数实验（2026-08-23 CST）

为完成步骤 2～4，已从同一个 B Epoch 10 checkpoint 重新启动单 seed2026 的 6 epoch 训练：

- `rl_buffer_size=10000`；
- `train_epochs=6`，即 3 次 rollout 和 3 次 update；
- 初始 `rl_sampling_noise_scale=0.1`；
- 其他参数保持步骤 1 控制实验不变；
- update 指标新增 `reward_std_p10`、`reward_std_p50`、`reward_std_p90`。

训练输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_6ep_seed2026_from_b10
```

当前已开始第一个 rollout，仍未进行 fixed200 评测。第一个 6 epoch 阶段完成后，根据 reward std 分位数判断是否启动 `noise_scale=0.2` 的第二阶段；最终只评测通过步骤 1～4 确定的 checkpoint。

`noise_scale=0.1` 的 6 epoch 阶段已完成，生成 Epoch 2/4/6 三个 checkpoint。三个 update 的 reward std 分位数为：

| Epoch | p10 | p50 | p90 | mean |
|---:|---:|---:|---:|---:|
| 2 | 0.00190 | 0.00635 | 0.01079 | 0.00635 |
| 4 | 0.00182 | 0.00588 | 0.00994 | 0.00588 |
| 6 | 0.00155 | 0.00499 | 0.00844 | 0.00499 |

候选组内 reward 差异没有扩大，反而逐轮收缩；相对平均 reward 约 `5.4` 仍然很小。因此触发步骤 4：保持 B10 起点、buffer、epoch、reward、anchor 和学习率不变，只把 `rl_sampling_noise_scale` 从 `0.1` 提高到 `0.2`，重新启动单 seed2026 的 6 epoch 训练。

第二阶段输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_6ep_noise02_seed2026_from_b10
```

当前第二阶段已开始第一个 rollout；仍未启动 fixed200 评测。

`noise_scale=0.2` 的 6 epoch 阶段已完成。reward std 分位数如下：

| Epoch | p10 | p50 | p90 | mean |
|---:|---:|---:|---:|---:|
| 2 | 0.00427 | 0.01392 | 0.02356 | 0.01392 |
| 4 | 0.00385 | 0.01278 | 0.02170 | 0.01278 |
| 6 | 0.00348 | 0.01208 | 0.02067 | 0.01208 |

相对于 `noise_scale=0.1` 的 Epoch 6（p50=`0.00499`、p90=`0.00844`），候选 reward 区分度约提高到 2 倍以上。步骤 1～4 已完成，选用 noise02 的 Epoch 6 checkpoint 开始 fixed200 闭环评测：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_noise02_fixed200_eval
```

该评测与已有 B Epoch 10 结果使用同一个 `mini-val-fixed-200-rl-v4` 场景集合；评测完成后再比较 score、安全、进度和舒适性指标。

首次启动因评测脚本中 checkpoint 目录名的连字符/下划线写法错误而立即退出，未执行任何场景；已修正路径并重新启动。当前日志已显示 `Building simulations from 200 scenarios`，正式进入 fixed200 评测阶段。

fixed200 已完成：`200/200` 成功、`0` 个仿真失败。结果如下：

| 指标 | B Epoch 10 | RL buffer10k/noise02 Epoch 6 | 差值 |
|---|---:|---:|---:|
| score | 0.90456672 | 0.89151622 | -0.01305050 |
| no ego-at-fault collision | 0.9575 | 0.9575 | 0 |
| drivable area compliance | 0.9500 | 0.9350 | -0.0150 |
| progress along expert route | 0.91206773 | 0.91309464 | +0.00102691 |
| TTC | 0.9350 | 0.9350 | 0 |
| speed-limit compliance | 0.99543114 | 0.99548354 | +0.00005240 |
| comfort | 0.9950 | 0.9950 | 0 |

score 逐场景为 `55/39/106` 胜/负/平。总分下降由 3 个 `following_lane_without_lead` 场景触发 drivable-area 从 `1` 降到 `0` 导致，这 3 个场景都来自日志 `2021.10.05.07.10.04_veh-52_01442_01802`。排除这 3 个硬退化场景后，其余 197 个场景的 score 平均差为 `+0.00031412`。

结论：扩大 buffer、增加 update 轮数和提高采样噪声后，RL 在大多数场景中呈现小幅正向趋势，但当前 reward 缺少 drivable-area 硬约束，少数越界场景会通过 NuPlan 乘法/硬门控指标把总收益完全抹掉；当前 checkpoint 不能判为正收益模型。

### 16.62 负收益原因与实际改动核对（2026-08-23 CST）

将此前 fixed200 小幅正收益的 seed2026 `args.json` 与本次 noise02 Epoch 6 的 `args.json` 逐项比较，除实验名称/输出路径外，行为相关参数只有三个差异：

```text
rl_buffer_size:            1024 -> 10000
train_epochs:                 2 -> 6
rl_sampling_noise_scale:    0.1 -> 0.2
```

未修改 reward 权重、reward 计算逻辑、学习率、expert anchor、encoder 冻结、detach、模型结构、planner 推理噪声或 fixed200 场景集合。代码层面只在 `rl/loss.py` 增加了 p10/p50/p90 日志指标，这些值不参与 total loss 和反向传播。

负收益的数值来源非常集中：3 个场景各自从约 `0.88～0.90` 降为 `0`，三者合计对 200 场景均值造成约 `-0.01336`；其余 197 个场景贡献约 `+0.00031`，最终净差为 `-0.01305`。NuPlan 的 drivable-area 是硬乘法/门控类指标，单个场景从 1 降为 0 会把该场景总分清零。

机制上，buffer 扩大和 3 次 update 使策略比原 1 次 update 更充分地偏离 B10；`noise_scale=0.2` 又让训练候选具有更大的几何差异。当前 reward 用 soft lane reward 和 progress guard 排序，但没有使用 NuPlan drivable-area polygon 的同等硬门槛，因此少数推进更积极但越界的候选仍可能形成更新信号。需要强调：由于最终实验同时改变了 buffer、update 次数和候选噪声，仅凭最终 fixed200 不能严谨地把退化单独归因于其中一个参数；应利用现有 Epoch 2/4/6、noise01/noise02 checkpoint 在 3 个失败场景上做阶梯消融定位。

### 16.63 增加 drivable-area 硬约束并完成训练前验证（2026-08-23 CST）

#### 目标

针对 noise02 Epoch 6 fixed200 中 3 个场景 drivable-area 从 1 降为 0 的问题，
在 RL reward 的 safety gate 中加入候选级 drivable-area 硬约束，然后再重新训练。

#### 数据与实现边界

当前 NPZ 没有保存 NuPlan 原生 drivable-area polygon，只保存了 lane 的中心线、方向
以及到左右边界的相对向量。因此实现的是“基于缓存 lane 边界的 drivable-area 代理硬门”，
不是直接调用 NuPlan polygon scorer。候选车辆包络需要位于至少一条附近 lane corridor
内，且同时满足已有 risk/TTC 门；越界候选在组内存在合格候选时不能通过更高的
progress/follow/lane 奖励重新成为优选。

#### 代码修改

1. `HDP-nuplan/hdp_nuplan/rl/reward.py`

   - 新增 `safety_gate_require_drivable_area` 和
     `safety_gate_drivable_area_margin` 配置；默认关闭，避免改变历史实验。
   - 新增 `_drivable_area_compliance()`：计算候选轨迹每个时间点的自车矩形在 lane
     横向的投影余量，检查最近 8 条 lane，而不是只检查最近 1 条，兼容路口和换道。
   - 将 drivable-area 条件并入 `_apply_safety_gate()` 的 eligible 判定。
   - 如果专家轨迹本身无法被当前缓存 lane corridor 覆盖，则关闭该场景的硬门，避免
     把换道、路口或地图查询范围不足误判成候选越界。
   - 输出 `drivable_area_compliance`、`drivable_area_min_clearance` 和
     `drivable_area_gate_active` 供训练日志审计。

2. `HDP-nuplan/train_predictor_rl.py`

   - 新增两个命令行参数并写入 `args.json`：
     `--reward_safety_gate_require_drivable_area`、
     `--reward_safety_gate_drivable_area_margin`。

3. `HDP-nuplan/scripts/run_rl_buffer10k_6ep_seed2026.sh`

   - 正式重训练命令显式开启 `--reward_safety_gate_require_drivable_area true`；
   - 保留 `margin=0.1`、seed2026、buffer10000、6 epoch、noise0.2、no-detach、
     冻结 encoder 和其余既定参数不变。

#### 训练前验证

执行的代码检查和测试：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m py_compile \
  HDP-nuplan/hdp_nuplan/rl/reward.py HDP-nuplan/train_predictor_rl.py
CUDA_VISIBLE_DEVICES='' /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果：`50 passed`；代码编译通过。

在当前 10,000 NPZ cache 的前 1,000 个样本上，用专家未来轨迹进行 corridor sanity
check：`expert_pass=939/1000`，通过率 `0.939`；最小边界余量 p10=`0.0649978 m`、
p50=`0.6033382 m`，全体最小值=`-1.5934360 m`。

前 200 个样本的专家通过率为 `185/200=92.5%`。最近 1 条 lane 的初版只有
`104/200=52%`，已通过最近 8 条 lane 的候选检查修复，说明该修复不是为了放松
越界判定，而是减少路口/换道时的错误 lane 归属。仍有少量专家轨迹因 lane-change
或缓存覆盖不足无法通过；代码会在这些场景关闭硬门并记录 `drivable_area_gate_active`。

#### 下一步

训练前代码和单元测试已完成，尚未启动长时间 RL 重训练。下一步使用修改后的
`run_rl_buffer10k_6ep_seed2026.sh`，从同一个 B Epoch 10 checkpoint 重新训练，
完成后在同一个 `mini-val-fixed-200-rl-v4` 场景集合闭环评测，重点比较总 score、
drivable-area compliance、3 个原始退化场景、硬门 active/eligible 比例，以及
碰撞、TTC、进度和舒适性指标。

#### 重训练启动记录

第一次启动发现脚本默认 `RL_SAMPLING_NOISE_SCALE=0.1`，与本阶段应沿用的
noise02 配置不一致；该进程运行约 15 秒、尚未完成有效 batch，已安全停止且不作为
实验结果。随后使用 `RL_SAMPLING_NOISE_SCALE=0.2` 正确启动：

```bash
RL_SAMPLING_NOISE_SCALE=0.2 \
RL_RUN_TAG=buffer10k_6ep_gate_drivable_noise02_seed2026 \
nohup bash HDP-nuplan/scripts/run_rl_buffer10k_6ep_seed2026.sh \
  > HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/\
  rl_buffer10k_6ep_gate_drivable_noise02_seed2026_launcher.out 2>&1 &
```

正确训练进程已确认使用：

```text
rl_sampling_noise_scale=0.2
rl_buffer_size=10000
train_epochs=6
rl_detach_window_size=0
reward_safety_gate_require_drivable_area=true
seed=2026
```

当前日志已进入第 1 个 rollout，约 `367/5000` batch，速度约 `12 batch/s`，GPU
进程正常运行；训练完成后再启动 fixed200 闭环评测。

#### 训练性能修正

接入 top-k lane corridor 后，实际训练速度降至约 `3.5 batch/s`，原因是每个 batch
对 32 条候选轨迹和约 1,400 个地图点进行两次全量距离计算。该次训练在第 2 rollout
早期安全停止，已生成的 Epoch 2 checkpoint 保留作中间调试，不作为最终结果。

随后在 `_drivable_area_compliance()` 中对每条 20 点 lane 保留首点、尾点和每隔一个
中间点，将地图点数约减半。重新执行全套测试仍为 `50 passed`；前 1,000 个专家样本
通过率由 `0.939` 变为 `0.942`，说明这是计算优化而不是放松硬门。最终训练使用新的
`...gate_drivable_noise02_seed2026_v2_from_b10` 输出目录，从 B Epoch 10 重新开始。

#### v2 最终训练结果

最终训练于 `2026-08-23 19:19:57 CST` 完成，输出目录为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_6ep_gate_drivable_noise02_seed2026_v2_from_b10/training_log/hdp-rl-buffer10k_6ep_gate_drivable_noise02_seed2026_v2-from-full-mini-b10/2026-08-23-18:49:02/
```

关键 rollout 结果：

| epoch 阶段 | drivable gate active | drivable candidate pass | reward mean | reward std p50 |
|---|---:|---:|---:|---:|
| Epoch 1 rollout | 0.8504 | 0.9518 | 5.4361 | - |
| Epoch 3 rollout | 0.8504 | 0.9583 | 5.5205 | - |
| Epoch 5 rollout | 0.8504 | 0.9605 | 5.5436 | - |
| Epoch 6 update | - | - | 5.2363 | 0.01287 |

三次 update 均完成 `500` 个有效 update step，最终 total loss=`2.7787e-4`、
RL loss=`-2.5914e-5`、expert anchor loss=`3.0379e-3`。训练进程已正常退出，
当前正在执行同一 fixed200 闭环评测。

#### fixed200 评测启动

已启动脚本：

```bash
bash HDP-nuplan/scripts/evaluate_rl_buffer10k_gate_drivable_fixed200.sh
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_gate_drivable_noise02_fixed200_eval/
```

评测确认使用 `mini-val-fixed-200-rl-v4`、`Sequential`、同一 NuPlan mini DB 根目录，
并与 B Epoch 10 采用相同 runner 配置。启动后已完成 `2/200`；单场景约 40–45 秒，
预计总耗时约 2–2.5 小时。评测完成后脚本自动生成
`b10_vs_rl_gate_drivable_fixed200.json`，不需要手动合并结果。

#### fixed200 最终评测结果

评测已于 `2026-08-23 21:52:04 CST` 完成，200 个场景全部成功，失败数为 0。
RL 模型与 B Epoch 10 使用完全相同的 `mini-val-fixed-200-rl-v4` 场景集合和闭环
评测流程。结果如下：

| 指标 | B Epoch 10 | RL + drivable-area gate | 变化 |
|---|---:|---:|---:|
| score | 0.90457 | 0.89144 | -0.01313 |
| drivable_area_compliance | 0.95000 | 0.93500 | -0.01500 |
| no_ego_at_fault_collisions | 0.95750 | 0.95750 | 0 |
| time_to_collision_within_bound | 0.93500 | 0.93500 | 0 |
| ego_progress_along_expert_route | 0.91207 | 0.91248 | +0.00041 |
| speed_limit_compliance | 0.99543 | 0.99555 | +0.00011 |
| ego_is_comfortable | 0.99500 | 0.99500 | 0 |

本次实验没有观察到正收益：综合 score 下降约 `1.31` 个百分点，主要原因是
drivable-area compliance 从 `0.950` 降至 `0.935`，而碰撞、TTC、进度和舒适性
基本没有改善。该结果说明当前 lane-boundary 代理硬约束并没有等价于 NuPlan 的
精确 drivable-area polygon 约束，且 RL 更新仍可能改变闭环轨迹分布。

结果文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_gate_drivable_noise02_fixed200_eval/b10_vs_rl_gate_drivable_fixed200.json
```

#### reward 与 RL 更新目标重构

根据 fixed200 结果，下一轮实验不再启用 lane-boundary 代理形式的
drivable-area gate，仅保留代码、单元测试和诊断字段。训练脚本现在明确传入：

```text
reward_safety_gate_require_drivable_area=false
```

同时增加 `reward_objective_mode=nuplan_aligned`。该模式不再把安全、进度、路线、
舒适性和跟车指标直接相加，而是先转换为 `[0,1]` 质量分，再进行加权几何组合：

```text
R = safety^0.45
    * progress^0.25
    * route^0.15
    * comfort^0.10
    * follow^0.05
```

其中 `safety = risk_reward * no_collision`，碰撞会使安全质量降为 0；舒适性使用
`exp(-comfort_cost)` 转换。drivable-area 不进入该 reward，因为当前 NPZ 没有
NuPlan 原生 drivable-area polygon，继续使用 lane 边界代理会把地图表示误差带入
RL 目标。

RL 更新目标也从旧的指数权重改为 `softmax_positive`：

```text
w_g = G * softmax(advantage_g / temperature)
L_rollout = mean(w_g * L_diffusion_g)
L_total = 1.0 * L_rollout + 0.1 * L_expert_anchor
```

新权重始终为正，组内均值为 1，不再使用 `(w-1)` 产生负的回归权重。专家 anchor
继续使用真实 `ego_future` 的监督损失，限制 reward-weighted self-distillation
偏离 B Epoch 10 的原有能力。新实验的权重温度设为 `0.5`，其余训练设置保持
10,000 NPZ、seed 2026、6 epoch、buffer 10,000、noise 0.2、detach 关闭和
encoder 冻结。

#### 重构后的验证

修改后的代码通过：

```text
52 passed, 15 warnings
```

在真实 10,000 NPZ cache 的前 100 个样本上运行新的 aligned reward sanity check，
结果为 `accepted=true`：

| 检查项 | 通过率 |
|---|---:|
| lateral 增大路线代价 | 99/100 |
| jitter 增大舒适性代价 | 100/100 |
| reverse 降低前进进度 | 88/88 |
| reverse 增大倒退代价 | 88/88 |
| 移动场景 expert 优于 stop | 88/88 |
| collision 增大碰撞代价 | 79/79 |

sanity 报告：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/reward_sanity_nuplan_aligned.json
```

#### 重构后 RL 重训练启动

已从同一个 B Epoch 10 checkpoint 启动新的单 seed RL 训练：

```text
RL_RUN_TAG=buffer10k_6ep_aligned_reward_softmax_seed2026_v1
seed=2026
train_epochs=6
rl_buffer_size=10000
rl_weighting_mode=softmax_positive
rl_reward_temperature=0.5
rl_center_reward_weights=false
rl_expert_anchor_weight=0.1
reward_objective_mode=nuplan_aligned
reward_safety_gate_require_drivable_area=false
rl_detach_window_size=0
rl_freeze_encoder=true
```

训练输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_6ep_aligned_reward_softmax_seed2026_v1_from_b10/
```

启动时已完成 10,000 NPZ、manifest 和 B Epoch 10 checkpoint 校验，rollout 运行
正常，速度约 `12 batch/s`。训练完成后仍需使用相同的 fixed200 场景集合进行闭环
评测，不能仅根据训练 reward 判断是否带来正收益。

#### 重构后 fixed200 闭环评测启动

6 epoch RL 训练已于 `2026-08-23 23:19:22 CST` 正常结束，最终 checkpoint 为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_6ep_aligned_reward_softmax_seed2026_v1_from_b10/training_log/hdp-rl-buffer10k_6ep_aligned_reward_softmax_seed2026_v1-from-full-mini-b10/2026-08-23-22:46:17/model_epoch_6_trainloss_0.0025.pth
```

随后启动与 B Epoch 10 完全相同的 `mini-val-fixed-200-rl-v4` 顺序闭环评测：

```text
HDP-nuplan/scripts/evaluate_rl_aligned_reward_fixed200.sh
```

评测输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_aligned_reward_softmax_fixed200_eval/
```

评测完成后脚本将生成：

```text
b10_vs_rl_aligned_reward_fixed200.json
```

#### 重构后 fixed200 最终评测结果

评测于 `2026-08-24 02:09:54 CST` 完成，200 个场景全部成功，失败数为 0。

| 指标 | B Epoch 10 | aligned reward RL | 变化 |
|---|---:|---:|---:|
| score | 0.90457 | 0.87568 | -0.02889 |
| no_ego_at_fault_collisions | 0.95750 | 0.95250 | -0.00500 |
| drivable_area_compliance | 0.95000 | 0.93500 | -0.01500 |
| time_to_collision_within_bound | 0.93500 | 0.93000 | -0.00500 |
| ego_progress_along_expert_route | 0.91207 | 0.87370 | -0.03837 |
| ego_is_making_progress | 0.99000 | 0.99000 | 0 |
| driving_direction_compliance | 1.00000 | 1.00000 | 0 |
| speed_limit_compliance | 0.99543 | 0.99726 | +0.00183 |
| ego_is_comfortable | 0.99500 | 0.99500 | 0 |

本次重构没有带来正收益，综合 score 相对 B Epoch 10 下降约 `2.89` 个百分点。
虽然 speed-limit compliance 略有提高，但不足以抵消路线进度下降 `0.03837`、
drivable-area compliance 下降 `0.015`，以及碰撞/TTC 各下降 `0.005`。

这一结果否定了当前 `nuplan_aligned` 几何 reward 加 `softmax_positive` 自蒸馏目标
作为主方案。关闭 drivable-area gate 并没有恢复 drivable-area 指标，说明主要问题
不是 gate 本身，而是 reward-weighted rollout 回归仍然使策略偏离了监督模型；此外，
离线 NPZ reward 与真实闭环 NuPlan 指标之间仍存在明显模型误差。

结果文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_aligned_reward_softmax_fixed200_eval/b10_vs_rl_aligned_reward_fixed200.json
```

#### v4 评测集扩展：test14-hard 启动

根据项目已有的 `val14`、`test14-random` 和 `test14-hard` 评测协议，停止继续
扩大失败的 aligned-reward 实验，改用已经在 fixed200 上获得正收益的 v4 配置。
第一阶段先运行 `test14-hard`，同时评测 B Epoch 10、RL v4 seed42 和 RL v4
seed2026；只有困难集没有新增安全退化，才继续运行 `test14-random` 和 `val14`。

启动脚本：

```text
HDP-nuplan/scripts/evaluate_v4_test14_hard.sh
```

评测配置：

```text
filter: test14-hard
explicit scenario tokens: 272
runner: Sequential
planner: HDP
models: B Epoch 10, RL v4 seed42, RL v4 seed2026
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_test14_hard_eval/
```

当前 B Epoch 10 已开始执行，三个模型共约 816 个闭环场景，按当前单卡顺序
评测速度预计需要约 9～11 小时。该时间只是 test14-hard 阶段，后续 random/val14
会根据安全门禁结果决定是否启动。

#### v4 评测集扩展：test14-hard 实际结果与场景覆盖修正

评测于 `2026-08-24 03:30:33 CST` 完成。需要修正启动时的场景数量估计：
`test14-hard.yaml` 中虽然列出了 272 个显式 token，但当前配置的本地 mini DB
实际只解析出 7 个可用场景，因此三个模型各自评测 7 个相同场景，共 21 次闭环，
不是 272 × 3 = 816 次。

| 模型 | 场景数 | 成功/失败 | score | 相对 B10 | route progress | 相对 B10 | 碰撞 | TTC | drivable area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B Epoch 10 | 7 | 7/0 | 0.733806810 | — | 0.916825280 | — | 0.785714 | 0.714286 | 1.000000 |
| RL v4 seed42 | 7 | 7/0 | 0.733512116 | -0.000294693 | 0.915894315 | -0.000930965 | 0.785714 | 0.714286 | 1.000000 |
| RL v4 seed2026 | 7 | 7/0 | 0.734574737 | +0.000767928 | 0.920013698 | +0.003188418 | 0.785714 | 0.714286 | 1.000000 |

在这个 7 场景子集上，两个 RL seed 的碰撞、TTC 和 drivable-area 指标均与 B10
完全相同；seed42 的 score 略降，seed2026 的 score 略升。但样本数只有 7，
不能据此判断 RL 已经稳定带来正收益，也不能把它等同于完整 test14-hard。

结果文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_test14_hard_eval/b10_vs_rl_v4_test14_hard.json
```

#### v4 评测集扩展：test14-random 与 val14 启动

由于当前 mini DB 对官方筛选 token 的实际覆盖有限，先在同一数据源上统计并运行
可用场景，而不把配置文件中的理论数量误认为实际闭环数量：

| 评测配置 | 配置理论范围 | 当前 mini DB 实际可用场景 |
|---|---:|---:|
| test14-random | 14 类 × 20 | 186 |
| val14 | 显式 token / 官方 val split | 12 |

已于 `2026-08-24 03:55:39 CST` 启动三模型配对评测，顺序为 B Epoch 10、RL v4
seed42、RL v4 seed2026；先完成 `test14-random`，再自动完成 `val14`。所有模型
使用同一个 mini DB、同一个 filter 和顺序执行器，保证场景集合与推理流程一致。

启动脚本：

```text
HDP-nuplan/scripts/evaluate_v4_random_val14.sh
```

输出目录与日志：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_random_val14_eval/
```

预计 `test14-random` 约需 3～4 小时，随后 `val14` 约需 20～30 分钟；完成后分别
生成 `b10_vs_rl_v4_test14_random.json` 和 `b10_vs_rl_v4_val14.json`。

启动后复核：`2026-08-24 03:55:47 CST` 已成功进入 `test14-random`，NuPlan
日志确认本轮构建 `186 scenarios`，当前正在评测 B Epoch 10；后台脚本仍在运行，
不会提前进入 `val14`。脚本完成随机集三模型评测并生成汇总后，会自动继续 val14。

`2026-08-24 03:57 CST` 再次检查：当前进度为 B Epoch 10 的 `1/186`，进程 CPU
占用正常，日志没有异常或 traceback；GPU 显存仅约 296 MiB，说明当前闭环执行主要
由 CPU/仿真流程承担，不能用 GPU 利用率判断任务是否卡住。

`2026-08-24 03:59 CST` 检查：B Epoch 10 已推进到 `2/186`，后台进程仍正常运行，
尚未生成该模型的最终 `runner_report.parquet`；因此暂不启动下一个模型，保持严格
顺序和相同场景集合。

`2026-08-24 06:01 CST`，B Epoch 10 的 test14-random 已完成：`186/186`
成功、`0` 失败，耗时约 `2:05:47`，并已生成完整 runner report。原自动脚本随后
没有进入 seed42：原因是在 B10 长时间运行期间修改了同一 Bash 文件，旧 Bash
进程继续读取文件时出现行偏移并报语法错误。这是评测编排操作错误，不是模型、
checkpoint 或 NuPlan 场景失败；B10 结果无需重跑。

脚本已改为可恢复方式：每个阶段启动前检查 `runner_report.parquet`，完整报告存在
就跳过，否则执行该阶段。`2026-08-24 08:55:41 CST` 已恢复，日志确认跳过已完成
的 B10，当前开始 `rl-v4-seed42-test14-random`。恢复后不再修改正在运行的脚本。

`2026-08-24` 随后收到暂停指令。第一次 `SIGINT` 未能结束 NuPlan 子进程，之后使用
`SIGTERM` 结束整个评测进程组。暂停时 seed42 已运行到约 `15/186`，但没有生成
完整 `runner_report.parquet`，因此该部分不会被计入结果；B10 的完整报告仍保留。
当前所有评测进程已停止，后续恢复时脚本会跳过 B10，并重新完成 seed42、seed2026
以及 val14。

#### 按要求恢复 test14-random，并在其完成后暂停

收到“继续，当 `test14-random` 测试完就暂停”的指令后，未使用会自动进入 val14
的总脚本，而是新建并启动：

```text
HDP-nuplan/scripts/resume_v4_test14_random_only.sh
```

该脚本只执行 seed42 和 seed2026 的 test14-random；B Epoch 10 已有完整报告会被
跳过，随机集汇总生成后脚本立即结束，不启动 val14。启动时间：`2026-08-24
12:24:36 CST`；当前已进入 seed42 的 186 场景评测。

#### test14-random 最终结果

`2026-08-24 16:21:28 CST`，test14-random 三模型评测与汇总全部完成，三个模型
均为 `186/186` 成功、`0` 失败；脚本已按要求结束，未启动 val14。

| 指标 | B Epoch 10 | RL v4 seed42 | seed42 - B10 | RL v4 seed2026 | seed2026 - B10 |
|---|---:|---:|---:|---:|---:|
| score | 0.877131 | 0.874905 | -0.002226 | 0.870830 | -0.006301 |
| route progress | 0.941122 | 0.940656 | -0.000467 | 0.946382 | +0.005259 |
| no ego-at-fault collisions | 0.946237 | 0.946237 | 0 | 0.930108 | -0.016129 |
| drivable-area compliance | 0.946237 | 0.946237 | 0 | 0.951613 | +0.005376 |
| TTC within bound | 0.919355 | 0.908602 | -0.010753 | 0.892473 | -0.026882 |
| speed-limit compliance | 0.984241 | 0.984503 | +0.000262 | 0.983699 | -0.000541 |
| comfort | 0.967742 | 0.978495 | +0.010753 | 0.983871 | +0.016129 |

结论：在 186 场景 test14-random 上，两个 RL seed 都没有带来综合正收益。seed42
综合分下降约 `0.00223`，碰撞不变，但 TTC 下降约 `0.01075`；seed2026 虽然路线
进度、可行驶区域和舒适性提高，但碰撞下降约 `0.01613`、TTC 下降约 `0.02688`，
导致综合分下降约 `0.00630`。这表明 fixed200 上 seed2026 的微弱正收益没有推广到
test14-random，当前 RL v4 仍存在安全指标退化和跨场景集不稳定问题。

结果文件：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_random_val14_eval/b10_vs_rl_v4_test14_random.json
```

#### RL v5 test14-random 闭环评测启动（2026-08-24）

RL v5 训练完成后，启动了 B Epoch 10 与 RL v5 seed2026 的配对闭环评测：

```text
HDP-nuplan/scripts/evaluate_v5_test14_random.sh
```

两种模型均使用同一 `test14-random` 过滤器、同一 mini DB 目录、同一 `run_mini_closed_loop.sh`、同一 HDP planner 配置和顺序执行方式。输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_test14_random_eval/
```

启动时间：`2026-08-24 18:46:54 CST`。当前 NuPlan 已确认构建 `186 scenarios`，正在先评测 B Epoch 10；完成后自动评测 RL v5，并生成配对汇总 JSON。

随后复核发现，B Epoch 10 的完整逐场景汇总已经保存在：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_random_val14_eval/b10_vs_rl_v4_test14_random.json
```

其中 B Epoch 10 为 `186/186` 成功，指标和逐场景记录完整。因此已停止刚刚重复启动的 B10 闭环进程，避免重复消耗评测时间；新的评测脚本改为复用该 B10 汇总，只运行 RL v5。RL v5 评测于 `2026-08-24 18:48:36 CST` 重新启动，仍使用同一 `test14-random` 过滤器和 186 个场景。完成后将把旧 B10 汇总与新 RL v5 汇总合并为最终比较文件。

#### RL v5 test14-random 最终结果

`2026-08-24 20:53:41 CST`，RL v5 的 186 场景评测完成，`186/186` 成功、`0` 失败。与同一 B Epoch 10 逐场景汇总比较如下：

| 指标 | B Epoch 10 | RL v5 seed2026 | RL v5 - B10 |
|---|---:|---:|---:|
| score | 0.877131 | 0.871810 | -0.005322 |
| route progress | 0.941122 | 0.930414 | -0.010708 |
| no ego-at-fault collisions | 0.946237 | 0.946237 | 0 |
| drivable-area compliance | 0.946237 | 0.951613 | +0.005376 |
| TTC within bound | 0.919355 | 0.908602 | -0.010753 |
| speed-limit compliance | 0.984241 | 0.985456 | +0.001216 |
| comfort | 0.967742 | 0.973118 | +0.005376 |

逐场景比较确认两者 token 完全相同，共 186 个。score 提升 8 个、下降 108 个、相同 70 个；route progress 提升 1 个、下降 126 个、相同 59 个。碰撞指标没有任何场景退化；TTC 有两个场景从 1 下降到 0：`850050d6ef405220`（starting_left_turn）和 `ca949cb6eec45dd8`（waiting_for_pedestrian_to_cross）。

结论：安全候选筛选和正权重消除了 v4 seed2026 的碰撞退化，并将 TTC 退化从 `-0.026882` 缩小为 `-0.010753`，说明安全更新方向部分有效；但正权重 rollout 自蒸馏仍把大量场景拉向低路线进度的安全候选，导致 route progress 下降 `0.010708`，最终 score 仍下降 `0.005322`。因此 v5 不能判定为正收益模型，下一步应优先增加“候选路线进度不得显著低于专家/基线”的硬筛选或相对进度门槛，而不是继续扩大 epoch。

最终汇总：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_test14_random_eval/b10_vs_rl_v5_test14_random.json
```

#### RL v6 进度保护实验启动（2026-08-24）

针对 v5 路线进度下降，新增候选级进度保护门：候选必须同时满足 safety gate 和 `progress_guard_reward >= 0.9` 才能进入 rollout 自蒸馏。进度保护门默认关闭，不改变旧实验；本次 v6 显式开启。

启动脚本：

```text
HDP-nuplan/scripts/run_rl_v6_safety_progress_filtered_10k_seed2026.sh
```

本次仍从 B Epoch 10 分叉，使用同一 10,000 NPZ、buffer=10,000、正权重、seed2026、2 个 epoch；相对于 v5 只新增：

```text
rl_filter_progress_guard_candidates=true
rl_min_progress_guard_reward=0.9
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_progress_filtered_10k_seed2026_from_b10/
```

启动前完整测试结果：`55 passed, 15 warnings`。训练已后台启动，训练完成后先检查有效候选比例和 RL loss，再使用同一 `test14-random` 评测。

#### RL v5 正式训练启动（2026-08-24）

已从 B Epoch 10 checkpoint 启动单 seed2026 的 10,000 NPZ RL v5 训练。启动脚本：

```text
HDP-nuplan/scripts/run_rl_v5_safety_filtered_10k_seed2026.sh
```

训练输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_from_b10/
```

本次配置：`train_epochs=2`、`rl_buffer_size=10000`、`rl_weighting_mode=softmax_positive`、`rl_filter_safety_eligible_candidates=true`、`rl_center_reward_weights=false`、`rl_expert_anchor_weight=0.1`，其余关键 v4 参数保持不变。启动前已校验 cache 中存在 10,000 个 NPZ、manifest 包含 10,000 条记录，且起点 checkpoint 为 B Epoch 10 并包含 `model` 和 `ema_state_dict`。

启动时间：`2026-08-24 18:06:07 CST`。当前进程正常运行，GPU 利用率约 `80%`，显存约 `487 MiB`。训练完成后再进行固定 `test14-random` 闭环评测，不将训练中的中间状态作为最终结果。

#### RL v4 负收益原因分析与安全更新实现（2026-08-24）

本节针对 `test14-random` 的负收益进行代码和日志复核。186 个场景的结果如下：

| 模型 | score | route progress | collision | TTC | comfort |
|---|---:|---:|---:|---:|---:|
| B Epoch 10 | 0.877131 | 0.941122 | 0.946237 | 0.919355 | 0.967742 |
| RL v4 seed42 | 0.874905 | 0.940656 | 0.946237 | 0.908602 | 0.978495 |
| RL v4 seed2026 | 0.870830 | 0.946382 | 0.930108 | 0.892473 | 0.983871 |

seed42 的综合分下降 `0.002226`，seed2026 下降 `0.006301`。seed2026 虽然路线进度和舒适性提高，但碰撞指标下降 `0.016129`、TTC 指标下降 `0.026882`。因此问题不是“RL 完全没有改变策略”，而是 rollout 更新改变了策略，同时安全退化超过了进度和舒适性收益。

代码与训练日志显示有三个直接原因：

1. `rl_buffer_size=1024`。训练 rollout 约为 10,000 个场景，但 replay buffer 只保留最后 1,024 个样本，存在明显的末尾数据覆盖偏置。
2. v4 的 safety gate 只约有 `55.6%` 的场景组存在至少一个 TTC 合格候选；其余场景没有安全候选，但旧实现仍会用全部候选进行相对奖励更新。
3. v4 开启了 `rl_center_reward_weights=true`。原实现使用 `regression_weights = weights - 1`，低奖励候选会得到负权重。日志已直接记录：seed42 的 `reward_weighted_diffusion_loss=-5.97e-6`、`reward_weighted_waypoint_loss=-1.48e-3`、`rl_loss=-2.08e-5`；seed2026 的 `rl_loss=-1.88e-5`。负的 MSE 权重不是普通的“惩罚项”，而是在最小化目标中主动鼓励增大该候选的回归误差，容易造成碰撞/TTC 退化和 seed 敏感性。

#### 已实现的改进

在以下文件中加入了保守安全更新：

```text
HDP-nuplan/hdp_nuplan/rl/replay_buffer.py
HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py
HDP-nuplan/hdp_nuplan/rl/loss.py
HDP-nuplan/train_predictor_rl.py
HDP-nuplan/tests/test_rl_components.py
```

具体行为：

- rollout 时保存每个候选是否通过 safety gate 的 `candidate_mask`；
- 组内均值和标准差只在合格候选上计算；
- 没有安全候选，或只有一个安全候选的场景组，rollout loss 权重置零；该组仍可通过 expert anchor 保持原监督能力；
- 新增 `softmax_positive` 权重模式，保证 rollout 回归权重始终为正，不再使用 `w-1` 形成负损失；
- 新增 `eligible_candidate_fraction`、`eligible_candidates_per_group`、`no_eligible_group_fraction` 和 reward std 分位数，便于判断 RL 是否有足够有效的学习信号；
- 新增命令行开关 `--rl_filter_safety_eligible_candidates true`，默认关闭，因此不会改变旧实验的复现结果。

#### 100 场景 smoke 验证

启动脚本：

```text
HDP-nuplan/scripts/run_rl_safety_filtered_smoke100.sh
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_smoke_100_v2/rl-safety-filtered-positive-smoke/
```

本次使用 100 个场景、2 个 epoch、`softmax_positive`、安全候选筛选和 expert anchor=0.1。结果 checkpoint 已生成：

```text
HDP-nuplan/tmp/mini_train_smoke_100_v2/rl-safety-filtered-positive-smoke/training_log/hdp-rl-safety-filtered-positive-smoke100/2026-08-24-17:55:56/model_epoch_2_trainloss_0.6671.pth
```

关键日志：rollout 更新的 `rl_loss=0.004024`、`reward_weighted_diffusion_loss=0.002343`，均为正；安全合格候选比例为 `0.0375`，`no_eligible_group_fraction=0.9625`。这说明新逻辑确实屏蔽了绝大多数没有可靠安全候选的组，没有把它们错误地变成负向回归目标。由于 100 场景 smoke 的安全候选很少，不能据此宣称综合指标已经提升，只能说明实现和梯度方向符合预期。

验证命令及结果：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q HDP-nuplan/tests
```

```text
54 passed, 15 warnings in 5.00s
```

#### 下一步正式实验建议

正式 RL v5 应从 B Epoch 10 分叉，只改变为“完整覆盖 + 正权重 + 安全候选筛选”：

```text
rl_buffer_size=10000
rl_weighting_mode=softmax_positive
rl_filter_safety_eligible_candidates=true
rl_center_reward_weights=false
rl_expert_anchor_weight=0.1
train_epochs=2（先做短实验）
```

其余 v4 参数先保持不变。先在固定的 `test14-random` 上与 B Epoch 10 做配对评测，重点观察 score、collision、TTC 以及 `no_eligible_group_fraction`；若安全指标不退化，再扩大到 `val14`、`test14-hard`。

#### RL v5 训练完成与空间清理（2026-08-24）

在 smoke 通过后，已启动并完成上述 10,000 NPZ、seed2026、2 epoch 的正式 RL v5 训练。训练日志：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_from_b10/rl_train.log
```

训练结果 checkpoint：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_from_b10/training_log/hdp-rl-safety_filtered_positive_10k_seed2026-from-full-mini-b10/2026-08-24-18:06:12/model_epoch_2_trainloss_0.0011.pth
```

关键训练指标：`rl_loss=0.000755`、`reward_weighted_diffusion_loss=0.000458`、`reward_weighted_waypoint_loss=0.029751`，均为正；`buffer_size=10000`，`eligible_candidate_fraction=0.5271`，`no_eligible_group_fraction=0.47`。这只是训练阶段结果，是否带来闭环正收益仍需与 B Epoch 10 在相同 `test14-random` 场景上评测后确认。

随后清理了明确可再生的缓存：

- `NewDisk/wheels/torch241_py38_cu12`，约 2.7 GB 的离线 wheel 安装包缓存；
- `NewDisk/conda_pkgs/cache`、`NewDisk/.cache/pip`；
- 项目中的 `.mypy_cache`、`.pytest_cache`、`.ipynb_checkpoints` 和 `__pycache__`。

未删除 `HDP-nuplan/tmp` 中的 NPZ、DB、checkpoint、训练日志和评测结果。清理前可用空间约 9.3 GB，清理后约 12 GB；当前 10,000 个 RL 训练 NPZ 和 v5 checkpoint 均已复核存在。

#### HDP-nuplan/tmp 空间盘点（2026-08-24）

`HDP-nuplan/tmp` 当前约占 `126 GB`，主要不是缓存，而是以下三类文件：

| 目录 | 大小 | 内容 | 建议 |
|---|---:|---|---|
| `mini_train_full_306801_seed3407_v1` | 约 50 GB | 306,801 个完整 mini-train NPZ，约 46 GB；监督训练 checkpoint 约 1.4 GB | 若还要继续完整 mini-train 训练，保留；否则可只删除 `cache`，保留 B Epoch 10 checkpoint |
| `mini_train_balanced_10000_seed3407_v1` | 约 47 GB | 当前 10k NPZ 约 1.5 GB，历史 RL 闭环评测 `exp` 约 45 GB | 保留 `cache`、RL checkpoint、汇总 JSON；旧评测的 `exp` 可删除，但会失去逐场景 parquet |
| `mini_train_pilot_1000_seed3407_v1` | 约 6.7 GB | 早期 pilot NPZ、RL checkpoint 和闭环实验 | 当前正式流程不依赖，通常可整体删除 |
| `nuplan_train_formal_union_35869_seed3407_v1` | 约 8.5 GB | 35,869 NPZ 和两组监督训练结果 | 若不再复现实验 B/35,869 监督训练，可删除；否则保留 |
| `nuplan_train_formal_union_existing_v2` | 约 3.4 GB | 22,369 NPZ | 与当前 10k RL 训练无直接依赖，可在确认不再复现该数据集后删除 |
| `closed_loop_eval` | 约 4.9 GB | 历史闭环逐场景报告；汇总 JSON 很小 | 若只保留汇总指标，可删除其中 `exp`，保留 JSON |

当前 RL v5 只直接依赖：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache
HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_filtered_positive_10k_seed2026_from_b10/
```

因此，若目标是尽量保留当前 RL 项目且快速释放空间，第一优先级是删除历史闭环目录中的 `exp`，第二优先级是删除 `mini_train_pilot_1000_seed3407_v1`。完整 mini-train cache、35,869 NPZ 和 22,369 NPZ 暂不删除，等待用户确认是否放弃后续复现实验。

#### 用户确认后的实际清理（2026-08-24）

收到确认后，实际删除了以下内容：

```text
HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/
HDP-nuplan/tmp/nuplan_train_formal_union_35869_seed3407_v1/
HDP-nuplan/tmp/nuplan_train_formal_union_existing_v2/
HDP-nuplan/tmp/closed_loop_eval/exp/
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_fixed200_eval_retry/exp/
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_random_val14_eval/exp/
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_aligned_reward_softmax_fixed200_eval/exp/
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_gate_drivable_noise02_fixed200_eval/exp/
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_buffer10k_noise02_fixed200_eval/exp/
```

删除的是旧数据集、早期 pilot 实验和逐场景闭环仿真中间文件；保留了当前 10,000 NPZ、RL v5 checkpoint、B Epoch 10 checkpoint、历史汇总 JSON、训练日志和项目文档。`HDP-nuplan/tmp` 从约 `126 GB` 降至约 `74 GB`，`NewDisk` 可用空间从约 `12 GB` 增加至约 `61 GB`。已复核当前 10,000 NPZ 数量仍为 `10,000`。

#### RL v6 训练完成（2026-08-24）

`2026-08-24 21:20:02 CST`，RL v6 训练正常完成。Epoch 2 的关键指标：

```text
rl_loss=0.00067096
reward_weighted_diffusion_loss=0.00039750
reward_weighted_waypoint_loss=0.02734637
expert_anchor_loss=0.00342975
eligible_candidate_fraction=0.48028125
no_eligible_group_fraction=0.514
```

v6 的进度保护门实际生效：`progress_guard_eligible=0.503153125`，与 safety mask 取交集后，最终有效候选比例约 48.0%。Checkpoint：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_progress_filtered_10k_seed2026_from_b10/training_log/hdp-rl-safety_progress_filtered_10k_seed2026-from-full-mini-b10/2026-08-24-21:11:55/model_epoch_2_trainloss_0.0010.pth
```

#### RL v6 并行闭环评测启动（2026-08-24）

为缩短 186 场景 `test14-random` 闭环评测时间，采用 NuPlan devkit 已提供的 `SingleMachineParallelExecutor`，设置 2 个 thread worker。只改变场景调度方式，不改变 checkpoint、planner、scenario filter、NuPlan DB、metric 配置和场景内容；B Epoch 10 继续复用已有汇总 JSON，不重复运行。

启动脚本：

```text
HDP-nuplan/scripts/evaluate_v6_test14_random_parallel2.sh
```

输出目录：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safety_progress_filtered_10k_seed2026_test14_random_eval_parallel2/
```

启动时间：`2026-08-24 21:26:11 CST`。NuPlan 已构建 `186 scenarios` 并进入 `SingleMachineParallelExecutor`；启动后已确认至少 2 个场景并行产生结果，显存约 `417 MiB`，没有 OOM 或 traceback。预计总耗时约 1～1.5 小时。

#### RL v6 并行评测中断与恢复（2026-08-24）

后续检查发现并行进程已经退出，但没有生成完整 `runner_report.parquet` 或最终 JSON。输出目录中已完成 `152/186` 个场景的 metric 临时文件，未发现 OOM 或 Python traceback；因此不能把 152 个部分结果当成最终评测结果。

通过原 `test14-random` 汇总中的场景 token 与现有临时文件名比对，确定缺失场景数为 `34`。新建 `HDP-nuplan/hdp_nuplan/config/scenario_filter/test14-random-v6-missing34.yaml`，仅包含这 34 个 token；原有 152 个结果保留不动。

恢复脚本为 `HDP-nuplan/scripts/resume_v6_test14_random_missing34.sh`。它使用原实验 UID、原输出目录、同一个 v6 checkpoint 和 NuPlan 配置串行补跑缺失场景，随后检查临时文件总数必须为 `186`，再聚合全部 186 个场景的指标。最终结果写入 `b10_vs_rl_v6_test14_random.json`。恢复过程的详细说明另见 `HDP-nuplan/doc_hdp_nuplan/test14_random_v6评测恢复说明.md`。

恢复任务已于 `2026-08-24 23:43:42 CST` 启动。NuPlan 已确认本次只构建 `34 scenarios`，截至 `23:45` 已完成第 `1/34` 个补跑场景，metric 临时文件总数为 `153`（原有 152 个 + 新增 1 个）。
