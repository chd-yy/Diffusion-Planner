，# AutoDL 全量训练部署操作日志

## 1. 目标与原则

本次部署用于把本地已验证的 HDP-nuPlan 监督训练和 RL 管线迁移到 AutoDL RTX 4090，随后按分片方式处理 NuPlan 官方训练数据。为避免在环境、数据和模型同时变化时难以定位问题，验收顺序固定为：

1. 对齐 Python、PyTorch、NuPlan devkit 和 HDP-nuPlan；
2. 运行仓库单元测试；
3. 上传已验证的 10,000 场景缓存；
4. 用 128 场景完成监督训练 smoke；
5. smoke 通过后才进入官方数据分片预处理和大规模训练。

## 2. 实例与存储

部署日期：2026-08-09。

- GPU：NVIDIA GeForce RTX 4090 24 GB；
- 实际分配：20 CPU 核、90 GB 内存；
- 系统盘：30 GB，只保留系统级小文件；
- 数据盘：`/root/autodl-tmp`，当前总容量 170 GB；
- 项目、Conda 环境、数据、日志和 checkpoint 均放在数据盘。

目录布局：

```text
/root/autodl-tmp/
├── conda_envs/diffusion_planner
├── workspace/Diffusion-Planner
├── workspace/nuplan-devkit
├── processed
├── experiments
└── logs
```

注意：最初在另一容器状态中观察到过 250 GB，最终用于训练的实例实际显示为 170 GB，因此后续不能长期同时保存完整 NuPlan 原始数据和全部处理结果，必须采用分片处理与及时清理原始 DB 的方式。

## 3. 代码与 NuPlan devkit

HDP 工程由 Git 提交 `720025446c0cb86bb236e4903454864db0b71153` 导出：

```text
Diffusion-Planner-7200254.tar.gz
SHA256 fa86256b4e9d1996a85a90956455bb5beb38644a822570751f6434474b17926c
```

云端解压目录：

```text
/root/autodl-tmp/workspace/Diffusion-Planner
```

NuPlan devkit 固定为提交：

```text
e9241677997dd86bfc0bcd44817ab04fe631405b
```

归档文件：

```text
nuplan-devkit-e924167.tar.gz
SHA256 c7b3f15b249a34b2ad060321f270e381aaa52b8e26a7a0eb0d2a4dd6501e4bc5
```

云端以 editable 方式安装到：

```text
/root/autodl-tmp/workspace/nuplan-devkit
```

## 4. Conda 与关键依赖

环境创建在数据盘：

```bash
conda create \
  -p /root/autodl-tmp/conda_envs/diffusion_planner \
  python=3.9 pip -y
```

固定基础工具：

```text
Python       3.9.25
pip          24.0
setuptools   59.5.0
wheel        0.45.1
```

PyTorch 使用与本地一致的 CUDA 11.8 wheel：

```text
torch        2.0.0+cu118
torchvision  0.15.1+cu118
numpy        1.23.4
```

实例驱动报告 CUDA 13.0，表示驱动可支持的最高 CUDA 版本；新驱动可以运行 PyTorch 自带的 CUDA 11.8 runtime，不需要改装 CUDA 13.0 版 PyTorch。

GPU 验证结果：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

NuPlan 和 HDP 安装结果：

```text
nuplan-devkit  1.2.2
hdp-nuplan     1.0.0
```

`python -m pip check` 返回：

```text
No broken requirements found.
```

NuPlan 的 `requirements_torch.txt` 没有执行，因为它会安装旧版 PyTorch 1.9；实际执行的是 NuPlan 的通用 `requirements.txt` 和 HDP-nuPlan 的 `requirements_torch.txt`。

## 5. 仓库测试

云端执行：

```bash
cd /root/autodl-tmp/workspace/Diffusion-Planner
PYTHONPATH=HDP-nuplan \
python -m pytest -q HDP-nuplan/tests
```

结果：

```text
38 passed, 18 warnings in 5.12s
```

18 条 warning 来自 timm、Matplotlib 和扩散工具 docstring 的弃用或转义提示，没有测试失败，不阻塞训练。

## 6. 10,000 场景缓存上传

只打包本地缓存中的 10,000 个 NPZ，不包含旧训练日志和 checkpoint：

```text
/tmp/hdp-nuplan-balanced10k-cache.tar
SHA256 0fef328c494b5a5fc8e19ecc5de481f0857ae1cb8362c8688f8c700a13c34cd3
```

云端解压后目录：

```text
/root/autodl-tmp/processed/mini_train_balanced_10000_seed3407_v1/cache
```

验收结果：

```text
NPZ 数量：10000
目录大小：约 1.5 GB
```

训练 manifest 使用工程归档中已提交的：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/
diffusion_planner_training.json
```

该 manifest 是有序、无重复的 10,000 项 JSON 列表。

## 7. Encoder warm-start checkpoint

原 Diffusion-Planner checkpoint 从本地上传到：

```text
/root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth
```

校验结果：

```text
大小：47 MB
SHA256 7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45
```

smoke 启动时的加载审计为：

```text
loaded=151/151
parameters=1799040
decoder_loaded=0
```

这说明只迁移了全部兼容 encoder 张量，没有把原 Diffusion-Planner decoder 错误加载进 HDP decoder。

## 8. SSH 远程执行

为避免在聊天中传输密码，使用一把只服务于该 AutoDL 实例的 Ed25519 专用密钥。私钥只保留在本地 `~/.ssh`，云端仅保存公钥。

首次连接使用的旧公网端口返回 `Connection refused`。域名可以解析，但端口没有监听，说明失败发生在 SSH 身份验证之前，与公钥无关。重新从 AutoDL 当前实例页面复制 SSH 指令后，公网端口已变化，免密连接成功。

安全约束：

- 文档不记录 SSH 密码、私钥或公钥正文；
- SSH 端口可能随实例状态变化，始终以控制台当前指令为准；
- 长任务不能依赖交互式 SSH 存活，应使用守护会话、后台任务或外部监控。

## 9. 云端监督训练 smoke

从 10,000 项 manifest 中按既有顺序取前 128 项，生成：

```text
/root/autodl-tmp/processed/
mini_train_balanced_10000_seed3407_v1/smoke_128.json
```

关键参数：

```text
数据量                     128
batch size                  8
每 epoch batch 数          16
train epochs                2
warm-up epochs              1
learning rate               5e-4
planning hybrid loss        0.1
freeze encoder epochs       1
EMA                         true
DDP world size              1
```

执行入口：

```text
HDP-nuplan/train_predictor.py
```

训练输出目录：

```text
/root/autodl-tmp/experiments/hdp_cloud_smoke_128/training_log/
hdp-cloud-supervised-smoke-128/2026-08-09-16:08:59/
```

训练结果：

| epoch | encoder 状态 | epoch train loss | checkpoint |
|---:|---|---:|---|
| 1 | 冻结 | 31.6348 | `model_epoch_1_trainloss_31.6348.pth` |
| 2 | 解冻 | 9.7875 | `model_epoch_2_trainloss_9.7875.pth` |

进程退出码为 0，最终没有残留训练进程或 GPU 计算进程。两个 checkpoint 均可由 `torch.load` 读取，包含：

```text
model
ema_state_dict
optimizer
schedule
epoch
loss
wandb_id
```

同时生成 `latest.pth`、`args.json` 和 `encoder_warm_start_report.json`。这证明云端已完整打通：

```text
NPZ/manifest
  -> DataLoader
  -> encoder warm-start
  -> encoder 冻结与解冻
  -> HDP 前向
  -> loss 与反向传播
  -> optimizer 与 EMA
  -> checkpoint 保存和读取
```

该 128 场景结果只用于工程验收，不能用于报告模型性能或判断 RL 收益。

## 10. 当前结论与下一步

截至本次记录，云端训练环境和监督训练链路均已通过，不需要重复安装环境或继续修改模型代码。下一阶段应上传 NuPlan maps，随后只取官方 train 的第一批日志做约 50-log 数据分片预处理门禁。门禁需要验证：

1. 日志确实属于官方 train split；
2. 每个 shard 的 manifest、NPZ 数量和校验报告一致；
3. 单场景失败不会中断整个 shard；
4. 处理过程可断点续跑；
5. 估算每个日志的场景产量、耗时和磁盘占用；
6. 通过后再制定 100k 监督训练数据规模，而不是一次性下载完整原始数据。

上面的“下一阶段”已于同日继续执行，实际操作和结果记录在第 11 节以后。

## 11. 上传 NuPlan maps 并做完整性校验

### 11.1 决策依据

NuPlan 的地图要被每个 Scenario 的特征提取重复读取，因此地图应常驻 AutoDL 数据盘；原始 DB 则按 shard 临时下载。这样既避免重复传输地图，也不要求云端同时存放全部原始 train 数据。

本地地图目录：

```text
/home/yanjun/NewDisk/nuplan/dataset/maps
```

云端地图目录：

```text
/root/autodl-tmp/nuplan/maps
```

### 11.2 实际操作

先在本地对所有地图文件生成 SHA256 清单：

```bash
cd /home/yanjun/NewDisk/nuplan/dataset/maps
find . -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /tmp/nuplan-maps-v1.0.sha256
```

随后通过 SSH 的 tar 流直接传输目录。没有先在本地或云端生成额外地图压缩包，避免同时占用两份 1.4 GB 空间。传输后在云端逐项执行 SHA256 校验。

结果：

```text
地图总大小：1.4G
文件校验：7/7 passed
云端目录：/root/autodl-tmp/nuplan/maps
```

该步骤只上传 NuPlan map 文件，没有上传 camera/LiDAR sensor blobs。

## 12. 确认官方数据源与按日志下载方案

### 12.1 数据源核查

先尝试了 nuPlan-devkit 旧测试代码中出现的 `nuplan-production` bucket，结果为 HTTP 403。该 bucket 不能作为当前公开数据下载入口。

随后依据 NuPlan 官方下载页、官方 dataset setup 文档和 AWS Open Data 页面确认当前公开资源为：

```text
s3://motional-nuplan
region: ap-northeast-1
```

NuPlan v1.1 的 trainval 数据按城市放在 ZIP 中，而不是按单个 DB 暴露为独立 S3 对象。公开前缀为：

```text
https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/
public/nuplan-v1.1/
```

实测服务器支持 HTTP Range，返回状态码 206。因此可以只读取 ZIP 的中央目录和目标成员的数据区，不必先下载完整城市 ZIP。这个选择非常重要：例如 `vegas_1.zip` 约 154.4 GB，而首个 50-log shard 实际只需要其中约 7.77 GiB 的 DB。

### 12.2 新增按日志下载工具

新增：

```text
HDP-nuplan/requirements_data_download.txt
HDP-nuplan/scripts/download_nuplan_log_subset.py
HDP-nuplan/tests/test_nuplan_subset_download.py
```

新增依赖：

```text
remotezip==0.12.3
```

云端安装日志：

```text
/root/autodl-tmp/logs/remotezip_install.log
```

下载器的职责为：

1. 读取指定 NuPlan 日志清单；
2. 扫描 9 个 train ZIP 的中央目录，建立 `log_name -> archive/member` 索引；
3. 缓存索引，后续 shard 不重复扫描；
4. 只通过 HTTP Range 下载目标 DB；
5. 临时文件下载完成后原子改名；
6. 校验 ZIP CRC、SQLite 结构，可选执行 `PRAGMA quick_check`；
7. 已有有效 DB 自动跳过；
8. 输出文件大小、SHA256、来源 archive 和耗时报告。

云端索引位置：

```text
/root/autodl-tmp/nuplan/archive_index_train_v1.1.json
```

索引结果：

```text
train ZIP 数量：9
可索引 DB 数量：13,180
索引前几项与 HDP-nuplan/nuplan_train.json 一致
```

### 12.3 下载器首次缺陷及修复

首次实现用 `Path.stem` 规范化输入日志名。NuPlan 日志名自身包含多个点，例如：

```text
2021.05.12.19.36.12_veh-35_00005_00204
```

`Path.stem` 会把裸日志名误认为带扩展名的路径并截断末尾，导致索引查找失败。此时尚未下载任何 DB，没有产生错误原始数据。

修复方式：只在字符串明确以 `.db` 结尾时去掉该后缀，不再对裸日志名调用 `Path.stem`。同时增加带点日志名的回归测试。

## 13. 生成正式 100k 计划并下载 shard 00000

### 13.1 计划生成

正式计划位置：

```text
/root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json
```

配置：

```text
官方 train 日志：13,180
logs_per_shard：50
shard 数：264
目标 Scenario：100,000
seed：3407
```

目标数由计划生成器在所有日志之间确定性分配，264 个 shard 的目标合计严格为 100,000，不是每个 shard 单独向上取整。

首个 shard：

```text
shard_id：shard_00000
日志数：50
目标 Scenario：400
首日志：2021.05.12.19.36.12_veh-35_00005_00204
末日志：2021.05.13.17.53.42_veh-35_05077_05485
```

### 13.2 实际下载

下载目标：

```text
/root/autodl-tmp/nuplan/raw/shard_00000/trainval
```

执行逻辑等价于：

```bash
PY=/root/autodl-tmp/conda_envs/diffusion_planner/bin/python
PROJECT=/root/autodl-tmp/workspace/Diffusion-Planner

$PY "$PROJECT/HDP-nuplan/scripts/download_nuplan_log_subset.py" \
  --log_names_json \
    /root/autodl-tmp/nuplan/plans/nuplan_train_100k/shard_00000_logs.json \
  --output_dir /root/autodl-tmp/nuplan/raw/shard_00000/trainval \
  --index_path /root/autodl-tmp/nuplan/archive_index_train_v1.1.json \
  --report_path /root/autodl-tmp/nuplan/raw/shard_00000/download_report.json \
  --sqlite_quick_check
```

下载日志：

```text
/root/autodl-tmp/logs/download_shard_00000.log
```

实测结果：

```text
requested：50
downloaded：50
skipped：0
reprocessed：0
总字节：8,341,491,712 bytes，约 7.77 GiB
耗时：2,307.856 秒，约 38.5 分钟
SQLite quick_check：全部通过
计划清单、下载报告、落盘 DB：三者日志集合完全一致
临时文件残留：0
```

本次 50 个 DB 均来自公开数据的 `vegas_1.zip`。这是由日志本身所在 archive 决定的，不是代码固定指定城市。

## 14. 首次预处理失败与空日志诊断

### 14.1 `/usr/bin/time` 不存在

第一次尝试用 `/usr/bin/time -v` 测量峰值内存时，基础镜像没有该程序，命令退出码为 127，Python 预处理入口没有启动，也没有生成任何 NPZ。

处理方式：安装 Ubuntu 的 `time` 软件包。安装日志：

```text
/root/autodl-tmp/logs/apt_install_time.log
```

该软件包仅约 80 KB，不构成磁盘压力。

### 14.2 两个 DB 无法产生 NuPlan Scenario

真正启动 Builder 后，在进入 NPZ 处理前发现两个日志的 Scenario 数量为 0：

```text
2021.05.12.23.36.44_veh-35_00712_00774
2021.05.13.17.53.42_veh-35_03352_03415
```

数据库检查结果：

```text
第一个 DB：scene=4，lidar_pc=1240，scenario_tag=1850，时间跨度约 61.95 秒
第二个 DB：scene=4，lidar_pc=1260，scenario_tag=2615，时间跨度约 62.95 秒
```

因此这两个 DB 不是空文件，也不是下载损坏。NuPlan 官方 `get_scenarios_from_db` 对 scene 使用的有效行条件为：

```text
row_num >= 3 AND row_num < n.cnt - 1
```

只有 4 个 scene 时不存在满足条件的有效 scene 行，所以 ScenarioBuilder 按官方逻辑返回 0 个 Scenario。

第一次真实预处理因此在抽样阶段退出：

```text
耗时：8.70 秒
峰值 RSS：783,528 KB
生成 NPZ：0
```

## 15. 空日志兼容改造

### 15.1 为什么不能静默忽略

如果默认静默跳过，用户可能把错误路径、缺失 DB 或 split 配错误判为正常空日志。因此保留严格默认行为：未显式授权时，只要请求日志没有 Scenario 就报错。

正式全量流程新增显式开关：

```text
--allow_empty_logs
```

只有传入该参数时，抽样器才会：

1. 保留 shard 原始请求的全部日志清单；
2. 单独记录空日志及数量；
3. 在其余有效日志间重新分配该 shard 的完整目标数；
4. 保证目标仍为 400，而不是因两个空日志减少样本；
5. 让 merged report 同时保存 `log_count=请求日志数` 与
   `selected_log_count=实际贡献 NPZ 的日志数`。

修改文件：

```text
HDP-nuplan/data_process.py
HDP-nuplan/hdp_nuplan/data_process/sampling.py
HDP-nuplan/scripts/run_preprocessing_shard.py
HDP-nuplan/scripts/merge_preprocessing_shards.py
HDP-nuplan/tests/test_balanced_sampling.py
HDP-nuplan/tests/test_preprocessing_pipeline.py
```

测试新增了三类约束：严格模式仍拒绝空日志；显式允许时总目标数保持不变；merge 后空日志仍保留在 split 审计信息中，但不能出现在 `selected_per_log`。

代码同步云端时曾遇到一次临时 DNS 解析失败；第一次 `scp` 没有上传任何文件，重试后 6 个修改文件全部上传成功。该故障没有导致本地和云端版本混用。

## 16. shard 00000 正式预处理结果

### 16.1 执行命令

```bash
cd /root/autodl-tmp/workspace/Diffusion-Planner

/usr/bin/time -v env PYTHONUNBUFFERED=1 \
  /root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/run_preprocessing_shard.py \
  --plan /root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --shard_index 0 \
  --data_path /root/autodl-tmp/nuplan/raw/shard_00000/trainval \
  --map_path /root/autodl-tmp/nuplan/maps \
  --output_root /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --checksum_mode files \
  --allow_empty_logs
```

运行日志：

```text
/root/autodl-tmp/logs/preprocess_shard_00000.log
```

### 16.2 实测结果

```text
请求日志数：50
有效日志数：48
空日志数：2
目标 Scenario：400
processed：400
skipped：0
failed：0
manifest：400
NPZ：400
临时文件残留：0
状态：complete
```

性能：

```text
Scenario 特征处理进度耗时：约 3 分 03 秒
完整命令 wall time：3 分 11.78 秒
平均进度：约 2.17 scenario/s
CPU 利用率：242%
峰值 RSS：976,036 KB，约 953 MiB
NPZ 总大小：62,995,484 bytes，约 60.1 MiB
```

关键产物：

```text
/root/autodl-tmp/processed/nuplan_train_100k_gate/shard_00000/
  cache/
  manifest.json
  sampling_report.json
  processing_report.json
  checksums.json
  launch.json
```

### 16.3 断点续跑复验

完全相同的命令再次执行，结果为：

```text
processed：0
skipped：400
failed：0
manifest：400
wall time：7.76 秒
峰值 RSS：783,060 KB
```

复验日志：

```text
/root/autodl-tmp/logs/preprocess_shard_00000_resume.log
```

这证明已有可读 NPZ 会被快速跳过，任务中断后不需要删除整个 shard 重来。

## 17. 缓存校验、manifest 合并与复验

先校验单 shard：

```bash
$PY HDP-nuplan/scripts/validate_processed_cache.py \
  --cache_dir "$OUT/shard_00000/cache" \
  --manifest "$OUT/shard_00000/manifest.json" \
  --sampling_report "$OUT/shard_00000/sampling_report.json" \
  --expected_count 400 \
  --expected_log_count 48 \
  --output "$OUT/shard_00000/cache_validation_report.json"
```

再合并并从训练集共同根目录复验：

> 本节记录的是早期单 shard 门禁命令；当时合并器允许合并子集。当前正式命令行入口已经要求 `--plan` 并拒绝不足 264 个 shard 的子集，不能直接把本节命令用于冻结 100k 正式 manifest。

```bash
$PY HDP-nuplan/scripts/merge_preprocessing_shards.py \
  --shards_root "$OUT" \
  --output_manifest "$OUT/train_manifest.json" \
  --output_report "$OUT/train_merged_report.json"

$PY HDP-nuplan/scripts/validate_processed_cache.py \
  --cache_dir "$OUT" \
  --manifest "$OUT/train_manifest.json" \
  --sampling_report "$OUT/train_merged_report.json" \
  --expected_count 400 \
  --expected_log_count 48 \
  --output "$OUT/train_cache_validation_report.json"
```

其中：

```text
PY=/root/autodl-tmp/conda_envs/diffusion_planner/bin/python
OUT=/root/autodl-tmp/processed/nuplan_train_100k_gate
```

两次验证均为 `status=passed`：

```text
manifest_count：400
unique_manifest_count：400
npz_count：400
实际贡献 NPZ 的日志数：48
每日志最少/最多抽样：8 / 11
shape 错误：0
非有限值错误：0
cache/manifest 缺失或多余文件：0
```

合并报告同时记录：

```text
log_count：50
selected_log_count：48
empty_log_count：2
manifest_count：400
```

注意：校验器的 `--expected_log_count` 指“实际出现在 NPZ 中的日志数”，所以这里是 48；split 审计仍使用合并报告中的完整 50 个请求日志。

## 18. 本地与云端完整自动化测试

分别执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests

PYTHONPATH=HDP-nuplan \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果：

```text
本地：44 passed, 15 warnings in 5.42s
云端：44 passed, 15 warnings in 4.14s
```

15 个 warning 仍来自现有依赖的弃用提示，不是测试失败。

## 19. 官方 shard 监督训练门禁

### 19.1 目的

缓存验证只能证明 NPZ 的字段和 shape 正确，不能证明训练入口真的能按嵌套 manifest 加载并完成反向传播。因此对这 400 个官方场景执行 1 epoch 监督训练。该实验只做工程门禁，不代表模型最终性能。

### 19.2 执行命令

```bash
cd /root/autodl-tmp/workspace/Diffusion-Planner

/usr/bin/time -v env PYTHONUNBUFFERED=1 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 1 \
  --standalone \
  HDP-nuplan/train_predictor.py \
  --name hdp-official-shard-00000-gate \
  --save_dir /root/autodl-tmp/experiments/hdp_official_shard_gate \
  --train_set /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --train_set_list \
    /root/autodl-tmp/processed/nuplan_train_100k_gate/train_manifest.json \
  --normalization_file_path \
    /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/normalization.json \
  --train_epochs 1 \
  --save_utd 1 \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 0.0005 \
  --warm_up_epoch 1 \
  --planning_hybrid_loss 0.1 \
  --diffusion_model_type x_start \
  --diffusion_supervision_type x_start \
  --encoder_pretrained_model_path \
    /root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 0 \
  --use_wandb false
```

运行日志：

```text
/root/autodl-tmp/logs/train_official_shard_00000_gate.log
```

### 19.3 结果

```text
Dataset：400
batch size：8
batch 数：50
encoder warm-start：151/151 tensors
decoder 从旧 checkpoint 加载：0 tensors
encoder trainable：true
epoch train loss：12.1730
wall time：12.28 秒
峰值 RSS：1,293,000 KB
退出码：0
```

输出目录：

```text
/root/autodl-tmp/experiments/hdp_official_shard_gate/training_log/
hdp-official-shard-00000-gate/2026-08-09-17:17:21/
```

`latest.pth` 和 `model_epoch_1_trainloss_12.1730.pth` 均可成功加载，包含：

```text
model
ema_state_dict
optimizer
schedule
epoch
loss
wandb_id
```

因此首个官方分片已经打通：

```text
公开 train ZIP
  -> 按日志 Range 下载 DB
  -> SQLite 完整性校验
  -> NuPlan ScenarioBuilder
  -> 均衡抽样
  -> 400 NPZ
  -> 单 shard 校验
  -> 合并 manifest
  -> 根目录复验
  -> DataLoader
  -> 监督训练和 checkpoint
```

## 20. 容量实测与下一步判断

当前 AutoDL 数据盘：

```text
总容量：170G
已使用：19G
剩余：152G
```

本次门禁占用：

```text
50 个原始 DB：7.8G
400 个 NPZ 和报告：62M
maps：1.4G
```

基于首 shard 做线性外推时，100,000 个 NPZ 约为 15.7 GB；这是估算值，后续 shard 的场景分布可能使单文件大小变化。真正的空间瓶颈是原始 DB：如果保留全部 13,180 个 DB，远超当前 170 GB 数据盘。因此正式流程必须按 shard 下载、预处理、校验，然后删除已经产出且可恢复下载的原始 DB，只长期保留 maps、NPZ、manifest、报告和 checkpoint。

当前已经通过单 shard 门禁，但还不应立刻无人值守启动全部 264 个 shard。下一步应先连续处理额外 3 个 shard，并验证：

1. 不同 ZIP/城市的 Range 下载和地图读取；
2. 空日志比例是否稳定，100k 总目标能否持续守恒；
3. 4 shard 合并、train/val split 审计；
4. 每完成一个 shard 后删除原始 DB，确认磁盘峰值可控；
5. 实测总下载速度后再决定串行执行还是并行 2 个下载任务。

这 4-shard 门禁通过后，才进入 100k 全量预处理和正式监督训练。

## 21. 跨城市 4-shard 门禁启动

### 21.1 shard 选择依据

只读分析 `preprocessing_plan.json` 与 archive index 后发现，`shard_00001` 到 `shard_00034` 仍全部属于 `vegas_1`。如果机械地处理相邻三个 shard，只能继续验证同一 ZIP 和同一地图，不能尽早暴露跨城市地图或 archive 的问题。

因此保留已完成的 Las Vegas `shard_00000`，另外选择三个各含 50 个纯城市日志的分片：

```text
shard_00152：Pittsburgh 50 logs
shard_00175：Singapore 50 logs
shard_00197：Boston 50 logs
```

这四个 shard 合计覆盖 Las Vegas、Pittsburgh、Singapore、Boston 四个 NuPlan 城市。计划中的目标分别为 400、400、350、350，合计严格为 1,500 个 Scenario。后两个 shard 为 350，是因为 100,000 个目标按日志确定性分配余数，并非每个 50-log shard 都固定抽 400 个。

### 21.2 并行下载命令

为减少 4090 实例等待数据的租赁时间，三个不同 archive 的下载任务并行执行；每个任务仍在自身目录原子写入，并独立执行 SQLite quick check，不会写同一个 DB。

```bash
PROJECT=/root/autodl-tmp/workspace/Diffusion-Planner
PY=/root/autodl-tmp/conda_envs/diffusion_planner/bin/python
PLAN=/root/autodl-tmp/nuplan/plans/nuplan_train_100k
INDEX=/root/autodl-tmp/nuplan/archive_index_train_v1.1.json

for SHARD in 00152 00175 00197; do
  RAW=/root/autodl-tmp/nuplan/raw/shard_${SHARD}
  mkdir -p "$RAW/trainval"
  nohup env PYTHONUNBUFFERED=1 "$PY" \
    "$PROJECT/HDP-nuplan/scripts/download_nuplan_log_subset.py" \
    --log_names_json "$PLAN/shard_${SHARD}_logs.json" \
    --output_dir "$RAW/trainval" \
    --archive_index "$INDEX" \
    --report "$RAW/download_report.json" \
    --sqlite_quick_check \
    > "/root/autodl-tmp/logs/download_shard_${SHARD}.log" 2>&1 &
done
```

启动记录：

```text
shard_00152 PID 11571
shard_00175 PID 11573
shard_00197 PID 11575
```

`nohup` 使任务不依赖本地 SSH 连接；断线后下载仍会继续。最终完成时间、字节数、完整性和后续预处理结果将在本节继续补充。

### 21.3 下载实测结果

三个下载任务使用不同目录和报告并行执行。实测如下：

| shard | 城市/archive | 请求 DB | 本轮下载 | 本轮跳过 | 落盘字节 | 本轮报告耗时 |
|---|---|---:|---:|---:|---:|---:|
| 00152 | Pittsburgh | 50 | 50 | 0 | 2,058,637,312 | 815.10 秒 |
| 00175 | Singapore | 50 | 50 | 0 | 1,335,910,400 | 777.62 秒 |
| 00197 | Boston 续跑 | 50 | 33 | 17 | 2,197,708,800 | 852.36 秒 |

Pittsburgh 和 Singapore 完成后立即进入预处理，没有等待 Boston。这样可以让 CPU 特征提取与剩余网络下载重叠，减少实例空闲时间。

本地到 AutoDL SSH 入口在此期间多次出现：

```text
Connection refused
Temporary failure in name resolution
Connection timed out
```

每次都先检查原 PID，而不是重新执行命令。三个下载任务和后续预处理均由云端 `nohup` 托管，所以本地 SSH 轮询失败没有终止云端作业，也没有产生重复进程。

### 21.4 Boston 连接停滞及处置

Boston 首次进程 PID 为 `11575`。执行到第 18 个 DB：

```text
2021.09.14.14.57.08_veh-39_00645_00957.db
```

临时文件曾增长到 67,108,864 bytes，随后连续数分钟满足以下条件：

```text
临时文件大小不变
/proc/11575/io 的 write_bytes 不变
进程仍为 sleep/wait 状态
下载日志没有进入下一个 member
```

这证明不是“文件很大、计数尚未刷新”，而是 HTTP Range 读取连接实际停滞。原下载器没有显式 socket read timeout，因此进程可能永久等待。

处置顺序：

1. 只向确定的 PID `11575` 发送 `SIGTERM`；
2. 等待并确认该 PID 已退出；
3. 保留 17 个完整 DB；
4. 只删除唯一未完成临时文件：

```text
/root/autodl-tmp/nuplan/raw/shard_00197/trainval/
.2021.09.14.14.57.08_veh-39_00645_00957.db.11575.tmp
```

删除的临时文件为 67,108,864 bytes，不是完整 DB，无法作为有效数据恢复；对应 DB 可从官方 ZIP 重新下载。

### 21.5 下载器可靠性修复

修改：

```text
HDP-nuplan/scripts/download_nuplan_log_subset.py
HDP-nuplan/tests/test_nuplan_subset_download.py
```

新增默认参数：

```text
connect_timeout_seconds = 10
read_timeout_seconds = 60
max_member_retries = 3
retry_delay_seconds = 5
```

含义：

- 连接建立超过 10 秒失败；
- 单个 socket 连续 60 秒没有读到数据则判定该 Range 读取失败；
- 当前 ZIP member 在第一次失败后最多再试 3 次；
- 每次重试关闭失效 RemoteZip/HTTP 连接，等待 5 秒，重新打开 archive；
- `_copy_member_atomic()` 的 `finally` 删除本次未完成临时文件；
- 报告新增总 `retry_count` 和每个文件的 `download_attempts`。

新增测试用 `TimeoutError` 模拟第一次 member 下载失败，验证第二次会重新打开 archive、成功落盘并记录两次 attempt。目标测试本地和云端均为：

```text
4 passed
```

Boston 使用新脚本从 17 个已完成 DB 断点续跑，PID 为 `13852`。最终报告：

```text
requested_log_count：50
downloaded：33
skipped_existing：17
reprocessed_invalid：0
retry_count：0
total_bytes：2,197,708,800
elapsed_seconds：852.36
临时文件残留：0
```

`retry_count=0` 的原因是重新启动后新连接一直有数据，不需要触发自动重试；但 17 个已有 DB 的跳过行为和新超时版本均在真实任务中生效。首次停滞进程运行约 21 分钟，续跑约 14.2 分钟，因此 Boston 整体实际投入时间约 35 分钟；不能只把续跑报告的 852 秒当成完整首次下载时间。

### 21.6 三个新增城市的预处理与单 shard 校验

执行命令与 shard 00000 相同，只替换 `--shard_index` 和 `--data_path`，均显式使用：

```text
--checksum_mode files
--allow_empty_logs
```

结果：

| shard | 城市 | 请求日志 | 有效日志 | 空日志 | NPZ | 失败 | wall time | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00152 | Pittsburgh | 50 | 49 | 1 | 400 | 0 | 2:00.16 | 957,032 KB |
| 00175 | Singapore | 50 | 50 | 0 | 350 | 0 | 1:11.59 | 823,116 KB |
| 00197 | Boston | 50 | 43 | 7 | 350 | 0 | 2:00.28 | 860,536 KB |

三次 `validate_processed_cache.py` 均为 `status=passed`：

```text
Pittsburgh：400 NPZ，49 logs，每日志 8～10 个场景，62,998,000 bytes
Singapore：350 NPZ，50 logs，每日志固定 7 个场景，55,108,826 bytes
Boston：350 NPZ，43 logs，每日志 8～9 个场景，55,109,234 bytes
```

Boston 有 7/50 个日志无法生成官方 Scenario；如果只验证相邻 Las Vegas shard，不会提前看到这种差异。显式空日志报告和重分配确保 Boston 的目标仍严格为 350。

### 21.7 四 shard 合并与强校验

> 这是正式计划完整性门禁加入之前执行的四城市子集合并实验。当前 CLI 会正确拒绝将 4/264 个 shard 标记为正式完整数据；历史结果仅用于证明跨城市 NPZ 可以共同读取。

执行：

```bash
$PY HDP-nuplan/scripts/merge_preprocessing_shards.py \
  --shards_root /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --output_manifest \
    /root/autodl-tmp/processed/nuplan_train_100k_gate/train_manifest.json \
  --output_report \
    /root/autodl-tmp/processed/nuplan_train_100k_gate/train_merged_report.json
```

合并后再用 `validate_processed_cache.py` 从共同根目录读取全部嵌套 manifest。结果：

```text
shard_count：4
请求日志 log_count：200
有效日志 selected_log_count：190
空日志 empty_log_count：10
manifest_count：1,500
unique manifest：1,500
NPZ：1,500
失败：0
每日志场景范围：7～11
NPZ 总字节：236,211,544，约 225.3 MiB
状态：passed
```

10 个空日志由合并报告完整保留；它们不出现在 `selected_per_log`，但仍参与后续 train/val 日志泄漏审计。

### 21.8 四城市监督训练门禁

训练入口仍为：

```text
HDP-nuplan/train_predictor.py
```

关键命令参数：

```text
name：hdp-official-four-city-gate
train_set：/root/autodl-tmp/processed/nuplan_train_100k_gate
train_set_list：train_manifest.json
数据量：1,500
batch_size：8
完整 batch：187
train_epochs：1
encoder_pretrained_model_path：checkpoints/model.pth
freeze_encoder_epochs：0
model/supervision type：x_start/x_start
planning_hybrid_loss：0.1
```

结果：

```text
encoder warm-start：151/151 tensors
decoder_loaded：0
187/187 batch 完成
epoch train loss：9.4630
wall time：22.28 秒
峰值 RSS：1,379,412 KB
退出码：0
```

输出目录：

```text
/root/autodl-tmp/experiments/hdp_official_four_city_gate/training_log/
hdp-official-four-city-gate/2026-08-09-18:02:21/
```

`latest.pth` 与 `model_epoch_1_trainloss_9.4630.pth` 均可读取，loss 为 `9.463010787963867`，并包含模型、EMA、优化器、调度器、epoch 和日志 ID。

### 21.9 最终自动化测试

加入下载超时重试测试后，完整测试结果：

```text
本地：45 passed, 15 warnings in 4.95s
云端：45 passed, 15 warnings in 5.21s
```

## 22. 4-shard 门禁结论

四城市门禁已经完整通过，能够证明当前代码具备以下工程能力：

```text
官方公开 ZIP 按 member 下载
  -> SQLite 与 SHA256 校验
  -> 网络停滞超时和 member 级重试
  -> 下载断点续跑
  -> 四城市地图特征提取
  -> 合法空日志审计和目标数重分配
  -> NPZ 原子写入与缓存强校验
  -> 多 shard 合并
  -> 训练 DataLoader、反向传播和 checkpoint
```

此时可以进入 100k 预处理，但必须采用“下载一个 shard、处理并校验、保存下载报告、删除该 shard 原始 DB”的滚动策略。当前数据盘实测：

```text
总容量：170G
已用：25G
剩余：146G
四个 raw shard：约 13.2G
四 shard NPZ 与报告：230M
```

不能让 264 个 shard 的原始 DB 同时常驻数据盘。下一步先归档四份 `download_report.json` 到对应 processed shard，然后删除这四个已经验证且可从官方源恢复的 `trainval` 临时目录，以真实验证空间回收流程。

## 23. 下载报告归档与 raw DB 空间回收

先把每个下载报告复制到对应的 processed shard：

```text
/root/autodl-tmp/processed/nuplan_train_100k_gate/
  shard_00000/download_report.json
  shard_00152/download_report.json
  shard_00175/download_report.json
  shard_00197/download_report.json
```

复制后分别比较 raw 与 processed 两份报告的 SHA256，结果：

```text
shard_00000  3b00e18044dc85f51df28d80609e8bdd866f55c403820137f626558c959ddb26
shard_00152  9a90d81ca838b45cf575b2f3531a7678243b4d53a3a7a381a00be212ff8a80bd
shard_00175  31fa1950957b0729e9e84cef3a2dbcec7920facd7404cfa5593a63c1b16695be
shard_00197  0ae193d355506d10df0f1f03ec8e422ebfa707f5c1bf0cd8cc357f07b69c432a
```

四组源/目标哈希全部一致后，只删除以下四个明确目录：

```text
/root/autodl-tmp/nuplan/raw/shard_00000/trainval
/root/autodl-tmp/nuplan/raw/shard_00152/trainval
/root/autodl-tmp/nuplan/raw/shard_00175/trainval
/root/autodl-tmp/nuplan/raw/shard_00197/trainval
```

没有删除 shard 父目录和其中的原始 `download_report.json`；processed 目录中的 NPZ、manifest、sampling/processing/checksum/validation/download report、云端日志和训练 checkpoint 也均保留。

删除后的检查：

```text
四个 trainval 目录均不存在
raw 报告均存在
processed 报告均存在
raw 根目录总占用：112K
processed 4-shard 数据：230M
数据盘已用：25G -> 12G
数据盘可用：146G -> 159G
```

本次删除约 13.2 GB 可重新从官方公开 ZIP 下载的临时 DB。恢复时使用对应 `download_report.json` 中的日志名、archive、成员、大小和 SHA256 即可；处理后的 1,500 个 NPZ 不受影响。

空间回收流程已实测通过，后续 100k 可以滚动执行而不需要 300 GB 数据盘同时容纳全部原始 DB。

## 24. 操作日志云端同步核验

将本文件、`NuPlan全量数据云端预处理改造操作日志.md` 和 `requirements_data_download.txt` 同步到云端工程时，`scp` 末尾报告过一次 SSH `Connection timed out`。没有直接假设同步失败或盲目覆盖，而是等待连接恢复后分别计算本地与云端 SHA256。三份文件的本地/云端哈希完全一致，说明数据实际已传完，超时发生在连接收尾阶段。

云端文档目录：

```text
/root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan
```

因此完整操作日志同时保存在本地 Git 工作区和当前 AutoDL 数据盘工程副本中。

## 25. 100k 无人值守滚动执行器

### 25.1 改造原因

单 shard 下载和预处理已经验证，但连续运行 264 个 shard 还需要解决恢复位置、并发任务互斥、已完成 shard 跳过、raw 删除条件、磁盘下限和逐 shard 审计。因此新增：

```text
HDP-nuplan/scripts/run_preprocessing_range.py
HDP-nuplan/tests/test_preprocessing_range.py
```

### 25.2 单 shard 状态机

每个 worker 串行执行：

```text
读取计划和恢复位置
  -> 实时强校验已有 processed shard
  -> 已完整：跳过下载和预处理
  -> 未完整：检查剩余磁盘 >= 20 GiB
  -> 下载 DB 并执行 SQLite quick_check
  -> 预处理并审计官方空日志
  -> validate_processed_cache 强校验
  -> 原子归档 download_report 并核对 SHA256
  -> 只删除 raw_root/<shard_id>/trainval
  -> 原子更新 state 的 next_position
```

任一步骤抛异常时，worker 写入 `status=failed`、异常和完整 traceback，`next_position` 不前移，当前 raw DB 不删除。用同一命令重启后从失败位置继续，下载器和预处理器分别跳过已有完整 DB/NPZ。

已有产物首次强校验失败时会尝试下载和预处理续跑；续跑后第二次强校验仍失败则停止，不会归档报告或清理 raw。

### 25.3 三 worker 互斥分配

分配公式：

```text
shard_index % worker_count == worker_index
```

当前 `worker_count=3`，264 个 shard 恰好分成三个互斥集合：

```text
worker 0：0, 3, 6, ... 261，共 88 个
worker 1：1, 4, 7, ... 262，共 88 个
worker 2：2, 5, 8, ... 263，共 88 个
```

三者并集严格为 `0..263`。四个已完成门禁 shard 会被各自 worker 强校验并跳过：

```text
shard_00000 -> worker 0
shard_00152 -> worker 2
shard_00175 -> worker 1
shard_00197 -> worker 2
```

因此计划覆盖 264 个 shard，实际新增处理为 260 个。

### 25.4 删除与磁盘保护

raw 清理函数只允许：

```text
<raw_root>/<精确 shard_id>/trainval
```

它拒绝父目录逃逸、错误名称和非目录目标。只有 `processing_report.status=complete`、失败数为 0、NPZ 强校验通过、下载报告已归档且源/目标 SHA256 一致后才删除。

每个新 shard 下载前通过 `shutil.disk_usage()` 检查数据盘，默认至少保留 20 GiB；低于阈值时停止并保存状态，不开始下一次下载。

### 25.5 测试与一次同步故障

新增测试覆盖三 worker 分工完整且互斥、左闭右开范围选择、报告先归档再精确清理，以及 raw 已删除时的幂等行为。

结果：

```text
执行器与下载器定向测试：本地/云端均 8 passed
完整测试：本地 49 passed, 15 warnings in 5.31s
完整测试：云端 49 passed, 15 warnings in 4.19s
```

首次同步时 AutoDL 域名临时 DNS 失败，测试文件已上传但执行器文件未上传，云端收集阶段出现：

```text
ModuleNotFoundError: No module named 'run_preprocessing_range'
```

这是同步不完整而非逻辑测试失败。连接恢复后补传两个文件，定向和完整测试全部通过。

### 25.6 真实跳过 smoke

使用 `start_index=0, end_index=1, worker_count=1` 对已完成的 `shard_00000` 实跑。结果：

```text
status：complete
next_position：1
action：skipped_valid_complete
manifest/NPZ：400/400
有效日志：48
重新下载/预处理：否/否
raw_trainval_removed：false，因为此前已经删除
```

状态文件：

```text
/root/autodl-tmp/logs/rolling_smoke_shard_00000_state.json
```

### 25.7 正式后台命令

三个 worker 使用相同参数，只替换 `<W>`：

```bash
nohup /usr/bin/time -v env PYTHONUNBUFFERED=1 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
/root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /root/autodl-tmp/nuplan/archive_index_train_v1.1.json \
  --raw_root /root/autodl-tmp/nuplan/raw \
  --map_path /root/autodl-tmp/nuplan/maps \
  --output_root /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --state_path /root/autodl-tmp/logs/rolling_worker_<W>_of_3_state.json \
  --start_index 0 \
  --end_index 264 \
  --worker_index <W> \
  --worker_count 3 \
  --checksum_mode files \
  --connect_timeout_seconds 10 \
  --read_timeout_seconds 60 \
  --max_member_retries 3 \
  --retry_delay_seconds 5 \
  --min_free_gib 20 \
  > /root/autodl-tmp/logs/rolling_worker_<W>_of_3.log 2>&1 &
```

启动结果：

```text
worker 0 PID 17454
worker 1 PID 17455
worker 2 PID 17456
```

状态与日志分别为：

```text
/root/autodl-tmp/logs/rolling_worker_0_of_3_state.json
/root/autodl-tmp/logs/rolling_worker_1_of_3_state.json
/root/autodl-tmp/logs/rolling_worker_2_of_3_state.json
/root/autodl-tmp/logs/rolling_worker_0_of_3.log
/root/autodl-tmp/logs/rolling_worker_1_of_3.log
/root/autodl-tmp/logs/rolling_worker_2_of_3.log
```

### 25.8 启动后首次状态

约 23 秒后：

```text
worker 0：next_position=1，shard_00000 已跳过，正在下载 shard_00003
worker 1：next_position=0，正在下载 shard_00001
worker 2：next_position=0，正在下载 shard_00002
三个 worker status：running
新 DB 已完整落盘：9
数据盘已用/可用：13G / 158G
```

当前 100k 是后台进行中的预处理任务，不能表述为已经完成。三个 worker 全部 `status=complete` 后，还要合并 264 个 shard、验证精确 100,000 个 NPZ，之后才能开始正式监督训练。

### 25.9 第二次健康检查

文档同步后第一次读取状态时 SSH 入口再次 `Connection timed out`。没有重启后台任务；10 秒后连接恢复，检查结果为：

```text
worker 0：running，shard_00003，15/50 DB
worker 1：running，shard_00001，7/50 DB
worker 2：running，shard_00002，10/50 DB
三个 state 均无 error
已完成 processing report：4 个门禁 shard
现有 NPZ：1,500
worker 运行时间：约 3 分 06 秒
数据盘已用/可用：15G / 156G
```

三组下载仍在继续，当前没有自动重试、缓存错误或磁盘阈值告警。

## 26. 100k 后台任务实时监控记录

监控项目包括 worker 父进程、state 状态、当前 shard、完整 DB 数、当前临时文件大小、重试/错误关键字、已完成 shard/NPZ 和数据盘余量。

### 26.1 第一帧

```text
worker 0：running，shard_00003，16/50 DB，当前临时文件 16 MiB
worker 1：running，shard_00001，7/50 DB，当前临时文件约 344 MiB
worker 2：running，shard_00002，12/50 DB，当前临时文件 8 MiB
父 PID 17454/17455/17456 均存活，运行约 5 分 09 秒
state error：全部为 null
已完成 shard：4
NPZ：1,500
磁盘可用：156G
```

### 26.2 约 30 秒后的第二帧

```text
worker 0：16 -> 18 个完整 DB，临时文件约 72 MiB
worker 1：仍在同一个大 DB，临时文件约 344 -> 392 MiB
worker 2：12 -> 14 个完整 DB，临时文件约 112 MiB
三个 PID 均存活
```

完整 DB 数不变但临时文件增长，表示正在传输较大的 ZIP member，不属于停滞。

日志关键字检查：

```text
worker 0：retries=0，error_markers=0
worker 1：retries=0，error_markers=0
worker 2：retries=0，error_markers=0
```

三个实际子进程分别为 `download_nuplan_log_subset.py`，参数中的 10/60 秒超时和 3 次重试均已生效。

### 26.3 再约 30 秒后的第三帧

```text
worker 0：shard_00003，18/50 DB，临时文件约 136 MiB
worker 1：shard_00001，7/50 DB，临时文件约 488 MiB
worker 2：shard_00002，15/50 DB，已进入新 member，临时文件约 24 MiB
三个 worker：running
state error：全部为 null
数据盘已用/可用：16G / 155G
```

三帧数据证明三个下载任务都在持续产生 I/O，没有连接停滞、自动重试、异常退出或磁盘压力，因此本次监控不需要重启或修改任务。

## 27. 100,000 个 NPZ 占完整 train 数据的比例估算

### 27.1 统计口径

不能用 `100,000 / 13,180` 计算，因为 `13,180` 是官方 train 日志（DB）数量，而不是 Scenario 数量。当前数据流程中，一个日志可以产生数千个候选 Scenario；每个最终 NPZ 对应一个经过筛选并选中的 Scenario。

本节计算的是“100,000 个训练 Scenario 占完整官方 train 候选 Scenario 的比例”，不是原始 DB 磁盘容量占比，也不是日志覆盖率。当前采样计划会遍历官方 train 日志清单，因此日志来源覆盖范围与 Scenario 抽样比例是两个不同概念。

### 27.2 已完成分片的观测值

只读汇总云端 4 个门禁分片的 `sampling_report.json`：

```text
请求日志数：200
有可用 Scenario 的日志数：190
空日志数：10
候选 Scenario 总数：680,408
按全部请求日志计算的平均值：680,408 / 200 = 3,402.04 Scenario/日志
```

四个分片分别为：

```text
shard_00000：201,600
shard_00152：211,198
shard_00175：136,407
shard_00197：131,203
合计：680,408
```

### 27.3 外推结果

按官方 train 清单中的 13,180 个日志外推：

```text
预计完整候选 Scenario 数
= 680,408 / 200 * 13,180
= 44,838,887.2

100,000 个 NPZ 的估计覆盖率
= 100,000 / 44,838,887.2 * 100%
= 0.2230%
```

因此当前可以表述为：**100,000 个 NPZ 约占完整 nuPlan 官方 train 候选 Scenario 的 0.22%，约为 1/448。**

这是基于 200 个日志样本的工程估算，不是最终精确值。不同城市和日志时长存在差异；待全部 264 个 shard 完成后，应汇总所有 `sampling_report.json` 的 `unique_scenarios`，再用 `100,000 / 全量候选数` 得到精确比例。

## 28. 预处理耗时分析与 3 路扩展到 6 路

### 28.1 为什么预处理慢

实时资源检查结果：

```text
容器 CPU 配额：20 核（cpu.max = 2000000 / 100000）
容器内存上限：90 GiB
数据盘可用：146 GiB
GPU 利用率：0%
```

`nproc` 和 `free -h` 会显示宿主机的 208 核和约 1 TiB 内存，但容器 cgroup 的实际配额分别是 20 核和 90 GiB，评估并发时必须使用 cgroup 数值。

当前链路为：

```text
远程 ZIP Range 下载 50 个 DB
→ SQLite/大小/哈希校验
→ 从 DB 构造候选 Scenario
→ 提取地图、路线、邻车和轨迹特征
→ 写 NPZ、manifest、report 和 checksum
→ 强校验
→ 删除当前 shard 的 raw DB
```

主要瓶颈是远程下载，不是 GPU。实测一个包含 50 个日志的 shard：

```text
shard_00002：下载 4,816,138,240 bytes，耗时 748.89 秒，约 6.13 MiB/s
已归档的 4 个门禁 shard：单 shard 下载约 777～2,308 秒
350～400 个 Scenario 的特征提取：约 68～207 秒
```

旧的 3 路运行状态经常表现为两个 worker 下载、一个 worker 预处理；单个预处理进程约使用 2.2 个 CPU 核和 0.9 GiB RSS，GPU 为 0%。因此 3 路并未用满 20 核 CPU 和 90 GiB 内存，可以提高滚动 worker 数。

原“剩余约 2～3 天”是按较慢门禁 shard 给出的保守估计。结合正式任务的实时速度，3 路更合理的粗估为约 1～2 天；6 路若官方 S3 并发带宽能扩展，预计约 15～24 小时。网络速度和不同城市 DB 大小变化较大，因此不能把该时间作为保证值。

### 28.2 安全切换方法

不能在旧 3 路继续运行时直接另开一组覆盖相同范围的 6 路，否则可能同时写同一个 shard。采用以下切换过程：

1. 向旧的 3 个调度器及其 `/usr/bin/time` 外层进程发送 `SIGSTOP`，阻止其领取下一 shard。
2. 不终止已经开始的下载或 `data_process.py` 子进程，让 `shard_00001`、`shard_00002`、`shard_00003` 自然完成。
3. 确认三个 shard 的下载/处理报告已经落盘，且没有旧的下载、分片执行或数据处理子进程。
4. 向暂停的旧调度器发送 `SIGTERM`，再发送 `SIGCONT` 使终止信号生效。
5. 用新的 `_of_6` state 和 log 文件启动 6 路，已有 DB、NPZ 和报告由续跑逻辑校验并复用。

切换时的实际结果：

```text
shard_00001：50 个 DB 下载完成，尚未预处理
shard_00002：400 个 NPZ 和 processing report 完成
shard_00003：400 个 NPZ 和 processing report 完成
旧下载/预处理子进程：全部自然结束
旧 3 路调度进程：全部终止
```

第一次远程启动 6 路时，AutoDL SSH 域名出现临时 DNS 解析失败：

```text
ssh: Could not resolve hostname connect.cqa1.seetacloud.com:
Temporary failure in name resolution
```

该命令没有在云端执行，因此没有产生半启动 worker。连接恢复后完整重试一次。

### 28.3 6 路启动结果

6 个 `/usr/bin/time` 外层 PID：

```text
worker 0：19351
worker 1：19352
worker 2：19353
worker 3：19354
worker 4：19355
worker 5：19356
```

每个 worker 使用相同的计划范围 `[0, 264)`，通过 `shard_index % 6 == worker_index` 互斥分配，每路 44 个 shard。新的审计文件为：

```text
/root/autodl-tmp/logs/rolling_worker_<0..5>_of_6.log
/root/autodl-tmp/logs/rolling_worker_<0..5>_of_6_state.json
```

启动后首轮检查：

```text
worker 0：校验并跳过 shard_00000，开始 shard_00006
worker 1：复用 shard_00001 已下载的 DB，继续校验/预处理
worker 2：校验并接管 shard_00002，开始 shard_00008
worker 3：校验并接管 shard_00003，开始 shard_00009
worker 4：开始 shard_00004
worker 5：开始 shard_00005
```

因此本次并发切换没有删除或丢失已下载 DB、NPZ 和报告，也没有两个活动进程同时写同一 shard。

## 29. 为什么正式训练要等 100,000 个 NPZ

算法并不要求必须达到 100,000 个样本才能执行训练；此前的 100/1,500 场景 smoke test 已经验证训练程序可以运行。当前把精确 100,000 个 NPZ 作为正式监督训练门槛，是为了先固定训练数据集，再开始可复现的正式实验。

正式启动顺序为：

```text
264 个 shard 全部完成
→ 合并 shard manifest
→ 校验 NPZ 总数精确为 100,000
→ 校验文件存在性、重复项、分片报告和 checksum
→ 冻结正式训练 manifest
→ 启动监督训练
```

如果在 NPZ 仍持续增加时就开始训练，不同 epoch 可能看到不同数据，断点恢复后的数据分布也可能改变，难以准确复现实验。因此当前不会在后台预处理尚未完成时启动正式监督训练。

## 30. 6 路切换后的连续监控

### 30.1 第一帧（18:58:02 CST）

```text
worker 0：shard_00006，2 个完整 DB，临时文件 136 MiB
worker 1：shard_00001，复用 50 个完整 DB，正在预处理
worker 2：shard_00008，临时文件 56 MiB
worker 3：shard_00009，临时文件 72 MiB
worker 4：shard_00004，临时文件 216 MiB
worker 5：shard_00005，临时文件 56 MiB
完整 shard：6
当前 NPZ 文件：2,518
数据盘可用：151.0 GiB
六路 state：全部 running，error=None
日志：retry=0，error marker=0
```

进程核对确认 `shard_00001` 只有一个 `data_process.py` 写进程，其余五路为下载进程；不存在同一 shard 的重复写入。

### 30.2 第二帧（18:59:07 CST）

```text
NPZ：2,518 -> 2,641，增加 123
worker 0 临时文件：136 -> 152 MiB
worker 2 临时文件：56 -> 72 MiB
worker 3 临时文件：72 -> 96 MiB
worker 4：已完成 2 个 DB，当前临时文件 184 MiB
worker 5 临时文件：56 -> 96 MiB
完整 shard：仍为 6
数据盘可用：150.7 GiB
retry=0，error=0
```

下载临时文件和 NPZ 数量均持续增长，没有停滞。

### 30.3 第三帧（19:00:07 CST）

```text
shard_00001：处理完成并通过完整报告校验
完整 shard：6 -> 7
NPZ：2,641 -> 2,700
worker 1：next_position 0 -> 1，自动进入 shard_00007
数据盘可用：150.7 -> 157.2 GiB
其余 worker：继续 shard_00004/00005/00006/00008/00009
```

磁盘余量回升约 6.5 GiB，证明 `shard_00001` 的 raw DB 已在报告归档和强校验后自动回收。worker 1 随后自动领取 `shard_00007`，证明以下续跑闭环正常：

```text
复用已有下载
→ 预处理
→ 报告与 cache 强校验
→ 归档下载审计
→ 删除 raw DB
→ 更新 state
→ 领取下一 shard
```

本轮监控结束时，6 路均健康运行，无重试、无异常、无磁盘压力，不需要人工干预。

### 30.4 持续监控帧（19:01:24～19:02:25 CST）

19:01:24 的状态：

```text
6/6 Python 调度进程存活
6 个下载进程，0 个预处理进程
完整 shard：7
NPZ：2,700
磁盘可用：156.9 GiB
GPU：0%，1 MiB
全部 worker：running，error=None，retry=0
```

此时六路都处于下载阶段，因此没有 `data_process.py` 属于正常的阶段性状态。61 秒后的下载增量：

```text
shard_00006：临时文件 216 -> 232 MiB
shard_00007：24 -> 40 MiB
shard_00008：120 -> 160 MiB
shard_00009：136 -> 160 MiB
shard_00004：232 -> 256 MiB
shard_00005：136 -> 152 MiB
磁盘可用：156.9 -> 156.6 GiB
```

六个临时文件均增长，说明所有下载连接都在产生有效 I/O。完整 shard 和 NPZ 暂未增长，是因为这一帧仍处在下载同一批 DB 的阶段；日志仍为 retry=0、traceback=0，不需要干预。

### 30.5 调整监控频率

用户要求把常规状态检查从分钟级调整为每 10 分钟一次，避免过度频繁输出。后续采用以下规则：

```text
正常状态：每 10 分钟采集并汇报一次
异常状态：worker 退出、state=failed、重试累积、Traceback、磁盘低于安全阈值时立即汇报和处置
```

调整前最后一帧为 19:04:17 CST：6 路下载均继续增长，完整 shard 为 7，NPZ 为 2,700，磁盘可用 156.6 GiB，error/retry 均为 0。

## 31. 本地电脑协同生成 NPZ 的可行性

### 31.1 本地资源实测

```text
磁盘：221G 总量，126G 可用
CPU：16 个逻辑核
内存：15 GiB，总可用约 2.4 GiB
Swap：15 GiB，已使用 9.6 GiB
Python：3.9.25
PyTorch：2.0.0+cu118
GPU：RTX 4060 Laptop，CUDA 可用
nuPlan devkit：/home/yanjun/NewDisk/nuplan-devkit
nuPlan 地图：四城市地图齐全
本地 DB：108 个
remotezip：尚未安装
```

本地磁盘足够采用“下载一个 shard、处理、校验、删除 raw DB”的滚动方案。NPZ 预处理主要受网络、CPU、SQLite 和地图特征提取限制，RTX 4060 基本不参与；主要风险是本地内存和 Swap 压力，因此不适合复制云端的 6 路并发。

### 31.2 推荐协同方案

先用本地单 worker 处理一个云端短期内不会访问的尾部 shard，例如 `shard_00252`，测量下载速度、峰值内存和总耗时。验证稳定后，本地可顺序处理 `252～263`，而云端 6 路继续从低编号向上运行。

本地产物不能直接写入云端正式 shard 目录，安全同步流程应为：

```text
本地完成整个 shard
→ 本地执行 cache/manifest/report/checksum 强校验
→ 上传到云端唯一 staging 目录
→ 云端再次强校验
→ 确认正式 shard 目录尚不存在
→ 原子改名为正式 shard 目录
```

这样云端 worker 后续到达该编号时，会通过 `_validate_completed_shard()` 验证并跳过，不会重新下载和处理。如果云端已经先完成该 shard，则保留云端版本并放弃本地上传，禁止覆盖。

预计本地增加一路后的理想总吞吐提升约为 10%～17%，实际收益取决于家庭网络到 nuPlan S3 的速度。本地开始前还需要：

1. 安装与云端一致的 `remotezip`。
2. 从云端同步 `preprocessing_plan.json`、对应 log JSON 和 `archive_index_train_v1.1.json`。
3. 做一个尾部 shard 的单路基准测试。
4. 确认本地峰值内存、Swap 和下载速度可接受后，再决定是否继续尾部范围。

当前只完成可行性检查，尚未启动本地下载或处理，没有改变云端 6 路任务分配。

## 32. 19:00 后云端 CPU 接近 0% 的原因

19:14:41 CST 的 10 分钟监控帧显示，6 路恰好全部处于 `download_nuplan_log_subset.py` 下载阶段，没有 `data_process.py`。各下载进程的 `%CPU` 为：

```text
0.3%, 0.9%, 0.8%, 0.4%, 0.3%, 0.4%
合计约 3.1% 个单核
折算到容器 20 核配额：3.1% / 20 ≈ 0.155%
```

因此 AutoDL 面板显示 CPU 接近 0% 是正常的。下载进程主要等待 S3 网络和磁盘 I/O，CPU 只在 ZIP 解压、SQLite quick check 和 SHA256 阶段短时工作；GPU 为 0% 同样符合预期。

不能只凭 CPU 曲线判断任务是否停滞。与 19:04:17 的状态比较：

```text
shard_00006：完整 DB 4 -> 6
shard_00007：完整 DB 0 -> 2
shard_00008：完整 DB 1 -> 2
shard_00009：临时文件 176 -> 280 MiB
shard_00004：完整 DB 2 -> 4
shard_00005：临时文件 176 -> 304 MiB
磁盘可用：155.8 GiB
retry=0，traceback=0，disk error=0
```

六路都产生了有效下载增量，证明任务没有卡住。某个 shard 下载完成并进入 `data_process.py` 后，单个处理进程实测约占 `220%` CPU（约 2.2 核），云端 CPU 曲线会阶段性升高；下载和预处理交替运行，因此 CPU 曲线呈低谷和峰值是预期行为。

## 33. 19:17:07 实时进度

```text
完整 shard：7 / 264 = 2.65%
完整 shard 对应计划 NPZ：2,700 / 100,000 = 2.70%
文件系统 NPZ：2,700（当前没有部分预处理文件）
活动 6 路调度器：6
活动下载器：6
活动 data_process.py：0
retry：0
error：0
磁盘可用：155.5 GiB
```

当前 shard 下载状态：

```text
worker 0：shard_00006，6 个完整 DB，临时文件 64 MiB
worker 1：shard_00007，2 个完整 DB，临时文件 40 MiB
worker 2：shard_00008，2 个完整 DB，临时文件 112 MiB
worker 3：shard_00009，0 个完整 DB，临时文件 296 MiB
worker 4：shard_00004，4 个完整 DB，临时文件 40 MiB
worker 5：shard_00005，0 个完整 DB，临时文件 320 MiB
```

从 19:00 到本帧尚无新的完整 shard，但六路 DB 数或临时文件均有增长。这说明任务没有停止，但 6 路可能在竞争同一公网/S3 带宽。需要再取得一个完整 10 分钟窗口的总字节增量，比较 6 路聚合吞吐与先前 3 路实测，再决定是否保留 6 路；不能仅以 worker 数推断加速比。

## 34. 下载慢速诊断与用户要求停止云端

### 34.1 6 路与 3 路 A/B 测试

以六个活动 raw shard 的总文件字节数为口径：

```text
6 路窗口：约 53.7 秒增加 40 MiB，聚合约 0.74 MiB/s
临时暂停 3 路：约 72.2 秒增加 40 MiB，聚合约 0.55 MiB/s
```

测试结束后三个暂停进程均恢复为运行状态。6 路比 3 路只快约 34%，没有接近线性加速，说明继续增加云端 worker 的收益有限。

同一官方 ZIP 的精确 HTTP Range 测试出现很大波动：快速连接可在约 1.6～2.9 秒下载 8 MiB，慢连接会降到约 62 KiB/s 并持续 90 秒。现有 requests/RemoteZip 只配置“socket 完全无数据”的 read timeout；慢连接持续传输少量数据时不会超时，所以会长期拖住 worker。

curl 的低速检测门禁证明，可以通过 `--speed-limit` 和 `--speed-time` 在约 12～13 秒主动中止慢连接，而不是等待数分钟。为此新增了实验性 curl member 后端，并在本地和云端通过完整测试：

```text
本地：51 passed, 15 warnings
云端：51 passed, 15 warnings
```

真实 member 门禁没有切换正式任务。门禁同时发现：ZIP 中央目录的 extra field 与本地文件头可能不同，第一版压缩数据 offset 偏移 8 字节；另外 6 路占用带宽时 512 KiB/s 阈值过高。该门禁安全失败，没有生成正式 DB，也没有覆盖任何正式 shard。后续必须使用本地文件头计算精确 offset，并在低并发下载阶段重新标定阈值。

### 34.2 用户要求停止云端

2026-08-09 19:39 CST，用户明确要求先停止云端。停止顺序：

1. 先向 6 个 `run_preprocessing_range.py` 调度器发送 `SIGSTOP`，防止领取新 shard。
2. 确认当前只有下载子进程，没有 `data_process.py` 或 NPZ 写进程。
3. 终止 6 个 `download_nuplan_log_subset.py`。
4. 终止调度 Python 进程和 `/usr/bin/time` 外层进程。
5. 将 6 个 state 原子更新为 `stopped_by_user`，记录停止时间和原因。
6. 核对目标进程数为 0。

停止后的恢复点：

```text
worker 0：next_position=1，当前 shard_00006
worker 1：next_position=1，当前 shard_00007
worker 2：next_position=1，当前 shard_00008
worker 3：next_position=1，当前 shard_00009
worker 4：next_position=0，当前 shard_00004
worker 5：next_position=0，当前 shard_00005
完整 shard：7
完整 NPZ：2,700
残留未完成隐藏 .tmp：6 个
相关活动进程：0
数据盘：170G 总量，约 155G 可用
```

完整 `.db`、已完成 NPZ、manifest、report、checksum 和未完成 `.tmp` 均被保留，没有执行清理。当前云端不会继续下载、预处理或训练。恢复前应先完成下载/处理两阶段流水线设计，并明确处理 6 个旧 PID 临时文件的策略。

### 34.3 关机前最终核验

2026-08-09 19:40:43 CST，再次独立检查所有可能相关的下载、预处理和训练进程：

```text
target_processes=0
complete_shards=7
npz=2700
partial_tmp=6
free_gib=154.5
```

六个 worker 的 state 均为 `stopped_by_user`：worker 0～3 的 `next_position=1`，worker 4～5 的 `next_position=0`。随后执行 `sync`，返回 `filesystem_sync=done`，确保已完成数据和状态文件已提交到文件系统。

因此此时可以在 AutoDL 控制台执行“关机/停止实例”。不能选择“释放实例”“重置实例”或删除数据盘，否则已下载 DB、2,700 个 NPZ 和断点状态可能丢失。下次开机后先核对 state、完整 shard、残留 `.tmp` 和磁盘空间，再恢复预处理；不会自动启动正式监督训练。

## 35. 停机后的恢复代码改造

### 35.1 残留 `.tmp` 的性质与处理原则

停止时的 6 个隐藏 `.tmp` 是下载器原子写 DB 时留下的未完成文件，命名形式为：

```text
.<log_name>.db.<pid>.tmp
.<log_name>.db.<pid>.deflate.tmp
```

它们不是经过大小、SQLite、CRC 和 SHA256 校验的完整 DB，不能改名后复用。`remotezip` 后端也没有可靠的 member 断点续传语义，因此恢复时应重新下载当前未完成 DB。已经原子改名为 `<log_name>.db` 的完整文件仍按原逻辑校验并跳过，不会重复下载。

在 `download_nuplan_log_subset.py` 中新增了 `cleanup_stale_member_temporaries()`：

1. 只匹配当前目标 DB 的两类 PID 临时文件，不扫描或删除其他 DB。
2. 用 `os.kill(pid, 0)` 保守判断原进程是否存在。
3. 仅删除 PID 已不存在的临时文件；PID 存活或无权探测时一律保留。
4. 删除发生在对应 DB 确实需要重新下载之后，完整正式 DB 不受影响。
5. 下载报告增加 `stale_temporaries_removed`，便于审计恢复时实际清理数量。

这意味着下次开机恢复后，当前 6 个死亡 PID 临时文件会在各自 DB 被重新处理时逐个安全清理，不需要人工使用宽泛的 `rm` 命令。

### 35.2 专用恢复入口

新增：

```text
HDP-nuplan/scripts/resume_autodl_preprocessing_6.sh
```

该脚本固定恢复当前 `[0, 264)`、`worker_count=6` 的正式预处理任务，并执行以下门禁：

1. 核对 Python、项目脚本、100k 计划、archive index 和地图路径存在。
2. 若已有下载、预处理或训练进程，拒绝重复启动。
3. 要求数据盘至少保留 20 GiB。
4. 先执行下载器与滚动 worker 的定向测试。
5. 要求 `_of_6` 的六个断点 state 全部存在。
6. 六路均传入 `--resume`，沿用各自原 `next_position`。
7. 使用已验证过的 `remotezip` 后端；实验性 curl 门禁尚未切入正式任务。
8. 脚本只调用 `run_preprocessing_range.py`，没有监督训练或 RL 训练入口。

由于云端已经关机，本轮只在本地创建和验证恢复代码，没有尝试 SSH、没有启动云端任务。下次开机后还必须先把本地更新同步到云端项目目录，再执行恢复脚本。

### 35.3 验证结果

定向测试：

```text
11 passed in 0.34s
```

完整 HDP-nuPlan 测试集：

```text
52 passed, 15 warnings in 5.29s
```

恢复脚本检查：

```text
bash -n：通过
git diff --check：通过
```

新增测试覆盖：死亡 PID 的当前 DB 临时文件会被删除；活动 PID 的临时文件和其他 DB 的临时文件均保留。当前正式进度仍是 7 个完整 shard、2,700 个 NPZ；停机期间不会变化，也不会自动启动训练。

## 36. 统一的只读监控快照

### 36.1 为什么新增统一脚本

此前每次监控分别使用 `pgrep`、`find`、state 解析和 `df` 等临时命令，容易出现“文件系统 NPZ”“完整 shard 对应 NPZ”“部分 shard 已写 NPZ”三个口径混用。新增：

```text
HDP-nuplan/scripts/monitor_preprocessing.py
```

该脚本完全只读，不修改 state、DB、NPZ 或报告，也不会启动、停止任何进程。

### 36.2 完整 shard 的判定

一个 shard 必须同时满足以下条件才计入 `complete_shards`：

1. `processing_report.json` 可解析且 `status=complete`。
2. `failed=0`。
3. `manifest.json` 存在且是列表。
4. manifest 条目数等于计划中该 shard 的 `total_scenarios`。
5. processing report 的 `manifest_count` 同样等于计划目标。
6. `cache/*.npz` 的文件系统实数等于计划目标。

因此只写完部分 NPZ、只有报告、只有 manifest，或报告错误地声明完成，都不会被计为完整 shard。`actual_npz` 单独表示所有 shard 当前真实存在的 NPZ，包括部分 shard；`complete_shard_npz` 只包含通过上述门禁的 shard。

### 36.3 每帧输出内容

每次快照包含：

```text
phase 与 issues
完整 shard / 计划 shard
文件系统 NPZ / 100,000
完整 shard NPZ 与部分 shard NPZ
已归档 download report 的累计 retry_count
6 个 worker 的 status、next_position、current_shard、error
scheduler、downloader、preprocessor 和两类 trainer 进程数
raw DB 数、残留 .tmp 数与字节数
数据盘总量、使用量、可用量及 20 GiB 门槛
```

异常判定包括：worker 报错或状态损坏、state 显示 running 但没有 scheduler、调度器超过 6 个、磁盘低于 20 GiB、完整 shard 缺失下载审计，以及意外出现监督训练或 RL 训练进程。

### 36.4 云端使用命令

下次开机并同步本地代码后，执行一次简洁快照：

```bash
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/scripts/monitor_preprocessing.py \
  --plan /root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --output_root /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --raw_root /root/autodl-tmp/nuplan/raw \
  --state_dir /root/autodl-tmp/logs \
  --worker_count 6 \
  --min_free_gib 20
```

需要完整机器可读快照时，在末尾增加 `--json`。恢复运行后可由外部每 10 分钟执行一次；脚本自身不创建常驻轮询进程，避免关机、重连或监控器异常影响正式预处理。

### 36.5 验证

定向测试覆盖：

- 完整 shard 与部分 shard 分开计数。
- 文件系统 NPZ 和完整 shard NPZ 分开计数。
- raw DB、残留 `.tmp` 和归档重试数统计。
- 报告声称 complete 但 NPZ 缺失时拒绝计为完成。
- worker 失败和意外训练进程告警。

结果：

```text
监控脚本定向测试：2 passed in 0.12s
完整 HDP-nuPlan 测试集：54 passed, 15 warnings in 5.29s
py_compile：通过
git diff --check：通过
```

本轮云端仍处于关机状态，因此没有把停机期间不变的 `7 shard / 2,700 NPZ` 伪装成新的在线监控结果，也没有恢复下载、预处理或训练。

## 37. 正式合并门禁加固

### 37.1 审计发现的问题

原 `merge_preprocessing_shards.py` 只枚举当前能找到的 `shard_*/processing_report.json`。只要这些已发现 shard 各自标记为 complete，它就会把合并报告写成 `status=complete`；它并不知道原计划要求 264 个 shard。因此在当前只有 7 个完整 shard 时误执行旧命令，有可能得到一个仅含局部数据却名为 `train_manifest.json` 的文件。

早期单 shard和四城市门禁阶段使用这种子集合并能力是合理的，但它不能充当正式 100k 数据冻结入口。

### 37.2 当前正式 CLI 的强制条件

命令行入口现在必须传入：

```text
--plan /root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json
```

正式合并会先完成以下检查，任一失败都在写正式 manifest 之前退出：

1. 计划自身的 shard 数、连续索引、`shard_id`、日志总数和场景目标自洽。
2. 输出根目录的 shard 集合与计划精确相等；当前 7/264 会直接失败。
3. 264 个 processing report 全部存在，`status=complete` 且 `failed=0`。
4. 每个 shard 的日志清单、请求场景数与计划一致。
5. manifest、processing report、sampling report 和实际 NPZ 数量互相一致。
6. cache 中不能缺少 manifest 文件，也不能存在未列入 manifest 的额外 NPZ。
7. shard 之间不能重复日志或重复 NPZ 名称。
8. `cache_validation_report.json` 必须为 passed，其 manifest/sampling 哈希仍匹配。
9. `download_report.json` 必须完整覆盖该 shard 的计划日志，并汇总下载重试和字节数。
10. 默认读取 `checksums.json`，重新计算所有元数据及 100,000 个 NPZ 的 SHA256。
11. 最终总数必须精确等于计划的 264 shard、13,180 logs 和 100,000 NPZ。

合并报告现在还汇总所有 sampling report 的 `raw_scenarios`、`unique_scenarios`、去重数，并给出：

```text
sampled_fraction_of_unique_candidates
= 100000 / 全量 unique_scenarios
```

这会在全量完成后替代此前基于 200 个日志得到的约 0.22% 工程估算。

### 37.3 最终操作入口

新增：

```text
HDP-nuplan/scripts/finalize_autodl_preprocessing.sh
```

它只允许在下载、预处理和训练进程均为 0 时运行，顺序为：

```text
严格按原 plan 合并 264 个 shard
→ 全量 SHA256 复验
→ 生成 train_manifest.json 与 train_merged_report.json
→ validate_processed_cache 加载并检查全部 100,000 个 NPZ
→ 生成 train_cache_validation_report.json
→ 停止
```

脚本没有调用 `train_predictor.py`、`train_predictor_rl.py` 或 `torch.distributed.run`，因此不会自动启动正式训练。全部 shard 完成后仍需人工执行该脚本并审阅三个输出文件。

### 37.4 测试证据

新增测试证明正式合并会拒绝：

- 缺失计划 shard；
- 混入计划外 shard；
- cache 中存在未列入 manifest 的 NPZ；
- NPZ 内容变化但文件名不变的静默损坏；
- 计划、报告、日志、计数或哈希不一致。

验证结果：

```text
合并管线定向测试：8 passed
完整 HDP-nuPlan 测试集：57 passed, 15 warnings in 4.68s
merge_preprocessing_shards.py py_compile：通过
finalize_autodl_preprocessing.sh bash -n：通过
git diff --check：通过
```

相关通用文档的正式合并命令也已补上 `--plan`；历史单 shard/四城市子集合并记录增加了版本说明。本轮云端保持关机，最终脚本只在本地创建和检查，没有执行合并，也没有启动训练。正式进度仍为 7/264 shard、2,700/100,000 NPZ。

## 38. OpenDataLab nuPlan-v1.1 镜像核对

用户提供的目录：

```text
https://opendatalab.com/OpenDataLab/nuPlan-v1_dot_1/tree/main/raw
```

通过 OpenDataLab 公开文件树 API 读取到的实际文件不是官方九个 train ZIP，而是一个分卷 `tar.gz`：

```text
nuPlan-v1.1.tar.gz.0000 ～ nuPlan-v1.1.tar.gz.0011
```

前 11 卷各为 `107,374,182,400 bytes`（100 GiB），最后一卷为 `15,339,575,783 bytes`，总计：

```text
1,196,455,582,183 bytes
= 1,196.46 GB
= 1,114.29 GiB
```

OpenDataLab 为每个分卷提供独立 SHA256，元数据声明版本为 `nuPlan-v1.1`、publisher 为 Motional、publish URL 指向 nuPlan 官方页面。但这些信息只能说明它被标记为官方数据集镜像，不能单独证明其中每个 DB 与官方归档字节一致。

官方九个 train ZIP 的 HTTP Content-Length 合计为：

```text
1,017,278,085,874 bytes
= 1,017.28 GB
= 947.41 GiB
```

两者相差约 `179.18 GB`，且封装格式不同。这很可能是因为 OpenDataLab 分卷包含 train 之外的其他 v1.1 内容或采用了不同的重新打包方式；在读取内部文件树前不能把该推断当成已证实事实。

对当前工程而言，主要差异是访问方式：

- 官方每城市 ZIP 有中央目录，可以随机读取当前 shard 所需的 DB member。
- OpenDataLab 是一个被切成 12 段的连续 gzip/tar 流；必须取得全部分卷并按顺序拼接读取。
- 普通 `tar.gz` 不适合按 50 个日志随机提取；列出或搜索全部内容也需要顺序扫描压缩流。
- 若完整解压，除了约 1.196 TB 分卷本身，还要为解压后的数据预留大量额外空间；当前 170 GB 数据盘不能承担该操作。

因此当前优先建议仍是下载官方 Boston、Pittsburgh、Singapore、Vegas 1～6 共九个 Asia ZIP，并原样挂载，不整体解压。若 OpenDataLab 到国内文件存储的下载速度或价格明显更优，也可以把它作为候选来源，但必须：

1. 下载 `.0000`～`.0011` 全部 12 卷，不能只取部分。
2. 按 OpenDataLab 公布的 SHA256 校验每卷。
3. 在足够大的临时/文件存储中检查 tar 文件树。
4. 提取后将 train DB basename 集合与正式计划的 13,180 个日志精确比对。
5. 将每个 DB 的文件大小和 CRC32 与现有官方 ZIP archive index 比对。
6. 只有全部匹配后才允许作为正式预处理输入。

本轮只进行了公开元数据和 HTTP 头的只读核对，没有下载 1.196 TB 数据、没有改变云端状态，也没有恢复预处理或训练。

## 39. 无 GPU 实例下载官方 Boston train ZIP

### 39.1 本轮边界与存储位置

用户在无 GPU 模式下启动 AutoDL 实例，要求只下载官方页面中 Boston City 的 train log database ZIP，并观察下载速度。本轮明确不执行解压、DB 提取、NPZ 预处理、监督训练或 RL 微调。

官方归档 URL：

```text
https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip
```

HTTP `Content-Length`：

```text
38,161,149,300 bytes（约 35.54 GiB）
```

用户进一步指定下载到 AutoDL 数据盘，而不是文件存储 `/root/autodl-fs`。最终路径为：

```text
下载中的临时文件：
/root/autodl-tmp/nuplan/official_archives/.nuplan-v1.1_train_boston.zip.part

下载完成后的正式文件：
/root/autodl-tmp/nuplan/official_archives/nuplan-v1.1_train_boston.zip

下载日志：
/root/autodl-tmp/logs/nuplan-v1.1_train_boston.zip.download.log

后台包装进程 PID：
/root/autodl-tmp/logs/nuplan-v1.1_train_boston.zip.download.pid
```

目标修正时先正常停止原下载进程，将已经下载的 `44,220,416 bytes` 断点文件移动到 `/root/autodl-tmp` 后续传；没有重新下载已有内容。随后检查 `/root/autodl-fs`，确认没有遗留 Boston ZIP 或临时文件。

数据盘当时状态：

```text
总容量：170G
可用容量：155G
```

因此可以容纳 35.54 GiB 的 Boston ZIP。本轮不解压，所以不额外占用一份 DB 解压空间。

### 39.2 单连接基线测速

实例自带 `curl 7.68.0` 不支持 `--retry-all-errors`，第一次兼容性尝试立即退出，没有破坏临时文件。去掉该选项后使用断点续传，20 秒净增长结果为：

```text
begin_bytes=69,124,096
end_bytes=70,828,032
delta_bytes=1,703,936
speed=0.0813 MiB/s
speed=0.6816 Mbps
```

按这一速度下载完整归档需要约 5 天，因此不继续采用单连接 `curl`。

### 39.3 aria2 多连接续传与参数修正

在系统盘安装 `aria2 1.35.0`，新增占用约 6.2 MB。它只作为下载器使用，不处理 nuPlan 数据。随后在同一个 `.part` 文件上启用 16 分片连接续传：

```bash
aria2c \
  --continue=true \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=1M \
  --connect-timeout=15 \
  --timeout=60 \
  --max-tries=0 \
  --retry-wait=5 \
  --lowest-speed-limit=0 \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --summary-interval=10 \
  --dir=/root/autodl-tmp/nuplan/official_archives \
  --out=.nuplan-v1.1_train_boston.zip.part \
  'https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip'
```

第一次 aria2 配置曾使用：

```text
--lowest-speed-limit=32K
```

日志证明源站到该实例的单条连接经常只有 `23～32 KiB/s`，因此 aria2 将这些连接误判为过慢并主动中断，有效连接数从 10 降至 4，总速度降到约 `0.30～0.38 MiB/s`。这不是归档损坏，也不是磁盘故障。

处理方式是向 aria2 发送 `SIGINT`，让它保存 `.aria2` 断点控制文件，然后将最低速度阈值改为 0 并原地续传。已有下载内容没有删除。修正后观测到：

```text
有效连接：CN:16
当前进度：约 548 MiB / 35 GiB（1%）
瞬时速度：约 1.7～2.2 MiB/s
折算带宽：约 14～18 Mbps
aria2 ETA：约 4.5～5.5 小时（会随线路继续波动）
```

与单连接基线相比，多连接当前约快 21～27 倍。

aria2 采用分片随机写入，因此 `.part` 文件是稀疏文件：`stat` 显示的逻辑大小可能提前接近 35 GiB，不能用它代表真实完成进度。监控时应看 aria2 的 `MiB/35GiB` 和 `DL:` 字段，或者查看文件实际分配块数。

### 39.4 下载完成时的自动检查

后台包装命令只在 aria2 正常完成后执行以下操作：

1. 检查临时文件大小必须精确等于 `38,161,149,300 bytes`。
2. 使用 Python `zipfile` 读取 ZIP 中央目录，确认归档可识别且至少包含一个 member。
3. 两项检查都通过后，将 `.part` 原子改名为正式 `.zip`。

它不会解压 ZIP，也不会调用 `data_process.py`、`run_preprocessing_range.py`、`train_predictor.py` 或 `train_predictor_rl.py`。检查时也确认上述预处理和训练进程均未运行。

后续只读监控命令：

```bash
tr '\r' '\n' \
  < /root/autodl-tmp/logs/nuplan-v1.1_train_boston.zip.download.log \
  | tail -n 30

pgrep -af 'aria2c.*nuplan-v1.1_train_boston'

df -h /root/autodl-tmp
```

### 39.5 2026-08-10 安全暂停下载

用户要求中途暂停 Boston ZIP 下载。暂停前没有删除临时文件，也没有使用 `kill -9`；采用 `SIGINT` 让 aria2 正常结束并保存断点。

第一次安全选择脚本使用的匹配式同时命中了 aria2 子进程和外层下载包装进程。由于包装进程的名称不是 `aria2c`，安全检查主动拒绝发送信号，所以该次尝试没有改变下载状态。随后将选择条件收紧为：

```text
进程名必须严格等于 aria2c
且完整命令行必须包含 nuplan-v1.1_train_boston.zip
```

最终只向实际下载进程 PID `2153` 发送：

```bash
kill -INT 2153
```

暂停结果：

```text
暂停时间：2026-08-10T01:03:51+08:00
最后进度：13 GiB / 35 GiB（39%）
暂停前瞬时速度：796 KiB/s
Boston aria2 进程数：0
断点控制文件大小：4,908 bytes
```

断点文件已存在且与临时文件同时更新：

```text
/root/autodl-tmp/nuplan/official_archives/.nuplan-v1.1_train_boston.zip.part
/root/autodl-tmp/nuplan/official_archives/.nuplan-v1.1_train_boston.zip.part.aria2
```

aria2 日志明确记录：

```text
Shutdown sequence commencing...
Download ... not complete
aria2 will resume download if the transfer is restarted.
```

暂停后间隔 5 秒复查，临时文件的已分配块数和修改时间均未变化，外层包装进程与 aria2 子进程均已退出，确认下载已经静止。此时可以关闭实例；重新开机后应使用第 39.3 节相同的 aria2 参数与路径续传，不能删除 `.part` 或 `.part.aria2`。

## 40. U 盘 OpenDataLab `.0011` 分卷上传方案检查

用户准备将约 15.3 GB 的 `nuPlan-v1.1.tar.gz.0011` 从 U 盘传到 AutoDL。只读检查结果：

- 当前本机没有检测到已挂载的 U 盘，也没有在 `/media`、`/run/media`、`/mnt` 找到该文件，因此没有启动上传。
- 本机已有 `/usr/bin/rsync` 和 `/usr/bin/scp`，没有 `rclone`。
- 云端实例在线，`/root/autodl-tmp` 剩余约 141G，可以保存该分卷。
- 云端尚未安装 `rsync`；若采用推荐的断点续传方案，需要先在云端安装。
- `.0011` 是 12 个连续分卷中的最后一卷，单独上传后不能解压或训练。完整归档仍需要 `.0000～.0010`，总计约 1.196 TB，当前 170G 数据盘无法同时保存完整 OpenDataLab 分卷。

因此建议仅在确实需要备份该分卷时，用 `rsync --partial --append-verify` 经专用 SSH 密钥上传；压缩包不使用 `-z` 二次压缩。若目标仍是当前 Boston 官方 ZIP，则该 `.0011` 不能替代 Boston ZIP，应继续第 39 节的官方 ZIP 断点下载。

### 39.6 2026-08-10 恢复 Boston ZIP 快速续传

用户明确要求继续下载第 39.5 节暂停的两个断点文件。启动前检查结果：

```text
.part 文件：存在
.part.aria2 控制文件：存在，4,908 bytes
重复 aria2 进程：0
数据盘剩余：141G
```

随后使用第 39.3 节已经验证过的参数原地续传，关键设置为：

```text
--continue=true
--split=16
--max-connection-per-server=16
--lowest-speed-limit=0
--file-allocation=none
```

启动结果：

```text
外层校验包装进程 PID：1878
aria2 下载进程 PID：1884
有效连接数：16
```

外层包装进程仍负责下载完成后的精确大小检查、ZIP 中央目录检查和 `.part` 到正式 `.zip` 的改名；不会解压或触发预处理、监督训练、RL 微调。

续传初期直连速度在约 `0.34～0.69 MiB/s` 波动。为判断 AutoDL 网络加速是否有帮助，保持现有下载不停止，使用 `/etc/network_turbo` 对同一官方 URL 发起 100 MiB Range 只读测速，20 秒结果为：

```text
HTTP status：206
接收：887,266 bytes
平均速度：44,365 bytes/s（约 43.3 KiB/s）
```

该代理明显慢于 S3 直连，因此没有把正式下载切换到代理，也没有使用未经确认的镜像。aria2 已经使用单一服务器允许的 16 连接配置。

截至 `2026-08-10 01:30:42 +08:00`：

```text
aria2 显示：14 GiB / 35 GiB（39%，显示值有取整）
当前速度：525 KiB/s
当前 ETA：约 11 小时 56 分
连接数：16
下载进程：正常运行
```

### 39.7 当前下载速度瓶颈诊断

用户询问 Boston ZIP 下载速度受什么影响。保持下载运行并进行只读检查，结果为：

```text
aria2 连接数：16
aria2 CPU：约 0.6%
aria2 内存占比：接近 0
系统可用内存：约 961 GiB
数据盘可用空间：141G
观测速度：约 0.40～0.44 MiB/s
```

因此当前瓶颈不是 GPU、CPU、内存或磁盘容量。0.4 MiB/s 的写入量也远低于数据盘正常吞吐能力。

结合此前证据，主要瓶颈是 AutoDL 数据中心到 AWS `ap-northeast-1` 官方 S3 的跨云网络路径与共享出口拥塞：

- 单条连接此前多次只有约 `23～32 KiB/s`；16 条连接汇总后仍只有数百 KiB/s。
- 同一任务曾短时达到约 `1.7～2.2 MiB/s`，之后下降到约 `0.4 MiB/s`，说明线路质量和共享带宽随时间波动，而不是代码设置了固定限速。
- SSH 本身也多次出现 `Connection refused` 或 banner 阶段超时，进一步说明实例所在网络入口/出口存在波动；SSH 路径与 S3 路径并不完全相同，因此这只能作为云端网络不稳定的旁证。
- aria2 已使用该单源配置的 16 条连接；继续增加并发通常不能解决共享出口或跨境路由瓶颈，还可能增加重传与连接抖动。
- AutoDL `/etc/network_turbo` 代理对同一 S3 Range 的实测速度仅约 `43.3 KiB/s`，比直连更慢，故未使用。

用户所在地湖北只影响本地电脑到 AutoDL 的 SSH 体验，不直接决定“云端实例到 AWS S3”的下载速度；无 GPU 模式也不会降低这类纯 HTTP 下载的理论速度。当前能做的稳妥选择是保持 16 连接断点续传，等待共享线路恢复，或更换到对 AWS 东京线路更好的实例地区后从断点继续。

### 39.8 2026-08-10 第二次安全暂停

用户再次要求暂停 Boston ZIP 下载。暂停工具调用的界面回合曾被中断，但远端命令已经在中断前完成。后续重新连接并检查，确认无需重复发送信号。

结果：

```text
aria2 安全退出时间：2026-08-10 01:41:42 +08:00
复查时间：2026-08-10 01:47:39 +08:00
最后进度：14 GiB / 35 GiB（40%）
暂停前瞬时速度：335 KiB/s
该次运行平均速度：395 KiB/s
Boston aria2 进程数：0
断点控制文件：存在，4,908 bytes
```

aria2 日志包含正常的 `Shutdown sequence commencing` 与 `aria2 will resume download if the transfer is restarted`，不是异常崩溃。`.part` 与 `.part.aria2` 的最后修改时间一致，均为 `01:41:42`，复查时没有继续增长。已有数据和断点均已保留，可以关闭实例或以后使用第 39.6 节相同命令续传。

### 39.9 关机后以无卡模式恢复续传

用户关闭实例后重新以无 GPU 模式开机，并要求继续下载。启动前检查：

```text
实例：autodl-container-b8e548a0d5-61312194
.part 文件：存在，修改时间仍为暂停时的 01:41:42
.part.aria2：存在，4,908 bytes
已有 aria2 进程：0
数据盘可用空间：141G
```

这证明关机没有破坏数据盘断点。随后使用与第 39.6 节相同的 16 连接 aria2 参数恢复，未重新下载已有部分。

```text
恢复时间：2026-08-10 01:55:44 +08:00
恢复时已分配数据：15,321,165,824 bytes
新 aria2 PID：1275
连接数：16
恢复进度：14 GiB / 35 GiB（40%）
```

启动最初速度曾在约 `0.7～1.5 MiB/s` 波动；约 52 秒后稳定观测值为 `389 KiB/s`，对应动态 ETA 约 15 小时 55 分。无卡模式不影响 aria2 运行，本轮没有启动解压、NPZ 预处理、监督训练或 RL 微调。

## 41. 本地 Singapore 官方 ZIP 可新增 NPZ 分析

用户在本地下载：

```text
/home/yanjun/NewDisk/nuplan-v1.1_train_singapore.zip
```

本轮只读取 ZIP 中央目录和正式 100k plan，没有解压、上传或启动预处理。

### 41.1 ZIP 来源与内容核对

本地文件：

```text
大小：34,959,594,178 bytes（约 32.56 GiB）
ZIP member：2,398
其中 DB：2,396
```

官方 S3 对象头：

```text
HTTP：200 OK
Content-Length：34,959,594,178
ETag：fd44464d9ce2e3439b1124838f0f2890-4168
Last-Modified：2024-01-30 22:15:54 GMT
```

本地大小与官方对象精确一致，ZIP 中央目录能够正常读取。这里尚未执行耗时的全文件 CRC 解压测试，因此当前结论是“大小与目录结构通过”，不是完整字节哈希结论。

### 41.2 与正式 100k plan 匹配

将 ZIP 内 2,396 个 DB basename 与云端：

```text
/root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json
及 264 个 shard_*_logs.json
```

逐项匹配，结果：

```text
ZIP DB：2,396
正式 plan 匹配日志：2,396
ZIP 中计划外 DB：0
```

说明该 Singapore ZIP 中的全部 DB 都属于当前正式 100k 日志来源。

### 41.3 排除已有 2,700 NPZ 后的可新增量

当前已完成 7 个正式 shard，其中：

```text
shard_00175：纯 Singapore，已生成 350 NPZ
```

正式 plan 中共有 28 个“50 个日志全部来自 Singapore”的纯 Singapore shard，目标合计 9,800 NPZ。排除已完成的 `shard_00175` 后，还剩 27 个纯 Singapore shard：

```text
shard_00179
shard_00183 shard_00184 shard_00185 shard_00188 shard_00191
shard_00215 shard_00218 shard_00219 shard_00220 shard_00221 shard_00222
shard_00225 shard_00226 shard_00227 shard_00228 shard_00229
shard_00233 shard_00234 shard_00235 shard_00236 shard_00237 shard_00238
shard_00243 shard_00247 shard_00248 shard_00249
```

每个目标为 350 NPZ，因此可在不重复现有 2,700 个的前提下新增：

```text
27 × 350 = 9,450 NPZ
```

若全部成功，正式进度将从：

```text
2,700 / 100,000
```

提升到：

```text
12,150 / 100,000
```

其余 996 个 Singapore 日志分散在 42 个混合城市 shard 中。现有执行器按完整 shard 的 50 个日志和固定目标数运行，只有 Singapore ZIP 时不能正确完成这些混合 shard；需要同时提供其中其他城市的 DB，或者另行实现按 log 级别精确分配、合并与去重。为保持当前正式 manifest 和可复现性，本轮建议先只处理上述 27 个纯 Singapore shard。

匹配分析时从云端只读复制了约几百 KB 的 plan 文件到 `/tmp/tmp.YrSFUlEZoe/nuplan_train_100k`；未修改项目或云端 plan。第一次分析命令因包含临时目录清理而被安全策略在执行前拒绝，去掉清理后完成统计。

## 42. 本地 ZIP 定向预处理能力改造

### 42.1 为什么需要改造

原执行器主要面向在线归档下载和连续 shard 范围。当前 Singapore、Pittsburgh 已经是本地完整 ZIP，如果仍走网络下载会重复消耗时间；同时每个城市只有部分 shard 能由单个城市 ZIP 独立满足，不能简单运行一个连续编号区间。

因此本轮只改造数据编排层，不修改场景抽取、特征计算或 NPZ 内容：

- `download_nuplan_log_subset.py` 支持把本地 ZIP 路径或 `file://` 作为 archive，并新增 `zip` backend，直接从 ZIP 中抽取指定 DB member。
- `run_preprocessing_range.py` 新增 `--shard_indices`，可显式指定不连续 shard 白名单。
- `--archive NAME=PATH` 和 `--only_archive NAME` 会继续传给单 shard 下载器，避免从错误城市归档取 DB。
- 白名单、archive 和 only-archive 被写入 state JSON，使中断续跑仍使用同一数据边界。
- 保留 `--cleanup_raw`：每个 shard 生成并校验 NPZ 后立即删除临时解压 DB，只保留原 ZIP、NPZ 和报告。

这里选择“只运行纯城市 shard”，是因为正式 100k plan 的一个 shard 固定包含 50 个日志并要求产生固定数量样本。混合城市 shard 还需要其他城市 DB；仅使用单城市 ZIP 强行运行，会得到不完整 shard，并破坏正式 manifest 的可复现性。

### 42.2 验证

新增/扩展测试覆盖：

- 本地 ZIP member 抽取；
- local path 与 `file://`；
- `zip` backend 参数校验；
- `--shard_indices` 白名单选择；
- state 中断续跑；
- archive 参数向下游传递。

验证结果：

```text
定向测试：13 passed in 0.42s
HDP-nuplan/tests 全量测试：59 passed, 15 warnings in 7.26s
py_compile：通过
```

正式 100k plan 已持久化到：

```text
/home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k
```

其中包含 1 个主 plan 和 264 个 shard 日志清单，共 265 个文件；主 plan SHA256 为：

```text
04a1385a60fcead9aea6084298691b09d46b8fd6ae67578de5ce54bd4f381f18
```

## 43. Singapore 本地正式预处理

### 43.1 执行策略

排除云端已经完成的 `shard_00175` 后，显式选择第 41.3 节列出的 27 个纯 Singapore shard。先仅运行一个 shard 做 smoke test，再使用同一 state 文件续跑其余 shard。

核心执行参数为：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /home/yanjun/NewDisk/nuplan/indexes/archive_index_train_singapore_local.json \
  --raw_root /home/yanjun/NewDisk/nuplan/raw_singapore_local \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_singapore_local \
  --state_path /home/yanjun/NewDisk/logs/nuplan_singapore_local_preprocessing_state.json \
  --shard_indices 179 183 184 185 188 191 215 218 219 220 221 222 225 226 227 228 229 233 234 235 236 237 238 243 247 248 249 \
  --worker_index 0 --worker_count 1 \
  --checksum_mode files --cleanup_raw \
  --download_backend zip \
  --archive singapore=/home/yanjun/NewDisk/nuplan-v1.1_train_singapore.zip \
  --only_archive singapore \
  --min_free_gib 30
```

smoke 阶段额外使用 `--max_shards 1`；验证通过后去掉该参数。state 中的 `next_position` 从 1 继续，因此不会重复处理首 shard。

第一次曾尝试用普通后台子进程运行，但交互工具结束时其后代进程被回收；这不是 OOM 或预处理代码异常。之后改用持久 PTY 会话，已有 19 个有效 DB 在恢复时经过校验后被跳过，未重复写坏数据。

### 43.2 Singapore 最终结果

```text
state：complete
shard：27 / 27
每 shard：350 NPZ
NPZ 总数：9,450
唯一文件名：9,450
内部重复：0
无效 shard：0
残留临时 DB：0
输出占用：约 1.5G
```

state 时间：

```text
started_at_utc：2026-08-09T19:50:10.190823+00:00
updated_at_utc：2026-08-09T20:27:18.022709+00:00
```

输出目录：

```text
/home/yanjun/NewDisk/processed/nuplan_train_100k_gate_singapore_local
```

## 44. Pittsburgh ZIP 检查与本地正式预处理

### 44.1 ZIP 完整性与 plan 匹配

用户下载完成的文件：

```text
/home/yanjun/NewDisk/nuplan-v1.1_train_pittsburgh.zip
```

检查结果：

```text
本地大小：30,620,248,893 bytes
官方 Content-Length：30,620,248,893 bytes
ZIP member：1,562
DB：1,560
正式 plan 匹配 DB：1,560
计划外 DB：0
官方 Last-Modified：2024-01-30 22:14:06 GMT
官方 ETag：6d9100ba0b89c9b0e997cf99c1ef739e-3651
```

ZIP 中央目录可正常读取。本地大小与官方对象精确一致；本轮未把 multipart ETag 错当作普通 MD5。

正式 plan 中共有 5 个纯 Pittsburgh shard。其中 `shard_00152` 已包含在云端 2,700 个正式 NPZ 内，因此排除；本地只处理：

```text
shard_00155 shard_00159 shard_00166 shard_00190
```

其余 1,310 个 Pittsburgh 日志分布于 54 个混合城市 shard，需要对应的其他城市 ZIP 后再完整处理，本轮不生成残缺 shard。

### 44.2 smoke 与续跑命令

首先给以下命令增加 `--max_shards 1`，只运行 `shard_00155`：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /home/yanjun/NewDisk/nuplan/indexes/archive_index_train_pittsburgh_local.json \
  --raw_root /home/yanjun/NewDisk/nuplan/raw_pittsburgh_local \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_pittsburgh_local \
  --state_path /home/yanjun/NewDisk/logs/nuplan_pittsburgh_local_preprocessing_state.json \
  --shard_indices 155 159 166 190 \
  --worker_index 0 --worker_count 1 \
  --checksum_mode files --cleanup_raw \
  --download_backend zip \
  --archive pittsburgh=/home/yanjun/NewDisk/nuplan-v1.1_train_pittsburgh.zip \
  --only_archive pittsburgh \
  --min_free_gib 30
```

`shard_00155` smoke 结果：

```text
processing status：complete
validation status：passed
manifest / NPZ：350 / 350
日志数：46
失败：0
内部重复：0
与 Singapore 9,450 个 NPZ 的文件名交集：0
临时 DB：已删除
```

确认通过后，去掉 `--max_shards 1` 并使用同一 state 文件续跑。执行器从 `next_position=1` 开始，仅处理 `00159、00166、00190`。

### 44.3 Pittsburgh 最终结果

```text
state：complete
next_position：4
shard：4 / 4
每 shard：350 NPZ
NPZ 总数：1,400
唯一文件名：1,400
内部重复：0
无效 shard：0
残留临时 DB：0
输出占用：约 214M
```

四个 shard 的 `processing_report.json` 均为 `complete`、`failed=0`，`cache_validation_report.json` 均为 `passed`，manifest 与实际 NPZ 都是 350。

state 时间：

```text
started_at_utc：2026-08-10T04:09:00.845259+00:00
updated_at_utc：2026-08-10T04:16:41.060275+00:00
```

输出目录：

```text
/home/yanjun/NewDisk/processed/nuplan_train_100k_gate_pittsburgh_local
```

## 45. 两个本地城市的最终去重审计

文件名级审计结果：

```text
Singapore：9,450 NPZ
Pittsburgh：1,400 NPZ
Pittsburgh vs Singapore 交集：0
本地合计：10,850 NPZ
本地合计唯一文件名：10,850
```

云端已完成 shard 为：

```text
00000 00001 00002 00003 00152 00175 00197
```

本地运行的 Singapore/Pittsburgh shard 已显式排除 `00175` 和 `00152`，与云端 shard 编号无交集。因此按正式 plan 的 shard 分配，本地新增样本与云端 2,700 个样本在计划层面无重复，合并后的预期正式进度为：

```text
2,700 + 9,450 + 1,400 = 13,550 / 100,000
```

第一次执行云端实物文件名交集复核时，SSH 返回：

```text
ssh: connect to host connect.cqa1.seetacloud.com port 11156: Connection refused
```

该次失败没有被误记为“交集为 0”。稍后按相同命令重试，SSH 恢复，远端直接读取正式输出目录并与通过标准输入传入的本地文件名集合求交集，结果为：

```text
cloud_count = 2,700
incoming_count = 10,850
overlap = 0
```

因此现在已同时确认：本地两批实物文件互不重复，本地新增 10,850 个与云端正式 2,700 个也无文件名重复。三批可合并为 13,550 个唯一正式 NPZ。

本轮结束时 `/home/yanjun/NewDisk` 约剩余 63G。ZIP 保留，临时解压 DB 已清理，故后续可从 state 与报告复核，不需要重新生成已经通过门禁的 shard。

## 46. Boston ZIP 下载完成状态复核

在 `2026-08-10 12:32 CST` 只读检查云端，结果为：

```text
aria2 进程：0
下载日志状态：OK / download completed
下载完成时间：2026-08-10 10:05:30 +08:00
文件大小：38,161,149,300 bytes
.aria2 控制文件：不存在
正式 .zip 文件：尚不存在
```

随后使用项目 conda 环境中的 Python 只读取 ZIP 中央目录：

```text
zip_directory_readable：True
ZIP member：1,649
DB member：1,647
ZIP comment：0 bytes
```

因此 Boston ZIP 已经下载完成，目前没有继续占用带宽。文件仍名为：

```text
/root/autodl-tmp/nuplan/official_archives/.nuplan-v1.1_train_boston.zip.part
```

原因是 aria2 完成后，外层包装脚本调用系统 `python3` 做最终校验，但该镜像没有 `python3` 命令，日志报 `python3: command not found`，所以未执行最后的改名。下载本身和 ZIP 中央目录不受该后处理错误影响；正式改名与进一步处理应在完成额外校验后单独执行。

## 47. 并行上传本地 10,850 NPZ 与处理 Boston ZIP

### 47.1 开机后的边界检查

用户重新开机后，先只读核对云端：

```text
正式输出：7 shard / 2,700 NPZ
数据盘：170G，总剩余 119G
CPU：208 个逻辑核
内存：约 1.0 TiB
活动下载/预处理任务：0
rsync：/usr/bin/rsync
screen：/usr/bin/screen
```

实际地图路径为：

```text
/root/autodl-tmp/nuplan/maps
```

不是此前假设的 `/root/autodl-tmp/nuplan/dataset/maps`。目录内同时存在 Boston、Pittsburgh、Singapore 和 Las Vegas 的 map.gpkg。

云端脚本仍是旧版，只支持 `remotezip/curl`，不支持本地 ZIP 和离散 shard 白名单。因此将本地已经通过 59 项测试的两个脚本同步到云端：

```text
download_nuplan_log_subset.py
SHA256：38195811c10c5ed366d388fcef2e6c282536b19e37df5d8d3bede21f62d607f8

run_preprocessing_range.py
SHA256：29a2d35eb0dfbc14329c1ebff4a30e1255d56562ea9a573db0e655d37b9eb3c1
```

同步后云端 SHA256 与本地完全一致，`py_compile` 通过。

### 47.2 为什么使用并行但分离的目录

本轮同时执行两条流水线：

1. 本地 Singapore/Pittsburgh 结果上传到独立暂存目录；
2. 云端 Boston ZIP 做 CRC 和预处理，输出到独立 Boston 目录。

没有直接上传到已有 2,700 NPZ 的正式根目录。这样即使上传中断、ZIP 损坏或 Boston 处理失败，也不会把半成品混入正式缓存。只有三批数据全部通过 checksum 和去重门禁后，才复制到正式根目录。

## 48. 本地 10,850 NPZ 上传与校验

### 48.1 上传命令

目标暂存目录：

```text
/root/autodl-tmp/processed/nuplan_train_100k_gate_incoming_local
```

使用支持断点续传和传输校验的 rsync：

```bash
rsync -a --partial --append-verify --info=progress2 \
  -e 'ssh -i /home/yanjun/.ssh/id_ed25519_codex_autodl -p 11156' \
  /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_singapore_local \
  /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_pittsburgh_local \
  root@connect.cqa1.seetacloud.com:\
/root/autodl-tmp/processed/nuplan_train_100k_gate_incoming_local/
```

传输结果：

```text
传输字节：1,711,561,353
耗时：约 11 分 03 秒
平均速度：约 2.46 MB/s
rsync 退出码：0
```

### 48.2 上传后门禁

首先核对结构和报告：

```text
Singapore：27 shard / 9,450 NPZ
Pittsburgh：4 shard / 1,400 NPZ
合计：31 shard / 10,850 NPZ
唯一文件名：10,850
内部重复：0
无效 shard：0
与云端已有 2,700 NPZ 交集：0
与正在生成的 Boston NPZ 交集：0
残留 .partial：0
```

随后重新执行：

```bash
rsync -a --dry-run --checksum --itemize-changes ...
```

结果退出码为 0 且无差异输出，确认本地源与云端暂存目录逐文件一致。上传任务到此才被标记为完成。

## 49. Boston ZIP 全 CRC 与正式预处理

### 49.1 全 ZIP CRC 与正式命名

下载完成时 ZIP 仍名为 `.part`。本轮不是直接改名，而是使用项目 conda 环境的 Python 顺序读取全部 ZIP member，使 `zipfile` 验证每个 member 的解压流和 CRC：

```text
文件大小：38,161,149,300 bytes
member：1,649 / 1,649 通过
DB member：1,647
CRC 错误：0
```

全部通过后才原子改名为：

```text
/root/autodl-tmp/nuplan/official_archives/nuplan-v1.1_train_boston.zip
```

### 49.2 Boston shard 匹配

第一次匹配错误地对 plan 中已经没有 `.db` 后缀的字符串使用 `Path.stem`。由于日志名包含多个点，`Path.stem` 会把最后一个点之后的内容当成后缀移除，导致错误结果为 0。该结果没有用于执行。

修正为只显式移除 `.db` 后缀后，结果为：

```text
Boston DB：1,647
正式 plan 匹配日志：1,647
涉及 shard：70
纯 Boston shard：9
混合城市 shard：61，包含 1,217 个 Boston 日志
```

9 个纯 Boston shard 为：

```text
00197 00245 00256 00258 00259 00260 00261 00262 00263
```

其中 `00197` 已在原有 2,700 NPZ 中，因此只处理后 8 个。目标数量不是简单的 `8 × 350`：最后一个 `00263` 按正式 plan 只有 30 个日志和 210 个场景，因此总目标为：

```text
7 × 350 + 210 = 2,660 NPZ
```

61 个混合城市 shard 没有强行处理，因为还缺少对应的其他城市 DB。

### 49.3 冒烟和续跑

核心命令为：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
/root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /root/autodl-tmp/nuplan/indexes/archive_index_train_boston_local.json \
  --raw_root /root/autodl-tmp/nuplan/raw_boston_local \
  --map_path /root/autodl-tmp/nuplan/maps \
  --output_root /root/autodl-tmp/processed/nuplan_train_100k_gate_boston \
  --state_path /root/autodl-tmp/logs/nuplan_boston_local_preprocessing_state.json \
  --shard_indices 245 256 258 259 260 261 262 263 \
  --worker_index 0 --worker_count 1 \
  --checksum_mode files --cleanup_raw \
  --download_backend zip \
  --archive boston=/root/autodl-tmp/nuplan/official_archives/nuplan-v1.1_train_boston.zip \
  --only_archive boston \
  --min_free_gib 30
```

冒烟阶段额外传入 `--max_shards 1`，只处理 `00245`。结果：

```text
350 / 350 NPZ
processing：complete
validation：passed
failed：0
唯一文件名：350
与已有正式 NPZ 交集：0
残留临时 DB：0
```

之后去掉 `--max_shards 1`，同一 state 从 `next_position=1` 续跑其余 7 个。

### 49.4 Boston 最终结果

```text
state：complete
next_position：8
shard：8 / 8
NPZ：2,660
内部重复：0
无效 shard：0
残留临时 DB：0
```

state 时间：

```text
started_at_utc：2026-08-10T04:47:40.780951+00:00
finished_at_utc：2026-08-10T05:03:22.758908+00:00
```

上传和 Boston 预处理确实并行运行；两者均不依赖 GPU。

## 50. 三批数据强门禁与正式目录合并

### 50.1 合并前强门禁

对新增的 39 个 shard 逐一检查：

- shard 集合与显式白名单完全一致；
- `processing_report.status=complete` 且 `failed=0`；
- `cache_validation_report.status=passed`；
- manifest 数量、实际 NPZ 数量和 plan 目标一致；
- processing/download report 的日志列表与正式 plan 完全一致；
- 每个 NPZ 的实际 SHA256 与 `checksums.json` 一致；
- shard 内和跨根目录没有重复文件名。

门禁汇总：

```text
已有正式：2,700
Singapore：9,450，交集 0
Pittsburgh：1,400，交集 0
Boston：2,660，交集 0
合计：16,210
```

### 50.2 非破坏性合并

确认正式根目录不存在任何同名目标 shard、预处理进程已经结束后，使用 rsync 依次复制三个来源：

```text
Singapore：1,490,623,943 bytes
Pittsburgh：220,937,410 bytes
Boston：419,577,464 bytes
删除文件：0
```

暂存目录和独立 Boston 输出均保留，没有用移动或删除替代复制，便于出现问题时回溯。

第一次合并后 dry-run 只报告：

```text
.d..tpog... ./
```

它表示三个源目录与共同目标根目录 `./` 的时间、权限、owner/group 元数据不同，不是文件内容差异。随后使用 `--omit-dir-times` 并只比较文件内容，结果为：

```text
source_to_canonical_checksum_diff = 0
```

最终正式目录：

```text
/root/autodl-tmp/processed/nuplan_train_100k_gate
```

最终审计：

```text
shard：46
NPZ：16,210
唯一文件名：16,210
重复：0
manifest 数量逐 shard 合计：16,210
无效 shard：0
活动预处理任务：0
数据盘剩余：115G
```

一次验证命令最后显示退出码 1，是因为启用了 `set -euo pipefail`，而 `pgrep` 在没有匹配进程时按约定返回 1；此前所有数据审计已输出并通过。之后单独复查确认没有活动预处理任务，因此该退出码不是数据失败。

当前正式进度为：

```text
16,210 / 100,000 = 16.21%
```

尚缺 83,790 个 NPZ。因为 264 个 shard 尚未全部完成，本轮没有运行 `finalize_autodl_preprocessing.sh`，没有生成伪全量 `train_manifest.json`，也没有启动监督训练或 RL 训练。

## 51. 4×RTX 4090 上先训练 16,210 NPZ 的可行性检查

用户将实例切换为 4×RTX 4090 后，本轮只读检查，尚未启动训练。

### 51.1 硬件与运行时

```text
GPU：4 × NVIDIA GeForce RTX 4090
单卡显存：24,091 MiB（PyTorch 可见值）
检查时显存占用：约 1 MiB/卡
GPU 利用率：0%/卡
PyTorch：2.0.0+cu118
CUDA runtime：11.8
NCCL：2.14.3
torch.cuda.device_count()：4
数据盘剩余：115G
活动训练进程：0
```

因此硬件和 DDP 基础条件满足。

### 51.2 当前不能直接启动的两个原因

正式缓存根目录虽然已有：

```text
46 shard / 16,210 NPZ
```

但根目录现存的 `train_manifest.json` 是旧文件：

```text
manifest_count：1,500
shard_count：4
```

它只覆盖早期四城市门禁实验，直接传给 `train_predictor.py` 会只训练 1,500 条，而不是 16,210 条。该旧文件不能被误当成当前 partial manifest，也不应冒充最终 100k manifest。

仓库现有 `torch_run.sh` 也不能直接使用：

- `CUDA_VISIBLE_DEVICES` 和 `--nproc-per-node` 硬编码为 8 卡；
- Python、训练集和 manifest 路径仍为空；
- 末尾存在 `--batch_size 2048train` 拼写错误。

因此需要单独生成命名明确的 `train_manifest_16210.json`，并使用 4 卡命令启动。

### 51.3 推荐的阶段性训练定义

16,210 个 NPZ 足够进行阶段性监督训练，验证多卡收敛、损失趋势、checkpoint 和后续 RL 接口，但不能作为完整 100k 最终实验。当前数据城市分布也偏向 Singapore，完整数据补齐后仍需继续监督训练并重新做统一评测。

推荐配置：

```text
GPU / DDP rank：4
全局 batch size：32
单卡 batch size：8
num_workers：8/进程
阶段 epoch：20
warm-up：2 epoch
checkpoint 间隔：5 epoch
diffusion model/supervision：x_start / x_start
planning_hybrid_loss：0.1
```

单卡 batch 8 已被早期 smoke 验证过；四卡全局 batch 32 保持相同单卡显存压力。16,210 个样本在 `DistributedSampler + drop_last=True` 下，每轮约 506 个 DDP update step。

建议从原 Diffusion-Planner checkpoint 只迁移兼容 encoder：

```text
/root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth
```

不建议直接 resume 早期 1,500 样本的单 epoch checkpoint，因为那个实验的数据范围和仅 1 epoch 的调度器状态都不同。新建独立实验可获得更清晰、可复现的 16,210 阶段基线。

下一步应依次执行：生成并校验 16,210 专用 manifest；做四卡首批/首轮监控；确认无 NaN、无 OOM、四卡均有负载后继续 20 epoch。本节只记录方案，没有创建 manifest 或启动训练。

## 52. 生成 16,210 专用 partial manifest

### 52.1 首次尝试的安全失败

首次调用云端 `merge_preprocessing_shards.py` 的 `merge_shards(..., plan_path=None)` 时，云端脚本仍是旧版，报错：

```text
TypeError: merge_shards() got an unexpected keyword argument 'plan_path'
```

因为异常发生在创建输出目录之前，所以没有生成半成品，也没有覆盖根目录中旧的 1,500 条 `train_manifest.json`。随后验证器因目标 manifest 不存在而退出，这只是上游未生成文件的连带结果。

将本地已经通过测试的新版本同步到云端：

```text
merge_preprocessing_shards.py
SHA256：e98c483a7c1ef955f55cbc331ed32289fb042d9408e71244928a2baa04d9c5f5
```

云端 `py_compile` 通过后重新生成。

### 52.2 专用 manifest 结果

为了不与最终 100k manifest 混淆，文件放在独立目录：

```text
/root/autodl-tmp/processed/manifests/nuplan_train_partial_16210/
```

生成文件：

```text
train_manifest_16210.json
train_merged_report_16210.json
train_cache_validation_report_16210.json
smoke_128.json
```

强校验结果：

```text
manifest_count：16,210
manifest_unique：16,210
shard_count：46
selected_log_count：2,132
manifest SHA256：855ade50b4a81fe118f63605779e41ef6077a1bfbcfc623fb3f6e55e1190ad7e
cache validation：passed
实际 NPZ：16,210
smoke manifest：128
```

这里的 `selected_log_count=2,132` 是实际贡献了入选场景的日志数量，不等于 46 个 shard 中所有日志条目数量。旧的 1,500 manifest 保持原样，本轮训练显式传入新的绝对路径。

## 53. 四卡 DDP 冒烟训练

### 53.1 命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 4 --standalone \
  train_predictor.py \
  --name hdp-partial-16210-4gpu-smoke-128 \
  --save_dir /root/autodl-tmp/experiments/hdp_partial_16210_4gpu_smoke \
  --train_set /root/autodl-tmp/processed/nuplan_train_100k_gate \
  --train_set_list /root/autodl-tmp/processed/manifests/nuplan_train_partial_16210/smoke_128.json \
  --normalization_file_path /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/normalization.json \
  --encoder_pretrained_model_path /root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth \
  --freeze_encoder_epochs 0 \
  --diffusion_model_type x_start \
  --diffusion_supervision_type x_start \
  --planning_hybrid_loss 0.1 \
  --train_epochs 1 --warm_up_epoch 1 --save_utd 1 \
  --batch_size 32 --num_workers 4 --seed 3407
```

### 53.2 结果

```text
DDP rank：0、1、2、3 全部初始化成功
数据：128
全局 batch：32
每 rank DataLoader batch：8
每轮 DDP step：4
encoder warm-start：151 / 151 tensor
encoder 参数：1,799,040
decoder 从旧 checkpoint 加载：0
总模型参数：5,092,996
epoch loss：53.1319
退出码：0
```

没有 OOM、NaN、NCCL 或 unused-parameter 错误，checkpoint 正常保存。该 smoke 仅验证四卡和训练链路，不用于模型效果比较。

## 54. 16,210 样本四卡 20 epoch 阶段训练

### 54.1 后台启动

正式阶段训练使用独立 screen：

```text
screen：hdp16210_e20
日志：/root/autodl-tmp/logs/hdp_partial_16210_4gpu_e20.log
退出码文件：/root/autodl-tmp/logs/hdp_partial_16210_4gpu_e20.exit
```

关键命令参数：

```text
name：hdp-official-partial-16210-4gpu-e20
train_set：/root/autodl-tmp/processed/nuplan_train_100k_gate
train_set_list：train_manifest_16210.json
DDP：4 rank
train_epochs：20
global batch_size：32（每卡 8）
num_workers：8/进程
learning_rate：5e-4
warm_up_epoch：2
save_utd：5
model/supervision：x_start/x_start
planning_hybrid_loss：0.1
encoder warm-start：原 Diffusion-Planner model.pth
freeze_encoder_epochs：0
seed：3407
```

运行目录：

```text
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/
training_log/hdp-official-partial-16210-4gpu-e20/2026-08-10-13:22:54
```

启动后确认 4 个 rank 都识别到 16,210 条数据，单 rank 每轮 506 个 DataLoader batch。四卡典型显存约 1.7～1.8 GiB，利用率采样约 20%～53%。模型规模较小且需要读取大量 NPZ，因此没有吃满 24G 显存。

DDP 输出过以下 warning：

```text
find_unused_parameters=True was specified ... but did not find any unused parameters
```

这是额外遍历计算图的性能提示，不影响数值正确性；为避免在运行中改变已经通过 smoke 的代码，本轮没有停训修改。

### 54.2 loss 过程

20 个 epoch 的 train loss：

```text
01  5.3149
02  3.4918
03  2.2625
04  1.9107
05  1.6996
06  1.4918
07  1.4846
08  1.4149
09  1.3493
10  1.2882
11  1.2567
12  1.2501
13  1.1949
14  1.1085
15  1.0984
16  1.0891
17  1.0932
18  1.0328
19  1.0059
20  1.0450
```

总体从 5.3149 降到 1.0450。epoch 20 相比 epoch 19 有小幅回升，但仍低于 epoch 15；仅凭 train loss 不能判断闭环规划效果，需要下一步评测。

用户本地网络中途断开，训练没有中断，因为任务运行在云端独立 screen 中。重连时训练已继续到 epoch 17，说明后台隔离生效。

### 54.3 checkpoint 与最终验收

保存文件：

```text
model_epoch_5_trainloss_1.6996.pth
model_epoch_10_trainloss_1.2882.pth
model_epoch_15_trainloss_1.0984.pth
model_epoch_20_trainloss_1.0450.pth
latest.pth
```

每个 checkpoint 约 81,936,261 bytes。最终 checkpoint 包含：

```text
model
ema_state_dict
optimizer
schedule
epoch = 20
loss
wandb_id
```

`latest.pth` 与 epoch 20 checkpoint 的 SHA256 完全相同：

```text
eb8308739228517973468dc445be843a0a030afc5974bbc293caf14935f6210a
```

最终状态：

```text
训练开始：2026-08-10 13:22:54 +08:00
最终文件：2026-08-10 13:38:40 +08:00
耗时：约 15 分 46 秒
退出码：0
loss 条目：20
非有限 loss：0
OOM/NaN/NCCL/ChildFailedError：0
活动训练进程：0
GPU：全部恢复 0% / 约 1 MiB
数据盘剩余：115G
```

本轮已经完成阶段性监督训练，但还没有执行离线指标、NuPlan 闭环评测或 RL 微调，因此不能只凭训练 loss 宣称规划性能提升。

## 55. 16,210 场景监督 checkpoint 的单卡评测与运行时诊断

时间：2026-08-10。

目标：在训练结束后选择监督 checkpoint，并使用独立 open-loop 数据和 NuPlan 官方 closed-loop 指标验证规划性能。评测阶段不继续训练，也不启动 RL。

### 55.1 单卡实例和 checkpoint 检查

AutoDL 改为单张 RTX 4090 后，重新连接：

```bash
ssh -p 11156 root@connect.cqa1.seetacloud.com
```

检查结果：

```text
GPU：NVIDIA GeForce RTX 4090，24564 MiB
GPU 初始占用：1 MiB
数据盘：170G，总占用 56G，剩余 115G
活动训练/预处理/评测进程：0
```

训练目录仍完整保留：

```text
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/
training_log/hdp-official-partial-16210-4gpu-e20/2026-08-10-13:22:54
```

四个 checkpoint 均可由 `torch.load(..., map_location="cpu")` 读取，每个都包含 `ema_state_dict`，且 EMA 中有 260 个张量。`latest.pth` 与 epoch 20 checkpoint 的 SHA256 仍完全一致。

检查过程中出现过一次 `Connection refused`，原因是实例刚启动时 SSH 映射尚未恢复；未启动任何任务，也没有产生半文件，端口恢复后继续。

### 55.2 恢复独立 mini-val 1k 缓存

云端项目包保留了验证 manifest 和历史 JSON，但没有包含 1,000 个验证 NPZ。本机仍保留：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/
tmp/mini_val_balanced_1000_seed3407_v1/cache
```

本机缓存状态：

```text
NPZ：1000
大小：153M
来源：10 个官方 mini-val 日志
```

只上传缺失的 `cache/`：

```bash
tar -C \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1 \
  -cf - cache \
| ssh -i ~/.ssh/id_ed25519_codex_autodl -p 11156 \
  root@connect.cqa1.seetacloud.com \
  'mkdir -p /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1 &&
   tar -xf - -C /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1'
```

上传后使用 Python 同时核对 manifest、实际文件和训练清单：

```text
val manifest 条目：1000
val manifest 唯一条目：1000
实际 val NPZ：1000
缺失：0
多余：0
train manifest 条目：16210
train/val 按 NPZ 文件名交集：0
```

因此该验证集与 16,210 训练清单物理隔离，不是拿训练数据做验证。

### 55.3 四个 checkpoint 的 open-loop 排名

评测统一严格加载每个 checkpoint 的 EMA 权重，使用相同 val1k、三个随机种子重复、`batch_size=32`，单张 RTX 4090：

```bash
cd /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan

PY=/root/autodl-tmp/conda_envs/diffusion_planner/bin/python
RUN=/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/training_log/hdp-official-partial-16210-4gpu-e20/2026-08-10-13:22:54
OUT=/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation

$PY evaluate_checkpoints.py \
  --args_file "$RUN/args.json" \
  --checkpoint_dir "$RUN" \
  --data_dir /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 32 \
  --num_workers 8 \
  --repeats 3 \
  --seed 3407 \
  --device cuda \
  --output "$OUT/mini_val_1000_checkpoint_ranking_repeat3.json"
```

第一次发出命令时 SSH 端口短暂拒绝连接，命令以 255 退出，未进入 Python、未生成有效结果。确认端口恢复后原样重跑，退出码为 0。

排名：

| open-loop 排名 | epoch | train loss | ego planning | hybrid | val total loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 1.098373 | 0.029608 | 4.804853 | 0.510093 |
| 2 | 10 | 1.288177 | 0.032616 | 5.157435 | 0.548360 |
| 3 | 20 | 1.044985 | 0.032208 | 5.938132 | 0.626021 |
| 4 | 5 | 1.699569 | 0.067435 | 9.793811 | 1.046816 |

仅按 open-loop loss 会选择 epoch 15。epoch 20 的 train loss 更低，但 val loss 已明显回升，不能默认最后一个 checkpoint 最好。

结果文件：

```text
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation/
mini_val_1000_checkpoint_ranking_repeat3.json
mini_val_1000_checkpoint_ranking_repeat3.log
```

### 55.4 上传三场景 closed-loop DB

固定三场景只依赖以下三个官方 mini-val DB：

```text
2021.06.07.12.54.00_veh-35_01843_02314.db
2021.06.08.14.35.24_veh-26_02555_03004.db
2021.07.24.23.50.16_veh-17_01696_02071.db
```

原始合计约 726M。最初使用未压缩 tar 流，实测本地上行约 1 MiB/s；抽样确认 SQLite DB 使用 `gzip -1` 后约为原大小的 60%，因此停止未完成的临时传输，改为 gzip 流。远端同名未完成 DB 被完整流覆盖，训练数据和 checkpoint 没有改动。

```bash
tar -C /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  -czf - \
  2021.06.08.14.35.24_veh-26_02555_03004.db \
  2021.06.07.12.54.00_veh-35_01843_02314.db \
  2021.07.24.23.50.16_veh-17_01696_02071.db \
| ssh -i ~/.ssh/id_ed25519_codex_autodl -p 11156 \
  root@connect.cqa1.seetacloud.com \
  'mkdir -p /root/autodl-tmp/nuplan/mini_val_3_db &&
   tar -xzf - -C /root/autodl-tmp/nuplan/mini_val_3_db'
```

逐个比较本机与云端 SHA256，三份完全一致：

```text
180c6a98315444ac30b1d049e009f0a98777194eb3af65bba9f3c3b5545866b1
6ddbbdc835ed87bf3d32e87e2441be32e733450493c73f72871cba557420140e
292c4cb4e72e1c1295b9dd0c52a9bd1a56826319f6a2d593664964e30cccc47f
```

### 55.5 云端 closed-loop 初测与异常

先按照 open-loop 排名选择 epoch 15，后台执行：

```bash
env \
  NUPLAN_DATA_ROOT=/root/autodl-tmp/nuplan \
  NUPLAN_MAPS_ROOT=/root/autodl-tmp/nuplan/maps \
  NUPLAN_EXP_ROOT=/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/closed_loop \
  NUPLAN_DEVKIT_ROOT=/root/autodl-tmp/workspace/nuplan-devkit \
  NUPLAN_MINI_DB_ROOT=/root/autodl-tmp/nuplan/mini_val_3_db \
  DIFFUSION_PLANNER_PYTHON=/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_mini_closed_loop.sh \
  hdp-partial16210-epoch15-mini-val-3 \
  "$RUN/args.json" \
  "$RUN/model_epoch_15_trainloss_1.0984.pth" \
  mini-val-closed-loop-3 \
  hdp
```

运行成功 3/3、失败 0、总仿真约 2 分 17 秒，但官方均分异常为 0：

```text
score：0.0
no_ego_at_fault_collisions：1.0
drivable_area_compliance：0.0
ego_is_making_progress：0.333333
ego_progress_along_expert_route：0.164737
time_to_collision_within_bound：0.333333
speed_limit_compliance：0.0
ego_is_comfortable：0.0
```

为判断是模型还是环境，随后使用 SHA256 与本机完全相同的仓库发布版 Diffusion-Planner `model.pth + args.json` 做控制组。第一次控制组因为从 `HDP-nuplan` 子目录启动时父目录不在 `PYTHONPATH`，Hydra 报 `Could not find planner/diffusion_planner`，没有进入仿真；添加：

```bash
PYTHONPATH=/root/autodl-tmp/workspace/Diffusion-Planner
```

后重跑成功 3/3，但云端控制组同样得到 `score=0.0`。这与本机同场景历史 `0.9958288173` 矛盾，说明云端 closed-loop 结果不能直接归因于新 checkpoint。

### 55.6 云端运行时异常的受控诊断

完成以下逐项排查：

1. 发布版 `model.pth` 和 `args.json` 本机/云端 SHA256 完全一致。
2. Diffusion-Planner 的 planner、model、decoder、sampling、SDE、DPM-Solver、normalizer 源码哈希一致。
3. nuPlan devkit 全部 Python/YAML 文件树聚合哈希一致。
4. 四城 `map.gpkg` 和 `nuplan-maps-v1.0.json` SHA256 一致。
5. 三个 DB SHA256 一致。
6. Hydra 最终配置除实验名和绝对路径外一致。
7. NumPy、SciPy、Pandas、Shapely、SQLAlchemy、Hydra、OmegaConf、GeoPandas、Fiona、GDAL、PROJ、GEOS 版本一致。
8. 本机仅使用三个 DB 的软链接目录重跑发布版控制组，仍精确得到 `0.9958288173`，排除 DB 目录中包含 64 个或 3 个 DB 的影响。
9. 同一个 val NPZ、同一 seed 的发布版模型输出在两端几乎一致，末端 x 仅约 `7.6e-6 m` 差异。
10. 第一个在线场景首帧的 ego、lane、route、speed-limit、static-object 张量哈希完全一致；neighbor history 只有底层浮点末位差异，首帧预测约 `1e-5` 量级差异。

明确观察到的底层运行时差异：

```text
本机：PyTorch 2.0.0+cu118，但通过本机 LD_LIBRARY_PATH
      实际加载 CUDA 12.4 路径中的 cuDNN 8.9.7

云端：PyTorch 2.0.0+cu118，加载 torch/lib 自带 cuDNN 8.7.0
```

扩散规划器闭环会把每步很小的数值差异继续反馈给下一时刻；本次实测表明该三场景闭环对底层 CUDA 运行时敏感。当前证据能够证明“云端和本机运行时不等价”，但不能仅凭 cuDNN 版本断言唯一根因，因为实际 CUDA/cuBLAS 动态库路径也不同。

因此采取的评测口径是：

```text
open-loop：云端单卡，所有 checkpoint 使用同一运行时，可内部排名；
closed-loop：把 checkpoint 下载到产生历史基线的本机环境，统一重跑；
云端 score=0：保留为环境诊断结果，不作为模型质量结论。
```

### 55.7 本机可比环境下的四 checkpoint closed-loop 排名

从云端下载 `args.json` 和四个 checkpoint。本地 SHA256：

```text
epoch 5   c2f3043a6634a1e43f90836f0daa487ea940445ba0822660d3f3ba0fe61dee11
epoch 10  fe06bf3bf3ce2bc14aa16a7ddb9d9da8f591c86cb0f030577387ea5e63b1005d
epoch 15  d858c35219de8d9173a63352f5a3195e6f7b8e0d1fc1edf08fd0f4e963e1c00d
epoch 20  eb8308739228517973468dc445be843a0a030afc5974bbc293caf14935f6210a
```

四个 checkpoint 都通过 `torch.load` 校验，并统一加载 EMA。

每个 checkpoint 使用相同命令模板：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  <experiment_uid> \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/args.json \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/model_epoch_<N>_trainloss_<LOSS>.pth \
  mini-val-closed-loop-3 \
  hdp
```

四次均为成功 3/3、失败 0。统一排名：

| closed-loop 排名 | epoch | score | 路线进度 | making progress | 无责任碰撞 | 可行驶区域 | TTC | 限速 | 舒适性 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.954635 | 0.855022 | 1.000000 | 1.0 | 1.0 | 1.0 | 0.999765 | 1.0 |
| 2 | 20 | 0.892413 | 0.655770 | 1.000000 | 1.0 | 1.0 | 1.0 | 0.999941 | 1.0 |
| 3 | 15 | 0.867939 | 0.577450 | 1.000000 | 1.0 | 1.0 | 1.0 | 0.999943 | 1.0 |
| 4 | 10 | 0.651297 | 0.671831 | 0.666667 | 1.0 | 1.0 | 1.0 | 0.999926 | 1.0 |

历史同场景参考：

```text
发布版 Diffusion-Planner：0.995829
旧 HDP 10k epoch 10：    0.947107
```

阶段结论：

1. 当前 16,210 场景训练的最佳三场景 checkpoint 是 epoch 5，而不是 open-loop 选出的 epoch 15，也不是最后的 epoch 20。
2. epoch 5 在三场景上比旧 HDP 10k 高约 `0.00753`，但低于发布版 Diffusion-Planner。
3. epoch 5 的碰撞、道路、TTC、方向、限速和舒适性均通过，主要差距仍在路线进度。
4. 从 epoch 5 继续训练后，train/open-loop loss 总体下降，但 closed-loop 分数下降，说明当前监督目标与闭环进度并不完全一致。
5. 三场景只能作为门禁，不能作为最终模型结论；在启动 RL 前，应先对 epoch 5 扩大到固定 mini-val 20 场景，并与旧 HDP 10k 和发布版控制组同配置比较。

关键结果文件：

```text
云端 open-loop：
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation/mini_val_1000_checkpoint_ranking_repeat3.json

本机四 checkpoint closed-loop：
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/checkpoint_closed_loop_mini_val_3.json

本机 epoch 15 详细结果：
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15_mini_val_3_local_runtime.json

云端异常对照：
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation/mini_val_3_diffusion_control_vs_hdp_epoch15.json

本机发布版控制组复现：
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/closed_loop_cloud_diagnosis/diffusion_planner_local_control_summary.json
```

### 55.8 固定 20 场景门禁：三场景优势未能保持

按照上一节的决策，先把三场景排名第一的 epoch 5 扩大到固定
`mini-val-closed-loop-20`。为避免云端底层 CUDA 运行时造成不可比结果，仍在产生历史基线的本机
环境中执行：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  hdp-cloud16210-epoch5-local-runtime-mini-val-20 \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/args.json \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/model_epoch_5_trainloss_1.6996.pth \
  mini-val-closed-loop-20 \
  hdp
```

epoch 5 完成 20/20、失败 0，耗时约 16 分 42 秒。因为 epoch 5 的 20 场景结果低于旧
HDP 10k，为排除“早期 checkpoint 偶然失效、后期 checkpoint 更稳定”的可能，又使用完全相同
配置评测 epoch 20：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  hdp-cloud16210-epoch20-local-runtime-mini-val-20 \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/args.json \
  HDP-nuplan/tmp/cloud_partial_16210_eval/epoch15/model_epoch_20_trainloss_1.0450.pth \
  mini-val-closed-loop-20 \
  hdp
```

epoch 20 完成 20/20、失败 0，耗时约 17 分 17 秒。运行中两处出现 nuPlan 官方 route
extractor 的 `All route_list elements are empty` warning，但对应 simulation 均正常结束并写入指标；
该现象也曾在历史控制组出现，因此不作为模型失败。

统一对比结果：

| 模型 | score | 无责任碰撞 | 可行驶区域 | making progress | 行驶方向 | 路线进度 | TTC | 限速 | 舒适性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 发布版 Diffusion-Planner | 0.881446 | 0.90 | 0.95 | 1.00 | 1.000 | 0.960774 | 0.90 | 0.984866 | 0.95 |
| 旧 HDP 10k epoch 10 | 0.787373 | 0.95 | 1.00 | 0.90 | 0.975 | 0.668771 | 0.90 | 0.999990 | 1.00 |
| 新 HDP 16,210 epoch 20 | 0.674302 | 0.95 | 0.95 | 0.80 | 1.000 | 0.604205 | 0.90 | 0.999991 | 1.00 |
| 新 HDP 16,210 epoch 5 | 0.662552 | 0.90 | 0.95 | 0.85 | 0.925 | 0.590206 | 0.90 | 0.999964 | 1.00 |

相对旧 HDP 10k：

```text
epoch 5  score：-0.124821（约 -15.85%）
epoch 20 score：-0.113072（约 -14.36%）
```

结论：三场景上 epoch 5 的 `0.954635` 没有在 20 场景上保持，属于小样本乐观波动。epoch 20
虽比 epoch 5 高 `0.011750`，但仍低于旧 HDP 10k，主要问题是 `making progress` 和路线进度。
当前 16,210 模型不能替代旧 10k 监督基线，也不应作为下一轮 RL 起点。

结果文件：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/epoch5_mini_val_20_local_runtime.json
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_eval/epoch20_mini_val_20_local_runtime.json
```

### 55.9 训练策略诊断与停止条件

本轮与旧 10k 监督实验并非只改变了样本数量，关键配置同时发生了变化：

| 项目 | 旧 HDP 10k | 新 HDP 16,210 |
|---|---:|---:|
| 全局 batch size | 8 | 32 |
| 每 epoch update 数 | 1,250 | 506 |
| 比较 checkpoint 的累计 update 数 | epoch 10：12,500 | epoch 20：约 10,120 |
| planning_hybrid_loss | 0.01 | 0.1 |
| encoder 冻结 | 前 3 epoch | 0 epoch |
| checkpoint 间隔 | 1 epoch | 5 epoch |
| 数据构成 | mini balanced 10k | partial official 16,210，当前偏 Singapore |

因此“16,210 比 10,000 多”不等于优化更充分：四卡全局 batch 扩大到 32 后，20 epoch 的
参数更新次数反而少于旧 10k 的 epoch 10；同时混合损失权重、encoder 冻结策略和数据分布都发生
变化，无法把性能下降单独归因于某一个因素。

基于当前证据执行以下停止条件：

1. 不启动 RL；RL 应从可靠的监督策略微调，不能用 RL 掩盖监督基线退化。
2. 不继续对 epoch 10/15 做完整 20 场景穷举。三场景排名前两位的 epoch 5/20 均未通过旧
   10k 门禁，而 open-loop 排名又已被证明不能可靠预测 closed-loop 排名。
3. 保留 16,210 checkpoint 和所有评测结果，不删除、不覆盖。
4. 下一轮先做可归因的监督对照：固定数据与 seed，恢复旧基线的 batch 8、
   `planning_hybrid_loss=0.01`、encoder 前 3 epoch 冻结，并每 epoch 保存；至少达到旧 10k 的
   20 场景 `0.787373` 后，才进入 RL。
5. 100k 数据仍应继续准备，但不能把“等待全量数据”当作当前监督配置正确的证据。

这是一次有效的阶段性否定结果：训练链路、checkpoint、open-loop 和 closed-loop 均已跑通，但
当前大 batch/新损失配置没有获得正收益。项目下一步应先恢复可比监督基线，再逐项改变变量。

### 55.10 结果同步与资源状态

将本节日志和两个本机 20 场景汇总文件同步到云端：

```text
/root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/doc_hdp_nuplan/AutoDL全量训练部署操作日志.md
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation/epoch5_mini_val_20_local_runtime.json
/root/autodl-tmp/experiments/hdp_partial_16210_4gpu_e20/evaluation/epoch20_mini_val_20_local_runtime.json
```

两个结果 JSON 的本机/云端 SHA256 一致：

```text
epoch 5： 0e2a272f1a6a5cd6c2e51227bffa22aa315645848b3f7975b5fe0ef7fee2e23b
epoch 20：ba46c830ba88672644dfd718938e640bc0f9ac0b71ff912e291de081124f1727
```

同步后云端 GPU 为 `1 MiB / 0%`，没有活动的训练、仿真、checkpoint 评测或数据预处理任务。

## 56. 16,210 样本旧基线配置复现对照

时间：2026-08-10 16:48（Asia/Shanghai）。

### 56.1 为什么启动该对照

第 55 节证明 16,210 样本的四卡大 batch 实验没有通过旧 10k closed-loop 门禁，但那一轮同时
改变了 batch size、混合损失权重、encoder 冻结和数据分布，无法确定是数据本身还是训练配置
导致退化。

本轮不启动 RL，也不覆盖已有实验；继续使用相同 16,210 manifest 和 seed，只把可控训练配置
恢复为旧 10k 监督基线口径：

```text
全局 batch size：8（单卡）
planning_hybrid_loss：0.01
freeze_encoder_epochs：3
train_epochs：20
save_utd：1
learning_rate：5e-4
warm_up_epoch：2
encoder warm-start：发布版 Diffusion-Planner encoder
diffusion model/supervision：x_start/x_start
seed：3407
```

使用 20 epoch 而不是只运行到 10 epoch，是因为旧 10k 实验的 scheduler 总长度也是 20 epoch；
每 epoch 保存使后续可以按 open-loop 排名和 closed-loop 门禁选择中间 checkpoint。

### 56.2 后台启动命令

```bash
screen -dmS hdp16210_b8_baseline bash -lc '
  cd /root/autodl-tmp/workspace/Diffusion-Planner
  set -o pipefail
  CUDA_VISIBLE_DEVICES=0 \
  /root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
    --nnodes 1 \
    --nproc-per-node 1 \
    --standalone \
    HDP-nuplan/train_predictor.py \
    --name hdp-official-partial-16210-baselinecfg-b8-e20 \
    --save_dir /root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20 \
    --train_set /root/autodl-tmp/processed/nuplan_train_100k_gate \
    --train_set_list /root/autodl-tmp/processed/manifests/nuplan_train_partial_16210/train_manifest_16210.json \
    --normalization_file_path /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/normalization.json \
    --batch_size 8 \
    --train_epochs 20 \
    --save_utd 1 \
    --num_workers 8 \
    --learning_rate 0.0005 \
    --warm_up_epoch 2 \
    --planning_hybrid_loss 0.01 \
    --encoder_pretrained_model_path /root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth \
    --freeze_encoder_epochs 3 \
    --diffusion_model_type x_start \
    --diffusion_supervision_type x_start \
    --use_data_augment true \
    --augment_prob 0.5 \
    --use_ema true \
    --use_wandb false \
    --ddp true \
    --seed 3407 \
    --pin-mem \
    2>&1 | tee /root/autodl-tmp/logs/hdp_partial_16210_baselinecfg_b8_e20.log
  rc=${PIPESTATUS[0]}
  printf "%s\\n" "$rc" | tee /root/autodl-tmp/logs/hdp_partial_16210_baselinecfg_b8_e20.exit
  exit "$rc"
'
```

运行标识：

```text
screen：hdp16210_b8_baseline
日志：/root/autodl-tmp/logs/hdp_partial_16210_baselinecfg_b8_e20.log
退出码：/root/autodl-tmp/logs/hdp_partial_16210_baselinecfg_b8_e20.exit
输出根目录：/root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20
```

### 56.3 首批验收

启动后确认：

```text
Dataset Prepared：16,210
每 epoch DataLoader batch：2,026
encoder warm-start：151/151 tensor，1,799,040 parameters
decoder_loaded：0
模型参数量：5,092,996
Epoch 1 encoder trainable：False
初始 GPU：约 1,037 MiB，利用率采样 15%，功耗约 96 W
退出码文件：尚未生成，表示任务仍在运行
```

首批 loss 为有限值并快速下降，没有出现 OOM、NaN、NCCL 或 warm-start 缺失。DataLoader 的
8 个 worker 在进程列表中会显示与主训练进程相似的命令行，这是多进程数据加载的正常现象，
不是错误启动了多份训练。

### 56.4 首个 epoch 与 checkpoint 校验

第一次只读监控误用了云端 base shell 中不存在的 `python`，返回：

```text
bash: python: command not found
```

该命令与训练进程完全独立，没有停止或修改训练。后续监控统一使用：

```text
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python
```

监控期间 SSH 转发端口还短暂返回过一次 `Connection refused`，约数十秒后恢复；screen 内的训练
持续运行，没有重新启动任务。

epoch 1 结果：

```text
train loss：0.6138715744
checkpoint：model_epoch_1_trainloss_0.6139.pth
文件大小：67,417,699 bytes
完成后进入：Epoch 2/20
GPU 采样：约 1,041 MiB、14%～18%、约 93～95 W
```

使用 CPU `torch.load` 独立校验 epoch 1 checkpoint：

```text
epoch：1
loss：0.6138715744
model tensors：260
EMA tensors：260
optimizer state entries：109
checkpoint keys：model、ema_state_dict、optimizer、schedule、epoch、loss、wandb_id
```

同时重新读取自动保存的 `args.json`，确认 batch 8、20 epoch、每轮保存、学习率 `5e-4`、
`planning_hybrid_loss=0.01`、encoder 冻结 3 epoch、seed 3407 和 16,210 manifest 均正确。

epoch 1 checkpoint 比 encoder 已解冻实验小，是因为冻结期间 AdamW 只为实际产生梯度的参数创建
optimizer state；model 和 EMA 本身仍各自包含完整的 260 个 tensor，不是模型权重缺失。

### 56.5 训练完成状态

再次连接云端后确认本轮训练完整结束：

```text
完成 epoch：20/20
checkpoint：20 个，每 epoch 1 个
退出码：0
Traceback：0
NaN：0
CUDA OOM：0
ChildFailedError：0
NCCL error：0
训练结束后 GPU：1 MiB、0%
```

完整 train loss：

```text
01  0.6139
02  0.3767
03  0.2591
04  0.3281
05  0.2969
06  0.2727
07  0.2644
08  0.2616
09  0.2468
10  0.2397
11  0.2334
12  0.2388
13  0.2197
14  0.2275
15  0.2115
16  0.2107
17  0.2130
18  0.2059
19  0.2067
20  0.1928
```

epoch 4 的 loss 从 `0.2591` 回升到 `0.3281`，与 encoder 在前三轮冻结、第四轮开始解冻的边界
一致；此后总体继续下降。该现象符合配置预期，不是训练异常。

最终 checkpoint 校验：

```text
epoch：20
精确 loss：0.1928371340
model/EMA/optimizer 等完整状态均存在
optimizer state entries：260
epoch 20 SHA256：8ac7b5247a11b96c427704b29b018db3e14642645284eedd09e42ac95ad6c126
latest.pth SHA256：8ac7b5247a11b96c427704b29b018db3e14642645284eedd09e42ac95ad6c126
```

`latest.pth` 与 epoch 20 完全相同。训练成功只证明优化链路和 loss 收敛，尚不能证明闭环规划优于
旧 10k；下一步必须先做独立 val1k checkpoint 排名，再做固定 20 场景闭环门禁，暂不启动 RL。

## 57. 旧基线配置复现模型的 val1k 排名与 closed-loop 门禁

时间：2026-08-10。

目标：判断第 56 节训练得到的模型能否替代旧 HDP 10k 监督基线，并决定是否可以启动 RL。

### 57.1 评测前数据审计

云端检查：

```text
训练 manifest：16,210 条，16,210 个唯一文件名
验证 manifest：1,000 条，1,000 个唯一文件名
训练/验证 basename 交集：0
监督 checkpoint：20 个
```

因此 val1k 是独立验证集，不是从本轮训练 manifest 中重复读取样本。

### 57.2 云端 open-loop 全 checkpoint 排名

先对 20 个 checkpoint 各评测 1 次：

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/evaluate_checkpoints.py \
  --args_file /root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20/training_log/hdp-official-partial-16210-baselinecfg-b8-e20/2026-08-10-16:48:09/args.json \
  --checkpoint_dir /root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20/training_log/hdp-official-partial-16210-baselinecfg-b8-e20/2026-08-10-16:48:09 \
  --pattern 'model_epoch_*.pth' \
  --data_dir /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 32 \
  --num_workers 8 \
  --repeats 1 \
  --seed 3407 \
  --device cuda \
  --output /root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20/evaluation/mini_val_1000_checkpoint_ranking_repeat1.json
```

20 个 checkpoint 全部完成，退出码 0。单次排名前五为 epoch 20、17、16、18、3。由于评测脚本
执行较快，没有只复测前三名，而是对全部 20 个 checkpoint 使用相同参数执行 3 次复测，输出：

```text
/root/autodl-tmp/experiments/hdp_partial_16210_baselinecfg_b8_e20/evaluation/
mini_val_1000_checkpoint_ranking_repeat3.json
SHA256：a224612bda8849609ef0c29d42fef7cd8526db6681b11a37e36c50401b1611f1
```

repeat3 前十名：

| 排名 | epoch | mean total loss | 三次标准差 |
|---:|---:|---:|---:|
| 1 | 20 | 0.086157832 | 0.002397597 |
| 2 | 18 | 0.086316069 | 0.004898408 |
| 3 | 16 | 0.086788811 | 0.003484305 |
| 4 | 3 | 0.087275190 | 0.001052389 |
| 5 | 17 | 0.087445140 | 0.001893826 |
| 6 | 19 | 0.089570170 | 0.003269387 |
| 7 | 13 | 0.092728331 | 0.002456694 |
| 8 | 15 | 0.092852543 | 0.003305626 |
| 9 | 11 | 0.093935013 | 0.002552257 |
| 10 | 14 | 0.094047483 | 0.004238029 |

结果解析时第一次误用不存在的 `ranking` 字段并对字典执行切片，触发只读
`TypeError: unhashable type: 'slice'`；实际字段名为 `ranked`。该错误发生在结果生成之后，没有修改
JSON，也没有重新运行评测。

### 57.3 下载 closed-loop 候选

选择三个具有互补意义的 checkpoint：

```text
epoch 20：repeat3 open-loop 第 1，最终训练轮
epoch 18：repeat3 open-loop 第 2，后期候选
epoch 3： repeat3 open-loop 第 4，encoder 冻结阶段最后一轮
```

不只测试相邻的后期 checkpoint，是为了覆盖“冻结 encoder 的早期策略可能更适合闭环”的可能性。
使用支持断点文件保留的 rsync 下载到：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/
cloud_partial_16210_baselinecfg_eval/checkpoints/
```

本地/云端 SHA256 完全一致：

```text
args.json：57dc5ef3ded6400f630f8ee85185d5b957a3cc9070867edf753a91d706a4f3e1
epoch 3： 91be55860199b8a1954ae78f8f0e41b854124c3223ffee578ab87fafd7153ada
epoch 18：ad1ce58e541a98a2aad0f941caf8e44b27eb5c2baa6e586c36b7019d0b55506e
epoch 20：8ac7b5247a11b96c427704b29b018db3e14642645284eedd09e42ac95ad6c126
```

### 57.4 本地 closed-loop 启动中的两个无效尝试

第一次计划用本机 `screen` 托管仿真，但本机没有安装 screen：

```text
/bin/bash: screen: command not found
```

因此该命令没有创建仿真进程或结果。随后改由当前执行会话托管，并把输出重定向到独立日志。

第一次实际调用 epoch 20 时传入相对 `args.json` 路径。`run_mini_closed_loop.sh` 内部切换到
nuPlan-devkit 工作目录后，相对路径失效，立即报错：

```text
FileNotFoundError: ... cloud_partial_16210_baselinecfg_eval/checkpoints/args.json
```

该 run 没有开始任何场景。失败日志被保留；后续统一改用 checkpoint 和 args 的绝对路径，并使用
新的 `-abs` experiment UID，避免覆盖失败证据。

有效命令模板：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_baselinecfg_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  <experiment_uid>-abs \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_baselinecfg_eval/checkpoints/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_baselinecfg_eval/checkpoints/model_epoch_<N>_trainloss_<LOSS>.pth \
  mini-val-closed-loop-20 \
  hdp
```

### 57.5 三个候选的固定 20 场景结果

三个模型均在历史基线使用的本机运行时、相同 20 token、相同 challenge 和相同指标配置下执行：

```text
epoch 20：20 成功、0 失败，仿真 00:15:44
epoch 3： 20 成功、0 失败，仿真 00:16:05
epoch 18：20 成功、0 失败，仿真 00:16:01
```

每个 run 均出现 3 次 nuPlan route extractor 的
`All route_list elements are empty` warning；历史基线在相同 token 上也存在该数据现象，所有场景
仍正常完成并写入指标，因此不计为模型失败。

统一结果：

| 模型 | score | 无责任碰撞 | 可行驶区域 | making progress | 方向 | 路线进度 | TTC | 限速 | 舒适性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 发布版 Diffusion-Planner | 0.881446 | 0.90 | 0.95 | 1.00 | 1.000 | 0.960774 | 0.90 | 0.984866 | 0.95 |
| 旧 HDP 10k epoch 10 | 0.787373 | 0.95 | 1.00 | 0.90 | 0.975 | 0.668771 | 0.90 | 0.999990 | 1.00 |
| 新 baselinecfg epoch 3 | 0.675841 | 0.95 | 1.00 | 0.80 | 0.975 | 0.534877 | 0.90 | 0.999988 | 1.00 |
| 新 baselinecfg epoch 20 | 0.585445 | 1.00 | 0.95 | 0.70 | 1.000 | 0.491947 | 1.00 | 0.999991 | 1.00 |
| 新 baselinecfg epoch 18 | 0.513077 | 0.95 | 0.95 | 0.65 | 1.000 | 0.519112 | 0.95 | 0.999991 | 1.00 |

最终对比 JSON：

```text
/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/cloud_partial_16210_baselinecfg_eval/
final_candidates_vs_baselines_mini_val_20.json
SHA256：073a3389fa460bbc8491925a6fa078e44bf7f10995f3e9440a058e69f7a2cce5
```

### 57.6 决策

三个代表性候选均未达到旧 10k 的 `0.787373` 门禁，本轮最优 epoch 3 仍低 `0.111532`，约
`14.17%`。后期模型的碰撞、TTC 和舒适性很好，但 making-progress 与路线进度持续下降，表现为
更安全却更保守，并非数值发散。

恢复旧 10k 的 batch 8、`planning_hybrid_loss=0.01` 和 encoder 前 3 epoch 冻结后仍未恢复旧
10k 性能，说明四卡大 batch 并不是唯一原因。当前最强证据指向 16,210 partial 数据的城市/场景
分布偏差或样本质量差异；第 51 节已记录该 partial 集偏向 Singapore，而固定 mini-val 20 并非
同分布的 Singapore 主导集合。

执行以下停止条件：

1. 不从本轮 epoch 3/18/20 启动 RL。
2. 不继续穷举其余相邻 checkpoint 的 20 场景评测；open-loop 前两名和结构差异最大的冻结边界
   checkpoint 均未通过门禁，继续穷举的成本高且缺乏依据。
3. 旧 HDP 10k epoch 10 继续作为当前监督/RL 起点。
4. 下一步先审计 16,210 与旧 balanced 10k 的城市、日志、scenario type、轨迹速度/位移和
   route 有效率分布；根据审计结果构造去重、跨城市均衡的数据集，再重新监督训练。
5. 当前三个 checkpoint、open-loop 排名和 closed-loop 结果全部保留，作为项目中“训练 loss
   下降不保证闭环收益、数据分布需要门禁验证”的完整负结果证据。

### 57.7 结果保存与同步状态

本地最终对比 JSON 和本节日志已成功写盘。尝试同步到云端时，AutoDL 连接先后返回：

```text
Could not resolve hostname connect.cqa1.seetacloud.com
connect to host connect.cqa1.seetacloud.com port 11156: Connection refused
```

多次重试后端口仍拒绝连接，因此没有宣称同步成功。该连接问题发生在全部训练和评测结束之后，
不影响本地 closed-loop 结果。云端已经保存 repeat1/repeat3 open-loop JSON 和全部监督 checkpoint；
本地保存三个候选 checkpoint、三个有效 closed-loop run、日志以及最终统一对比 JSON。待实例 SSH
恢复后，只需同步本节 Markdown 和 `final_candidates_vs_baselines_mini_val_20.json`，不需要重新训练
或重新评测。

## 58. 为什么 16,210 模型低于旧 balanced 10k

时间：2026-08-10。

本节只做只读数据统计和因果边界分析，没有修改缓存、模型或训练配置。

### 58.1 最强证据：训练集与评测集的城市分布错位

直接逐 NPZ 读取 `map_name` 后，旧 balanced 10k 的城市构成为：

| 城市 | 数量 | 占比 |
|---|---:|---:|
| Las Vegas | 7,728 | 77.28% |
| Singapore | 908 | 9.08% |
| Boston | 682 | 6.82% |
| Pittsburgh | 682 | 6.82% |

独立 val1k 为：

| 城市 | 数量 | 占比 |
|---|---:|---:|
| Las Vegas | 700 | 70% |
| Singapore | 100 | 10% |
| Boston | 100 | 10% |
| Pittsburgh | 100 | 10% |

固定 closed-loop 20 的 17 个 val1k token 可直接由 NPZ 文件名前缀识别为 Las Vegas 11、
Boston 3、Singapore 3；三个固定 token 分别与同日志的已识别场景对应，两条属于 Las Vegas、
一条属于 Singapore。最终为：

```text
Las Vegas：13/20 = 65%
Boston：3/20 = 15%
Singapore：4/20 = 20%
Pittsburgh：0/20
```

旧 10k 因此与 val1k 和 closed-loop 20 的目标域高度一致。

16,210 partial 则是按“当时已经下载完成、可以组成完整 shard 的城市 ZIP”逐步拼成，并不是完整
100k 的随机代表性子集。已知纯城市部分为：

```text
Singapore：本地新增 9,450 + 已有 shard_00175 350 = 至少 9,800（60.46%）
Pittsburgh：本地新增 1,400 + 已有 shard_00152 400 = 至少 1,800（11.10%）
Boston：新增 2,660 + 已有 shard_00197 350 = 至少 3,010（18.57%）
其余四个早期 shard 合计：1,600
```

即使把未明确的 1,600 条全部算成 Las Vegas，16,210 中 Las Vegas 也最多只有 `9.87%`；实际值
只会更低或等于该上界。与旧 10k 的 `77.28%`、val1k 的 `70%` 和闭环集合的 `65%` 相差巨大。

因此“16,210 大于 10,000”并不表示对当前评测更有效：新增的主要是 Singapore 域，而不是目标
评测占主导的 Las Vegas 域。

### 58.2 日志采样密度和行为分布也不同

旧 10k 使用 44 个官方 mini-train 日志和 `balanced_logs`：

```text
每日志 227～229 场景
```

16,210 覆盖 2,132 个实际贡献日志，平均每日志只有约 `7.60` 条。可在本地完整统计的两个主要
新增城市为：

```text
Singapore：9,450 场景 / 1,282 日志，每日志 7～13 条
Pittsburgh：1,400 场景 / 175 日志，每日志 7～12 条
```

覆盖日志更广本身不是错误，但在 partial 数据尚未跨城市补齐时，它会变成“每个日志很稀、城市
又严重偏置”的训练集，而不是旧 10k 那种与 mini 评测域一致的密集均衡样本。

轨迹和场景类型统计进一步显示行为分布不同：

| 统计 | 旧 train10k | val1k | Singapore 9,450 | Pittsburgh 1,400 |
|---|---:|---:|---:|---:|
| `stationary` 占比 | 23.17% | 21.20% | 6.38% | 22.79% |
| `unknown` 占比 | 25.96% | 22.20% | 40.93% | 30.57% |
| 未来位移中位数/m | 22.72 | 12.19 | 37.50 | 33.65 |
| 未来位移小于 1m | 27.75% | 43.60% | 14.38% | 24.86% |
| route 非空比例 | 97.69% | 90.00% | 95.83% | 98.86% |

这说明新增主力 Singapore 数据比 mini train/val 更少静止、位移更大、`unknown` 更多。张量 shape、
有限值、checksum 和去重校验全部通过，只能证明数据文件有效，不能证明其统计分布适合当前评测。

### 58.3 为什么 open-loop 更低，closed-loop 却更差

新 baselinecfg 的 val1k repeat3 最优 loss 为 epoch 20 的 `0.086158`，低于旧 10k epoch 10 的
`0.119260`；但 closed-loop 分别为 `0.585445` 和 `0.787373`。这不是矛盾：

1. open-loop 每个样本都从专家真实状态出发，只衡量一次预测误差；
2. closed-loop 会把模型上一帧的输出反馈为下一帧输入，小偏差会累积；
3. val1k 中 `43.60%` 的样本未来位移小于 1m，而固定 20 场景构造时显式优先非静态、非
   `unknown` 场景；
4. 因此偏保守的模型可能在静态占比较高的平均 open-loop loss 上很好，却在运动场景闭环中停止或
   进展不足。

实际指标与该解释一致：新 epoch 20 的碰撞和 TTC 都是 `1.0`，但 making-progress 只有 `0.70`、
路线进度只有 `0.491947`。它不是失控，而是安全但过于保守。

“城市域错配导致模型在闭环 OOD 状态下趋向保守”是基于上述统计和指标的最合理推断；若要证明
严格因果，仍需构造城市均衡训练集做控制变量实验。

### 58.4 已排除或降级的原因

以下因素不能再作为主要解释：

1. **训练没有收敛**：20 epoch 正常结束，loss 从 `0.6139` 降到 `0.1928`，无 NaN/OOM。
2. **四卡大 batch 是唯一原因**：已经用相同 16,210 数据重跑 batch 8、hybrid loss `0.01`、
   encoder 冻结 3 epoch，最佳闭环仍只有 `0.675841`。
3. **checkpoint 损坏或加载错误**：model/EMA 完整，SHA256 一致，三次 closed-loop 均 20/20
   成功。
4. **训练/val 文件重复**：16,210 与 val1k basename 交集为 0。
5. **单纯 epoch 选择错误**：已测试 open-loop 第一、第二以及冻结边界 checkpoint，均未通过门禁。

### 58.5 当前结论

按证据强弱排序：

```text
主因：16,210 是城市严重偏置的 partial 集，与 Las-Vegas 主导的 mini 评测域错配。
次因：每日志采样很稀，场景类型和运动幅度与旧 10k/val1k 不同。
放大器：open-loop 平均 loss 对静态样本友好，不能约束运动场景中的闭环 progress。
已排除为唯一主因：四卡 batch、loss 权重、encoder 冻结、训练失败、checkpoint 损坏。
```

所以问题不是“更多数据反而必然更差”，而是“一个不完整且偏 Singapore 的 16.21% 子集，在当前
Las-Vegas 主导评测上不如与评测域匹配的 balanced mini 10k”。完整 100k 补齐或重新构造跨城市
均衡子集后，结论可能改变，必须重新训练并用同一 closed-loop 门禁验证。

## 59. 方案一：构建与独立 mini-val 城市比例一致的 11,040 训练集

时间：2026-08-10。

### 59.1 目的、控制变量和目标数量

本轮选择“方案一”：保留旧 balanced 10k 的全部样本，只补充缺少的城市，使训练集城市比例与
独立 mini-val 的 `70:10:10:10` 一致。目的不是宣称得到适用于所有 NuPlan 域的通用最优分布，
而是做一个可解释的控制变量实验，验证第 58 节发现的城市域错配是否是 16,210 模型退化的主因。

旧 10k 中 Las Vegas 已有 7,728 条并保持不动，因此总量由下式唯一确定：

```text
7,728 / 70% = 11,040
```

每个非 Las Vegas 城市目标为 `11,040 × 10% = 1,104`。相对旧 10k 的补充量为：

| 城市 | 旧 10k | 目标 | 新增 |
|---|---:|---:|---:|
| Las Vegas | 7,728 | 7,728 | 0 |
| Singapore | 908 | 1,104 | 196 |
| Boston | 682 | 1,104 | 422 |
| Pittsburgh | 682 | 1,104 | 422 |
| 合计 | 10,000 | 11,040 | 1,040 |

训练数据只允许来自官方 train。独立 mini-val 的 1,000 个 NPZ、10 个日志及其 token 全部作为硬
排除项，不能用于补足训练数量。

### 59.2 本地数据盘点和 Boston 按需下载

本地已有的官方 train 候选为：

```text
Singapore：9,450 NPZ
Pittsburgh：1,400 NPZ
Boston：0 个独立于旧 10k 的现成候选
```

Singapore 和 Pittsburgh 足够；Boston 还差 422 个。为避免下载完整的约 35 GB Boston ZIP，
使用 HTTP Range 只读取 ZIP 中目标 DB。安装项目固定依赖：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pip install -r HDP-nuplan/requirements_data_download.txt
```

使用官方地址：

```text
https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/
nuplan-v1.1_train_boston.zip
```

先对 `shard_00245` 做单分片门禁：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/run_preprocessing_range.py \
  --plan /home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k/preprocessing_plan.json \
  --archive_index /home/yanjun/NewDisk/nuplan/indexes/archive_index_train_boston_remote_targetmatch.json \
  --raw_root /home/yanjun/NewDisk/nuplan/raw_boston_targetmatch \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --output_root /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_source \
  --state_path /home/yanjun/NewDisk/logs/nuplan_targetmatch_boston_state.json \
  --shard_indices 245 256 \
  --worker_index 0 --worker_count 1 \
  --max_shards 1 \
  --checksum_mode files --cleanup_raw \
  --download_backend remotezip \
  --archive boston=https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip \
  --only_archive boston \
  --connect_timeout_seconds 15 \
  --read_timeout_seconds 120 \
  --max_member_retries 5 \
  --retry_delay_seconds 5 \
  --min_free_gib 30
```

结果：

```text
50/50 DB 下载成功
下载 1,761,095,680 bytes，耗时 265.83 s
成员重试 2 次，均恢复成功
生成 350/350 NPZ，failed=0
48 个有效日志
逐文件 checksum 和缓存校验通过
原始 DB 在校验通过后自动清理
```

继续 `shard_00256` 时，本地代理/网络连续出现 `SSLZeroReturnError`、TLS handshake timeout 和
`ProxyError`。第一次 5 次重试、第二次 10 次重试均失败。失败状态和 traceback 完整保留在：

```text
/home/yanjun/NewDisk/logs/nuplan_targetmatch_boston_state.json
/home/yanjun/NewDisk/logs/nuplan_targetmatch_boston_preprocess_resume.log
```

失败前已有 12 个通过 SQLite 检查的完整 DB 写盘。为避免继续依赖异常外网，直接从这些 train DB
生成 100 个补充候选：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/raw_boston_targetmatch/shard_00256/trainval \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source/cache \
  --log_names_json /home/yanjun/NewDisk/nuplan/plans/nuplan_train_100k/shard_00256_logs.json \
  --total_scenarios 100 \
  --seed 3663 \
  --sampling_strategy balanced_logs \
  --allow_empty_logs true \
  --shard_id boston_shard_00256_partial12 \
  --output_list_path /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source/manifest.json \
  --sampling_report_path /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source/sampling_report.json \
  --processing_report_path /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source/processing_report.json \
  --checksum_path /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source/checksums.json \
  --checksum_mode files \
  --skip_existing true \
  --fail_on_error true
```

NuPlan valid-scene 查询在 12 个 DB 中识别出 10 个有效日志，另外 2 个 DB 没有有效 scenario；最终
每个有效日志抽 10 条，得到 100/100 NPZ，`failed=0`。Boston 因而有 `350 + 100 = 450` 个
候选，足够选出所需 422 个。

### 59.3 可复现数据集构建脚本

新增：

```text
HDP-nuplan/scripts/build_city_target_dataset.py
HDP-nuplan/tests/test_build_city_target_dataset.py
```

脚本执行以下门禁后才原子落盘：

1. 读取 base、候选源和 validation NPZ 的 `map_name/log_name/scenario_type/token`；
2. 排除与 base/validation 重复的文件名和 token，并禁止使用 validation 日志；
3. 优先从 base 尚未覆盖的新 train 日志中按日志轮询抽样，固定 seed；
4. 检查最终文件名和 token 均唯一，城市数量精确；
5. 在同一文件系统内使用 hardlink，不重复占用 NPZ 数据块；
6. 输出 manifest、sampling report、selection report 和构建哈希。

同时修复 `audit_dataset_splits.py`：旧报告若没有 `log_names`，从 `selected_per_log` 的键恢复日志
集合。修复前 mini-val 被错误显示为 `val_log_count=0`；修复后正确显示 10。新增回归测试防止该
显示错误再次出现。

构建命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/build_city_target_dataset.py \
  --base_cache HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache \
  --base_manifest HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json \
  --source sg-one-north /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_singapore_local 196 \
  --source us-ma-boston /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_source 350 \
  --source us-ma-boston /home/yanjun/NewDisk/processed/nuplan_targetmatch_boston_partial12_source 72 \
  --source us-pa-pittsburgh-hazelwood /home/yanjun/NewDisk/processed/nuplan_train_100k_gate_pittsburgh_local 422 \
  --output_dir HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1 \
  --seed 3407 \
  --val_cache HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --val_manifest HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --val_report HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/sampling_report.json
```

新增样本的日志覆盖：

```text
Singapore 196 条来自 196 个新日志
Boston 前 350 条来自 48 个新日志
Boston 后 72 条来自 10 个新日志
Pittsburgh 422 条来自 175 个新日志
最终训练集共 473 个日志
```

### 59.4 严格校验结果

逐文件校验命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/validate_processed_cache.py \
  --cache_dir HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache \
  --manifest HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/diffusion_planner_training.json \
  --sampling_report HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/sampling_report.json \
  --expected_count 11040 \
  --expected_log_count 473 \
  --output HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache_validation_report.json
```

结果：

```text
manifest_count=11040
unique_manifest_count=11040
npz_count=11040
log_count=473
total_bytes=1,738,624,268
所有必需键、shape、有限数值和 scalar 元数据检查通过
```

防泄漏审计：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/scripts/audit_dataset_splits.py \
  --train_manifest HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/diffusion_planner_training.json \
  --train_report HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/sampling_report.json \
  --val_manifest HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --val_report HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/sampling_report.json \
  --output HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/train_val_split_audit.json
```

结果：

```text
train_log_count=473
val_log_count=10
overlapping_log_count=0
overlapping_npz_count=0
```

构建阶段还直接比较了 NPZ 元数据 token，token 交集同样为 0。最终城市数量精确为：

```text
Las Vegas  7,728（70%）
Singapore  1,104（10%）
Boston     1,104（10%）
Pittsburgh 1,104（10%）
```

DataLoader 冒烟读取成功：`dataset_len=11040`，batch size 4 返回 11 个张量，shape 与训练入口
一致。hardlink 抽查为相同 inode、链接数 2，说明本地没有复制第二份 NPZ 数据块。

完整测试：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q HDP-nuplan/tests
```

结果：`61 passed, 15 warnings in 4.81s`。warnings 均来自已安装依赖的弃用提示，不是测试失败。

关键产物 SHA256：

```text
808d3b170d7067785e2120d7f4cffd81b427bdbae6f7fc62c9a634e5206e8bac  diffusion_planner_training.json
d64d841bee3e71ca0224e3858c273dfd9e3f8bb7c7f1ad918c7e71bb3b84ef3d  sampling_report.json
e8c507462b9e2f88885037a3fcb59faa0facece546800370710ae6786c2bbd90  selection_report.json
98ebf21504af950f24713ab752da670b3615c2b17e5519e0af2070b0a390fe5f  cache_validation_report.json
883df3d702b7c20264d116b4dd187998e24669833c2b6f1adbd5dce2a9a94105  train_val_split_audit.json
```

### 59.5 当前停止点和下一步

本地数据构建和训练前门禁已完成。下一步应把该 11,040 数据集同步到 AutoDL，然后严格复用旧
10k 的训练设置（batch size 8、`planning_hybrid_loss=0.01`、encoder 冻结前 3 epoch、seed
3407、每 epoch 保存）。这里不是“整个模型随机初始化”：与旧 10k 一样，从发布版
Diffusion-Planner 只迁移兼容 encoder，HDP decoder 保持新初始化。之后先在同一 val1k 做
checkpoint 排名，再在完全相同的固定 20 场景上与旧 10k epoch 10 比较；只有监督模型通过门禁，
才允许启动 RL。启动时的设计意图是把城市比例作为主要数据变量；但第 61 节事后审计确认，旧 10k
与新 11,040 还存在训练代码版本、优化步数和硬件差异，因此本轮不是严格单变量实验，不能把结果
完全归因于城市比例。

### 59.6 云端同步与二次校验

AutoDL SSH 恢复后确认：

```text
GPU：NVIDIA GeForce RTX 4090，24,092 MiB free
数据盘：112 GB free
Python：3.9.25
```

最初使用完整 rsync 时，11,040 个小文件产生较大协议开销；传到约 47 MB 后主动中止。随后尝试
完整 tar 流，发现会重复传输云端已有的旧 10k，又主动中止。两个未完成目标都只是本轮生成的
重复副本，删除后不可恢复，但本地源、云端旧 10k 和所有实验均未删除或覆盖。

最终方案利用“新 11,040 = 旧 10,000 + 新增 1,040”：

1. 校验云端旧缓存精确包含 10,000 个 NPZ；
2. 用 `cp -al` 在新目标缓存建立 10,000 个 hardlink；
3. 由两个 manifest 做集合差，只传输 1,040 个新增文件；
4. 单独同步 manifest、报告、脚本、测试和本操作日志。

新增文件传输结果：

```text
1,040/1,040 files
163,772,860 bytes
耗时约 3 分 23 秒
```

云端目标：

```text
/root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/
mini_train_targetmatch_11040_seed3407_v1
```

不能只依赖文件数；传输结束后在云端重新执行第 59.4 节的逐文件校验和 split 审计。结果：

```text
NPZ_COUNT=11040
HARDLINKED_BASE=10000
cache validation=passed
train_log_count=473
val_log_count=10
overlapping_log_count=0
overlapping_npz_count=0
```

云端三个输入元数据哈希与本地完全一致：

```text
808d3b170d7067785e2120d7f4cffd81b427bdbae6f7fc62c9a634e5206e8bac  diffusion_planner_training.json
d64d841bee3e71ca0224e3858c273dfd9e3f8bb7c7f1ad918c7e71bb3b84ef3d  sampling_report.json
e8c507462b9e2f88885037a3fcb59faa0facece546800370710ae6786c2bbd90  selection_report.json
```

云端重新生成的 `cache_validation_report.json` 哈希与本地不同，是因为报告内记录了不同机器的
绝对 `cache_dir/manifest_path/sampling_report_path`；其校验状态和数据统计一致，不属于数据差异。

云端目标测试：`10 passed, 14 warnings in 2.97s`。

### 59.7 正式监督训练启动

启动前确认没有其他 `train_predictor.py` 训练进程、没有 screen session，且新实验目录不存在。
使用单卡启动，`num_workers=4` 也与旧 10k 的 `args.json` 一致：

```bash
screen -dmS hdp11040_b8 bash -lc '
  cd /root/autodl-tmp/workspace/Diffusion-Planner
  set -o pipefail
  CUDA_VISIBLE_DEVICES=0 \
  /root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run \
    --nnodes 1 \
    --nproc-per-node 1 \
    --standalone \
    HDP-nuplan/train_predictor.py \
    --name hdp-targetmatch-11040-b8-e20 \
    --save_dir /root/autodl-tmp/experiments/hdp_targetmatch_11040_b8_e20 \
    --train_set /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/cache \
    --train_set_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_train_targetmatch_11040_seed3407_v1/diffusion_planner_training.json \
    --normalization_file_path /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/normalization.json \
    --batch_size 8 \
    --train_epochs 20 \
    --save_utd 1 \
    --num_workers 4 \
    --learning_rate 0.0005 \
    --warm_up_epoch 2 \
    --planning_hybrid_loss 0.01 \
    --encoder_pretrained_model_path /root/autodl-tmp/workspace/Diffusion-Planner/checkpoints/model.pth \
    --freeze_encoder_epochs 3 \
    --diffusion_model_type x_start \
    --diffusion_supervision_type x_start \
    --use_data_augment true \
    --augment_prob 0.5 \
    --use_ema true \
    --use_wandb false \
    --ddp true \
    --seed 3407 \
    --pin-mem \
    2>&1 | tee /root/autodl-tmp/logs/hdp_targetmatch_11040_b8_e20.log
  rc=${PIPESTATUS[0]}
  printf "%s\\n" "$rc" | tee /root/autodl-tmp/logs/hdp_targetmatch_11040_b8_e20.exit
  exit "$rc"
'
```

运行标识：

```text
screen：hdp11040_b8
日志：/root/autodl-tmp/logs/hdp_targetmatch_11040_b8_e20.log
退出码：/root/autodl-tmp/logs/hdp_targetmatch_11040_b8_e20.exit
输出：/root/autodl-tmp/experiments/hdp_targetmatch_11040_b8_e20
```

启动门禁：

```text
Dataset Prepared：11,040
每 epoch：1,380 batch
encoder warm-start：151/151 tensor，1,799,040 parameters
decoder_loaded：0
模型参数量：5,092,996
Epoch 1 encoder trainable：False
GPU 显存：约 923 MiB
退出码文件：尚未生成，训练正在运行
```

首批 checkpoint：

```text
epoch 1 train loss：0.5724
epoch 2 train loss：0.3220
epoch 3 train loss：0.1848
```

冻结控制符合预期：epoch 1～3 均输出 `Encoder trainable: False`；epoch 4 自动切换为 `True`，
显存由约 927 MiB 增至约 1,803 MiB，说明 encoder 梯度和优化器状态开始参与训练。前三轮无
NaN、OOM 或进程退出。

### 59.8 训练后自动 open-loop 排名 watcher

为了避免监督训练完成后 GPU 空闲，另建 `hdp11040_eval_watch` screen。它每 30 秒只检查训练
退出码，不在训练期间占用 GPU；仅当监督退出码为 0 时，自动定位本轮唯一 `args.json`，对全部
`model_epoch_*.pth` 在独立 val1k 上做 3 次排名：

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/evaluate_checkpoints.py \
  --args_file <本轮 run_dir>/args.json \
  --checkpoint_dir <本轮 run_dir> \
  --pattern 'model_epoch_*.pth' \
  --data_dir /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 32 \
  --num_workers 8 \
  --repeats 3 \
  --seed 3407 \
  --device cuda \
  --output /root/autodl-tmp/experiments/hdp_targetmatch_11040_b8_e20/evaluation/mini_val_1000_checkpoint_ranking_repeat3.json
```

watcher 产物：

```text
screen：hdp11040_eval_watch
日志：/root/autodl-tmp/logs/hdp_targetmatch_11040_val1k_repeat3.log
退出码：/root/autodl-tmp/logs/hdp_targetmatch_11040_val1k_repeat3.exit
结果：/root/autodl-tmp/experiments/hdp_targetmatch_11040_b8_e20/evaluation/
      mini_val_1000_checkpoint_ranking_repeat3.json
```

当前不能提前写入排名或闭环结论：监督训练和 watcher 都仍在运行。排名完成后必须先核对全部 20
个 checkpoint、三次方差和 EMA 加载，再选择有互补意义的候选做固定 20 场景 closed-loop；不会
未经审阅直接启动 RL。

## 60. target-matched 11,040 训练完成与监督门禁结论

时间：2026-08-10。

目标：完成第 59 节训练后的同机 open-loop 公平复测和本机固定 20 场景 closed-loop 门禁，判断
target-matched 11,040 模型能否替代旧 HDP 10k epoch 10，并决定是否允许进入 RL。

### 60.1 监督训练和自动排名完成

云端监督训练完成 20/20 epoch，训练退出码为 0；自动 val1k 排名也完成，退出码为 0。训练过程
未出现 NaN、OOM 或 DDP 异常。后四个 checkpoint 为：

```text
epoch 17 train loss：0.1334
epoch 18 train loss：0.1452
epoch 19 train loss：0.1286
epoch 20 train loss：0.1309
```

对全部 20 个 checkpoint 使用相同 `mini_val_balanced_1000_seed3407_v1`、3 次重复和 seed 3407
评测。前三名为：

| 排名 | epoch | train loss | mean val loss | 三次总体标准差 |
|---:|---:|---:|---:|---:|
| 1 | 10 | 0.158698 | 0.097193610 | 0.002532390 |
| 2 | 20 | 0.130939 | 0.101721517 | 0.003404818 |
| 3 | 5 | 0.215934 | 0.103657300 | 0.002109147 |

train loss 最低的后期模型不是 val1k 最优模型，不能用最终 epoch 或 train loss 单独选模。第一候选
因此选择 epoch 10；最终 epoch 20 作为训练阶段不同、具有互补意义的第二候选。

### 60.2 云端同机旧 10k 公平复测

虽然本机已有旧 10k epoch 10 的历史 val1k repeat3 结果，为排除机器、代码和运行时差异，将旧
checkpoint 与对应 `args.json` 上传到云端独立参考目录：

```text
/root/autodl-tmp/reference/old10k_epoch10
```

上传前后 SHA256 一致：

```text
6902eb677c94a33287fc496878b1783d719fe187c7c45abdfe5d9a5ff7269258  model_epoch_10_trainloss_0.1380.pth
910c90a3fbdf5f4736334bcac86531fa4006d8223ad49686f1654dcad0b67b18  args.json
```

复测命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/conda_envs/diffusion_planner/bin/python \
HDP-nuplan/evaluate_checkpoints.py \
  --args_file /root/autodl-tmp/reference/old10k_epoch10/args.json \
  --checkpoint_dir /root/autodl-tmp/reference/old10k_epoch10 \
  --pattern model_epoch_10_trainloss_0.1380.pth \
  --data_dir /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/cache \
  --data_list /root/autodl-tmp/workspace/Diffusion-Planner/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 32 \
  --num_workers 8 \
  --repeats 3 \
  --seed 3407 \
  --device cuda \
  --output /root/autodl-tmp/reference/old10k_epoch10/mini_val_1000_repeat3_currentcloud.json
```

结果退出码为 0。完全相同云端协议下：

| 模型 | ego planning loss | hybrid loss | total loss |
|---|---:|---:|---:|
| 旧 10k epoch 10 | 0.041640617 | 8.159039513 | 0.123231010 |
| target-matched 11,040 epoch 10 | 0.033100349 | 6.409326234 | 0.097193610 |
| 新减旧 | -0.008540268 | -1.749713279 | -0.026037400 |
| 相对变化 | -20.51% | -21.45% | -21.13% |

三个 repeat 的 total loss 也逐次全部更低：

```text
旧 10k：0.128007320，0.118280233，0.123405477
新 11,040：0.098885719，0.093614047，0.099081065
```

因此 open-loop 改善不是由单次随机波动造成。但根据第 55～58 节的经验，open-loop 不能替代
closed-loop 门禁。

### 60.3 checkpoint 下载和本机运行时约束

第 55 节已经实测云端 CUDA 运行时会使同一三场景控制组得到与本机历史结果矛盾的 `score=0`，
所以本轮 closed-loop 仍在产生旧基线的本机环境执行。下载并校验两个候选：

```text
d0b4817cbbd1f8dcbd442f6c9d7c24fc8aecc6dffbab50e4896caa4d352657b6  args.json
fb55d6e48136ac0daba20402ca6b3ac9a7cc1e5c2f2e9d6a3d4791335f61cd2e  model_epoch_10_trainloss_0.1587.pth
81e3a3e6302d6e7232a95f4a882cf7f0ac0c8688c3886fbbf2eb960ef5737491  model_epoch_20_trainloss_0.1309.pth
```

本地与云端 SHA256 完全一致。评测输入固定为：

```text
challenge：closed_loop_nonreactive_agents
scenario filter：mini-val-closed-loop-20
scenario 数：20
simulation seed：0
worker：sequential
planner：HDP
本机 Python：/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
```

命令模板：

```bash
env \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_eval/closed_loop \
  DIFFUSION_PLANNER_PYTHON=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
bash HDP-nuplan/scripts/run_mini_closed_loop.sh \
  <experiment_uid> \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_eval/checkpoints/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/targetmatch_11040_eval/checkpoints/model_epoch_<N>_trainloss_<LOSS>.pth \
  mini-val-closed-loop-20 \
  hdp
```

两次运行均为 20 成功、0 失败：

```text
epoch 10：00:14:55
epoch 20：00:15:43
```

每次均出现 3 个 `All route_list elements are empty` warning；旧 10k 和发布版控制组在同一固定
token 上也有该数据 warning，相关场景均完成并写入指标，因此不计为模型失败。

### 60.4 固定 20 场景最终结果

统一汇总结果：

| 模型 | score | 无责任碰撞 | 可行驶区域 | making progress | 方向 | 路线进度 | TTC | 限速 | 舒适性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 HDP 10k epoch 10 | 0.787373 | 0.95 | 1.00 | 0.90 | 0.975 | 0.668771 | 0.90 | 0.999990 | 1.00 |
| 新 11,040 epoch 10 | 0.729273 | 0.90 | 0.95 | 0.90 | 1.000 | 0.637729 | 0.90 | 0.999992 | 1.00 |
| 新 11,040 epoch 20 | 0.614740 | 0.90 | 0.85 | 0.85 | 1.000 | 0.657656 | 0.85 | 0.999991 | 0.95 |

相对旧 10k：

```text
epoch 10 score：-0.058100（-7.38%）
epoch 20 score：-0.172634（-21.93%）
```

epoch 10 的主要退化集中在两个安全关键场景：

1. `36b0447fd8235d0c`（`traversing_crosswalk`）：可行驶区域指标由 1 降为 0；
2. `b2c0d1c5589e540f`（`following_lane_with_slow_lead`）：无责任碰撞和 TTC 均由 1 降为 0。

除此之外，多数运动场景的路线进度小幅下降。epoch 20 虽然路线进度比 epoch 10 略高，但可行驶
区域、making progress、TTC 和舒适性进一步退化，所以其总分更低。继续训练降低 train loss 没有
转化为闭环收益。

### 60.5 决策和下一步

本轮监督门禁判定为 **不通过**：

1. 不从 target-matched 11,040 的 epoch 10 或 epoch 20 启动 RL；
2. 旧 HDP 10k epoch 10 继续作为当前监督基线和 RL 起点；
3. 不因 val1k 降低 21.13% 宣称规划性能提高，固定 20 场景已经给出相反证据；
4. target-matched 数据方案是有效改进方向：其 epoch 10 已把先前 baselinecfg epoch 20 的
   `0.585445` 提高到 `0.729273`，但仍未恢复旧 10k 的 `0.787373`；
5. 下一轮应先检查新增 1,040 样本与旧 10k 的速度、未来位移、scenario type、route 有效率和
   两个失败场景邻域的覆盖，针对安全/可行驶区域构造数据质量门禁；不得直接用 RL 掩盖监督退化；
6. 若继续监督实验，应先在当前代码和同一云端环境重训旧 10k 控制组；在控制组建立后，再固定
   代码、硬件、样本数和优化步数，只调整数据选择，才能形成可归因对照。

### 60.6 结果文件与异常记录

关键本地文件：

```text
HDP-nuplan/tmp/targetmatch_11040_eval/open_loop/old10k_epoch10_repeat3_currentcloud.json
HDP-nuplan/tmp/targetmatch_11040_eval/open_loop/targetmatch_11040_checkpoint_ranking_repeat3.json
HDP-nuplan/tmp/targetmatch_11040_eval/final_candidates_vs_old10k_mini_val_20.json
HDP-nuplan/tmp/targetmatch_11040_eval/epoch10_mini_val_20.log
HDP-nuplan/tmp/targetmatch_11040_eval/epoch20_mini_val_20.log
```

结果 SHA256：

```text
9c4ad04cc5550a37a8fb337af370902fe6bf1622d88e1eb3ddb31131eafc27d9  old10k_epoch10_repeat3_currentcloud.json
798f188b59c20579b7f244c0f213e402a28f47d524718db20ec70cb866be162a  targetmatch_11040_checkpoint_ranking_repeat3.json
3c8c3f7d81f22853e95210a3c30a745de46de1c4b6e6e8349aa073f0e20dfc50  final_candidates_vs_old10k_mini_val_20.json
```

传输 epoch 20 和归档 open-loop JSON 时各遇到一次 AutoDL SSH `Connection refused`；只读连接
随后恢复，使用 rsync 重新传输成功并通过 SHA256。失败尝试没有产生不完整 checkpoint、没有覆盖
实验，也没有影响两次本机 closed-loop 结果。

## 61. 旧 10k 与新 11,040 训练过程一致性复核

时间：2026-08-10。

触发原因：闭环结果显示旧 10k epoch 10 强于新模型后，重新核查“两个实验是否真的只改变数据”。
结论是：**命令行超参数基本一致，但训练过程不完全一致，不能把闭环差值单独归因于新增数据。**

### 61.1 一致的部分

逐键比较两个 `args.json`，二者都包含 49 个参数。除实验名、输出路径、训练缓存路径和 manifest
路径外，其余参数值完全一致，包括：

```text
seed=3407
batch_size=8
train_epochs=20
learning_rate=5e-4
warm_up_epoch=2
planning_hybrid_loss=0.01
freeze_encoder_epochs=3
diffusion_model_type=x_start
diffusion_supervision_type=x_start
use_data_augment=true
augment_prob=0.5
use_ema=true
num_workers=4
```

encoder 初始化 checkpoint 和 normalization 文件的本机/云端 SHA256 也完全一致：

```text
7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45  checkpoints/model.pth
c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a  HDP-nuplan/normalization.json
```

旧训练的 warm-start 报告证明 151/151 个 encoder tensor、1,799,040 个参数被加载，decoder 加载
数为 0；新训练的启动报告相同。因此初始化来源和模型结构不是已发现的差异。

### 61.2 不一致的部分

#### 训练代码版本不同

旧 10k 在 2026-08-01 训练，使用当时的工作区代码；新 11,040 在 2026-08-10 使用提交
`7200254` 的归档代码训练。8 月 8 日在逐行审计中修复了两个会影响监督优化的实现：

1. `detached_integral()` 原代码使用 `shifted[:, :, :detach_window_size]`。对监督张量
   `[B,T,2]` 而言，这会切到最后的 xy 特征维，并把两个特征全部清零，使 hybrid waypoint loss
   实际使用完整 `cumsum` 梯度；新代码改为 `shifted[..., :detach_window_size, :]`，只允许最近
   10 个时间步传播积分梯度。前向轨迹相同，但反向梯度不同。
2. 旧 `VPSDE_linear.transform("x_start->x_start")` 会先绕行到 noise 再转换回来，分母中的
   `1e-6` 会产生微小数值和梯度误差；新代码对 `src == tgt` 直接返回输入。

其中第一项明确改变 hybrid loss 的梯度路径。尽管其总损失权重只有 0.01，也足以使非凸优化走向
不同参数区域。因此“相同 args”不等于“相同训练算法”。

#### 每 epoch 的 optimizer update 数不同

`DataLoader(drop_last=True)` 且 batch size 为 8：

```text
旧 10k：每 epoch 10,000 / 8 = 1,250 batch
新 11,040：每 epoch 11,040 / 8 = 1,380 batch
```

到候选 epoch 10 时，旧模型执行 12,500 次 update，新模型执行 13,800 次 update，多 10.4%。
两者 scheduler 按 epoch 更新，因此相同 epoch 的学习率阶段相同，但在每个学习率阶段经历的梯度
步数不同。

#### 相同 seed 不会保留共享样本的训练顺序

`DistributedSampler(shuffle=True)` 的 permutation 依赖数据集长度。10,000 和 11,040 使用相同
seed 时会生成不同排列；新训练不是“先完全重复旧 10k，再追加 1,040”。共享的 10,000 条样本会
以不同顺序出现，并匹配到不同的数据增强随机数、扩散时间 `t` 和噪声 `z`，优化轨迹因此改变。

#### 运行硬件不同

旧模型在本机 RTX 4060 Laptop GPU 训练，新模型在云端 RTX 4090 训练。二者均使用 PyTorch
2.0.0+cu118，且训练入口设置 `cudnn.deterministic=True`、`benchmark=False`，但不同 GPU、驱动和
底层 kernel 仍不能保证 bitwise 完全一致。当前没有证据表明硬件是主要原因，但它是严格复现实验
必须消除的混杂变量。

### 61.3 数据本身的精确差异

集合关系通过 manifest 验证：旧 10k 是新 11,040 的严格子集，新增文件恰好 1,040 个且无重复。
但采样结构变化很大：

| 项目 | 旧 10k | 新增 1,040 | 新 11,040 |
|---|---:|---:|---:|
| 日志数 | 44 | 429 个新日志 | 473 |
| 平均每个新增日志样本 | - | 约 2.4 | - |
| 最终位移均值 | 32.06 m | 40.08 m | 32.81 m |
| 最终位移小于 1 m | 27.65% | 20.29% | 26.96% |
| route lane 有效 | 97.69% | 94.62% | 97.40% |

新增样本城市构成为 Singapore 196、Boston 422、Pittsburgh 422，没有新增 Las Vegas。它把最终
城市比例调整为 LV/SG/Boston/Pittsburgh=`70/10/10/10`，但新增样本按新日志广度选择，没有对安全
关键行为均衡：新增 1,040 条中 `following_lane_with_slow_lead` 只有 1 条，`traversing_crosswalk`
只有 11 条。epoch 10 的两个主要失败场景正属于 slow-lead 和 crosswalk 类别。这是“覆盖不足”的
证据，但还不是新增数据导致退化的因果证明。

### 61.4 对“旧 10k 更强”的准确解释

旧 10k 只是在当前固定 20 场景 closed-loop 上更强；新模型在同机 val1k open-loop total loss 上
反而低 21.13%。因此不能概括成旧模型所有能力都更强。准确表述是：

```text
旧 10k checkpoint：固定 20 场景安全性和综合闭环分数更好。
新 11,040 checkpoint：平均 open-loop 轨迹拟合更好，但反馈闭环中的碰撞、可行驶区域和进度更差。
```

由于训练代码、更新次数、样本顺序、数据构成和硬件同时变化，当前实验只能证明“新训练组合得到的
checkpoint 没有通过旧基线门禁”，不能证明“多出的 1,040 条数据必然有害”。

### 61.5 下一项真正可归因的实验

优先级最高的控制实验是：在当前 `7200254` 代码、同一台云端 4090、同一环境和相同命令下，重新
训练旧 10k。它能先回答代码修复和硬件变化是否已经改变旧 10k 的结果。

若需要进一步只检验城市/数据构成，应构造固定 10,000 样本的 target-matched replacement 集：

1. 保持总样本数 10,000、每 epoch 1,250 batch、总 update 数和 LR schedule 一致；
2. 保持当前代码、GPU、seed、初始化 checkpoint 和 normalization 不变；
3. 用目标城市样本替换旧集合中的等量样本，而不是把 1,040 条直接追加；
4. 再执行相同 val1k repeat3 和固定 20 场景 closed-loop。

只有完成上述控制后，才能把差值主要解释为数据构成影响。

## 62. target-matched 11,040 的论文一致监督起点

时间：2026-08-11。

项目目标现改为验证“论文方案 RL 在新数据上能否相对其配对监督起点产生正收益”。因此本轮不再
执行第 61.5 节的数据因果控制实验，而是保留其结论作为边界，转入以下配对实验：

```text
11,040 supervised（planning_hybrid_loss=0.1）→ 同 checkpoint 的论文 RL → 同场景闭环配对
```

监督启动配置、哈希、判定标准和完整命令记录在
`HDP_RL_论文方案迁移操作日志.md` 第 19 节。第一次连接云端
`connect.cqa1.seetacloud.com:11156` 返回 `Connection refused`。截至记录时训练尚未启动；没有
创建云端输出目录，也没有占用 GPU。恢复连接后，先做只读验收再执行日志中的唯一启动命令。

连接随后恢复。验收确认单卡 RTX 4090 空闲、数据盘可用 111GB、11,040 个 NPZ 与三项输入哈希
正确、核心训练代码与本地哈希一致、目标目录不存在；全部测试为 `56 passed`。监督训练于
`2026-08-11 00:15:30 CST` 在 `hdp11040_paper_sup` 后台启动，实际 `args.json` 已确认
`planning_hybrid_loss=0.1`。同时启动 `hdp11040_paper_eval`，仅在训练成功后自动执行 val1k × 3
checkpoint 排名。完整命令、路径和启动证据见论文迁移日志第 19.4 节；RL 不会在选模和闭环监督
基线完成前自动启动。

首轮已正常落盘：`model_epoch_1_trainloss_3.8623.pth`，大小 67,417,699 bytes；随后进入 epoch 2，
未出现 NaN、Traceback 或退出。该检查排除了 screen 空启动，但不用于选模。

前三轮 encoder 均为冻结状态，loss 依次为 `3.8623、2.6349、1.4178`；epoch 4 日志变为
`Encoder trainable: True`，显存从约 927 MiB 增至约 1,803 MiB，解冻路径验证通过。00:24:53
已有5个 checkpoint 并进入 epoch 6。停止本地前台轮询不影响云端训练和自动排名 screen。

01:01 复核确认训练与 val1k 排名退出码均为 0，20 个 checkpoint 全部生成，GPU 已空闲。val1k
三次均值排名第一的是 epoch 10（`0.613795605`），第二为 epoch 14（`0.619970966`），第三为
epoch 9（`0.631156356`）。当前只完成监督训练和 open-loop 选模，尚未启动 RL；下一步先下载
epoch 10，在本机固定 20 场景建立其配对监督基线。详细排名和哈希见论文迁移日志第 19.5 节。

后续已完成完整配对实验：epoch 10 监督闭环为 `0.703840943`；从新 11,040 中按城市比例固定抽取
1,000 条执行论文 RL 后，RL 闭环为 `0.704483158`，平均提高 `0.091244%`，route progress 提高
`3.600599%`，TTC 从 `0.80` 提高到 `0.85`，collision 和 drivable area 不变。但一个
`following_lane_without_lead` 场景的 driving-direction compliance 从 1 降至 0.5，整体均值从
1.000 降至 0.975。因此结论为“带方向副作用的弱正收益”，不是稳健全面提升。完整数据选择、RL
参数、checkpoint 哈希和逐场景结果见论文迁移日志第 19.6～19.9 节。
