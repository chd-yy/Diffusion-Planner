# HDP-nuPlan mini 监督训练与 RL 迁移操作日志

## 1. 任务目标

- 使用现有 NuPlan mini 数据验证 `原始 DB -> HDP NPZ -> 监督训练 -> checkpoint -> RL 微调` 链路。
- 训练阶段只使用官方 `train` 划分，不把 `val/test` 混入训练。
- 使用原有 Conda 环境 `/home/yanjun/NewDisk/conda_envs/diffusion_planner`。
- 完整记录关键命令、判断依据、结果、错误及修复，便于后续复盘。

## 2. 操作边界

- 仓库起始提交：`7571a3b`。
- 工作区在开始前已有大量未提交的 HDP-nuPlan 迁移修改；本次保留这些修改，不执行 reset、checkout、clean 或提交。
- 首轮只处理 100 个训练场景并训练 2 epoch，确认链路可靠后才扩大规模。
- **当前 RL scorer 是基于缓存张量的离线近似 scorer**；它不是 NAVSIM PDM 闭环 scorer 的完全等价实现。

## 3. 环境与数据基线

记录时间：`2026-07-31 23:22:41 CST (+0800)`。

### 3.1 软件与硬件

| 项目 | 实际值 |
|---|---|
| Python | `3.9.25` |
| PyTorch | `2.0.0+cu118` |
| CUDA runtime | `11.8` |
| CUDA 可用 | `True` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| 显存 | `8188 MiB` |
| NVIDIA Driver | `580.173.02` |
| nuPlan-devkit | `/home/yanjun/NewDisk/nuplan-devkit/nuplan/__init__.py` |
| 磁盘剩余 | `137G` |

环境检查命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python --version
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -c \
  'import torch, nuplan; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), nuplan.__file__)'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
df -h /home/yanjun/NewDisk/Diffusion-Planner /home/yanjun/NewDisk/nuplan/dataset
```

### 3.2 本地 DB 数据

| 目录 | DB 数 | 总大小 | 平均大小 |
|---|---:|---:|---:|
| `/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini` | 64 | 13.37 GiB | 213.85 MiB |
| `/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1/trainval` | 36 | 0.19 GiB | 5.42 MiB |

划分核验结论：

- mini 的 64 个 DB 与 nuPlan-devkit 的 `nuplan.yaml` 取交集后得到 44 train、10 val、10 test。
- `nuplan-v1.1/trainval` 中的 36 个 DB 全部匹配官方 val 清单，不能作为正式训练数据。
- `HDP-nuplan/nuplan_train.json` 包含 13,180 个官方 train 日志名；用于 mini 数据目录时，`ScenarioFilter` 只会匹配本地存在的 44 个 train 日志。

## 4. 执行记录

### 4.1 建立日志与记录基线

状态：完成。

操作：

1. 读取 Git 提交及工作区状态。
2. 核验 Python、PyTorch、CUDA、GPU、nuPlan-devkit 和磁盘空间。
3. 统计两个 NuPlan DB 目录的文件数量与体积。

结果：环境可用，RTX 4060 只有 8 GiB 显存，因此监督训练从全局 batch size 8 开始，RL 从更小的 batch/group 配置开始。

后续步骤会继续追加到本文档。

### 4.2 生成 mini 官方划分清单并修复预处理复现问题

状态：完成。

判断与修改：

1. 原 `data_process.py` 将 NPZ 文件清单固定写入当前目录的 `diffusion_planner_training.json`，不同实验容易相互覆盖。
2. 原 `--shuffle_scenarios` 使用 `type=bool`；Python 中 `bool("False")` 为真，因此命令行无法可靠关闭随机打乱。
3. 新增 `--output_list_path`，让每个实验把数据清单写入自己的运行目录。
4. 新增显式布尔解析，支持 `true/false、yes/no、1/0`。
5. NPZ 文件名排序后写入 JSON，保证清单顺序可复现。
6. 新增 `scripts/build_mini_log_splits.py`，读取本地 DB 文件名与 nuPlan-devkit 官方 `nuplan.yaml`，生成互斥的 train/val/test JSON，并在发现遗漏或重叠时直接报错。

执行命令：

```bash
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
$PY -m py_compile HDP-nuplan/data_process.py HDP-nuplan/scripts/build_mini_log_splits.py
$PY HDP-nuplan/scripts/build_mini_log_splits.py \
  --mini-db-dir /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --splitter-yaml /home/yanjun/NewDisk/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml \
  --output-dir HDP-nuplan/config/mini_splits
```

结果：

| 清单 | 数量 |
|---|---:|
| `config/mini_splits/mini_train_logs.json` | 44 |
| `config/mini_splits/mini_val_logs.json` | 10 |
| `config/mini_splits/mini_test_logs.json` | 10 |

额外核验：生成的 44 个 mini-train 日志与“mini 本地 DB 和原 `nuplan_train.json` 的交集”完全一致；`comm -3` 无输出。三个清单合计覆盖全部 64 个 mini DB，生成脚本未发现重叠或遗漏。

### 4.3 预处理 100 个 mini-train 场景

状态：完成。

第一次运行目录：`tmp/mini_train_smoke_100_v1`。

执行命令：

```bash
RUN_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v1
mkdir -p "$RUN_ROOT/cache"
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path "$RUN_ROOT/cache" \
  --log_names_json ./config/mini_splits/mini_train_logs.json \
  --output_list_path "$RUN_ROOT/diffusion_planner_training.json" \
  --total_scenarios 100 \
  --shuffle_scenarios false
```

第一次运行结果：

- `ScenarioFilter` 返回 100 个场景。
- 100/100 成功处理，约 52 秒。
- 生成 100 个唯一 NPZ，共 14.95 MiB，平均 153.08 KiB。
- 清单中无缺失文件；所有数值张量均无 NaN/Inf。
- 张量形状与模型约定一致。

审计时发现，原 NPZ 只保存 `map_name/token`，没有 `log_name/scenario_type`。这不影响当前离线监督训练，但有两个长期问题：

1. 无法从缓存直接审计 train/val/test 来源；
2. 后续若接入按原始场景重建的 NuPlan/PDM scorer，缺少日志定位信息。

因此在 `DataProcessor` 中新增 `log_name` 和 `scenario_type` 元数据，然后保留 v1、不覆盖旧产物，在新目录 `tmp/mini_train_smoke_100_v2` 使用相同命令重跑。第二次处理约 47 秒，仍生成 100/100 个场景。

第二次完整校验结果：

| 校验项 | 结果 |
|---|---:|
| manifest 文件数 | 100 |
| 唯一文件数 | 100 |
| 缺失文件 | 0 |
| 非有限数值 | 0 |
| 非 train 日志场景 | 0 |
| 100 场景涉及的 train 日志 | 35/44 |
| 场景类型 | 15 种 |
| v1/v2 原数值张量不一致 | 0 |
| Dataset 长度 | 100 |
| 单样本监督张量数 | 11 |

固定形状：

```text
ego_current_state          (10,)
ego_agent_future           (80, 3)
neighbor_agents_past       (32, 21, 11)
neighbor_agents_future     (32, 80, 3)
lanes                      (70, 20, 12)
lanes_speed_limit          (70, 1)
lanes_has_speed_limit      (70, 1)
route_lanes                (25, 20, 12)
route_lanes_speed_limit    (25, 1)
route_lanes_has_speed_limit (25, 1)
static_objects             (5, 10)
```

结论：使用 `shuffle=false` 时 v1/v2 选中了完全相同的 100 个场景，新增元数据没有改变任何模型输入或监督张量。后续训练使用 v2。

### 4.4 监督训练冒烟测试

状态：完成。

训练前先执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q HDP-nuplan/tests
```

结果：`3 passed in 3.08s`。

#### 尝试 1：warm-up 参数不适用于 2 epoch smoke

初始命令使用 `train_epochs=2`，但遗漏了覆盖默认的 `warm_up_epoch=5`。调度器初始化时触发：

```text
AssertionError: epoch >= warm_up_epoch
```

该错误发生在训练开始前，没有更新模型。修复方式是在 smoke 命令中增加 `--warm_up_epoch 1`。失败运行只生成了参数文件，目录为：

```text
training_log/hdp-mini-supervised-smoke/2026-07-31-23:27:30/
```

#### 尝试 2：DDP 条件分支未使用参数

加入 `--warm_up_epoch 1` 后，第一个 batch 成功前向和反向，但第二个 batch 报错：

```text
RuntimeError: Expected to have finished reduction in the prior iteration...
Parameter indices which did not receive grad for rank 0: 58
```

训练器自身的梯度诊断进一步定位到：

```text
module.encoder.encoder.lane_encoder.unknown_speed_emb.weight
```

根因：车道速度限制编码器包含合法的数据依赖分支。一个 batch 如果所有有效车道都有速度限制，就不会使用 `unknown_speed_emb`；另一个 batch 可能会使用。因此 DDP 必须允许条件性未使用参数。

修复：监督训练的 DDP 包装增加 `find_unused_parameters=True`。失败运行目录：

```text
training_log/hdp-mini-supervised-smoke/2026-07-31-23:27:44/
```

#### 尝试 3：训练成功

最终关键参数：

```text
GPU                 1 x RTX 4060 Laptop 8 GiB
global batch size   8
num_workers         2
train epochs        2
warm-up epochs      1
model type          x_start
supervision type    x_start
hybrid loss weight  0.01
model parameters    5,092,996
```

事后审计说明：本次 smoke 配置的基础学习率是 `5e-4`，但当
`warm_up_epoch=1` 时，旧调度器创建了 `LinearLR(total_iters=0,
start_factor=0.1)`，实际学习率被固定成 `5e-5`。该问题当时不影响“链路是否能跑通”的
判断，但会影响优化强度；问题在 1000 场景 pilot 中被 TensorBoard 审计发现并修复，
详见 4.8。这里保留原始结果，不重写历史。

成功运行目录：

```text
tmp/mini_train_smoke_100_v2/training_log/hdp-mini-supervised-smoke/2026-07-31-23:28:29/
```

结果：

| Epoch | 平均训练损失 |
|---:|---:|
| 1 | 6.1090 |
| 2 | 3.7177 |

两轮之间损失下降约 39.1%。该结果只证明代码能够学习这 100 个训练样本，不能当作泛化性能结论。

生成文件：

```text
model_epoch_1_trainloss_6.1090.pth
model_epoch_2_trainloss_3.7177.pth
latest.pth
args.json
```

checkpoint 校验：

| 项目 | 结果 |
|---|---:|
| `latest.pth` 大小 | 78.14 MiB |
| checkpoint epoch | 2 |
| checkpoint loss | 3.7176840305 |
| EMA state tensors | 260 |
| EMA 非有限张量 | 0 |
| strict load missing keys | 0 |
| strict load unexpected keys | 0 |
| 加载后模型参数 | 5,092,996 |

结论：该 `latest.pth` 与当前 HDP-nuPlan 模型严格兼容，可以作为 RL 冒烟测试的监督起点；但只有 100 个场景、2 epoch，不能视为训练完成的高质量规划模型。

### 4.5 RL 微调冒烟测试

状态：完成。

目标：验证“监督 checkpoint 加载 -> 分组扩散采样 -> 离线奖励 -> replay buffer -> 奖励加权扩散更新 -> RL checkpoint”整条链路。

关键配置：

```text
train scenes             100
global batch size        4
group size               4
rollout diffusion steps  2
RL epochs                2（epoch 1 rollout，epoch 2 update）
requested learning rate  1e-5
actual learning rate     1e-6（旧 warm-up 边界问题，后续已修复）
replay buffer capacity   128
freeze encoder           true
```

监督起点：

```text
tmp/mini_train_smoke_100_v2/training_log/hdp-mini-supervised-smoke/
2026-07-31-23:28:29/latest.pth
```

RL 输出：

```text
tmp/mini_train_smoke_100_v2/training_log/hdp-mini-rl-smoke/
2026-07-31-23:29:41/
```

#### Rollout 结果

- 监督权重加载：`missing=0, unexpected=0`。
- 25 个 batch 全部完成。
- 最终 replay buffer：100 个场景，每个场景 4 条候选轨迹。
- 平均奖励：`-14.52398`。
- 平均 progress：`-0.16992`。
- 无碰撞候选比例：`0.3925`。
- 平均 collision cost：`0.33397`。
- 平均 route cost：`0.31850`。
- 平均 comfort cost：`105.91037`。
- 平均 backward cost：`0.10486`。
- imitation cost 仅记录未加权，均值：`16.10886`。

#### Update 结果

- 25 个更新 batch 全部完成。
- 平均总损失：`0.09191936`。
- reward-weighted diffusion loss：`0.08541688`。
- reward-weighted waypoint loss：`0.65024803`，乘 `0.01` 后进入总损失。
- 平均 reward：`-14.65681`。
- 平均组内最大 reward：`-14.64782`。
- 平均组内最小 reward：`-14.66562`。
- replay buffer 在更新阶段保持 100 个场景。

#### RL checkpoint 校验

| 项目 | 结果 |
|---|---:|
| `latest.pth` 大小 | 64.29 MiB |
| checkpoint epoch | 2 |
| checkpoint loss | 0.0919193617 |
| EMA tensors | 260 |
| EMA 非有限张量 | 0 |
| strict load missing keys | 0 |
| strict load unexpected keys | 0 |

冻结审计使用 RL checkpoint 中的即时 `model` 权重与监督 EMA 起点比较：

- encoder 改变张量：0，证明 `rl_freeze_encoder=true` 生效；
- decoder 改变张量：109；
- decoder 最大绝对参数变化：`2.59727e-05`。

EMA 的 encoder 出现约 `1e-7` 量级舍入差异，这是 EMA 重复执行 `decay * ema + (1-decay) * model` 的浮点舍入，并非优化器更新；即时 model encoder 逐张量完全相等。

#### 当前 RL 结果的限制

1. 当前奖励是缓存张量上的离线近似，不是 NAVSIM 使用的 PDM simulator/scorer，也不是 NuPlan 官方闭环分数。
2. 监督模型只训练了 100 场景、2 epoch，因此 progress 为负、轨迹质量低是预期现象。
3. comfort cost 均值约 105.9，在当前权重下贡献约 -10.59，是总奖励的主要项之一；正式训练前需要在更可靠的监督 checkpoint 上重新统计并校准奖励尺度。
4. 当前只用 2 个采样步，4 个候选的平均组内 reward 范围约 0.0178，候选差异偏小。正式 RL 应提高采样步数，并检查候选多样性后再确定 group size 和 temperature。
5. 该次命令请求的学习率为 `1e-5`，但受 4.4 所述调度器边界问题影响，实际为 `1e-6`；因此它仍只作为软件链路冒烟测试，不用于质量结论。

结论：RL 软件链路和梯度更新已通过，但该 RL checkpoint 只具有工程冒烟价值，不代表获得了有效驾驶策略。

### 4.6 全量 mini-train 成本估算与 pilot 决策

状态：完成。

在直接处理全部场景前，先用 SQLite 只读统计 44 个 mini-train DB：

| 数据项 | 行数 |
|---|---:|
| `scenario_tag` | 638,483 |
| `lidar_pc` | 359,061 |

这里不能把 638,483 直接解释成 638,483 个唯一训练样本：同一 lidar token 可能具有多个
scenario tag，而当前 NPZ 以场景 token 命名，重复 token 会覆盖。它只用于量级估计。

100 场景缓存平均约 153 KiB，据此外推：

| 场景量 | 粗略缓存体积 |
|---:|---:|
| 1,000 | 0.15 GiB |
| 100,000 | 14.6 GiB |
| 638,483（上界估计） | 93.2 GiB |

结合 100 场景预处理约 47 秒、8 GiB 显存以及当前 RL scorer 仍是离线近似，直接启动全部
mini 数据会产生数十 GiB 缓存并可能运行数十小时，却不能回答链路和配置是否正确。因此先执行
固定随机种子的 1000 场景 pilot，再用独立 mini-val 选择 checkpoint。该决策不删除任何已有
数据，也没有启动无法快速验证的大规模作业。

### 4.7 固定种子预处理 1000 个 mini-train 场景

状态：完成。

为了让随机场景选择可复现，`data_process.py` 新增 `--seed`，默认值为 3407，并在场景打乱前
调用 `random.seed(args.seed)`。运行目录与 100 场景 smoke 隔离：

```text
tmp/mini_train_pilot_1000_seed3407_v1/
```

执行命令：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
RUN_ROOT=$PWD/tmp/mini_train_pilot_1000_seed3407_v1
mkdir -p "$RUN_ROOT/cache"

$PY data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path "$RUN_ROOT/cache" \
  --log_names_json ./config/mini_splits/mini_train_logs.json \
  --output_list_path "$RUN_ROOT/diffusion_planner_training.json" \
  --total_scenarios 1000 \
  --shuffle_scenarios true \
  --seed 3407
```

结果：

| 校验项 | 结果 |
|---|---:|
| 运行时间 | 约 7 分 37 秒 |
| manifest 条目 | 1,000 |
| 唯一 NPZ | 1,000 |
| manifest 缺失文件 | 0 |
| 必需 key 缺失 | 0 |
| 非有限数值 | 0 |
| 非 train 日志场景 | 0 |
| 覆盖 train 日志 | 43/44 |
| 场景类型 | 26 种 |
| 缓存体积 | 150.204 MiB |
| 平均 NPZ 大小 | 153.81 KiB |

唯一未抽中的 train 日志为：

```text
2021.06.23.20.43.31_veh-16_03607_04007
```

这表示“固定种子抽取的 1000 场景没有命中该日志”，不表示该日志不属于 train。主要场景类型：

| 场景类型 | 数量 |
|---|---:|
| `stationary` | 331 |
| `traversing_traffic_light_intersection` | 135 |
| `stationary_in_traffic` | 117 |
| `traversing_pickup_dropoff` | 97 |
| `medium_magnitude_speed` | 88 |
| `high_magnitude_speed` | 58 |
| `traversing_intersection` | 42 |
| `stationary_at_traffic_light_without_lead` | 28 |

数据分布明显偏向静止场景，这是 NuPlan scenario tags 和当前随机抽样共同形成的分布；pilot
阶段保留真实分布，不在没有依据时人为重采样。

### 4.8 1000 场景监督训练、学习率边界问题与修复

状态：完成。

#### 第一次 10 epoch 训练

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-supervised-pilot-1000/2026-07-31-23:40:13/
```

关键命令与后续修复版相同，仅实验名为 `hdp-mini-supervised-pilot-1000`：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python

$PY -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor.py \
  --name hdp-mini-supervised-pilot-1000 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 \
  --num_workers 2 \
  --train_epochs 10 \
  --warm_up_epoch 1 \
  --save_utd 1 \
  --learning_rate 5e-4
```

训练本身成功，10 轮平均总损失为：

```text
1.78936553, 0.63427436, 0.60414821, 0.53490794, 0.54646307,
0.47613153, 0.43857890, 0.44161072, 0.43290779, 0.39427850
```

但读取 TensorBoard event 后发现，10 轮实际学习率始终是 `5e-5`，不是命令请求的 `5e-4`。
根因是：

```python
LinearLR(start_factor=0.1, total_iters=warm_up_epoch - 1)
```

当 `warm_up_epoch=1` 时，`total_iters=0`，PyTorch 2.0 中优化器学习率被乘以 0.1 后没有
恢复到基础学习率。因此 4.4 的监督 smoke 和 4.5 的 RL smoke 也分别实际使用了 `5e-5`
和 `1e-6`。这不是 checkpoint 损坏，但属于必须修复的配置语义错误。

修复内容：

1. `hdp_nuplan/utils/lr_schedule.py`：当 `warm_up_epoch <= 1` 时返回恒等
   `MultiplicativeLR`，保持调用方基础学习率；大于 1 时保留原线性 warm-up。
2. 新增 `tests/test_lr_schedule.py`，断言创建调度器及执行一次 `scheduler.step()` 后
   `5e-4` 均保持不变。
3. `train_predictor.py` 新增 `--log_unused_parameters`，默认关闭逐参数 debug 输出；DDP
   的 `find_unused_parameters=True` 仍保留，因为 lane encoder 确有条件分支。

修复后测试结果：

```text
4 passed
```

#### 修复后重新训练

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/
```

复现命令就是上一个命令把 `--name` 改为：

```text
hdp-mini-supervised-pilot-1000-lrfix
```

TensorBoard 核验实际学习率 10 轮均为 `0.0005`。训练指标：

| Epoch | 总损失 | ego planning loss | hybrid loss |
|---:|---:|---:|---:|
| 1 | 1.275325 | 0.428528 | 84.679749 |
| 2 | 0.783230 | 0.256010 | 52.721958 |
| 3 | 0.687917 | 0.224180 | 46.373718 |
| 4 | 0.614624 | 0.194547 | 42.007671 |
| 5 | 0.577879 | 0.192521 | 38.535816 |
| 6 | 0.494645 | 0.165043 | 32.960110 |
| 7 | 0.466989 | 0.153704 | 31.328487 |
| 8 | 0.480345 | 0.159984 | 32.036125 |
| 9 | 0.436284 | 0.146611 | 28.967258 |
| 10 | 0.415616 | 0.138525 | 27.709061 |

checkpoint 校验：

| 项目 | 结果 |
|---|---:|
| `latest.pth` 大小 | 78.14 MiB |
| checkpoint epoch | 10 |
| checkpoint loss | 0.4156156182 |
| strict load missing keys | 0 |
| strict load unexpected keys | 0 |
| 非有限数值 | 0 |

低学习率版本的训练集最终损失略低于修复版，但不能据此选模型，因为二者需要在未参与训练的
数据上比较。因此没有直接选择训练 loss 更低的 checkpoint，而是建立独立 mini-val 缓存。

### 4.9 独立 mini-val 缓存与监督 checkpoint 选择

状态：完成。

#### 生成 100 个验证场景

验证缓存与训练缓存物理隔离，且只接收 10 个官方 mini-val 日志：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
VAL_ROOT=$PWD/tmp/mini_val_pilot_100_seed3407_v1
mkdir -p "$VAL_ROOT/cache"

$PY data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path "$VAL_ROOT/cache" \
  --log_names_json ./config/mini_splits/mini_val_logs.json \
  --output_list_path "$VAL_ROOT/diffusion_planner_validation.json" \
  --total_scenarios 100 \
  --shuffle_scenarios true \
  --seed 3407
```

校验结果：100 个唯一 NPZ、无缺失、无 NaN/Inf、无 train/test 日志混入，覆盖 9/10 个 val
日志和 16 种场景类型。未抽中的 val 日志为：

```text
2021.08.24.13.12.55_veh-45_00386_00472
```

#### 增加固定种子验证入口

新增 `evaluate_predictor.py`，行为如下：

- 从训练 `args.json` 重建当前 HDP 模型；
- 优先严格加载 checkpoint 的 `ema_state_dict`；
- `model.eval()` 关闭 dropout；
- 在固定 seed 3407、3408、3409 下重复计算 3 次 diffusion validation loss；
- 按样本数加权，不按 batch 数平均；
- 同时输出均值和每次结果到 JSON。

首次执行时失败，报错表现为验证 loss 需要 decoder 输出 `score`，但 `model.eval()` 使旧 decoder
只返回 `prediction`。根因是旧代码错误地用 `self.training` 同时决定 dropout 模式和输出契约。

修复 `hdp_nuplan/model/module/decoder.py`：只要输入同时包含
`sampled_trajectories` 和 `diffusion_time` 就返回训练/验证需要的 `score`；普通推理输入缺少这两个
字段时仍返回 `prediction`。这样验证可以处于 eval 模式，同时不会破坏 planner 推理接口。

验证命令模板：

```bash
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
$PY evaluate_predictor.py \
  --args_file <对应监督运行目录>/args.json \
  --checkpoint <checkpoint>/latest.pth \
  --data_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_pilot_100_seed3407_v1/cache \
  --data_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_val_pilot_100_seed3407_v1/diffusion_planner_validation.json \
  --batch_size 8 \
  --num_workers 2 \
  --repeats 3 \
  --seed 3407 \
  --device cuda \
  --output <输出 JSON>
```

三次固定种子均值：

| checkpoint | 实际 LR | ego planning | hybrid | 总损失 |
|---|---:|---:|---:|---:|
| 首次训练 | `5e-5` | 0.394656 | 77.665734 | 1.171313 |
| LR 修复版 | `5e-4` | 0.238385 | 38.430509 | 0.622690 |

LR 修复版的三次总损失分别为：

```text
0.6074072, 0.6399706, 0.6206935
```

输出文件：

```text
tmp/mini_val_pilot_100_seed3407_v1/eval_supervised_lr5e-5.json
tmp/mini_val_pilot_100_seed3407_v1/eval_supervised_lr5e-4.json
```

结论：使用验证总损失 `0.622690` 的 LR 修复版作为 RL 起点。这个选择基于独立日志划分，
不是基于训练集 loss。

### 4.10 1000 场景 RL pilot

状态：完成。

监督起点：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/latest.pth
```

相对 100 场景 RL smoke 的配置调整：

- 采样步数由 2 增至 5，提高候选轨迹质量与差异；
- comfort 权重由 0.1 降至 0.01，避免 comfort cost 单项压倒碰撞和路线信号；
- replay buffer 容量 1024，可容纳全部 1000 个场景；
- 使用修复后真实的 `1e-5` 学习率；
- 冻结场景 encoder，只更新 decoder。

执行命令：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python

$PY -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-rl-pilot-1000 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/latest.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 \
  --num_workers 2 \
  --train_epochs 5 \
  --warm_up_epoch 1 \
  --save_utd 1 \
  --learning_rate 1e-5 \
  --rl_group_size 4 \
  --rl_rollout_steps 5 \
  --rl_buffer_update_epoch 5 \
  --rl_buffer_size 1024 \
  --rl_freeze_encoder true \
  --reward_comfort_weight 0.01
```

输出目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-pilot-1000/2026-07-31-23:49:42/
```

#### RL 前 rollout 指标

| 指标 | 值 |
|---|---:|
| reward | 1.29423668 |
| progress | 2.70126349 |
| no-collision 比例 | 0.74975000 |
| collision cost | 0.09572496 |
| route cost | 0.21621123 |
| comfort cost | 23.16818842 |
| backward cost | 0.00188407 |
| imitation cost（只记录） | 3.90034562 |
| 最终 replay 场景 | 1,000 |

原始控制台的 epoch 汇总曾显示 `buffer_size=504`。检查代码确认，它把每个 batch 后的容量
`8,16,...,1000` 求了平均；`(8 + 1000) / 2 = 504`。进度条和后续 update 均确认真实最终容量
为 1000。已把 `rollout_epoch()` 的汇总改成返回最终 `len(replay_buffer)`，该修改只修正日志
语义，不改变 buffer 内容、采样或梯度。

#### RL 更新指标

epoch 1 仅 rollout，epoch 2--5 从同一 1000 场景 replay buffer 更新：

| Epoch | update loss | 实际 LR | buffer |
|---:|---:|---:|---:|
| 2 | 0.01809396 | `1e-5` | 1,000 |
| 3 | 0.00833132 | `1e-5` | 1,000 |
| 4 | 0.00672711 | `1e-5` | 1,000 |
| 5 | 0.00594997 | `1e-5` | 1,000 |

最后一轮细分指标：

```text
reward_weighted_diffusion_loss = 0.0044241902
reward_weighted_waypoint_loss  = 0.1525780185
reward_mean                    = 1.4379798203
reward_max                     = 1.4421156067
reward_min                     = 1.4335499274
advantage_mean                 = 9.5498879e-06
weight_mean                    = 1.5679315758
```

更新阶段的 `reward_mean` 来自 replay 抽样 batch，不能与完整数据 rollout 的平均 reward 直接
相减；因此另做同配置、固定 seed 的完整 post-rollout。

#### RL checkpoint 审计

| 项目 | 结果 |
|---|---:|
| `latest.pth` 大小 | 64.29 MiB |
| checkpoint epoch | 5 |
| checkpoint loss | 0.0059499704 |
| strict state key 缺失/多余 | 0 / 0 |
| model 非有限数值 | 0 |
| EMA 非有限数值 | 0 |
| `module.encoder.*` 张量 | 151 |
| encoder 改变张量 | 0 |
| `module.decoder.*` 张量 | 109 |
| decoder 改变张量 | 109 |
| decoder 最大绝对变化 | 0.00350105 |

注意：decoder 内部有名为 `route_encoder` 的子模块。若仅用“键名是否包含 encoder”统计，
会把这 26 个应当更新的 decoder 参数误判成冻结 encoder 发生变化。本次冻结审计使用严格前缀
`module.encoder.`，结果为逐张量完全相等。

### 4.11 RL 后独立验证与同配置 rollout

状态：完成。

#### mini-val diffusion loss

使用 4.9 完全相同的 100 个 val 场景、batch size、eval 模式和 3 个 seed，仅替换为 RL
`latest.pth`。输出：

```text
tmp/mini_val_pilot_100_seed3407_v1/eval_rl_pilot_1000.json
```

结果：

| 指标 | RL 前 | RL 后 | 相对变化 |
|---|---:|---:|---:|
| ego planning loss | 0.238385 | 0.231356 | -2.95% |
| hybrid loss | 38.430509 | 36.227327 | -5.73% |
| 总损失 | 0.622690 | 0.593629 | -4.67% |

RL 后三次总损失：

```text
0.58603070, 0.59472764, 0.60012966
```

所以这次短 RL pilot 没有造成监督目标灾难性退化，在该 100 场景验证集上反而略有改善。
这仍不是闭环驾驶效果证明。

#### 固定 seed 的 post-rollout

使用 RL EMA checkpoint 再运行 1 个只 rollout、不更新的 epoch；除
`--pretrained_model_path` 改成 RL `latest.pth`、`--train_epochs 1`、实验名改成
`hdp-mini-rl-pilot-1000-post-eval` 外，其余配置与 4.10 相同。

| 指标 | RL 前 | RL 后 |
|---|---:|---:|
| reward | 1.29423668 | 1.29568970 |
| progress | 2.70126349 | 2.70512732 |
| no-collision 比例 | 0.749750 | 0.755000 |
| collision cost | 0.09572496 | 0.09562183 |
| route cost | 0.21621123 | 0.21645447 |
| comfort cost | 23.16818842 | 23.45995709 |
| backward cost | 0.00188407 | 0.00216523 |
| imitation cost | 3.90034562 | 3.84941584 |

完整离线平均 reward 增加 `0.00145303`，约 `0.11%`。no-collision 和 imitation 略有改善，
route/comfort/backward 略有变差；这是单个固定 seed、短更新上的微小变化，不能宣称具有统计
显著性。它证明的是更新后 scorer 没有明显崩溃，而不是已经得到成熟 RL 策略。

## 5. 最终验证

最终执行：

```bash
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
$PY -m py_compile \
  HDP-nuplan/data_process.py \
  HDP-nuplan/evaluate_predictor.py \
  HDP-nuplan/train_predictor.py \
  HDP-nuplan/train_predictor_rl.py \
  HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py
$PY -m pytest -q HDP-nuplan/tests
```

结果：

```text
4 passed in 3.11s
```

产物占用：

| 目录 | 大小 |
|---|---:|
| `tmp/mini_train_pilot_1000_seed3407_v1` | 943 MiB |
| `tmp/mini_val_pilot_100_seed3407_v1` | 16 MiB |

关键产物 SHA-256（用于后续确认文件没有被覆盖）：

| 产物 | SHA-256 |
|---|---|
| 监督 `latest.pth` | `d8e5b049d9a39cf33c2461d01a302039dae4417800f5f3c52fa06a2dbc454c2d` |
| RL `latest.pth` | `311d6544530c792e32dc231d108a175ba2089bae663a576d72d885c0e9634941` |
| train1000 manifest | `2c8dac2687ed365db2e50069f9284f9498a63f0c9fe51c8969f400bec1591372` |
| val100 manifest | `52e75dca74f2db88cea678a0fadabd3d66d65fc953af76e35881d41e6c941efc` |
| 监督验证 JSON | `7f53e0274a7aea238effedd738aa1736cbc52a1fedb28f023d04c85afd576201` |
| RL 验证 JSON | `a4e966c2cd03c303827da0f3ed3f9f092d675fdd939ec172e0b259277285920a` |

最终状态矩阵：

| 环节 | 状态 | 证据 |
|---|---|---|
| 官方 mini train/val/test 日志隔离 | 通过 | 44/10/10，互斥且覆盖 64 DB |
| 100 场景数据 smoke | 通过 | 100/100、张量有限、来源为 train |
| 2 epoch 监督 smoke | 通过 | strict checkpoint 可加载 |
| 100 场景 RL smoke | 通过 | rollout/update/checkpoint 链路完整 |
| 1000 场景固定种子缓存 | 通过 | 1000/1000、43 个 train 日志 |
| 10 epoch 监督 pilot | 通过 | 实际 LR 已审计并修复 |
| 独立 mini-val 选择模型 | 通过 | `0.622690 < 1.171313` |
| 1000 场景 RL pilot | 通过 | loss `0.018094 -> 0.005950` |
| encoder 冻结 | 通过 | 151 个 encoder 张量变化为 0 |
| RL 后 mini-val | 通过 | 总损失 `0.622690 -> 0.593629` |
| 单元/语法测试 | 通过 | 4 tests passed，py_compile 通过 |

## 6. 本次新增或修改的关键文件

以下是本轮为“mini 监督训练 -> RL pilot -> 独立验证”直接新增或修改的文件；仓库在本轮开始前
已有其他 HDP-nuPlan changes，不能把整个 `git diff` 都归因于本轮：

| 文件 | 作用 |
|---|---|
| `scripts/build_mini_log_splits.py` | 从官方 splitter 建立 44/10/10 清单并校验互斥性 |
| `config/mini_splits/*.json` | 固定 mini train/val/test 日志清单 |
| `data_process.py` | 正确解析布尔参数、指定输出清单、排序、固定 seed |
| `hdp_nuplan/data_process/data_processor.py` | NPZ 增加 `log_name/scenario_type` 审计字段 |
| `train_predictor.py` | DDP 条件未使用参数支持与可选 debug 输出 |
| `hdp_nuplan/utils/lr_schedule.py` | 修复 `warm_up_epoch=1` 的 0.1 倍 LR 边界问题 |
| `tests/test_lr_schedule.py` | 回归测试基础 LR 不被错误缩放 |
| `evaluate_predictor.py` | 固定 seed 的独立 NPZ 验证入口 |
| `hdp_nuplan/model/module/decoder.py` | eval 模式支持 diffusion loss 的输入驱动输出契约 |
| `hdp_nuplan/rl/train_epoch_rl.py` | rollout buffer 最终容量的正确日志语义 |
| `docs/mini_supervised_rl_operation_log.md` | 本文档 |

没有执行 `git add`、`git commit`、`git reset`、`git checkout` 或 `git clean`。全部代码修改、日志、
缓存和 checkpoint 继续作为工作区 changes/未跟踪产物保留，符合“先别提交修改版本”的约束。

## 7. 能得出的结论与不能得出的结论

可以确认：

1. NuPlan mini 的官方 train 日志可以稳定转换为当前 HDP-nuPlan 所需 NPZ。
2. 当前迁移版能够在原 `diffusion_planner` Conda 环境中完成监督训练、EMA checkpoint、分组
   diffusion rollout、离线奖励、replay 更新和 RL checkpoint。
3. 1000 场景 pilot 中，监督 checkpoint 的选择经过独立 mini-val，而非只看训练 loss。
4. 短 RL 更新确实只更新 decoder，没有破坏 encoder，并通过了独立监督目标后测。

仍不能确认：

1. 这不是全部 44 个 train DB 的全量训练，只是固定种子 1000 场景 pilot。
2. val 只有 100 场景、9/10 个日志，不是完整官方 val 评测。
3. 当前 scorer 使用缓存张量近似 progress、collision、route、comfort 等项；它没有执行 NuPlan
   simulator，也没有复现 NAVSIM 的 PDM simulator/scorer。
4. diffusion validation loss 和离线 reward 都不能替代 NuPlan 官方闭环指标。
5. reward 仅提升约 0.11%，不能据此宣称 RL 已获得显著驾驶收益。

因此，准确的完成状态是：**NuPlan mini 上的可复现监督训练 + 离线 diffusion-based RL pilot
已完整跑通并完成独立后测；全量训练和真正 NuPlan/PDM 闭环 RL 尚未完成。**

进度更新：上述结论描述的是 4.11 完成时的状态。第 9 节已经继续完成 NuPlan 官方
`closed_loop_nonreactive_agents` 三模型同场景评测；仍未完成的是“用 PDM/仿真器奖励直接训练”
和大规模训练，而不是闭环评测入口。

## 8. 后续扩大规模时的推荐顺序

1. 先把固定种子场景数扩大到 10k，复用独立 val100，观察缓存、训练时间、奖励尺度和候选
   多样性；不要立即把 `scenario_tag` 行数当作唯一场景数全量处理。
2. 建立覆盖全部 10 个 mini-val 日志的验证清单；用于闭环时还要保存 scenario token、log name、
   map name 并从原始 DB 重建 scenario。
3. 将离线 scorer 替换为 NuPlan simulator/PDM 闭环 scorer，或明确把当前方法命名为
   offline reward-weighted diffusion fine-tuning，避免与 NAVSIM HDP 的闭环设计混为一谈。
4. 每次扩大规模都保留三个门槛：checkpoint strict load、独立 val 不显著退化、官方闭环指标
   不下降；三者通过后再扩大 RL epoch 和 group size。

## 9. NuPlan 官方闭环评测

记录时间：`2026-08-01`。

### 9.1 目标与评测协议

状态：完成。

目标是补齐项目从 checkpoint 到 NuPlan 官方仿真的最后一段，并在完全相同场景和配置上比较：

1. 仓库自带、经过完整训练的原始 Diffusion-Planner checkpoint；
2. 本次 1000 场景 HDP-nuPlan 监督 checkpoint；
3. 从该监督模型继续训练的 HDP-nuPlan RL checkpoint。

统一协议：

```text
challenge       closed_loop_nonreactive_agents
scenario source official mini-val logs
scenario count  3
simulation seed 0
worker          sequential
controller      two_stage_controller（challenge 默认）
metrics         NuPlan simulation_closed_loop_nonreactive_agents
aggregator      closed_loop_nonreactive_agents_weighted_average
GPU             CUDA_VISIBLE_DEVICES=0
```

使用 sequential worker 是为了让三个模型按相同顺序独占 GPU，避免 8 GiB 显存上并行模型实例造成
OOM，也避免 Ray 调度噪声影响 smoke 阶段的可复现性。

### 9.2 固定 3 个 mini-val 场景

从 4.9 的 val100 缓存读取 `token/log_name/scenario_type`，优先选择不同驾驶状态。最初考虑的
SG high-speed token `f5419153ed095c3e` 可以从 DB 重建，但 `mission_goal=None`；因为官方闭环
路线评测需要有效目标，没有勉强使用该场景，而是替换为同为 high-speed、目标有效的 token。

最终场景写入：

```text
hdp_nuplan/config/scenario_filter/mini-val-closed-loop-3.yaml
```

| token | 类型 | log | mission goal | route roadblocks |
|---|---|---|---|---:|
| `134f95eed3775e22` | high_magnitude_speed | `2021.06.08.14.35.24_veh-26_02555_03004` | 有效 | 28 |
| `08764235932a5530` | stationary_in_traffic | `2021.06.07.12.54.00_veh-35_01843_02314` | 有效 | 21 |
| `263da852dba35cac` | traversing_traffic_light_intersection | `2021.07.24.23.50.16_veh-17_01696_02071` | 有效 | 24 |

只读 scenario builder 核验返回 3/3 场景；每个场景约 20 秒、400/401 个原始迭代点，地图分别
覆盖 Las Vegas 场景。配置使用 `remove_invalid_goals=true`，防止未来清单被替换后静默混入
无任务目标场景。

### 9.3 Hydra 配置检查与首次失败

先使用 `--cfg job --resolve`，确认以下 override 已生效：

- `scenario_builder.db_files` 指向本地 64 个 mini DB 目录；
- scenario filter 只含上述 3 个 token；
- planner args/checkpoint 都是绝对路径；
- 输出目录位于 `tmp/closed_loop_eval`；
- worker 为 sequential；
- official closed-loop metrics 与 weighted aggregator 已加载。

第一次真实运行目录：

```text
tmp/closed_loop_eval/exp/simulation/closed_loop_nonreactive_agents/
supervised-one-smoke/
```

在构建 planner 时失败，尚未执行模型推理：

```text
TypeError: Config.__init__() got an unexpected keyword argument 'guidance_fn'
```

根因：`HyperDiffusionPlanner` 已使用独立的 HDP `Config(args_file)`，但
`hyper_diffusion_planner.yaml` 还残留原 Diffusion-Planner 的 `guidance_fn: null`。Hydra 会递归
实例化 config，并把该字段作为构造参数传入，因此接口不匹配。

修复：从 HDP planner YAML 删除多余 `guidance_fn`；不修改模型、checkpoint 或仿真器。失败
目录保留用于复盘。

### 9.4 单场景闭环 smoke

修复后使用相同命令增加：

```text
scenario_filter.limit_total_scenarios=1
experiment_uid=supervised-one-smoke-v2
```

结果：

| 项目 | 结果 |
|---|---:|
| 成功仿真 | 1 |
| 失败仿真 | 0 |
| 仿真阶段耗时 | 约 41 秒 |
| 官方 metrics parquet | 17 个 |
| weighted aggregator | 成功 |
| runner report | 成功 |
| NuBoard 文件 | 成功 |
| summary PDF | 成功 |

这一场是 `stationary_in_traffic / 08764235932a5530`。HDP 监督模型能够完整运行，但发生 2 次
车辆责任碰撞，`no_ego_at_fault_collisions=0`、`time_to_collision_within_bound=0`，最终场景
score 为 0。这里第一次把“软件能跑”和“策略质量好”明确区分开。

### 9.5 可复现运行与汇总脚本

新增：

```text
scripts/run_mini_closed_loop.sh
scripts/summarize_closed_loop_metrics.py
```

运行脚本会检查 Python、args、checkpoint 和 DB 目录，设置当前机器的数据/地图/实验根目录，
并支持 `hdp` 与 `diffusion` 两种 planner。默认使用固定 3 场景和官方 non-reactive challenge。

HDP 监督：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
bash scripts/run_mini_closed_loop.sh \
  supervised-mini-val-3 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/latest.pth
```

HDP-RL：

```bash
bash scripts/run_mini_closed_loop.sh \
  rl-mini-val-3 \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-supervised-pilot-1000-lrfix/2026-07-31-23:43:06/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-rl-pilot-1000/2026-07-31-23:49:42/latest.pth
```

原始 Diffusion-Planner baseline：

```bash
bash scripts/run_mini_closed_loop.sh \
  diffusion-planner-mini-val-3 \
  /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/args.json \
  /home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth \
  mini-val-closed-loop-3 \
  diffusion
```

三模型汇总：

```bash
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/closed_loop_eval/exp/simulation/closed_loop_nonreactive_agents
$PY scripts/summarize_closed_loop_metrics.py \
  --run diffusion_planner="$ROOT/diffusion-planner-mini-val-3" \
  --run hdp_supervised="$ROOT/supervised-mini-val-3" \
  --run hdp_rl="$ROOT/rl-mini-val-3" \
  --output tmp/closed_loop_eval/mini_val_3_three_model_comparison.json
```

汇总器只保留 aggregator 中 `log_name` 非空的真实场景行，排除 scenario-type 汇总行和
`final_score` 行；同时从 `runner_report.parquet` 合并规划耗时和仿真成功状态。

### 9.6 三模型官方闭环结果

三个运行均为 `3/3` 成功、`0` 失败。

均值对比：

| 模型 | 官方 score | 无责任碰撞 | 路线进度 | TTC 合规 | 可行驶区 | 舒适 | 单次规划均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diffusion-Planner | 0.995829 | 1.000000 | 0.986696 | 1.000000 | 1.000000 | 1.000000 | 0.222596 s |
| HDP 监督 | 0.297406 | 0.333333 | 0.755465 | 0.333333 | 1.000000 | 1.000000 | 0.211528 s |
| HDP-RL | 0.297931 | 0.333333 | 0.758802 | 0.333333 | 1.000000 | 1.000000 | 0.222320 s |

逐场景 score：

| token / 类型 | Diffusion-Planner | HDP 监督 | HDP-RL |
|---|---:|---:|---:|
| `087642...` stationary_in_traffic | 1.000000 | 0.000000 | 0.000000 |
| `134f95...` high_magnitude_speed | 0.990394 | 0.000000 | 0.000000 |
| `263da8...` traffic_light_intersection | 0.997092 | 0.892217 | 0.893794 |

HDP-RL 相对 HDP 监督：

```text
score 绝对变化          +0.00052569
score 相对变化          +0.1768%
路线进度绝对变化        +0.00333681
规划耗时相对变化        +5.10%
碰撞/TTC 场景比例变化   0
```

两者在 stationary 和 high-speed 场景都发生责任碰撞，weighted score 被乘性安全指标归零；
traffic-light 场景均成功，RL 仅有非常小的路线进度提升。

### 9.7 结果解释

1. 原始 Diffusion-Planner 在同一 3 场景得到 `0.995829`，证明 DB、地图、controller、metrics、
   token 选择和机器环境本身可正常工作；HDP 的低分不能归因于仿真环境失效。
2. 原始 baseline checkpoint 是仓库发布的成熟 checkpoint，而 HDP 只有随机抽取的 1000 场景、
   10 epoch；二者训练数据和成熟度不相等，所以这张表是“当前工程基线”，不是公平架构优劣论文
   结论。
3. HDP 监督与 RL 的运行成功率、输入适配、轨迹输出和推理耗时都正常；最主要缺口是策略质量。
4. 当前离线 RL reward 只提升约 0.18% official score，且没有改善碰撞比例，不能把 val diffusion
   loss 的改善解释为闭环驾驶收益。
5. 因此下一步不应继续增加 RL epoch。应先扩大并改善监督训练起点，再以官方闭环安全指标作为
   checkpoint 门槛，然后重新进行短 RL。

### 9.8 产物与最终测试

核心对比结果：

```text
tmp/closed_loop_eval/mini_val_3_three_model_comparison.json
SHA-256: a829de02113f5e6d267ba07aeb264bd983f0c13633aa03b130bba1eb10abd523
```

每个正式运行目录都包含：

```text
aggregator_metric/*.parquet
metrics/*.parquet
runner_report.parquet
summary/summary.pdf
*.nuboard
code/hydra/config.yaml
code/hydra/overrides.yaml
log.txt
```

闭环评测目录总大小约 225 MiB。

新增 `tests/test_closed_loop_summary.py`，构造一个真实场景行和一个 `final_score` 聚合行，验证
汇总器只统计真实场景并正确合并 runtime。最终执行：

```bash
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
$PY -m py_compile \
  HDP-nuplan/scripts/summarize_closed_loop_metrics.py \
  HDP-nuplan/hdp_nuplan/planner/planner.py
bash -n HDP-nuplan/scripts/run_mini_closed_loop.sh
$PY -m pytest -q HDP-nuplan/tests
```

结果：

```text
5 passed in 3.23s
```

本阶段没有执行 `git add`、`git commit`、`git reset`、`git checkout` 或 `git clean`。所有闭环
配置、脚本、测试、日志和仿真产物仍作为工作区 changes 保留。

## 10. 当前项目状态与下一阶段门槛

现在项目已经具有完整可复现链路：

```text
NuPlan DB
  -> 官方日志划分
  -> NPZ 缓存
  -> 监督训练/EMA checkpoint
  -> 独立 open-loop validation
  -> offline reward-weighted diffusion RL
  -> RL 前后 validation
  -> NuPlan 官方 closed-loop simulation
  -> 三模型相同场景指标汇总
```

下一阶段建议从 10k 监督训练开始，而不是继续当前 1000 场景 checkpoint 的 RL：

1. train 扩大到 10k，并保证 44 个 train 日志全部覆盖；
2. val 扩大到 1k，并保证 10 个 val 日志全部覆盖；
3. 每个监督 checkpoint 先通过 open-loop val，再跑固定 3 场景；
4. 只有无责任碰撞比例明显高于 `1/3`，才进入 20 场景 closed-loop 和新一轮 RL；
5. RL 后必须同时满足 official score 不下降、碰撞比例不下降、open-loop val 不显著退化。

这套门槛的原因是当前最大问题已经由 official metrics 明确定位为碰撞，而不是代码能否运行。

## 11. 10k 阶段准备：encoder warm-start 与按日志均衡采样

### 11.1 为什么先改监督起点与采样方法

第 9 节已经证明 1000 场景 HDP 的主要问题是闭环碰撞，而不是软件链路中断。直接继续训练当前
RL checkpoint，无法解决监督起点弱、日志覆盖不均的问题。因此本阶段先做两项受控改动：

1. 只从仓库发布的成熟 Diffusion-Planner checkpoint 迁移结构完全兼容的 encoder；HDP decoder
   继续随机初始化，避免把语义和输出结构不同的 decoder 参数强行迁移。
2. 训练集和验证集先按日志分配基础配额，再随机补齐余数；这样 44 个 train 日志和 10 个 val
   日志都能被覆盖，且固定 seed 后可复现。

这两项改动不改变 HDP 的损失函数、decoder 结构或 RL 算法，只改善监督训练的数据覆盖与初始化。

### 11.2 encoder 兼容性审计

源 checkpoint：

```text
/home/yanjun/NewDisk/Diffusion-Planner/checkpoints/model.pth
```

审计结果：

| 项目 | 结果 |
|---|---:|
| HDP encoder 目标张量 | 151 |
| 名称和 shape 都兼容 | 151 |
| 可迁移参数量 | 1,799,040 |
| 缺失 encoder key | 0 |
| shape 不匹配 | 0 |
| 加载 decoder 张量 | 0 |

实现位置：

```text
hdp_nuplan/utils/train_utils.py
train_predictor.py
tests/test_encoder_warm_start.py
```

`load_encoder_warm_start()` 优先读取 `ema_state_dict`，兼容 `module.` 前缀，只接受
`encoder.*` 且名称、shape 完全一致的张量，并生成 `encoder_warm_start_report.json`。训练入口新增：

```text
--encoder_pretrained_model_path
--freeze_encoder_epochs
```

`--resume_model_path` 与 `--encoder_pretrained_model_path` 互斥，因为前者用于恢复完整训练状态，
后者只用于新实验的 encoder 初始化。

### 11.3 按日志均衡采样实现

实现位置：

```text
hdp_nuplan/data_process/sampling.py
data_process.py
tests/test_balanced_sampling.py
```

数据入口新增：

```text
--sampling_strategy {random,balanced_logs}
--sampling_report_path
```

`balanced_logs` 的确定性流程为：按最终 NPZ 文件名去重、按日志分组、每个日志先取
`total_scenarios // log_count` 条，再从剩余候选中按固定 seed 补齐余数。manifest 不再扫描目录中
可能存在的历史文件，而是只由本次选中的场景生成，并逐项检查预期 NPZ 是否落盘。

测试中曾出现一次导入冲突：测试导入顶层 `data_process.py` 时命中了父项目同名文件。处理方式是
把纯采样逻辑移动到包内 `hdp_nuplan/data_process/sampling.py`，让 CLI 和测试共同引用同一个实现。
修复后执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python -m pytest -q \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tests
```

结果：

```text
10 passed in 3.29s
```

### 11.4 44 日志真实均衡采样 smoke

运行目录：

```text
tmp/balanced_sampling_smoke_44/
```

核心参数：

```text
total_scenarios=44
sampling_strategy=balanced_logs
seed=3407
log_names_json=config/mini_splits/mini_train_logs.json
```

候选池共 306,801 条场景，最终 44 个训练日志各选 1 条。校验结果：manifest 44 条、唯一文件名
44 个、NPZ 44 个、日志覆盖 44/44、所有数值数组均为有限值。采样报告保存在：

```text
tmp/balanced_sampling_smoke_44/sampling_report.json
```

### 11.5 warm-start/冻结解冻训练 smoke

第一次启动失败于 normalization 路径：命令误写为
`hdp_nuplan/utils/normalization.json`，实际文件位于项目根目录 `normalization.json`。此时尚未初始化
训练。第二次使用相对数据路径后，数据集只读到 1 条，`drop_last=True` 导致 0 个 batch，并在
汇总空 loss 字典时触发 `KeyError: 'loss'`。这两次均未完成任何 epoch，失败目录保留用于复盘。

第三次将 normalization、cache 和 manifest 全部改为绝对路径，运行 3 epoch：

```text
batch_size=4
learning_rate=5e-4
warm_up_epoch=2
encoder_pretrained_model_path=checkpoints/model.pth
freeze_encoder_epochs=2
```

成功运行目录：

```text
tmp/balanced_sampling_smoke_44/training_log/hdp-balanced44-warmstart-smoke/
2026-08-01-00:35:26/
```

结果：

| epoch | encoder 状态 | train loss | 相对源 checkpoint 改变的 encoder 张量 |
|---:|---|---:|---:|
| 1 | frozen | 8.0108 | 0/151 |
| 2 | frozen | 6.8077 | 0/151 |
| 3 | trainable | 1.9122 | 151/151 |

第 3 轮 encoder 最大绝对参数变化为 `0.0053733736`。因此冻结阶段并非只修改日志字段，而是
checkpoint 中的 encoder 确实逐字节不变；解冻后参数也确实参与更新。warm-start 报告再次确认
加载 151/151 个 encoder 张量、1,799,040 个参数，decoder 加载数为 0。

### 11.6 正式缓存构建计划

在上述测试通过后，启动以下两个独立且无数据泄漏的缓存：

```text
train: tmp/mini_train_balanced_10000_seed3407_v1
       44 个官方 mini-train 日志，10,000 场景，seed=3407

val:   tmp/mini_val_balanced_1000_seed3407_v1
       10 个官方 mini-val 日志，1,000 场景，seed=3407
```

每个缓存完成后都必须检查：采样配额、日志覆盖、manifest 数量与唯一性、manifest/NPZ 精确对应、
必需字段、shape、dtype 和数值有限性。正式训练只有在这两份缓存全部通过校验后才开始。

为避免再次依赖一次性内联命令，新增：

```text
scripts/validate_processed_cache.py
tests/test_validate_processed_cache.py
```

该脚本还会检查 manifest 按名称排序、缓存目录不存在陈旧额外 NPZ、NPZ 中的实际日志计数与
`sampling_report.json` 完全相等，并把 manifest 与采样报告的 SHA-256 写入验收 JSON。新增两个
测试分别覆盖正常缓存和陈旧额外 NPZ 的拒绝路径。当前完整测试结果更新为：

```text
12 passed in 4.39s
```

为避免默认使用最后一个 epoch，新增 `evaluate_checkpoints.py`：在同一进程内复用 val
DataLoader，逐个严格加载 EMA state，记录 epoch/train loss/validation metrics，并按独立验证集
total loss 升序排名。计划先用一次重复扫描所有 epoch，再对排名靠前的候选做三次重复确认，最后才
进入固定 3 场景闭环门禁。新增排名测试后，完整测试更新为：

```text
13 passed, 1 warning in 6.18s
```

该 warning 是环境中 `timm.models.layers` 的既有弃用提示，不是本次逻辑失败。

### 11.7 train10k / val1k 实际构建与验收

train10k 使用 44 个官方 mini-train 日志、`balanced_logs`、seed 3407。预处理成功处理
`10,000/10,000` 条，DataProcessor 阶段耗时 `1:08:59`。严格验收结果：

| 项目 | 结果 |
|---|---:|
| manifest / unique / NPZ | 10,000 / 10,000 / 10,000 |
| 日志覆盖 | 44/44 |
| 每日志数量 | 227–229 |
| 缺失或额外 NPZ | 0 |
| 总字节数 | 1,574,851,408 |
| manifest SHA-256 | `1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514` |
| sampling report SHA-256 | `f6941f7f887ab3fb304b71362066af5a7c13af0905784ba07ce02654e6cb33bf` |

val1k 使用 10 个官方 mini-val 日志和相同 seed。预处理成功处理 `1,000/1,000` 条，耗时
`8:07`。严格验收结果：

| 项目 | 结果 |
|---|---:|
| manifest / unique / NPZ | 1,000 / 1,000 / 1,000 |
| 日志覆盖 | 10/10 |
| 每日志数量 | 100 |
| 缺失或额外 NPZ | 0 |
| 总字节数 | 157,491,232 |
| manifest SHA-256 | `051e06f59464f5249dceff4f511071ce851062f0516ad531e6aaf47f5dddc924` |
| sampling report SHA-256 | `269b77697ef54c392e9a1f30ab5d9321596aee39bd79022dccbb1965d323f72f` |

完整报告分别位于：

```text
tmp/mini_train_balanced_10000_seed3407_v1/cache_validation_report.json
tmp/mini_val_balanced_1000_seed3407_v1/cache_validation_report.json
```

两份报告均为 `status: passed`，且逐 NPZ 验证了默认 HDP 张量 shape 和有限值。因此 train 与
val 不仅在日志列表上隔离，最终落盘文件也分别经过可重复验收。

## 12. train10k 完整训练、选模与闭环验收

本节记录从已验收的 train10k/val1k 缓存开始，到监督训练、RL 微调、固定场景闭环对比和最终
模型决策的完整过程。所有 Python 命令均使用原 Diffusion-Planner 环境：

```text
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
```

### 12.1 监督训练配置与运行结果

训练入口为 `train_predictor.py`，训练数据为 44 个 mini-train 日志均衡采样得到的 10,000 条
场景。关键配置如下：

| 参数 | 值 |
|---|---:|
| seed | 3407 |
| train epochs | 20 |
| batch size | 8 |
| learning rate | 5e-4 |
| warm-up epochs | 2 |
| encoder source | 根目录发布版 Diffusion-Planner `checkpoints/model.pth` |
| frozen encoder epochs | 3 |
| EMA | true |
| checkpoint interval | 1 epoch |

运行目录：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/
```

20 个 epoch 全部完成并保存 checkpoint，最后一轮训练 loss 为 `0.1046`。参数检查证明：第 3 轮
checkpoint 的 151 个 encoder 张量与源 checkpoint 全部完全一致；第 4 轮解冻后 151/151 个张量
均已改变，最大绝对变化为 `0.0737883`。这验证了正式训练中的冻结和解冻路径确实生效。

监控训练时曾因 tqdm 输出积压，把终端显示的旧 epoch 误认为训练停滞。后续改用 checkpoint 的
mtime、文件数量和进程状态交叉判断，确认训练一直在正常前进。该经验说明：长任务不能只依据被
截断或延迟刷新的进度条判断状态。

### 12.2 用独立 val1k 选 checkpoint

使用 `evaluate_checkpoints.py` 对 20 个 EMA checkpoint 在同一个 val1k DataLoader 上先各评估
1 次，再对前三名各重复 3 次。一次扫描的前八名为：

| 排名 | epoch | val total loss |
|---:|---:|---:|
| 1 | 10 | 0.114719564 |
| 2 | 5 | 0.117446960 |
| 3 | 6 | 0.118460128 |
| 4 | 9 | 0.120081278 |
| 5 | 7 | 0.124292410 |
| 6 | 18 | 0.125227629 |
| 7 | 8 | 0.125693097 |
| 8 | 20 | 0.127031341 |

前三名三次重复后的均值为：epoch 10=`0.119260203`、epoch 5=`0.121007727`、
epoch 6=`0.122915955`，因此正式选中 epoch 10，而不是默认采用最后一轮。

选中 checkpoint：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/
model_epoch_10_trainloss_0.1380.pth
```

checkpoint SHA-256：

```text
6902eb677c94a33287fc496878b1783d719fe187c7c45abdfe5d9a5ff7269258
```

对照旧 1k pilot checkpoint，在同一 val1k 上重复 3 次：旧监督模型为 `0.700395189`，旧 RL
模型为 `0.663651541`，新监督模型相对旧监督模型降低 `82.9724%`。排名与复测证据位于：

```text
tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_ranking_repeat1.json
tmp/mini_val_balanced_1000_seed3407_v1/checkpoint_evaluation_epoch10_repeat3.json
```

两者 SHA-256 分别为：

```text
781618094f2930bbb0ebf2bdde659ddb50b13037e98a0de1ab30d0a4d2f928b3
f6e28afabad198f206d7c4556ae046ef2958e0e30c814387d497f8697a4f1142
```

### 12.3 固定 3 场景闭环门禁

先在与旧模型、发布版 Diffusion-Planner 完全相同的 3 个 mini-val 场景上运行官方
`closed_loop_nonreactive_agents`。新监督模型 3/3 成功、0 失败，结果为：

| 模型 | score | collision | TTC | progress |
|---|---:|---:|---:|---:|
| 发布版 Diffusion-Planner | 0.995829 | 1.000000 | 1.000000 | 0.986696 |
| 旧 HDP 监督 1k | 0.297406 | 0.333333 | 0.333333 | 0.755465 |
| 旧 HDP RL 1k | 0.297931 | 0.333333 | 0.333333 | 0.758802 |
| 新 HDP 监督 10k epoch 10 | 0.947107 | 1.000000 | 1.000000 | 0.830792 |

新模型消除了旧模型在该固定集合上的碰撞和 TTC 失败，达到进入更大闭环集合的门禁条件。四模型
对比证据：

```text
tmp/closed_loop_eval/mini_val_3_four_model_comparison.json
SHA-256: 390c4cbb285d80e2e6580e356e3719829adc2329eec3683d260ddf44a7f1b0d1
```

### 12.4 构造固定 20 场景集合

为了比 3 场景 smoke 更稳定，同时保持本地项目成本可控，构造固定 20 场景集合。规则为：保留原
固定 3 场景；候选来自 val1k 对应的 10 个官方 mini-val 日志；使用 NuPlan 官方
`remove_invalid_goals` 逻辑验证 mission goal；同一日志内起始时间至少间隔 20 秒；seed 固定为
3407，并尽量在有效日志间均衡。

构造过程中出现过三次有效失败，均保留在思路记录中：

1. 仅按 `route_lanes` 非零判断时，某日志没有候选，最终只能得到 18 条；
2. 首版 20 token 中只有 12 条通过官方 mission-goal 校验，证明“缓存 route 非零”不等于
   “ScenarioBuilder 可用于闭环”；
3. 只验证 val1k manifest 时漏掉了固定 3 场景，因为固定 token 不一定被 val1k 抽样选中；最终
   改为“固定 3 + val1k 候选”的并集后再执行统一校验。

最终集合 20 个 token 全部唯一、20/20 可由 ScenarioBuilder 构建、20/20 mission goal 有效，
覆盖 8 个有效日志且满足日志内最小时间间隔。10 个 val 日志中有 2 个在该规则下没有合格候选，
因此没有强行凑成每日志覆盖。

配置和完整选择审计分别为：

```text
hdp_nuplan/config/scenario_filter/mini-val-closed-loop-20.yaml
config/mini_splits/mini_val_closed_loop_20_selection.json
```

两种模型运行时均在相同 3 条场景上出现 `route_list empty` 警告，但所有 runner 均完成且指标文件
齐全。因此该警告被记录为数据/route 提取现象，不被误判为某一模型独有失败。

### 12.5 新 HDP 与发布版 Diffusion-Planner 的同场景 20 次闭环对比

新监督 HDP 运行目录：

```text
tmp/closed_loop_eval/exp/simulation/closed_loop_nonreactive_agents/
hdp-balanced10k-epoch10-mini-val-20/
```

运行 20/20 成功、0 失败，耗时约 14 分 23 秒。随后在相同 token、相同 challenge 和相同指标
配置下运行发布版 Diffusion-Planner，20/20 成功、0 失败，耗时约 14 分 52 秒。聚合结果如下：

| 指标 | 发布版 Diffusion-Planner | HDP 10k epoch 10 | HDP - DP |
|---|---:|---:|---:|
| overall score | 0.881446 | 0.787373 | -0.094072 |
| no collision | 0.900000 | 0.950000 | +0.050000 |
| TTC within bound | 0.900000 | 0.900000 | 0.000000 |
| expert-route progress | 0.960774 | 0.668771 | -0.292004 |
| drivable-area compliance | 0.950000 | 1.000000 | +0.050000 |
| comfort | 0.950000 | 1.000000 | +0.050000 |
| mean planner runtime/s | 0.232155 | 0.227081 | -0.005074 |

这说明当前 HDP 已不是旧 pilot 中“频繁碰撞”的状态：它在 collision、drivable area 和 comfort
上略好，TTC 持平，但明显更保守，路线进度比发布版低 `0.292004`，从而使总分低 `0.094072`。
对这个项目而言，下一轮优化目标应是“在不损失安全性的前提下恢复 progress”，而不是继续只加大
碰撞惩罚。

完整同场景对比：

```text
tmp/closed_loop_eval/mini_val_20_diffusion_vs_hdp10k.json
SHA-256: 4b3682fb51ce6c2878996e462fb35a383edef648ef6c549db2863193d2781e40
```

### 12.6 train10k RL 微调与拒绝决策

RL 从已经选中的监督 epoch 10 启动，而不是从架构不兼容的发布版完整 checkpoint 启动。关键参数：

| 参数 | 值 |
|---|---:|
| group size | 4 |
| rollout steps | 5 |
| epochs | 5（第 1 轮收集，后 4 轮更新） |
| batch size | 8 |
| replay buffer | 10,000 |
| learning rate | 1e-5 |
| freeze encoder | true |
| progress / collision / route / comfort weights | 1 / 10 / 1 / 0.01 |

运行目录：

```text
tmp/mini_train_balanced_10000_seed3407_v1/training_log/
hdp-mini-rl-balanced10k-from-epoch10/2026-08-01-02:57:57/
```

任务正常退出，最终 replay buffer 为 10,000，最后更新轮总 loss=`0.009807715`、reward mean=
`2.17165795`。这证明迁移后的离线 diffusion RL 训练链路能够在 NuPlan 缓存上完整运行。

但是“训练跑通”不等于“模型被接受”。四个 RL checkpoint 在 val1k 上重复 3 次的结果为：

| epoch | val total loss |
|---:|---:|
| 2 | 0.153701405 |
| 3 | 0.193612728 |
| 4 | 0.213826350 |
| 5 | 0.221683908 |

最佳 RL epoch 2 相比监督 epoch 10 的 `0.119260203` 退化 `28.8790%`。继续将最佳 RL 候选送入
固定 3 场景闭环，3/3 成功，但 score 从 `0.947106568` 降到 `0.939914612`（下降
`0.7594%`），progress 从 `0.830792377` 降到 `0.807774850`，collision/TTC 均保持 1.0。

因此按照预先设定的两级门禁，拒绝 RL checkpoint，不再为已失败候选消耗一次 20 场景评测成本。
最终交付模型仍是监督 epoch 10。被拒绝的最佳 RL checkpoint 仍保留，便于复盘：

```text
model_epoch_2_trainloss_0.0195.pth
SHA-256: 5c56796dd16fc1f83a68d77f591dfae2d9ad79eea76cb41ee79397ebc3caef7f
```

RL 排名与固定 3 场景对比证据：

```text
tmp/mini_val_balanced_1000_seed3407_v1/rl_checkpoint_ranking_repeat3.json
SHA-256: fd3a5b3ed630f139bfa2750ff14dc80d63dc5a7a18df7b5b77ab1c0852d40084

tmp/closed_loop_eval/mini_val_3_hdp10k_supervised_vs_rl.json
SHA-256: f2da849185ec2055cf74909a56f3e1764703bc6565cea9458f89aa94d9751ce9
```

### 12.7 当前项目结论

截至本轮，NuPlan 迁移已经形成一条完整、可复盘的工程链路：

```text
官方 mini 日志划分
→ 按日志均衡预处理并严格验收缓存
→ 发布版 DP encoder warm-start
→ HDP 监督训练
→ 独立 val1k checkpoint 选择
→ 固定 3 场景闭环门禁
→ 固定 20 场景同场景基线对比
→ NuPlan 离线 diffusion RL 微调
→ val + closed-loop 两级接受/拒绝决策
```

最终接受：`HDP 监督 epoch 10`。最终拒绝：`HDP RL epoch 2–5`。项目成功证明了 NAVSIM 风格的
reward-weighted diffusion RL 训练设计可以迁移到 NuPlan 数据接口并完整运行；同时实验也如实说明，
当前离线近似 reward 没有带来闭环收益。这个负结果不是迁移失败，而是通过验证集和闭环门禁避免
把退化模型当成改进模型。

当前最明确的模型短板是闭环 progress 偏低。若继续迭代，应优先改进 reward 与 NuPlan 官方
闭环 progress/route 指标的一致性，再重新训练；不建议在当前 reward 下盲目增加 RL epoch。

### 12.8 最终代码与配置校验

收尾时补充了 checkpoint 文件名的自然排序，避免字典序把 epoch 10 排在 epoch 2 前面；同时新增
固定 20 场景配置测试，持续检查 token 数量、唯一性、固定 3 场景包含关系、有效 goal 和日志内
最小时间间隔。使用指定 conda 环境执行：

```bash
python -m py_compile \
  train_predictor.py train_predictor_rl.py evaluate_predictor.py \
  evaluate_checkpoints.py scripts/validate_processed_cache.py \
  scripts/summarize_closed_loop_metrics.py \
  hdp_nuplan/utils/train_utils.py hdp_nuplan/data_process/sampling.py

bash -n scripts/run_mini_closed_loop.sh
python -m json.tool \
  config/mini_splits/mini_val_closed_loop_20_selection.json >/dev/null
python -m pytest -q tests
```

最终结果为 `15 passed, 1 warning in 3.67s`；warning 来自环境内 timm 的既有弃用提示。Python
静态编译、Shell 语法和 JSON 解析均通过。整个过程中未执行 `git add`、`git commit`、`git reset`
或 `git clean`，所有迁移代码、配置、文档和实验产物仍保留为工作区 changes。

### 12.9 后续扩散参数化与 Detached Integral 审计（2026-08-08）

代码逐行复盘时完成两项修正：

1. `VPSDE_linear.transform()` 对 `src == tgt` 直接返回输入，避免
   `x_start -> noise -> x_start` 等恒等转换因两次 `1e-6` 分母产生额外计算和微小数值误差。
2. `detached_integral()` 接收的实际张量为 `[B*G,T,2]`，原实现使用
   `shifted[:, :, :detach_window_size]`，误把最后一个特征维当成时间维；默认窗口 10 会把两个
   shifted 张量全部清零，使反向传播退化为普通完整 `cumsum`。现改为
   `shifted[..., :detach_window_size, :]`，同样修正无梯度前缀张量，从而明确操作倒数第二个时间维。

新增梯度回归测试同时验证：

- `detached_integral(u, W)` 的前向结果与 `torch.cumsum(u, dim=-2)` 一致；
- 只对最后一个积分位置反向传播时，梯度仅落在最近 `W` 个输入时间步。

使用指定环境执行：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests/test_rl_components.py
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果分别为 `4 passed` 和 `16 passed, 1 warning`，warning 仍为 timm 的既有弃用提示。该修复不改变
Detached Integral 的前向轨迹数值，但会改变 hybrid waypoint loss 的梯度路径；此前已经完成的
监督/RL 实验使用的是修复前的完整累积梯度，历史结果不受文件修改追溯影响，若要评价修复后的训练
效果需要重新训练。上述修改和测试仍保留为工作区 changes，未提交。

### 12.10 NuPlan reward v2、1000 场景受控 RL pilot 与拒绝决策（2026-08-08）

#### 12.10.1 为什么修改 reward

12.6 的第一版 RL 虽然能完整运行，但 reward 与 NuPlan 官方闭环指标存在明显语义差距：碰撞仅按
中心点距离判断，progress/backward 只使用局部坐标系的 x 方向，comfort 没有接入当前 ego 速度和
加速度。继续扩大 epoch 或数据量不能修复这些定义问题，因此先修改 reward，再做小规模门禁。

reward v2 的改动如下：

1. 动态车辆和静态物体使用带朝向、长宽的 OBB 分离距离；`reward_collision_distance=0.5` 的语义
   改为 OBB 外额外 0.5 m 安全余量。邻车尺寸来自 `neighbor_agents_past[..., 6:8]`，静态物体尺寸
   来自 `static_objects[..., 4:6]`，缺失时使用配置默认尺寸。
2. progress 和 backward 由轨迹相邻点位移在最近 route 点切向量上的投影计算，不再把局部 x
   方向等同于路线前进方向；progress 除以 10 做尺度归一化。
3. comfort 在预测轨迹前拼接当前原点，并使用 `ego_current_state[..., 4:8]` 中的当前速度、加速度
   作为差分边界；各项超限值裁剪到 5，默认 comfort 权重保持 0.01，避免 jerk 主导总奖励。
4. rollout 调用增加 `neighbor_agents_past` 和 `ego_current_state`，但 replay 数据结构和扩散 RL loss
   不变。参数名 `reward_collision_distance` 为兼容已有命令保留，含义已在代码注释中明确。

涉及文件：

```text
hdp_nuplan/rl/reward.py
hdp_nuplan/rl/train_epoch_rl.py
train_predictor_rl.py
tests/test_rl_components.py
scripts/validate_reward_v2.py
```

#### 12.10.2 单元测试与真实 NPZ 定向验证

新增测试覆盖：OBB 尺寸敏感碰撞、route 切向进度、当前状态参与 comfort、comfort 裁剪以及
Detached Integral 梯度窗口。完整测试命令：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果为 `19 passed, 1 warning in 3.87s`；warning 仍是 timm 的既有弃用提示。

随后使用 100 个真实 NuPlan NPZ 构造 expert、stop、lateral、reverse、jitter 和 collision 六类
轨迹，运行：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  scripts/validate_reward_v2.py \
  --cache-dir tmp/mini_train_smoke_100_v2/cache \
  --manifest tmp/mini_train_smoke_100_v2/diffusion_planner_training.json \
  --output tmp/mini_train_smoke_100_v2/reward_v2_validation_100.json \
  --max-scenes 100 --minimum-pass-rate 0.8 --strict
```

第一次实现对全部场景强制检查 reverse/stop 排序，结果没有达到阈值。逐场景核查发现失败主要来自
红灯、拥堵等专家最终位移不足 1 m 的近静止场景：对近静止轨迹坐标取反并不能构造有意义的逆行
样本，专家也不应被强制判定优于停车。修正验证协议后，仅在最终位移至少 1 m 的 66 个 moving
scene 上检查 reverse/stop；阈值仍保持 80%，不是降低验收标准。严格验证最终通过：

| 检查项 | 通过/总数 |
|---|---:|
| lateral 增大 route cost | 98/100 |
| jitter 增大 comfort cost | 100/100 |
| reverse 降低 progress（moving） | 66/66 |
| reverse 增大 backward cost（moving） | 66/66 |
| expert reward 高于 stop（moving） | 66/66 |
| collision 增大 collision cost | 99/100 |

报告：

```text
tmp/mini_train_smoke_100_v2/reward_v2_validation_100.json
SHA-256: bf49083166bae893e760310ab92b15eefe01d26d1e87478e5c669b78b7220b7f
```

#### 12.10.3 100 场景端到端 smoke

先用已有 100 场景缓存确认 reward v2、修复后的 Detached Integral、replay 和 checkpoint 保存能
共同工作。该 smoke 的监督起点只是早期 100 场景模型，只用于软件验证：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-rl-smoke-reward-v2-detachfix \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v2 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v2/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v2/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v2/training_log/hdp-mini-supervised-smoke/2026-07-31-23:28:29/latest.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 4 --num_workers 0 --train_epochs 2 --warm_up_epoch 1 \
  --save_utd 1 --learning_rate 1e-5 --rl_group_size 4 --rl_rollout_steps 2 \
  --rl_buffer_update_epoch 5 --rl_buffer_size 128 --rl_freeze_encoder true \
  --reward_comfort_weight 0.01 --reward_collision_distance 0.5
```

任务成功退出，rollout buffer=`100`，更新 loss=`0.06531676`，无 NaN、无 OOM，checkpoint 成功
保存。由于早期监督起点质量很弱，rollout 的 no-collision 仅为 0.13；该结果只证明软件链路通过，
不能评价 reward 或模型质量。运行目录：

```text
tmp/mini_train_smoke_100_v2/training_log/
hdp-mini-rl-smoke-reward-v2-detachfix/2026-08-08-11:29:47/
```

#### 12.10.4 1000 场景、5 步 rollout 受控 pilot

正式 pilot 固定使用 12.2 已接受、且完成 val1k 与闭环门禁的监督 epoch 10 作为唯一变量控制起点；
训练数据使用已有 1000 场景 manifest，避免 reward 尚未过门禁时直接消耗 10k 训练成本：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-rl-pilot-1000-reward-v2-detachfix-from-10k-epoch10 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 --num_workers 2 --train_epochs 5 --warm_up_epoch 1 \
  --save_utd 1 --learning_rate 1e-5 --rl_group_size 4 --rl_rollout_steps 5 \
  --rl_buffer_update_epoch 5 --rl_buffer_size 1024 --rl_freeze_encoder true \
  --reward_comfort_weight 0.01 --reward_collision_distance 0.5
```

第 1 轮遍历全部 1000 场景并收集 1000 条 replay；后 4 轮各执行 125 个 update batch。rollout
均值为 reward=`0.646413`、progress=`2.293903`、no-collision=`0.87375`、collision cost=
`0.146750`、route cost=`0.164487`、comfort cost=`1.412167`。各更新轮 total loss 为：

| epoch | total loss |
|---:|---:|
| 2 | 0.0480465 |
| 3 | 0.0222（checkpoint 文件四舍五入值） |
| 4 | 0.0185677 |
| 5 | 0.0168292 |

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-pilot-1000-reward-v2-detachfix-from-10k-epoch10/2026-08-08-11:31:39/
```

#### 12.10.5 val1k 与官方闭环两级门禁

使用 `evaluate_checkpoints.py` 在固定 val1k、seed 3407–3409 上重复评估三次。四个 RL checkpoint
结果：

| epoch | val total loss |
|---:|---:|
| 2 | 0.118971 |
| 3 | 0.119678 |
| 4 | 0.121728 |
| 5 | 0.124878 |

同一当前代码下重测监督 epoch 10 为 `0.119260`。因此仅 RL epoch 2 在 open-loop 上有约 0.24%
的微小改善，后续更新已经逐步退化。证据：

```text
tmp/mini_val_balanced_1000_seed3407_v1/rl_reward_v2_detachfix_pilot1000_ranking_repeat3.json
SHA-256: fc3b391ce4873f23a656e24665d8e4cfc96372c942a8f3968673237d0d775df6

tmp/mini_val_balanced_1000_seed3407_v1/supervised_epoch10_currentcode_repeat3.json
SHA-256: d8249e23bda33dfd946d1bc6b756832098ebee9db5c3030152bca6f98b0c9d12
```

将唯一候选 epoch 2 送入固定 3 场景 `closed_loop_nonreactive_agents`，3/3 仿真成功、0 失败：

```text
model_epoch_2_trainloss_0.0480.pth
SHA-256: 07a1fd791d8be63ace3f0a6b578f3e42bb09094fd595edc462706cb3ef7f33c9
```

| 指标 | 监督 epoch 10 | reward v2 RL epoch 2 | RL - 监督 |
|---|---:|---:|---:|
| overall score | 0.947107 | 0.936489 | -0.010618 |
| expert-route progress | 0.830792 | 0.796813 | -0.033979 |
| no collision | 1.000000 | 1.000000 | 0 |
| TTC within bound | 1.000000 | 1.000000 | 0 |
| comfort | 1.000000 | 1.000000 | 0 |

RL 的 score 相对下降约 1.12%，progress 相对下降约 4.09%，三个场景的 progress 都低于监督
模型。离线 reward 中的 progress 改为 route 投影后方向语义更合理，但它仍使用专家未来邻车和静态
场景近似，不包含 simulator 闭环交互；当前配置仍然使模型更保守，未解决 12.5 已识别的 progress
短板。因此拒绝该 RL checkpoint，不继续运行 10k RL，也不消耗固定 20 场景评测成本。最终接受
模型仍是监督 epoch 10。

闭环运行目录与汇总证据：

```text
tmp/closed_loop_eval/exp/simulation/closed_loop_nonreactive_agents/
hdp-rl-reward-v2-pilot1000-epoch2-mini-val-3/

tmp/closed_loop_eval/mini_val_3_supervised_vs_rl_reward_v2_pilot1000.json
SHA-256: 1a4605e620e44e03dae4231ff03d60bf98e077927feb2483b87cad4a8aeccbd2
```

本轮结论不是“reward v2 无效”，而是：几何碰撞、路线方向和 comfort 边界定义已通过定向测试，
训练也稳定；但当前离线 reward-weighted fine-tuning 没有通过 NuPlan 官方闭环收益门禁。下一步若
继续改进，应优先把 rollout/reward 接到 NuPlan simulator 状态转移或官方 metric proxy，而不是
扩大当前离线 RL 的数据量或 epoch。全过程未执行提交，代码、日志和实验产物均保留为 changes。

### 12.11 监督/RL checkpoint 同噪声行为归因（2026-08-08）

12.10 的闭环结果只能说明 RL 退化，不能区分 reward 错位、数据过拟合、EMA 保存问题或更新目标
问题。因此新增 `scripts/compare_checkpoint_behavior.py`，在同一 batch 的两个模型采样前重置相同
随机种子，使两边接收完全相同的初始扩散噪声，并计算 reward 分项、路径长度、速度、停车比例、
加速度、jerk、ADE/FDE、组内候选差异和逐场景 paired delta。

核心命令使用以下固定路径；train/val 和采样协议只修改对应参数：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
PY=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
ARGS=tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-rl-pilot-1000-reward-v2-detachfix-from-10k-epoch10/2026-08-08-11:31:39/args.json
SUP=tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth
RL=tmp/mini_train_pilot_1000_seed3407_v1/training_log/hdp-mini-rl-pilot-1000-reward-v2-detachfix-from-10k-epoch10/2026-08-08-11:31:39/model_epoch_2_trainloss_0.0480.pth
VAL=tmp/mini_val_balanced_1000_seed3407_v1

$PY scripts/compare_checkpoint_behavior.py \
  --args-file "$ARGS" --supervised-checkpoint "$SUP" --rl-checkpoint "$RL" \
  --data-dir "$VAL/cache" --data-list "$VAL/diffusion_planner_validation.json" \
  --batch-size 16 --num-workers 2 --repeats 3 --seed 3407 \
  --num-samples 4 --diffusion-steps 5 --device cuda \
  --output "$VAL/behavior_supervised_vs_rl_reward_v2_rollout5_group4_repeat3.json"

$PY scripts/compare_checkpoint_behavior.py \
  --args-file "$ARGS" --supervised-checkpoint "$SUP" --rl-checkpoint "$RL" \
  --data-dir "$VAL/cache" --data-list "$VAL/diffusion_planner_validation.json" \
  --batch-size 16 --num-workers 2 --repeats 3 --seed 3407 \
  --num-samples 1 --diffusion-steps 10 --device cuda \
  --output "$VAL/behavior_supervised_vs_rl_reward_v2_inference10_single_repeat3.json"
```

train1k 使用相同命令，将 data-dir/data-list 替换为
`tmp/mini_train_pilot_1000_seed3407_v1/cache` 和其中的 training manifest。即时模型对照额外传入
`--supervised-weight-source ema_state_dict --rl-weight-source model`。

首先运行独立 val1k 三次重复。RL rollout 协议 `4 candidates × 5 steps` 的主要结果：

| 指标 | 监督 epoch 10 | RL epoch 2 EMA | RL - 监督 |
|---|---:|---:|---:|
| reward | 1.245572 | 1.141913 | -0.103659 |
| progress | 2.311822 | 2.256419 | -0.055403 |
| path length/m | 23.456154 | 22.902174 | -0.553980 |
| mean speed/m/s | 2.932019 | 2.862772 | -0.069248 |
| low-speed step fraction | 0.487184 | 0.492796 | +0.005611 |
| collision cost | 0.082535 | 0.087394 | +0.004858 |
| comfort cost | 1.398125 | 1.421943 | +0.023819 |
| ADE/m | 2.382748 | 2.550254 | +0.167506 |

闭环 planner 协议 `1 trajectory × 10 steps` 同样三次重复：

| 指标 | 监督 epoch 10 | RL epoch 2 EMA | RL - 监督 |
|---|---:|---:|---:|
| reward | 1.313272 | 1.191847 | -0.121425 |
| progress | 2.358922 | 2.292237 | -0.066685 |
| path length/m | 23.921527 | 23.252980 | -0.668547 |
| mean speed/m/s | 2.990191 | 2.906623 | -0.083568 |
| low-speed step fraction | 0.494333 | 0.500954 | +0.006621 |
| collision cost | 0.080289 | 0.085839 | +0.005549 |
| comfort cost | 1.390425 | 1.412445 | +0.022020 |
| ADE/m | 2.202330 | 2.382100 | +0.179770 |

10 步协议中，RL 仅在 `0.1%` 的配对轨迹上提高 progress、仅在 `1.77%` 上提高 reward。5 步和
10 步退化方向一致，因此采样步数差异不是主因。

随后在实际 RL train1k 上运行相同 `4×5` 三次重复。reward 从 `0.643565` 降到 `0.480735`，
progress 从 `2.294749` 降到 `2.239211`，path length 从 `23.248677 m` 降到 `22.699921 m`。
train 和 val 同时退化，排除了“只在独立验证集过拟合”的解释。

第一次加入 group 指标时，报告末尾相关性计算失败：候选级 reward 有 `B*G` 个值，场景级 group
指标只有 `B` 个值，NumPy 拒绝拼接不同长度数组。采样和指标计算未受影响，但 JSON 没有落盘。
修复为不同统计层级不计算 Pearson correlation，并新增回归测试后重跑成功。

组内诊断发现 train1k 监督候选的 reward 标准差中位数只有 `0.007111`，progress 标准差中位数
`0.006437`，候选终点两两距离中位数仅 `0.098388 m`。候选差异很小，但当前
`group_advantage_weights()` 除以每组自身标准差后再执行 `exp(advantage)`，使非常微弱的差异也能
得到与大差异相似的相对权重；实际训练 weight mean 约 `1.59`。

reward 排序本身没有偏好更短轨迹：train1k 组内 reward-progress 相关系数中位数为 `0.998305`，
最高 reward 候选相对组均值的 progress 平均增加 `0.017117`，path length 平均增加
`0.167490 m`，collision cost 平均降低 `0.003948`。因此最终模型变短不是 reward 直接选择停车，
而是加权自蒸馏没有稳定学习这种很小的候选差异。

最后强制加载 RL checkpoint 的即时 `model` 而非 EMA：train1k reward=`-0.804320`、path length=
`18.067617 m`，比 RL EMA 的 `0.480735` 和 `22.699921 m` 更差。EMA 只是平滑并减轻了更新漂移，
不是退化根因。

本轮将下一步优先级从“接 simulator reward”修正为“先修 RL update”：小 reward 方差组不加权、
组内权重归一化为均值 1、加入真实专家轨迹监督 anchor，并建立 10–20 update step 的快速门禁。
如果离线 train reward 都不能提高，接入更昂贵的 simulator-in-the-loop 没有意义。

详细结论整理在：

```text
doc_hdp_nuplan/RL_checkpoint_behavior_diagnosis.md
```

主要证据 SHA-256：

```text
bea3693dfa0744511ea2eccc78adb619794dd45b8451225a9d8e9193c27cd24c  val1k 4×5 repeat3
e70dc48deb4758a0db41acba347f72f01ef1441629f13cd35fb7347c9e0c257e  val1k 1×10 repeat3
437bca111fab174bb9678b0134f689c5b3456d1931c249ba425b97c292400752  train1k 4×5 repeat3
a48a156c013ba754efdac59f45653f49f72cbd00a64bfc8c041a8143bcd17193  train1k group preference
c728c72077a9c8cc484879564c47577a3e7aca433c686846f98b18bc91329c45  train1k RL model vs EMA baseline
```

### 12.12 RL update v2 与 20-step 短步门禁（2026-08-08）

根据 12.11 的归因结果，本轮没有直接扩大 RL 训练，而是先修正 update：

1. 组内 reward 标准差小于 `0.01` 时，该组 rollout 权重置零，避免放大近乎重合候选的数值差异；
2. 有效组的 `exp(advantage)` 权重按组归一化到均值 1，固定有效学习率；
3. 总 loss 加入权重 `0.1` 的真实 `ego_future` 监督扩散 anchor；
4. 增加 `rl_max_update_steps_per_epoch`，本轮只允许 20 个 optimizer step；
5. 增加 `rl_sampling_noise_scale`，单独控制 RL 候选多样性，不改变普通 planner 的 0.1 默认值。

对应修改位于：

```text
hdp_nuplan/rl/loss.py
hdp_nuplan/rl/train_epoch_rl.py
hdp_nuplan/model/hyper_diffusion_planner.py
hdp_nuplan/model/module/decoder.py
train_predictor_rl.py
scripts/compare_checkpoint_behavior.py
tests/test_rl_components.py
```

#### 12.12.1 noise=0.1 的第一次门禁

训练命令的完整可复现形式如下。历史第一次运行发生在参数公开到 CLI 之前，使用的是采样函数默认
`noise_scale=0.1`；下列命令显式写出该值，语义相同：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-rl-update-v2-gate20-from-10k-epoch10 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 --num_workers 2 --train_epochs 2 --warm_up_epoch 1 \
  --save_utd 1 --learning_rate 1e-5 --rl_group_size 4 --rl_rollout_steps 5 \
  --rl_sampling_noise_scale 0.1 --rl_buffer_update_epoch 5 --rl_buffer_size 1024 \
  --rl_freeze_encoder true --rl_min_reward_std 0.01 --rl_normalize_weights true \
  --rl_expert_anchor_weight 0.1 --rl_max_update_steps_per_epoch 20 \
  --reward_comfort_weight 0.01 --reward_collision_distance 0.5
```

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-update-v2-gate20-from-10k-epoch10/2026-08-08-14:54:54/
```

该门禁 checkpoint 的 SHA-256 为
`064d5eeb58ac7c8b42745e97e4fb50a62964656d169053261eff9450ba24fa09`。

rollout 与旧实验相同；20 次更新的 total loss=`0.0479703`、RL loss=`0.0417714`、expert anchor
loss=`0.0619890`。参与 RL loss 的场景组比例为 `0.48125`，有效组平均权重为 1。该次运行早于
`update_steps` 日志修复，汇总中显示的 `10.5` 是对步骤序号错误求均值，实际执行了 20 步。

使用 `compare_checkpoint_behavior.py`，在 train1k/val1k 上分别用共同随机数、4 candidates、5 steps、
3 repeats 比较监督起点和新 checkpoint：

| 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 |
|---|---:|---:|---:|---:|
| train1k | -0.006054 | -0.002522 | -0.025493 m | +0.008956 m |
| val1k | -0.004725 | -0.002534 | -0.025700 m | +0.007946 m |

虽然退化远小于旧 RL，但 reward 未转正，因此第一次门禁拒绝。

#### 12.12.2 提高候选多样性并进行第二次门禁

先只对监督模型运行 group diagnostics。将 noise 从 0.1 提高到 0.2 后，reward 标准差中位数由
`0.007111` 提高到 `0.014335`，终点两两距离中位数由 `0.098388 m` 提高到 `0.197904 m`；因此
保留其他配置，只将 `--rl_sampling_noise_scale` 改为 `0.2` 重跑上述训练命令，name 改为：

```text
hdp-mini-rl-update-v2-noise02-gate20-from-10k-epoch10
```

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-update-v2-noise02-gate20-from-10k-epoch10/2026-08-08-14:59:35/
```

该门禁 checkpoint 的 SHA-256 为
`1ea298958724eb63692944b63b554be4519a55a826875af867452fe221137a25`。

rollout reward=`0.682813`、progress=`2.308244`；20 次更新的 total loss=`0.0456518`、RL loss=
`0.0394467`、expert anchor loss=`0.0620518`。active group fraction=`0.525`、active weight mean=`1`、
reward std mean=`0.198923`，步骤计数修复后正确记录 `update_steps=20`。

第一次启动行为评估时误写 checkpoint 时间目录为 `14:58:29`，立即得到 `FileNotFoundError`，没有
加载模型、没有生成不完整报告，也没有影响训练产物。改为实际目录 `14:59:35` 后重跑成功：

| 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 |
|---|---:|---:|---:|---:|
| train1k | -0.008393 | -0.002555 | -0.025663 m | +0.008876 m |
| val1k | -0.004312 | -0.002563 | -0.025924 m | +0.007775 m |

第二次门禁仍未通过。停止完整 epoch、10k RL 和官方闭环评测，两个 v2 checkpoint 均仅作为诊断
产物，不作为候选模型。当前接受模型仍是监督 epoch 10。

报告与 SHA-256：

```text
bb97a9822a77757a2ae48c35acb517166b72ef8e6a042fab58b31f73fa0ac873  noise0.1 train1k
472d38b52b5b4667a56f28670360ad947d8de52374e397ee626d623779c49f6c  noise0.1 val1k
5558977c52bfb2d7bd713e91a989a1fa90b08be2f9155143174e9f7a14d161fe  noise0.2 train1k
7d1437319608cb3caf3e4b42184b8835b1abdf4314a66cfc5e3e25eba4e3ac78  noise0.2 val1k
064a201f1973bbc49db3da0515ddd6e0cb01bbbd062176bf8821a38e7fcad834  noise0.2 group diagnostics
```

详细解释另见：

```text
doc_hdp_nuplan/RL_update_v2_gate_result.md
```

本轮结论：update v2 将旧 RL 的 reward/path 漂移压低约一个数量级，但没有产生正收益。下一轮应
改变 update 目标（如 best-of-N 偏好更新并加 KL/参数漂移约束），继续沿用 20-step train/val 门禁；
在门禁转正前不扩大训练。全过程未执行 commit，所有代码、文档和实验产物均保留为 changes。

### 12.13 忠实迁移 HDP-NAVSIM 候选协议（2026-08-08）

#### 12.13.1 源码核对

重新核对 `HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py`、`scoring.py`、
`model/rl_utils.py` 和 `config/agent/dp_vla_rl_agent.yaml`，并与
`/home/yanjun/NewDisk/Hyper-Diffusion-Planner/HDP-navsim` 中的对应核心文件执行 `diff -u`；agent 和
配置文件无差异。

确认原 HDP-NAVSIM 不是 Best-of-N、PPO 或 GRPO，而是：每场景 10 条候选、5-step diffusion、
前 5 个 epoch 施加 0.5 m 局部轨迹平移、PDM score、组内 z-score、`exp(reward) * diffusion MSE`。
它没有低方差门控、权重归一化、专家 anchor 或 KL。配置里的 `progress_weight`、`ttc_weight`、
`comfortable_weight`、`bc_data` 和运行时 `only_ep` 没有进入当前核心 reward/loss 路径。

#### 12.13.2 实现

新增：

```text
hdp_nuplan/rl/trajectory_augmentation.py
```

对 `[B,G,T,4]` 的 `[x,y,cos_yaw,sin_yaw]`，每条候选采样一对沿时间共享的局部纵向/横向高斯
偏移，再根据每个点航向旋转到全局坐标；`cos/sin` 原样保留。`train_predictor_rl.py` 默认
`rl_group_size` 从 8 对齐为 10，并新增：

```text
--rl_trajectory_augmentation_std 0.5
--rl_trajectory_augmentation_epochs 5
```

`rollout_epoch()` 接收零基 epoch，仅在 `epoch < 5` 时增强；行为比较工具新增默认关闭的
`--trajectory-augmentation-std`，防止正常推理评估误加增强。新增两个单元测试，验证局部坐标变换、
时间共享、航向不变和 `std=0` no-op。

#### 12.13.3 20-step 训练命令

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-rl-navsim-protocol-g10-aug05-gate20-from-10k-epoch10 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 --num_workers 2 --train_epochs 2 --warm_up_epoch 1 \
  --save_utd 1 --learning_rate 1e-5 --rl_group_size 10 --rl_rollout_steps 5 \
  --rl_sampling_noise_scale 0.2 --rl_trajectory_augmentation_std 0.5 \
  --rl_trajectory_augmentation_epochs 5 --rl_buffer_update_epoch 5 \
  --rl_buffer_size 1024 --rl_freeze_encoder true --rl_min_reward_std 0.01 \
  --rl_normalize_weights true --rl_expert_anchor_weight 0.1 \
  --rl_max_update_steps_per_epoch 20 --reward_comfort_weight 0.01 \
  --reward_collision_distance 0.5
```

运行成功，无 OOM、NaN 或异常。运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-rl-navsim-protocol-g10-aug05-gate20-from-10k-epoch10/2026-08-08-17:16:42/
```

rollout 指标：reward=`0.331166`、progress=`2.311024`、collision cost=`0.175123`、route cost=
`0.194760`、comfort cost=`3.057588`。20-step update：total loss=`0.078735`、RL loss=`0.072015`、
anchor loss=`0.067193`、reward std mean=`1.062877`、active group fraction=`1.0`、update steps=`20`。

checkpoint：

```text
model_epoch_2_trainloss_0.0787.pth
SHA-256: b862784c9d4a369488b27e833f960fe173619e357f2de8da14f002730dda578d
```

#### 12.13.4 正常推理门禁

使用 `compare_checkpoint_behavior.py` 比较监督起点和新 checkpoint。train 与 val 都使用
`num-samples=10`、`diffusion-steps=5`、`sampling-noise-scale=0.2`、3 repeats 和共同随机数；明确传入
`--trajectory-augmentation-std 0`，因为部署推理不会增加训练扰动。

| 数据 | reward 变化 | progress 变化 | path length 变化 | collision cost 变化 | ADE 变化 |
|---|---:|---:|---:|---:|---:|
| train1k | -0.007442 | -0.002649 | -0.026392 m | +0.000481 | +0.009038 m |
| val1k | -0.004216 | -0.002656 | -0.026712 m | +0.000158 | +0.007910 m |

train 主门禁失败后没有扩大训练；val1k 只作为跨数据集归因证据，未运行完整 epoch、10k 或闭环。

#### 12.13.5 候选多样性诊断

另外用监督模型、10 candidates、augmentation 0.5 m、repeat 1 生成诊断报告。中位数：reward std=
`0.110409`、reward range=`0.366213`、progress std=`0.053150`、endpoint diversity=`0.967435 m`、
reward-progress correlation=`0.703297`。最高 reward 候选相对组均值的 progress 中位增量为
`0.071681`，path length 中位增量为 `0.439728 m`。

此前 group=4、noise=0.2、无增强时，reward std 中位数仅 `0.014335`，endpoint diversity 仅
`0.197904 m`。因此 NAVSIM 候选协议成功解决了候选过度重合，但更新后正常模型仍然变短，剩余问题
不能再归因于低候选多样性。

证据与 SHA-256：

```text
78a325d6bb105666be0723a1a3e180e7b6065edf943d05173d0d62ffdcfd076e  train1k repeat3
5bfb0de309c8768d415c564c383d73629fed0d08f2f5cf49fd5f7a1956c55658  val1k repeat3
a06e0cd81b45cb52105a5c9f3ac5a07c9295f533b4ee3e9e0d981d0e90c0b9c0  group diagnostics
```

详细报告：

```text
doc_hdp_nuplan/HDP_NAVSIM_RL_protocol_migration_gate.md
```

全量测试结果：`26 passed, 1 warning`。该 warning 仍是第三方 `timm.models.layers` 弃用提示。
本轮没有执行 commit；代码、checkpoint、JSON 和文档全部保留为 changes。

### 12.14 update 目标控制实验与 Best-of-N（2026-08-08）

#### 12.14.1 为什么先做控制实验

noise=0.1、noise=0.2 和 NAVSIM group10+augmentation 三轮20-step更新的正常推理路径都缩短约
`0.026 m`。候选分布变化很大但输出漂移接近，说明必须先区分 rollout 自蒸馏和额外监督训练各自
造成的变化，不能直接继续调候选参数。

在 `train_epoch_rl.py` 新增：

```text
L_total = rl_rollout_loss_weight * L_rollout
        + rl_expert_anchor_weight * L_expert
```

`train_predictor_rl.py` 新增 `--rl_rollout_loss_weight`，默认1；两个权重不能同时为0。日志新增
`weighted_rl_loss` 和 `weighted_expert_anchor_loss`。单元测试显式验证 rollout 权重为0时其梯度为0。

#### 12.14.2 Anchor-only 20-step 控制

训练命令沿用12.13.3的完整命令，只做以下替换：

```text
--name hdp-mini-control-anchor-only-g10-aug05-gate20-from-10k-epoch10
--rl_rollout_loss_weight 0
--rl_expert_anchor_weight 1.0
```

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-control-anchor-only-g10-aug05-gate20-from-10k-epoch10/2026-08-08-17:41:04/
```

rollout 与12.13完全一致。20步 update：total/expert loss=`0.068052`、`weighted_rl_loss=0`、
`weighted_expert_anchor_loss=0.068052`。正常推理继续使用10 candidates、noise0.2、augmentation0、
3 repeats和共同随机数：

| 数据 | reward 变化 | progress 变化 | path length 变化 | ADE 变化 |
|---|---:|---:|---:|---:|
| train1k | -0.000834 | -0.000388 | -0.004016 m | +0.001818 m |
| val1k | -0.000743 | -0.000385 | -0.003994 m | +0.001697 m |

对比12.13完整RL的路径变化 `-0.026392/-0.026712 m`，anchor-only 将 train/val 退化分别减少约
`84.8%/85.0%`。所以约85%的路径漂移来自 rollout 自蒸馏，而不是仅由专家 anchor/EMA 更新流程造成。

证据 SHA-256：

```text
e1c9ccd58899867e87a877d320888079d22b94e743c9ce2250ba48c22e49c051  anchor-only checkpoint
7e6feb04bfc9715844287a9d6d5f10d6ed71db7e60beaf94bb76d45d579666d3  train1k repeat3
2e38af0abf2f2071fbce540b88aa84a970bb25559edf2510955399ff25341316  val1k repeat3
```

#### 12.14.3 Best-of-N 改进分支

新增 `--rl_candidate_weighting {exponential,best_of_n}`，默认 `exponential` 保持原行为。
`best_of_n` 对每个有效组只保留最高 reward 候选；归一化开启时胜者权重为G，使最终 mean 的损失
尺度保持为1。低方差组继续全部跳过。该逻辑有单元测试覆盖。

本次关闭0.5 m轨迹平移，避免拟合首个未来点已经整体偏移的增强目标；采用自然扩散候选和保守损失
比例。完整训练命令为：

```bash
cd /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m torch.distributed.run --nnodes=1 --nproc-per-node=1 --standalone \
  train_predictor_rl.py \
  --name hdp-mini-bestofn-g10-noaug-gate20-from-10k-epoch10 \
  --save_dir /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1 \
  --train_set /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/cache \
  --train_set_list /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1/diffusion_planner_training.json \
  --pretrained_model_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/training_log/hdp-mini-supervised-balanced10k-warmstart/2026-08-01-01:56:22/model_epoch_10_trainloss_0.1380.pth \
  --normalization_file_path /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json \
  --batch_size 8 --num_workers 2 --train_epochs 2 --warm_up_epoch 1 \
  --save_utd 1 --learning_rate 1e-5 --rl_group_size 10 --rl_rollout_steps 5 \
  --rl_sampling_noise_scale 0.2 --rl_trajectory_augmentation_std 0 \
  --rl_trajectory_augmentation_epochs 5 --rl_buffer_update_epoch 5 \
  --rl_buffer_size 1024 --rl_freeze_encoder true --rl_min_reward_std 0.01 \
  --rl_normalize_weights true --rl_candidate_weighting best_of_n \
  --rl_rollout_loss_weight 0.25 --rl_expert_anchor_weight 1.0 \
  --rl_max_update_steps_per_epoch 20 --reward_comfort_weight 0.01 \
  --reward_collision_distance 0.5
```

运行目录：

```text
tmp/mini_train_pilot_1000_seed3407_v1/training_log/
hdp-mini-bestofn-g10-noaug-gate20-from-10k-epoch10/2026-08-08-17:45:28/
```

rollout reward=`0.691586`、progress=`2.311038`。update 中 active group fraction=`0.55625`、
weighted rollout loss=`0.010049`、weighted expert loss=`0.057369`、total loss=`0.067418`、20步。

正常推理门禁：

| 数据 | reward 变化 | progress 变化 | path length 变化 | collision cost 变化 | ADE 变化 |
|---|---:|---:|---:|---:|---:|
| train1k | -0.002097 | -0.000916 | -0.009667 m | +0.000121 | +0.003684 m |
| val1k | -0.001501 | -0.000933 | -0.009743 m | +0.000059 | +0.003572 m |

Best-of-N 比全候选稳定，但比 anchor-only 更差，说明新增 rollout 分量在 train/val 上仍为负贡献。
门禁拒绝，不运行完整 epoch、10k 或闭环。

证据 SHA-256：

```text
6a9cd99372c8795f13e0d57153c93609330a871c27efc84bb73af224e60fda81  Best-of-N checkpoint
cff3bf1aae763caed0fe9af60790708254155e3e205da9b6fd3649c8c37a3ce6  train1k repeat3
096ee92210b99c6d07bfc07da806933d320ec230a588439b72094812431fd8e3  val1k repeat3
```

#### 12.14.4 最终收束

三类 update 的正常推理路径退化从小到大为：anchor-only约 `0.004 m`、Best-of-N约 `0.0097 m`、
全候选加权约 `0.0265 m`。继续调整 rollout/anchor 权重只会在已经验证为负的区间内插值，不再具有
合理实验收益。当前离线 RL 分支正式停止扩大训练，最终接受模型仍为监督 epoch10。

详细说明：

```text
doc_hdp_nuplan/RL_update_objective_control_and_bestofn.md
```

全量测试结果：`28 passed, 1 warning`。未执行 commit，全部保留为 changes。

### 12.15 移除 Best-of-N，恢复 NAVSIM 主线（2026-08-08）

用户明确决定不使用 Best-of-N，继续采用 HDP-NAVSIM 的全候选奖励加权思想。因此从当前代码删除：

```text
--rl_candidate_weighting
group_advantage_weights(..., weighting_mode=...)
best_of_n 单候选权重分支
对应单元测试
```

当前唯一候选权重流程恢复为：

```text
10条候选
→ 组内 reward 标准化并裁剪
→ exp(temperature * advantage)
→ 有效组内权重均值归一化为1
→ 低方差组跳过
→ 全部有效候选参与 diffusion loss
```

`rl_rollout_loss_weight` 作为12.14控制实验和后续诊断开关保留，默认值仍为1，不改变 NAVSIM 主线
训练行为。12.14 的 Best-of-N checkpoint、JSON 和报告保留为历史负实验，不作为当前代码功能或候选
模型；历史命令中的 `--rl_candidate_weighting best_of_n` 在当前代码中不再可执行。

移除后全量测试：`27 passed, 1 warning`；`py_compile` 与 `git diff --check` 均通过，代码范围内搜索
不再存在 `best_of_n`、`rl_candidate_weighting` 或 `weighting_mode`。未执行 commit。
