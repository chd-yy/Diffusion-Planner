# HDP-nuPlan `data_process.py` 流程

## 1. 必要语法

- `argparse`：从命令行读取数据路径、场景数量和张量上限。
- `if __name__ == "__main__"`：直接执行脚本时启动预处理。
- `set` + `sorted`：去重并稳定排序 NPZ 文件名。
- `SingleMachineParallelExecutor`：并行构建 NuPlan 场景对象；不负责并行写 NPZ。

## 2. 整体作用

`data_process.py` 是原始 NuPlan DB 到 HDP 训练缓存的入口：

```text
NuPlan DB + 地图 + 日志清单
→ 构建并筛选 Scenario
→ 可选按日志均衡抽样
→ 提取自车、邻车、地图和未来标签
→ 每场景保存一个 NPZ
→ 校验预期文件并生成 manifest
```

入口文件：`HDP-nuplan/data_process.py`

特征处理：`HDP-nuplan/hdp_nuplan/data_process/data_processor.py`
均衡抽样：`HDP-nuplan/hdp_nuplan/data_process/sampling.py`

## 3. 主流程

### 3.1 解析参数并准备目录

主要输入：

| 参数 | 作用 |
|---|---|
| `data_path` | NuPlan `.db` 所在目录 |
| `map_path` | NuPlan 地图目录 |
| `log_names_json` | 允许使用的日志名列表 |
| `save_path` | NPZ 输出目录 |
| `total_scenarios` | 目标场景数 |
| `sampling_strategy` | `random` 或 `balanced_logs` |
| `output_list_path` | manifest 输出路径 |

脚本固定 Python 随机种子，并创建 `save_path`。

### 3.2 构建和筛选场景

1. 读取 `log_names_json`。
2. 使用 `NuPlanScenarioBuilder` 连接 DB 和 `nuplan-maps-v1.0`。
3. `ScenarioFilter` 按日志名、数量、是否打乱等条件筛选。
4. `SingleMachineParallelExecutor` 并行执行 `builder.get_scenarios()`。

当前过滤器不限制场景类型和地图，不删除无效 mission goal，并启用
`expand_scenarios=True`。

两种抽样策略的区别：

| 策略 | ScenarioBuilder 阶段 | 后处理 |
|---|---|---|
| `random` | 直接应用 `total_scenarios` 和 `shuffle_scenarios` | 不再抽样 |
| `balanced_logs` | 读取指定日志的全部场景，不打乱、不提前限量 | 去重后按日志配额抽样 |

`balanced_logs` 的规则：

1. 以 `<map_name>_<token>.npz` 为唯一键去重；
2. 每个日志先取 `total_scenarios // 日志数` 条；
3. 配额不足部分从所有剩余场景中按 `seed` 随机补齐；
4. 按最终 NPZ 文件名排序；
5. 保存 `sampling_report.json`，记录候选数、去重数和每日志数量。

该策略要求每个指定日志至少存在一个场景、唯一候选总数足够，并且
`total_scenarios >= 日志数`。

### 3.3 提取特征并写 NPZ

`DataProcessor.work()` 顺序遍历选中场景。每个场景执行：

1. 以当前自车位姿为局部坐标原点；
2. 提取 2 秒历史（20 帧历史加当前帧）；
3. 处理最多 32 个动态参与者和 5 个静态物体；
4. 在 100 米范围提取 lane、边界、限速、红绿灯和 route；
5. 修正断裂的 route roadblock；
6. 提取 8 秒、80 帧的自车和邻车未来标签；
7. 计算当前自车速度、加速度等附加状态；
8. 保存为 `<map_name>_<scenario_token>.npz`。

默认主要字段和 shape：

| 字段 | shape |
|---|---:|
| `ego_current_state` | `(10,)` |
| `ego_agent_future` | `(80, 3)` |
| `neighbor_agents_past` | `(32, 21, 11)` |
| `neighbor_agents_future` | `(32, 80, 3)` |
| `lanes` | `(70, 20, 12)` |
| `route_lanes` | `(25, 20, 12)` |
| `static_objects` | `(5, 10)` |

NPZ 还保存 lane/route 限速及其有效标志，以及 `map_name`、`log_name`、
`scenario_type`、`token` 元数据。

注意：场景对象的构建使用并行 worker，但 `DataProcessor.work()` 本身是顺序写盘。

#### 3.3.1 第 139–140 行具体发生了什么

```python
processor = DataProcessor(args)
processor.work(scenarios)
```

`args` 是 `parser.parse_args()` 返回的 `argparse.Namespace`，包含全部命令行参数和默认值。
在 100 场景 smoke 命令中，其有效内容等价于：

```text
data_path=/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini
map_path=/home/yanjun/NewDisk/nuplan/dataset/maps
save_path=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v1/cache
scenarios_per_type=None
total_scenarios=100
shuffle_scenarios=False
seed=3407
sampling_strategy=random
log_names_json=./config/mini_splits/mini_train_logs.json
output_list_path=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_smoke_100_v1/diffusion_planner_training.json
sampling_report_path=None
agent_num=32
static_objects_num=5
lane_len=20
lane_num=70
route_len=20
route_num=25
```

第 139 行把整个 `args` 作为配置对象传给构造函数，但 `DataProcessor.__init__()` 实际只读取：

| `args` 字段 | 保存到处理器中的配置 |
|---|---|
| `save_path` | NPZ 输出目录 |
| `agent_num` | 最大动态参与者数 |
| `static_objects_num` | 最大静态物体数 |
| `lane_num` / `lane_len` | lane 数量及每条 lane 点数 |
| `route_num` / `route_len` | route lane 数量及点数 |

构造函数还固定设置：历史 `2s/20` 帧、未来 `8s/80` 帧、查询半径 `100m`、
最多 10 个行人/自行车，以及需要提取的地图图层。`data_path`、`map_path`、采样参数等已在
前面的 ScenarioBuilder/ScenarioFilter 阶段使用，`DataProcessor` 不再读取它们。

第 140 行传入的是筛选完成的 `scenarios`，不是 `args`。在该 smoke 中它包含 100 个
`NuPlanScenario`。`work()` 顺序处理每个场景，提取历史、未来、地图和元数据，并通过
`np.savez()` 写出一个 `<map_name>_<token>.npz`。该方法没有显式返回值，结果体现在磁盘文件中。

### 3.4 生成 manifest

处理结束后，脚本根据本次选中的场景重新计算预期文件名：

1. 文件名去重并排序；
2. 检查每个预期 NPZ 是否存在，缺失则报错；
3. 将文件名列表写入 `output_list_path`。

manifest 只包含本次场景，不会因为输出目录中存在历史 NPZ 而把它们加入训练清单。
但该脚本不会拒绝额外历史 NPZ；需要严格检查时使用
`scripts/validate_processed_cache.py`。

## 4. 推荐调用方式

使用绝对脚本路径，避免误运行仓库根目录的同名文件：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/data_process.py \
  --data_path /home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini \
  --map_path /home/yanjun/NewDisk/nuplan/dataset/maps \
  --save_path /path/to/run/cache \
  --log_names_json /path/to/mini_train_logs.json \
  --output_list_path /path/to/run/diffusion_planner_training.json \
  --sampling_strategy balanced_logs \
  --total_scenarios 10000 \
  --seed 3407
```

## 5. 项目中常见问题

- 为什么按最终 NPZ 文件名去重？同一 lidar token 可能有多个 scenario tag，否则会覆盖同一文件。
- 为什么均衡抽样不在 Builder 中直接限量？必须先看到各日志的完整候选池才能分配配额。
- manifest 为什么不扫描整个缓存目录？避免旧 NPZ 静默混入新实验。
- 最容易出错在哪里？工作目录导致同名脚本误调用、日志清单与 DB 不匹配、输出目录残留旧 NPZ，以及 route/map 数据异常。
