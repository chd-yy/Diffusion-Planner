# 导入 os 模块，用于文件夹创建、文件列表读取等操作
import os

# 导入 argparse 模块，用于从命令行接收参数
import argparse

# 导入 json 模块，用于读取和保存 json 文件
import json

# 导入项目中自定义的数据处理器 DataProcessor
# 该类通常负责把 nuPlan 原始 scenario 数据转换成模型训练所需的缓存数据，例如 .npz 文件
from diffusion_planner.data_process.data_processor import DataProcessor

# nuPlan 提供的单机并行执行器
# 用于在构建 scenario 时并行读取和处理数据，提高数据加载效率
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

# nuPlan 的场景过滤器
# 用于按照场景类型、log 名称、场景数量、是否打乱等条件筛选 scenario
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# nuPlan 数据库场景构建器
# 用于从 nuPlan 数据库文件中构建可供规划模型训练或评估使用的 scenario 对象
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

# ScenarioMapping 用于场景类型映射
# 这里虽然导入了，但当前代码中没有实际使用
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


# 定义一个函数，用于组织并返回 ScenarioFilter 所需的过滤参数
# 这些参数会在后面通过 *get_filter_parameters(...) 的形式传入 ScenarioFilter
def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    # 场景类型过滤条件
    # None 表示不限制具体的 scenario type，即保留所有类型的场景
    scenario_types = None

    # 指定需要包含的 scenario token 列表
    # 如果 scenario_tokens 不为 None，则只保留这些 token 对应的场景
    scenario_tokens                      # List of scenario tokens to include

    # 按 log 文件名称过滤场景
    # 这里将外部传入的 log_names 赋值给局部变量 log_names
    log_names = log_names                # Filter scenarios by log names

    # 地图名称过滤条件
    # None 表示不限制地图名称
    map_names = None                     # Filter scenarios by map names

    # 每种场景类型最多保留多少个 scenario
    # 例如设置为 100，则每种 scenario type 最多取 100 个
    num_scenarios_per_type               # Number of scenarios per type

    # 限制总场景数量
    # 如果是整数，表示最多保留多少个场景
    # 如果是浮点数，通常表示按比例采样
    # 该限制可以叠加在 num_scenarios_per_type 之后
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type

    # 时间戳间隔阈值
    # 用于过滤初始 lidar 时间戳之间间隔不足的场景
    # None 表示不启用该过滤条件
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps

    # 自车最小位移过滤条件
    # 可用于去除自车移动距离过小的场景
    # None 表示不启用该过滤条件
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    # 是否展开多采样场景
    # True 表示把 multi-sample scenario 展开成多个 single-sample scenario
    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios

    # 是否移除任务目标无效的场景
    # False 表示不移除 mission goal 无效的场景
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid

    # 是否打乱场景顺序
    # shuffle 参数由外部传入，通常训练时设为 True
    shuffle                              # Whether to shuffle the scenarios

    # 自车起步速度阈值
    # 用于筛选自车速度从低于某阈值逐渐达到该阈值的场景
    # None 表示不启用
    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below

    # 自车停车速度阈值
    # 用于筛选自车速度从高于某阈值逐渐降低到该阈值的场景
    # None 表示不启用
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above

    # 速度噪声容忍阈值
    # 速度变化小于等于该值时可被视为噪声
    # None 表示不启用
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    # 按照 nuPlan ScenarioFilter 构造函数需要的顺序返回所有过滤参数
    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


# Python 脚本入口
# 当该文件被直接运行时，下面的代码才会执行
# 如果该文件被其他文件 import，则不会执行
if __name__ == "__main__":

    # 创建命令行参数解析器
    # description 用于说明该脚本的功能
    parser = argparse.ArgumentParser(description='Data Processing')

    # 原始 nuPlan 数据路径 
    # 默认路径为 /data/nuplan-v1.1/trainval
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')

    # nuPlan 地图数据路径
    # 默认路径为 /data/nuplan-v1.1/maps
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    # 处理后数据的保存路径
    # 默认保存到当前目录下的 ./cache 文件夹
    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')

    # 每种场景类型采样多少个场景
    # default=None 表示不对每种类型单独限制数量
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')

    # 限制总场景数量
    # 默认只处理 10 个场景，通常用于快速测试数据处理流程是否正常
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')

    # 是否打乱 scenario 顺序
    # 注意：argparse 中 type=bool 的行为可能和直觉不同，但这里保持原代码不变
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    # 动态交通参与者数量
    # 例如周围车辆、行人、骑行者等 agent
    # 这里默认最多保留 32 个 agent
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    # 静态障碍物数量
    # 例如锥桶、路障等静态目标
    # 默认最多保留 5 个
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    # 每条 lane polyline 采样的点数
    # 默认每条车道线保留 20 个点
    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)

    # 车道数量
    # 默认最多保留 70 条相关 lane
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    # 每条 route lane 采样的点数
    # route 表示导航路线相关的车道序列
    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)

    # route lane 数量
    # 默认最多保留 25 条导航相关 lane
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)

    # 解析命令行参数，并保存到 args 中
    args = parser.parse_args()

    # 创建处理后数据的保存文件夹
    # exist_ok=True 表示如果该文件夹已经存在，则不会报错
    os.makedirs(args.save_path, exist_ok=True)

    # sensor_root 表示传感器原始数据路径
    # 当前数据处理流程不使用传感器原始数据，因此设置为 None
    sensor_root = None

    # db_files 表示指定的数据库文件列表
    # 设置为 None 时，NuPlanScenarioBuilder 会根据 data_path 自动查找数据库文件
    db_files = None

    # Only preprocess the training data
    # 读取训练集 log 名称列表
    # nuplan_train.json 中通常保存需要用于训练的数据 log name
    # 后续 ScenarioFilter 会根据这些 log_names 只筛选训练数据
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:

        # 从 json 文件中读取 log_names
        log_names = json.load(file)

    # 指定 nuPlan 地图版本
    map_version = "nuplan-maps-v1.0"    

    # 创建 nuPlan 场景构建器
    # args.data_path: nuPlan 原始数据库路径
    # args.map_path: 地图路径
    # sensor_root: 传感器数据路径，这里为 None
    # db_files: 指定数据库文件，这里为 None
    # map_version: 地图版本
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)

    # 创建 scenario 过滤器
    # get_filter_parameters(...) 会返回 ScenarioFilter 所需的一组参数
    # 这里主要使用：
    # args.scenarios_per_type 控制每类场景数量
    # args.total_scenarios 控制总场景数量
    # args.shuffle_scenarios 控制是否打乱
    # log_names 控制只使用训练集 log
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    # 创建单机并行执行器
    # use_process_pool=True 表示使用多进程池并行处理
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    # 根据 scenario_filter 从 nuPlan 数据库中构建场景列表
    # scenarios 是后续 DataProcessor 需要处理的核心输入
    scenarios = builder.get_scenarios(scenario_filter, worker)

    # 打印筛选并构建得到的场景总数
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    # 删除 worker、builder 和 scenario_filter
    # 这样可以释放一部分内存资源，避免后续数据处理阶段占用过多内存
    del worker, builder, scenario_filter

    # 创建数据处理器
    # DataProcessor 会根据 args 中的配置处理 scenario
    processor = DataProcessor(args)

    # 开始处理所有 scenarios
    # 通常会将每个 scenario 转换为模型训练所需的结构化数据，并保存为 .npz 文件
    processor.work(scenarios)

    # 遍历保存路径，找出所有以 .npz 结尾的文件
    # 这些文件就是处理后的训练样本缓存文件
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    # Save the list to a JSON file
    # 将所有 .npz 文件名保存到 diffusion_planner_training.json
    # 训练脚本可以通过读取该 json 文件知道有哪些缓存样本可用于训练
    with open('./diffusion_planner_training.json', 'w') as json_file:

        # 将 .npz 文件名列表写入 json 文件
        # indent=4 表示以缩进格式保存，方便人工查看
        json.dump(npz_files, json_file, indent=4)

    # 打印最终保存的 .npz 文件数量
    print(f"Saved {len(npz_files)} .npz file names")