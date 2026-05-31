# 导入 numpy，用于数组构造、数值计算、坐标表示、保存 npz 文件等
import numpy as np

# 导入 tqdm，用于在遍历 scenarios 时显示进度条
from tqdm import tqdm

# Point2D 是 nuPlan 中的二维点类型，这里用于表示自车当前位置的 x、y 坐标
from nuplan.common.actor_state.state_representation import Point2D

# 导入 route roadblock 修正函数
# 用于修正导航 route 中可能存在的断连、缺失 roadblock、环路等问题
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction

# 导入 agent 相关数据处理函数
from diffusion_planner.data_process.agent_process import (
agent_past_process, 
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)

# 导入地图矢量特征提取和地图特征处理函数
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process

# 导入自车历史、未来和当前状态处理函数
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states

# 导入 numpy 数据转模型输入 tensor 的函数
from diffusion_planner.data_process.utils import convert_to_model_inputs


# 数据处理器类
# 主要用于两种场景：
# 1. observation_adapter：推理阶段，将 nuPlan 在线仿真中的历史观测转成模型输入；
# 2. work：离线数据预处理阶段，将 scenario 批量处理并保存为 .npz 训练数据。
class DataProcessor(object):

    # 初始化数据处理器，读取配置参数，并设置固定的数据处理超参数
    def __init__(self, config):

        # 处理后数据保存目录
        # 从 config 中读取 save_path，如果不存在则为 None
        self._save_dir = getattr(config, "save_path", None) 

        # 历史时间范围，单位为秒
        # 这里表示使用过去 2 秒的历史信息
        self.past_time_horizon = 2 # [seconds]

        # 历史轨迹采样点数
        # 这里假设采样频率为 10Hz，因此 2 秒对应 20 个历史采样点
        self.num_past_poses = 10 * self.past_time_horizon 

        # 未来时间范围，单位为秒
        # 这里表示预测未来 8 秒轨迹
        self.future_time_horizon = 8 # [seconds]

        # 未来轨迹采样点数
        # 10Hz 采样下，8 秒对应 80 个未来采样点
        self.num_future_poses = 10 * self.future_time_horizon

        # 动态 agent 的最大数量
        # 例如车辆、行人、自行车等，超过该数量会按距离和类别规则筛选
        self.num_agents = config.agent_num

        # 静态目标最大数量
        # 例如交通锥、路障、施工区标志等
        self.num_static = config.static_objects_num

        # 行人和自行车数量上限
        # 防止行人/自行车过多时挤占车辆 agent 的名额
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.

        # 地图查询半径，单位 m
        # 以当前自车位置为中心，提取半径 100m 内的地图元素
        self._radius = 100 # [m] query radius scope relative to the current pose.

        # 需要提取的地图特征类型
        # LANE：车道中心线
        # LEFT_BOUNDARY：车道左边界
        # RIGHT_BOUNDARY：车道右边界
        # ROUTE_LANES：导航路线相关车道
        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.

        # 每种地图特征最多保留多少个元素
        # 普通 lane、左右边界使用 config.lane_num
        # route lane 使用 config.route_num
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.

        # 每个地图元素最多保留多少个点
        # 普通 lane、左右边界使用 config.lane_len
        # route lane 使用 config.route_len
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    # 推理阶段使用的观测适配函数
    # 作用：将在线仿真时的 history_buffer、交通灯、地图和 route 信息转换成模型可以直接输入的 tensor 字典
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''

        # 推理阶段不需要自车完整历史 ego_agent_past
        # 这里只保留 None，后续 agent_past_process 会跳过 ego 历史处理
        ego_agent_past = None # inference no need ego_agent_past

        # 获取当前自车状态
        # history_buffer.current_state[0] 通常是当前时刻 EgoState
        ego_state = history_buffer.current_state[0]

        # 构造自车当前位置 Point2D，用于地图查询
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

        # 构造 anchor ego state
        # 格式为 [x, y, heading]
        # 后续所有 agent、地图点都会转换到这个自车局部坐标系下
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''

        # 获取历史观测缓存
        # 包括过去若干帧以及当前帧的 tracked objects
        observation_buffer = history_buffer.observation_buffer # Past observations including the current

        # 将历史观测中的动态目标转换为数组列表
        # neighbor_agents_past：每一帧的动态 agent 数组
        # neighbor_agents_types：每一帧中对应 agent 的类型
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)

        # 从当前帧观测中提取静态目标
        # observation_buffer[-1] 表示当前帧
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])

        # 处理动态 agent 历史和静态目标
        # 输出：
        # 第一个返回值是 ego，这里用 _ 忽略，因为推理阶段不需要 ego_agent_past
        # neighbor_agents_past：固定数量、局部坐标系下的动态 agent 历史特征
        # 第三个返回值是 selected_indices，这里用 _ 忽略
        # static_objects：固定数量、局部坐标系下的静态目标特征
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''

        # Simply fixing disconnected routes without pre-searching for reference lines
        # 对 route_roadblock_ids 进行修正
        # 主要处理 route 断连、缺失中间 roadblock、环路等问题
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )

        # 从地图中提取自车附近的矢量地图特征
        # coords：地图几何信息，例如 lane 中心线、左右边界等
        # traffic_light_data：lane 对应的交通灯编码
        # speed_limit：lane 限速信息
        # lane_route：每条 lane 对应的 roadblock id
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )

        # 将原始矢量地图数据转换为模型输入格式
        # 包括固定数量截断/补零、插值到固定点数、转换到自车坐标系、构造 route_lanes 等
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

        
        # 组织推理阶段模型输入数据
        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}

        # 将地图特征加入 data
        data.update(vector_map)

        # 将 numpy 数组转换为 torch Tensor，并移动到指定 device
        data = convert_to_model_inputs(data, device)

        # 返回模型可直接使用的输入
        return data
    
    # Use for data preprocess
    # 离线数据预处理主函数
    # 输入一批 scenarios，逐个提取自车、agent、地图、未来轨迹等信息，并保存为 .npz 文件
    def work(self, scenarios):

        # 遍历所有 scenario，并用 tqdm 显示处理进度
        for scenario in tqdm(scenarios):

            # 当前 scenario 所属地图名称
            map_name = scenario._map_name

            # 当前 scenario 的唯一 token
            token = scenario.token

            # 当前 scenario 对应的地图 API
            map_api = scenario.map_api        

            '''
            ego & agents past
            '''

            # 当前时刻自车状态
            ego_state = scenario.initial_ego_state

            # 当前自车位置，用于地图查询
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

            # 当前自车位姿 [x, y, heading]
            # 作为坐标转换的 anchor
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

            # 从 scenario 中提取自车历史轨迹和对应时间戳
            # ego_agent_past：历史自车状态数组
            # time_stamps_past：历史时间戳数组
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            # 获取当前帧 tracked objects
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            # 获取过去 self.past_time_horizon 秒内的 tracked objects
            # num_samples=self.num_past_poses 表示历史采样点数
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]

            # 将过去 tracked objects 与当前帧 tracked objects 拼接
            # 形成完整的历史观测序列
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]

            # 将历史观测序列中的动态目标转换为数组列表
            # neighbor_agents_past：每帧动态 agent 数组
            # neighbor_agents_types：每帧动态 agent 类型
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            
            # 从当前帧提取静态目标
            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            # 处理自车历史、动态 agent 历史、静态目标
            # ego_agent_past：转换到自车局部坐标系后的自车历史
            # neighbor_agents_past：固定数量动态 agent 历史特征
            # neighbor_indices：被选中的动态 agent 原始索引，后续用于提取这些 agent 的未来轨迹
            # static_objects：固定数量静态目标特征
            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            
            '''
            Map
            '''

            # 获取 scenario 原始导航 route roadblock id 列表
            route_roadblock_ids = scenario.get_route_roadblock_ids()

            # 获取当前 iteration=0 时刻的交通灯状态
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

            # 如果 route 不为空，则进行 route roadblock 修正
            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )

            # 从地图中提取自车附近的矢量地图信息
            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            # 将地图信息处理成模型输入格式
            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

            '''
            ego & agents future
            '''

            # 提取自车未来轨迹
            # 输出通常是相对于当前自车坐标系的未来轨迹点
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            # 再次获取当前帧 tracked objects
            # 当前帧会作为未来序列的起点
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            # 获取未来 self.future_time_horizon 秒内的 tracked objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            # 未来观测序列由当前帧 + 未来帧组成
            sampled_future_observations = [present_tracked_objects] + future_tracked_objects

            # 将未来观测序列中的动态目标转换为数组列表
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)

            # 根据历史阶段选中的 neighbor_indices，提取对应 agent 的未来轨迹
            # 输出固定为 self.num_agents 个 agent 的未来轨迹标签
            neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


            '''
            ego current
            '''

            # 根据自车历史状态和时间戳计算当前扩展自车状态
            # 包括 x、y、cos heading、sin heading、vx、vy、ax、ay、steering angle、yaw rate
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # gather data
            # 汇总单个 scenario 的所有训练数据
            data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}

            # 加入地图特征
            data.update(vector_map)

            # 保存当前 scenario 的处理结果到磁盘
            self.save_to_disk(self._save_dir, data)

    # 将单个 scenario 的处理结果保存成 .npz 文件
    def save_to_disk(self, dir, data):

        # 保存路径格式：
        # {save_dir}/{map_name}_{token}.npz
        # 文件中保存 data 字典中的所有键值对
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)