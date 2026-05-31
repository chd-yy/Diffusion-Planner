"""
Module: Agent Data Preprocessing Functions
Description: This module contains functions for agents related data processing.

Categories:
    1. Get list of agent array from raw data
    2. Get agents array for model input
"""

# 导入 numpy，用于数组构造、距离计算、排序、三角函数、padding 等数值操作
import numpy as np

# 导入 Dict 类型注解，用于说明字典类型变量
from typing import Dict

# AgentInternalIndex 是 nuPlan 中用于索引 agent 状态数组各字段的工具类
# 例如 track_token、x、y、heading、vx、vy、width、length 等字段在数组中的列号
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex

# TrackedObjectType 表示感知目标的类别
# 例如 VEHICLE、PEDESTRIAN、BICYCLE、BARRIER、TRAFFIC_CONE 等
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# DetectionsTracks 是 nuPlan 中一种检测结果封装类型
# 其中包含 tracked_objects，也就是当前帧检测/跟踪到的目标集合
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

# 坐标转换函数
# 用于将全局坐标系下的自车、agent、静态目标状态转换到以当前自车为中心的局部坐标系下
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative

# =====================
# 1. Get list of agent array from raw data
# =====================
# 第 1 部分：从 nuPlan 原始 tracked object 数据中提取 agent 数组


# 从一帧 tracked_objects 中提取指定类别 agent 的状态，并转换成 numpy 数组
# tracked_objects：当前帧所有被检测/跟踪到的目标
# track_token_ids：用于将字符串形式的 track_token 映射成整数 id，保证同一个目标在不同帧中 id 一致
# object_types：需要保留的目标类别，例如车辆、行人、自行车
def _extract_agent_array(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a array.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a array as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a array.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated array and the updated track_token_ids dict.
    """

    # 从当前帧 tracked_objects 中筛选出指定类别的目标
    # 例如只保留 VEHICLE、PEDESTRIAN、BICYCLE
    agents = tracked_objects.get_tracked_objects_of_types(object_types)

    # 用于记录每个 agent 的类别
    # 后续会根据类别生成 one-hot 特征
    agent_types = []

    # 初始化输出数组
    # 行数为当前帧符合类别要求的 agent 数量
    # 列数为 AgentInternalIndex.dim()，即 nuPlan 规定的 agent 状态维度
    output = np.zeros((len(agents), AgentInternalIndex.dim()), dtype=np.float64)

    # 当前已经分配过的最大整数 track id
    # 如果遇到新的 track_token，就从这个值开始分配新的整数 id
    max_agent_id = len(track_token_ids)

    # 遍历当前帧中的每个 agent
    for idx, agent in enumerate(agents):

        # 如果该 agent 的 track_token 之前没有出现过
        # 则为它分配一个新的整数 id
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1

        # 获取该 agent 对应的整数 track id
        track_token_int = track_token_ids[agent.track_token]

        # 将整数 track id 写入 output 的 track_token 字段
        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)

        # 写入 agent 在全局坐标系下的 x 方向速度
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x

        # 写入 agent 在全局坐标系下的 y 方向速度
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y

        # 写入 agent 的航向角 heading
        output[idx, AgentInternalIndex.heading()] = agent.center.heading

        # 写入 agent 包围盒宽度
        output[idx, AgentInternalIndex.width()] = agent.box.width

        # 写入 agent 包围盒长度
        output[idx, AgentInternalIndex.length()] = agent.box.length

        # 写入 agent 中心点全局 x 坐标
        output[idx, AgentInternalIndex.x()] = agent.center.x

        # 写入 agent 中心点全局 y 坐标
        output[idx, AgentInternalIndex.y()] = agent.center.y

        # 保存 agent 的类别
        agent_types.append(agent.tracked_object_type)

    # 返回当前帧 agent 状态数组、更新后的 track_token_ids 字典、当前帧 agent 类别列表
    return output, track_token_ids, agent_types


# 将多帧 past_tracked_objects 转换成 agent 数组列表
# 每一帧会调用 _extract_agent_array 转换成一个 numpy 数组
def sampled_tracked_objects_to_array_list(past_tracked_objects):
    """
    Arrayifies the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each array as described in `_extract_agent_array()`.
    :param past_tracked_objects: The tracked objects to arrayify.
    :return: The arrayified objects.
    """

    # 指定需要保留的动态目标类别
    # 这里只处理车辆、行人、自行车
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]

    # output 保存每一帧转换后的 agent 数组
    output = []

    # output_types 保存每一帧中每个 agent 的类别
    output_types = []

    # track_token_ids 用于在所有帧之间维护 track_token 到整数 id 的映射
    # 这样同一个目标在不同时间帧中具有一致的整数 id
    track_token_ids = {}

    # 遍历所有历史帧
    for i in range(len(past_tracked_objects)):

        # 如果当前帧是 DetectionsTracks 类型，则需要取出其中的 tracked_objects
        if type(past_tracked_objects[i]) == DetectionsTracks:
            track_object = past_tracked_objects[i].tracked_objects

        # 如果当前帧本身就是 tracked_objects，则直接使用
        else:
            track_object = past_tracked_objects[i]

        # 将当前帧 tracked_objects 转换成 agent 状态数组
        arrayified, track_token_ids, agent_types = _extract_agent_array(track_object, track_token_ids, object_types)

        # 保存当前帧的 agent 数组
        output.append(arrayified)

        # 保存当前帧 agent 类型
        output_types.append(agent_types)

    # 返回多帧 agent 数组列表和对应的 agent 类型列表
    return output, output_types


# 将当前帧静态目标转换成数组
# 静态目标包括施工区标志、路障、交通锥、通用静态物体等
def sampled_static_objects_to_array_list(present_tracked_objects):

    # 指定需要保留的静态目标类型
    static_object_types = [TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT
                    ]

    # 如果输入是 DetectionsTracks 类型，则取出其中的 tracked_objects
    if type(present_tracked_objects) == DetectionsTracks:
        present_tracked_objects = present_tracked_objects.tracked_objects

    # 从当前帧 tracked_objects 中筛选静态目标
    static_obj = present_tracked_objects.get_tracked_objects_of_types(static_object_types)

    # 保存静态目标类别
    agent_types = []

    # 初始化静态目标数组
    # 每个静态目标包含 5 个字段：
    # x、y、heading、width、length
    output = np.zeros((len(static_obj), 5), dtype=np.float64)

    # 遍历每个静态目标
    for idx, agent in enumerate(static_obj):

        # 写入静态目标中心点 x 坐标
        output[idx, 0] = agent.center.x

        # 写入静态目标中心点 y 坐标
        output[idx, 1] = agent.center.y

        # 写入静态目标航向角
        output[idx, 2] = agent.center.heading

        # 写入静态目标包围盒宽度
        output[idx, 3] = agent.box.width

        # 写入静态目标包围盒长度
        output[idx, 4] = agent.box.length

        # 保存静态目标类别
        agent_types.append(agent.tracked_object_type)

    # 返回静态目标数组和静态目标类别列表
    return output, agent_types


# =====================
# 2. Get agents array for model input
# =====================
# 第 2 部分：将 agent 历史和未来数据处理成模型输入需要的固定尺寸数组


# 过滤 agent 序列，只保留在关键帧中出现过的 agent
# reverse=False 时，以第一帧为关键帧
# reverse=True 时，以最后一帧为关键帧
def _filter_agents_array(agents, reverse: bool = False):
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] arrays, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """

    # 选择用于过滤的目标帧
    # reverse=True 用于历史轨迹时，通常保留当前帧出现的 agent
    # reverse=False 用于未来轨迹时，通常保留起始帧出现的 agent
    target_array = agents[-1] if reverse else agents[0]

    # 遍历每一帧 agent 数组
    for i in range(len(agents)):

        # rows 用于保存当前帧中被保留下来的 agent 行
        rows = []

        # 遍历当前帧中的每个 agent
        for j in range(agents[i].shape[0]):

            # 如果目标帧中至少存在一个 agent
            if target_array.shape[0] > 0:

                # 读取当前 agent 的 track_token 整数 id
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())])

                # 判断当前 agent id 是否出现在目标帧 target_array 中
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:, AgentInternalIndex.track_token()]).max()
                )

                # 如果该 agent 在目标帧中出现，则保留当前帧中的这一行
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        # 如果当前帧有需要保留的 agent，则将 rows 堆叠成数组
        if len(rows) > 0:
            agents[i] = np.stack(rows)

        # 如果当前帧没有需要保留的 agent，则返回一个空数组
        # 列数保持为原来的 agent 状态维度
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)

    # 返回过滤后的多帧 agent 数组
    return agents


# 对 agent 轨迹进行 padding
# 核心作用：当某个 agent 在某些帧缺失时，用最近一次可用状态进行填充
# 这样每一帧中的 agent 顺序和数量保持一致
def _pad_agent_states(agent_trajectories, reverse: bool):
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """


    # track_token 在 agent 状态数组中的列索引
    track_id_idx = AgentInternalIndex.track_token()

    # 如果 reverse=True，则先反转时间顺序
    # 这样可以从当前帧向过去填充历史轨迹
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    # 选择反转后序列的第一帧作为关键帧
    # 该帧决定最终保留哪些 agent，以及每个 agent 对应输出数组中的哪一行
    key_frame = agent_trajectories[0]

    # id_row_mapping 用于建立 track_id 到输出行号的映射
    id_row_mapping: Dict[int, int] = {}

    # 遍历关键帧中的所有 agent id
    for idx, val in enumerate(key_frame[:, track_id_idx]):

        # 将 agent 的整数 track id 映射到固定行号 idx
        id_row_mapping[int(val)] = idx

    # current_state 保存当前已经填充好的 agent 状态
    # 初始全为 0，形状与 key_frame 一致
    current_state = np.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=np.float64)

    # 遍历每一帧 agent 数据
    for idx in range(len(agent_trajectories)):

        # 当前帧 agent 数据
        frame = agent_trajectories[idx]

        # Update current frame
        # 遍历当前帧中的每个 agent
        for row_idx in range(frame.shape[0]):

            # 根据当前 agent 的 track id 找到它在输出数组中固定对应的行
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx])]

            # 用当前帧检测到的状态更新 current_state 中对应行
            current_state[mapped_row, :] = frame[row_idx, :]

        # Save current state
        # 将当前状态保存为这一帧的完整 agent 状态
        # 如果某些 agent 当前帧缺失，则会保留前一帧最近可用状态
        agent_trajectories[idx] = current_state.copy()

    # 如果前面做了时间反转，这里再反转回来，恢复原始时间顺序
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    # 返回 padding 后的 agent 轨迹
    return agent_trajectories


# 对 agent 未来轨迹进行 zero padding
# 与 _pad_agent_states 不同，这里没有用最近状态补齐，而是缺失位置保持为 0
def _pad_agent_states_with_zeros(agent_trajectories):

    # 以第一帧为关键帧，决定需要预测哪些 agent
    key_frame = agent_trajectories[0]

    # track_token 字段所在列
    track_id_idx = AgentInternalIndex.track_token()

    # 初始化 padding 后的轨迹数组
    # 形状为 [num_frames, num_agents_in_key_frame, agent_state_dim]
    pad_agent_trajectories = np.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=np.float32)

    # 遍历每一帧
    for idx in range(len(agent_trajectories)):

        # 当前帧 agent 数据
        frame = agent_trajectories[idx]

        # 当前帧中所有 agent 的 track id
        mapped_rows = frame[:, track_id_idx]

        # 遍历关键帧中的每一个 agent 行号
        for row_idx in range(key_frame.shape[0]):

            # 如果关键帧中的第 row_idx 个 agent 在当前帧中也存在
            # 这里依赖 track_token 被编码成从 0 开始的整数 id
            if row_idx in mapped_rows:

                # 将当前帧对应 agent 的状态填入输出数组
                pad_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx]==row_idx]

    # 返回缺失位置为 0 的 agent 轨迹数组
    return pad_agent_trajectories


# 处理自车历史、动态 agent 历史、静态目标信息，生成模型输入格式
def agent_past_process(past_ego_states, past_tracked_objects, tracked_objects_types, num_agents, static_objects, static_objects_types, num_static, max_ped_bike, anchor_ego_state):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input array data of the ego past.
    :param past_tracked_objects: The input array data of agents in the past.
    :param tracked_objects_types: The type of agents in the past.
    :param num_agents: Clip the number of agents.
    :param static_objects: The input array data of static objects in the past.
    :param static_objects_types: The type of static objects in the past.
    :param num_static: Clip the number of static objects.
    :param max_ped_bike: Clip the total number of ped and bike.
    :param anchor_ego_state: Ego current state
    :return: ego, agents, selected_indices, static_objects
    """

    # 动态 agent 最终基础状态维度
    # 包括 x、y、cos heading、sin heading、vx、vy、length、width
    agents_states_dim = 8 # x, y, cos h, sin h, vx, vy, length, width

    # 自车历史状态
    ego_history = past_ego_states

    # 动态 agent 历史状态
    agents = past_tracked_objects

    # 如果存在自车历史状态，则转换到以 anchor_ego_state 为中心的局部坐标系
    if past_ego_states is not None:
        ego = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)

    # 如果没有自车历史状态，则 ego 设为 None
    else:
        ego = None

    # 过滤 agent 历史，只保留当前帧出现过的 agent
    # reverse=True 表示以最后一帧，也就是当前帧作为过滤关键帧
    agent_history = _filter_agents_array(agents, reverse=True)

    # 取最后一帧 agent 类型
    # 因为前面过滤后只保留当前帧存在的 agent，因此类别以当前帧为准
    agent_types = tracked_objects_types[-1]

    # 如果当前帧没有任何动态 agent
    if agent_history[-1].shape[0] == 0:

        # Return zero array when there are no agents in the scene
        # 返回一个空 agent 数组
        # 形状为 [num_frames, 0, agents_states_dim]
        agents_array = np.zeros((len(agent_history), 0, agents_states_dim))

    # 如果存在动态 agent
    else:

        # 保存转换到自车局部坐标系后的每一帧 agent 状态
        local_coords_agent_states = []

        # 对 agent 历史进行 padding，使每一帧 agent 数量和顺序一致
        padded_agent_states = _pad_agent_states(agent_history, reverse=True)

        # 遍历 padding 后的每一帧 agent 状态
        for agent_state in padded_agent_states:

            # 将 agent 状态从全局坐标系转换到自车局部坐标系
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    
        # Calculate yaw rate
        # 初始化动态 agent 基础特征数组
        # 形状为 [num_frames, num_agents, 8]
        agents_array = np.zeros(
            (len(local_coords_agent_states), local_coords_agent_states[0].shape[0], agents_states_dim)
        )

        # 遍历每一帧局部坐标系下的 agent 状态
        for i in range(len(local_coords_agent_states)):

            # 写入局部 x 坐标
            agents_array[i, :, 0] = local_coords_agent_states[i][:, AgentInternalIndex.x()].squeeze()

            # 写入局部 y 坐标
            agents_array[i, :, 1] = local_coords_agent_states[i][:, AgentInternalIndex.y()].squeeze()

            # 用 cos heading 表示航向角
            # 这样可以避免 heading 在 pi/-pi 附近不连续的问题
            agents_array[i, :, 2] = np.cos(local_coords_agent_states[i][:, AgentInternalIndex.heading()].squeeze())

            # 用 sin heading 表示航向角
            agents_array[i, :, 3] = np.sin(local_coords_agent_states[i][:, AgentInternalIndex.heading()].squeeze())

            # 写入局部坐标系下 x 方向速度
            agents_array[i, :, 4] = local_coords_agent_states[i][:, AgentInternalIndex.vx()].squeeze()

            # 写入局部坐标系下 y 方向速度
            agents_array[i, :, 5] = local_coords_agent_states[i][:, AgentInternalIndex.vy()].squeeze()

            # 写入 agent 宽度
            agents_array[i, :, 6] = local_coords_agent_states[i][:, AgentInternalIndex.width()].squeeze()

            # 写入 agent 长度
            agents_array[i, :, 7] = local_coords_agent_states[i][:, AgentInternalIndex.length()].squeeze()

    # 初始化静态目标基础特征数组
    # 每个静态目标 6 维：
    # x、y、cos heading、sin heading、width、length
    static_objects_array = np.zeros((static_objects.shape[0], 6))

    # 如果当前帧存在静态目标
    if static_objects.shape[0] != 0:

        # 将静态目标从全局坐标系转换到自车局部坐标系
        local_coords_static_objects_states = convert_absolute_quantities_to_relative(static_objects, anchor_ego_state, 'static')

        # 写入静态目标局部 x 坐标
        static_objects_array[:, 0] = local_coords_static_objects_states[:, 0]

        # 写入静态目标局部 y 坐标
        static_objects_array[:, 1] = local_coords_static_objects_states[:, 1]

        # 写入 cos heading
        static_objects_array[:, 2] = np.cos(local_coords_static_objects_states[:, 2])

        # 写入 sin heading
        static_objects_array[:, 3] = np.sin(local_coords_static_objects_states[:, 2])

        # 写入静态目标宽度
        static_objects_array[:, 4] = local_coords_static_objects_states[:, 3]

        # 写入静态目标长度
        static_objects_array[:, 5] = local_coords_static_objects_states[:, 4]


    '''
    Post-process the agents array to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
    '''

    # Initialize the result array
    # 初始化最终动态 agent 模型输入数组
    # 形状为 [num_agents, num_frames, 11]
    # 其中 11 = 8 个基础状态特征 + 3 个类别 one-hot 特征
    agents = np.zeros((num_agents, agents_array.shape[0], agents_array.shape[-1] + 3), dtype=np.float32)

    # 计算每个 agent 在最后一帧相对于自车的距离
    # agents_array[-1, :, :2] 表示当前帧所有 agent 的局部 x、y 坐标
    distance_to_ego = np.linalg.norm(agents_array[-1, :, :2], axis=-1)

    # Sort indices by distance
    # 按距离从近到远排序，得到 agent 索引
    sorted_indices = np.argsort(distance_to_ego)

    # Collect the indices of pedestrians and bicycles
    # 从排序后的索引中筛选行人和自行车索引
    ped_bike_indices = [i for i in sorted_indices if agent_types[i] in (TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE)]

    # 从排序后的索引中筛选车辆索引
    vehicle_indices = [i for i in sorted_indices if agent_types[i] == TrackedObjectType.VEHICLE]

    # If the total number of available agents is less than or equal to num_agents, no need to filter further
    # 如果当前 agent 总数不超过 num_agents，则直接选择距离最近的前 num_agents 个
    if len(ped_bike_indices) + len(vehicle_indices) <= num_agents:
        selected_indices = sorted_indices[:num_agents]

    # 如果 agent 总数超过 num_agents，则需要进行类别约束下的筛选
    else:

        # Limit the number of pedestrians and bicycles to max_ped_bike, while retaining the remaining ones for later use
        # 优先限制行人和自行车的数量最多为 max_ped_bike
        selected_ped_bike_indices = ped_bike_indices[:max_ped_bike]

        # 剩余没有被选中的行人和自行车索引
        remaining_ped_bike_indices = ped_bike_indices[max_ped_bike:]

        # Combine the limited pedestrians/bicycles and all available vehicles
        # 先组合受限数量的行人/自行车和所有车辆
        selected_indices = selected_ped_bike_indices + vehicle_indices

        # If the combined selection is still less than num_agents, fill the remaining slots with additional pedestrians and bicycles
        # 如果组合后仍不足 num_agents，则用剩余行人/自行车补齐
        remaining_slots = num_agents - len(selected_indices)

        # 如果还有空位，就继续添加剩余行人/自行车
        if remaining_slots > 0:
            selected_indices += remaining_ped_bike_indices[:remaining_slots]

        # Sort and limit the selected indices to num_agents
        # 最后按照距离重新排序，并截断到 num_agents 个
        selected_indices = sorted(selected_indices, key=lambda idx: distance_to_ego[idx])[:num_agents]

    # Populate the final agents array with the selected agents' features
    # 将选中的 agent 写入最终 agents 数组
    for i, j in enumerate(selected_indices):

        # 写入第 j 个原始 agent 的历史状态特征
        agents[i, :, :agents_array.shape[-1]] = agents_array[:, j, :agents_array.shape[-1]]

        # 如果该 agent 是车辆，则类别 one-hot 为 [1,0,0]
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_array.shape[-1]:] = [1, 0, 0]  # Mark as VEHICLE

        # 如果该 agent 是行人，则类别 one-hot 为 [0,1,0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_array.shape[-1]:] = [0, 1, 0]  # Mark as PEDESTRIAN

        # 否则认为该 agent 是自行车，则类别 one-hot 为 [0,0,1]
        else:  # TrackedObjectType.BICYCLE
            agents[i, :, agents_array.shape[-1]:] = [0, 0, 1]  # Mark as BICYCLE


    # 初始化最终静态目标模型输入数组
    # 形状为 [num_static, 10]
    # 其中 10 = 6 个基础状态特征 + 4 个类别 one-hot 特征
    static_objects = np.zeros((num_static, static_objects_array.shape[-1]+4), dtype=np.float32)

    # 计算每个静态目标到自车的距离
    static_distance_to_ego = np.linalg.norm(static_objects_array[:, :2], axis=-1)

    # 按距离从近到远排序，并最多保留 num_static 个静态目标
    static_indices = list(np.argsort(static_distance_to_ego))[:num_static]

    # 遍历被选中的静态目标
    for i, j in enumerate(static_indices):

        # 写入静态目标基础特征
        static_objects[i, :static_objects_array.shape[-1]] = static_objects_array[j, :static_objects_array.shape[-1]]

        # 如果静态目标是施工区标志，则 one-hot 为 [1,0,0,0]
        if static_objects_types[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, static_objects_array.shape[-1]:] = [1, 0, 0, 0]

        # 如果静态目标是路障，则 one-hot 为 [0,1,0,0]
        elif static_objects_types[j] == TrackedObjectType.BARRIER:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 1, 0, 0]

        # 如果静态目标是交通锥，则 one-hot 为 [0,0,1,0]
        elif static_objects_types[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 1, 0]

        # 其他静态目标类型归为 GENERIC_OBJECT，对应 one-hot 为 [0,0,0,1]
        else:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 0, 1]

    # 如果 ego 不为空，则转换为 float32，方便作为模型输入
    if ego is not None:
        ego = ego.astype(np.float32)

    # 返回：
    # ego：局部坐标系下的自车历史状态
    # agents：固定数量动态 agent 历史特征
    # selected_indices：被选中的动态 agent 原始索引
    # static_objects：固定数量静态目标特征
    return ego, agents, selected_indices, static_objects


# 处理动态 agent 的未来轨迹标签
# 主要用于训练阶段生成周围 agent 的未来预测监督信号
def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    
    # 过滤未来帧 agent，只保留第一帧中出现过的 agent
    # 这样未来轨迹与当前选中的 agent 可以保持对应关系
    agent_future = _filter_agents_array(future_tracked_objects)

    # 保存转换到自车局部坐标系后的未来 agent 状态
    local_coords_agent_states = []

    # 遍历未来每一帧 agent 状态
    for agent_state in agent_future:

        # 将未来 agent 状态从全局坐标系转换到自车局部坐标系
        local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))

    # 对未来 agent 状态做 zero padding
    # 缺失的 agent 未来状态保持为 0
    padded_agent_states = _pad_agent_states_with_zeros(local_coords_agent_states)

    # fill agent features into the array
    # 初始化最终未来轨迹数组
    # 形状为 [num_agents, future_steps, 3]
    # 其中 3 表示未来 x、y、heading
    # padded_agent_states.shape[0]-1 表示去掉当前帧，只保留真正未来帧
    agent_futures = np.zeros(shape=(num_agents, padded_agent_states.shape[0]-1, 3), dtype=np.float32)

    # 遍历 agent_index 中记录的已选 agent 索引
    # i 是最终输出中的 agent 序号
    # j 是该 agent 在 padded_agent_states 中的索引
    for i, j in enumerate(agent_index):

        # 取出该 agent 从第 1 帧开始的未来 x、y、heading
        # 第 0 帧通常是当前帧，因此跳过
        agent_futures[i] = padded_agent_states[1:, j, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]

    # 返回未来 agent 轨迹标签
    return agent_futures