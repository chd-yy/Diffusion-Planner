# 导入 numpy，用于数组构造、数值计算、三角函数、角度归一化、裁剪等操作
import numpy as np

# 导入 numpy.typing，用于类型注解
# npt.NDArray[np.float32] 表示返回值是 numpy 数组，元素类型期望为 float32
import numpy.typing as npt

# 导入 List 类型注解，用于标注列表类型
from typing import List

# TimePoint 是 nuPlan 中表示时间戳的类型
# 通常包含 time_us，即以微秒为单位的时间
from nuplan.common.actor_state.state_representation import TimePoint

# EgoState 是 nuPlan 中表示自车状态的类
# 其中包含自车位姿、速度、加速度、车辆动力学状态等信息
from nuplan.common.actor_state.ego_state import EgoState

# EgoInternalIndex 是 nuPlan 中用于访问自车状态数组字段索引的工具类
# 例如 x、y、heading、vx、vy、ax、ay 在数组中的列号
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex

# 将绝对坐标系下的位姿序列转换为相对于当前自车坐标系的位姿序列
# 通常用于生成自车未来轨迹标签
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

# 获取 Chrysler Pacifica 车辆参数
# 这里主要用于获取 wheel_base，用于由 yaw_rate 和速度估算转向角
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


# 从 scenario 中提取自车历史状态数组和对应历史时间戳数组
def get_ego_past_array_from_scenario(scenario, num_past_poses, past_time_horizon):
    
    # 获取当前时刻的自车状态
    # scenario.initial_ego_state 通常表示该 scenario 在 iteration=0 时刻的自车状态
    current_ego_state = scenario.initial_ego_state
    
    # 从 scenario 中获取自车过去一段时间的历史轨迹
    # iteration=0 表示以当前 scenario 的初始时刻为参考
    # num_samples=num_past_poses 表示采样多少个过去状态
    # time_horizon=past_time_horizon 表示向过去回看的时间长度
    past_ego_states = scenario.get_ego_past_trajectory(
        iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
    )

    # 将历史自车状态转换成 list，并把当前自车状态拼接到最后
    # 因此 sampled_past_ego_states 包含：
    # 过去 num_past_poses 个状态 + 当前状态
    sampled_past_ego_states = list(past_ego_states) + [current_ego_state]

    # 将 EgoState 对象列表转换成 numpy 数组
    # 每一行对应一个时刻的自车状态
    past_ego_states_array = sampled_past_ego_states_to_array(sampled_past_ego_states)

    
    # 获取过去轨迹对应的时间戳
    # 同样是 iteration=0、num_samples=num_past_poses、time_horizon=past_time_horizon
    # 最后拼接当前 scenario 的 start_time，使时间戳数量与自车状态数量一致
    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
        )
    ) + [scenario.start_time]

    # 定义内部函数：将 TimePoint 列表转换成 numpy 数组
    # 每个 TimePoint 取 time_us 字段，即微秒级时间戳
    def sampled_past_timestamps_to_array(past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:

        # 提取所有时间戳的 time_us
        flat = [t.time_us for t in past_time_stamps]

        # 转换成 int64 类型的 numpy 数组
        # 虽然函数注解中写的是 np.float32，但这里源代码实际返回的是 np.int64
        return np.array(flat, dtype=np.int64)

    # 将 TimePoint 列表转换成时间戳数组
    past_time_stamps_array = sampled_past_timestamps_to_array(past_time_stamps)

    # 返回自车历史状态数组和历史时间戳数组
    return past_ego_states_array, past_time_stamps_array


# 将 EgoState 对象列表转换成固定格式的 numpy 数组
# 输出数组每一行对应一个时刻的自车状态
def sampled_past_ego_states_to_array(past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:

    # 初始化输出数组
    # 行数为历史自车状态数量
    # 列数为 7，对应：
    # x、y、heading、vx、vy、ax、ay
    output = np.zeros((len(past_ego_states), 7), dtype=np.float64)

    # 遍历每一个历史自车状态
    for i in range(0, len(past_ego_states), 1):

        # 写入自车后轴中心点的 x 坐标
        output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x

        # 写入自车后轴中心点的 y 坐标
        output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y

        # 写入自车后轴坐标系的航向角 heading
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading

        # 写入自车后轴处的 x 方向速度
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x

        # 写入自车后轴处的 y 方向速度
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y

        # 写入自车后轴处的 x 方向加速度
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x

        # 写入自车后轴处的 y 方向加速度
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y
        
    # 返回形状为 [num_past_states, 7] 的自车历史状态数组
    return output


# 从 scenario 中提取自车未来轨迹，并转换成相对于当前自车坐标系的形式
def get_ego_future_array_from_scenario(scenario, current_ego_state, num_future_poses, future_time_horizon):

    # 获取自车未来轨迹的绝对状态序列
    # iteration=0 表示从当前 scenario 初始时刻开始
    # num_samples=num_future_poses 表示未来采样点数量
    # time_horizon=future_time_horizon 表示未来预测时间范围
    future_trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0, num_samples=num_future_poses, time_horizon=future_time_horizon
    )

    # Get all future poses of the ego relative to the ego coordinate system
    # 将未来轨迹从全局坐标系转换到当前自车坐标系
    # current_ego_state.rear_axle 是当前自车后轴位姿，作为局部坐标系参考
    # [state.rear_axle for state in future_trajectory_absolute_states] 是未来每一帧自车后轴位姿
    future_trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_ego_state.rear_axle, [state.rear_axle for state in future_trajectory_absolute_states]
    )

    # 返回自车未来相对轨迹
    # 通常每个点包含相对 x、相对 y、相对 heading
    return future_trajectory_relative_poses


# 根据自车历史状态和时间戳，计算当前时刻额外自车状态
# 主要包括：
# 1. 将 heading 转成 cos heading 和 sin heading；
# 2. 根据相邻两帧 heading 差计算 yaw_rate；
# 3. 根据 yaw_rate、速度、轴距估算 steering_angle
def calculate_additional_ego_states(ego_agent_past, time_stamp):

    # transform haeding to cos h, sin h and calculate the steering_angle and yaw_rate for current state

    # 取最后一帧作为当前状态
    #[x,  y,  heading,  vx,  vy,  ax,  ay]
    current_state = ego_agent_past[-1]

    # 取倒数第二帧作为上一时刻状态
    prev_state = ego_agent_past[-2]

    # 计算当前帧与上一帧之间的时间间隔
    # time_stamp 单位是微秒，因此乘以 1e-6 转换成秒
    dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

    # 取当前速度
    # 这里使用 current_state[3]，对应前面数组中的 vx
    cur_velocity = current_state[3]

    # 计算当前帧和上一帧的航向角差
    angle_diff = current_state[2] - prev_state[2]

    # 将角度差归一化到 [-pi, pi) 范围
    # 防止 heading 跨越 pi/-pi 时出现异常大的角度差
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # 根据角度差和时间间隔计算横摆角速度 yaw_rate
    yaw_rate = angle_diff / dt

    # 如果当前车速过低，则认为转向角和 yaw_rate 不可靠
    # 因为低速或接近静止时，航向角微小抖动会导致 yaw_rate 被放大
    if abs(cur_velocity) < 0.2:

        # 车辆几乎停止时，转向角置为 0
        steering_angle = 0.0

        # 车辆几乎停止时，yaw_rate 也置为 0
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable

    # 如果车速足够大，则根据自行车模型近似计算转向角
    else:
        # yaw_rate = v * κ
        # κ = tan(δ) / L，其中 δ 是前轮转向角steering_angle，L 是轴距
        # yaw_rate = v * tan(δ) / L
        # 根据车辆运动学自行车模型：
        # yaw_rate = v / L * tan(delta)
        # 因此：
        # delta = arctan(yaw_rate * L / v)
        # 这里 L 使用 Pacifica 车辆参数中的 wheel_base
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity)
        )

        # 将转向角裁剪到 [-2/3*pi, 2/3*pi] 范围内
        # 防止异常 yaw_rate 或速度导致过大的估计转向角
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)

        # 将 yaw_rate 裁剪到 [-0.95, 0.95] 范围内
        # 防止异常航向角跳变导致 yaw_rate 过大
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

    # 初始化当前时刻的扩展自车状态
    # ego_agent_past.shape[1] 原本是 7 维：
    # x、y、heading、vx、vy、ax、ay
    # 这里额外增加 3 个维度：
    # sin/cos 表示引入后会替换 heading 表示，同时再加入 steering_angle 和 yaw_rate
    # 最终 current 长度为 10
    current = np.zeros((ego_agent_past.shape[1] + 3), dtype=np.float32)

    # 写入当前 x、y 坐标
    current[:2] = current_state[:2]

    # 用 cos heading 表示航向角
    # 相比直接使用 heading，cos/sin 可以避免角度在 pi/-pi 处的不连续问题
    current[2] = np.cos(current_state[2])

    # 用 sin heading 表示航向角
    current[3] = np.sin(current_state[2])

    # 写入当前速度和加速度相关状态
    # current_state[3:7] 对应 vx、vy、ax、ay
    # 写入 current[4:8]
    current[4:8] = current_state[3:7]

    # 写入估算得到的前轮转向角
    current[8] = steering_angle

    # 写入估算得到的横摆角速度
    current[9] = yaw_rate

    # 返回当前扩展自车状态
    return current