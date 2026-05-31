"""
Module: Coordination Transformation Functions and Numpy-Tensor Transformation
Description: This module contains functions for transforming the coordination to ego-centric coordination and Numpy-Tensor transformation.

Categories:
    1. Ego, agent, static coordination transformation
    2. Map coordination transformation
    3. Numpy-Tensor transformation
"""

# 导入 numpy，用于矩阵运算、三角函数、数组拼接、坐标变换等数值计算
import numpy as np

# 导入 torch，用于将 numpy 数据转换为模型可用的 Tensor，并放到指定 device 上
import torch

# 从 nuPlan 的 agents_preprocessing 工具中导入索引类
# EgoInternalIndex：用于访问自车状态数组中的字段位置，例如 x、y、heading、vx、vy、ax、ay 等
# AgentInternalIndex：用于访问周围动态交通参与者状态数组中的字段位置，例如 x、y、heading、vx、vy 等
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex


# =====================
# 1. Ego, agent, static coordination transformation
# =====================
# 第 1 部分：自车、动态 agent、静态目标的坐标系转换函数
# 主要作用是把全局坐标系下的位置、航向角、速度、加速度等信息转换到以自车为中心的局部坐标系下


# 将 global_states1 表示的一批局部位姿转换到 global_states2 对应的局部坐标系中
# 更直观地说：
# global_states1 中有很多个目标位姿；
# global_states2 是目标参考坐标系，也就是 anchor；
# 该函数会计算 global_states1 相对于 global_states2 的位姿变换矩阵
def _local_to_local_transforms(global_states1, global_states2):
    """
    Converts the global_states1' local coordinates to global_states2's local coordinates.
    """

    # 将 global_states2=[x,y,heading] 转换成 3×3 齐次变换矩阵
    # 该矩阵表示 global_states2 坐标系在全局坐标系下的位置和朝向
    local_xform = _state_se2_array_to_transform_matrix(global_states2)

    # 求 local_xform 的逆矩阵
    # 作用：从全局坐标系转换到 global_states2 对应的局部坐标系
    local_xform_inv = np.linalg.inv(local_xform)

    # 将 global_states1 中的一批 [x,y,heading] 位姿批量转换成 N×3×3 齐次变换矩阵
    transforms = _state_se2_array_to_transform_matrix_batch(global_states1)

    # 左乘 local_xform_inv，相当于把 global_states1 的位姿从全局坐标系转换到 global_states2 的局部坐标系
    transforms = np.matmul(local_xform_inv, transforms)

    # 返回变换后的批量齐次变换矩阵
    return transforms


# 将单个 SE(2) 状态 [x,y,heading] 转换为 3×3 齐次变换矩阵
# SE(2) 表示二维平面中的刚体位姿，包括平移 x、y 和旋转 heading
def _state_se2_array_to_transform_matrix(input_data):


    # 读取 x 坐标，并转成 float 类型
    x: float = float(input_data[0])

    # 读取 y 坐标，并转成 float 类型
    y: float = float(input_data[1])

    # 读取航向角 heading，并转成 float 类型
    h: float = float(input_data[2])

    # 计算航向角的 cos 值，用于构造旋转矩阵
    cosine = np.cos(h)

    # 计算航向角的 sin 值，用于构造旋转矩阵
    sine = np.sin(h)

    # 返回二维刚体变换的齐次矩阵：
    # [ cos(h), -sin(h), x ]
    # [ sin(h),  cos(h), y ]
    # [   0.0,     0.0, 1 ]
    # 其中左上角 2×2 是旋转矩阵，第三列前两项是平移量
    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]]
    )


# 将一批 SE(2) 状态 [x,y,heading] 批量转换为 N×3×3 齐次变换矩阵
# 输入 input_data 的形状通常为 [N,3]
# 每一行表示一个位姿：[x,y,heading]
def _state_se2_array_to_transform_matrix_batch(input_data):

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    # 将原始 [x,y,heading] 转换成 [x,y,cos(heading),sin(heading),1]
    # 这样后续可以通过矩阵乘法一次性拼出齐次变换矩阵中的 9 个元素
    processed_input = np.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            np.cos(input_data[:, 2]),
            np.sin(input_data[:, 2]),
            np.ones_like(input_data[:, 0]),
        )
    )

    # See below for reshaping example
    # 该矩阵用于把 [x,y,cos,sin,1] 映射成齐次变换矩阵展开后的 9 个元素
    # 最终目标形式为：
    # [cos, -sin, x, sin, cos, y, 0, 0, 1]
    reshaping_array = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 array, and then reshapes to a Nx3x3 array
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    # processed_input @ reshaping_array 得到形状为 [N,9] 的矩阵展开形式
    # reshape(-1,3,3) 再把每 9 个元素恢复为一个 3×3 齐次变换矩阵
    return (processed_input @ reshaping_array).reshape(-1, 3, 3)


# 将一批 N×3×3 的齐次变换矩阵转换回 N×3 的 SE(2) 状态数组
# 输出每一行为 [x,y,heading]
def _transform_matrix_to_state_se2_array_batch(input_data):
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 array of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted array.
    """

    # 取出每个 3×3 变换矩阵的第一列
    # 第一列包含 [cos(heading), sin(heading), 0]，可用于恢复航向角
    first_columns = input_data[:, :, 0].reshape(-1, 3)

    # 通过 arctan2(sin, cos) 从旋转矩阵中恢复 heading
    angles = np.arctan2(first_columns[:, 1], first_columns[:, 0])

    # 取出每个变换矩阵的第三列
    # 第三列通常是 [x,y,1]，其中前两项就是平移量
    result = input_data[:, :, 2]

    # 将第三列中的最后一项 1 替换为恢复出的 heading
    # 因此 result 的每一行从 [x,y,1] 变成 [x,y,heading]
    result[:, 2] = angles

    # 返回形状为 [N,3] 的 [x,y,heading]
    return result


# 将全局坐标系下的一批 SE(2) 位姿转换到 local_state 所定义的局部坐标系下
# 常用于把周围车辆、静态障碍物、历史轨迹等转换到自车坐标系
def _global_state_se2_array_to_local(
    global_states, local_state
):
    """
    Transforms the StateSE2 in array from to the frame of reference in local_frame.

    :param global_states: A array of Nx3, where the columns are [x, y, heading].
    :param local_state: A array of [x, y, h] of the frame to which to transform.
    :return: The transformed coordinates.
    """

    # 将 local_state=[x,y,heading] 转换为齐次变换矩阵
    # local_state 通常是当前自车位姿
    local_xform = _state_se2_array_to_transform_matrix(local_state)

    # 求逆矩阵，用于从全局坐标系转换到 local_state 对应的局部坐标系
    local_xform_inv = np.linalg.inv(local_xform)

    # 将 global_states 中的一批全局位姿转换为批量齐次变换矩阵
    transforms = _state_se2_array_to_transform_matrix_batch(global_states)

    # 对每个全局位姿左乘 local_xform_inv
    # 得到它们在 local_state 局部坐标系下的位姿
    transforms = np.matmul(local_xform_inv, transforms)

    # 将变换后的齐次矩阵恢复为 [x,y,heading] 数组
    output = _transform_matrix_to_state_se2_array_batch(transforms)

    # 返回局部坐标系下的位姿
    return output


# 将全局坐标系下的速度向量转换到以 anchor_heading 为朝向的局部坐标系下
# 输入 velocity 的形状通常为 [N,2]，每一行为 [vx,vy]
def _global_velocity_to_local(velocity, anchor_heading):

    # 计算局部坐标系 x 方向速度
    # 这里相当于把全局速度向量投影到自车朝向方向
    velocity_x = velocity[:, 0] * np.cos(anchor_heading) + velocity[:, 1] * np.sin(anchor_heading)

    # 计算局部坐标系 y 方向速度
    # 这里相当于把全局速度向量投影到自车横向方向
    velocity_y = velocity[:, 1] * np.cos(anchor_heading) - velocity[:, 0] * np.sin(anchor_heading)

    # 将局部 vx、vy 重新拼成 [N,2] 数组
    return np.stack([velocity_x, velocity_y], axis=-1)




# 将自车、动态 agent 或静态目标的绝对状态转换为相对于自车的局部状态
# agent_type 用于区分输入状态属于：
# 'ego'：自车历史或未来状态；
# 'agent'：周围动态交通参与者；
# 'static'：静态障碍物或静态目标
def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent or ego history to ego-centric coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """

    # 从 ego_state 中取出自车当前位姿 [x,y,heading]
    # 这个位姿将作为局部坐标系的原点和朝向
    ego_pose = np.array(
        [
            float(ego_state[EgoInternalIndex.x()]),
            float(ego_state[EgoInternalIndex.y()]),
            float(ego_state[EgoInternalIndex.heading()]),
        ],
        dtype=np.float64,
    )

    # 如果转换对象是 ego 自身
    # 通常用于把自车历史轨迹或未来轨迹转换到当前自车坐标系下
    if agent_type == 'ego':

        # 从 agent_state 中取出自车状态的 [x,y,heading] 三列
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]

        # 计算这些自车位姿相对于 ego_pose 的局部变换矩阵
        transforms = _local_to_local_transforms(agent_global_poses, ego_pose)

        # 将局部变换矩阵恢复成 [x,y,heading]
        transformed_poses = _transform_matrix_to_state_se2_array_batch(transforms)

        # 用转换后的局部 x 覆盖原始 agent_state 中的 x
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0]

        # 用转换后的局部 y 覆盖原始 agent_state 中的 y
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1]

        # 用转换后的局部 heading 覆盖原始 agent_state 中的 heading
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2]

        # local vel,acc to local
        # 取出自车状态中的局部速度 [vx,vy]
        agent_local_vel = agent_state[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy()]]

        # 取出自车状态中的局部加速度 [ax,ay]
        agent_local_acc = agent_state[:, [EgoInternalIndex.ax(), EgoInternalIndex.ay()]]

        # 将二维速度 [vx,vy] 扩展成齐次形式 [vx,vy,0]，再增加最后一维，方便与 3×3 变换矩阵相乘
        # 这里第三维使用 0，是因为速度向量只需要旋转，不应该受到平移影响
        agent_local_vel = np.expand_dims(np.concatenate((agent_local_vel, np.zeros((agent_local_vel.shape[0], 1))), axis=-1), axis=-1)

        # 将二维加速度 [ax,ay] 扩展成齐次形式 [ax,ay,0]，再增加最后一维，方便与 3×3 变换矩阵相乘
        # 同样第三维使用 0，表示加速度向量只旋转，不平移
        agent_local_acc = np.expand_dims(np.concatenate((agent_local_acc, np.zeros((agent_local_acc.shape[0], 1))), axis=-1), axis=-1)

        # 使用前面计算得到的 transforms 对速度向量进行坐标变换
        # squeeze(axis=-1) 去掉最后一个大小为 1 的维度
        transformed_vel = np.matmul(transforms, agent_local_vel).squeeze(axis=-1)

        # 使用 transforms 对加速度向量进行坐标变换
        transformed_acc = np.matmul(transforms, agent_local_acc).squeeze(axis=-1)

        # 写回转换后的局部速度 vx
        agent_state[:, EgoInternalIndex.vx()] = transformed_vel[:, 0]

        # 写回转换后的局部速度 vy
        agent_state[:, EgoInternalIndex.vy()] = transformed_vel[:, 1]

        # 写回转换后的局部加速度 ax
        agent_state[:, EgoInternalIndex.ax()] = transformed_acc[:, 0]

        # 写回转换后的局部加速度 ay
        agent_state[:, EgoInternalIndex.ay()] = transformed_acc[:, 1]

    # 如果转换对象是动态交通参与者 agent
    # 例如周围车辆、行人、骑行者等
    elif agent_type == 'agent':

        # 从 agent_state 中取出动态 agent 的全局位姿 [x,y,heading]
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]

        # 从 agent_state 中取出动态 agent 的全局速度 [vx,vy]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]

        # 将动态 agent 的全局位姿转换到 ego_pose 对应的自车局部坐标系
        transformed_poses = _global_state_se2_array_to_local(agent_global_poses, ego_pose)

        # 将动态 agent 的全局速度转换到自车局部坐标系
        transformed_velocities = _global_velocity_to_local(agent_global_velocities, ego_pose[-1])

        # 写回转换后的局部 x
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0]

        # 写回转换后的局部 y
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1]

        # 写回转换后的局部 heading
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2]

        # 写回转换后的局部速度 vx
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0]

        # 写回转换后的局部速度 vy
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1]

    # 如果转换对象是静态目标
    # 静态目标通常只包含位置、朝向、尺寸、类别等信息，不包含速度和加速度
    elif agent_type == 'static':

        # 取出静态目标的全局位姿 [x,y,heading]
        # 这里没有使用 AgentInternalIndex，而是直接取第 0、1、2 列
        agent_global_poses = agent_state[:, [0, 1, 2]]

        # 将静态目标的全局位姿转换到自车局部坐标系
        transformed_poses = _global_state_se2_array_to_local(agent_global_poses, ego_pose)

        # 写回转换后的局部 x
        agent_state[:, 0] = transformed_poses[:, 0]

        # 写回转换后的局部 y
        agent_state[:, 1] = transformed_poses[:, 1]

        # 写回转换后的局部 heading
        agent_state[:, 2] = transformed_poses[:, 2]

    # 返回转换后的 agent_state
    # 注意：该函数是在原数组上原地修改，然后返回同一个数组
    return agent_state


# =====================
# 2. Map coordination transformation
# =====================
# 第 2 部分：地图元素坐标转换函数
# 主要用于将车道线、路线、边界线等地图 polyline 坐标从全局坐标系转换到自车局部坐标系


# 将一组二维坐标点 [x,y] 转换到 anchor_state 定义的局部坐标系下
# 与前面的 SE(2) 位姿转换不同，这里只转换坐标点，不处理 heading
def coordinates_to_local_frame(
    coords, anchor_state, precision = None
):
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <np.array: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate array. If None, then it will be inferred from the input precisions.
    :return: <np.array: num_coords, 2> Transformed coordinates.
    """

    # 检查 coords 的维度是否合法
    # 要求 coords 必须是二维数组，并且第二维大小为 2，即每个点是 [x,y]
    if len(coords.shape) != 2 or coords.shape[1] != 2:

        # 如果形状不满足要求，则直接抛出异常，避免后续矩阵乘法出错
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    # 如果没有显式指定计算精度
    if precision is None:

        # 要求 coords 和 anchor_state 的 dtype 一致
        # 如果两者精度不同，又没有指定 precision，可能导致隐式类型转换问题
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")

        # 使用 coords 自身的数据类型作为中间计算精度
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    # 如果 coords 中没有任何坐标点，则直接返回
    # 这样可以避免空数组在后续 pad 或矩阵运算中触发异常
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    # 将 anchor_state=[x,y,heading] 转换为全局坐标系下的齐次变换矩阵
    transform = _state_se2_array_to_transform_matrix(anchor_state)

    # 对齐次变换矩阵求逆
    # 作用：把全局坐标点转换到 anchor_state 对应的局部坐标系
    transform = np.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    # 将二维点 [x,y] 扩展为齐次坐标 [x,y,1]
    # 这样旋转和平移可以统一通过 3×3 矩阵乘法完成
    coords = np.pad(coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1.0)

    # Perform the transformation, transposing so the shapes match
    # 执行坐标变换
    # coords.T 的形状为 [3,num_coords]，左乘 transform 后仍为 [3,num_coords]
    coords = np.matmul(transform, coords.T)

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    # 转置回 [num_coords,3]
    result = coords.T

    # 去掉齐次坐标的最后一维，只保留 [x,y]
    result = result[:, :2]

    # 返回局部坐标系下的二维点
    return result



# 将一组地图矢量元素的坐标从全局坐标系转换到自车局部坐标系
# coords 的形状通常为 [num_elements,num_points,2]
# 例如：70 条 lane，每条 lane 20 个点，每个点 2 维坐标，则形状为 [70,20,2]
def vector_set_coordinates_to_local_frame(
    coords,
    avails,
    anchor_state,
    output_precision = np.float32,
):
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <np.array: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <np.array: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output array.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """


    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    # 读取地图元素数量和每个元素包含的点数
    num_map_elements, num_points_per_element, _ = coords.shape

    # 将三维数组拉平成二维点集
    # 这样可以复用 coordinates_to_local_frame 对所有点一次性做坐标变换
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    # 将所有地图点从全局坐标系转换到 anchor_state 对应的局部坐标系
    # precision=np.float64 表示中间计算使用双精度，减少坐标变换误差
    coords = coordinates_to_local_frame(coords, anchor_state, precision=np.float64)

    # Reshape to original dimensionality
    # 将二维点集恢复成原来的地图元素结构
    # 形状重新变回 [num_map_elements,num_points_per_element,2]
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    # 将输出转换为指定精度
    # 默认 np.float32，更适合深度学习模型输入，显存占用更小
    coords = coords.astype(output_precision)

    # ignore zero-padded data
    # avails 是有效点 mask
    # 对于 padding 出来的无效地图点，将其坐标重新置为 0，避免模型误认为这些点是真实地图信息
    coords[~avails] = 0.0

    # 返回局部坐标系下的地图元素坐标
    return coords


# =====================
# 3. Numpy-Tensor transformation
# =====================
# 第 3 部分：Numpy 到 PyTorch Tensor 的转换函数
# 用于把数据处理阶段生成的 numpy 数组转换成模型前向推理或训练所需的 torch.Tensor


# 将字典形式的 numpy 数据转换为模型输入 Tensor
# data 通常是一个 dict，key 是特征名称，value 是对应的 numpy 数组
# device 表示 Tensor 要放置的位置，例如 'cpu' 或 'cuda'
def convert_to_model_inputs(data, device):

    # 创建一个新的字典，用于保存转换后的 Tensor 数据
    tensor_data = {}

    # 遍历输入数据字典中的每一个键值对
    for k, v in data.items():

        # 如果 v 是 numpy 数组，并且数据类型是 bool
        # 例如 mask、availability 等布尔型特征
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:

            # 将 bool numpy 数组转换为 torch.bool 类型 Tensor
            # unsqueeze(0) 在最前面增加 batch 维度
            # to(device) 将 Tensor 移动到指定设备上
            tensor_data[k] = torch.tensor(v, dtype=torch.bool).unsqueeze(0).to(device)

        # 其他类型的数据默认转换为 float32 Tensor
        # 例如坐标、速度、加速度、地图点等连续数值特征
        else:

            # 将数据转换为 torch.float32 类型 Tensor
            # unsqueeze(0) 增加 batch 维度，使单个样本变成 batch_size=1 的输入形式
            # to(device) 将 Tensor 移动到 CPU 或 GPU
            tensor_data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)

    # 返回模型可直接使用的输入字典
    return tensor_data