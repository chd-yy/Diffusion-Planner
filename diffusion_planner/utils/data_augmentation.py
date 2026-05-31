# 导入 PyTorch，用于张量计算、批量矩阵乘法、轨迹扰动、轨迹插值等
import torch

# 导入 numpy，用于使用 pi、角度归一化等数值计算
import numpy as np

# 导入类型注解
# List：列表类型
# Optional：可选类型
# Tuple、cast 当前代码中导入但未实际使用
# Union：用于表示 angle 可以是 np.ndarray 或 torch.Tensor
from typing import List, Optional, Tuple, Union, cast

# 获取 nuPlan 中 Pacifica 车辆参数
# 这里主要用到 wheel_base 轴距，用于根据 yaw_rate 反推 steering angle
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


# 需要用五次多项式重新修正的未来轨迹点数
# 20 个点，在 TIME_INTERVAL=0.1s 时，对应未来 2s
NUM_REFINE = 20

# 轨迹修正时间范围，单位为秒
REFINE_HORIZON = 2.0

# 轨迹采样时间间隔，单位为秒
# 0.1s 对应 10Hz 轨迹采样频率
TIME_INTERVAL = 0.1


# 对二维向量进行批量坐标变换
# vector 的最后一维必须是 2，表示二维向量 [x, y]、[vx, vy]、[ax, ay] 或 [cos, sin]
def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    bias: (B, ..., 2)
    """

    # 保存原始 vector 的形状，后面矩阵乘法结束后需要恢复成该形状
    shape = vector.shape

    # 获取 batch size
    B = vector.shape[0]

    # 计算除了 batch 维和最后的二维坐标维之外，还有多少个中间维度
    # 例如 vector.shape=(B, N, T, 2)，则 nexpand=2
    nexpand = vector.ndim - 2

    # 如果传入 bias，则说明当前处理的是“点坐标”
    # 点坐标需要先减去平移量，例如减去当前自车位置 center_xy
    # 如果没有传入 bias，则通常说明当前处理的是“向量”
    # 例如速度、加速度、方向向量，它们只需要旋转，不需要平移
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)

    # 将 vector 拉平成 (B, N, 2)，再转置为 (B, 2, N)
    # 这样可以用 torch.bmm 做批量矩阵乘法
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1) # (B, 2, N1 * N2 ...)

    # transform_mat 的形状是 (B, 2, 2)
    # vector 的形状是 (B, 2, N)
    # torch.bmm 后得到 (B, 2, N)
    # 再 permute 回 (B, N, 2)，最后 reshape 回原始形状
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape) # (B, ..., 2)


# 对 heading 航向角进行批量坐标变换
# heading 是角度标量，不能像 [x, y] 那样直接左乘矩阵
# 所以这里先把 heading 转成方向向量 [cos heading, sin heading]，再旋转，最后用 atan2 恢复角度
def heading_transform(heading, transform_mat):
    """
    heading: (B, ...)
    transform_mat: (B, 2, 2)
    """

    # 获取 batch size
    B = heading.shape[0]

    # 保存 heading 原始形状，后面需要恢复
    shape = heading.shape

    # 计算 heading 除 batch 维外还有多少个维度
    nexpand = heading.ndim - 1

    # 将 heading 展平为 (B, N)，方便统一处理
    heading = heading.reshape(B, -1)

    # 将旋转矩阵 reshape 为 (B, 1, 2, 2)
    # 这样可以和 heading 的 N 维进行广播
    transform_mat = transform_mat.reshape(B, 1, 2, 2)

    # 对方向向量 [cos heading, sin heading] 做旋转
    # 旋转后的 x 分量作为 atan2 的第二个输入
    # 旋转后的 y 分量作为 atan2 的第一个输入
    # 最后 reshape 回原始 heading 形状
    #  transform_mat =
    # [cosθ, -sinθ]
    # [sinθ,  cosθ]
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0] + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0] + torch.sin(heading) * transform_mat[..., 0, 1]
    ).reshape(*shape)


# 状态扰动增强类
# 主要用于训练阶段的数据增强：
# 1. 随机扰动当前自车状态；
# 2. 用五次多项式重新连接扰动后的当前状态和未来轨迹；
# 3. 把自车、邻车、地图、静态目标等全部转换到扰动后的新自车坐标系下。
class StatePerturbation():
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    # 初始化状态扰动器
    def __init__(
        self,
        low: List[float] = [-0., -0.75, -0.35, -1, -0.5, -0.2, -0.1, 0., -0.],
        high: List[float] = [0., 0.75, 0.35, 1, 0.5, 0.2, 0.1, 0., 0.],
        augment_prob: float = 0.5,
        normalize=True,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """
        Initialize the augmentor,
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        """

        # 数据增强概率
        self._augment_prob = augment_prob

        # 是否归一化的标志
        # 当前代码中保存了该变量，但后续没有直接使用
        self._normalize = normalize

        # 指定计算设备，例如 cpu 或 cuda
        self._device = torch.device(device)

        # 扰动下界，转成 tensor 并放到指定设备
        self._low = torch.tensor(low).to(self._device)

        # 扰动上界，转成 tensor 并放到指定设备
        self._high = torch.tensor(high).to(self._device)

        # 获取车辆轴距
        # 后续根据 yaw_rate 和速度估算 steering angle 时使用
        self._wheel_base = get_pacifica_parameters().wheel_base
        
        # 五次多项式修正的时间范围
        self.refine_horizon = REFINE_HORIZON

        # 五次多项式修正的点数
        self.num_refine = NUM_REFINE

        # 轨迹采样时间间隔
        self.time_interval = TIME_INTERVAL
        
        # 五次多项式边界约束使用的终止时间
        # 这里是 2.0 + 0.1 = 2.1
        T = REFINE_HORIZON + TIME_INTERVAL

        # 构造五次多项式系数求解矩阵的逆矩阵
        # 五次多项式形式为：
        # p(t)=a0+a1*t+a2*t^2+a3*t^3+a4*t^4+a5*t^5
        # 6 个系数由起点位置、起点速度、起点加速度、终点位置、终点速度、终点加速度确定
        # 构造五次多项式边界条件矩阵 M 的逆矩阵
        # 五次多项式形式为：
        # p(t)=a0+a1*t+a2*t^2+a3*t^3+a4*t^4+a5*t^5
        #
        # 其一阶导数为：
        # p'(t)=a1+2*a2*t+3*a3*t^2+4*a4*t^3+5*a5*t^4
        #
        # 其二阶导数为：
        # p''(t)=2*a2+6*a3*t+12*a4*t^2+20*a5*t^3
        #
        # 该矩阵对应 6 个边界条件：
        # 第 1 行：p(0)，起点位置约束
        # 第 2 行：p'(0)，起点速度约束
        # 第 3 行：p''(0)，起点加速度约束
        # 第 4 行：p(T)，终点位置约束
        # 第 5 行：p'(T)，终点速度约束
        # 第 6 行：p''(T)，终点加速度约束
        #
        # 矩阵方程为：
        # M * [a0,a1,a2,a3,a4,a5]^T = [p0,v0,acc0,pT,vT,accT]^T
        #
        # 因此：
        # [a0,a1,a2,a3,a4,a5]^T = M^{-1} * [p0,v0,acc0,pT,vT,accT]^T
        #
        # 这里提前保存 M^{-1}，后续可以直接通过矩阵乘法求 x 和 y 方向的多项式系数。
        self.coeff_matrix = torch.linalg.inv(torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ], device=device, dtype=torch.float32))

        # 构造时间幂矩阵
        # torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE) 生成 0.1 到 2.0 的 20 个采样时刻
        # torch.arange(6) 生成幂次 0 到 5
        # 每一行对应 [1, t, t^2, t^3, t^4, t^5]
        self.t_matrix = torch.pow(torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(1), 
                                  torch.arange(6).unsqueeze(0)).to(device=device)  # shape (B, N+1)

    # 让 StatePerturbation 对象可以像函数一样调用
    # 输入模型数据和轨迹标签，输出扰动并重新坐标变换后的数据
    def __call__(self, inputs, ego_future, neighbors_future):

        # 对当前自车状态进行随机扰动
        # aug_flag 表示哪些 batch 样本真正应用扰动
        # aug_ego_current_state 是扰动后的当前自车状态
        aug_flag, aug_ego_current_state = self.augment(inputs)

        # 根据扰动后的当前自车状态，对 ego_future 前 2 秒轨迹进行五次多项式插值修正
        interpolated_ego_future = self.interpolation_future_trajectory(aug_ego_current_state, ego_future)

        # 只替换需要增强的样本的当前自车状态
        inputs['ego_current_state'][aug_flag] = aug_ego_current_state[aug_flag]

        # 只替换需要增强的样本的未来自车轨迹
        ego_future[aug_flag] = interpolated_ego_future[aug_flag]

        # 将输入、ego_future、neighbors_future 全部转换到新的自车中心坐标系
        return self.centric_transform(inputs, ego_future, neighbors_future)
        
    # 对当前自车状态进行随机扰动
    def augment(
        self,
        inputs
    ):
        # Only aug current state

        # 复制当前自车状态，避免直接修改原始输入
        ego_current_state = inputs['ego_current_state'].clone()

        # 获取 batch size
        B = ego_current_state.shape[0]

        # 生成增强标志
        # torch.rand(B) >= self._augment_prob 为 True 的样本会被增强
        # 同时排除低速样本：abs(vx)<2.0 的样本不增强
        aug_flag = (torch.rand(B) >= self._augment_prob).bool().to(self._device) & ~(abs(ego_current_state[:, 4]) < 2.0)

        # 生成 [0,1] 均匀分布随机数
        random_tensor = torch.rand(B, len(self._low)).to(self._device)

        # 将随机数缩放到 [low, high] 区间
        # 对应扰动维度：
        # [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate]
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        # 初始化新状态
        # 9 维对应 [x, y, heading, vx, vy, ax, ay, steering_angle, yaw_rate]
        new_state = torch.zeros((B, 9), dtype=torch.float32).to(self._device)

        # 原始 ego_current_state 格式为：
        # [x, y, cos heading, sin heading, vx, vy, ax, ay, steering angle, yaw rate]
        # 因为原始坐标已经是 ego-centric，所以 x、y、heading 初始按 0 处理
        # 这里把 vx、vy、ax、ay、steering angle、yaw rate 写入 new_state 的后 6 维
        new_state[:, 3:] = ego_current_state[:, 4:10] # x, y, h is 0 because of ego-centric, update vx, vy, ax, ay, steering angle, yaw rate

        # 加上随机扰动
        new_state = new_state + scaled_random_tensor

        # 限制纵向速度不小于 0，避免扰动出倒车速度
        new_state[:, 3] = torch.max(new_state[:, 3], torch.tensor(0.0, device=new_state.device))

        # 限制 yaw_rate，避免扰动后横摆角速度过大
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)


        # 写回扰动后的 x、y
        ego_current_state[:, :2] = new_state[:, :2]

        # 将扰动后的 heading 转成 cos heading
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])

        # 将扰动后的 heading 转成 sin heading
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])

        # 写回扰动后的 vx、vy、ax、ay
        ego_current_state[:, 4:8] = new_state[:, 3:7]

        # 写回扰动后的 steering angle 和 yaw rate
        ego_current_state[:, 8:10] = new_state[:, -2:] # steering angle, yaw rate

        # update steering angle and yaw rate

        # 取当前纵向速度
        cur_velocity = ego_current_state[:, 4]

        # 取当前 yaw_rate
        yaw_rate = ego_current_state[:, 9] 

        # 初始化 steering_angle
        steering_angle = torch.zeros_like(cur_velocity)

        # 初始化新的 yaw_rate
        new_yaw_rate = torch.zeros_like(yaw_rate)

        # 低速 mask
        # 当速度过低时，yaw_rate 和 steering_angle 的估计通常不可靠
        mask = torch.abs(cur_velocity) < 0.2

        # 非低速样本 mask
        not_mask = ~mask

        # 对非低速样本，根据运动学自行车模型反推转向角：
        # yaw_rate = v / L * tan(delta)
        # delta = atan(yaw_rate * L / abs(v))
        steering_angle[not_mask] = torch.atan(yaw_rate[not_mask] * self._wheel_base / torch.abs(cur_velocity[not_mask]))

        # 限制转向角范围，避免异常值
        steering_angle[not_mask] = torch.clamp(steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi)

        # 非低速样本保留 yaw_rate
        new_yaw_rate[not_mask] = yaw_rate[not_mask]


        # 写回修正后的 steering_angle
        ego_current_state[:, 8] = steering_angle

        # 写回修正后的 yaw_rate
        ego_current_state[:, 9] = new_yaw_rate

        # 返回增强标志和扰动后的当前自车状态
        return aug_flag, ego_current_state
    

    # 将角度归一化到 [-pi, pi) 范围内
    def normalize_angle(self, angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    

    # 根据当前自车状态中的 cos heading 和 sin heading 构造二维旋转矩阵
    def get_transform_matrix_batch(self, cur_state):

        # 取出当前 heading 的 cos 和 sin
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        # 构造映射矩阵
        # processed_input=[cos, sin]
        # 乘以 reshaping_tensor 后得到：
        # [cos, sin, -sin, cos]
        # reshape 后得到：
        # [[ cos,  sin],
        #  [-sin,  cos]]
        # 这个矩阵用于把坐标转换到当前自车坐标系下
        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ], dtype=torch.float32
        ).to(processed_input.device)

        # 批量生成 2×2 旋转矩阵
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)
        
    # 将所有输入特征转换到扰动后的自车中心坐标系下
    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor, 
        neighbors_future: torch.Tensor,
    ):

        # 复制当前自车状态
        cur_state = inputs['ego_current_state'].clone()

        # 当前自车位置 [x, y]
        # 点坐标转换时需要先减去该中心点
        center_xy = cur_state[:, :2]

        # 根据当前自车 heading 构造旋转矩阵
        transform_matrix = self.get_transform_matrix_batch(cur_state)
        
        # ego xy
        # 转换自车当前状态中的位置坐标 x、y
        inputs["ego_current_state"][..., :2] = vector_transform(inputs["ego_current_state"][..., :2], transform_matrix, center_xy)

        # ego cos sin
        # 转换自车 heading 的方向向量 [cos, sin]
        inputs["ego_current_state"][..., 2:4] = vector_transform(inputs["ego_current_state"][..., 2:4], 
                                                          transform_matrix)

        # ego vx, vy
        # 转换自车速度向量，只旋转，不平移
        inputs["ego_current_state"][..., 4:6] = vector_transform(inputs["ego_current_state"][..., 4:6], 
                                                          transform_matrix)

        # ego ax, ay
        # 转换自车加速度向量，只旋转，不平移
        inputs["ego_current_state"][..., 6:8] = vector_transform(inputs["ego_current_state"][..., 6:8], 
                                                          transform_matrix)
        
        # ego future xy
        # 转换自车未来轨迹位置
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)

        # 转换自车未来轨迹 heading
        ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)

        # neighbor past xy
        # 构造邻居历史轨迹的 padding mask
        # 如果前 6 维全为 0，则认为该位置是 padding
        mask = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1) == 0

        # 转换邻居历史位置 x、y
        inputs["neighbor_agents_past"][..., :2] = vector_transform(inputs["neighbor_agents_past"][..., :2], 
                                                               transform_matrix, center_xy)

        # neighbor past cos sin
        # 转换邻居历史 heading 的方向向量 [cos, sin]
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(inputs["neighbor_agents_past"][..., 2:4], 
                                                          transform_matrix)

        # neighbor past vx, vy
        # 转换邻居历史速度向量
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(inputs["neighbor_agents_past"][..., 4:6], 
                                                                transform_matrix)

        # 将 padding 位置重新置 0
        # 避免坐标变换后 padding 从全 0 变成非 0
        inputs["neighbor_agents_past"][mask] = 0.
        
        # neighbor future xy
        # 构造邻居未来轨迹的 padding mask
        # 如果 x、y 都为 0，则认为该位置是 padding
        mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0

        # 转换邻居未来位置 x、y
        neighbors_future[..., :2] = vector_transform(neighbors_future[..., :2], transform_matrix, center_xy)

        # 转换邻居未来 heading
        neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)

        # 将 padding 位置重新置 0
        neighbors_future[mask] = 0.

        # lanes
        # 构造 lane 的 padding mask
        # lane 前 8 维通常包括中心线坐标、中心线方向向量、到左右边界的相对向量
        mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0

        # 转换 lane 中心线位置
        inputs["lanes"][..., :2] = vector_transform(inputs["lanes"][..., :2], 
                                                transform_matrix, center_xy)

        # 转换 lane 方向向量
        inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], 
                                                transform_matrix)

        # 转换中心线到左边界的相对向量
        inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], 
                                                transform_matrix)

        # 转换中心线到右边界的相对向量
        inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], 
                                                transform_matrix)

        # 将 padding 位置重新置 0
        inputs["lanes"][mask] = 0.

        # route_lanes
        # 构造 route_lanes 的 padding mask
        mask = torch.sum(torch.ne(inputs["route_lanes"][..., :8], 0), dim=-1) == 0

        # 转换 route_lanes 中心线位置
        inputs["route_lanes"][..., :2] = vector_transform(inputs["route_lanes"][..., :2], 
                                                      transform_matrix, center_xy)

        # 转换 route_lanes 方向向量
        inputs["route_lanes"][..., 2:4] = vector_transform(inputs["route_lanes"][..., 2:4], 
                                                       transform_matrix)

        # 转换 route_lanes 到左边界的相对向量
        inputs["route_lanes"][..., 4:6] = vector_transform(inputs["route_lanes"][..., 4:6], 
                                                       transform_matrix)

        # 转换 route_lanes 到右边界的相对向量
        inputs["route_lanes"][..., 6:8] = vector_transform(inputs["route_lanes"][..., 6:8], 
                                                       transform_matrix)

        # 将 padding 位置重新置 0
        inputs["route_lanes"][mask] = 0.  

        # static objects xy
        # 构造静态目标 padding mask
        mask = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) == 0

        # 转换静态目标位置 x、y
        inputs["static_objects"][..., :2] = vector_transform(inputs["static_objects"][..., :2], 
                                                               transform_matrix, center_xy)

        # static objects cos sin
        # 转换静态目标 heading 的方向向量 [cos, sin]
        inputs["static_objects"][..., 2:4] = vector_transform(inputs["static_objects"][..., 2:4], 
                                                          transform_matrix)

        # 将 padding 位置重新置 0
        inputs["static_objects"][mask] = 0.  

        # 返回坐标变换后的输入、自车未来轨迹和邻居未来轨迹
        return inputs, ego_future, neighbors_future
    
    # 对扰动后的自车未来轨迹进行五次多项式插值修正
    # 目的是让扰动后的当前状态和原始未来轨迹之间平滑衔接
    def interpolation_future_trajectory(self, aug_current_state, ego_future):
        """
        refine future trajectory with quintic spline interpolation
        
        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, 80, 3) future trajectory of the ego vehicle
            
        Returns:
            ego_future: refined future trajectory of the ego vehicle
        """
        
        # 修正的未来轨迹点数
        P = self.num_refine

        # 采样时间间隔
        dt = self.time_interval

        # 修正时间范围
        T = self.refine_horizon

        # batch size
        B = aug_current_state.shape[0]

        # 将时间幂矩阵扩展到 batch 维
        # M_t 形状为 (B, P, 6)
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)

        # 将五次多项式系数矩阵扩展到 batch 维
        # A 形状为 (B, 6, 6)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        # 起点边界条件
        # x0, y0：扰动后的当前自车位置
        # theta0：从扰动后当前位置指向未来第 P/2 个点的方向，用作起点 heading
        # v0：当前速度模长
        # a0：当前加速度模长
        # omega0：当前 yaw_rate
        x0, y0, theta0, v0, a0, omega0 = (
            aug_current_state[:, 0], 
            aug_current_state[:, 1], 
            torch.atan2((ego_future[:, int(P/2), 1] - aug_current_state[:, 1]), (ego_future[:, int(P/2), 0] - aug_current_state[:, 0])), 
            torch.norm(aug_current_state[:, 4:6], dim=-1), 
            torch.norm(aug_current_state[:, 6:8], dim=-1), 
            aug_current_state[:, 9]
        )
        
        # 终点边界条件
        # xT, yT, thetaT：第 P 个未来轨迹点的位置和 heading
        # vT：由第 P 点和第 P-1 点的位置差估计速度
        # aT：由二阶差分估计加速度
        # omegaT：由 heading 差分估计 yaw_rate
        xT, yT, thetaT, vT, aT, omegaT = (
            ego_future[:, P, 0],
            ego_future[:, P, 1],
            ego_future[:, P, 2],
            torch.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1) / dt,
            torch.norm(ego_future[:, P, :2] - 2 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2], dim=-1) / dt**2,
            self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt
        )

        # Boundary conditions
        # x 方向五次多项式的 6 个边界条件：
        # 起点 x、起点 x 方向速度、起点 x 方向加速度、终点 x、终点 x 方向速度、终点 x 方向加速度
        sx = torch.stack([
            x0, 
            v0*torch.cos(theta0), 
            a0*torch.cos(theta0) - v0*torch.sin(theta0)*omega0, 
            xT, 
            vT*torch.cos(thetaT), 
            aT*torch.cos(thetaT) - vT*torch.sin(thetaT)*omegaT
        ], dim=-1)
        
        # y 方向五次多项式的 6 个边界条件：
        # 起点 y、起点 y 方向速度、起点 y 方向加速度、终点 y、终点 y 方向速度、终点 y 方向加速度
        sy = torch.stack([
            y0, 
            v0*torch.sin(theta0), 
            a0*torch.sin(theta0) + v0*torch.cos(theta0)*omega0, 
            yT, 
            vT*torch.sin(thetaT), 
            aT*torch.sin(thetaT) + vT*torch.cos(thetaT)*omegaT
        ], dim=-1)

        # 计算 x 方向五次多项式系数
        ax = A @ sx[:, :, None] # B, 6, 1

        # 计算 y 方向五次多项式系数
        ay = A @ sy[:, :, None] # B, 6, 1

        # 根据五次多项式系数计算插值轨迹 x 坐标
        traj_x = M_t @ ax

        # 根据五次多项式系数计算插值轨迹 y 坐标
        traj_y = M_t @ ay

        # 根据相邻点连线方向计算轨迹 heading
        # 第一个 heading 由第一个插值点和当前扰动位置之间的方向计算
        # 后续 heading 由相邻插值点之间的方向计算
        traj_heading = torch.cat([
            torch.atan2(traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)),
            torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0])
        ], dim=1)

        # 拼接修正后的前 P 个轨迹点和原始 ego_future 从 P 点开始的后续轨迹
        # 前 P 个点使用五次多项式修正结果
        # P 及之后的点保留原始未来轨迹
        return torch.concatenate([torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1), ego_future[:, P:, :]], axis=1)