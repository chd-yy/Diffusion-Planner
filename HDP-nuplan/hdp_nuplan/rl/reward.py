# dataclass 用于快速定义只保存配置参数的数据类。
#
# 使用 @dataclass 后，NuPlanRewardConfig 会自动获得 __init__ 等方法，
# 因此可以直接写：
#
# config = NuPlanRewardConfig(
#     progress_weight=1.0,
#     collision_weight=10.0,
# )
from dataclasses import dataclass

# Dict：
# 表示字典类型。
#
# Optional[T]：
# 表示变量既可以是 T 类型，也可以是 None。
#
# Tuple：
# 表示函数返回多个固定类型的对象。
from typing import Dict, Optional, Tuple

# PyTorch 用于所有轨迹、距离、奖励的张量计算。
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# NuPlan 奖励函数配置
# ----------------------------------------------------------------------
#
# 这里定义的不是可学习参数，而是人工设置的奖励函数超参数。
#
# 最终奖励总体形式为：
#
# reward
# =
# progress_weight * progress
# - collision_weight * collision_cost
# - route_weight * route_cost
# - comfort_weight * comfort_cost
# - backward_weight * backward_cost
# - imitation_weight * imitation_cost
#
# 因此：
#
# 1. progress 是正奖励；
# 2. collision、route、comfort、backward、imitation 都作为代价扣除。
#
# 这意味着强化学习并不是直接使用 nuPlan 官方完整闭环 scorer，
# 而是在缓存张量上构造一个本地、轻量、可批量计算的近似奖励函数。
@dataclass
class NuPlanRewardConfig:
    """基于 NuPlan 缓存张量的离线候选轨迹奖励配置。"""

    # 相邻轨迹点的时间间隔，单位通常为秒。
    #
    # 默认：
    #
    # dt = 0.1 s
    #
    # 对应 10 Hz 的轨迹采样频率。
    #
    # 后面计算速度、加速度和 jerk 时会使用：
    #
    # velocity
    # =
    # Δposition / dt
    #
    # acceleration
    # =
    # Δvelocity / dt
    #
    # jerk
    # =
    # Δacceleration / dt
    dt: float = 0.1

    # 前进进度奖励权重。
    #
    # progress 越大，最终奖励越高。
    progress_weight: float = 1.0

    # 碰撞/危险接近惩罚权重。
    #
    # 默认值为 10.0，是所有项中最大的权重，
    # 表明安全距离违反是当前奖励函数中最重要的惩罚项。
    collision_weight: float = 10.0

    # 偏离导航路线的惩罚权重。
    route_weight: float = 1.0

    # 舒适性惩罚权重。
    #
    # 当前代码中的舒适性只考虑：
    #
    # 1. 加速度模长超过 acceleration_limit 的部分；
    # 2. jerk 模长超过 jerk_limit 的部分。
    # v2 默认使用 pilot 已验证的较小权重，避免三阶差分产生的 comfort
    # 数值压倒碰撞、路线和进度信号。
    comfort_weight: float = 0.01

    # 沿局部 x 轴倒退的惩罚权重。
    backward_weight: float = 1.0

    # 模仿专家真实未来轨迹的惩罚权重。
    #
    # 默认值为 0.0，
    # 因此默认情况下 imitation_cost 会被计算，
    # 但不会影响最终 reward。
    imitation_weight: float = 0.0

    # v2 中表示有向包围盒之间的安全余量（米），不再表示中心距离阈值。
    collision_distance: float = 0.5

    # 自车和缺失尺寸目标的几何回退值，单位为米。
    ego_width: float = 2.0
    ego_length: float = 4.8
    default_agent_width: float = 2.0
    default_agent_length: float = 4.8

    # route-aligned progress 的尺度归一化距离，单位为米。
    progress_normalization: float = 10.0

    # 单时间步 acceleration/jerk 归一化超限值的截断上限。
    comfort_violation_clip: float = 5.0

    # 路线距离代价的截断上限。
    #
    # 如果某个候选轨迹点距离最近 route 点超过 5 m，
    # 其距离仍按照 5 m 计算。
    #
    # 这样可以防止极端离路线轨迹产生无限大的 route_cost。
    max_route_distance: float = 5.0

    # 舒适性评价中的加速度模长阈值。
    #
    # 当：
    #
    # ||a_t||_2 <= 4 m/s^2
    #
    # 时，该时间步不会产生 acceleration_cost。
    acceleration_limit: float = 4.0

    # 舒适性评价中的 jerk 模长阈值。
    #
    # 当：
    #
    # ||j_t||_2 <= 8
    #
    # 时，该时间步不会产生 jerk_cost。
    jerk_limit: float = 8.0

    # 【论文 HDP-RL】多目标奖励权重，来自论文 Table 6。
    risk_weight: float = 1.0
    follow_weight: float = 3.0
    lane_weight: float = 2.5

    # 【NuPlan 小数据适配，非论文原式】论文依赖 70M 帧 IL 先验抑制停车
    # 退化；mini 数据上的弱先验需要一个有界进度保护项。默认 0 保持论文
    # Eq. (25) 不变，只有实验显式传入正权重时才参与总奖励。
    progress_guard_weight: float = 0.0
    # route-aligned progress 已除以 progress_normalization；0.2 对应默认 2 m。
    progress_guard_stop_tolerance: float = 0.2

    # 【NuPlan 小数据适配，非论文原式】方向约束代价权重。
    #
    # 默认 0 保持论文 Eq. (25) 和历史实验不变；显式设置正值后，从
    # multi-reward 中扣除 direction_cost，防止只优化 lane/progress 时产生
    # “仍靠近车道中心线，但运动方向或车头方向与路线相反”的 reward hacking。
    direction_guard_weight: float = 0.0

    # 方向余弦低于 margin 的部分才产生代价。0 表示只惩罚超过 90 度的
    # 反向运动/朝向；取值范围必须为 [-1, 1]。
    direction_motion_cosine_margin: float = 0.0
    direction_heading_cosine_margin: float = 0.0

    # direction_cost 中运动方向和车头方向两个分量的相对权重。
    direction_motion_weight: float = 1.0
    direction_heading_weight: float = 1.0

    # 位移过小时运动方向不稳定，因此不参与 motion direction 统计，单位为米。
    direction_min_displacement: float = 1e-3

    # 【对齐 NuPlan 官方 driving_direction_compliance】官方指标统计过去 1 秒
    # 沿车道 baseline 的累计进度；反向超过 2 m 记为部分违规，超过 6 m 记为
    # 完全违规。这里用相同时间窗和阈值构造可连续排序的代理 cost。
    direction_time_horizon: float = 1.0
    direction_compliance_threshold: float = 2.0
    direction_violation_threshold: float = 6.0

    # 【NuPlan 适配】论文只说明 TTC/THW/OCC 使用速度自适应 shaping，
    # 未公开具体阈值。以下参数全部开放，便于后续通过验证集标定。
    risk_speed_reference: float = 15.0
    ttc_safe_low_speed: float = 2.0
    ttc_safe_high_speed: float = 4.0
    thw_critical: float = 0.5
    thw_safe_low_speed: float = 1.0
    thw_safe_high_speed: float = 2.0
    occupancy_safe_min: float = 0.5
    occupancy_safe_max: float = 3.0
    occupancy_time_headway: float = 0.2
    rear_end_collision_penalty: float = 0.3

    # 【NuPlan 适配】跟车奖励的 ACC 风格参数。
    follow_time_gap_low_speed: float = 1.0
    follow_time_gap_high_speed: float = 2.0
    follow_min_spacing: float = 2.0
    follow_speed_tolerance: float = 2.0
    follow_comfort_acceleration: float = 2.0
    follow_comfort_deceleration: float = 3.0
    leader_lateral_margin: float = 0.5

    # 【NuPlan 适配】地图缓存含车道中心线和左右边界偏移；当边界缺失时，
    # 使用 1.75 m 作为半车道宽回退值。
    lane_half_width_fallback: float = 1.75
    lane_change_ratio: float = 0.5


# ----------------------------------------------------------------------
# NuPlan 候选轨迹奖励计算器
# ----------------------------------------------------------------------
#
# 该类输入一组候选轨迹：
#
# trajectories.shape
# =
# [B, G, T, 4]
#
# 其中：
#
# B：batch size
# G：同一个场景生成的候选轨迹数量，即 group size
# T：未来规划时间步数
# 4：轨迹状态维度，通常为 [x,y,cos(heading),sin(heading)]
#
# 输出：
#
# rewards.shape
# =
# [B,G]
#
# 即每个场景中的每一条候选轨迹最终得到一个标量奖励。
class NuPlanTensorRewardScorer:
    """使用 NuPlan 预处理张量计算候选轨迹奖励。

    这是 NAVSIM PDM reward 的本地可运行替代层。接口刻意与模型解耦，
    后续可以换成完整 NuPlan/PDM 闭环 scorer，而无需修改 RL 训练器。
    """

    # 初始化奖励计算器。
    def __init__(self, config: Optional[NuPlanRewardConfig] = None):
        # 如果调用方传入 config，就使用指定配置。
        #
        # 如果：
        #
        # config is None
        #
        # 则创建一份默认 NuPlanRewardConfig。
        self.config = config or NuPlanRewardConfig()

    # ------------------------------------------------------------------
    # 碰撞/安全距离代价
    # ------------------------------------------------------------------
    #
    # 输入：
    #
    # trajectories：
    # [B,G,T,4]
    #
    # neighbors_future：
    # 通常为 [B,N,T_n,4]
    #
    # neighbor_mask：
    # 通常为 [B,N,T_n]
    # True 表示该邻车未来状态无效。
    #
    # static_objects：
    # 静态障碍物张量。
    #
    # 输出：
    #
    # collision_cost：
    # [B,G]
    #
    # no_collision：
    # [B,G]
    #
    # 需要特别注意：
    #
    # 当前实现计算的是“轨迹点中心之间的欧氏距离”，
    # 并没有显式考虑车辆长宽、朝向和 bounding box 重叠。
    def _collision_cost_center_distance_legacy(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        static_objects: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 从候选轨迹形状中读取：
        #
        # batch_size = B
        # group_size = G
        # horizon = T
        #
        # 最后一维轨迹状态维度用 _ 忽略。
        batch_size, group_size, horizon, _ = trajectories.shape

        # 自车候选轨迹长度和邻车未来轨迹长度可能不同。
        #
        # 因此只比较双方都存在的时间范围：
        #
        # T_compare
        # =
        # min(T_ego,T_neighbor)
        neighbor_horizon = min(horizon, neighbors_future.shape[-2])

        # 提取自车未来二维坐标。
        #
        # 原 trajectories：
        #
        # [B,G,T,4]
        #
        # trajectories[:, :, None, ...] 增加邻车维度：
        #
        # [B,G,1,T,2]
        #
        # 这样后面可以依靠广播机制同时与所有 N 个邻车比较。
        ego_xy = trajectories[:, :, None, :neighbor_horizon, :2]

        # 提取邻车未来二维坐标。
        #
        # 原形状近似：
        #
        # [B,N,T,4]
        #
        # neighbors_future[:, None, ...] 插入候选轨迹维度：
        #
        # [B,1,N,T,2]
        #
        # 后续与 ego_xy 相减时，会广播为：
        #
        # [B,G,N,T,2]
        neighbor_xy = neighbors_future[:, None, :, :neighbor_horizon, :2]

        # neighbor_mask 中 True 表示该邻车状态无效。
        #
        # 因此使用 ~ 取反：
        #
        # valid=True
        #
        # 表示该邻车对应时间步的数据有效。
        #
        # 插入 group 维度后，逻辑形状为：
        #
        # [B,1,N,T]
        #
        # 随后可以自动广播到 G 条候选轨迹。
        valid = ~neighbor_mask[:, None, :, :neighbor_horizon]

        # ------------------------------------------------------------------
        # 动态障碍物距离
        # ------------------------------------------------------------------

        # ego_xy - neighbor_xy：
        #
        # [B,G,N,T,2]
        #
        # 对最后二维坐标求 L2 范数：
        #
        # d
        # =
        # sqrt(
        #     (x_ego-x_neighbor)^2
        #     +
        #     (y_ego-y_neighbor)^2
        # )
        #
        # 得到：
        #
        # distances.shape
        # =
        # [B,G,N,T]
        distances = torch.linalg.vector_norm(ego_xy - neighbor_xy, dim=-1)

        # 创建与 distances 相同形状的 +∞ 张量。
        #
        # 对无效 padding 邻车状态，我们希望它永远不会成为最小距离，
        # 因此将其距离设置为 +∞。
        inf = torch.full_like(distances, float("inf"))

        # valid=True：
        # 保留真实距离。
        #
        # valid=False：
        # 替换成 +∞。
        distances = torch.where(valid, distances, inf)

        # distances 原形状：
        #
        # [B,G,N,T]
        #
        # flatten(2) 会从第 2 维开始全部展平：
        #
        # [B,G,N*T]
        #
        # 然后 min(dim=-1)：
        #
        # 在所有邻车、所有未来时间步中寻找一个全局最小距离。
        #
        # 因此：
        #
        # min_dynamic[b,g]
        # =
        # min_{n,t}
        # ||p_ego^{b,g,t}-p_neighbor^{b,n,t}||_2
        #
        # 最终：
        #
        # min_dynamic.shape
        # =
        # [B,G]
        min_dynamic = distances.flatten(2).min(dim=-1).values

        # ------------------------------------------------------------------
        # 静态障碍物距离
        # ------------------------------------------------------------------

        # 只有确实传入 static_objects，
        # 并且场景中静态物体维度大于 0 时才计算。
        if static_objects is not None and static_objects.shape[1] > 0:

            # 取静态物体前两个状态量作为二维位置：
            #
            # [x,y]
            #
            # 逻辑形状近似：
            #
            # [B,S,2]
            static_xy = static_objects[..., :2]

            # 判断哪些静态障碍物是真实有效对象。
            #
            # 如果一个静态物体所有特征全部等于 0：
            #
            # static_valid=False
            #
            # 认为它只是 padding。
            #
            # torch.any(...,dim=-1) 后：
            #
            # static_valid.shape
            # =
            # [B,S]
            static_valid = torch.any(static_objects != 0, dim=-1)

            # 计算每一条候选轨迹、每一个未来时间步
            # 到每一个静态障碍物的欧氏距离。
            #
            # trajectories[:, :, :, None, :2]
            #
            # 形状：
            #
            # [B,G,T,1,2]
            #
            # static_xy[:, None, None, :, :]
            #
            # 形状：
            #
            # [B,1,1,S,2]
            #
            # 相减后广播为：
            #
            # [B,G,T,S,2]
            #
            # 对最后坐标维求 L2 范数后：
            #
            # [B,G,T,S]
            static_distances = torch.linalg.vector_norm(
                trajectories[:, :, :, None, :2] - static_xy[:, None, None, :, :],
                dim=-1,
            )

            # 对无效的静态障碍物 padding，将距离替换为 +∞。
            #
            # static_valid：
            #
            # [B,S]
            #
            # 插入两个维度后：
            #
            # [B,1,1,S]
            #
            # 自动广播到：
            #
            # [B,G,T,S]
            static_distances = torch.where(
                static_valid[:, None, None, :],
                static_distances,
                torch.full_like(static_distances, float("inf")),
            )

            # static_distances：
            #
            # [B,G,T,S]
            #
            # flatten(2)：
            #
            # [B,G,T*S]
            #
            # min：
            #
            # 在所有未来时刻、所有静态障碍物之间
            # 找到整条轨迹的最小静态障碍距离：
            #
            # min_static[b,g]
            # =
            # min_{t,s}
            # ||p_ego^{b,g,t}-p_static^{b,s}||_2
            min_static = static_distances.flatten(2).min(dim=-1).values

            # 动态障碍物和静态障碍物分别得到一个最小距离。
            #
            # 最终取二者中更危险的那个：
            #
            # min_distance
            # =
            # min(min_dynamic,min_static)
            min_distance = torch.minimum(min_dynamic, min_static)

        # 如果不存在静态障碍物，
        # 则安全距离完全由动态交通参与者决定。
        else:
            min_distance = min_dynamic

        # 无任何障碍物时 min_distance 为 inf，对应零碰撞代价。

        # ------------------------------------------------------------------
        # 将最小障碍物距离转换成碰撞代价
        # ------------------------------------------------------------------
        #
        # 定义：
        #
        # d_safe
        # =
        # collision_distance
        #
        # d_min
        # =
        # min_distance
        #
        # 当前代价为：
        #
        # collision_cost
        # =
        # ReLU(d_safe-d_min)/d_safe
        #
        # 因此：
        #
        # 当 d_min >= d_safe：
        #
        # collision_cost=0
        #
        # 当 d_min < d_safe：
        #
        # collision_cost
        # =
        # (d_safe-d_min)/d_safe
        #
        # 例如：
        #
        # d_safe=2.5 m
        # d_min=2.0 m
        #
        # collision_cost
        # =
        # (2.5-2.0)/2.5
        # =
        # 0.2
        #
        # 当 d_min=0 时：
        #
        # collision_cost=1
        #
        # 所以这实际上是一个“安全距离违反程度”，
        # 而不仅仅是严格几何碰撞事件。
        collision_cost = torch.relu(
            self.config.collision_distance - min_distance
        ) / self.config.collision_distance

        # 如果场景中没有任何有效障碍物：
        #
        # min_distance=inf
        #
        # 则某些中间计算可能产生非有限值。
        #
        # 对所有非有限 collision_cost 显式替换成 0，
        # 表示没有障碍物时不存在碰撞惩罚。
        collision_cost = torch.where(
            torch.isfinite(collision_cost),
            collision_cost,
            torch.zeros_like(collision_cost),
        )

        # 生成一个二值安全指标。
        #
        # 如果最小距离满足：
        #
        # d_min >= d_safe
        #
        # 则：
        #
        # no_collision=1
        #
        # 否则：
        #
        # no_collision=0
        #
        # 注意这里的 no_collision 实际更准确地说是：
        #
        # “没有进入 collision_distance 定义的危险区域”。
        no_collision = (min_distance >= self.config.collision_distance).float()

        # 返回连续碰撞代价以及二值安全指标。
        return collision_cost, no_collision

    @staticmethod
    def _normalize_direction(direction: torch.Tensor) -> torch.Tensor:
        """归一化二维方向；无效零方向回退为局部 x 正方向。"""
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1
        return torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)

    def _rectangle_signed_separation(
        self,
        ego_xy: torch.Tensor,
        ego_direction: torch.Tensor,
        object_xy: torch.Tensor,
        object_direction: torch.Tensor,
        object_width: torch.Tensor,
        object_length: torch.Tensor,
    ) -> torch.Tensor:
        """使用二维 OBB 分离轴定理计算有符号间距，输出 [B,G,N,T]。

        正值表示两个矩形分离，0 表示接触，负值表示重叠。该近似使用
        rollout 航向和缓存尺寸，比单纯中心距离更符合车辆几何。
        """
        ego_forward = self._normalize_direction(ego_direction)[:, :, None]
        ego_lateral = torch.stack(
            [-ego_forward[..., 1], ego_forward[..., 0]], dim=-1
        )
        object_forward = self._normalize_direction(object_direction)[:, None]
        object_lateral = torch.stack(
            [-object_forward[..., 1], object_forward[..., 0]], dim=-1
        )

        delta = object_xy[:, None] - ego_xy[:, :, None]
        object_half_width = object_width[:, None, :, None] * 0.5
        object_half_length = object_length[:, None, :, None] * 0.5
        ego_half_width = self.config.ego_width * 0.5
        ego_half_length = self.config.ego_length * 0.5

        separations = []
        for axis in (ego_forward, ego_lateral, object_forward, object_lateral):
            center_projection = torch.abs(torch.sum(delta * axis, dim=-1))
            ego_radius = (
                ego_half_length
                * torch.abs(torch.sum(ego_forward * axis, dim=-1))
                + ego_half_width
                * torch.abs(torch.sum(ego_lateral * axis, dim=-1))
            )
            object_radius = (
                object_half_length
                * torch.abs(torch.sum(object_forward * axis, dim=-1))
                + object_half_width
                * torch.abs(torch.sum(object_lateral * axis, dim=-1))
            )
            separations.append(center_projection - ego_radius - object_radius)

        # 矩形只要在任一分离轴上存在正间距就不相交，因此取四轴最大值。
        return torch.stack(separations, dim=-1).max(dim=-1).values

    def _collision_cost(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        static_objects: Optional[torch.Tensor],
        neighbor_agents_past: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用尺寸和航向感知的 OBB 近似计算动态/静态碰撞代价。"""
        batch_size, group_size, horizon, _ = trajectories.shape
        ego_xy = trajectories[..., :2]
        ego_direction = trajectories[..., 2:4]
        inf_result = trajectories.new_full((batch_size, group_size), float("inf"))
        min_dynamic = inf_result

        neighbor_count = neighbors_future.shape[1]
        neighbor_horizon = min(horizon, neighbors_future.shape[-2])
        if neighbor_count > 0 and neighbor_horizon > 0:
            if neighbor_agents_past is not None:
                current_neighbors = neighbor_agents_past[:, :neighbor_count, -1]
                neighbor_width = current_neighbors[..., 6]
                neighbor_length = current_neighbors[..., 7]
                neighbor_width = torch.where(
                    neighbor_width > 0,
                    neighbor_width,
                    torch.full_like(neighbor_width, self.config.default_agent_width),
                )
                neighbor_length = torch.where(
                    neighbor_length > 0,
                    neighbor_length,
                    torch.full_like(neighbor_length, self.config.default_agent_length),
                )
            else:
                neighbor_width = trajectories.new_full(
                    (batch_size, neighbor_count), self.config.default_agent_width
                )
                neighbor_length = trajectories.new_full(
                    (batch_size, neighbor_count), self.config.default_agent_length
                )

            dynamic_separation = self._rectangle_signed_separation(
                ego_xy[..., :neighbor_horizon, :],
                ego_direction[..., :neighbor_horizon, :],
                neighbors_future[..., :neighbor_horizon, :2],
                neighbors_future[..., :neighbor_horizon, 2:4],
                neighbor_width,
                neighbor_length,
            )
            dynamic_separation = torch.where(
                ~neighbor_mask[:, None, :, :neighbor_horizon],
                dynamic_separation,
                torch.full_like(dynamic_separation, float("inf")),
            )
            min_dynamic = dynamic_separation.flatten(2).min(dim=-1).values

        min_static = inf_result
        if static_objects is not None and static_objects.shape[1] > 0:
            static_valid = torch.any(static_objects != 0, dim=-1)
            static_width = torch.where(
                static_objects[..., 4] > 0,
                static_objects[..., 4],
                torch.full_like(static_objects[..., 4], self.config.default_agent_width),
            )
            static_length = torch.where(
                static_objects[..., 5] > 0,
                static_objects[..., 5],
                torch.full_like(static_objects[..., 5], self.config.default_agent_length),
            )
            static_xy = static_objects[..., None, :2].expand(
                -1, -1, horizon, -1
            )
            static_direction = static_objects[..., None, 2:4].expand(
                -1, -1, horizon, -1
            )
            static_separation = self._rectangle_signed_separation(
                ego_xy,
                ego_direction,
                static_xy,
                static_direction,
                static_width,
                static_length,
            )
            static_separation = torch.where(
                static_valid[:, None, :, None],
                static_separation,
                torch.full_like(static_separation, float("inf")),
            )
            min_static = static_separation.flatten(2).min(dim=-1).values

        min_separation = torch.minimum(min_dynamic, min_static)
        margin = max(float(self.config.collision_distance), 1e-6)
        collision_cost = (
            torch.relu(margin - min_separation) / margin
        ).clamp(max=1.0)
        collision_cost = torch.where(
            torch.isfinite(collision_cost),
            collision_cost,
            torch.zeros_like(collision_cost),
        )
        no_collision = (min_separation >= 0).to(trajectories.dtype)
        return collision_cost, no_collision

    # ------------------------------------------------------------------
    # 导航路线偏离代价
    # ------------------------------------------------------------------
    #
    # 对候选轨迹的每一个未来位置，
    # 找到距离导航 route polyline 最近的 route 点，
    # 再对整条轨迹的最近距离取平均。
    #
    # 最终：
    #
    # route_cost ∈ [0,1]
    #
    # 越接近导航路线，route_cost 越小。
    def _route_cost(
        self,
        trajectories: torch.Tensor,
        route_lanes: torch.Tensor,
    ) -> torch.Tensor:

        # trajectories：
        #
        # [B,G,T,4]
        batch_size, group_size, horizon, _ = trajectories.shape

        # 初始化最终路线代价：
        #
        # result.shape=[B,G]
        #
        # new_zeros 会继承 trajectories 的：
        #
        # 1. device；
        # 2. dtype。
        result = trajectories.new_zeros(batch_size, group_size)

        # 提取所有 route lane 点的二维位置：
        #
        # route_lanes[..., :2]
        #
        # 然后将 lane 数量和 lane 内部采样点数量展平成一个点集：
        #
        # route_xy.shape
        # =
        # [B,R,2]
        #
        # 其中 R 表示当前场景所有 route 采样点的总数。
        route_xy = route_lanes[..., :2].reshape(batch_size, -1, 2)

        # 判断 route point 是否有效。
        #
        # 前四维全部为 0 的 route point 被认为是 padding。
        #
        # torch.any(...,dim=-1)：
        #
        # 只要前四个状态中任意一个非零，就认为该 route 点有效。
        #
        # reshape 后：
        #
        # route_valid.shape=[B,R]
        route_valid = torch.any(route_lanes[..., :4] != 0, dim=-1).reshape(
            batch_size, -1
        )

        # 每个场景的有效 route 点数量不同，逐场景计算可避免 padding 参与最短距离。
        for batch_idx in range(batch_size):

            # 取出当前场景中所有有效导航路线点：
            #
            # valid_points.shape
            # =
            # [R_valid,2]
            valid_points = route_xy[batch_idx][route_valid[batch_idx]]

            # 如果当前场景完全没有有效 route 点，
            # 则直接跳过。
            #
            # 此时 result[batch_idx] 保持初始化值 0。
            if valid_points.numel() == 0:
                continue

            # 当前场景有 G 条候选轨迹，每条 T 个点。
            #
            # trajectories[batch_idx, :, :, :2]
            #
            # 原形状：
            #
            # [G,T,2]
            #
            # reshape(-1,2) 后：
            #
            # [G*T,2]
            candidate_points = trajectories[batch_idx, :, :, :2].reshape(-1, 2)

            # torch.cdist 计算两个点集之间的两两欧氏距离。
            #
            # candidate_points：
            #
            # [G*T,2]
            #
            # valid_points：
            #
            # [R_valid,2]
            #
            # 得到：
            #
            # distances.shape
            # =
            # [G*T,R_valid]
            #
            # 其中：
            #
            # distances[i,j]
            # =
            # ||candidate_point_i-route_point_j||_2
            distances = torch.cdist(candidate_points, valid_points)

            # 对每一个候选轨迹点，
            # 找距离它最近的 route 点：
            #
            # d_nearest(p_t)
            # =
            # min_j ||p_t-r_j||_2
            #
            # min(dim=-1) 后：
            #
            # [G*T]
            #
            # 再恢复：
            #
            # [G,T]
            nearest = distances.min(dim=-1).values.reshape(group_size, horizon)

            # 对 route distance 进行截断、平均和归一化。
            #
            # 对每个轨迹点：
            #
            # d_clip
            # =
            # min(d_nearest,d_max)
            #
            # 然后整条轨迹取平均：
            #
            # mean_t(d_clip)
            #
            # 最后除以：
            #
            # d_max=max_route_distance
            #
            # 因此：
            #
            # route_cost
            # =
            # mean_t[
            #     min(d_t,d_max)
            # ]/d_max
            #
            # 当轨迹完全贴合 route：
            #
            # route_cost≈0
            #
            # 当所有轨迹点都至少距离 route 5 m：
            #
            # route_cost=1
            result[batch_idx] = (
                nearest.clamp(max=self.config.max_route_distance).mean(dim=-1)
                / self.config.max_route_distance
            )

        # 返回每个场景、每条候选轨迹的 route_cost。
        return result

    def _route_motion_metrics(
        self,
        trajectories: torch.Tensor,
        route_lanes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算沿最近 route 切向的有符号进度和倒退代价，均为 [B,G]。"""
        batch_size, group_size, horizon, _ = trajectories.shape
        scale = max(float(self.config.progress_normalization), 1e-6)

        # 无有效 route 时回退为原局部 x 定义，保证异常缓存仍可评分。
        progress = trajectories[..., -1, 0] / scale
        backward_cost = torch.relu(
            -torch.diff(trajectories[..., 0], dim=-1)
        ).mean(dim=-1)

        route_xy = route_lanes[..., :2].reshape(batch_size, -1, 2)
        route_vector = route_lanes[..., 2:4].reshape(batch_size, -1, 2)
        route_valid = torch.any(route_lanes[..., :4] != 0, dim=-1).reshape(
            batch_size, -1
        )
        route_valid = route_valid & (
            torch.linalg.vector_norm(route_vector, dim=-1) > 1e-6
        )

        origin = torch.zeros_like(trajectories[:, :, :1, :2])
        displacement = torch.diff(
            torch.cat([origin, trajectories[..., :2]], dim=-2), dim=-2
        )

        for batch_idx in range(batch_size):
            valid_points = route_xy[batch_idx][route_valid[batch_idx]]
            valid_vectors = route_vector[batch_idx][route_valid[batch_idx]]
            if valid_points.numel() == 0:
                continue

            candidate_points = trajectories[batch_idx, :, :, :2].reshape(-1, 2)
            nearest_index = torch.cdist(candidate_points, valid_points).argmin(dim=-1)
            tangent = self._normalize_direction(valid_vectors[nearest_index]).reshape(
                group_size, horizon, 2
            )
            signed_motion = torch.sum(displacement[batch_idx] * tangent, dim=-1)
            progress[batch_idx] = signed_motion.sum(dim=-1) / scale
            backward_cost[batch_idx] = torch.relu(-signed_motion).mean(dim=-1)

        return progress, backward_cost

    def _direction_metrics(
        self,
        trajectories: torch.Tensor,
        route_lanes: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """计算候选轨迹相对路线切向的方向指标，所有输出均为 ``[B,G]``。

        返回值依次为：

        1. ``direction_cost``：运动与航向违规的加权平均，范围 ``[0,1]``；
        2. ``motion_alignment``：有效位移方向与路线切向的平均余弦；
        3. ``heading_alignment``：车头方向与路线切向的平均余弦；
        4. ``reverse_fraction``：有符号位移为负的时间步比例；
        5. ``min_progress_in_1s``：滑动时间窗内最小有符号累计进度，单位米；
        6. ``compliance_score_approx``：按 NuPlan 2 m/6 m 阈值近似的 1/0.5/0 分。

        无有效 route 时使用 ego 局部坐标系正 x 方向作为回退切向，保证异常
        cache 仍能得到有限指标。静止时间步不参与 motion_alignment 和
        reverse_fraction，避免停车场景因数值噪声被误判为逆向。
        """
        batch_size, group_size, horizon, _ = trajectories.shape
        cfg = self.config

        origin = torch.zeros_like(trajectories[:, :, :1, :2])
        displacement = torch.diff(
            torch.cat([origin, trajectories[..., :2]], dim=-2), dim=-2
        )
        displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
        motion_valid = displacement_norm > float(cfg.direction_min_displacement)
        motion_direction = displacement / displacement_norm.clamp_min(1e-6)[..., None]

        # [cos(yaw), sin(yaw)] 可能在异常候选中接近零；只对有效航向归一化。
        heading = trajectories[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1)
        heading_valid = heading_norm > 1e-6
        heading_direction = heading / heading_norm.clamp_min(1e-6)[..., None]

        # 默认回退到 ego 局部正 x 方向。
        tangent = torch.zeros_like(displacement)
        tangent[..., 0] = 1.0

        route_xy = route_lanes[..., :2].reshape(batch_size, -1, 2)
        route_vector = route_lanes[..., 2:4].reshape(batch_size, -1, 2)
        route_valid = torch.any(route_lanes[..., :4] != 0, dim=-1).reshape(
            batch_size, -1
        )
        route_valid = route_valid & (
            torch.linalg.vector_norm(route_vector, dim=-1) > 1e-6
        )

        for batch_idx in range(batch_size):
            valid_points = route_xy[batch_idx][route_valid[batch_idx]]
            valid_vectors = route_vector[batch_idx][route_valid[batch_idx]]
            if valid_points.numel() == 0:
                continue

            candidate_points = trajectories[batch_idx, :, :, :2].reshape(-1, 2)
            nearest_index = torch.cdist(candidate_points, valid_points).argmin(dim=-1)
            tangent[batch_idx] = self._normalize_direction(
                valid_vectors[nearest_index]
            ).reshape(group_size, horizon, 2)

        motion_cosine = torch.sum(motion_direction * tangent, dim=-1)
        heading_cosine = torch.sum(heading_direction * tangent, dim=-1)
        signed_motion = torch.sum(displacement * tangent, dim=-1)

        motion_denominator = motion_valid.sum(dim=-1).clamp_min(1)
        heading_denominator = heading_valid.sum(dim=-1).clamp_min(1)
        motion_alignment = (
            torch.where(motion_valid, motion_cosine, torch.zeros_like(motion_cosine))
            .sum(dim=-1)
            / motion_denominator
        )
        heading_alignment = (
            torch.where(
                heading_valid, heading_cosine, torch.zeros_like(heading_cosine)
            ).sum(dim=-1)
            / heading_denominator
        )
        reverse_fraction = (
            ((motion_cosine < 0) & motion_valid).sum(dim=-1) / motion_denominator
        )

        # 对 cosine∈[-1,1] 使用 margin-aware 归一化，使每个分量严格落在 [0,1]。
        motion_margin = float(cfg.direction_motion_cosine_margin)
        heading_margin = float(cfg.direction_heading_cosine_margin)
        motion_cost_per_step = (
            torch.relu(motion_margin - motion_cosine)
            / max(motion_margin + 1.0, 1e-6)
        ).clamp(0, 1)
        heading_cost_per_step = (
            torch.relu(heading_margin - heading_cosine)
            / max(heading_margin + 1.0, 1e-6)
        ).clamp(0, 1)
        motion_cosine_cost = (
            torch.where(
                motion_valid,
                motion_cost_per_step,
                torch.zeros_like(motion_cost_per_step),
            ).sum(dim=-1)
            / motion_denominator
        )

        # NuPlan 官方实现按过去 time_horizon 秒累计 baseline progress。其
        # n_horizon 切片包含当前帧，因此稳定阶段实际覆盖 n_horizon+1 个采样点。
        # 使用左侧补零的 unfold，可为每个规划点得到同长度的滑窗累计进度。
        n_horizon = max(
            int(round(float(cfg.direction_time_horizon) / float(cfg.dt))), 1
        )
        window_size = min(n_horizon + 1, horizon)
        window_progress = F.pad(
            signed_motion, (window_size - 1, 0)
        ).unfold(-1, window_size, 1).sum(dim=-1)
        min_progress_in_window = window_progress.min(dim=-1).values
        max_negative_progress = torch.relu(-min_progress_in_window)

        compliance_threshold = max(
            float(cfg.direction_compliance_threshold), 1e-6
        )
        violation_threshold = max(
            float(cfg.direction_violation_threshold), compliance_threshold + 1e-6
        )
        # 从发生反向运动开始连续增加，达到官方 2 m 部分违规阈值时饱和为 1。
        # 与只在越过阈值后才惩罚相比，这能在候选尚未违规时提供预防性排序信号。
        progress_window_cost = (
            max_negative_progress / compliance_threshold
        ).clamp(0, 1)
        motion_cost = torch.maximum(motion_cosine_cost, progress_window_cost)
        compliance_score_approx = torch.where(
            max_negative_progress < compliance_threshold,
            torch.ones_like(max_negative_progress),
            torch.where(
                max_negative_progress < violation_threshold,
                torch.full_like(max_negative_progress, 0.5),
                torch.zeros_like(max_negative_progress),
            ),
        )
        heading_cost = (
            torch.where(
                heading_valid,
                heading_cost_per_step,
                torch.zeros_like(heading_cost_per_step),
            ).sum(dim=-1)
            / heading_denominator
        )

        motion_weight = float(cfg.direction_motion_weight)
        heading_weight = float(cfg.direction_heading_weight)
        total_weight = max(motion_weight + heading_weight, 1e-6)
        direction_cost = (
            motion_weight * motion_cost + heading_weight * heading_cost
        ) / total_weight

        return (
            direction_cost.clamp(0, 1),
            motion_alignment.clamp(-1, 1),
            heading_alignment.clamp(-1, 1),
            reverse_fraction.clamp(0, 1),
            min_progress_in_window,
            compliance_score_approx,
        )

    # ------------------------------------------------------------------
    # 舒适性代价
    # ------------------------------------------------------------------
    #
    # 当前舒适性完全根据候选轨迹二维位置有限差分计算：
    #
    # position
    # ->
    # velocity
    # ->
    # acceleration
    # ->
    # jerk
    #
    # 最后只惩罚超过阈值的加速度和 jerk。
    def _comfort_cost_finite_difference_legacy(
        self, trajectories: torch.Tensor
    ) -> torch.Tensor:

        # 提取候选轨迹的二维位置：
        #
        # xy.shape=[B,G,T,2]
        xy = trajectories[..., :2]

        # 使用一阶有限差分计算速度：
        #
        # v_t
        # =
        # (p_{t+1}-p_t)/dt
        #
        # 如果原轨迹长度为 T，
        # 则 velocity 的时间长度变为 T-1。
        velocity = torch.diff(xy, dim=-2) / self.config.dt

        # 对速度再次差分得到加速度：
        #
        # a_t
        # =
        # (v_{t+1}-v_t)/dt
        #
        # 时间长度变为 T-2。
        acceleration = torch.diff(velocity, dim=-2) / self.config.dt

        # 对加速度再次差分得到 jerk：
        #
        # j_t
        # =
        # (a_{t+1}-a_t)/dt
        #
        # 时间长度变为 T-3。
        jerk = torch.diff(acceleration, dim=-2) / self.config.dt

        # ------------------------------------------------------------------
        # 加速度代价
        # ------------------------------------------------------------------
        #
        # 先计算二维加速度模长：
        #
        # ||a_t||_2
        # =
        # sqrt(a_x^2+a_y^2)
        #
        # 然后只惩罚超过阈值的部分：
        #
        # ReLU(
        #     ||a_t||_2-a_limit
        # )
        #
        # 对所有未来时间步取平均，再除以 a_limit：
        #
        # acceleration_cost
        # =
        # mean_t[
        #     ReLU(||a_t||_2-a_limit)
        # ]/a_limit
        #
        # 因此：
        #
        # ||a_t||<=a_limit
        #
        # 不会产生该项惩罚。
        acceleration_cost = torch.relu(
            torch.linalg.vector_norm(acceleration, dim=-1)
            - self.config.acceleration_limit
        ).mean(dim=-1) / self.config.acceleration_limit

        # ------------------------------------------------------------------
        # Jerk 代价
        # ------------------------------------------------------------------
        #
        # 同理：
        #
        # jerk_cost
        # =
        # mean_t[
        #     ReLU(||j_t||_2-j_limit)
        # ]/j_limit
        #
        # 只有 jerk 超过阈值后才产生惩罚。
        jerk_cost = torch.relu(
            torch.linalg.vector_norm(jerk, dim=-1) - self.config.jerk_limit
        ).mean(dim=-1) / self.config.jerk_limit

        # 最终舒适性代价为两项直接相加：
        #
        # comfort_cost
        # =
        # acceleration_cost
        # +
        # jerk_cost
        #
        # 当前代码没有分别为 acceleration 和 jerk 设置额外权重。
        return acceleration_cost + jerk_cost

    def _comfort_cost(
        self,
        trajectories: torch.Tensor,
        ego_current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用当前运动状态补齐差分边界，并截断极端超限值。"""
        xy = trajectories[..., :2]
        origin = torch.zeros_like(xy[..., :1, :])
        velocity = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
        velocity = velocity / self.config.dt

        if ego_current_state is None:
            current_velocity = torch.zeros_like(velocity[:, :, :1])
            current_acceleration = torch.zeros_like(velocity[:, :, :1])
        else:
            current_velocity = ego_current_state[:, None, None, 4:6].expand(
                -1, trajectories.shape[1], -1, -1
            )
            current_acceleration = ego_current_state[:, None, None, 6:8].expand(
                -1, trajectories.shape[1], -1, -1
            )

        acceleration = torch.diff(
            torch.cat([current_velocity, velocity], dim=-2), dim=-2
        ) / self.config.dt
        jerk = torch.diff(
            torch.cat([current_acceleration, acceleration], dim=-2), dim=-2
        ) / self.config.dt

        clip = max(float(self.config.comfort_violation_clip), 0.0)
        acceleration_violation = torch.relu(
            torch.linalg.vector_norm(acceleration, dim=-1)
            - self.config.acceleration_limit
        ) / self.config.acceleration_limit
        jerk_violation = torch.relu(
            torch.linalg.vector_norm(jerk, dim=-1) - self.config.jerk_limit
        ) / self.config.jerk_limit
        if clip > 0:
            acceleration_violation = acceleration_violation.clamp(max=clip)
            jerk_violation = jerk_violation.clamp(max=clip)

        return acceleration_violation.mean(dim=-1) + jerk_violation.mean(dim=-1)

    def _ego_motion(
        self,
        trajectories: torch.Tensor,
        ego_current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回候选轨迹的速度、速度标量和纵向加速度，形状分别为 [B,G,T,2]/[B,G,T]/[B,G,T]。"""
        xy = trajectories[..., :2]
        origin = torch.zeros_like(xy[..., :1, :])
        velocity = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
        velocity = velocity / self.config.dt
        speed = torch.linalg.vector_norm(velocity, dim=-1)

        if ego_current_state is None:
            current_velocity = torch.zeros_like(velocity[..., :1, :])
        else:
            current_velocity = ego_current_state[:, None, None, 4:6].expand(
                -1, trajectories.shape[1], -1, -1
            )
        acceleration = torch.diff(
            torch.cat([current_velocity, velocity], dim=-2), dim=-2
        ) / self.config.dt
        forward = self._normalize_direction(trajectories[..., 2:4])
        longitudinal_acceleration = torch.sum(acceleration * forward, dim=-1)
        return velocity, speed, longitudinal_acceleration

    def _neighbor_geometry(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_agents_past: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """构造动态目标的 OBB 间距、相对位置和速度，核心张量为 [B,G,N,T]。"""
        batch_size, group_size, horizon, _ = trajectories.shape
        neighbor_count = neighbors_future.shape[1]
        horizon = min(horizon, neighbors_future.shape[-2])
        shape = (batch_size, group_size, neighbor_count, horizon)

        if neighbor_count == 0 or horizon == 0:
            empty = trajectories.new_empty(shape)
            return {
                "separation": empty,
                "longitudinal": empty,
                "lateral": empty,
                "valid": torch.empty(shape, dtype=torch.bool, device=trajectories.device),
                "vehicle": torch.empty(shape, dtype=torch.bool, device=trajectories.device),
                "neighbor_velocity": trajectories.new_empty((*shape, 2)),
            }

        if neighbor_agents_past is None:
            width = trajectories.new_full(
                (batch_size, neighbor_count), self.config.default_agent_width
            )
            length = trajectories.new_full(
                (batch_size, neighbor_count), self.config.default_agent_length
            )
            current_xy = neighbors_future[..., :1, :2]
            vehicle = torch.ones(
                batch_size, neighbor_count, dtype=torch.bool, device=trajectories.device
            )
        else:
            current = neighbor_agents_past[:, :neighbor_count, -1]
            width = torch.where(
                current[..., 6] > 0,
                current[..., 6],
                torch.full_like(current[..., 6], self.config.default_agent_width),
            )
            length = torch.where(
                current[..., 7] > 0,
                current[..., 7],
                torch.full_like(current[..., 7], self.config.default_agent_length),
            )
            current_xy = current[..., None, :2]
            # agent feature 8 是 vehicle one-hot；兼容缺少类型维度的旧缓存。
            vehicle = current[..., 8] > 0.5 if current.shape[-1] > 8 else torch.ones_like(width, dtype=torch.bool)

        ego_xy = trajectories[..., :horizon, :2]
        ego_forward = self._normalize_direction(trajectories[..., :horizon, 2:4])
        ego_lateral = torch.stack([-ego_forward[..., 1], ego_forward[..., 0]], dim=-1)
        neighbor_xy = neighbors_future[..., :horizon, :2]
        delta = neighbor_xy[:, None] - ego_xy[:, :, None]
        longitudinal = torch.sum(delta * ego_forward[:, :, None], dim=-1)
        lateral = torch.sum(delta * ego_lateral[:, :, None], dim=-1)

        separation = self._rectangle_signed_separation(
            ego_xy,
            trajectories[..., :horizon, 2:4],
            neighbor_xy,
            neighbors_future[..., :horizon, 2:4],
            width,
            length,
        )
        valid = ~neighbor_mask[:, None, :neighbor_count, :horizon]
        vehicle = vehicle[:, None, :, None].expand_as(valid)

        neighbor_velocity = torch.diff(
            torch.cat([current_xy, neighbor_xy], dim=-2), dim=-2
        ) / self.config.dt
        neighbor_velocity = neighbor_velocity[:, None].expand(
            -1, group_size, -1, -1, -1
        )
        return {
            "separation": separation,
            "longitudinal": longitudinal,
            "lateral": lateral,
            "valid": valid,
            "vehicle": vehicle,
            "neighbor_velocity": neighbor_velocity,
            "width": width,
            "length": length,
        }

    @staticmethod
    def _masked_min(values: torch.Tensor, valid: torch.Tensor, dims) -> torch.Tensor:
        """无有效对象时返回 1；否则在指定维度取最保守的最小分数。"""
        masked = torch.where(valid, values, torch.ones_like(values))
        for dim in sorted((dims if isinstance(dims, tuple) else (dims,)), reverse=True):
            masked = masked.min(dim=dim).values
        return masked

    def _risk_reward(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        static_objects: Optional[torch.Tensor],
        neighbor_agents_past: Optional[torch.Tensor],
        ego_current_state: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """论文式连续风险奖励：对 TTC、THW、OCC 在对象和时间上取最小值。"""
        cfg = self.config
        batch_size, group_size, horizon, _ = trajectories.shape
        ego_velocity, ego_speed, _ = self._ego_motion(trajectories, ego_current_state)
        geometry = self._neighbor_geometry(
            trajectories, neighbors_future, neighbor_mask, neighbor_agents_past
        )

        dynamic_horizon = geometry["separation"].shape[-1]
        if geometry["separation"].numel() == 0:
            ttc_reward = trajectories.new_ones(batch_size, group_size)
            thw_reward = trajectories.new_ones(batch_size, group_size)
            safety_reward = trajectories.new_ones(batch_size, group_size)
        else:
            ego_velocity_dynamic = ego_velocity[..., :dynamic_horizon, :][:, :, None]
            delta = (
                neighbors_future[:, None, :, :dynamic_horizon, :2]
                - trajectories[:, :, None, :dynamic_horizon, :2]
            )
            line_of_sight = self._normalize_direction(delta)
            relative_velocity = geometry["neighbor_velocity"] - ego_velocity_dynamic
            closing_speed = -torch.sum(relative_velocity * line_of_sight, dim=-1)
            ttc = geometry["separation"].clamp_min(0) / closing_speed.clamp_min(1e-3)

            speed_ratio = (
                ego_speed[..., :dynamic_horizon] / cfg.risk_speed_reference
            ).clamp(0, 1)
            safe_ttc = cfg.ttc_safe_low_speed + speed_ratio * (
                cfg.ttc_safe_high_speed - cfg.ttc_safe_low_speed
            )
            ttc_score = (ttc / safe_ttc[:, :, None]).clamp(0, 1)
            ttc_score = torch.where(
                closing_speed > 1e-3, ttc_score, torch.ones_like(ttc_score)
            )

            # 非响应式 replay 中，位于自车后方的碰撞按论文减弱为 0.3 代价。
            overlap = geometry["separation"] <= 0
            rear_end = geometry["longitudinal"] < 0
            collision_score = torch.where(
                rear_end,
                torch.full_like(ttc_score, 1.0 - cfg.rear_end_collision_penalty),
                torch.zeros_like(ttc_score),
            )
            ttc_score = torch.where(overlap, collision_score, ttc_score)
            ttc_reward = self._masked_min(ttc_score, geometry["valid"], (2, 3))

            if neighbor_agents_past is None:
                half_length = trajectories.new_full(
                    (batch_size, geometry["longitudinal"].shape[2]),
                    cfg.default_agent_length * 0.5,
                )
            else:
                half_length = geometry["length"] * 0.5
            lane_corridor = (
                cfg.ego_width * 0.5
                + geometry["width"][:, None, :, None] * 0.5
                + cfg.leader_lateral_margin
            )
            leader = (
                geometry["valid"]
                & geometry["vehicle"]
                & (geometry["longitudinal"] > 0)
                & (geometry["lateral"].abs() <= lane_corridor)
            )
            bumper_gap = (
                geometry["longitudinal"]
                - cfg.ego_length * 0.5
                - half_length[:, None, :, None]
            ).clamp_min(0)
            dynamic_speed = ego_speed[:, :, None, :dynamic_horizon]
            time_headway = bumper_gap / dynamic_speed.clamp_min(0.1)
            safe_thw = cfg.thw_safe_low_speed + speed_ratio * (
                cfg.thw_safe_high_speed - cfg.thw_safe_low_speed
            )
            thw_score = (
                (time_headway - cfg.thw_critical)
                / (safe_thw[:, :, None] - cfg.thw_critical).clamp_min(1e-3)
            ).clamp(0, 1)
            thw_score = torch.where(
                dynamic_speed > 0.1,
                thw_score,
                torch.ones_like(thw_score),
            )
            thw_reward = self._masked_min(thw_score, leader, (2, 3))

            collision_penalty = torch.where(
                rear_end,
                torch.full_like(ttc_score, cfg.rear_end_collision_penalty),
                torch.ones_like(ttc_score),
            )
            collision_penalty = torch.where(overlap, collision_penalty, torch.zeros_like(collision_penalty))
            max_dynamic_penalty = torch.where(
                geometry["valid"], collision_penalty, torch.zeros_like(collision_penalty)
            ).flatten(2).max(dim=-1).values
            safety_reward = 1.0 - max_dynamic_penalty

        occupancy_reward = trajectories.new_ones(batch_size, group_size)
        if static_objects is not None and static_objects.shape[1] > 0:
            static_valid = torch.any(static_objects != 0, dim=-1)
            static_width = torch.where(
                static_objects[..., 4] > 0,
                static_objects[..., 4],
                torch.full_like(static_objects[..., 4], cfg.default_agent_width),
            )
            static_length = torch.where(
                static_objects[..., 5] > 0,
                static_objects[..., 5],
                torch.full_like(static_objects[..., 5], cfg.default_agent_length),
            )
            static_xy = static_objects[..., None, :2].expand(-1, -1, horizon, -1)
            static_direction = static_objects[..., None, 2:4].expand(-1, -1, horizon, -1)
            separation = self._rectangle_signed_separation(
                trajectories[..., :2], trajectories[..., 2:4],
                static_xy, static_direction, static_width, static_length,
            )
            safe_distance = (
                cfg.occupancy_safe_min + cfg.occupancy_time_headway * ego_speed
            ).clamp(max=cfg.occupancy_safe_max)
            occupancy_score = (
                separation.clamp_min(0) / safe_distance[:, :, None].clamp_min(1e-3)
            ).clamp(0, 1)
            static_valid = static_valid[:, None, :, None].expand_as(occupancy_score)
            occupancy_reward = self._masked_min(occupancy_score, static_valid, (2, 3))
            static_collision = torch.where(
                static_valid, separation <= 0, torch.zeros_like(static_valid)
            ).flatten(2).any(dim=-1)
            safety_reward = torch.where(static_collision, torch.zeros_like(safety_reward), safety_reward)

        risk_reward = torch.minimum(torch.minimum(ttc_reward, thw_reward), occupancy_reward)
        risk_reward = torch.minimum(risk_reward, safety_reward).clamp(0, 1)
        return {
            "risk_reward": risk_reward,
            "safety_reward": safety_reward.clamp(0, 1),
            "ttc_reward": ttc_reward.clamp(0, 1),
            "thw_reward": thw_reward.clamp(0, 1),
            "occupancy_reward": occupancy_reward.clamp(0, 1),
        }

    def _following_reward(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_agents_past: Optional[torch.Tensor],
        ego_current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """论文式 ACC 跟车奖励；没有前车时返回中性值 1。"""
        cfg = self.config
        batch_size, group_size, _, _ = trajectories.shape
        geometry = self._neighbor_geometry(
            trajectories, neighbors_future, neighbor_mask, neighbor_agents_past
        )
        if geometry["separation"].numel() == 0:
            return (
                trajectories.new_ones(batch_size, group_size),
                trajectories.new_zeros(batch_size, group_size),
            )

        horizon = geometry["separation"].shape[-1]
        ego_velocity, ego_speed, longitudinal_acceleration = self._ego_motion(
            trajectories, ego_current_state
        )
        lane_corridor = (
            cfg.ego_width * 0.5
            + geometry["width"][:, None, :, None] * 0.5
            + cfg.leader_lateral_margin
        )
        leader_valid = (
            geometry["valid"]
            & geometry["vehicle"]
            & (geometry["longitudinal"] > 0)
            & (geometry["lateral"].abs() <= lane_corridor)
        )
        gap = (
            geometry["longitudinal"]
            - cfg.ego_length * 0.5
            - geometry["length"][:, None, :, None] * 0.5
        ).clamp_min(0)
        candidate_gap = torch.where(
            leader_valid, gap, torch.full_like(gap, float("inf"))
        )
        leader_gap, leader_index = candidate_gap.min(dim=2)
        has_leader = torch.isfinite(leader_gap)
        leader_gap = torch.where(has_leader, leader_gap, torch.zeros_like(leader_gap))

        neighbor_velocity = geometry["neighbor_velocity"]
        gather_index = leader_index[:, :, None, :, None].expand(-1, -1, 1, -1, 2)
        leader_velocity = torch.gather(neighbor_velocity, 2, gather_index).squeeze(2)
        ego_forward = self._normalize_direction(trajectories[..., :horizon, 2:4])
        leader_speed = torch.sum(leader_velocity * ego_forward, dim=-1).clamp_min(0)
        speed = ego_speed[..., :horizon]
        speed_ratio = (speed / cfg.risk_speed_reference).clamp(0, 1)
        ideal_time_gap = cfg.follow_time_gap_low_speed + speed_ratio * (
            cfg.follow_time_gap_high_speed - cfg.follow_time_gap_low_speed
        )

        actual_time_gap = leader_gap / speed.clamp_min(0.1)
        gap_score = (1.0 - (actual_time_gap - ideal_time_gap).abs() / ideal_time_gap.clamp_min(1e-3)).clamp(0, 1)
        target_spacing = cfg.follow_min_spacing + ideal_time_gap * speed
        spacing_score = (1.0 - (leader_gap - target_spacing).abs() / target_spacing.clamp_min(1.0)).clamp(0, 1)
        speed_tolerance = torch.maximum(
            torch.full_like(speed, cfg.follow_speed_tolerance), 0.25 * speed
        )
        speed_score = (1.0 - (speed - leader_speed).abs() / speed_tolerance).clamp(0, 1)

        acceleration = longitudinal_acceleration[..., :horizon]
        acceleration_violation = (
            torch.relu(acceleration - cfg.follow_comfort_acceleration)
            / max(cfg.follow_comfort_acceleration, 1e-3)
            + torch.relu(-cfg.follow_comfort_deceleration - acceleration)
            / max(cfg.follow_comfort_deceleration, 1e-3)
        )
        comfort_score = (1.0 - acceleration_violation).clamp(0, 1)

        per_step = (gap_score + spacing_score + speed_score + comfort_score) * 0.25
        valid_count = has_leader.sum(dim=-1)
        reward = (per_step * has_leader).sum(dim=-1) / valid_count.clamp_min(1)
        reward = torch.where(valid_count > 0, reward, torch.ones_like(reward))
        leader_fraction = valid_count.to(trajectories.dtype) / max(horizon, 1)
        return reward.clamp(0, 1), leader_fraction

    def _lane_reward(
        self,
        trajectories: torch.Tensor,
        lanes: torch.Tensor,
        ego_future: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按最近车道中心线距离计算 lane reward，并屏蔽专家换道/离道场景。"""
        cfg = self.config
        batch_size, group_size, horizon, _ = trajectories.shape
        result = trajectories.new_ones(batch_size, group_size)
        mask = trajectories.new_zeros(batch_size, group_size)
        lane_xy = lanes[..., :2].reshape(batch_size, -1, 2)
        lane_vector = lanes[..., 2:4].reshape(batch_size, -1, 2)
        lane_valid = torch.any(lanes[..., :4] != 0, dim=-1).reshape(batch_size, -1)

        if lanes.shape[-1] >= 8:
            left_width = torch.linalg.vector_norm(lanes[..., 4:6], dim=-1)
            right_width = torch.linalg.vector_norm(lanes[..., 6:8], dim=-1)
            half_width = (left_width + right_width) * 0.5
            half_width = torch.where(
                half_width > 0.5,
                half_width,
                torch.full_like(half_width, cfg.lane_half_width_fallback),
            ).reshape(batch_size, -1)
        else:
            half_width = trajectories.new_full(
                lane_valid.shape, cfg.lane_half_width_fallback
            )

        for batch_idx in range(batch_size):
            valid = lane_valid[batch_idx]
            if not torch.any(valid):
                continue
            centers = lane_xy[batch_idx][valid]
            tangents = self._normalize_direction(lane_vector[batch_idx][valid])
            widths = half_width[batch_idx][valid]

            points = trajectories[batch_idx, :, :, :2].reshape(-1, 2)
            distances = torch.cdist(points, centers)
            nearest_distance, nearest_index = distances.min(dim=-1)
            candidate_width = widths[nearest_index].clamp_min(0.5)
            candidate_score = (1.0 - nearest_distance / candidate_width).clamp(0, 1)
            result[batch_idx] = candidate_score.reshape(group_size, horizon).mean(dim=-1)

            use_lane_reward = True
            if ego_future is not None:
                expert = ego_future[batch_idx, :horizon, :2]
                expert_distance, expert_index = torch.cdist(expert, centers).min(dim=-1)
                expert_width = widths[expert_index].clamp_min(0.5)
                expert_tangent = tangents[expert_index]
                expert_lateral = torch.stack(
                    [-expert_tangent[..., 1], expert_tangent[..., 0]], dim=-1
                )
                expert_offset = torch.sum(
                    (expert - centers[expert_index]) * expert_lateral, dim=-1
                )
                expert_off_lane = torch.any(expert_distance > expert_width)
                lateral_shift = (expert_offset[-1] - expert_offset[0]).abs()
                lane_change = lateral_shift > cfg.lane_change_ratio * torch.minimum(
                    expert_width[0], expert_width[-1]
                )
                use_lane_reward = not bool((expert_off_lane | lane_change).item())

            if use_lane_reward:
                mask[batch_idx] = 1
            else:
                # 被 mask 的场景给所有候选相同中性值，不改变组内优势排序。
                result[batch_idx] = 1

        return result.clamp(0, 1), mask

    # ------------------------------------------------------------------
    # 整体轨迹奖励函数
    # ------------------------------------------------------------------
    #
    # @torch.no_grad() 表示整个奖励计算过程不构建自动求导图。
    #
    # 这非常符合当前 HDP-RL 的设计：
    #
    # reward 并不是通过：
    #
    # ∂reward/∂trajectory
    #
    # 直接反向传播给扩散模型。
    #
    # 而是先将奖励转换为样本权重，
    # 再通过 reward-weighted diffusion regression 更新模型。
    #
    # 因此奖励函数本身不需要可微。
    @torch.no_grad()
    def _legacy_reward(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        route_lanes: torch.Tensor,
        static_objects: Optional[torch.Tensor] = None,
        ego_future: Optional[torch.Tensor] = None,
        neighbor_agents_past: Optional[torch.Tensor] = None,
        ego_current_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # 明确要求候选轨迹必须是四维张量：
        #
        # trajectories.shape
        # =
        # [B,G,T,4]
        #
        # 否则说明上游 sample() 输出格式不符合当前奖励器接口。
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, G, T, 4]")

        # ------------------------------------------------------------------
        # 1. 碰撞/危险距离代价
        # ------------------------------------------------------------------
        #
        # collision_cost：
        #
        # 连续安全距离违反程度。
        #
        # no_collision：
        #
        # 0/1 二值安全指标。
        collision_cost, no_collision = self._collision_cost(
            trajectories,
            neighbors_future,
            neighbor_mask,
            static_objects,
            neighbor_agents_past,
        )

        # ------------------------------------------------------------------
        # 2. 导航路线偏离代价
        # ------------------------------------------------------------------
        route_cost = self._route_cost(trajectories, route_lanes)

        # ------------------------------------------------------------------
        # 3. 舒适性代价
        # ------------------------------------------------------------------
        comfort_cost = self._comfort_cost(trajectories, ego_current_state)

        # NuPlan 缓存以当前自车为局部原点，x 轴通常朝当前车头方向。

        # ------------------------------------------------------------------
        # 4. 前进进度奖励
        # ------------------------------------------------------------------
        #
        # trajectories[..., -1, 0]：
        #
        # 取轨迹最后一个未来时间点的 x 坐标。
        #
        # 在当前局部坐标定义下：
        #
        # x>0
        #
        # 通常表示车辆向前运动。
        #
        # 因此：
        #
        # progress
        # =
        # x_final / 10
        #
        # 这里除以 10.0 只是一个人工尺度归一化。
        #
        # 例如：
        #
        # 最终前进 20 m：
        #
        # progress=2
        #
        # 最终前进 5 m：
        #
        # progress=0.5
        #
        # 该项不是严格意义上的 route progress，
        # 而是局部 x 方向上的终点位移近似。
        # reward v2 使用每个轨迹点最近 route 向量的切向投影，而不是终点局部 x，
        # 因而转弯场景中的合法前进也能得到正进度。
        progress, backward_cost = self._route_motion_metrics(
            trajectories, route_lanes
        )

        # ------------------------------------------------------------------
        # 5. 倒退惩罚
        # ------------------------------------------------------------------
        #
        # trajectories[...,0]：
        #
        # 取所有未来时间点的局部 x 坐标：
        #
        # [B,G,T]
        #
        # torch.diff(...,dim=-1)：
        #
        # Δx_t
        # =
        # x_{t+1}-x_t
        #
        # 如果车辆正常向前：
        #
        # Δx_t>=0
        #
        # 那么：
        #
        # -Δx_t<=0
        #
        # ReLU(-Δx_t)=0
        #
        # 不产生惩罚。
        #
        # 如果某一步向后：
        #
        # Δx_t<0
        #
        # 那么：
        #
        # ReLU(-Δx_t)>0
        #
        # 产生倒退惩罚。
        #
        # 最终对所有时间步取平均：
        #
        # backward_cost
        # =
        # mean_t[
        #     ReLU(-(x_{t+1}-x_t))
        # ]
        # backward_cost 已由上面的 route-aligned signed motion 同时给出；
        # 无有效 route 时函数内部才回退到局部 x 差分。

        # ------------------------------------------------------------------
        # 6. 模仿专家轨迹代价
        # ------------------------------------------------------------------
        #
        # 如果传入 ego_future，
        # 就计算候选轨迹与数据集中真实自车未来轨迹的平均位置误差。
        if ego_future is not None:

            # 候选轨迹和专家轨迹的时间长度可能不同，
            # 因此只比较共同存在的 horizon。
            horizon = min(trajectories.shape[-2], ego_future.shape[-2])

            # 候选轨迹位置：
            #
            # trajectories[..., :horizon, :2]
            #
            # 逻辑形状：
            #
            # [B,G,T,2]
            #
            # ego_future 原形状：
            #
            # [B,T,4]
            #
            # ego_future[:,None,...] 插入 group 维：
            #
            # [B,1,T,2]
            #
            # 广播后可同时与 G 条候选轨迹比较。
            #
            # 每个时间步的 imitation distance：
            #
            # d_t
            # =
            # ||p_candidate_t-p_expert_t||_2
            #
            # 然后沿时间取平均：
            #
            # imitation_cost
            # =
            # mean_t(d_t)
            #
            # 这实际上类似一个 ADE。
            imitation_cost = torch.linalg.vector_norm(
                trajectories[..., :horizon, :2]
                - ego_future[:, None, :horizon, :2],
                dim=-1,
            ).mean(dim=-1)

        # 如果没有专家未来轨迹，
        # 则 imitation_cost 直接设置成与 progress 同形状的 0。
        else:
            imitation_cost = torch.zeros_like(progress)

        # 为后续公式使用配置简写 cfg。
        cfg = self.config

        # ------------------------------------------------------------------
        # 7. 汇总最终奖励
        # ------------------------------------------------------------------
        #
        # 最终 reward 为：
        #
        # R
        # =
        # w_progress * R_progress
        # - w_collision * C_collision
        # - w_route * C_route
        # - w_comfort * C_comfort
        # - w_backward * C_backward
        # - w_imitation * C_imitation
        #
        # 具体写成：
        #
        # R
        # =
        # progress_weight * progress
        # - collision_weight * collision_cost
        # - route_weight * route_cost
        # - comfort_weight * comfort_cost
        # - backward_weight * backward_cost
        # - imitation_weight * imitation_cost
        #
        # 所以：
        #
        # reward 越大
        #
        # 表示该候选轨迹在当前人工指标下越优。
        #
        # 之后 reward_weighted_diffusion_loss 会比较
        # 同一场景 G 条候选轨迹的 reward，
        # 并增加高奖励轨迹在扩散训练中的权重。
        rewards = (
            cfg.progress_weight * progress
            - cfg.collision_weight * collision_cost
            - cfg.route_weight * route_cost
            - cfg.comfort_weight * comfort_cost
            - cfg.backward_weight * backward_cost
            - cfg.imitation_weight * imitation_cost
        )

        # ------------------------------------------------------------------
        # 保存所有分项指标
        # ------------------------------------------------------------------
        #
        # details 不仅用于日志，
        # 还可以帮助判断最终奖励提高到底来自哪里。
        #
        # 例如：
        #
        # reward 上升可能是：
        #
        # 1. progress 增大；
        # 2. collision_cost 减小；
        # 3. route_cost 减小；
        # 4. comfort_cost 减小。
        #
        # 如果只记录总 reward，
        # 很难判断模型是否通过某种 reward hacking 提高了分数。
        details = {
            # 最终综合奖励。
            "reward": rewards,

            # 终点局部 x 方向前进距离的归一化奖励。
            "progress": progress,

            # 是否始终保持在 collision_distance 之外。
            "no_collision": no_collision,

            # 动态/静态障碍物最小距离对应的连续安全惩罚。
            "collision_cost": collision_cost,

            # 与导航路线的平均归一化距离。
            "route_cost": route_cost,

            # 加速度和 jerk 超限产生的舒适性惩罚。
            "comfort_cost": comfort_cost,

            # 局部 x 坐标发生反向变化产生的惩罚。
            "backward_cost": backward_cost,

            # 相对于专家未来位置的平均欧氏距离。
            "imitation_cost": imitation_cost,
        }

        # 返回：
        #
        # rewards：
        # [B,G]
        #
        # details：
        # 每项同样通常为 [B,G]，
        # 便于 rollout_epoch 统计各个奖励分量。
        return rewards, details

    def _progress_guard_reward(
        self,
        trajectories: torch.Tensor,
        ego_future: Optional[torch.Tensor],
        route_lanes: torch.Tensor,
        candidate_progress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回相对专家进度的有界保护奖励，形状为 [B,G]、范围为 [0,1]。

        移动场景只要求候选达到专家记录进度，超过专家不会继续增益；停车场景
        则奖励接近专家的低进度。这样可以抑制 stop hacking，同时避免鼓励超速。
        """
        if candidate_progress is None:
            candidate_progress, _ = self._route_motion_metrics(
                trajectories, route_lanes
            )
        if ego_future is None:
            return torch.ones_like(candidate_progress)

        expert_progress, _ = self._route_motion_metrics(
            ego_future[:, None], route_lanes
        )
        tolerance = max(float(self.config.progress_guard_stop_tolerance), 1e-6)
        moving = expert_progress > tolerance
        moving_reward = (candidate_progress / expert_progress.clamp_min(tolerance)).clamp(
            0, 1
        )
        stopped_reward = (
            1 - (candidate_progress - expert_progress).abs() / tolerance
        ).clamp(0, 1)
        return torch.where(moving, moving_reward, stopped_reward)

    @torch.no_grad()
    def __call__(
        self,
        trajectories: torch.Tensor,
        neighbors_future: torch.Tensor,
        neighbor_mask: torch.Tensor,
        route_lanes: torch.Tensor,
        static_objects: Optional[torch.Tensor] = None,
        ego_future: Optional[torch.Tensor] = None,
        neighbor_agents_past: Optional[torch.Tensor] = None,
        ego_current_state: Optional[torch.Tensor] = None,
        lanes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算论文 D.5 的 multi-reward，并保留旧指标用于退化诊断。"""
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, G, T, 4]")
        if trajectories.shape[-1] != 4:
            raise ValueError("trajectory state must be [x, y, cos(yaw), sin(yaw)]")

        risk = self._risk_reward(
            trajectories,
            neighbors_future,
            neighbor_mask,
            static_objects,
            neighbor_agents_past,
            ego_current_state,
        )
        follow_reward, leader_fraction = self._following_reward(
            trajectories,
            neighbors_future,
            neighbor_mask,
            neighbor_agents_past,
            ego_current_state,
        )
        lane_source = route_lanes if lanes is None else lanes
        lane_reward, lane_mask = self._lane_reward(
            trajectories, lane_source, ego_future
        )

        # 旧指标主要用于 reward hacking/运动退化诊断；progress/direction guard
        # 只有在显式设置非零权重时才作为 NuPlan 小数据适配加入论文三项奖励。
        progress, backward_cost = self._route_motion_metrics(
            trajectories, route_lanes
        )
        (
            direction_cost,
            motion_alignment,
            heading_alignment,
            reverse_fraction,
            min_progress_in_1s,
            direction_compliance_score_approx,
        ) = self._direction_metrics(trajectories, route_lanes)
        progress_guard_reward = self._progress_guard_reward(
            trajectories,
            ego_future,
            route_lanes,
            candidate_progress=progress,
        )

        cfg = self.config
        rewards = (
            cfg.risk_weight * risk["risk_reward"]
            + cfg.follow_weight * follow_reward
            + cfg.lane_weight * lane_reward
            + cfg.progress_guard_weight * progress_guard_reward
            - cfg.direction_guard_weight * direction_cost
        )

        route_cost = self._route_cost(trajectories, route_lanes)
        comfort_cost = self._comfort_cost(trajectories, ego_current_state)
        collision_cost, no_collision = self._collision_cost(
            trajectories,
            neighbors_future,
            neighbor_mask,
            static_objects,
            neighbor_agents_past,
        )

        details = {
            "reward": rewards,
            **risk,
            "follow_reward": follow_reward,
            "leader_fraction": leader_fraction,
            "lane_reward": lane_reward,
            "lane_reward_mask": lane_mask,
            "progress_guard_reward": progress_guard_reward,
            "progress": progress,
            "direction_cost": direction_cost,
            "motion_alignment": motion_alignment,
            "heading_alignment": heading_alignment,
            "reverse_fraction": reverse_fraction,
            "min_progress_in_1s": min_progress_in_1s,
            "direction_compliance_score_approx": (
                direction_compliance_score_approx
            ),
            "no_collision": no_collision,
            "collision_cost": collision_cost,
            "route_cost": route_cost,
            "comfort_cost": comfort_cost,
            "backward_cost": backward_cost,
        }
        return rewards, details
