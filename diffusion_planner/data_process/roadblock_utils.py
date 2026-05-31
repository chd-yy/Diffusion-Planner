# 导入 numpy，用于数组计算、角度归一化、距离计算、矩阵运算等
import numpy as np

# 导入双端队列 deque，用于实现广度优先搜索 BFS 的队列
from collections import deque

# 导入类型注解工具
# Dict：字典类型
# Optional：可选类型，可以是某种类型，也可以是 None
# Tuple：元组类型
# Union：联合类型，可以是多种类型之一
# List：列表类型
from typing import Dict, Optional, Tuple, Union, List

# nuPlan 中的自车状态类，包含自车的位置、姿态、速度等信息
from nuplan.common.actor_state.ego_state import EgoState

# nuPlan 中的二维位姿表示类，通常包含 x、y、heading
from nuplan.common.actor_state.state_representation import StateSE2

# nuPlan 抽象地图接口，用于查询 roadblock、lane、map object 等地图元素
from nuplan.common.maps.abstract_map import AbstractMap

# roadblock 图中的边对象类型
# 在 nuPlan 中，roadblock 和 roadblock connector 都可以看成路线图中的节点/边式结构
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject

# 地图语义层枚举，例如 ROADBLOCK、ROADBLOCK_CONNECTOR 等
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# 基于 STRTree 的占用地图构造工具
# 这里主要用于判断 roadblock connector 的多边形是否发生相交，从而检测路线环路
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory

# 再次导入 AbstractMap
# 这里是重复导入，虽然没有必要，但保持源代码不变
from nuplan.common.maps.abstract_map import AbstractMap

# 再次导入 RoadBlockGraphEdgeMapObject
# 这里也是重复导入，保持源代码不变
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject


# 将角度归一化到 [-pi, pi) 范围内
# 这样可以避免角度差出现 2pi 周期导致的错误
# 例如 179° 和 -179° 实际只差 2°，直接相减却会得到 358°
def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# 定义一个用于在 roadblock graph 上执行广度优先搜索的类
# 它可以从某个起始 roadblock 出发，向前或向后搜索目标 roadblock
class BreadthFirstSearchRoadBlock:
    """
    A class that performs iterative breadth first search. The class operates on the roadblock graph.
    """

    # 初始化 BFS 搜索器
    # start_roadblock_id：起始 roadblock 的 id
    # map_api：nuPlan 地图接口，用于根据 id 查询 roadblock 对象
    # forward_search：是否沿行驶方向向前搜索；如果为 False，则沿 incoming_edges 反向搜索
    def __init__(
        self, start_roadblock_id: int, map_api: Optional[AbstractMap], forward_search: str = True
    ):
        """
        Constructor of BreadthFirstSearchRoadBlock class
        :param start_roadblock_id: roadblock id where graph starts
        :param map_api: map class in nuPlan
        :param forward_search: whether to search in driving direction, defaults to True
        """

        # 保存地图 API，用于后续根据 roadblock id 获取地图对象
        self._map_api: Optional[AbstractMap] = map_api

        # 初始化 BFS 队列
        # 队列中先放入起始 roadblock，再放入 None 作为当前搜索深度结束的标记
        # None 的作用是区分 BFS 的层级，也就是记录搜索深度
        self._queue = deque([self.id_to_roadblock(start_roadblock_id), None])

        # 用于记录每个 roadblock 的父节点
        # key 通常是 roadblock_id + depth
        # value 是它的上一个 roadblock
        # 后续可以通过 _parent 从终点反向构造路径
        self._parent: Dict[str, Optional[RoadBlockGraphEdgeMapObject]] = dict()

        # 保存搜索方向
        # True 表示沿 outgoing_edges 正向搜索
        # False 表示沿 incoming_edges 反向搜索
        self._forward_search = forward_search

        #  lazy loaded
        # 目标 roadblock id 列表
        # 在调用 search 时才会赋值，因此这里先设为 None
        self._target_roadblock_ids: List[str] = None

    # 执行 BFS 搜索，寻找从起点到目标 roadblock 的路径
    # target_roadblock_id 可以是单个字符串，也可以是字符串列表
    # max_depth 表示最大搜索深度，避免在地图图结构中无限搜索
    def search(
        self, target_roadblock_id: Union[str, List[str]], max_depth: int
    ) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
        """
        Apply BFS to find route to target roadblock.
        :param target_roadblock_id: id of target roadblock
        :param max_depth: maximum search depth
        :return: tuple of route and whether a path was found
        """

        # 如果目标 roadblock id 是单个字符串，则统一包装成列表
        # 这样后续可以统一用 in 判断当前 roadblock 是否属于目标集合
        if isinstance(target_roadblock_id, str):
            target_roadblock_id = [target_roadblock_id]

        # 保存目标 roadblock id 列表
        self._target_roadblock_ids = target_roadblock_id

        # 取出 BFS 起始 roadblock
        start_edge = self._queue[0]

        # Initial search states
        # 是否找到路径的标志
        path_found: bool = False

        # 当前记录的终点 roadblock
        # 初始设为起点
        end_edge: RoadBlockGraphEdgeMapObject = start_edge

        # 当前记录的终点深度
        end_depth: int = 1

        # 当前 BFS 搜索深度
        depth: int = 1

        # 起点没有父节点
        # key 中加入 depth，是为了区分不同深度下可能出现的同一个 roadblock id
        self._parent[start_edge.id + f"_{depth}"] = None

        # BFS 主循环，只要队列不为空就继续搜索
        while self._queue:

            # 从队列左侧弹出当前 roadblock
            current_edge = self._queue.popleft()

            # Early exit condition
            # 如果当前深度超过最大搜索深度，则提前停止搜索
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            # current_edge 为 None 说明当前深度这一层已经遍历完
            # 接下来进入下一层搜索
            if current_edge is None:

                # 深度加 1
                depth += 1

                # 在队列末尾继续插入 None，用于标记下一层的结束
                self._queue.append(None)

                # 如果队列开头也是 None，说明没有新的 roadblock 入队
                # 搜索已经结束
                if self._queue[0] is None:
                    break

                # 进入下一轮循环
                continue

            # Goal condition
            # 如果当前 roadblock 满足目标条件，则记录终点并停止搜索
            if self._check_goal_condition(current_edge, depth, max_depth):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            # 根据搜索方向选择邻居节点
            # forward_search=True：沿车辆行驶方向，访问 outgoing_edges
            # forward_search=False：反向搜索，访问 incoming_edges
            neighbors = (
                current_edge.outgoing_edges if self._forward_search else current_edge.incoming_edges
            )

            # Populate queue
            # 遍历当前 roadblock 的所有邻居 roadblock
            for next_edge in neighbors:

                # if next_edge.id in self._candidate_lane_edge_ids_old:
                # 将邻居 roadblock 加入 BFS 队列
                self._queue.append(next_edge)

                # 记录该邻居 roadblock 的父节点为 current_edge
                # 深度为 depth+1，因为 next_edge 是下一层节点
                self._parent[next_edge.id + f"_{depth + 1}"] = current_edge

                # 即使没有找到目标，也持续更新 end_edge 和 end_depth
                # 这样如果搜索失败，仍然可以返回一条已经搜索到的路径
                end_edge = next_edge
                end_depth = depth + 1

        # 根据记录的 end_edge 和 end_depth 反向构造路径
        # 同时返回 path_found 表示是否真正找到了目标 roadblock
        return self._construct_path(end_edge, end_depth), path_found

    # 根据 roadblock id 从地图 API 中获取对应 roadblock 对象
    # 先尝试普通 ROADBLOCK，再尝试 ROADBLOCK_CONNECTOR
    def id_to_roadblock(self, id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Retrieves roadblock from map-api based on id
        :param id: id of roadblock
        :return: roadblock class
        """

        # 尝试从地图中获取普通 roadblock
        block = self._map_api._get_roadblock(id)

        # 如果普通 roadblock 没找到，则尝试获取 roadblock connector
        block = block or self._map_api._get_roadblock_connector(id)

        # 返回找到的地图对象
        return block

    # 检查是否应该因为超过最大深度而终止搜索
    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: whether depth exceeds the target depth.
        """

        # 如果当前深度超过最大允许深度，则结束搜索
        return depth > max_depth

    # 检查当前 roadblock 是否为目标 roadblock
    def _check_goal_condition(
        self,
        current_edge: RoadBlockGraphEdgeMapObject,
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: edge to check.
        :param depth: current depth to check.
        :param max_depth: maximum depth the edge should be at.
        :return: True if the lane edge is contain the in the target roadblock. False, otherwise.
        """

        # 当前 roadblock id 在目标 id 列表中，并且当前深度没有超过最大深度，则认为找到目标
        return current_edge.id in self._target_roadblock_ids and depth <= max_depth

    # 从搜索终点出发，根据 _parent 字典反向恢复路径
    def _construct_path(
        self, end_edge: RoadBlockGraphEdgeMapObject, depth: int
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Constructs a path when goal was found.
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of RoadBlockGraphEdgeMapObject
        """

        # 路径初始化为终点 roadblock
        path = [end_edge]

        # 同时记录路径中的 roadblock id
        path_id = [end_edge.id]

        # 只要当前节点存在父节点，就持续向前回溯
        while self._parent[end_edge.id + f"_{depth}"] is not None:

            # 将父节点加入路径
            path.append(self._parent[end_edge.id + f"_{depth}"])

            # 记录父节点 id
            path_id.append(path[-1].id)

            # 当前节点更新为父节点
            end_edge = self._parent[end_edge.id + f"_{depth}"]

            # 深度向前回退一层
            depth -= 1

        # 如果是正向搜索，回溯得到的路径顺序是终点到起点
        # 因此需要 reverse 成起点到终点
        if self._forward_search:
            path.reverse()
            path_id.reverse()

        # 返回 roadblock 对象路径和 id 路径
        return (path, path_id)


# 根据自车当前位置，从地图中确定自车当前所在或最可能所在的 roadblock
# route_roadblocks_dict 是导航路线中的 roadblock 字典，用于优先选择位于 route 上的候选 roadblock
def get_current_roadblock_candidates(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    Determines a set of roadblock candidate where ego is located
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param heading_error_thresh: maximum heading error, defaults to np.pi/4
    :param displacement_error_thresh: maximum displacement, defaults to 3
    :return: tuple of most promising roadblock and other candidates
    """

    # 获取自车后轴中心位姿
    # nuPlan 中通常使用 rear_axle 表示自车参考点
    ego_pose: StateSE2 = ego_state.rear_axle

    # 初始化候选 roadblock 列表
    roadblock_candidates = []

    # 定义需要查询的地图语义层
    # ROADBLOCK：普通道路块
    # ROADBLOCK_CONNECTOR：连接不同 roadblock 的道路连接块，常见于路口区域
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]

    # 查询自车当前位置附近半径 1.0 米内的 roadblock 和 roadblock connector
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=1.0, layers=layers
    )

    # 将普通 roadblock 和 roadblock connector 合并为候选集合
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    # 如果半径 1m 范围内没有查询到候选 roadblock
    # 则退而求其次，寻找最近的 roadblock 或 roadblock connector
    if not roadblock_candidates:

        # 分别在 ROADBLOCK 和 ROADBLOCK_CONNECTOR 图层中查询最近地图对象
        for layer in layers:

            # 获取距离自车最近的地图对象 id 以及距离
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )

            # 根据 id 和 layer 获取对应地图对象
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            # 如果找到，则加入候选列表
            if roadblock:
                roadblock_candidates.append(roadblock)

    # on_route_candidates：位于导航路线上的候选 roadblock
    # on_route_candidate_displacement_errors：对应的横向/空间距离误差
    on_route_candidates, on_route_candidate_displacement_errors = [], []

    # candidates：不在导航路线上的候选 roadblock
    # candidate_displacement_errors：对应的距离误差
    candidates, candidate_displacement_errors = [], []

    # 每个候选 roadblock 的最小距离误差
    # 用于最后没有符合阈值候选时的兜底选择
    roadblock_displacement_errors = []

    # 每个候选 roadblock 的最小航向误差
    roadblock_heading_errors = []

    # 遍历附近所有候选 roadblock
    for idx, roadblock in enumerate(roadblock_candidates):

        # 初始化当前 roadblock 内部所有 lane 的最小距离误差和航向误差
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        # 遍历当前 roadblock 内部的所有 lane
        for lane in roadblock.interior_edges:

            # 获取 lane 中心线的离散路径点
            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path

            # 将离散路径点中的二维坐标提取为 numpy 数组
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )

            # 计算自车当前位置到该 lane 每个离散点的欧氏距离
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(axis=-1) ** 0.5

            # 找到距离自车最近的 lane 离散点索引
            argmin = np.argmin(lane_state_distances)

            # 计算最近 lane 点的航向与自车航向之间的误差
            # normalize_angle 用于避免角度跨越 pi/-pi 时出现异常大误差
            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading - ego_pose.heading)
            )

            # 最近 lane 点与自车之间的距离误差
            displacement_error = lane_state_distances[argmin]

            # 更新当前 roadblock 内部的最小距离误差
            # 同时保存对应的航向误差
            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            # 如果航向误差和距离误差都满足阈值要求
            # 则认为当前 roadblock 可以作为自车所在 roadblock 的候选
            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):

                # 如果该 roadblock 在导航路线中，则优先加入 on_route_candidates
                if roadblock.id in route_roadblocks_dict.keys():
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)

                # 如果不在导航路线中，则加入普通候选 candidates
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        # 保存当前 roadblock 的最小距离误差
        roadblock_displacement_errors.append(lane_displacement_error)

        # 保存当前 roadblock 的最小航向误差
        roadblock_heading_errors.append(lane_heading_error)

    # 如果存在位于导航路线上的候选 roadblock，则优先返回其中距离误差最小的一个
    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )

    # 如果不存在 on-route 候选，但存在满足误差阈值的普通候选，则返回普通候选中距离最近的一个
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close roadblock
    # 如果没有任何满足航向和距离阈值的候选，则兜底返回距离误差最小的 roadblock
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )


# 修正 route_roadblock_ids
# 主要解决三个问题：
# 1. 自车当前所在 roadblock 不在原始 route 中；
# 2. route 中相邻 roadblock 不连通，需要搜索中间缺失 roadblock；
# 3. route 中存在环路，需要裁剪
def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """

    # 将 route_roadblock_ids 转换成 id -> roadblock 对象的字典
    route_roadblock_dict = {}

    # 遍历原始路线中的每个 roadblock id
    for id_ in route_roadblock_ids:

        # 优先尝试从 ROADBLOCK 图层获取地图对象
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)

        # 如果不是普通 roadblock，则尝试从 ROADBLOCK_CONNECTOR 图层获取
        block = block or map_api.get_map_object(
            id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )

        # 保存到路线字典中
        route_roadblock_dict[id_] = block

    # 根据自车当前位置，确定当前最可能所在的 roadblock
    # 同时返回所有候选 starting_block_candidates
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict
    )

    # 提取当前候选 roadblock 的 id 列表
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    # 将路线字典中的 roadblock 对象转换为列表
    route_roadblocks = list(route_roadblock_dict.values())

    # 将路线字典中的 id 转换为列表
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    # 修正 1：如果自车当前所在 roadblock 不在原始 route 中
    if starting_block.id not in route_roadblock_ids:

        # Backward search if current roadblock not in route
        # 情况 1：从原始 route 的第一个 roadblock 出发，反向搜索当前自车所在 roadblock
        # 如果能搜到，说明原始 route 缺少了前面一段 roadblock，需要补到 route 开头
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )

        # 执行反向 BFS 搜索，目标是当前自车可能所在的 starting_block_ids
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        # 如果反向搜索找到了路径
        if path_found:

            # 将搜索得到的路径补到 route 开头
            # path[:-1] 去掉最后一个，因为最后一个通常已经是原始 route 的起点，避免重复
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        # 如果反向搜索没有找到
        else:

            # Forward search to any route roadblock
            # 情况 2：从当前自车所在 roadblock 出发，向前搜索原始 route 前几个 roadblock
            # 如果能搜到，说明应该从当前 roadblock 接入后续 route
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )

            # 搜索目标为原始 route 的前三个 roadblock
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            # 如果正向搜索找到了连接到原始 route 的路径
            if path_found:

                # 找到 path 最后一个 roadblock 在原始 route 中的位置
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )

                # 删除原始 route 中已经被 path 覆盖之前的部分
                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

                # 将从当前自车 roadblock 搜出来的路径插入到 route 开头
                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    # 修正 2：检查 route 中相邻 roadblock 是否真正连通
    # 如果不连通，则通过 BFS 搜索中间缺失的 roadblock
    roadblocks_to_append = {}

    # 遍历 route 中每一对相邻 roadblock
    for i in range(len(route_roadblocks) - 1):

        # 获取后一个 roadblock 的所有 incoming_edges 的 id
        # 如果前一个 roadblock 真的是后一个 roadblock 的前驱，那么它的 id 应该出现在 incoming_edges 中
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]

        # 判断当前 roadblock 是否是下一个 roadblock 的前驱
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        # 如果已经连通，则不需要修正
        if is_incoming:
            continue

        # 如果不连通，则从当前 roadblock 出发向前搜索下一个 roadblock
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )

        # 执行正向 BFS，寻找从 route_roadblock_ids[i] 到 route_roadblock_ids[i+1] 的连接路径
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        # 如果找到了路径，并且路径长度至少为 3
        # 说明中间存在缺失 roadblock
        if path_found and path and len(path) >= 3:

            # 去掉路径首尾
            # 首是当前 roadblock，尾是下一个 roadblock，它们已经在原始 route 中
            # 中间部分才是需要插入的缺失 roadblock
            path, path_id = path[1:-1], path_id[1:-1]

            # 记录第 i 个位置后需要插入的中间 roadblock
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    # 将缺失的中间 roadblock 插入原始 route
    offset = 1

    # 遍历所有需要插入的位置和路径
    for i, (path, path_id) in roadblocks_to_append.items():

        # 在第 i 个 roadblock 后插入缺失 roadblock
        # offset 用于修正前面已经插入元素导致的索引偏移
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id

        # 更新偏移量
        offset += len(path)

    # Fix 3: cut route-loops
    # 修正 3：移除 route 中可能存在的环路
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    # 返回修正后的 route roadblock id 列表
    return route_roadblock_ids


# 移除路线中的环路
# 主要用于处理 roadblock connector 在交叉口区域出现重复或相交，导致 route 形成 loop 的情况
def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    # 用于保存已经遍历过的 roadblock connector 多边形占用区域
    roadblock_occupancy_map = None

    # 记录检测到 loop 的位置索引
    loop_idx = None

    # 遍历 route 中的所有 roadblock
    for idx, roadblock in enumerate(route_roadblocks):

        # loops only occur at intersection, thus searching for roadblock-connectors.
        # 只对 roadblock connector 做环路检测
        # 因为路线环路通常发生在路口连接区域
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":

            # 如果占用地图还没有创建
            if not roadblock_occupancy_map:

                # 用当前 connector 的 polygon 创建 STRTree 占用地图
                # STRTree 可以高效查询几何对象之间是否相交
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id]
                )

                # 第一个 connector 只用于初始化，不做相交检测
                continue

            # 构建 STRTree 和 id 索引
            strtree, index_by_id = roadblock_occupancy_map._build_strtree()

            # 查询当前 roadblock polygon 与已有 polygon 可能相交的对象索引
            indices = strtree.query(roadblock.polygon)

            # 如果存在潜在相交对象
            if len(indices) > 0:

                # 遍历所有可能相交的几何对象
                for geom in strtree.geometries.take(indices):

                    # 计算当前 polygon 与已有 polygon 的交集面积
                    area = geom.intersection(roadblock.polygon).area

                    # 如果交集面积大于 1，认为发生明显重叠，路线可能形成环路
                    if area > 1:

                        # 记录当前 loop 开始位置
                        loop_idx = idx
                        break

                # 如果已经找到 loop，则退出外层循环
                if loop_idx:
                    break

            # 如果没有形成 loop，则把当前 connector 插入占用地图
            # 后续 connector 会与它进行相交检测
            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    # 如果检测到 loop
    if loop_idx:

        # 裁剪掉 loop_idx 及其之后的路线部分
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    # 返回去除环路后的 route_roadblocks 和 route_roadblock_ids
    return route_roadblocks, route_roadblock_ids