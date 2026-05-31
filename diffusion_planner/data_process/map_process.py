"""
Module: Map Data Preprocessing Functions
Description: This module contains functions for Map related data processing.

Categories:
    1. Get lanes, speed limit, traffic light and lane's roadblock ids
    2. Get maps array for model input
"""

# 导入类型注解
# List：列表类型
# Dict：字典类型
# Tuple：元组类型
# Set：集合类型
from typing import List, Dict, Tuple, Set

# 导入 numpy，用于数组构造、距离计算、排序、插值结果组织等数值计算
import numpy as np

# 从 shapely 中导入 LineString
# 主要用于将一条离散 polyline 转成连续线段对象，然后在该线段上按距离插值采样
from shapely import LineString

# nuPlan 中的二维点类型，通常表示地图点坐标，例如车道中心线点、边界点等
from nuplan.common.actor_state.state_representation import Point2D

# nuPlan 的抽象地图接口，用于查询附近车道、车道连接器、地图多边形等对象
from nuplan.common.maps.abstract_map import AbstractMap

# 用于计算地图对象与某个点之间的距离
# 后面会用它按照距离自车远近对地图对象排序
from nuplan.common.maps.nuplan_map.utils import get_distance_between_map_object_and_point

# TrafficLightStatusData：交通灯状态数据类型
# SemanticMapLayer：地图语义层枚举，例如 LANE、LANE_CONNECTOR 等
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer

# 从 nuPlan 的 vector_builder_utils 中导入矢量地图相关工具和数据结构
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
MapObjectPolylines, 
VectorFeatureLayer, 
LaneSegmentLaneIDs, 
VectorFeatureLayerMapping, 
LaneSegmentTrafficLightData,
get_traffic_light_encoding,
get_map_object_polygons
)

# 导入坐标转换函数
# 用于将全局坐标系下的地图点转换到以当前自车为中心的局部坐标系下
from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame


# =====================
# 1. Get lanes, speed limit, traffic light and lane's roadblock ids
# =====================
# 第 1 部分：从地图 API 中提取自车附近的车道中心线、左右边界、限速、交通灯状态和 roadblock id


# 提取自车附近一定半径内的 lane 和 lane connector 的 polyline 信息
# 包括中心线、左边界、右边界、lane id、限速信息、roadblock id 等
def _get_lane_polylines(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
        lane_speed_limit: lane's speed limit.
        lane_has_speed_limit: whether lane has speed limit.
        lane_roadblock_ids: lane's roadblock ids.
    """

    # 保存每条 lane 或 lane connector 的中心线 polyline
    # 每个元素是一条 polyline，每条 polyline 内部包含数量不固定的 Point2D 点
    lanes_mid: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]

    # 保存每条 lane 或 lane connector 的左边界 polyline
    lanes_left: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]

    # 保存每条 lane 或 lane connector 的右边界 polyline
    lanes_right: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]

    # 保存每条 lane 或 lane connector 的 id
    lane_ids: List[str] = []  # shape: [num_lanes]

    # 保存每条 lane 或 lane connector 的限速值，单位通常是 m/s
    lane_speed_limit = []

    # 保存每条 lane 或 lane connector 是否存在有效限速
    lane_has_speed_limit = []

    # 保存每条 lane 或 lane connector 所属的 roadblock id
    # 后续会用它判断该 lane 是否在导航路线 route 上
    lane_roadblock_ids = []

    # 需要查询的地图语义层
    # LANE：普通车道
    # LANE_CONNECTOR：连接车道，通常出现在路口、转弯连接区域
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]

    # 从地图中查询以 point 为中心、radius 为半径范围内的 lane 和 lane connector
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    # 用于合并不同语义层中的地图对象
    map_objects = []

    # 将 LANE 和 LANE_CONNECTOR 两类对象合并到 map_objects 中
    for layer_name in layer_names:
        map_objects += layers[layer_name]

    # sort by distance to query point
    # 按照地图对象到查询点 point 的距离从近到远排序
    # 这样后续截断 max_elements 时，会优先保留离自车更近的地图元素
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    # 遍历排序后的所有 lane / lane connector 地图对象
    for map_obj in map_objects:

        # center lane
        # 提取当前地图对象的中心线 baseline path
        # map_obj.baseline_path.discrete_path 是离散路径点序列
        # 每个 node 有 x、y 坐标，这里重新包装成 Point2D
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]

        # 将该 lane 的中心线加入 lanes_mid
        lanes_mid.append(baseline_path_polyline)

        # boundaries
        # 提取当前 lane 的左边界离散点
        lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])

        # 提取当前 lane 的右边界离散点
        lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])

        # lane ids
        # 保存当前 lane 或 lane connector 的 id
        lane_ids.append(map_obj.id)

        # speed limit
        # 如果当前地图对象没有限速信息，则将限速设为 0.0，并标记 has_speed_limit 为 False
        if map_obj.speed_limit_mps is None:
            lane_speed_limit.append(0.0)
            lane_has_speed_limit.append(False)

        # 如果存在限速信息，则保存真实限速值，并标记 has_speed_limit 为 True
        else:
            lane_speed_limit.append(map_obj.speed_limit_mps)
            lane_has_speed_limit.append(True)

        
        # 保存当前 lane 所属的 roadblock id
        # 后续可通过 roadblock id 判断该 lane 是否属于导航路线
        lane_roadblock_ids.append(map_obj.get_roadblock_id())

    # 将提取结果包装成 nuPlan 使用的 MapObjectPolylines / LaneSegmentLaneIDs 等结构后返回
    return (
        MapObjectPolylines(lanes_mid),
        MapObjectPolylines(lanes_left),
        MapObjectPolylines(lanes_right),
        LaneSegmentLaneIDs(lane_ids),
        lane_speed_limit,
        lane_has_speed_limit,
        lane_roadblock_ids
    )


# 提取自车附近的矢量地图信息
# 根据 map_features 决定要提取哪些地图特征，例如 LANE、LEFT_BOUNDARY、RIGHT_BOUNDARY 等
def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        speed_limit: Lane's speed limit
        lane_route: route lane
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """

    # coords 用于保存不同地图特征对应的 polyline 数据
    # 例如 coords["LANE"] 保存 lane centerline，coords["LEFT_BOUNDARY"] 保存左边界
    coords: Dict[str, MapObjectPolylines] = {}

    # traffic_light_data 用于保存 lane 对应的交通灯 one-hot 编码
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}

    # speed_limit 用于保存 lane 是否有限速以及限速值
    speed_limit = {}

    # feature_layers 保存字符串形式 map_features 转换后的枚举类型
    feature_layers: List[VectorFeatureLayer] = []

    # 遍历用户希望提取的地图特征名称
    for feature_name in map_features:

        # 尝试将字符串 feature_name 转换为 VectorFeatureLayer 枚举
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])

        # 如果 feature_name 不属于合法的 VectorFeatureLayer，则抛出错误
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    # 如果需要提取 LANE，则调用 _get_lane_polylines 提取车道中心线、边界、id、限速和 roadblock 信息
    if VectorFeatureLayer.LANE in feature_layers:

        # lanes_mid：车道中心线
        # lanes_left：车道左边界
        # lanes_right：车道右边界
        # lane_ids：车道 id
        # lane_speed_limit：车道限速
        # lane_has_speed_limit：车道是否有限速
        # lane_route：车道所属 roadblock id
        lanes_mid, lanes_left, lanes_right, lane_ids, lane_speed_limit, lane_has_speed_limit, lane_route = _get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        # 保存 lane 中心线 polyline
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # 将是否有限速转换为 bool numpy 数组
        speed_limit['lane_has_speed_limit'] = np.array(lane_has_speed_limit, dtype=np.bool_)

        # 将限速值转换为 float32 numpy 数组
        speed_limit['lane_speed_limit'] = np.array(lane_speed_limit, dtype=np.float32)
        

        # lane traffic light data
        # 根据 lane_ids 和当前时刻 traffic_light_status_data 生成每条 lane 的交通灯编码
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        # 如果需要左边界，则保存左边界 polyline
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)

        # 如果需要右边界，则保存右边界 polyline
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)


    # extract generic map objects
    # 提取其他 polygon 类型地图对象
    # 例如某些地图层可能不是 lane polyline，而是 polygon 表示的区域
    for feature_layer in feature_layers:

        # 判断当前 feature_layer 是否属于可用的 polygon 地图层
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():

            # 根据语义地图层提取当前自车附近的 polygon 对象
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )

            # 保存 polygon 类型地图特征
            coords[feature_layer.name] = polygons

    # 返回：
    # coords：地图几何信息
    # traffic_light_data：交通灯信息
    # speed_limit：限速信息
    # lane_route：每条 lane 对应的 roadblock id
    return coords, traffic_light_data, speed_limit, lane_route


# =====================
# 2. Get maps array for model input
# =====================
# 第 2 部分：将前面提取到的原始地图 polyline 数据转换成模型可直接输入的固定尺寸数组

# 对一条 polyline 重新采样，使其点数固定为 num_point
# 输入 line 通常是一条车道中心线或边界线的点序列
def _interpolate_points(line, num_point):

    # 将输入的离散点序列转换成 shapely 的 LineString 对象
    # LineString 可以看作一条连续折线，支持按照距离插值
    line = LineString(line)

    # 在 [0, line.length] 范围内均匀取 num_point 个距离值
    # 对每个距离 d 调用 line.interpolate(d)，得到折线上的插值点
    # 最后将所有插值点坐标拼接成一个 numpy 数组
    new_line = np.concatenate([line.interpolate(d).coords._coords for d in np.linspace(0, line.length, num_point)])

    # 返回固定点数的 polyline，形状通常为 [num_point, 2]
    return new_line


# 将 lane 相关数据转换成固定尺寸数组
# 主要完成：
# 1. 按距离自车远近选取最多 max_elements 条 lane；
# 2. 每条 lane 插值成 max_points 个点；
# 3. 同步处理中心线、左右边界、交通灯、限速和 route 信息；
# 4. 不足 max_elements 的部分用 0 padding
def _convert_lane_to_fixed_size(ego_pose, feature_coords, speed_limit, lane_route, left_boundary, right_boundary, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim):

    # 如果存在交通灯数据，则要求每条 lane 都应该有对应的交通灯编码
    # 如果数量不一致，说明数据对齐有问题，直接报错
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")
    
    # 取出每条 lane 是否存在限速的信息
    lane_has_speed_limit = speed_limit['lane_has_speed_limit']

    # 取出每条 lane 的限速值
    lane_speed_limit = speed_limit['lane_speed_limit']

    # trim or zero-pad elements to maintain fixed size
    # 初始化 lane 中心线数组
    # 形状为 [max_elements, max_points, 2]
    # max_elements 表示最多保留多少条 lane
    # max_points 表示每条 lane 固定采样多少个点
    coords_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)

    # 初始化左边界数组
    left_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)

    # 初始化右边界数组
    right_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)

    # 初始化是否有限速数组，形状为 [max_elements, 1]
    lane_has_speed_limit_array = np.zeros((max_elements, 1), dtype=np.bool_)

    # 初始化限速值数组，形状为 [max_elements, 1]
    lane_speed_limit_array = np.zeros((max_elements, 1), dtype=np.float32)

    # 保存被选中 lane 对应的 roadblock id
    lane_routes = []
        
    # 初始化有效性 mask
    # True 表示该位置是真实 lane，False 表示该位置是 padding
    avails_array = np.zeros((max_elements, max_points), dtype=np.bool_)

    # 初始化交通灯数组
    # 如果 feature_tl_data 不为 None，则形状为 [max_elements, max_points, traffic_light_encoding_dim]
    # 如果没有交通灯数据，则 tl_data_array 为 None
    tl_data_array = (
        np.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=np.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    # 计算每条 lane 到自车的最近距离，并存入 mapping
    # 后面根据距离排序，优先保留离自车最近的 lane
    mapping = {}

    # 遍历所有 lane 的中心线坐标
    for i, e in enumerate(feature_coords):

        # e 的形状通常为 [num_points, 2]
        # ego_pose[None, :2] 取自车位置 [x, y] 并增加一个维度，便于广播计算
        # np.linalg.norm(..., axis=-1).min() 表示该 lane 上所有点到自车的最小距离
        dist = np.linalg.norm(e - ego_pose[None, :2], axis=-1).min()

        # 保存 lane 索引到距离的映射
        mapping[i] = dist

    # 按照距离从小到大排序
    mapping = sorted(mapping.items(), key=lambda item: item[1])

    # 截取距离最近的 max_elements 条 lane
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    # 遍历筛选出的 lane，并写入固定尺寸数组
    for idx, element_idx in enumerate(sorted_elements):

        # element_idx 是一个二元组，通常为 (原始 lane 索引, 距离)
        # element_idx[0] 表示原始 lane 索引
        element_coords = feature_coords[element_idx[0]]

        # 取出该 lane 对应的左边界
        left_coords = left_boundary[element_idx[0]]

        # 取出该 lane 对应的右边界
        right_coords = right_boundary[element_idx[0]]

        # interpolate to maintain fixed size if the number of points is not enough
        # 将中心线插值成固定 max_points 个点
        element_coords = _interpolate_points(element_coords, max_points)

        # 将左边界插值成固定 max_points 个点
        left_coords = _interpolate_points(left_coords, max_points)

        # 将右边界插值成固定 max_points 个点
        right_coords = _interpolate_points(right_coords, max_points)

        # 将处理后的中心线写入固定尺寸数组
        coords_array[idx] = element_coords

        # 将处理后的左边界写入固定尺寸数组
        left_array[idx] = left_coords

        # 将处理后的右边界写入固定尺寸数组
        right_array[idx] = right_coords

        # 将当前 idx 对应的整条 lane 标记为有效数据
        avails_array[idx] = True  # specify real vs zero-padded data

        # 写入该 lane 是否有限速
        lane_has_speed_limit_array[idx] = lane_has_speed_limit[element_idx[0]]

        # 写入该 lane 的限速值
        lane_speed_limit_array[idx] = lane_speed_limit[element_idx[0]]

        # 保存该 lane 对应的 roadblock id
        lane_routes.append(lane_route[element_idx[0]])

        # 如果存在交通灯数组和交通灯原始数据，则写入该 lane 的交通灯编码
        if tl_data_array is not None and feature_tl_data is not None:
            tl_data_array[idx] = feature_tl_data[element_idx[0]]

    # 返回固定尺寸的 lane 中心线、左右边界、交通灯、有效性 mask、限速和 route 信息
    return coords_array, left_array, right_array, tl_data_array, avails_array, lane_has_speed_limit_array, lane_speed_limit_array, lane_routes


# 根据当前查询半径内实际提取到的 roadblock，裁剪导航 route
# 目的是只保留进入查询范围后连续的一段 route roadblock，保证 route_lanes 特征是连通的
def _prune_route_by_connectivity(route_roadblock_ids: List[str], roadblock_ids: Set[str]) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """

    # 保存裁剪后的 route roadblock id
    pruned_route_roadblock_ids: List[str] = []

    # route_start 表示 route 是否已经进入当前查询半径范围
    # False 时，说明还没有遇到第一个在半径范围内的 route roadblock
    route_start = False  # wait for route to come into query radius before declaring broken connection

    # 按照原始 route 顺序遍历 roadblock id
    for roadblock_id in route_roadblock_ids:

        # 如果当前 route roadblock 在查询半径内提取到的 roadblock 集合中
        if roadblock_id in roadblock_ids:

            # 将其加入裁剪后的 route
            pruned_route_roadblock_ids.append(roadblock_id)

            # 标记 route 已经进入查询范围
            route_start = True

        # 如果 route 已经进入过查询范围，但当前 roadblock 又不在查询范围内
        # 说明当前半径内的连续 route 已经中断，因此停止
        elif route_start:  # connection broken
            break

    # 返回当前查询范围内连续的一段 route roadblock id
    return pruned_route_roadblock_ids


# 对 lane polyline 进行后处理，将中心线、方向向量、左右边界相对向量和交通灯状态拼接成模型输入特征
def _lane_polyline_process(polylines, left_boundary, right_boundary, avails, traffic_light):

    # 每个 lane 点最终特征维度为 12
    # 组成一般为：
    # polyline 坐标 2 维
    # polyline_vector 方向向量 2 维
    # polyline_to_left 到左边界向量 2 维
    # polyline_to_right 到右边界向量 2 维
    # traffic_light 交通灯 one-hot 4 维
    # 总共 2+2+2+2+4=12
    dim = 12

    # 初始化新的 lane 特征数组
    # 形状为 [num_lanes, num_points, 12]
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    # 遍历每一条 lane
    for i in range(polylines.shape[0]):

        # avails[i][0] 为 True 表示第 i 条 lane 是真实数据，不是 padding
        if avails[i][0]: 

            # 取出第 i 条 lane 的中心线坐标，形状为 [num_points, 2]
            polyline = polylines[i]

            # 计算中心线每个点到下一个点的方向向量
            # polyline[1:] 是第 1 到最后一个点
            # polyline[:-1] 是第 0 到倒数第二个点
            polyline_vector = polyline[1:]-polyline[:-1]

            # 在最后补一个 0 向量
            # 因为 N 个点只能计算出 N-1 个相邻点方向向量，需要补齐到 N 个点
            polyline_vector = np.insert(polyline_vector, polyline_vector.shape[0] , 0, axis=0)

            # 判断左边界点序列方向是否与中心线方向一致
            # 如果左边界最后一个点距离中心线起点更近，说明边界方向可能与中心线相反，需要翻转
            if np.linalg.norm(left_boundary[i, -1] - polyline[0]) < np.linalg.norm(left_boundary[i, 0] - polyline[0]):
                left_boundary[i] = np.flip(left_boundary[i], axis=0)

            # 判断右边界点序列方向是否与中心线方向一致
            # 如果右边界最后一个点距离中心线起点更近，说明右边界方向可能与中心线相反，需要翻转
            if np.linalg.norm(right_boundary[i, -1] - polyline[0]) < np.linalg.norm(right_boundary[i, 0] - polyline[0]):
                right_boundary[i] = np.flip(right_boundary[i], axis=0)

            # 计算中心线点指向左边界对应点的向量
            polyline_to_left = left_boundary[i] - polyline

            # 计算中心线点指向右边界对应点的向量
            polyline_to_right = right_boundary[i] - polyline


            # 将中心线坐标、中心线方向向量、左边界相对向量、右边界相对向量和交通灯状态拼接
            # 拼接后每个点形成 12 维特征
            new_polylines[i] = np.concatenate([polyline, polyline_vector, polyline_to_left, polyline_to_right, traffic_light[i]], axis=-1)  

    # 返回 lane 特征数组
    return new_polylines



# 地图数据主处理函数
# 输入原始矢量地图信息，将其转换成 Diffusion Planner 模型需要的固定维度地图输入
def map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, map_features, max_elements, max_points):
    """
    This function process the data from the raw vector set map data.
    :param route_roadblock_ids: route road block ids.
    :param anchor_ego_state: ego current state.
    :param coords: dictionary mapping feature name to polyline vector sets.
    :param traffic_light_data: traffic light status of lanes.
    :param speed_limit: speed limit of lanes.
    :param lane_route: road block ids of lanes.
    :param map_features: Name of map features to extract.
    :param max_elements: clip the number of map elements.
    :param max_points: clip the number of point for each element.
    :return: dict of the map elements.
    """

    # list_array_data 用于将 nuPlan 的 MapObjectPolylines / LaneSegmentTrafficLightData
    # 转换成普通 Python list + numpy array 的形式，便于后续统一处理
    list_array_data = {}

    # 遍历 coords 中每一种地图特征
    # feature_name 例如 LANE、LEFT_BOUNDARY、RIGHT_BOUNDARY
    # feature_coords 是对应特征的 polyline 集合
    for feature_name, feature_coords in coords.items():

        # 保存当前地图特征下的每个地图元素坐标
        list_feature_coords = []

        # Pack coords into array list
        # feature_coords.to_vector() 会把 MapObjectPolylines 转换为可遍历的点序列集合
        for element_coords in feature_coords.to_vector():

            # 将每个地图元素的坐标转换成 np.float64 数组
            list_feature_coords.append(np.array(element_coords, dtype=np.float64))

        # 以 coords.特征名 的形式保存
        list_array_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into array list if it exists
        # 如果当前 feature_name 有对应交通灯数据，则一并转换成 numpy 数组
        if feature_name in traffic_light_data:

            # 保存当前特征下每个元素对应的交通灯编码
            list_feature_tl_data = []

            # 遍历交通灯编码数据
            for element_tl_data in traffic_light_data[feature_name].to_vector():

                # 转换成 np.float64 数组
                list_feature_tl_data.append(np.array(element_tl_data, dtype=np.float64))

            # 以 traffic_light_data.特征名 的形式保存
            list_array_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    """
    
    # array_output 用于保存固定尺寸、转换到自车坐标系后的中间地图特征
    array_output = {}

    # 获取交通灯编码维度，通常为 4
    # 对应 green、yellow、red、unknown 四种状态
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    # 按照指定的 map_features 逐类处理地图特征
    for feature_name in map_features:

        # 如果当前 feature_name 的坐标数据存在，则进行处理
        if f"coords.{feature_name}" in list_array_data:

            # 取出当前特征的坐标列表
            feature_coords = list_array_data[f"coords.{feature_name}"]

            # 如果当前特征有交通灯数据，则取出；否则设为 None
            feature_tl_data = (
                list_array_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_array_data
                else None
            )

            # 如果当前处理的是 LANE，则需要同时处理中心线、左右边界、交通灯、限速和 route 信息
            if feature_name == 'LANE':

                # 将 lane 相关数据转换成固定尺寸数组
                coords, left_coords, right_coords, tl_data, avails, lane_has_speed_limit_array, lane_speed_limit_array, lane_routes = _convert_lane_to_fixed_size(
                        anchor_ego_state,
                        feature_coords,
                        speed_limit,
                        lane_route,
                        list_array_data[f"coords.LEFT_BOUNDARY"],
                        list_array_data[f"coords.RIGHT_BOUNDARY"],
                        feature_tl_data,
                        max_elements[feature_name],
                        max_points[feature_name],
                        traffic_light_encoding_dim
                        if feature_name
                        in [
                            VectorFeatureLayer.LANE.name,
                        ]
                        else None,
                )

                # 将左边界从全局坐标系转换到自车局部坐标系
                left_coords = vector_set_coordinates_to_local_frame(left_coords, avails, anchor_ego_state)

                # 将右边界从全局坐标系转换到自车局部坐标系
                right_coords = vector_set_coordinates_to_local_frame(right_coords, avails, anchor_ego_state)

                # 保存转换后的左边界
                array_output[f"vector_set_map.coords.LEFT_BOUNDARY"] = left_coords

                # 保存转换后的右边界
                array_output[f"vector_set_map.coords.RIGHT_BOUNDARY"] = right_coords

                '''
                Get roadblock polygon
                '''

                # lane_on_route 用于标记每条 lane 是否属于当前导航 route
                lane_on_route = []

                # 从 route_roadblock_ids 中筛选出当前半径内 lane_routes 中存在的 roadblock
                # 即：只保留当前地图提取范围内可见的 route roadblock
                pruned_lane_roadblock_ids = [route for route in route_roadblock_ids if route in lane_routes]

                # 对 route roadblock 进一步按照连通性裁剪，避免 route_lanes 中出现不连续片段
                pruned_route_roadblock_ids = _prune_route_by_connectivity(route_roadblock_ids, pruned_lane_roadblock_ids)

                # 遍历每条 lane 对应的 roadblock id
                for route in lane_routes:

                    # 判断该 lane 是否在裁剪后的导航 route 上
                    lane_on_route.append(route in pruned_route_roadblock_ids)

            # 左右边界已经在处理 LANE 时一起处理了
            # 因此单独遇到 LEFT_BOUNDARY 或 RIGHT_BOUNDARY 时直接跳过
            elif feature_name == 'LEFT_BOUNDARY' or feature_name == 'RIGHT_BOUNDARY':
                continue
            
            # 将当前特征 coords 从全局坐标系转换到自车局部坐标系
            coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_ego_state)

            # 保存当前特征的局部坐标
            array_output[f"vector_set_map.coords.{feature_name}"] = coords

            # 保存当前特征的有效性 mask
            array_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            # 如果存在交通灯数据，则保存交通灯数组
            if tl_data is not None:
                array_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data


    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    """

    # 对转换后的地图元素做进一步后处理，生成最终模型输入
    for feature_name in map_features:

        # 处理 lane 特征
        if feature_name == "LANE":

            # 取出自车局部坐标系下的 lane 中心线
            polylines = array_output[f'vector_set_map.coords.{feature_name}']

            # 取出自车局部坐标系下的左边界
            left_boundary = array_output[f"vector_set_map.coords.LEFT_BOUNDARY"]

            # 取出自车局部坐标系下的右边界
            right_boundary = array_output[f"vector_set_map.coords.RIGHT_BOUNDARY"]

            # 取出 lane 对应的交通灯状态
            traffic_light_state = array_output[f'vector_set_map.traffic_light_data.{feature_name}']

            # 取出 lane 有效性 mask
            avails = array_output[f'vector_set_map.availabilities.{feature_name}']

            # 将 lane 中心线、方向向量、边界相对向量和交通灯状态拼接成 12 维 lane 特征
            vector_map_lanes = _lane_polyline_process(polylines, left_boundary, right_boundary, avails, traffic_light_state)

        # 处理 route lane 特征
        elif feature_name == "ROUTE_LANES":

            # loc 表示已经写入 route_lanes 的数量
            loc = 0

            # TODO: add has speed limit
            # 初始化 route lane 特征数组
            # 形状为 [max_route_lanes, num_points, lane_feature_dim]
            # 其中 num_points 和 lane_feature_dim 与 vector_map_lanes 保持一致
            vector_map_route_lanes = np.zeros((max_elements["ROUTE_LANES"], vector_map_lanes.shape[-2], vector_map_lanes.shape[-1]), dtype=np.float32)

            # 初始化 route lane 限速数组
            route_lanes_speed_limit = np.zeros((max_elements["ROUTE_LANES"], 1), dtype=np.float32)

            # 初始化 route lane 是否有限速的数组
            route_lanes_has_speed_limit = np.zeros((max_elements["ROUTE_LANES"], 1), dtype=np.bool_)

            # 遍历所有 lane 的 route 标记
            for i in range(len(lane_on_route)):

                # 如果第 i 条 lane 在导航 route 上，则加入 route_lanes
                if lane_on_route[i] == True:
                    vector_map_route_lanes[loc] = vector_map_lanes[i]
                    route_lanes_speed_limit[loc] = lane_speed_limit_array[i]
                    route_lanes_has_speed_limit[loc] = lane_has_speed_limit_array[i]
                    loc += 1

                # 如果 route_lanes 数量已经达到上限，则停止写入
                if loc == max_elements["ROUTE_LANES"]:
                    break

        # 其他 feature 暂不做额外后处理
        else:
            pass

    # 组织最终输出字典
    # lanes：所有附近 lane 的特征
    # lanes_speed_limit：所有 lane 的限速
    # lanes_has_speed_limit：所有 lane 是否有限速
    # route_lanes：属于导航路线的 lane 特征
    # route_lanes_speed_limit：route lane 的限速
    # route_lanes_has_speed_limit：route lane 是否有限速
    vector_map_output = {'lanes': vector_map_lanes, 'lanes_speed_limit': lane_speed_limit_array, 'lanes_has_speed_limit': lane_has_speed_limit_array, \
                         'route_lanes': vector_map_route_lanes, 'route_lanes_speed_limit': route_lanes_speed_limit, 'route_lanes_has_speed_limit': route_lanes_has_speed_limit}

    # 返回模型需要的地图输入
    return vector_map_output