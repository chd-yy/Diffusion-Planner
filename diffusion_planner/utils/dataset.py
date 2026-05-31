# 导入 os 模块
# 这里主要用于拼接数据文件路径
import os

# 从 PyTorch 中导入 Dataset 基类
# 自定义数据集通常需要继承 Dataset，并实现 __len__ 和 __getitem__
from torch.utils.data import Dataset

# 导入自定义工具函数
# openjson：读取 json 文件，通常用于读取训练样本文件名列表
# opendata：读取 .npz 数据文件，返回 numpy 加载后的数据对象
from diffusion_planner.utils.train_utils import openjson, opendata


# 定义 Diffusion Planner 的训练数据集类
# 继承 torch.utils.data.Dataset，使其可以被 DataLoader 调用
class DiffusionPlannerData(Dataset):

    # 初始化数据集
    # data_dir：.npz 数据文件所在目录
    # data_list：保存 .npz 文件名列表的 json 文件路径
    # past_neighbor_num：历史输入中最多使用多少个邻居 agent
    # predicted_neighbor_num：未来预测中最多预测多少个邻居 agent
    # future_len：未来轨迹长度，当前代码中保存了该参数，但 __getitem__ 中没有直接使用
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len):

        # 保存数据目录
        self.data_dir = data_dir

        # 读取 json 文件，得到所有样本文件名列表
        # 例如 data_list 里可能保存：
        # ["map1_token1.npz", "map2_token2.npz", ...]
        self.data_list = openjson(data_list)

        # 保存历史邻居 agent 的最大数量
        # 后续会用它截取 neighbor_agents_past
        self._past_neighbor_num = past_neighbor_num

        # 保存需要预测未来轨迹的邻居 agent 最大数量
        # 后续会用它截取 neighbor_agents_future
        self._predicted_neighbor_num = predicted_neighbor_num

        # 保存未来轨迹长度
        # 当前代码中没有直接裁剪 ego_future 或 neighbor_future，但该配置可能用于外部保持接口一致
        self._future_len = future_len

    # 返回数据集样本数量
    # DataLoader 会调用该函数判断一个 epoch 有多少个样本
    def __len__(self):

        # 样本数量等于 data_list 中保存的文件名数量
        return len(self.data_list)

    # 根据索引 idx 读取一个训练样本
    # DataLoader 每次取数据时会调用该函数
    def __getitem__(self, idx):

        # 根据 idx 找到对应的 .npz 文件名
        # os.path.join(self.data_dir, self.data_list[idx]) 拼接完整路径
        # opendata 读取该 .npz 文件
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))

        # 读取当前自车状态
        # 通常包含：
        # x、y、cos heading、sin heading、vx、vy、ax、ay、steering angle、yaw rate
        ego_current_state = data['ego_current_state']

        # 读取自车未来轨迹真值
        # 通常形状类似 [future_steps, 3]
        # 3 维一般为 x、y、heading
        ego_agent_future = data['ego_agent_future']

        # 读取邻居 agent 历史轨迹
        # 只保留前 self._past_neighbor_num 个 agent
        # 用于控制输入给模型的历史邻居数量
        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]

        # 读取邻居 agent 未来轨迹真值
        # 只保留前 self._predicted_neighbor_num 个 agent
        # 用于控制模型需要预测的邻居数量
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        # 读取附近 lane 特征
        # 通常包含 lane 中心线坐标、方向向量、边界相对向量、交通灯编码等
        lanes = data['lanes']

        # 读取每条 lane 的限速值
        lanes_speed_limit = data['lanes_speed_limit']

        # 读取每条 lane 是否存在有效限速
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        # 读取导航路线相关的 lane 特征
        # route_lanes 是 lanes 中属于当前导航路线的部分
        route_lanes = data['route_lanes']

        # 读取 route_lanes 对应的限速值
        route_lanes_speed_limit = data['route_lanes_speed_limit']

        # 读取 route_lanes 是否存在有效限速
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        # 读取静态目标特征
        # 例如交通锥、路障、施工区标志等
        static_objects = data['static_objects']

        # 将读取出的各类特征重新组织成字典
        # key 是模型训练阶段使用的字段名
        data = {
            # 当前自车状态
            "ego_current_state": ego_current_state,

            # 自车未来轨迹真值
            "ego_future_gt": ego_agent_future,

            # 邻居 agent 历史轨迹输入
            "neighbor_agents_past": neighbor_agents_past,

            # 邻居 agent 未来轨迹真值
            "neighbors_future_gt": neighbor_agents_future,

            # 普通 lane 地图特征
            "lanes": lanes,

            # 普通 lane 限速
            "lanes_speed_limit": lanes_speed_limit,

            # 普通 lane 是否有限速
            "lanes_has_speed_limit": lanes_has_speed_limit,

            # 导航路线 lane 特征
            "route_lanes": route_lanes,

            # 导航路线 lane 限速
            "route_lanes_speed_limit": route_lanes_speed_limit,

            # 导航路线 lane 是否有限速
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,

            # 静态目标特征
            "static_objects": static_objects,
        }

        # 返回 data 字典中所有 value 组成的 tuple
        # 注意：这里不是返回字典，而是返回 tuple(data.values())
        # 因此训练代码中需要按照这个固定顺序解包
        return tuple(data.values())