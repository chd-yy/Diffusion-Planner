import os
from torch.utils.data import Dataset

from hdp_nuplan.utils.train_utils import openjson, opendata

class DiffusionPlannerData(Dataset):
    def __init__(
        self,
        data_dir,
        data_list,
        past_neighbor_num,
        predicted_neighbor_num,
        future_len,
        return_metadata=False,
    ):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        # RL rollout 需要用场景文件名重新加载对应的 NuPlan 场景特征；
        # 默认关闭以保持原监督训练返回的 11 个张量完全不变。
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.data_list)

    def _load_item(self, file_name):
        data = opendata(os.path.join(self.data_dir, file_name))

        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
        }

        return tuple(data.values())

    def get_by_name(self, file_name):
        """按缓存文件名读取场景，供 Replay Buffer 在 RL 更新阶段回放。"""
        return self._load_item(file_name)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        tensors = self._load_item(file_name)
        if self.return_metadata:
            return (*tensors, file_name)
        return tensors
