# 导入 copy 和 deepcopy
# copy 是浅拷贝，只复制最外层对象；
# deepcopy 是深拷贝，会递归复制内部对象。
# 当前代码中实际只使用了 copy，deepcopy 虽然导入了但没有使用。
from copy import copy, deepcopy

# 导入 PyTorch，用于 tensor 构造、设备迁移、归一化计算等
import torch

# 导入读取 json 文件的工具函数
# openjson 会读取 normalization_file_path 对应的 json 文件，并解析成 Python 字典
from diffusion_planner.utils.train_utils import openjson


# StateNormalizer 用于对扩散模型要预测/生成的状态数据进行归一化和反归一化
# 这里的 state 通常包括 ego 未来轨迹状态和 neighbor 未来轨迹状态
class StateNormalizer:

    # 初始化函数
    # mean：状态均值
    # std：状态标准差
    def __init__(self, mean, std):

        # 将 mean 转换成 torch.Tensor
        # 这样后续可以直接和模型中的 tensor 数据做归一化计算
        self.mean = torch.as_tensor(mean)

        # 将 std 转换成 torch.Tensor
        # std 用于除法归一化，表示每个状态维度的标准差
        self.std = torch.as_tensor(std)

    # 类方法：从 json 文件中构造 StateNormalizer
    # cls 表示当前类本身，相当于 StateNormalizer
    @classmethod
    def from_json(cls, args):

        # 从 args.normalization_file_path 中读取归一化参数
        # data 通常包含 ego、neighbor 等字段
        data = openjson(args.normalization_file_path)

        # 构造 mean
        # 第一项是 ego 的 mean
        # 后面 predicted_neighbor_num 项都是 neighbor 的 mean
        # 这样最终 mean 的结构对应：
        # [ego, neighbor_1, neighbor_2, ..., neighbor_N]
        # 其中 neighbor 的 mean 都是一样的，因为我们假设所有邻居的状态分布相同
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num

        # 构造 std
        # 第一项是 ego 的 std
        # 后面 predicted_neighbor_num 项都是 neighbor 的 std
        # 结构与 mean 保持一致
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num

        # 返回一个 StateNormalizer 实例
        return cls(mean, std)
    
    # 使 StateNormalizer 对象可以像函数一样被调用
    # 例如 normalizer(data)
    # 作用：对输入状态 data 做标准化
    def __call__(self, data):

        # 标准化公式：
        # normalized = (data - mean) / std
        # self.mean.to(data.device) 是为了确保 mean 和 data 在同一个设备上，例如都在 GPU 或都在 CPU
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    # 反归一化函数
    # 将标准化后的数据恢复到原始物理量尺度
    def inverse(self, data):

        # 反归一化公式：
        # original = normalized * std + mean
        return data * self.std.to(data.device) + self.mean.to(data.device)

    # 将当前 normalizer 中的 mean 和 std 转成普通 Python 字典
    # 常用于保存、打印或写入配置文件
    def to_dict(self):

        # detach()：从计算图中分离，避免梯度追踪
        # cpu()：移动到 CPU
        # numpy()：转成 numpy 数组
        # tolist()：转成 Python list，方便 json 序列化
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


# ObservationNormalizer 用于对模型输入观测数据进行归一化和反归一化
# 这里的 observation 通常包括地图、agent 历史、静态目标等输入特征
class ObservationNormalizer:

    # 初始化函数
    # normalization_dict 是一个字典，保存不同输入特征对应的 mean 和 std
    def __init__(self, normalization_dict):

        # 保存归一化参数字典
        # 格式通常类似：
        # {
        #   "neighbor_agents_past": {"mean": tensor(...), "std": tensor(...)},
        #   "lanes": {"mean": tensor(...), "std": tensor(...)},
        #   ...
        # }
        self._normalization_dict = normalization_dict

    # 类方法：从 json 文件中构造 ObservationNormalizer
    @classmethod
    def from_json(cls, args):

        # 如果 args 本身就是字符串，则认为它直接是 json 文件路径
        if isinstance(args, str):
            path = args

        # 否则认为 args 是参数对象，从其中读取 normalization_file_path
        else:
            path = args.normalization_file_path

        # 读取 json 文件中的归一化参数
        data = openjson(path)

        # 初始化新的归一化参数字典
        ndt = {}

        # 遍历 json 中的每个字段
        for k, v in data.items():

            # 跳过 ego 和 neighbor
            # 因为 ego 和 neighbor 的未来状态归一化由 StateNormalizer 负责
            # ObservationNormalizer 只处理输入观测特征
            if k not in ["ego", "neighbor"]:

                # 将每个特征的 mean 和 std 转成 float32 tensor
                ndt[k]= {"mean": torch.tensor(v["mean"], dtype=torch.float32), "std": torch.tensor(v["std"], dtype=torch.float32)}

        # 返回 ObservationNormalizer 实例
        return cls(ndt)

    # 使 ObservationNormalizer 对象可以像函数一样被调用
    # 例如 obs_normalizer(data)
    # 作用：对输入 data 字典中的各个特征做归一化
    def __call__(self, data):

        # 对 data 做浅拷贝
        # 这样 norm_data 是一个新的字典对象，但其中的 tensor 仍然和原始 data 共享引用
        norm_data = copy(data)

        # 遍历每个需要归一化的特征及其 mean/std
        for k, v in self._normalization_dict.items():

            # 如果当前归一化字典中的 key 不存在于 data 中，则跳过
            # 这样可以兼容某些输入特征缺失的情况
            if k not in data:  # Check if key `k` exists in `data`
                continue

            # 构造 padding mask
            # torch.ne(data[k], 0) 判断每个元素是否不等于 0
            # torch.sum(..., dim=-1) 对最后一个特征维度求和
            # 如果最后一维所有元素都是 0，则说明该点/该对象可能是 padding
            # mask=True 的位置表示 padding 数据
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0

            # 对当前特征做标准化
            # normalized = (x - mean) / std
            # mean 和 std 需要移动到 data[k] 所在设备上
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)

            # 对 padding 位置重新置 0
            # 避免原本全 0 的 padding 数据经过归一化后变成非零值
            norm_data[k][mask] = 0

        # 返回归一化后的数据字典
        return norm_data

    # 对观测数据进行反归一化
    # 将标准化后的输入特征恢复到原始尺度
    def inverse(self, data):

        # 对 data 做浅拷贝
        norm_data = copy(data)

        # 遍历每个需要反归一化的特征及其 mean/std
        for k, v in self._normalization_dict.items():

            # 如果当前 key 不存在于 data 中，则跳过
            if k not in data:  # Check if key `k` exists in `data`
                continue

            # 构造 padding mask
            # 这里同样认为最后一维全 0 的位置是 padding
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0

            # 反归一化公式：
            # original = normalized * std + mean
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)

            # padding 位置重新置 0
            # 保证无效填充数据仍然保持为 0
            norm_data[k][mask] = 0

        # 返回反归一化后的数据字典
        return norm_data

    # 将 ObservationNormalizer 中保存的归一化参数转换为普通 Python 字典
    # 常用于保存、打印或 json 序列化
    def to_dict(self):

        # 外层遍历每个特征 k
        # 内层遍历该特征的 mean 和 std
        # 将 tensor 转成 list，方便保存为 json
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}