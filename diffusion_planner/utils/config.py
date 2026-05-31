# 导入 json 模块
# 用于读取 args_file 中保存的配置文件内容
import json

# 导入 PyTorch
# 这里主要用于 torch.as_tensor，将列表形式的 mean/std 转换成 tensor
import torch

# 导入状态归一化器和观测归一化器
# StateNormalizer：用于归一化/反归一化模型生成的状态，例如 ego 和 neighbor 轨迹状态
# ObservationNormalizer：用于归一化/反归一化模型输入观测，例如地图、agent 历史、静态目标等
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer


# Config 类用于加载推理或训练所需的配置
# 它会从 json 文件中读取参数，并把这些参数注册成当前对象的属性
# 同时还会恢复 state_normalizer、observation_normalizer 和 guidance_fn
class Config:
    
    # 初始化 Config 对象
    # args_file：配置文件路径，通常是一个 json 文件
    # guidance_fn：引导函数，用于扩散模型推理阶段的 guidance，例如碰撞约束、速度约束、舒适性约束等
    def __init__(
            self,
            args_file,
            guidance_fn
    ):

        # 打开配置文件 args_file
        # 'r' 表示只读模式
        with open(args_file, 'r') as f:

            # 将 json 文件内容读取为 Python 字典
            # args_dict 中保存了模型配置、归一化参数、推理参数等
            args_dict = json.load(f)
            
        # 遍历配置字典中的所有键值对
        for key, value in args_dict.items():

            # 将每个配置项动态设置为当前 Config 对象的属性
            # 例如 args_dict 中有 {"predicted_neighbor_num": 10}
            # 那么执行后就可以通过 self.predicted_neighbor_num 访问
            setattr(self, key, value)

        # 根据配置文件中保存的 state_normalizer 参数重新构造 StateNormalizer 对象
        # self.state_normalizer 原本通常是一个字典：
        # {
        #     "mean": ...,
        #     "std": ...
        # }
        # 这里将其转换为真正可调用的 StateNormalizer
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])

        # 根据配置文件中保存的 observation_normalizer 参数重新构造 ObservationNormalizer 对象
        # self.observation_normalizer 原本通常是一个嵌套字典：
        # {
        #     "neighbor_agents_past": {"mean": ..., "std": ...},
        #     "lanes": {"mean": ..., "std": ...},
        #     ...
        # }
        self.observation_normalizer = ObservationNormalizer({

            # 遍历每个观测特征的归一化参数
            # k 是特征名称，例如 "neighbor_agents_past"、"lanes"、"static_objects"
            # v 是该特征对应的 mean/std 字典
            k: {

                # 将当前特征的 mean 转换为 torch.Tensor
                # 这样后续可以直接和模型输入 tensor 做归一化计算
                'mean': torch.as_tensor(v['mean']),

                # 将当前特征的 std 转换为 torch.Tensor
                # std 用于执行标准化公式：x_norm = (x - mean) / std
                'std': torch.as_tensor(v['std'])

            # 遍历 self.observation_normalizer 字典中的所有特征
            } for k, v in self.observation_normalizer.items()
        })
        
        # 保存 guidance 函数
        # 该函数通常会在扩散采样过程中提供额外梯度或能量约束
        # 用于实现安全、舒适、目标速度、可行驶区域等引导行为
        self.guidance_fn = guidance_fn