# 导入 Python 内置的 warnings 模块。
# 该模块用于控制程序运行过程中产生的警告信息。
import warnings

# 导入 PyTorch。
# 该项目使用 PyTorch 加载模型参数、执行张量运算，并将模型部署到 CPU 或 GPU。
import torch

# 导入 NumPy。
# NumPy 主要用于处理模型输出的轨迹数组，以及执行反正切函数等数值运算。
import numpy as np

# 从 typing 模块中导入类型标注工具。
#
# Deque：
#   双端队列类型。在当前代码中用于表示按时间顺序保存的自车历史状态。
#
# Dict：
#   字典类型。用于标注模型输入、模型输出以及模型权重字典。
#
# List：
#   列表类型。用于表示由多个轨迹状态组成的序列。
#
# Type：
#   类型对象。用于声明 observation_type() 返回的是某一种观测数据类型。
from typing import Deque, Dict, List, Type


# 忽略程序运行过程中产生的所有 warning。
#
# 这样做可以让终端输出更加简洁，但也可能隐藏一些值得关注的问题，
# 例如依赖库版本不兼容、接口即将弃用或张量类型不匹配等警告。
warnings.filterwarnings("ignore")


# EgoState 表示 nuPlan 中某一个时刻的自车状态。
#
# 一个 EgoState 通常包含：
# 1. 自车的位置；
# 2. 自车的航向角；
# 3. 自车速度；
# 4. 自车加速度；
# 5. 车辆尺寸；
# 6. 时间戳；
# 7. 转向角等动态信息。
from nuplan.common.actor_state.ego_state import EgoState

# InterpolatableState 是一种可以进行时间插值的状态类型。
#
# 轨迹中的相邻状态之间通常存在时间间隔。
# 当仿真器查询任意时刻的轨迹状态时，可以利用插值获得近似结果。
from nuplan.common.utils.interpolatable_state import InterpolatableState

# TrajectorySampling 用于描述轨迹的采样方式。
#
# 它通常包含：
# 1. time_horizon：轨迹覆盖的总时间长度；
# 2. num_poses：轨迹中采样点的数量；
# 3. interval_length：相邻采样点之间的时间间隔。
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# AbstractTrajectory 是 nuPlan 中所有轨迹类的抽象基类。
#
# Planner 最终需要返回 AbstractTrajectory 或其子类对象，
# 供仿真器、轨迹跟踪控制器和评测模块使用。
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

# InterpolatedTrajectory 是支持插值的轨迹实现。
#
# 它内部保存离散状态序列，并允许根据时间戳查询轨迹中的状态。
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Observation 是 nuPlan 中观测信息的抽象类型。
#
# DetectionsTracks 表示带有目标跟踪信息的环境观测。
# 其中通常包括周围车辆、行人、自行车等交通参与者的轨迹信息。
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks

# transform_predictions_to_states 用于将模型输出的相对轨迹点转换为 nuPlan 的状态序列。
#
# 神经网络通常预测自车局部坐标系中的未来轨迹。
# 但是 nuPlan 仿真器需要使用 EgoState 等标准状态对象。
#
# 该函数会结合：
# 1. 模型输出的未来轨迹；
# 2. 当前自车状态；
# 3. 预测时间范围；
# 4. 相邻轨迹点时间间隔；
#
# 将模型预测结果转换为可以被仿真器使用的轨迹状态列表。
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states

# 导入 nuPlan 中规划器的抽象接口及其相关数据结构。
#
# AbstractPlanner：
#   所有规划器必须继承的抽象基类。
#
# PlannerInitialization：
#   仿真开始前传入规划器的初始化信息，例如地图接口和导航路线。
#
# PlannerInput：
#   每一次规划时传入规划器的输入信息，例如历史轨迹、目标检测结果和红绿灯状态。
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)


# 导入 Diffusion Planner 项目中真正执行神经网络推理的模型。
#
# 当前文件的 DiffusionPlanner 类负责对接 nuPlan 仿真框架；
# Diffusion_Planner 类负责执行扩散模型的推理过程。
from diffusion_planner.model.diffusion_planner import Diffusion_Planner

# 导入数据处理器。
#
# DataProcessor 负责将 nuPlan 中的历史状态、交通参与者、
# 地图元素和红绿灯状态转换为神经网络可以接收的张量。
from diffusion_planner.data_process.data_processor import DataProcessor

# 导入配置类。
#
# Config 通常保存模型结构、数据处理参数、归一化方式和推理参数。
from diffusion_planner.utils.config import Config


# 定义一个恒等函数。
#
# 恒等函数会直接返回传入的 predictions，不做任何修改。
#
# 参数：
# ego_state：
#   当前自车状态。该函数中没有使用该参数。
#
# predictions：
#   模型预测结果。
#
# 返回值：
#   原样返回 predictions。
#
# 注意：
#   在当前代码片段中，这个函数没有被调用。
#   它可能用于兼容某些接口，或者是此前调试过程中保留的函数。
def identity(ego_state, predictions):
    return predictions


# 定义用于接入 nuPlan 仿真框架的 Diffusion Planner 规划器。
#
# 该类继承 AbstractPlanner，因此必须实现 nuPlan 规定的核心接口。
#
# 该类的主要任务可以概括为：
#
# 1. 加载训练好的扩散模型权重；
# 2. 接收 nuPlan 仿真器提供的当前环境信息；
# 3. 将环境信息转换为模型输入；
# 4. 调用扩散模型生成未来轨迹；
# 5. 将模型输出转换为 nuPlan 可以识别的轨迹对象；
# 6. 将轨迹返回给仿真器。
class DiffusionPlanner(AbstractPlanner):

    # 初始化 DiffusionPlanner 对象。
    #
    # config：
    #   Diffusion Planner 项目的配置对象。
    #
    # ckpt_path：
    #   模型检查点文件的路径。
    #   检查点文件通常包含训练完成后的神经网络参数。
    #
    # past_trajectory_sampling：
    #   历史轨迹的采样参数。
    #   它决定模型可以观察多长时间的历史信息，以及历史轨迹包含多少个采样点。
    #
    # future_trajectory_sampling：
    #   未来轨迹的采样参数。
    #   它决定模型需要预测多长时间的未来轨迹，以及输出多少个轨迹点。
    #
    # enable_ema：
    #   是否加载指数滑动平均模型参数。
    #   EMA 是 Exponential Moving Average 的缩写。
    #   训练过程中对模型权重进行滑动平均，通常可以获得更加稳定的推理效果。
    #
    # device：
    #   模型运行设备。
    #   当前代码只支持 "cpu" 和 "cuda"。
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        # 检查 device 参数是否合法。
        #
        # 如果 device 既不是 "cpu" 也不是 "cuda"，程序会直接抛出异常。
        assert device in ["cpu", "cuda"], f"device {device} not supported"

        # 如果用户要求使用 GPU，则进一步检查当前环境中 CUDA 是否可用。
        #
        # torch.cuda.is_available() 返回 False 时，说明：
        # 1. 机器可能没有 NVIDIA GPU；
        # 2. NVIDIA 驱动可能没有正确安装；
        # 3. 当前 PyTorch 可能不是 CUDA 版本；
        # 4. CUDA 环境可能配置不正确。
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        # 保存未来轨迹的预测时间范围，单位为秒。
        #
        # 例如：
        # future_trajectory_sampling.time_horizon = 8.0
        #
        # 表示模型需要预测未来 8 秒内的自车轨迹。
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 

        # 计算相邻两个未来轨迹点之间的时间间隔，单位为秒。
        #
        # 计算公式为：
        # 轨迹总时长 / 轨迹点数量
        #
        # 例如：
        # 未来轨迹长度为 8 秒，共预测 16 个轨迹点，
        # 则相邻轨迹点之间的时间间隔为 0.5 秒。
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        # 保存配置对象。
        #
        # 后续构造扩散模型和数据处理器时都会使用该配置。
        self._config = config

        # 保存模型检查点路径。
        #
        # initialize() 方法会使用该路径加载模型权重。
        self._ckpt_path = ckpt_path

        # 保存历史轨迹采样参数。
        self._past_trajectory_sampling = past_trajectory_sampling

        # 保存未来轨迹采样参数。
        self._future_trajectory_sampling = future_trajectory_sampling

        # 保存是否加载 EMA 权重。
        self._ema_enabled = enable_ema

        # 保存模型推理设备。
        self._device = device

        # 创建真正执行扩散模型推理的神经网络对象。
        #
        # 此时仅完成模型结构的构造。
        # 训练后的模型参数会在 initialize() 中加载。
        self._planner = Diffusion_Planner(config)

        # 创建数据处理器。
        #
        # 数据处理器负责将 nuPlan 的原始仿真数据转换为模型输入张量。
        self.data_processor = DataProcessor(config)
        
        # 从配置对象中获取输入归一化器。
        #
        # 神经网络训练时通常会对输入数据进行归一化。
        # 推理时必须使用相同的归一化方式，否则数据分布会与训练阶段不一致。
        self.observation_normalizer = config.observation_normalizer


    # 返回当前规划器的名称。
    #
    # nuPlan 可以使用该名称标识当前使用的规划算法。
    def name(self) -> str:
        """
        Inherited.
        """

        # 返回规划器名称。
        return "diffusion_planner"
    

    # 声明该规划器需要哪一种观测数据。
    #
    # nuPlan 仿真器会根据返回值准备相应的环境观测信息。
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """

        # 返回 DetectionsTracks。
        #
        # 这说明 Diffusion Planner 需要获取周围交通参与者的目标跟踪信息。
        return DetectionsTracks


    # 初始化规划器。
    #
    # 该方法通常会在仿真正式开始之前调用一次。
    #
    # 主要工作包括：
    # 1. 保存地图接口；
    # 2. 保存导航路线信息；
    # 3. 从检查点文件中加载模型参数；
    # 4. 将模型切换到推理模式；
    # 5. 将模型移动到指定设备。
    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """

        # 保存 nuPlan 提供的地图接口。
        #
        # 后续数据处理过程中，需要使用地图接口提取道路、车道线、
        # 路口和其他矢量地图元素。
        self._map_api = initialization.map_api

        # 保存导航路线中包含的 roadblock ID。
        #
        # roadblock 可以理解为 nuPlan 地图中描述道路结构的一类元素。
        # 这些 ID 用于帮助规划器识别自车应当沿着哪一条路线行驶。
        self._route_roadblock_ids = initialization.route_roadblock_ids

        # 如果检查点路径不为空，则加载训练好的模型参数。
        if self._ckpt_path is not None:

            # 从检查点文件中读取状态字典。
            #
            # map_location=self._device 表示：
            # 将检查点中的张量直接加载到指定设备。
            #
            # 例如：
            # self._device = "cpu"  时，权重加载到 CPU；
            # self._device = "cuda" 时，权重加载到 GPU。
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device)
            
            # 如果启用了 EMA，则读取 ema_state_dict。
            #
            # EMA 权重通常是训练过程中模型参数的指数滑动平均结果。
            # 与某一个训练步骤的瞬时权重相比，它通常更加稳定。
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']

            # 如果没有启用 EMA，则尝试读取普通模型权重。
            else:

                # 某些检查点文件使用 "model" 作为键保存模型参数。
                #
                # 如果检查点中包含该键，就将 state_dict 更新为其中保存的模型参数。
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']

            # 处理分布式训练保存的模型参数名称。
            #
            # DDP 是 Distributed Data Parallel 的缩写。
            #
            # 使用 DDP 训练模型时，参数名称通常会带有 "module." 前缀。
            #
            # 例如：
            # 原始参数名称：
            # encoder.weight
            #
            # DDP 保存后的参数名称：
            # module.encoder.weight
            #
            # 当前字典推导式会：
            # 1. 只保留以 "module." 开头的参数；
            # 2. 删除参数名称开头的 "module."；
            # 3. 将处理后的参数名称和对应张量保存到新的字典中。
            #
            # 注意：
            # 这里严格保留了原始代码。
            #
            # 如果某个检查点中的参数名称本来就不包含 "module." 前缀，
            # 那么新的 model_state_dict 可能会变成空字典。
            # 是否会遇到这一问题，取决于项目中实际使用的检查点格式。
            # 本次仅添加注释，没有修改原始逻辑。
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}

            # 将处理后的模型参数加载到扩散模型中。
            #
            # load_state_dict() 会按照参数名称匹配模型结构中的各个权重张量。
            self._planner.load_state_dict(model_state_dict)

        # 如果没有提供检查点路径，则不加载训练好的参数。
        else:

            # 打印提示信息。
            #
            # 此时模型使用的是随机初始化参数，
            # 输出轨迹通常不具备实际意义。
            print("load random model")
        
        # 将模型切换为评估模式。
        #
        # eval() 会影响 Dropout、BatchNorm 等模块的行为。
        #
        # 推理阶段通常必须调用 eval()，
        # 以保证模型使用确定性的推理逻辑。
        self._planner.eval()

        # 将模型移动到指定设备。
        #
        # 例如：
        # self._device = "cpu"  时，模型在 CPU 上运行；
        # self._device = "cuda" 时，模型在 GPU 上运行。
        self._planner = self._planner.to(self._device)

        # 保存初始化信息。
        #
        # 当前代码片段中没有继续使用该成员变量，
        # 但项目中的其他逻辑可能会访问它。
        self._initialization = initialization


    # 将 nuPlan 仿真器提供的 PlannerInput 转换为神经网络输入张量。
    #
    # PlannerInput 中包含：
    # 1. 自车历史状态；
    # 2. 周围交通参与者的历史状态；
    # 3. 当前红绿灯信息；
    # 4. 其他仿真上下文。
    #
    # 返回值是一个字典。
    # 字典中的每一个键通常对应一种模型输入特征。
    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:

        # 读取历史观测信息。
        #
        # history 中通常包括：
        # 1. 自车过去若干帧的状态；
        # 2. 周围目标过去若干帧的检测和跟踪结果；
        # 3. 当前时刻的观测信息。
        history = planner_input.history

        # 获取红绿灯状态，并将其转换为列表。
        #
        # planner_input.traffic_light_data 可能是可迭代对象，
        # 转换为列表后便于数据处理器遍历。
        traffic_light_data = list(planner_input.traffic_light_data)

        # 调用 DataProcessor 的 observation_adapter() 方法构造模型输入。
        #
        # 该方法接收：
        # 1. history：自车和周围目标的历史状态；
        # 2. traffic_light_data：红绿灯状态；
        # 3. self._map_api：地图接口；
        # 4. self._route_roadblock_ids：导航路线；
        # 5. self._device：张量需要存放的设备。
        #
        # 返回值 model_inputs 是一个字典。
        # 字典中的值通常是 PyTorch 张量。
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # 返回构造好的神经网络输入。
        return model_inputs


    # 将神经网络输出转换为 nuPlan 可以使用的轨迹状态列表。
    #
    # outputs：
    #   扩散模型的输出字典。
    #
    # ego_state_history：
    #   自车历史状态队列。
    #   转换相对轨迹时，需要使用当前自车状态作为参考坐标系。
    #
    # 返回值：
    #   由多个 InterpolatableState 构成的列表。
    #   这些状态描述自车在未来不同时刻的位置和姿态。
    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        # 从模型输出字典中读取预测轨迹。
        #
        # outputs['prediction'] 是模型输出的轨迹张量。
        #
        # [0, 0] 表示取出：
        # 1. batch 维度中的第 0 个样本；
        # 2. 轨迹候选维度中的第 0 条轨迹。
        #
        # detach()：
        #   将张量从 PyTorch 的计算图中分离。
        #   推理阶段不需要继续进行梯度计算。
        #
        # cpu()：
        #   将张量移动到 CPU。
        #   NumPy 无法直接处理位于 GPU 上的张量。
        #
        # numpy()：
        #   将 PyTorch 张量转换为 NumPy 数组。
        #
        # astype(np.float64)：
        #   将数组元素转换为 64 位浮点数。
        #
        # 根据原始注释，最终数组形状为：
        # T × 4
        #
        # 其中：
        # T 表示未来轨迹点数量。
        #
        # 根据后续代码可以推断，每个轨迹点的 4 个数值大概率依次表示：
        # 1. x 坐标；
        # 2. y 坐标；
        # 3. 航向方向的余弦值；
        # 4. 航向方向的正弦值。
        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4

        # 根据航向方向的正弦值和余弦值计算航向角。
        #
        # np.arctan2(y, x) 用于计算二维向量的方向角。
        #
        # 这里传入：
        # y = predictions[:, 3]
        # x = predictions[:, 2]
        #
        # 因此计算的是：
        # atan2(sin(heading), cos(heading))
        #
        # 与直接预测航向角相比，预测 sin 和 cos 通常更加稳定，
        # 因为航向角在 -pi 和 pi 的边界处存在跳变问题。
        #
        # [..., None] 会在数组末尾增加一个维度。
        #
        # 原始形状：
        # T
        #
        # 增加维度后的形状：
        # T × 1
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]

        # 重新构造轨迹数组。
        #
        # predictions[..., :2] 取出每一个轨迹点的 x 和 y 坐标。
        #
        # heading 是刚才计算得到的航向角。
        #
        # np.concatenate(..., axis=-1) 表示沿最后一个维度拼接。
        #
        # 拼接前：
        # x、y 坐标形状为 T × 2；
        # heading 形状为 T × 1。
        #
        # 拼接后：
        # predictions 形状为 T × 3。
        #
        # 每一个轨迹点变为：
        # [x, y, heading]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        # 将模型输出的轨迹点转换为 nuPlan 标准状态。
        #
        # transform_predictions_to_states() 通常会完成以下工作：
        #
        # 1. 将模型预测的局部坐标系轨迹转换到全局坐标系；
        # 2. 根据自车当前状态恢复轨迹的绝对位置；
        # 3. 根据时间间隔为每个轨迹点生成对应时间戳；
        # 4. 构造 nuPlan 可以识别的状态对象。
        #
        # 参数说明：
        #
        # predictions：
        #   形状为 T × 3 的未来轨迹点。
        #   每个点包含 x、y 和 heading。
        #
        # ego_state_history：
        #   自车历史状态。
        #   其中最新的自车状态通常用于确定局部坐标系原点。
        #
        # self._future_horizon：
        #   未来轨迹覆盖的总时间长度。
        #
        # self._step_interval：
        #   相邻两个未来轨迹点之间的时间间隔。
        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        # 返回 nuPlan 标准状态序列。
        return states
    

    # 根据当前仿真输入生成未来轨迹。
    #
    # 这是规划器最核心的方法。
    #
    # nuPlan 仿真器通常会在每一个规划周期调用该方法。
    #
    # 整体流程为：
    #
    # PlannerInput
    #     ↓
    # 数据预处理
    #     ↓
    # 输入归一化
    #     ↓
    # Diffusion Planner 推理
    #     ↓
    # 提取预测轨迹
    #     ↓
    # 转换为 nuPlan 状态
    #     ↓
    # 构造可插值轨迹
    #     ↓
    # 返回给仿真器
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """

        # 将 nuPlan 仿真输入转换为模型可以接收的张量字典。
        inputs = self.planner_input_to_model_inputs(current_input)

        # 对模型输入进行归一化。
        #
        # 必须保证推理阶段使用的归一化方式与训练阶段一致。
        #
        # 否则，模型接收到的数据分布会发生变化，
        # 可能导致预测轨迹质量明显下降。
        inputs = self.observation_normalizer(inputs)        

        # 调用扩散模型生成未来轨迹。
        #
        # self._planner(inputs) 会执行模型前向推理。
        #
        # 返回值包含两个部分：
        # 1. 第一个返回值：当前代码中不需要使用，因此使用下划线接收；
        # 2. outputs：模型输出字典，其中包含预测轨迹。
        _, outputs = self._planner(inputs)

        # 将模型输出转换为可插值轨迹。
        #
        # 处理流程为：
        #
        # outputs
        #     ↓
        # outputs_to_trajectory()
        #     ↓
        # 未来状态列表
        #     ↓
        # InterpolatedTrajectory
        #
        # current_input.history.ego_states：
        #   自车历史状态队列。
        #
        # InterpolatedTrajectory：
        #   支持根据任意时间点查询轨迹状态。
        #   后续仿真控制器可以利用该轨迹执行轨迹跟踪。
        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        # 将最终轨迹返回给 nuPlan 仿真器。
        #
        # 在闭环仿真中，轨迹跟踪控制器会根据该轨迹计算车辆控制量，
        # 然后更新自车在下一帧的状态。
        return trajectory
