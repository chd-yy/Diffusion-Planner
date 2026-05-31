# 从 typing 模块中导入 List 类型。
#
# List 通常用于为列表变量添加类型标注，例如：
#
# guidance_fns: List[Callable]
#
# 注意：
# 当前文件中尚未直接使用 List，但仍然保留该导入语句。
from typing import List

# 导入 PyTorch。
#
# 当前文件主要使用 PyTorch 完成：
# 1. 张量变形；
# 2. 张量加法；
# 3. 梯度路径切断；
# 4. 重复采样；
# 5. 随机噪声生成；
# 6. NaN 数值检查。
import torch

# 导入线性 Variance Preserving SDE。
#
# VPSDE_linear 用于描述正向扩散过程中的噪声强度变化规律。
#
# 当前文件中，sde 仅在后面的注释代码中使用。
# 如果启用对应代码，可以根据扩散时间 t 计算边缘分布标准差。
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear

# 导入碰撞规避引导函数。
#
# collision_guidance_fn 会根据自车与周围车辆之间的距离，
# 构造一个用于扩散采样过程的奖励值。
#
# 后续 GuidanceWrapper 会统一调用该函数。
from diffusion_planner.model.guidance.collision import collision_guidance_fn


# 设置每个输入样本的重复采样次数。
#
# 当前值为 1，表示每个样本只保留一份。
#
# 注意：
# 与 N 有关的 repeat_interleave 代码目前处于注释状态，
# 因此该变量暂时不会影响程序执行结果。
N = 1

# 创建线性 VP-SDE 实例。
#
# 当前对象仅在后面的注释代码中使用。
#
# 如果启用对应代码，可以通过：
#
# sde.marginal_prob_std(t_input)
#
# 计算当前扩散时间对应的噪声标准差。
sde = VPSDE_linear()


# 定义 GuidanceWrapper。
#
# 该类用于统一管理一个或多个 guidance 函数。
#
# 在扩散模型采样过程中，可以通过该包装器：
# 1. 根据模型输出修正输入轨迹；
# 2. 将归一化状态还原为物理量；
# 3. 调用一个或多个 guidance 函数；
# 4. 累加不同 guidance 函数产生的奖励值；
# 5. 返回总奖励，用于后续梯度引导。
class GuidanceWrapper:

    # 初始化 GuidanceWrapper。
    def __init__(self):

        # 创建 guidance 函数列表。
        #
        # 当前只启用了一个引导函数：
        #
        # collision_guidance_fn
        #
        # 如果需要同时启用多个引导目标，
        # 可以继续在列表中添加其他 guidance 函数。
        #
        # 例如：
        # 1. 碰撞规避；
        # 2. 道路边界约束；
        # 3. 舒适性约束；
        # 4. 目标点到达约束；
        # 5. 轨迹平滑性约束。
        self._guidance_fns = [
            collision_guidance_fn
        ]

    # 定义 GuidanceWrapper 实例被调用时的行为。
    #
    # 定义 __call__ 后，可以像调用普通函数一样调用该类的实例：
    #
    # guidance_wrapper = GuidanceWrapper()
    # energy = guidance_wrapper(x_in, t_input, cond, ...)
    #
    # 参数说明：
    #
    # x_in：
    # 当前扩散采样过程中的轨迹张量。
    # 在当前函数开头，它通常是一个压平后的三维张量：
    #
    # [B, P, D]
    #
    # 其中：
    # 1. B 表示 batch size；
    # 2. P 表示参与预测的车辆数量；
    # 3. D 表示压平后的轨迹状态维度。
    #
    # t_input：
    # 当前扩散时间。
    #
    # cond：
    # 条件信息。
    # 当前包装器不会直接处理 cond，
    # 而是将其继续传递给具体的 guidance 函数。
    #
    # *args：
    # 接收额外的位置参数。
    #
    # **kwargs：
    # 接收额外的关键字参数。
    # 当前函数会从中读取：
    # 1. state_normalizer；
    # 2. observation_normalizer；
    # 3. model；
    # 4. model_condition；
    # 5. inputs。
    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        This function is a wrapper for the guidance functions in the model.
        """

        # 初始化总能量或总奖励。
        #
        # 当前初始值是 Python 整数 0。
        #
        # 当第一次执行：
        #
        # energy += guidance_fn(...)
        #
        # 后，energy 通常会转换为 PyTorch 张量。
        #
        # 如果启用了多个 guidance 函数，
        # 它们返回的奖励会依次累加。
        energy = 0
        
        # 从 kwargs 中读取状态归一化器。
        #
        # 模型通常在归一化后的状态空间中进行训练和预测。
        #
        # 后续会调用：
        #
        # state_normalizer.inverse(...)
        #
        # 将模型内部状态恢复到原始物理量尺度。
        state_normalizer = kwargs["state_normalizer"]

        # 从 kwargs 中读取观测量归一化器。
        #
        # 后续会调用：
        #
        # observation_normalizer.inverse(...)
        #
        # 将场景输入数据恢复到原始物理量尺度。
        observation_normalizer = kwargs["observation_normalizer"]
      
        # 读取输入轨迹张量的形状。
        #
        # x_in 的形状为：
        #
        # [B, P, D]
        #
        # 其中：
        # 1. B 表示 batch size；
        # 2. P 表示车辆数量；
        # 3. D 表示压平后的轨迹状态维度。
        #
        # 下划线 _ 表示第三个维度的具体数值
        # 在当前语句中不需要单独保存。
        B, P, _ = x_in.shape

        # 从 kwargs 中读取扩散模型。
        #
        # 后续会调用：
        #
        # model(x_in, t_input, **model_condition)
        #
        # 根据当前轨迹状态和扩散时间，
        # 生成模型预测结果。
        model = kwargs["model"]

        # 从 kwargs 中读取传递给模型的条件参数。
        #
        # model_condition 通常是一个字典，
        # 后续通过 **model_condition 展开后传递给模型。
        #
        # 其中可能包含：
        # 1. 场景编码；
        # 2. 历史轨迹特征；
        # 3. 地图特征；
        # 4. 条件掩码；
        # 5. 其他模型输入。
        model_condition = kwargs["model_condition"]
      
        # 根据模型输出计算轨迹修正量。
        #
        # model(x_in, t_input, **model_condition)：
        # 使用当前轨迹 x_in、扩散时间 t_input 和模型条件执行一次前向推理。
        #
        # model(...).detach()：
        # 切断模型输出的梯度路径。
        #
        # x_in.detach()：
        # 切断当前输入在该减法路径中的梯度传播。
        #
        # 两者相减后：
        #
        # x_fix = 模型预测结果 - 当前输入
        #
        # 因此，x_fix 表示模型预测结果相对于当前输入的修正量。
        #
        # 注意：
        # 由于参与减法的两个张量都执行了 detach()，
        # x_fix 本身不会保留梯度路径。
        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()

        # 将压平后的轨迹修正量恢复为四维状态格式。
        #
        # 原始形状通常为：
        #
        # [B, P, D]
        #
        # 恢复后的形状为：
        #
        # [B, P, T, 4]
        #
        # 其中：
        # 1. B 表示 batch size；
        # 2. P 表示车辆数量；
        # 3. T 表示轨迹时间点数量；
        # 4. 4 表示每个时间点的状态维度。
        #
        # 在当前项目中，每个状态通常可以表示为：
        #
        # (x, y, cos_h, sin_h)
        x_fix = x_fix.reshape(B, P, -1, 4)

        # 将所有车辆在第 0 个时间点上的修正量设置为 0。
        #
        # x_fix[:, :, 0] 表示：
        # 1. 所有 batch；
        # 2. 所有车辆；
        # 3. 第 0 个时间点；
        # 4. 该时间点的全部 4 个状态分量。
        #
        # 这样可以固定轨迹起始状态，
        # 防止 guidance 过程修改当前时刻的已知状态。
        x_fix[:, :, 0] = 0.0

        # 将轨迹修正量重新压平，并加回输入轨迹。
        #
        # x_fix.reshape(B, P, -1)：
        # 将四维轨迹修正量恢复为与 x_in 相同的三维形式。
        #
        # x_in + x_fix：
        # 对当前轨迹应用模型预测得到的修正量。
        #
        # 由于：
        #
        # x_fix = 模型预测结果 - 原始输入
        #
        # 因此，在未将部分修正量置零的位置上，
        # 更新后的 x_in 数值等于模型预测结果。
        #
        # 对于第 0 个时间点，
        # 由于修正量被置零，
        # 更新后的 x_in 会保留原始输入值。
        x_in = x_in + x_fix.reshape(B, P, -1)
      
        # 以下代码目前处于注释状态，不会执行。
        #
        # 如果启用，可以将每个样本重复 N 次。
        #
        # 这样可以针对同一个输入场景生成多个候选轨迹，
        # 再分别执行 guidance 计算。
        #
        # torch.repeat_interleave(x_in, N, dim=0)：
        # 沿 batch 维度重复轨迹张量。
        #
        # torch.repeat_interleave(t_input, N, dim=0)：
        # 沿 batch 维度重复扩散时间。
        #
        # 字典推导式：
        #
        # {k: torch.repeat_interleave(v, N, dim=0) for k, v in kwargs["inputs"].items()}
        #
        # 会对 inputs 字典中的每个张量执行相同的 batch 维度复制。
        #
        # 注意：
        # 如果启用该代码，还需要确认后续使用的 B
        # 是否需要同步调整为重复后的 batch size。
        # x_in = torch.repeat_interleave(x_in, N, dim=0) # [B * N, P, T, 4]
        # t_input = torch.repeat_interleave(t_input, N, dim=0) # [B * N]        
        # kwargs["inputs"] = {k: torch.repeat_interleave(v, N, dim=0) for k, v in kwargs["inputs"].items()}
      
        # 以下代码目前处于注释状态，不会执行。
        #
        # 如果启用，可以根据当前扩散时间重新向自车轨迹加入随机噪声。
        #
        # 第一步：
        #
        # sigma_t = sde.marginal_prob_std(t_input)
        #
        # 根据线性 VP-SDE 计算当前时间对应的边缘分布标准差。
        #
        # 第二步：
        #
        # sigma_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
        #
        # 对标准差进行进一步缩放。
        #
        # 第三步：
        #
        # torch.randn_like(x_in[:, :1])
        #
        # 生成与自车轨迹形状一致的标准高斯噪声。
        #
        # 第四步：
        #
        # x_in[:, :1] + sigma_t[:, None, None] * torch.randn_like(x_in[:, :1])
        #
        # 仅向第 0 个车辆，也就是自车轨迹中加入随机噪声。
        #
        # 第五步：
        #
        # torch.cat([...], dim=1)
        #
        # 将加入噪声后的自车轨迹与保持不变的邻居车辆轨迹重新拼接。
        #
        # 注意：
        # 当前代码位于 x_in 恢复为四维形状之前。
        # 如果启用，需要确认 sigma_t 的广播维度
        # 与 x_in 的实际三维形状是否一致。
        # sigma_t = sde.marginal_prob_std(t_input)
        # sigma_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
        # x_in = torch.cat([x_in[:, :1] + sigma_t[:, None, None] * torch.randn_like(x_in[:, :1]), x_in[:, 1:]], dim=1)
      
        # 将轨迹状态还原到原始物理量尺度。
        #
        # 第一步：
        #
        # x_in.reshape(B, P, -1, 4)
        #
        # 将压平后的轨迹恢复为：
        #
        # [B, P, T, 4]
        #
        # 第二步：
        #
        # state_normalizer.inverse(...)
        #
        # 对归一化后的轨迹状态执行逆变换。
        #
        # 例如：
        # 1. 将归一化位置恢复为实际距离；
        # 2. 将缩放后的状态恢复为原始尺度；
        # 3. 使碰撞检测能够在物理空间中进行。
        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))

        # 将场景输入数据还原到原始物理量尺度。
        #
        # kwargs["inputs"] 通常是一个字典，
        # 其中可能包含：
        # 1. 邻居车辆历史轨迹；
        # 2. 邻居车辆尺寸；
        # 3. 有效性掩码；
        # 4. 其他观测量。
        #
        # observation_normalizer.inverse(...) 会对字典中的观测量执行逆归一化。
        #
        # 注意：
        # 当前语句会直接替换 kwargs["inputs"] 的值。
        kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])
      
        # 遍历所有已注册的 guidance 函数。
        #
        # 当前列表中只有：
        #
        # collision_guidance_fn
        #
        # 如果后续增加更多函数，
        # 它们会按照 self._guidance_fns 中的顺序依次执行。
        for guidance_fn in self._guidance_fns:

            # 调用当前 guidance 函数，并累加奖励值。
            #
            # 传入参数：
            #
            # x_in：
            # 已经恢复到原始物理尺度的轨迹状态。
            #
            # t_input：
            # 当前扩散时间。
            #
            # cond：
            # 条件信息。
            #
            # **kwargs：
            # 其他辅助参数。
            #
            # 对于 collision_guidance_fn，
            # kwargs 中通常还需要包含：
            # 1. inputs；
            # 2. model；
            # 3. state_normalizer；
            # 4. observation_normalizer；
            # 5. 其他运行参数。
            #
            # 如果存在多个 guidance 函数，
            # energy 会变为多个奖励值之和。
            energy += guidance_fn(x_in, t_input, cond, **kwargs)

        # 以下代码目前处于注释状态，不会执行。
        #
        # 这段代码展示了另一种组合 guidance 奖励的方式：
        # 1. 单独计算两个 guidance 函数的输出；
        # 2. 根据 energy2 的大小，在 energy1 和 energy2 之间进行选择。
        #
        # 当前正式实现采用的是直接累加方式。
        # energy1 = self._guidance_fns[0](x_in, t_input, cond, **kwargs)
        # energy2 = self._guidance_fns[1](x_in, t_input, cond, **kwargs)
        
        # 以下代码目前处于注释状态，不会执行。
        #
        # 如果启用：
        #
        # 当 energy2 < 1 时：
        #
        # energy = energy1
        #
        # 否则：
        #
        # energy = energy2
        #
        # 这种逻辑可用于根据某个条件动态选择不同引导目标。
        # energy = energy1 if energy2 < 1 else energy2
        
        # 检查最终奖励中是否存在 NaN。
        #
        # torch.isnan(energy)：
        # 判断 energy 中每一个元素是否为 NaN。
        #
        # .any()：
        # 判断是否至少存在一个 NaN。
        #
        # not：
        # 要求结果中不存在任何 NaN。
        #
        # 如果检测到 NaN，
        # assert 会立即抛出 AssertionError，
        # 防止异常数值继续进入后续扩散采样流程。
        assert not torch.isnan(energy).any()
          
        # 返回累计后的 guidance 奖励。
        #
        # 后续采样器通常会进一步计算：
        #
        # energy 对 x_in 的梯度
        #
        # 并使用该梯度调整扩散模型的采样方向。
        return energy