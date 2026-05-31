# 从 typing 模块中导入 Dict 类型。
# Dict 用于标注函数参数的类型，表示该参数应当是一个字典。
from typing import Dict

# 导入 PyTorch。
# 该代码使用 PyTorch 的神经网络模块、张量运算以及关闭梯度计算的上下文管理器。
import torch

# 导入项目中封装的 DPM-Solver 实现。
# 后续会使用该模块中的：
# 1. NoiseScheduleVP：构造扩散模型的噪声调度；
# 2. model_wrapper：统一模型预测函数的输入输出形式；
# 3. DPM_Solver：执行反向扩散采样。
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


# 定义基于 DPM-Solver++ 的扩散模型采样函数。
#
# 该函数的整体流程为：
# 1. 根据给定参数创建线性噪声调度器；
# 2. 将原始神经网络模型包装为 DPM-Solver 可调用的标准预测函数；
# 3. 创建 DPM-Solver++ 求解器；
# 4. 从初始高斯噪声 x_T 出发，执行多步反向扩散；
# 5. 返回去噪后的生成结果。
def dpm_sampler(
        # 已训练完成的扩散模型。
        # 模型需要继承 torch.nn.Module，并且应当包含 model_type 属性。
        # model_type 用于说明模型预测的是噪声、原始样本、速度变量还是 score。
        model: torch.nn.Module, 

        # 扩散过程终点时刻 T 的样本。
        # 在常见扩散模型中，x_T 通常是从标准高斯分布中采样得到的随机噪声。
        # DPM-Solver 将从 x_T 开始逐步反向去噪。
        x_T, 

        # 传递给模型的额外参数。
        # 例如条件信息、场景编码结果、轨迹约束或其他上下文信息。
        #
        # 注意：这里使用了可变字典作为默认参数。
        # 代码本身可以运行，但在编写新代码时，通常更推荐将默认值设为 None，
        # 再在函数内部创建字典，以避免多次函数调用共享同一个字典对象。
        other_model_params: Dict={}, 

        # DPM-Solver 的采样步数。
        # 步数越多，通常采样精度越高，但推理时间也会增加。
        # 这里默认使用 10 步。
        diffusion_steps=10,

        # 创建噪声调度器 NoiseScheduleVP 时使用的额外参数。
        # 例如 beta_0、beta_1、continuous_beta_0、continuous_beta_1 等。
        #
        # 当前函数内部已经固定 schedule='linear'，
        # 因此该字典用于补充线性噪声调度的其他配置。
        noise_schedule_params: Dict = {},

        # 创建模型包装函数 model_wrapper 时使用的额外参数。
        # 例如时间输入的处理方式、分类器引导参数或条件引导配置等。
        model_wrapper_params: Dict = {},

        # 创建 DPM_Solver 求解器时使用的额外参数。
        # 例如校正函数、动态阈值策略或其他求解器选项。
        dpm_solver_params: Dict = {},

        # 执行 dpm_solver.sample(...) 时使用的额外参数。
        # 例如采样起止时间、较低阶求解器设置或其他采样控制选项。
        sample_params: Dict = {}
    ):
    
    # 关闭 PyTorch 的自动求导功能。
    #
    # 采样阶段只需要使用训练完成的模型进行前向推理，
    # 不需要计算梯度，也不需要执行反向传播。
    #
    # 使用 torch.no_grad() 可以：
    # 1. 减少显存占用；
    # 2. 降低计算开销；
    # 3. 加快采样速度。
    with torch.no_grad():

        # 创建 Variance Preserving（VP）类型的噪声调度器。
        #
        # 噪声调度器用于描述扩散过程中：
        # 1. 原始数据 x_0 的保留比例；
        # 2. 高斯噪声的注入比例；
        # 3. 信噪比随时间 t 的变化规律；
        # 4. DPM-Solver 进行时间离散化和高阶更新时所需的中间量。
        #
        # schedule='linear' 表示使用线性 beta 调度。
        # 也就是说，噪声强度 beta_t 随时间 t 线性变化。
        #
        # **noise_schedule_params 会将字典中的键值对展开，
        # 并作为额外参数传递给 NoiseScheduleVP。
        noise_schedule = dpm.NoiseScheduleVP(
            schedule='linear',
            **noise_schedule_params
        )

        # 将原始扩散模型包装为 DPM-Solver 使用的标准预测函数。
        #
        # 不同扩散模型可能采用不同的训练目标，例如：
        # 1. "noise"：预测加入到样本中的噪声；
        # 2. "x_start"：直接预测原始无噪声样本 x_0；
        # 3. "v"：预测 velocity 参数化变量；
        # 4. "score"：预测数据分布的 score，即对数概率密度的梯度。
        #
        # model_wrapper 会根据 model_type 对模型输出进行统一转换，
        # 使 DPM-Solver 可以按照标准形式调用模型。
        model_fn = dpm.model_wrapper(
            # 传入已经训练完成的扩散模型。
            model,  # use your noise prediction model here

            # 传入上面创建的噪声调度器。
            # 包装函数需要根据噪声调度计算不同时间步对应的参数。
            noise_schedule,

            # 读取模型自身保存的预测类型。
            # 注释中列举了可选类型：
            # "noise"、"x_start"、"v" 或 "score"。
            model_type=model.model_type,  # or "x_start" or "v" or "score"

            # 将额外的模型参数传递给原始模型。
            # 例如条件生成任务中的场景特征、地图信息或历史轨迹等。
            model_kwargs=other_model_params,

            # 将其他包装器配置参数展开并传入 model_wrapper。
            **model_wrapper_params
        )

        # 创建 DPM-Solver 求解器。
        #
        # DPM-Solver 将扩散模型的反向生成过程视为一个常微分方程
        # 或随机微分方程的数值求解问题，并使用高阶数值方法进行快速采样。
        #
        # 相比逐步执行大量去噪步骤的方法，
        # DPM-Solver 通常能够在较少步数下得到较好的生成结果。
        #
        # algorithm_type="dpmsolver++" 表示使用 DPM-Solver++。
        # DPM-Solver++ 通常对数据预测形式进行更新，
        # 在较少采样步数下具有较好的稳定性。
        #
        # **dpm_solver_params 会将额外的求解器配置展开并传入 DPM_Solver。
        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params) # w.o. dynamic thresholding

        # 原始代码中的说明：
        # 采样步数设置为 10 到 20 时，通常可以生成质量较好的样本。
        # 当 steps=20 时，采样结果通常已经接近收敛。
        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.

        # 从随机噪声 x_T 出发，执行反向扩散采样。
        #
        # 求解器会调用包装后的模型函数 model_fn，
        # 逐步估计反向扩散轨迹，并最终得到接近 x_0 的生成结果。
        sample_dpm = dpm_solver.sample(
            # 扩散过程终点处的初始噪声。
            # 反向扩散过程将从该张量开始。
            x_T,

            # 设置反向扩散过程的离散采样步数。
            # 默认值由函数参数 diffusion_steps 决定，默认为 10。
            steps=diffusion_steps,

            # 使用二阶 DPM-Solver 更新公式。
            #
            # 与一阶方法相比，二阶方法会利用更多局部信息，
            # 通常可以在较少采样步数下获得更高的数值精度。
            order=2,

            # 按照 logSNR 均匀划分采样节点。
            #
            # SNR 表示 Signal-to-Noise Ratio，即信噪比。
            # logSNR 表示信噪比的对数形式。
            #
            # 扩散过程在不同时间段的变化速度并不均匀。
            # 使用 logSNR 进行离散化，可以让采样节点更合理地覆盖
            # 高噪声区域和低噪声区域。
            skip_type="logSNR",

            # 使用多步法进行数值求解。
            #
            # multistep 方法会利用前面若干个时间节点上的模型预测结果，
            # 计算当前时间节点的更新值。
            #
            # 相比每一步都重新构造中间节点的 singlestep 方法，
            # multistep 方法通常可以减少模型调用次数。
            method="multistep",

            # 在完成主要采样步骤后，继续执行一次到 t=0 的去噪操作。
            #
            # 这样可以进一步减小最终样本中残留的噪声，
            # 使输出更加接近无噪声样本 x_0。
            denoise_to_zero=True,

            # 将其他采样控制参数展开并传递给 sample 方法。
            **sample_params
        )

    # 返回 DPM-Solver++ 生成的最终样本。
    #
    # 输出 sample_dpm 通常与输入 x_T 具有相同的张量形状，
    # 但其中的随机噪声已经经过反向扩散过程逐步转换为目标样本。
    return sample_dpm