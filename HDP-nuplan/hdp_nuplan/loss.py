# 导入类型注解：Any 表示任意值类型，Dict 表示字典，Tuple 表示元组；
# Callable 和 List 是原实现保留的类型工具，当前函数体没有直接使用。
from typing import Any, Callable, Dict, List, Tuple
# 导入 PyTorch，用于轨迹差分、随机采样、扩散加噪和损失计算。
import torch
# nn.Module 用于标注传入的模型类型。
import torch.nn as nn

# 【HDP 与原 Diffusion-Planner 的区别：包命名空间】归一化器从
# diffusion_planner.utils.normalizer 切换到 hdp_nuplan.utils.normalizer。
from hdp_nuplan.utils.normalizer import StateNormalizer
# 【HDP 与原 Diffusion-Planner 的区别：积分轨迹损失】HDP 新增位移积分工具，
# 用于把模型预测的相邻时刻运动增量恢复成位置轨迹；原版 loss 没有该依赖。
from hdp_nuplan.utils.traj_kinematics import detached_integral


# HDP 单个 batch 的扩散损失函数，主要完成：
# 1. 把自车未来位置转换成相邻时刻运动增量并归一化；
# 2. 采样扩散时间和高斯噪声，构造加噪轨迹；
# 3. 把模型输出转换到指定监督空间并计算自车扩散损失；
# 4. 对预测增量积分，计算位置轨迹混合损失。
def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    # 【HDP 与原 Diffusion-Planner 的区别：SDE 接口】原版只接收 marginal_prob
    # 函数；HDP 接收完整 sde 对象，因为还需要调用 transform 转换扩散参数化。
    sde,

    # futures 仍保持 (ego_future, neighbors_future, neighbor_future_mask) 的兼容接口。
    futures: Tuple[torch.Tensor, torch.Tensor],
    
    # 自车运动状态归一化器，以及由调用方创建并由本函数填充的 loss 字典。
    norm: StateNormalizer,
    loss: Dict[str, Any],

    # model_type 表示模型输出参数化。
    model_type: str,
    # 【HDP 与原 Diffusion-Planner 的区别：独立监督空间】原版只用 model_type，
    # HDP 新增 supervision_type，使模型输出类型与训练监督目标可以不同。
    supervision_type: str = None,
    # 扩散时间的最小值，避免采样到 t=0 引发数值问题。
    eps: float = 1e-3,
    # 积分轨迹混合损失的梯度窗口；0 表示普通 cumsum，不使用 stop-gradient。
    detach_window_size: int = 0,
):
    # 从兼容 tuple 中解包自车未来、邻车未来和邻车 padding mask。
    ego_future, neighbors_future, neighbor_future_mask = futures
    # 【HDP 与原 Diffusion-Planner 的区别：仅自车规划】原版把自车和邻车未来轨迹
    # 组织成联合扩散目标，并计算 neighbor_prediction_loss；HDP 只对自车计算扩散损失，
    # neighbors_future 及其 mask 仅为兼容数据集接口而保留，不再参与预测目标。
    # B 是 batch size，T 是未来时间步数，末维是 [x, y, cos(heading), sin(heading)]。
    B, T, _ = ego_future.shape
    # 【HDP 与原 Diffusion-Planner 的区别：扩散状态表示】原版直接对自车和邻车的
    # 未来位置轨迹加噪，并在时间轴前拼接当前状态；HDP 对自车 x/y 做相邻时刻差分，
    # 学习局部位移/速度增量，且不再拼接一个不加噪的当前状态时间步。
    # 在序列前补零后做 diff，使第一个 x/y 增量相对于自车坐标系原点计算。
    ego_future_vel = torch.diff(
        torch.cat([torch.zeros_like(ego_future[:, :1, :], device=ego_future.device), ego_future], dim=-2),
        dim=-2
    )
    # heading 不做时间差分，恢复为原轨迹中的 cos/sin 表示。
    ego_future_vel[..., 2:] = ego_future[..., 2:]
    # 对运动增量执行状态归一化；all_gt 形状为 [B, T, 4]。
    all_gt = norm(ego_future_vel)

    # 为每个 batch 样本从 [eps, 1] 随机采样一个扩散时间，形状为 [B]。
    t = torch.rand(B, device=all_gt.device) * (1 - eps) + eps # [B,]
    # 采样与自车运动增量同形状的标准高斯噪声，形状为 [B, T, 4]。
    z = torch.randn_like(all_gt, device=all_gt.device) # [B, T, 4]

    # 根据 SDE 的边缘概率得到时刻 t 的均值和标准差。
    mean, std = sde.marginal_prob(all_gt, t)
    # 把 std 从 [B] 整理为可与 [B, T, 4] 广播运算的形状 [B, 1, 1]。
    std = std.view(-1, *([1] * (len(all_gt.shape)-1)))

    # 执行扩散前向加噪：x_t = mean + std * z。
    xT = mean + std * z
    
    # 在原场景观测中加入加噪后的自车运动序列和扩散时间，组成模型输入。
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    # 【HDP 与原 Diffusion-Planner 的区别：模型输出维度】原版 Decoder 联合输出
    # [B, ego+neighbors, current+future, 4]；HDP 只输出自车未来 [B, T, 4]，
    # 因而不需要去掉 agent 维或额外的当前状态时间步。
    _, decoder_output = model(merged_inputs) # [B, T, 4]
    # "score" 是沿用的输出字典键，张量实际表示哪种参数化由 model_type 决定。
    model_pred = decoder_output["score"] # [B, T, 4]
    
    ##########################################################################
    # Transformation of model prediction and loss space
    # The model outputs *model_type* and is supervised with *supervision_type*
    ##########################################################################
    # 未显式指定监督空间时，保持与原版兼容：监督类型等于模型输出类型。
    supervision_type = supervision_type if supervision_type is not None else model_type
    # 例如 x_start->noise 表示把模型的 x_start 输出转换成 noise 后再计算 loss。
    pred_pattern = f"{model_type}->{supervision_type}"
    # 【HDP 与原 Diffusion-Planner 的区别：预测空间转换】原版直接根据 model_type
    # 计算 loss；HDP 先把模型输出统一转换到 supervision_type 对应空间。
    score = sde.transform(pred_pattern, model_pred, t, xT)

    # score 监督使用稳定形式 ||score * std + noise||^2，避免小方差下数值放大。
    if supervision_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1) # to avoid exploding variance
    # x_start 监督：预测去噪后的自车运动增量，与归一化真值 all_gt 做平方误差。
    elif supervision_type == "x_start":
        dpm_loss = torch.sum((score - all_gt)**2, dim=-1)
    # 【HDP 与原 Diffusion-Planner 的区别：新增 noise/v 损失】原版没有下面两个分支；
    # noise 分支拟合采样噪声 z；v 分支只在实际使用 v 监督时，才把 noise 转换成
    # 扩散 v-prediction target，避免其他监督类型产生无用计算和临时张量。
    # 这里的 v 是扩散参数化，不是车辆的物理速度。
    elif supervision_type == "noise":
        dpm_loss = torch.sum((score - z)**2, dim=-1)
    elif supervision_type == "v":
        v = sde.transform("noise->v", z, t, xT)
        dpm_loss = torch.sum((score - v)**2, dim=-1) 
    # 【HDP 与原 Diffusion-Planner 的区别：无邻车预测损失】HDP 直接对全部自车
    # 时间步求平均作为 ego_planning_loss，不再筛选邻车有效点或生成 neighbor_prediction_loss。
    loss["ego_planning_loss"] = dpm_loss.mean()
    
    ##########################################################################
    #                              Hybrid Loss                               #
    # Integration performed in \tau_0 space
    ##########################################################################
    # 【HDP 与原 Diffusion-Planner 的区别：混合轨迹损失】原版只计算扩散参数空间
    # 的自车/邻车损失；HDP 新增位置轨迹约束，弥补逐点运动增量误差的长期累积。
    # 无论 model_type 是什么，先把原始模型输出转换成 x_start，即去噪后的运动增量。
    pred_v = sde.transform(f"{model_type}->x_start", model_pred, t, xT)
    # 把归一化运动增量还原到真实数值尺度。
    pred_v = norm.inverse(pred_v)
    # 只取 x/y 增量并沿时间积分，恢复自车预测位置轨迹。
    # 本参数必须由训练入口显式控制，不能再把 Detached Integral 窗口写死为 10。
    # 当前正式监督训练传 0，使位置损失向此前全部 displacement 传播梯度。
    pred_x = detached_integral(
        pred_v[..., :2],
        detach_window_size=detach_window_size,
    )
    # 预测位置与自车未来 x/y 真值的均方误差，作为混合损失项。
    loss["ego_planning_hybrid_loss"] = torch.sum((pred_x - ego_future[..., :2])**2, dim=-1).mean()

    # 检查扩散逐点损失中是否出现 NaN，发现数值异常时立即终止并输出噪声信息。
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    # 返回已填充的 loss 字典和 Decoder 原始输出，保持与训练调用方的接口一致。
    return loss, decoder_output
