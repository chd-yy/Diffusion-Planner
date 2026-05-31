# 导入类型注解工具
# Any：表示任意类型
# Callable：表示可调用对象，比如函数
# Dict：字典类型
# List：列表类型
# Tuple：元组类型
from typing import Any, Callable, Dict, List, Tuple

# 导入 PyTorch
# 用于张量计算、随机采样、拼接、mask 操作、loss 计算等
import torch

# 导入 PyTorch 神经网络模块
# nn.Module 用于标注 model 的类型
import torch.nn as nn

# 导入状态归一化器
# StateNormalizer 用于对未来轨迹状态进行归一化和反归一化
from diffusion_planner.utils.normalizer import StateNormalizer


# 扩散模型训练损失函数
# 该函数主要完成：
# 1. 将 ego future 和 neighbor future 组织成联合未来轨迹；
# 2. 对未来轨迹加入扩散噪声；
# 3. 将加噪轨迹输入模型；
# 4. 根据模型输出计算扩散损失；
# 5. 分别统计 ego planning loss 和 neighbor prediction loss。
def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   

    # 从 futures 中解包未来轨迹数据
    # ego_future：自车未来轨迹真值，通常形状为 [B, T, 4]
    # neighbors_future：邻居车辆未来轨迹真值，通常形状为 [B, Pn, T, 4]
    # neighbor_future_mask：邻居未来轨迹 mask，True 通常表示该位置无效或 padding
    ego_future, neighbors_future, neighbor_future_mask = futures

    # 将无效 mask 取反，得到有效 mask
    # neighbors_future_valid=True 表示该邻居未来轨迹点有效，可以参与 loss 计算
    # 形状通常为 [B, Pn, T]
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    # 读取 neighbors_future 的形状
    # B：batch size
    # Pn：邻居 agent 数量
    # T：未来时间步数
    # _：状态维度，通常为 4
    B, Pn, T, _ = neighbors_future.shape

    # 取出当前自车状态的前 4 维
    # 通常是 [x, y, cos heading, sin heading]
    # 同时取出每个邻居 agent 历史最后一帧的前 4 维，作为邻居当前状态
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]

    # 构造邻居当前状态 mask
    # 如果 neighbors_current 的前 4 维全为 0，说明该邻居当前状态是 padding
    # neighbor_current_mask=True 表示该邻居当前状态无效
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

    # 将邻居当前状态 mask 和邻居未来状态 mask 拼接
    # neighbor_current_mask.unsqueeze(-1)：形状从 [B, Pn] 变成 [B, Pn, 1]
    # neighbor_future_mask：形状通常为 [B, Pn, T]
    # 拼接后 neighbor_mask 形状为 [B, Pn, 1+T]
    # 其中第 0 个时间步对应当前状态，后面 T 个时间步对应未来状态
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    # 将 ego_future 和 neighbors_future 拼接成统一的未来轨迹张量
    # ego_future[:, None, :, :] 会在 agent 维度上增加一维，使 ego 也变成一个 agent
    # 拼接后 gt_future 形状为 [B, 1+Pn, T, 4]
    # 第 0 个 agent 是 ego，后面是邻居 agents
    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]

    # 将 ego 当前状态和 neighbors 当前状态拼接
    # ego_current[:, None] 形状为 [B, 1, 4]
    # neighbors_current 形状为 [B, Pn, 4]
    # 拼接后 current_states 形状为 [B, 1+Pn, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    # P 表示参与扩散建模的 agent 数量
    # P = 1 个 ego + Pn 个 neighbor
    P = gt_future.shape[1]

    # 为每个 batch 样本随机采样一个扩散时间 t
    # t 从 [eps, 1] 中采样，避免 t=0 造成数值问题
    # 形状为 [B]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]

    # 采样标准高斯噪声 z
    # 形状与 gt_future 相同，为 [B, P, T, 4]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    # 将当前状态和归一化后的未来轨迹拼接
    # current_states[:, :, None, :]：形状为 [B, P, 1, 4]
    # norm(gt_future)：对未来轨迹做状态归一化，形状为 [B, P, T, 4]
    # all_gt 形状为 [B, P, 1+T, 4]
    # 第 0 个时间步是当前状态，后面是未来轨迹
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)

    # 对邻居 agent 的无效位置置 0
    # all_gt[:, 1:] 表示只处理邻居，不处理 ego
    # neighbor_mask=True 的位置表示 padding 或无效状态
    all_gt[:, 1:][neighbor_mask] = 0.0

    # 根据扩散边缘概率函数 marginal_prob 计算加噪分布的 mean 和 std
    # 输入 all_gt[..., 1:, :] 表示只对未来轨迹部分加噪，不对当前状态加噪
    # mean 形状通常为 [B, P, T, 4]
    # std 形状通常为 [B] 或可广播形状
    mean, std = marginal_prob(all_gt[..., 1:, :], t)

    # 将 std reshape 成可以和 [B, P, T, 4] 广播相乘的形状
    # 如果 std 原来是 [B]，这里会变成 [B, 1, 1, 1]
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    # 根据扩散前向过程生成加噪轨迹 xT
    # xT = mean + std * z
    # z 是标准高斯噪声
    xT = mean + std * z

    # 将当前状态拼回到加噪未来轨迹前面
    # 当前状态不加噪，作为条件输入的一部分
    # xT 形状变为 [B, P, 1+T, 4]
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    # 构造输入模型的 merged_inputs
    # 在原始 inputs 基础上额外加入：
    # sampled_trajectories：加噪后的轨迹
    # diffusion_time：当前扩散时间 t
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    # 将输入送入模型
    # decoder_output 中通常包含模型预测的 score 或 x_start
    # decoder_output["score"] 形状通常为 [B, P, 1+T, 4]
    _, decoder_output = model(merged_inputs) # [B, P, 1 + T, 4]

    # 去掉当前状态时间步，只保留未来时间步对应的模型输出
    # score 形状为 [B, P, T, 4]
    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]

    # 如果模型类型是 score
    # 模型预测的是 score，也就是噪声相关的得分函数
    # 对应 denoising score matching 形式的损失：
    # score * std 应该接近 -z
    # 因此损失写成 (score * std + z)^2
    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)

    # 如果模型类型是 x_start
    # 模型直接预测干净的原始未来轨迹 x_start
    # 因此和 all_gt 中归一化后的未来轨迹真值做 MSE
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    # 取出邻居预测损失
    # dpm_loss[:, 1:, :] 表示去掉第 0 个 ego，只保留邻居 agent
    # neighbors_future_valid 用于筛选有效邻居未来点
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    # 如果有效邻居未来点数量大于 0，则计算邻居预测损失均值
    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()

    # 如果没有有效邻居未来点，则邻居预测损失设为 0
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    # 计算 ego planning loss
    # dpm_loss[:, 0, :] 表示第 0 个 agent，也就是 ego 的未来轨迹损失
    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    # 检查 dpm_loss 中是否存在 NaN
    # 如果存在 NaN，则直接触发断言错误，便于调试训练不稳定问题
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    # 返回更新后的 loss 字典和模型 decoder 输出
    return loss, decoder_output