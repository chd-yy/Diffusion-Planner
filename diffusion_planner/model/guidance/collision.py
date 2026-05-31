# 导入 Python 标准库中的 os 模块。
# os 通常用于文件路径、环境变量和操作系统相关操作。
# 注意：在当前文件中，os 尚未被直接使用，但仍然保留原始导入语句。
import os

# 导入 PyTorch。
# 该文件使用 PyTorch 完成张量运算、自动求导、矩阵变换和卷积操作。
import torch

# 导入 PyTorch 神经网络模块。
# nn 通常用于构建神经网络层。
# 注意：在当前文件中，nn 尚未被直接使用，但仍然保留原始导入语句。
import torch.nn as nn

# 导入 PyTorch 中常用的函数式接口，并命名为 F。
# 当前文件使用了：
# 1. F.pad：在时间序列两端补充数据；
# 2. F.conv1d：使用一维卷积平滑梯度序列。
import torch.nn.functional as F

# 从 nuPlan 中导入车辆参数相关接口。
#
# VehicleParameters：
# 用于描述车辆尺寸、轴距等参数的数据结构。
#
# get_pacifica_parameters：
# 返回 Chrysler Pacifica 车辆模型对应的参数。
#
# 注意：VehicleParameters 在当前文件中尚未被直接使用，
# 但仍然保留原始导入语句。
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters


# 获取自车的长度和宽度。
#
# ego_size 的格式为：
#
# [length, width]
#
# 这里使用 nuPlan 中预定义的 Pacifica 车型参数作为自车尺寸。
ego_size = [get_pacifica_parameters().length, get_pacifica_parameters().width]


# 自车参考点到车辆几何中心附近的纵向偏移距离。
#
# 在轨迹规划中，模型预测的位置可能对应后轴中心等车辆参考点，
# 而碰撞检测通常需要使用车辆包围盒的几何中心。
#
# 后续代码会沿着车辆航向方向增加该偏移量。
COG_TO_REAR = 1.67

# 碰撞距离惩罚的裁剪尺度。
#
# 当两车矩形间距小于该数值时，
# 碰撞引导函数会产生非零惩罚。
#
# 当前值为 1.0，表示在包围盒外侧 1 米以内开始产生惩罚。
CLIP_DISTANCE = 1.0

# 碰撞检测包围盒的膨胀量。
#
# 后续代码会在车辆长度和宽度上都加上该数值，
# 使碰撞检测更加保守。
INFLATION = 1.0


# 批量计算两组矩形之间的有符号距离。
#
# 该函数使用了与分离轴定理类似的思想：
# 1. 提取两个矩形的边方向；
# 2. 将矩形顶点投影到候选轴上；
# 3. 判断每个候选轴上的投影区间是否存在间隔；
# 4. 根据是否存在间隔，返回正距离或负距离。
#
# 返回值的符号含义：
# 1. distance > 0：两个矩形分离，数值表示候选轴上的最小间隔；
# 2. distance < 0：两个矩形重叠，绝对值反映穿透程度；
# 3. distance = 0：两个矩形边界接触。
#
# 注意：
# 当矩形沿对角线方向分离时，该函数返回的是候选轴上的最小间隔，
# 不一定等于两个矩形之间严格意义上的欧氏最短距离。
def batch_signed_distance_rect(rect1, rect2):
    '''
    rect1: [B, 4, 2]
    rect2: [B, 4, 2]
    
    return [B] (signed distance between two rectangles)
    '''

    # 读取批量大小 B。
    #
    # rect1 的形状为：
    #
    # [B, 4, 2]
    #
    # 其中：
    # 1. B 表示批量中的矩形数量；
    # 2. 4 表示每个矩形的四个顶点；
    # 3. 2 表示每个顶点的二维坐标 (x, y)。
    B, _, _ = rect1.shape

    # 为每一对矩形构造四个候选投影轴。
    #
    # 每个矩形只提取两条相邻边：
    # 1. 顶点 0 指向顶点 1 的边；
    # 2. 顶点 1 指向顶点 2 的边。
    #
    # 对于矩形而言，两条相邻边互相垂直，
    # 已经能够表示该矩形的两个局部坐标轴。
    #
    # 将 rect1 和 rect2 的局部轴合并后，
    # norm_vec 的形状为：
    #
    # [B, 4, 2]
    #
    # 注意：
    # 变量名虽然是 norm_vec，但当前代码实际保存的是边方向向量，
    # 而不是显式计算得到的法向量。
    # 对矩形而言，使用两条互相垂直的边方向同样可以完成投影检测。
    norm_vec = torch.stack([rect1[:, 0] - rect1[:, 1], 
                             rect1[:, 1] - rect1[:, 2], 
                             rect2[:, 0] - rect2[:, 1], 
                             rect2[:, 1] - rect2[:, 2]], dim=1) # [B, 4, 2]

    # 对每个候选轴进行归一化。
    #
    # torch.norm(..., dim=2, keepdim=True) 计算每个二维向量的长度。
    #
    # 归一化后，每个候选轴的长度为 1。
    # 这样，投影区间差值可以直接反映空间距离尺度。
    #
    # 注意：
    # 如果输入矩形存在长度或宽度为 0 的退化情况，
    # 某个边向量的模长可能为 0，从而产生除零风险。
    # 当前代码未额外处理这种边界情况。
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)
    
    # 将 rect1 的四个顶点投影到四个候选轴上。
    #
    # norm_vec 的形状为：
    #
    # [B, 4, 2]
    #
    # rect1 的形状为：
    #
    # [B, 4, 2]
    #
    # einsum 表达式：
    #
    # 'bij,bkj->bik'
    #
    # 表示对最后一个坐标维度 j 求和。
    #
    # 对于每一个批次 b：
    # 1. i 表示第 i 个候选轴；
    # 2. k 表示 rect1 的第 k 个顶点；
    # 3. 输出值表示对应顶点在对应轴上的标量投影。
    #
    # 输出 proj1 的形状为：
    #
    # [B, 4, 4]
    proj1 = torch.einsum('bij,bkj->bik', norm_vec, rect1) # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]

    # 对每一个候选轴，取 rect1 四个顶点投影中的最小值和最大值。
    #
    # 每个轴上的矩形投影可以表示为一个一维区间：
    #
    # [proj1_min, proj1_max]
    #
    # 两个输出张量的形状均为：
    #
    # [B, 4]
    proj1_min, proj1_max = proj1.min(dim=2)[0], proj1.max(dim=2)[0] # [B, 4] [B, 4]
    
    # 将 rect2 的四个顶点投影到相同的四个候选轴上。
    #
    # 输出 proj2 的形状同样为：
    #
    # [B, 4, 4]
    proj2 = torch.einsum('bij,bkj->bik', norm_vec, rect2) # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]

    # 对每一个候选轴，取 rect2 投影区间的最小值和最大值。
    #
    # 两个输出张量的形状均为：
    #
    # [B, 4]
    proj2_min, proj2_max = proj2.min(dim=2)[0], proj2.max(dim=2)[0] # [B, 4] [B, 4]
    
    # 计算两个投影区间在不同方向上的相对距离。
    #
    # 对于每一个候选轴，计算：
    #
    # proj1_min - proj2_max
    #
    # 以及：
    #
    # proj2_min - proj1_max
    #
    # 如果两个投影区间重叠，
    # 上述两个值均小于 0。
    #
    # 如果两个投影区间分离，
    # 至少有一个值大于或等于 0。
    #
    # 将两个方向的结果拼接后，overlap 的形状为：
    #
    # [B, 8]
    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max], dim=1) # [B, 8]
    
    # 保留非负间隔，将负值替换为较大的占位值 1e5。
    #
    # 当矩形分离时，后续需要找到最小的正间隔。
    #
    # 将负值替换成较大数值后，
    # 使用 min(...) 就可以忽略已经重叠的方向。
    positive_distance = torch.where(overlap < 0, 1e5, overlap)
    
    # 判断两个矩形是否在所有候选轴上都存在投影重叠。
    #
    # 如果 overlap 的八个值全部小于 0，
    # 则矩形在所有候选轴上的投影均重叠，
    # 因而两个矩形发生重叠。
    #
    # is_overlap 的形状为：
    #
    # [B]
    is_overlap = (overlap < 0).all(dim=1)

    # 根据矩形是否重叠，返回不同形式的有符号距离。
    #
    # 当矩形重叠时：
    #
    # overlap.max(dim=1).values
    #
    # 返回最接近 0 的负值，
    # 表示从所有候选轴中选取最小穿透程度。
    #
    # 当矩形分离时：
    #
    # positive_distance.min(dim=1).values
    #
    # 返回所有非负间隔中的最小值。
    #
    # distance 的形状为：
    #
    # [B]
    distance = torch.where(is_overlap, overlap.max(dim=1).values, positive_distance.min(dim=1).values)   
    
    # 返回每一对矩形的有符号距离。
    return distance


# 将由中心点、航向和尺寸描述的矩形转换为四个顶点。
#
# 输入 rect 的每一行包含：
#
# (x, y, cos_h, sin_h, l, w)
#
# 其中：
# 1. x, y：矩形中心坐标；
# 2. cos_h, sin_h：航向角的余弦值和正弦值；
# 3. l：矩形长度；
# 4. w：矩形宽度。
#
# 输出为：
#
# [B, 4, 2]
#
# 表示每一个矩形的四个二维顶点。
def center_rect_to_points(rect):
    '''
    rect: [B, 6] (x, y, cos_h, sin_h, l, w)
    
    return [B, 4, 2] (4 points of the rectangle)
    '''
    
    # 获取矩形批量大小 B。
    #
    # rect 的形状为：
    #
    # [B, 6]
    B, _ = rect.shape

    # 拆分矩形参数。
    #
    # xy：
    # 矩形中心坐标，形状为 [B, 2]。
    #
    # cos_h：
    # 航向角余弦值，形状为 [B]。
    #
    # sin_h：
    # 航向角正弦值，形状为 [B]。
    #
    # lw：
    # 矩形长度和宽度，形状为 [B, 2]。
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]
    
    # 根据航向角构造二维旋转矩阵。
    #
    # 每一个矩形对应一个旋转矩阵：
    #
    # [[ cos_h, -sin_h ],
    #  [ sin_h,  cos_h ]]
    #
    # rot 的形状为：
    #
    # [B, 2, 2]
    #
    # 该矩阵用于将车辆局部坐标系中的顶点偏移量
    # 旋转到全局坐标系。
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(-1, 2, 2) # [B, 2, 2]

    # 根据长度和宽度，生成矩形四个顶点在局部坐标系中的偏移量。
    #
    # 顶点符号组合为：
    #
    # [[ 1,  1],
    #  [-1,  1],
    #  [-1, -1],
    #  [ 1, -1]]
    #
    # 除以 2 后再分别乘以长度和宽度，
    # 可以得到四个顶点相对于矩形中心的局部偏移：
    #
    # [[ l / 2,  w / 2],
    #  [-l / 2,  w / 2],
    #  [-l / 2, -w / 2],
    #  [ l / 2, -w / 2]]
    #
    # 输出 lw 的形状为：
    #
    # [B, 4, 2]
    lw = torch.einsum('bj,ij->bij', lw, torch.tensor([[1., 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device) / 2) # [B, 2] * [4, 2] -> [B, 4, 2]

    # 使用旋转矩阵，将局部坐标系中的四个顶点偏移
    # 转换到全局坐标系。
    #
    # 输入：
    #
    # lw：[B, 4, 2]
    # rot：[B, 2, 2]
    #
    # 输出：
    #
    # lw：[B, 4, 2]
    lw = torch.einsum('bij,bkj->bik', lw, rot) # [B, 4, 2] * [B, 2, 2] -> [B, 4, 2]
    
    # 为旋转后的四个局部偏移量加上矩形中心坐标。
    #
    # xy[:, None, :] 的形状为：
    #
    # [B, 1, 2]
    #
    # PyTorch 会将其广播到：
    #
    # [B, 4, 2]
    #
    # 最终得到矩形四个顶点在全局坐标系中的坐标。
    rect = xy[:, None, :] + lw # [B, 4, 2]
    
    # 返回矩形的四个顶点。
    return rect


# 根据预测轨迹计算碰撞规避引导信号。
#
# 该函数的主要流程为：
# 1. 提取自车轨迹和邻居车辆轨迹；
# 2. 将车辆位置和尺寸转换为矩形包围盒；
# 3. 计算自车与邻居车辆的有符号距离；
# 4. 对过近或重叠的车辆构造碰撞惩罚；
# 5. 使用自动求导计算惩罚对自车位置的梯度；
# 6. 将梯度转换到自车局部坐标系；
# 7. 关闭纵向引导，仅保留经过卷积平滑的横向引导；
# 8. 将平滑后的引导向量转换回全局坐标系；
# 9. 构造一个线性奖励函数并返回。
#
# 返回值可用于扩散模型采样阶段的碰撞规避引导。
def collision_guidance_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """
    x: [B * Pn+1, T + 1, 4]
    t: [B, 1],
    inputs: Dict[str, torch.Tensor]
    """

    # 从 x 中读取四个维度。
    #
    # 根据该行代码，函数实际期望 x 是四维张量：
    #
    # [B, P, T, 4]
    #
    # 其中：
    # 1. B：批量大小；
    # 2. P：参与预测的车辆数量，通常包括自车和邻居车辆；
    # 3. T：时间点数量，通常包括当前时刻；
    # 4. 4：每个轨迹点的状态维度。
    #
    # 状态的四个分量通常为：
    #
    # (x, y, cos_h, sin_h)
    #
    # 注意：
    # 函数文档字符串中记录的 x 形状与此处实际解包方式并不完全一致。
    # 当前注释仅说明原始代码的实际行为，不修改原始实现。
    B, P, T, _ = x.shape

    # 读取邻居车辆当前时刻的掩码。
    #
    # neighbor_current_mask 的形状为：
    #
    # [B, Pn]
    #
    # 其中 Pn 表示邻居车辆数量。
    #
    # 后续代码通过 ~neighbor_current_mask 选择参与碰撞检测的车辆。
    # 因此，原始实现假设掩码值为 False 的位置需要保留。
    neighbor_current_mask = inputs["neighbor_current_mask"] # [B, Pn]
    
    # 将 x 调整为：
    #
    # [B, P, ?, 4]
    #
    # 当前 reshape 通常不会改变实际形状，
    # 但可以显式保证最后一个维度为 4。
    x: torch.Tensor = x.reshape(B, P, -1, 4)

    # 判断当前扩散时间是否位于碰撞引导生效区间。
    #
    # 引导区间为：
    #
    # 0.005 < t < 0.1
    #
    # 注意：
    # 当前代码使用 Python 关键字 and。
    # 如果 t 是包含多个元素的张量，
    # PyTorch 可能无法直接将其转换为单个布尔值。
    #
    # 该写法通常要求：
    # 1. t 是标量；
    # 2. 或者 t 只包含一个元素；
    # 3. 或者上游调用逻辑保证该表达式可以转换为单个布尔值。
    #
    # 当前注释仅说明原始实现，不修改判断表达式。
    mask_diffusion_time = (t < 0.1 and t > 0.005)

    # 根据扩散时间控制梯度是否继续传播。
    #
    # 当 mask_diffusion_time 为 True 时：
    #
    # 保留原始 x，允许梯度继续传播。
    #
    # 当 mask_diffusion_time 为 False 时：
    #
    # 使用 x.detach()，切断梯度传播。
    #
    # torch.where 只改变梯度路径，不改变张量数值。
    x = torch.where(mask_diffusion_time, x, x.detach())
    
    # 重新构造轨迹状态。
    #
    # 前两个分量：
    #
    # x[:, :, :, :2]
    #
    # 表示二维位置 (x, y)，保持原有梯度路径。
    #
    # 后两个分量：
    #
    # x[:, :, :, 2:]
    #
    # 表示航向方向向量，通常为 (cos_h, sin_h)。
    #
    # 代码先使用 detach() 切断航向向量的梯度传播，
    # 再除以其模长，使航向方向归一化为单位向量。
    #
    # 因此：
    # 1. 位置分量仍可参与梯度计算；
    # 2. 航向分量会参与数值计算；
    # 3. 但梯度不会通过航向分量继续传播。
    #
    # 注意：
    # 如果航向向量的模长为 0，
    # 当前实现可能产生除零风险。
    x = torch.cat([x[:, :, :, :2], 
                    x[:, :, :, 2:].detach() / torch.norm(x[:, :, :, 2:].detach(), dim=-1, keepdim=True)
                ], dim=-1) # [B, P + 1, T, 4]
    
    # 提取自车未来轨迹。
    #
    # x[:, :1, 1:, :] 表示：
    # 1. 保留批量维度；
    # 2. 只取第 0 个车辆，即自车；
    # 3. 跳过索引 0 对应的当前时刻；
    # 4. 取未来所有轨迹点；
    # 5. 保留完整状态维度。
    #
    # ego_pred 的形状为：
    #
    # [B, 1, T_future, 4]
    ego_pred = x[:, :1, 1:, :] # [B, 1, T, 4]

    # 提取自车航向向量的两个分量。
    #
    # cos_h 和 sin_h 的形状均为：
    #
    # [B, 1, T_future, 1]
    cos_h, sin_h = ego_pred[..., 2:3], ego_pred[..., 3:4]

    # 将自车预测位置沿着航向方向平移 COG_TO_REAR。
    #
    # 新的位置为：
    #
    # x_new = x_old + cos_h * COG_TO_REAR
    #
    # y_new = y_old + sin_h * COG_TO_REAR
    #
    # 该操作通常用于将轨迹模型使用的车辆参考点
    # 转换到更适合构造车辆包围盒的位置。
    #
    # 航向分量保持不变。
    ego_pred = torch.cat([ego_pred[..., 0:1] + cos_h * COG_TO_REAR, ego_pred[..., 1:2] + sin_h * COG_TO_REAR, ego_pred[..., 2:]], dim=-1)
    
    # 提取所有邻居车辆的未来预测轨迹。
    #
    # x[:, 1:, 1:, :] 表示：
    # 1. 跳过第 0 个车辆，即跳过自车；
    # 2. 跳过索引 0 对应的当前时刻；
    # 3. 保留所有邻居车辆的未来轨迹。
    #
    # neighbors_pred 的形状为：
    #
    # [B, Pn, T_future, 4]
    neighbors_pred = x[:, 1:, 1:, :] # [B, P, T, 4]
    
    # 重新读取邻居预测轨迹的形状。
    #
    # 此处：
    # 1. Pn 表示邻居车辆数量；
    # 2. T 表示未来预测时间步数量。
    #
    # 注意：
    # 这里会覆盖函数开头读取到的变量 T。
    # 函数开头的 T 通常包含当前时刻，
    # 此处的 T 则只表示未来时刻数量。
    B, Pn, T, _ = neighbors_pred.shape

    # 拼接自车未来轨迹和邻居车辆未来轨迹。
    #
    # predictions 的车辆维度顺序为：
    # 1. 第 0 个车辆：自车；
    # 2. 后续车辆：邻居车辆。
    #
    # neighbors_pred.detach() 会切断邻居车辆轨迹的梯度路径。
    #
    # 因此，碰撞引导会调整自车轨迹，
    # 但不会通过该路径反向调整邻居车辆轨迹。
    #
    # predictions 的形状为：
    #
    # [B, Pn + 1, T, 4]
    predictions = torch.cat([ego_pred, neighbors_pred.detach()], dim=1) # [B, P + 1, T, 4]
    
    # 拼接自车尺寸和邻居车辆尺寸。
    #
    # 第一部分：
    #
    # torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(B, 1, 1)
    #
    # 将自车尺寸 [length, width] 复制到每个批次。
    #
    # 输出形状为：
    #
    # [B, 1, 2]
    #
    # 第二部分：
    #
    # inputs["neighbor_agents_past"][:, :Pn, -1, [7, 6]]
    #
    # 从邻居车辆历史状态的最后一个时间点中，
    # 提取索引 7 和索引 6 对应的两个尺寸分量。
    #
    # 根据当前拼接顺序，这两个分量被当作：
    #
    # [length, width]
    #
    # 使用。
    #
    # 输出 lw 的形状为：
    #
    # [B, Pn + 1, 2]
    lw = torch.cat([torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(B, 1, 1),
                    inputs["neighbor_agents_past"][:, :Pn, -1, [7, 6]]], dim=1) # [B, P, 2]
    
    # 为每个车辆的每个预测时间步构造矩形参数。
    #
    # predictions 包含：
    #
    # (x, y, cos_h, sin_h)
    #
    # lw 包含：
    #
    # (length, width)
    #
    # lw.unsqueeze(2) 在时间维度增加一个轴。
    #
    # expand(-1, -1, T, -1) 将尺寸复制到所有未来时间步。
    #
    # + INFLATION 会同时增加车辆长度和车辆宽度，
    # 从而得到更保守的包围盒。
    #
    # bbox 的每个元素包含：
    #
    # (x, y, cos_h, sin_h, inflated_length, inflated_width)
    #
    # bbox 的形状为：
    #
    # [B, Pn + 1, T, 6]
    bbox = torch.cat([
        predictions,
        lw.unsqueeze(2).expand(-1, -1, T, -1) + INFLATION
    ], dim=-1) # [B, P, T, 6]
    
    # 将中心点形式的矩形包围盒转换为四个顶点。
    #
    # 第一步：
    #
    # bbox.reshape(-1, 6)
    #
    # 将批次、车辆和时间维度展平，
    # 使 center_rect_to_points(...) 可以批量处理所有矩形。
    #
    # 第二步：
    #
    # center_rect_to_points(...)
    #
    # 将每个矩形转换为四个顶点。
    #
    # 第三步：
    #
    # reshape(B, Pn + 1, T, 4, 2)
    #
    # 恢复批次、车辆和时间维度。
    #
    # bbox 的最终形状为：
    #
    # [B, Pn + 1, T, 4, 2]
    bbox = center_rect_to_points(bbox.reshape(-1, 6)).reshape(B, Pn + 1, T, 4, 2)
    
    # 为每一个邻居车辆复制一份自车包围盒，
    # 并筛选需要参与碰撞检测的邻居车辆。
    #
    # bbox[:, :1, :, :, :]：
    #
    # 只提取自车包围盒，形状为：
    #
    # [B, 1, T, 4, 2]
    #
    # expand(-1, Pn, -1, -1, -1)：
    #
    # 将自车包围盒沿邻居车辆维度复制，
    # 得到：
    #
    # [B, Pn, T, 4, 2]
    #
    # [~neighbor_current_mask]：
    #
    # 选择掩码值为 False 的邻居车辆。
    #
    # reshape(-1, 4, 2)：
    #
    # 将批次、有效邻居车辆和时间维度展平，
    # 得到若干个需要检查的自车矩形。
    ego_bbox = bbox[:, :1, :, :, :].expand(-1, Pn, -1, -1, -1)[~neighbor_current_mask].reshape(-1, 4, 2)

    # 提取邻居车辆包围盒，
    # 使用相同的掩码筛选参与碰撞检测的邻居车辆，
    # 并将批次、车辆和时间维度展平。
    #
    # neighbor_bbox 的形状为：
    #
    # [N, 4, 2]
    #
    # 其中 N 表示所有有效车辆对在所有未来时间步上的矩形数量。
    neighbor_bbox = bbox[:, 1:, :, :, :][~neighbor_current_mask].reshape(-1, 4, 2)
    
    # 计算自车与各个邻居车辆包围盒之间的有符号距离。
    #
    # distances 的形状为：
    #
    # [N]
    #
    # 其中：
    # 1. distances < 0：包围盒发生重叠；
    # 2. distances = 0：包围盒边界接触；
    # 3. distances > 0：包围盒相互分离。
    distances = batch_signed_distance_rect(ego_bbox, neighbor_bbox)

    # 根据有符号距离构造裁剪后的风险值。
    #
    # 原始表达式为：
    #
    # 1 - distances / CLIP_DISTANCE
    #
    # 再与 0 比较并取较大值。
    #
    # 当 CLIP_DISTANCE = 1.0 时：
    #
    # 1. distances < 0：
    #    clip_distances > 1，表示包围盒已经重叠；
    #
    # 2. 0 <= distances < 1：
    #    0 < clip_distances <= 1，表示包围盒尚未重叠，但距离较近；
    #
    # 3. distances >= 1：
    #    clip_distances = 0，表示包围盒距离足够远，不产生惩罚。
    clip_distances = torch.maximum(1 - distances / CLIP_DISTANCE, torch.tensor(0.0, device=distances.device))
            
    # 根据碰撞风险构造标量奖励。
    #
    # clip_distances 被划分为两组：
    #
    # 第一组：
    #
    # clip_distances > 1
    #
    # 表示包围盒已经发生重叠。
    #
    # 第二组：
    #
    # clip_distances <= 1
    #
    # 表示尚未重叠，或者距离已经足够远。
    #
    # 对两组风险分别求和，
    # 再除以对应的非零风险数量，
    # 相当于分别计算两类风险的平均强度。
    #
    # 分母中的 detach() 会阻止计数逻辑参与梯度传播。
    #
    # 分母增加 1e-5，
    # 用于避免某一组不存在有效元素时出现除零问题。
    #
    # 两类风险相加后使用 exp() 放大惩罚，
    # 最前面的负号使 reward 为负值：
    #
    # 风险越高，reward 越小。
    reward = - (torch.sum(clip_distances[clip_distances > 1]) / (torch.sum((clip_distances[clip_distances > 1].detach() > 0).float()) + 1e-5) +
                torch.sum(clip_distances[clip_distances <= 1]) / (torch.sum((clip_distances[clip_distances <= 1].detach() > 0).float()) + 1e-5)).exp()
    
    # 计算碰撞奖励 reward 对轨迹张量 x 的梯度。
    #
    # torch.autograd.grad(...) 的参数含义：
    #
    # reward.sum()：
    # 将 reward 汇总为标量，便于求梯度。
    #
    # x：
    # 指定需要求导的变量。
    #
    # retain_graph=True：
    # 求导后保留计算图，便于后续继续使用。
    #
    # allow_unused=True：
    # 允许 x 中存在没有参与 reward 计算的部分。
    #
    # [0]：
    # 取返回梯度列表中的第一个张量。
    #
    # [:, 0, :, :2]：
    # 只保留：
    # 1. 第 0 个车辆，即自车；
    # 2. 所有时间点；
    # 3. 前两个维度，即位置 (x, y)。
    #
    # 注意：
    # 当前代码注释写作 [B, T, 2]。
    # 从前面的索引逻辑看，这里通常包含当前时刻和未来时刻，
    # 因而其时间长度通常比 neighbors_pred 的未来时间长度多 1。
    x_aux = torch.autograd.grad(reward.sum(), x, retain_graph=True, allow_unused=True)[0][:, 0, :, :2] # [B, T, 2]
    
    # 将未来时间步数量增加 1。
    #
    # 前面重新赋值后的 T 只表示未来预测时间步数量。
    #
    # x_aux 和 x[:, 0, :, 2:] 通常还包含当前时刻，
    # 因此这里将 T 增加 1，使后续 reshape 的时间长度与 x 对齐。
    T += 1

    # 根据自车每个时间点的航向向量构造坐标变换矩阵。
    #
    # x[:, 0, :, 2:] 的形状为：
    #
    # [B, T, 2]
    #
    # 每个航向向量为：
    #
    # [cos_h, sin_h]
    #
    # 与固定张量进行 einsum 运算后得到：
    #
    # [cos_h, sin_h, -sin_h, cos_h]
    #
    # reshape 后，每个时间点对应矩阵：
    #
    # [[ cos_h,  sin_h],
    #  [-sin_h,  cos_h]]
    #
    # 该矩阵可以将全局坐标系中的二维向量
    # 转换到车辆局部坐标系。
    #
    # x_mat 的形状为：
    #
    # [B, T, 2, 2]
    x_mat = torch.einsum("btd,nd->btn", x[:, 0, :, 2:], torch.tensor([[1., 0], [0, 1], [0, -1], [1, 0]], device=x.device)).reshape(B, T, 2, 2)
    
    # 将全局坐标系中的碰撞梯度 x_aux
    # 转换到自车局部坐标系。
    #
    # 转换后：
    # 1. 第 0 个分量可理解为局部纵向分量；
    # 2. 第 1 个分量可理解为局部横向分量。
    #
    # x_aux 的形状保持为：
    #
    # [B, T, 2]
    x_aux = torch.einsum("btij,btj->bti", x_mat, x_aux)

    # 原始代码中保留的一种可选处理方式。
    #
    # 如果启用该行：
    # 1. 只保留前 5 个时间步的引导信号；
    # 2. 将后续时间步的引导信号置零。
    #
    # 当前该行处于注释状态，不会执行。
    # x_aux = torch.cat([x_aux[:, :5], torch.zeros_like(x_aux[:, 5:])], dim=1)
    
    # 重新构造局部坐标系中的引导信号。
    #
    # 最终 x_aux 的两个分量分别表示：
    # 1. 局部纵向引导；
    # 2. 局部横向引导。
    #
    # 当前实现有意关闭纵向引导，
    # 仅保留经过时间平滑后的横向引导。
    x_aux = torch.stack([  

        # 构造纵向引导分量。
        #
        # x_aux[..., 0] 表示局部纵向梯度。
        #
        # torch.linspace(0, 1, T)：
        # 生成长度为 T 的等间距序列。
        #
        # -torch.linspace(...)：
        # 将其变为从 0 到 -1 的序列。
        #
        # exp()：
        # 得到随时间衰减的权重。
        #
        # unsqueeze(0).repeat(T, 1)：
        # 将权重复制为 T x T 的矩阵。
        #
        # torch.tril(...)：
        # 只保留下三角部分，
        # 表示每一个时间点只累计当前时刻及其之前的影响。
        #
        # torch.einsum("bt,it->bi", ...)：
        # 对时间维度进行加权累积。
        #
        # 最后的 * 0：
        # 将纵向引导完全置零。
        #
        # 因此，虽然前面计算了纵向梯度，
        # 当前实现不会将其用于最终引导。
        torch.einsum("bt,it->bi", x_aux[..., 0], torch.tril((-torch.linspace(0, 1, T, device=x.device)).exp().unsqueeze(0).repeat(T, 1))) * 0,

        # 构造横向引导分量。
        #
        # x_aux[:, None, :, 1]：
        # 取出局部横向梯度，
        # 并增加一个通道维度。
        #
        # 原始形状：
        #
        # [B, T]
        #
        # 增加通道维度后：
        #
        # [B, 1, T]
        #
        # F.pad(..., (10, 10), mode='replicate')：
        # 在时间序列左右两端各补充 10 个元素，
        # 并使用边界值复制模式。
        #
        # F.conv1d(...)：
        # 使用长度为 21 的一维卷积核对横向梯度进行平滑。
        #
        # 卷积核为：
        #
        # exp(-linspace(-2, 2, 21)^2 / 4)
        #
        # 其形状类似高斯核：
        # 1. 中心位置权重较大；
        # 2. 距离中心较远的位置权重较小；
        # 3. 可以降低横向引导在相邻时间点之间的剧烈波动。
        #
        # 注意：
        # 当前卷积核没有显式除以权重总和，
        # 因此它执行的是加权平滑和放大，而不是严格归一化的加权平均。
        F.conv1d(
            F.pad(x_aux[:, None, :, 1], (10, 10), mode='replicate'), 
            torch.ones(1, 1, 21, device=x.device) * \
            (- torch.linspace(-2, 2, 21, device=x.device) ** 2 / 4).exp()
        )[:, 0] * 1.0

    # 将纵向和横向分量重新堆叠到最后一个维度。
    #
    # x_aux 的形状为：
    #
    # [B, T, 2]
    ], dim=2)

    # 将局部坐标系中的引导向量重新转换回全局坐标系。
    #
    # 这里使用 x_mat 的转置形式：
    #
    # "btji"
    #
    # 因而该操作对应前面局部坐标变换的逆变换。
    #
    # 转换后：
    # 1. x_aux[..., 0] 表示全局 x 方向引导；
    # 2. x_aux[..., 1] 表示全局 y 方向引导。
    #
    # 输出形状为：
    #
    # [B, T, 2]
    x_aux = torch.einsum("btji,btj->bti", x_mat, x_aux) # [B, T, 2]
    
    # 根据处理后的引导向量构造线性奖励函数。
    #
    # x_aux.detach()：
    # 将平滑后的引导向量视为常量，
    # 防止梯度继续回传到前面的碰撞距离计算过程。
    #
    # x[:, 0, :, :2]：
    # 取出自车在所有时间点的二维位置。
    #
    # 两者逐元素相乘并在时间维度和坐标维度上求和后，
    # 得到每个批次样本对应的标量奖励。
    #
    # reward 的形状为：
    #
    # [B]
    #
    # 由于 x_aux 已经 detach，
    # 对该 reward 再次求自车位置梯度时，
    # 可以直接得到经过处理后的引导方向。
    reward = torch.sum(x_aux.detach() * x[:, 0, :, :2], dim=(1, 2))
    
    # 将奖励乘以 3.0 后返回。
    #
    # 该系数用于调整碰撞规避引导信号的整体强度。
    return 3.0 * reward