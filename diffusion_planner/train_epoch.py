# 导入 tqdm，用于在训练过程中显示 batch 级别的进度条
from tqdm import tqdm

# 导入 PyTorch，用于张量计算、GPU 同步、反向传播等
import torch

# 从 torch 中导入 nn 模块
# 这里主要用 nn.utils.clip_grad_norm_ 做梯度裁剪
from torch import nn

# 导入状态扰动增强类
# 用于训练阶段对当前自车状态进行扰动，并同步调整未来轨迹标签
from diffusion_planner.utils.data_augmentation import StatePerturbation   

# 导入工具函数，用于统计一个 epoch 内各类 loss 的平均值
from diffusion_planner.utils.train_utils import get_epoch_mean_loss

# 导入分布式训练相关工具
# 包括 get_model、get_rank、reduce_and_average_losses 等
from diffusion_planner.utils import ddp

# 导入扩散模型训练损失函数
from diffusion_planner.loss import diffusion_loss_func


# 单个 epoch 的训练函数
# data_loader：训练数据加载器
# model：扩散规划模型
# optimizer：优化器
# args：训练参数配置
# ema：指数滑动平均模型，用于维护更稳定的模型参数
# aug：数据增强器，可选；如果不为 None，则对输入和标签做状态扰动增强
def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation=None):

    # 用于保存当前 epoch 中每个 batch 的 loss 字典
    # 后续会用 get_epoch_mean_loss 统计平均 loss
    epoch_loss = []

    # 将模型设置为训练模式
    # 会启用 dropout、batchnorm 的训练行为
    model.train()

    # 如果使用 DDP，则先同步 CUDA
    # 确保前面的 GPU 操作完成，避免多进程训练时状态不同步
    if args.ddp:
        torch.cuda.synchronize()

    # 使用 tqdm 包装 data_loader，显示训练进度条
    # desc="Training" 表示进度条名称
    # unit="batch" 表示进度单位是 batch
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:

        # 遍历当前 epoch 的每一个 batch
        for batch in data_epoch:

            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            # 将 batch 中的输入特征整理成模型需要的 inputs 字典
            # 并移动到 args.device 指定的设备上，例如 cuda 或 cpu
            inputs = {
                # 当前自车状态
                # 通常包含 x、y、cos heading、sin heading、vx、vy、ax、ay、steering angle、yaw rate
                'ego_current_state': batch[0].to(args.device),

                # 邻居 agent 历史轨迹
                # 通常形状类似 [B, num_agents, history_len, feature_dim]
                'neighbor_agents_past': batch[2].to(args.device),

                # 周围车道线特征
                'lanes': batch[4].to(args.device),

                # 周围车道线限速值
                'lanes_speed_limit': batch[5].to(args.device),

                # 周围车道线是否有限速
                'lanes_has_speed_limit': batch[6].to(args.device),

                # 导航路线相关 lane 特征
                'route_lanes': batch[7].to(args.device),

                # route lane 的限速值
                'route_lanes_speed_limit': batch[8].to(args.device),

                # route lane 是否有限速
                'route_lanes_has_speed_limit': batch[9].to(args.device),

                # 静态目标特征，例如交通锥、路障、施工区标志等
                'static_objects': batch[10].to(args.device)

            }

            # 自车未来轨迹真值
            # 通常形状类似 [B, future_len, 3]
            # 3 维一般为 x、y、heading
            ego_future = batch[1].to(args.device)

            # 邻居 agent 未来轨迹真值
            # 通常形状类似 [B, num_predicted_agents, future_len, 3]
            neighbors_future = batch[3].to(args.device)

            # Normalize to ego-centric
            # 如果提供了数据增强器，则进行状态扰动增强
            # 增强器会：
            # 1. 扰动当前自车状态；
            # 2. 修正自车未来轨迹；
            # 3. 将自车、邻车、地图等重新转换到扰动后的自车坐标系下
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            # 将 ego_future 中的 heading 角度表示转换成 cos/sin 表示
            # 原始 ego_future[..., :2] 是 x、y
            # ego_future[..., 2] 是 heading
            # 转换后 ego_future 变成 [x, y, cos heading, sin heading]
            # 这样可以避免 heading 在 pi/-pi 附近的不连续问题
            ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            # 构造邻居未来轨迹的 padding mask
            # 如果 neighbors_future 的前三维 [x, y, heading] 全为 0，
            # 则认为该时间步是 padding 或无效数据
            # mask=True 表示该邻居未来轨迹点无效
            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0

            # 同样将邻居未来轨迹中的 heading 转换为 cos/sin
            # 转换后 neighbors_future 变成 [x, y, cos heading, sin heading]
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            # 将原本是 padding 的邻居未来轨迹点重新置 0
            # 因为全 0 的 heading 经过 cos/sin 转换后会变成 [cos0, sin0]=[1,0]
            # 如果不重新置 0，padding 会被误认为真实数据
            neighbors_future[mask] = 0.

            # 对观测输入进行归一化
            # 例如 neighbor_agents_past、lanes、route_lanes、static_objects 等
            # padding 位置会在 normalizer 内部保持为 0
            inputs = args.observation_normalizer(inputs)
                  
            # call the mdoel
            # 清空优化器中的历史梯度
            optimizer.zero_grad()

            # 初始化当前 batch 的 loss 字典
            loss = {}

            # 调用扩散模型损失函数
            # ddp.get_model(model, args.ddp)：
            # 如果使用 DDP，则取 model.module；
            # 如果不用 DDP，则直接取 model。
            #
            # sde.marginal_prob：
            # 用于根据扩散时间 t 计算前向加噪分布的 mean 和 std。
            #
            # (ego_future, neighbors_future, mask)：
            # 传入自车未来真值、邻车未来真值和邻车未来 mask。
            loss, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type
            )

            # 总 loss 由两部分组成：
            # 1. neighbor_prediction_loss：周围 agent 未来预测损失
            # 2. ego_planning_loss：自车规划轨迹损失
            # args.alpha_planning_loss 用于调节自车规划损失的权重
            loss['loss'] = loss['neighbor_prediction_loss'] + args.alpha_planning_loss * loss['ego_planning_loss']

            # 取出总 loss 的 Python 数值，用于 tqdm 显示
            total_loss = loss['loss'].item()

            # loss backward
            # 对总 loss 进行反向传播，计算模型参数梯度
            loss['loss'].backward()

            # 梯度裁剪
            # 将梯度范数限制在 5 以内，防止梯度爆炸导致训练不稳定
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # 根据梯度更新模型参数
            optimizer.step()

            # 更新 EMA 模型参数
            # EMA 会维护模型参数的指数滑动平均版本，通常推理或评估时更稳定
            ema.update(model)

            # 如果使用 DDP，则同步 CUDA
            # 确保当前 step 的 GPU 操作完成
            if args.ddp:
                torch.cuda.synchronize()
            
            # 在 tqdm 进度条右侧显示当前 batch 的 loss
            data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))

            # 保存当前 batch 的 loss 字典
            epoch_loss.append(loss)

    # 统计当前 epoch 中每个 loss 项的平均值
    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    # 如果使用 DDP，则对所有进程上的 loss 进行 reduce 和平均
    # 得到全局平均 loss
    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    # 只在主进程 rank=0 打印训练 loss
    # 避免多进程重复输出日志
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    # 返回当前 epoch 的平均 loss 字典，以及其中的总 loss
    return epoch_mean_loss, epoch_mean_loss['loss']