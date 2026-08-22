# 导入 tqdm，用于显示当前 epoch 的 batch 级训练进度。
from tqdm import tqdm
# 导入 PyTorch，用于张量计算、设备同步和反向传播。
import torch
# nn.utils.clip_grad_norm_ 用于执行梯度裁剪。
from torch import nn

# 【HDP 与原 Diffusion-Planner 的区别：包命名空间】训练流程复用相同的数据增强、
# epoch loss 汇总和 DDP 工具接口，但实现从 diffusion_planner.* 切换到 hdp_nuplan.*。
# StatePerturbation 对当前自车状态施加扰动，并同步调整未来轨迹标签和场景坐标。
from hdp_nuplan.utils.data_augmentation import StatePerturbation   
# get_epoch_mean_loss 汇总一个 epoch 内各 batch 的平均损失。
from hdp_nuplan.utils.train_utils import get_epoch_mean_loss
# DDP 工具负责获取原始模型、进程 rank，以及跨进程汇总 loss。
from hdp_nuplan.utils import ddp
# diffusion_loss_func 计算 HDP 的自车扩散损失和积分轨迹混合损失。
from hdp_nuplan.loss import diffusion_loss_func


# 完成一个 epoch 的训练：整理 batch、计算 loss、反向传播、更新参数和 EMA，并汇总指标。
# data_loader：训练数据加载器；model：HDP 模型；optimizer：优化器；args：训练配置；
# ema：模型参数的指数滑动平均副本；aug：可选的状态扰动增强器。
def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation=None):
    # 保存当前 epoch 每个 batch 的标量 loss，最后统一计算各项平均值。
    # 不能直接保存带梯度的 loss 张量，否则其反向计算图会被保留到 epoch 结束；
    # 在完整 mini-train 的 38,350 个 batch 上会造成 CPU RAM 持续增长并触发 OOM。
    epoch_loss = []

    # 切换到训练模式，启用 Dropout、DropPath 等训练阶段行为。
    model.train()

    # 使用 DDP 时先等待当前进程已经提交到 CUDA 的操作完成；这只是设备同步，不是跨 rank barrier。
    if args.ddp:
        torch.cuda.synchronize()

    # tqdm 包装 data_loader：进度条名称为 Training，迭代单位为 batch。
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        # 依次处理当前 epoch 的每个 batch。
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
            # 按数据集约定的 tuple 顺序取出模型观测，并移动到训练设备。
            inputs = {
                # 当前自车状态。
                'ego_current_state': batch[0].to(args.device),

                # 周围动态交通参与者的历史状态。
                'neighbor_agents_past': batch[2].to(args.device),

                # 周围 lane 几何、限速值以及限速是否已知。
                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),

                # 导航 route lane 几何、限速值以及限速是否已知。
                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),

                # 交通锥、路障等静态目标的状态。
                'static_objects': batch[10].to(args.device)

            }

            # 自车未来轨迹真值，形状通常为 [B, future_len, 3]，末维是 x、y、heading。
            ego_future = batch[1].to(args.device)
            # 邻车未来轨迹仍供状态扰动增强和兼容 loss 接口使用；HDP 不对其计算预测损失。
            neighbors_future = batch[3].to(args.device)
            # Normalize to ego-centric
            # 若启用增强，则扰动当前自车状态，并同步变换输入、自车未来和邻车未来轨迹。
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            # 把自车未来轨迹由 [x, y, heading] 转为 [x, y, cos(heading), sin(heading)]，
            # 避免 heading 在 -pi 与 pi 边界处不连续。
            ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            # 邻车未来点的 x、y、heading 全为 0 时视为 padding，mask=True 表示无效。
            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            # 与自车一致，把邻车 heading 转换为 cos/sin 表示。
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            # padding 的 heading=0 经 cos/sin 后会变成 [1, 0]，因此必须重新整体置零。
            neighbors_future[mask] = 0.
            # 对自车、邻车、lane、route lane 和静态目标等模型观测执行归一化。
            inputs = args.observation_normalizer(inputs)
                  
            # call the model
            # 清除上一 batch 残留的参数梯度。
            optimizer.zero_grad()
            # loss 函数会向该字典写入各个损失分量。
            loss = {}

            # 【HDP 与原 Diffusion-Planner 的区别：扩散过程接口】原版只把
            # sde.marginal_prob 传给 loss；HDP 传入完整 sde，因为 loss 内除前向加噪外，
            # 还要在 noise、v、x_start、score 等参数化之间执行 transform。
            # 【HDP 与原 Diffusion-Planner 的区别：监督目标】原版由
            # diffusion_model_type 同时决定模型输出和监督空间；HDP 额外传入
            # diffusion_supervision_type，允许模型输出类型与监督目标类型独立配置。
            # HDP 的 loss 只监督自车运动增量；neighbors_future 和 mask 仍传入，
            # 仅用于保持数据集、数据增强与 loss 调用接口兼容。
            loss, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type,
                args.diffusion_supervision_type,
                detach_window_size=args.planning_detach_window_size,
            )
            
            # 【HDP 与原 Diffusion-Planner 的区别：总损失组成】原版使用
            # neighbor_prediction_loss + alpha_planning_loss * ego_planning_loss，
            # 联合优化邻车预测与自车规划；HDP 不再训练邻车预测，改为自车运动增量的
            # 扩散损失 + planning_hybrid_loss * 积分后位置轨迹损失。
            loss['loss'] = loss['ego_planning_loss'] + args.planning_hybrid_loss * loss['ego_planning_hybrid_loss']

            # 转为 Python 数值，仅用于进度条显示；反向传播仍使用上面的 loss 张量。
            total_loss = loss['loss'].item()

            # loss backward
            # 根据总损失构建反向梯度。
            loss['loss'].backward()
            
            # 【HDP 与原 Diffusion-Planner 的区别：无梯度诊断】原版反向传播后直接进行
            # 梯度裁剪；HDP 可通过 log_unused_parameters 输出当前 batch 未获得梯度的参数，
            # 用于定位 lane 限速条件分支在 DDP 下产生的条件性未使用参数。
            if getattr(args, 'log_unused_parameters', False):
                print("Parameters without gradients:")
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"{name}")

            # 将全模型梯度范数裁剪到 5，降低梯度爆炸导致训练不稳定的风险。
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            # 使用裁剪后的梯度更新可训练参数。
            optimizer.step()

            # 用本轮更新后的模型参数更新 EMA 副本，供后续验证和推理使用。
            ema.update(model)

            # DDP 模式下等待当前进程在该 batch 提交的 CUDA 操作完成后再继续。
            if args.ddp:
                torch.cuda.synchronize()
            
            # 【HDP 与原 Diffusion-Planner 的区别：进度条指标】原版只显示组合后的 loss；
            # HDP 分别显示总损失、自车运动增量扩散损失和积分轨迹混合损失，便于观察
            # 两种自车监督项各自的变化。
            data_epoch.set_postfix(total_loss='{:.4f}'.format(total_loss), velocity_loss='{:.4f}'.format(loss['ego_planning_loss'].item()), hybrid_loss='{:.4f}'.format(loss['ego_planning_hybrid_loss'].item()))
            # 只保存脱离计算图的 Python 标量，训练数值不变，同时避免跨 batch 保留计算图。
            epoch_loss.append(
                {
                    key: value.detach().item() if torch.is_tensor(value) else value
                    for key, value in loss.items()
                }
            )

    # 分别计算当前 epoch 中各个 loss 项的 batch 平均值。
    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    # DDP 模式下汇总所有进程的 loss，再除以 world size 得到全局平均值。
    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    # 只允许主进程输出日志，避免多个 rank 重复打印同一个 epoch 结果。
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    # 返回完整的 epoch 平均 loss 字典，以及其中的总损失。
    return epoch_mean_loss, epoch_mean_loss['loss']
