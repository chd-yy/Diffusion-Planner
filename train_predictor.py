# 导入 os 模块。
# 该模块用于处理与操作系统相关的功能。
# 在当前脚本中，主要使用 os.makedirs() 创建训练日志目录，
# 并使用 os.path.join() 拼接 args.json 的保存路径。
import os
# 导入 PyTorch。
# PyTorch 负责张量计算、GPU 运行、分布式训练同步等功能。
import torch
# 导入 argparse。
# argparse 用于解析命令行参数，使训练超参数可以在启动脚本时灵活设置。
import argparse
# 从 PyTorch 中导入 optim 优化器模块。
# 当前脚本后续使用 optim.AdamW() 创建 AdamW 优化器。
from torch import optim
# 从 timm 库中导入 ModelEma。
# EMA 是 Exponential Moving Average，即指数滑动平均。
# 它会维护一份平滑后的模型参数，通常可用于获得更稳定的验证或推理效果。
from timm.utils import ModelEma
# 导入数据加载器 DataLoader 和分布式采样器 DistributedSampler。
# DataLoader 负责按批次读取数据。
# DistributedSampler 负责在多进程训练时为不同进程分配不同的数据子集。
from torch.utils.data import DataLoader, DistributedSampler
# 导入 PyTorch 的分布式数据并行封装 DDP。
# DDP 是 DistributedDataParallel 的缩写。
# 它会让多个训练进程分别在不同 GPU 上计算梯度，并在反向传播时同步梯度。
from torch.nn.parallel import DistributedDataParallel as DDP

# 导入 Diffusion Planner 的主体模型。
# 该模型通常由场景编码器和扩散解码器组成，用于生成未来规划轨迹。
from diffusion_planner.model.diffusion_planner import Diffusion_Planner

# 导入训练辅助函数：
# set_seed()：设置随机种子；
# save_model()：保存训练检查点；
# resume_model()：从检查点恢复训练状态。
from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
# 导入归一化器：
# ObservationNormalizer 用于归一化场景观测特征；
# StateNormalizer 用于归一化轨迹或状态相关数据。
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
# 导入带有 warm-up 的余弦退火学习率调度器。
# 它会先逐步提高学习率，再按照余弦规律调整学习率。
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
# 导入训练日志记录器，并在当前文件中重命名为 Logger。
# 从命名可以看出，该记录器用于 TensorBoard 日志，也可能封装了其他日志平台。
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
# 导入状态扰动数据增强模块。
# 该模块用于对训练样本进行一定概率的扰动，以增强模型泛化能力。
from diffusion_planner.utils.data_augmentation import StatePerturbation
# 导入训练数据集类。
# DiffusionPlannerData 负责读取并组织 Diffusion Planner 训练样本。
from diffusion_planner.utils.dataset import DiffusionPlannerData
# 导入项目内部封装的 DDP 工具。
# 该模块负责初始化分布式环境、查询 world size，以及从 DDP 包装中取出原始模型。
from diffusion_planner.utils import ddp

# 导入单个 epoch 的训练函数。
# train_epoch() 会遍历训练集，执行前向传播、损失计算、反向传播和参数更新。
from diffusion_planner.train_epoch import train_epoch

# 将命令行中的字符串转换为布尔值。
#
# argparse 默认不会自动把字符串 "true" 或 "false" 转成 bool。
# 这个辅助函数允许用户通过多种常见写法指定布尔参数。
def boolean(v):
    # 如果传入值本身已经是 bool，则直接返回，不再做字符串处理。
    if isinstance(v, bool):
        return v
    # 将字符串转换为小写，并判断它是否属于常见的“真值”写法。
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # 如果字符串属于常见的“假值”写法，则返回 False。
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # 如果输入既不是可识别的真值，也不是可识别的假值，则报错。
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 解析训练脚本的命令行参数。
#
# 该函数集中定义：
# 1. 数据路径与数据形状；
# 2. DataLoader 参数；
# 3. 训练超参数；
# 4. 模型结构参数；
# 5. 分布式训练参数；
# 6. 归一化器。
def get_args():
    # Arguments
    # 创建命令行参数解析器。
    # description 会在执行 python xxx.py --help 时显示。
    parser = argparse.ArgumentParser(description='Training')
    # 实验名称。
    # 它会用于日志打印和训练结果目录命名。
    parser.add_argument('--name', type=str, help='log name (default: "diffusion-planner-training")', default="diffusion-planner-training")
    # 模型检查点和日志的基础保存目录。
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data
    # 训练数据路径。
    parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    # 训练数据列表文件路径。
    # 具体格式需要结合 DiffusionPlannerData 的实现确认。
    parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)

    # future_len：未来轨迹包含的时间点数量。
    # 默认值为 80，表示模型需要输出或监督 80 个未来时刻。
    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    # time_len：历史序列包含的时间点数量。
    # 默认值为 21。
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    # agent_state_dim：每个动态交通参与者在单个历史时刻的状态特征维度。
    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    # agent_num：场景中最多保留的动态交通参与者数量。
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    # static_objects_state_dim：每个静态物体的状态特征维度。
    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    # static_objects_num：场景中最多保留的静态物体数量。
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    # lane_len：每条车道中心线或地图折线保留的采样点数量。
    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    # lane_state_dim：每个车道采样点的特征维度。
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    # lane_num：场景中最多保留的车道数量。
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    # route_len：导航路线中每条路线车道保留的采样点数量。
    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    # route_state_dim：导航路线采样点的状态特征维度。
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    # route_num：最多保留的导航路线车道数量。
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # DataLoader parameters
    # augment_prob：执行状态扰动数据增强的概率。
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    # normalization_file_path：归一化统计量文件路径。
    # 该 JSON 文件通常保存各类特征的均值、标准差或其他缩放参数。
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    # 是否启用数据增强。
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    # DataLoader 使用的工作进程数量。
    parser.add_argument('--num_workers', default=4, type=int)
    # --pin-mem 用于启用 DataLoader 的 pin_memory。
    # 固定页内存有时可以提高 CPU 到 GPU 的数据传输效率。
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # --no-pin-mem 用于显式关闭 pin_memory。
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    # 默认启用 pin_memory。
    parser.set_defaults(pin_mem=True)
    
    # Training
    # 设置随机种子，以提高实验的可复现性。
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    # 训练总 epoch 数。
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=500)
    # save_utd：模型保存间隔。
    # 默认每训练 20 个 epoch 保存一次检查点。
    parser.add_argument('--save_utd', type=int, help='save frequency', default=20)
    # 全局 batch size。
    # 在 DDP 场景下，每个进程实际使用的 batch size 会在后面除以 world size。
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=2048)
    # 初始学习率。
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    # warm-up 持续的 epoch 数。
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    # Encoder 中使用的 drop path 比例。
    # drop path 是一种随机深度正则化方法。
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    # Decoder 中使用的 drop path 比例。
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    # 规划损失的权重系数。
    # 该参数会在训练损失计算中决定 planning loss 的相对重要程度。
    parser.add_argument('--alpha_planning_loss', type=float, help='coefficient of planning loss (default: 1.0)', default=1.0)

    # 运行设备。
    # 默认使用 CUDA。
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    # 是否维护 EMA 模型参数。
    parser.add_argument('--use_ema', default=True, type=boolean)

    # Model
    # Encoder 的网络深度。
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    # Decoder 的网络深度。
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    # 多头注意力中的注意力头数量。
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    # 隐藏特征维度。
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    # 扩散模型的预测目标类型。
    # score：预测 score function；
    # x_start：预测干净轨迹 x_0。
    # 当前默认值为 x_start。
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    # 需要联合预测的周围车辆数量。
    # 该参数与训练数据组织方式和 Decoder 输出维度有关。
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=10)
    # 用于恢复训练的检查点路径。
    # 默认值为 None，表示从头开始训练。
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default=None)

    # 是否启用 Weights & Biases 日志平台。
    parser.add_argument('--use_wandb', default=False, type=boolean)
    # 实验备注信息。
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    # 是否启用分布式训练。
    parser.add_argument('--ddp', default=True, type=boolean, help='use ddp or not')
    # 分布式训练通信端口。
    parser.add_argument('--port', default='22323', type=str, help='port')

    # 解析命令行输入，生成 args 对象。
    # 后续可以通过 args.xxx 的形式访问各项参数。
    args = parser.parse_args()

    # 根据 normalization.json 构造状态归一化器，并保存到 args 中。
    # 这样模型、数据集或训练函数都可以通过 args.state_normalizer 使用它。
    args.state_normalizer = StateNormalizer.from_json(args)
    # 根据 normalization.json 构造观测归一化器。
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    # 返回包含所有训练配置和归一化器的参数对象。
    return args

# 执行完整的模型训练流程。
#
# 该函数负责：
# 1. 初始化 DDP；
# 2. 创建日志目录并保存参数；
# 3. 设置随机种子；
# 4. 构造数据集与 DataLoader；
# 5. 创建 Diffusion Planner；
# 6. 创建 EMA 模型、优化器和学习率调度器；
# 7. 根据需要恢复训练；
# 8. 循环调用 train_epoch()。
def model_training(args):

    # init ddp
    # 初始化分布式训练环境。
    #
    # global_rank：
    #   当前进程在所有训练进程中的全局编号。
    #
    # rank：
    #   当前进程在本机设备中的编号，通常可用于选择对应 GPU。
    #
    # 第三个返回值在当前脚本中没有使用，因此用下划线接收。
    #
    # 注意：这里传入的是固定值 True，而不是 args.ddp。
    # 这是原始代码的写法，本版本仅添加注释，没有修改逻辑。
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    # 只让 global_rank 为 0 的主进程执行日志目录创建和参数保存。
    # 这样可以避免多个进程同时写入同一个文件。
    if global_rank == 0:
        # Logging
        # 打印当前实验名称。
        print("------------- {} -------------".format(args.name))
        # 打印全局 batch size。
        print("Batch size: {}".format(args.batch_size))
        # 打印学习率。
        print("Learning rate: {}".format(args.learning_rate))
        # 打印运行设备。
        print("Use device: {}".format(args.device))

        # 如果指定了恢复训练路径，则沿用该路径作为 save_path。
        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            # 延迟导入 datetime，仅在需要创建新实验目录时使用。
            from datetime import datetime
            # 获取当前系统时间。
            time = datetime.now()
            # 将当前时间格式化为字符串，用于区分不同训练任务。
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            # 按照“保存根目录 / training_log / 实验名称 / 时间戳”的形式创建路径。
            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            # 创建目录。
            # exist_ok=True 表示目录已经存在时不会抛出异常。
            os.makedirs(save_path, exist_ok=True)

        # Save args
        # vars(args) 将 argparse.Namespace 转换为普通字典。
        args_dict = vars(args)
        # 将归一化器对象转换为可写入 JSON 的字典。
        # 其他普通参数保持原值不变。
        args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

        # 延迟导入 mmengine 的 dump()。
        # dump() 用于将配置字典写入文件。
        from mmengine.fileio import dump
        # 将本次训练的完整配置保存为 args.json。
        # 这样可以在复现实验时查看当时使用的超参数。
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    # 非主进程不创建日志目录。
    else:
        # 将非主进程的 save_path 设为 None。
        save_path = None

    # set seed
    # 为当前进程设置随机种子。
    #
    # 在基础种子上加 global_rank，可以使不同训练进程拥有不同的随机序列，
    # 避免所有 GPU 完全重复地进行相同随机操作。
    set_seed(args.seed + global_rank)

    # training parameters
    # 读取训练总 epoch 数，保存为局部变量。
    train_epochs = args.train_epochs
    # 读取全局 batch size。
    batch_size = args.batch_size
    
    # set up data loaders
    # 根据 use_data_augment 决定是否创建状态扰动器。
    #
    # 如果启用数据增强，则构造 StatePerturbation；
    # 否则将 aug 设为 None。
    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    # 构造训练数据集。
    #
    # 参数依次包括：
    # 1. 训练数据路径；
    # 2. 训练数据列表；
    # 3. 最多保留的动态交通参与者数量；
    # 4. 需要预测的周围车辆数量；
    # 5. 未来轨迹长度。
    train_set = DiffusionPlannerData(args.train_set, args.train_set_list, args.agent_num, args.predicted_neighbor_num, args.future_len)
    # 创建分布式采样器。
    #
    # world size 表示总训练进程数。
    # 每个进程根据自己的 global_rank 读取不同的数据切片。
    # shuffle=True 表示每轮训练时打乱数据顺序。
    train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    # 创建 DataLoader。
    #
    # sampler=train_sampler：
    #   使用分布式采样器控制每个进程读取的数据。
    #
    # batch_size=batch_size//ddp.get_world_size()：
    #   将全局 batch size 均分给所有训练进程。
    #
    # pin_memory=args.pin_mem：
    #   控制是否使用固定页内存。
    #
    # drop_last=True：
    #   丢弃最后不足一个完整 batch 的样本。
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
   
    # 只让主进程打印训练集大小。
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    # 如果启用了 DDP，则执行进程同步屏障。
    # 所有进程都运行到这里以后，程序才会继续向下执行。
    if args.ddp:
        torch.distributed.barrier()

    # set up model
    # 根据 args 创建 Diffusion Planner 模型。
    diffusion_planner = Diffusion_Planner(args)
    # 将模型移动到指定设备。
    #
    # 当 args.device == 'cuda' 时，使用 rank 选择当前进程对应的 GPU；
    # 否则将模型移动到 args.device 指定的设备。
    diffusion_planner = diffusion_planner.to(rank if args.device == 'cuda' else args.device)

    # 如果启用了 DDP，则使用 DistributedDataParallel 包装模型。
    if args.ddp:
        # device_ids=[rank] 指定当前进程绑定的 GPU。
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])

    # 如果启用了 EMA，则创建 EMA 模型。
    #
    # 注意：后续 train_epoch()、resume_model() 和 save_model() 都会使用 model_ema。
    # 如果将 --use_ema 设为 False，需要检查项目其他位置是否已经兼容这种情况。
    # 这是对原始代码行为的提醒，本版本没有修改任何逻辑。
    if args.use_ema:
        # 创建 EMA 参数容器。
        model_ema = ModelEma(
            diffusion_planner,
            # decay 越接近 1，EMA 参数变化越平滑。
            decay=0.999,
            # 指定 EMA 参数存放的设备。
            device=args.device,
        )
    
    # 只让主进程打印模型总参数量。
    if global_rank == 0:
        # ddp.get_model() 用于在必要时取出 DDP 包装内部的原始模型。
        # p.numel() 返回单个参数张量中的元素数量。
        # 对所有参数求和即可得到模型参数总量。
        print("Model Params: {}".format(sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())))

    # optimizer
    # 构造优化器参数组。
    # 当前脚本将模型全部可训练参数放入同一个参数组，并设置统一学习率。
    params = [{'params': ddp.get_model(diffusion_planner, args.ddp).parameters(), 'lr': args.learning_rate}]

    # 创建 AdamW 优化器。
    # AdamW 在 Adam 的基础上使用解耦的权重衰减策略。
    optimizer = optim.AdamW(params)
    # 创建学习率调度器。
    # 调度器会根据总 epoch 数和 warm-up epoch 数动态调整学习率。
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    # 如果提供了检查点路径，则恢复模型和训练状态。
    if args.resume_model_path is not None:
        # 打印恢复训练所使用的检查点路径。
        print(f"Model loaded from {args.resume_model_path}")
        # 恢复：
        # 1. 模型参数；
        # 2. 优化器状态；
        # 3. 学习率调度器状态；
        # 4. 初始 epoch；
        # 5. wandb 日志 ID；
        # 6. EMA 参数。
        #
        # 这样可以从中断位置继续训练，而不是重新开始。
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device)
    # 如果没有提供检查点，则从第 0 个 epoch 开始训练。
    else:
        # 初始 epoch 设为 0。
        init_epoch = 0
        # 没有需要恢复的 wandb 日志 ID。
        wandb_id = None

    # logger
    # 创建训练日志记录器。
    #
    # 参数包括实验名称、备注、完整配置、恢复日志 ID、保存目录和进程编号。
    # 具体记录到哪些平台，需要结合 TensorBoardLogger 的内部实现确认。
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    # 在正式开始 epoch 循环前再次同步所有 DDP 进程。
    if args.ddp:
        torch.distributed.barrier()

    # begin training
    # 开始训练循环。
    # range(init_epoch, train_epochs) 支持从检查点保存的 epoch 继续训练。
    for epoch in range(init_epoch, train_epochs):
        # 只让主进程打印训练进度。
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        # 执行一个 epoch 的训练。
        #
        # train_epoch() 通常会完成：
        # 1. 遍历训练 DataLoader；
        # 2. 将 batch 放入模型；
        # 3. 计算训练损失；
        # 4. 执行反向传播；
        # 5. 更新模型参数；
        # 6. 更新 EMA 参数。
        #
        # 返回值：
        # train_loss：通常是包含不同损失项的字典；
        # train_total_loss：通常是该 epoch 的总损失统计量。
        train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args, model_ema, aug)
        


        # 仅由主进程记录日志和保存模型，避免多个进程重复写文件。
        if global_rank == 0:
            # 从优化器中读取当前学习率。
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            # 将各项训练损失写入日志。
            # 字典推导式会在每个损失名称前增加 train_loss/ 前缀。
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            # 将当前学习率写入日志。
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)

            # 每隔 save_utd 个 epoch 保存一次模型。
            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                # 保存模型检查点。
                #
                # 保存内容包括模型、优化器、调度器、当前 epoch、
                # 总损失、日志 ID 和 EMA 模型参数。
                save_model(diffusion_planner, optimizer, scheduler, save_path, epoch, train_total_loss, wandb_logger.id, model_ema.ema)
                # 打印模型保存路径。
                print(f"Model saved in {save_path}\n")

        # 在每个 epoch 结束后更新学习率。
        scheduler.step()
        # 通知 DistributedSampler 下一轮 epoch 编号。
        # 这样采样器可以在下一轮使用新的随机顺序打乱数据。
        train_sampler.set_epoch(epoch + 1)

# 只有直接执行当前 Python 文件时，下面的代码才会运行。
#
# 如果该文件被其他模块 import，则不会自动开始训练。
if __name__ == "__main__":

    # 解析命令行参数。
    args = get_args()
    
    # Run
    # 启动完整训练流程。
    model_training(args)
