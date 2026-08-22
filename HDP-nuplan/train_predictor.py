# 当前文件与仓库根目录 train_predictor.py 都是完整的监督训练入口。
# 【HDP 与原 Diffusion-Planner 的区别】凡是两者执行逻辑不同的位置，均使用这一统一前缀标注；
# 其余无前缀中文注释来自原训练入口中仍适用于 HDP 的说明。
# 导入 os：创建训练日志目录，并拼接 args.json、warm-start 报告等输出路径。
import os
# 导入 PyTorch：负责张量计算、设备迁移、分布式同步和 checkpoint 加载。
import torch
# argparse 用于把命令行中的数据路径、模型结构和训练超参数解析为 args。
import argparse
# optim 提供后续使用的 AdamW 优化器。
from torch import optim
# ModelEma 维护模型参数的指数滑动平均副本，保存后用于更稳定的验证和推理。
from timm.utils import ModelEma
# DataLoader 按 batch 读取数据；DistributedSampler 为不同 DDP 进程划分数据。
from torch.utils.data import DataLoader, DistributedSampler
# DDP 在多个训练进程间同步反向传播得到的梯度。
from torch.nn.parallel import DistributedDataParallel as DDP

# 【HDP 与原 Diffusion-Planner 的区别：模型类型】原入口导入 Diffusion_Planner；
# 当前入口改为 HDP 独立的 Hyper_Diffusion_Planner。
from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner

# 【HDP 与原 Diffusion-Planner 的区别：训练工具】公共的随机种子、保存和恢复工具切换到
# hdp_nuplan 命名空间；另外新增只迁移兼容 encoder 的 warm-start 和冻结/解冻工具。
from hdp_nuplan.utils.train_utils import (
    load_encoder_warm_start,
    resume_model,
    save_model,
    set_encoder_trainable,
    set_seed,
)
# 【HDP 与原 Diffusion-Planner 的区别：包命名空间】归一化、调度器、日志、增强、数据集和 DDP
# 工具的接口保持一致，但导入路径由 diffusion_planner.* 切换为 hdp_nuplan.*。
# ObservationNormalizer 归一化场景输入；StateNormalizer 归一化扩散轨迹状态。
from hdp_nuplan.utils.normalizer import ObservationNormalizer, StateNormalizer
# 带 warm-up 的余弦退火学习率调度器。
from hdp_nuplan.utils.lr_schedule import CosineAnnealingWarmUpRestarts
# 训练日志记录器，统一记录 loss、学习率和实验元数据。
from hdp_nuplan.utils.tb_log import TensorBoardLogger as Logger
# StatePerturbation 对训练场景和对应标签做一致的状态扰动增强。
from hdp_nuplan.utils.data_augmentation import StatePerturbation
# DiffusionPlannerData 从预处理 NPZ 缓存及 manifest 中读取监督训练样本。
from hdp_nuplan.utils.dataset import DiffusionPlannerData
# 项目封装的 DDP 工具，负责初始化、rank/world-size 查询和解包原始模型。
from hdp_nuplan.utils import ddp

# 【HDP 与原 Diffusion-Planner 的区别：单轮训练实现】原入口导入
# diffusion_planner.train_epoch；当前入口导入 hdp_nuplan.train_epoch。两者的 batch 级损失
# 差异位于各自 train_epoch.py 中，而不是当前入口文件中。
from hdp_nuplan.train_epoch import train_epoch

# argparse 不会把字符串 "true"/"false" 自动转换为 bool，此函数兼容常见真假写法。
def boolean(v):
    # 输入本身已经是 bool 时直接返回。
    if isinstance(v, bool):
        return v
    # 接受 yes、true、t、y、1 等真值写法。
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # 接受 no、false、f、n、0 等假值写法。
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # 无法识别时让 argparse 给出明确错误，而不是静默使用错误配置。
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 集中定义数据形状、DataLoader、训练、模型、checkpoint 和 DDP 参数。
def get_args():
    # Arguments
    # description 会显示在 `python train_predictor.py --help` 的帮助信息中。
    parser = argparse.ArgumentParser(description='Training')
    # 实验名称用于日志打印及 `training_log/<name>/<timestamp>` 目录命名。
    # 【HDP 与原 Diffusion-Planner 的区别：默认实验名】默认值从 diffusion-planner-training
    # 改为 hyper-diffusion-planner-training，避免两类模型的训练输出目录混淆。
    parser.add_argument('--name', type=str, help='log name (default: "hyper-diffusion-planner-training")', default="hyper-diffusion-planner-training")
    # checkpoint 和训练日志的保存根目录。
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data
    # 预处理 NPZ 缓存目录。
    parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    # manifest 文件，列出本次训练实际读取的样本名称。
    parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)

    # 自车未来轨迹长度，默认 80 个采样点。
    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    # 历史观测时间长度，默认 21 个采样点。
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    # 单个动态交通参与者的历史状态特征维度。
    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    # 场景中最多保留的动态交通参与者数量。
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    # 单个静态物体的状态维度及场景中最多保留的静态物体数量。
    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    # 每条 lane 的采样点数、单点特征维度以及最多保留的 lane 数量。
    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    # route lane 的采样点数、单点特征维度以及最多保留的 route lane 数量。
    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # DataLoader parameters
    # 状态扰动增强概率。
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    # 保存观测和轨迹归一化统计量的 JSON 文件。
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    # 是否启用状态扰动数据增强。
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    # 每个 DataLoader 使用的 CPU worker 数量。
    parser.add_argument('--num_workers', default=4, type=int)
    # 【HDP 与原 Diffusion-Planner 的区别：无梯度诊断】HDP 新增该调试参数；
    # 原入口没有这个选项。
    # 调试开关：输出当前 batch 中没有梯度的参数；正常训练默认关闭。
    parser.add_argument('--log_unused_parameters', default=False, type=boolean)
    # 固定页内存通常能提高 CPU 到 GPU 的异步传输效率。
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # 提供显式关闭 pin_memory 的命令行选项。
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    # 默认启用 pin_memory。
    parser.set_defaults(pin_mem=True)
    
    # Training
    # 固定随机种子，提高实验可复现性；各 DDP rank 会在此基础上加自己的 global_rank。
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    # 训练总 epoch 数。
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=500)
    # checkpoint 保存间隔，默认每 20 个 epoch 保存一次。
    parser.add_argument('--save_utd', type=int, help='save frequency', default=20)
    # 全局 batch size，后续会除以 DDP world size 得到单进程 batch size。
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=2048)
    # AdamW 初始学习率。
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    # 学习率 warm-up 持续的 epoch 数。
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    # 完整恢复通常连同 scheduler 状态一起恢复；受控学习率分叉实验可显式丢弃旧 scheduler，
    # 并以当前命令行的 learning_rate / warm_up_epoch 重建调度器。
    parser.add_argument(
        '--reset_lr_schedule_on_resume',
        default=False,
        type=boolean,
        help='reset optimizer lr and rebuild scheduler after loading a full checkpoint',
    )
    # Encoder/Decoder 的 stochastic-depth（drop path）比例。
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    # 【HDP 与原 Diffusion-Planner 的区别：损失权重】原入口提供 alpha_planning_loss，
    # 用于组合邻车预测损失与自车规划损失；HDP 不训练邻车预测，改为控制自车积分轨迹
    # 混合损失的权重。
    # HDP 新增的积分轨迹损失权重；原 Diffusion-Planner 使用 alpha_planning_loss
    # 加权邻车预测损失，而 HDP 不再训练邻车预测。
    # 【论文 HDP】Table 6 使用 omega=0.1；后续 RL 必须沿用同一权重。
    parser.add_argument('--planning_hybrid_loss', type=float, help='coefficient of planning hybrid loss (default: 0.1)', default=0.1)
    # 积分轨迹混合损失的反向传播窗口。0 表示关闭 stop-gradient，使用普通
    # torch.cumsum；正数表示只允许最近若干 displacement 接收位置损失梯度。
    # 当前正式监督训练要求不启用 detach，因此默认值和运行命令均使用 0。
    parser.add_argument(
        '--planning_detach_window_size',
        type=int,
        default=0,
        help='hybrid waypoint integral gradient window; 0 disables detach',
    )

    # 模型和训练张量使用的设备，默认使用 CUDA。
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    # 是否维护 EMA 参数副本；当前训练、恢复和保存流程默认按启用 EMA 使用。
    parser.add_argument('--use_ema', default=True, type=boolean)

    # Model
    # Encoder/Decoder 深度、注意力头数和隐藏维度共同定义 HDP 的网络规模。
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    # 【HDP 与原 Diffusion-Planner 的区别：扩散参数化】原入口的 diffusion_model_type 只支持
    # score/x_start；HDP 扩展为 v/noise/x_start/score。
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model prediction', choices=['v', 'x_start', 'noise', 'score'], default='x_start')
    # HDP 还把模型输出类型与监督目标类型分开配置。
    # HDP 允许“模型输出参数化”和“监督目标参数化”分开设置，便于统一训练 noise/v/x_start/score。
    parser.add_argument('--diffusion_supervision_type', type=str, help='type of diffusion model supervision', choices=['v', 'x_start', 'noise', 'score'], default='x_start')

    # decoder
    # 【HDP 与原 Diffusion-Planner 的区别：邻车预测】原模型使用 predicted_neighbor_num
    # 控制联合预测的邻车数量；HDP Decoder 不输出邻车未来轨迹，该参数仅为兼容数据集接口保留。
    # 该参数仅为复用数据集接口保留；HDP 的 Decoder 已不输出邻车轨迹。
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict [Warning] Neighbor prediction is deprecated in HDP', default=10)
    # 完整恢复一次训练，包含模型、优化器、调度器、epoch 和 EMA 状态。
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default=None)
    # 【HDP 与原 Diffusion-Planner 的区别：encoder 迁移训练】原入口只有完整 resume；
    # HDP 新增 encoder-only warm-start，并可在训练初始阶段冻结 encoder 指定轮数。
    parser.add_argument(
        '--encoder_pretrained_model_path',
        type=str,
        default=None,
        help='checkpoint used to warm-start only compatible encoder tensors',
    )
    parser.add_argument(
        '--freeze_encoder_epochs',
        type=int,
        default=0,
        help='number of initial epochs that keep a warm-started encoder frozen',
    )

    # 是否启用 Weights & Biases；notes 保存实验备注。
    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    # 是否启用分布式训练，以及进程组使用的通信端口。
    parser.add_argument('--ddp', default=True, type=boolean, help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    # 将命令行输入解析为 argparse.Namespace，后续通过 args.xxx 读取。
    args = parser.parse_args()
    if args.planning_detach_window_size < 0:
        raise ValueError('planning_detach_window_size must be non-negative')

    # 根据 normalization.json 创建状态和观测归一化器，供 train_epoch 和模型共同使用。
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    # 返回训练所需的全部配置和已经构造好的归一化器。
    return args

# 执行完整训练生命周期：初始化 DDP、构造数据与模型、恢复状态、循环训练并保存 checkpoint。
def model_training(args):

    # 【HDP 与原 Diffusion-Planner 的区别：参数合法性】原入口没有 encoder warm-start 参数，
    # 因此没有这些检查；HDP 禁止完整 resume 与 encoder-only 初始化同时使用，并校验冻结轮数。
    if args.resume_model_path is not None and args.encoder_pretrained_model_path is not None:
        raise ValueError('resume_model_path and encoder_pretrained_model_path are mutually exclusive')
    if args.reset_lr_schedule_on_resume and args.resume_model_path is None:
        raise ValueError('reset_lr_schedule_on_resume requires resume_model_path')
    if args.freeze_encoder_epochs < 0:
        raise ValueError('freeze_encoder_epochs must be non-negative')
    # 从完整 checkpoint 恢复时，encoder 权重已包含在 checkpoint 中，仍需保留原实验的
    # freeze_encoder_epochs，保证恢复后的冻结/解冻时机与中断前一致。
    if (
        args.freeze_encoder_epochs > 0
        and args.encoder_pretrained_model_path is None
        and args.resume_model_path is None
    ):
        raise ValueError(
            'freeze_encoder_epochs requires encoder_pretrained_model_path or resume_model_path'
        )

    # 初始化分布式环境。
    # global_rank 是当前进程在全部进程中的编号；rank 是本机设备编号；第三项未使用。
    # 这里沿用原入口写法传入固定 True，实际配置仍通过 args 交给内部 DDP 工具处理。
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    # 只有主进程创建输出目录、保存参数并打印实验信息，避免多个进程竞争写同一文件。
    if global_rank == 0:
        # Logging
        # 打印最常用的实验配置，便于启动后快速核对。
        print("------------- {} -------------".format(args.name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        # 恢复训练时沿用原保存路径；新实验按名称和时间戳创建独立目录。
        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            # datetime 仅在创建新实验目录时使用。
            from datetime import datetime
            time = datetime.now()
            # 时间戳用于区分同名实验的多次运行。
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            # exist_ok=True 允许目标目录已经存在。
            os.makedirs(save_path, exist_ok=True)

        # Save args
        # vars() 把 Namespace 转成普通字典；归一化器需要先转成可 JSON 序列化的字典。
        args_dict = vars(args)
        args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

        # 保存完整 args.json，供恢复训练、离线评估和 NuPlan Planner 推理时复用。
        from mmengine.fileio import dump
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    else:
        # 非主进程不负责文件输出。
        save_path = None

    # set seed
    # 不同 rank 使用不同随机序列，避免各 GPU 完全重复随机增强和扩散噪声。
    set_seed(args.seed + global_rank)

    # training parameters
    # 读取为局部变量，后续构造调度器和单进程 batch size 时使用。
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    # 根据开关创建状态扰动器；关闭增强时向 train_epoch 传入 None。
    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    # 数据集读取预处理缓存；predicted_neighbor_num 在 HDP 中只保留数据接口兼容作用。
    train_set = DiffusionPlannerData(args.train_set, args.train_set_list, args.agent_num, args.predicted_neighbor_num, args.future_len)
    # 每个 rank 通过 DistributedSampler 获得不同数据切片，并在每个 epoch 重新打乱。
    train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    # 全局 batch size 均分到各进程；drop_last=True 丢弃不足一个完整 batch 的尾部样本。
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
   
    # 仅主进程打印数据集大小，避免 DDP 重复输出。
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    # 等待所有进程完成 DataLoader 构造后再继续创建模型。
    if args.ddp:
        torch.distributed.barrier()

    # 【HDP 与原 Diffusion-Planner 的区别：模型构建】原入口实例化 Diffusion_Planner；
    # 当前入口实例化 Hyper_Diffusion_Planner，并在请求时只加载兼容 encoder，输出加载报告。
    # set up model
    diffusion_planner = Hyper_Diffusion_Planner(args)
    # encoder-only warm-start 只在用户显式提供源 checkpoint 时执行。
    if args.encoder_pretrained_model_path is not None:
        warm_start_report = load_encoder_warm_start(
            diffusion_planner,
            args.encoder_pretrained_model_path,
        )
        # 只让主进程保存和打印加载审计，避免并发写同一个 JSON。
        if global_rank == 0:
            from mmengine.fileio import dump
            dump(
                warm_start_report,
                os.path.join(save_path, 'encoder_warm_start_report.json'),
                file_format='json',
                indent=4,
            )
            print(
                'Encoder warm-start: '
                f"loaded={warm_start_report['loaded_tensor_count']}/"
                f"{warm_start_report['target_encoder_tensor_count']}, "
                f"parameters={warm_start_report['loaded_parameter_count']}, "
                f"decoder_loaded={warm_start_report['decoder_tensor_count_loaded']}"
            )
    # CUDA 模式按本地 rank 选择 GPU；CPU 模式使用 args.device。
    diffusion_planner = diffusion_planner.to(rank if args.device == 'cuda' else args.device)

    # 【HDP 与原 Diffusion-Planner 的区别：DDP 包装】原入口只传 device_ids；HDP 的限速
    # embedding 存在数据依赖分支，因此额外启用 find_unused_parameters=True。
    if args.ddp:
        # 车道速度限制编码包含数据依赖分支；某些 batch 可能不使用
        # unknown_speed_emb 或 speed_limit_emb，因此需要启用未使用参数检测。
        diffusion_planner = DDP(
            diffusion_planner,
            device_ids=[rank],
            find_unused_parameters=True,
        )

    # 创建 EMA 参数副本。decay 越接近 1，参数变化越平滑。
    # 与原入口一致，后续训练、恢复和保存路径均假定 model_ema 已创建。
    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
    
    # ddp.get_model() 在 DDP 模式下返回内部 module，否则直接返回原模型。
    if global_rank == 0:
        print("Model Params: {}".format(sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())))

    # optimizer
    # 将当前可训练模型参数交给 AdamW；encoder 后续通过 requires_grad 控制是否产生梯度。
    params = [{'params': ddp.get_model(diffusion_planner, args.ddp).parameters(), 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)
    # 先 warm-up，再按余弦计划更新学习率。
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    # 完整 resume 会同时恢复模型、优化器、调度器、起始 epoch、日志 ID 和 EMA。
    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device)
        if args.reset_lr_schedule_on_resume:
            # 受控分叉实验保留模型、EMA 和 AdamW 动量，但覆盖 checkpoint 中的旧学习率，
            # 再按当前参数重建 scheduler；warm_up_epoch<=1 时即得到恒定学习率。
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
                param_group['initial_lr'] = args.learning_rate
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer,
                train_epochs,
                args.warm_up_epoch,
            )
            print(
                'Learning-rate schedule reset after resume: '
                f'lr={args.learning_rate}, warm_up_epoch={args.warm_up_epoch}'
            )
    else:
        # 新实验从 epoch 0 开始，并创建新的日志运行。
        init_epoch = 0
        wandb_id = None

    # DistributedSampler 的 epoch 不在 checkpoint 中；恢复时显式对齐到下一训练 epoch，
    # 避免每次重启都从 sampler epoch 0 的相同 shuffle 顺序重新开始。
    train_sampler.set_epoch(init_epoch)

    # logger
    # rank 信息交给 Logger，由其避免多进程重复记录。
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    # 所有进程完成模型、优化器和日志器初始化后再进入训练循环。
    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        # 【HDP 与原 Diffusion-Planner 的区别：逐轮冻结控制】原入口直接开始 train_epoch；
        # HDP 每轮先根据 freeze_encoder_epochs 设置 encoder 是否参与反向传播，并输出当前状态。
        base_model = ddp.get_model(diffusion_planner, args.ddp)
        encoder_trainable = epoch >= args.freeze_encoder_epochs
        set_encoder_trainable(base_model, encoder_trainable)
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
            print(f"Encoder trainable: {encoder_trainable}")
        # 调用 HDP 的单 epoch 实现，执行前向、loss、反向传播、优化器和 EMA 更新。
        train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args, model_ema, aug)
        


        # 只有主进程记录指标和保存 checkpoint，避免多个 rank 同时写文件。
        if global_rank == 0:
            # 读取当前优化器学习率。
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            # 字典推导式为每项 loss 加上 train_loss/ 前缀；学习率使用 lr/ 前缀。
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)

            # 每隔 save_utd 个 epoch 保存一次完整训练状态和 EMA 权重。
            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                save_model(diffusion_planner, optimizer, scheduler, save_path, epoch, train_total_loss, wandb_logger.id, model_ema.ema)
                print(f"Model saved in {save_path}\n")

        # epoch 结束后更新学习率。
        scheduler.step()
        # 为下一轮设置 epoch，使 DistributedSampler 使用新的确定性随机顺序。
        train_sampler.set_epoch(epoch + 1)

# 只有直接执行该文件时才解析参数并开始训练；被其他模块 import 时不会自动运行。
if __name__ == "__main__":

    # 解析命令行参数。
    args = get_args()
    
    # Run
    # 启动完整监督训练。
    model_training(args)
