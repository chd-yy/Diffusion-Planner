# 该脚本用于：
# 1. 加载已经完成模仿学习预训练的 Hyper Diffusion Planner；
# 2. 在预先缓存的 nuPlan 数据上生成多组候选轨迹；
# 3. 使用可微的张量奖励函数为候选轨迹计算奖励；
# 4. 将候选轨迹及其奖励存入经验回放缓冲区；
# 5. 使用组内相对优势和奖励加权扩散损失微调规划模型。
#
# 整体训练流程可以概括为：
#
# 预训练 HDP
#     ↓
# 对同一场景生成多条候选轨迹
#     ↓
# 计算 progress、collision、route、comfort 等奖励
#     ↓
# 对组内奖励进行标准化，得到相对优势
#     ↓
# 将优势转换为扩散损失的样本权重
#     ↓
# 更新扩散规划器参数
#
# 需要特别注意：
# 该文件主要负责搭建训练入口和组织 rollout/update 两个阶段。
# 候选轨迹生成、奖励计算、组内优势和加权扩散损失的具体实现，
# 分别位于 hdp_nuplan.rl 和 hdp_nuplan.rl.train_epoch_rl 中。

"""在 NuPlan 缓存数据上进行奖励加权的 HDP 扩散策略微调。"""

# argparse 用于解析命令行参数。
import argparse

# datetime 用于生成当前训练任务的时间戳，
# 从而为每次训练创建独立的日志和模型保存目录。
from datetime import datetime

# os 用于拼接文件路径、创建文件夹等文件系统操作。
import os

# mmengine.fileio.dump 用于将训练参数保存为 JSON 文件。
from mmengine.fileio import dump

# PyTorch 核心库。
import torch

# optim 中包含 AdamW 等优化器。
from torch import optim

# DistributedDataParallel 用于多 GPU 数据并行训练。
#
# 每张 GPU 对应一个独立训练进程：
# 1. 每个进程持有一份完整模型；
# 2. 每个进程处理不同的数据子集；
# 3. 反向传播时自动同步并平均模型梯度。
from torch.nn.parallel import DistributedDataParallel as DDP

# DataLoader 用于按 batch 加载数据。
#
# DistributedSampler 用于在 DDP 模式下将数据集划分给不同进程，
# 避免不同 GPU 重复处理相同样本。
from torch.utils.data import DataLoader, DistributedSampler

# ModelEma 用于维护模型参数的指数滑动平均版本。
#
# EMA 参数更新形式近似为：
#
# ema_parameter
# =
# decay * ema_parameter
# +
# (1 - decay) * current_parameter
#
# EMA 模型通常比训练过程中瞬时参数更加平滑和稳定。
from timm.utils import ModelEma

# Hyper_Diffusion_Planner 是当前要进行强化学习微调的 HDP 模型。
from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner

# NuPlanReplayBuffer：
# 保存 rollout 阶段生成的候选轨迹、场景信息、奖励和训练所需变量。
#
# NuPlanRewardConfig：
# 定义奖励函数中各个分项的权重和阈值。
#
# NuPlanTensorRewardScorer：
# 使用 PyTorch 张量批量计算候选轨迹的奖励。
from hdp_nuplan.rl import (
    NuPlanReplayBuffer,
    NuPlanRewardConfig,
    NuPlanTensorRewardScorer,
)

# rollout_epoch：
# 使用当前策略对场景生成多组候选轨迹、计算奖励并填充 replay buffer。
#
# update_epoch：
# 从 replay buffer 读取候选轨迹，计算奖励加权扩散损失并更新模型。
from hdp_nuplan.rl.train_epoch_rl import rollout_epoch, update_epoch

# 项目中封装的 DDP 初始化、进程信息和分布式通信工具。
from hdp_nuplan.utils import ddp

# DiffusionPlannerData 用于加载预处理后保存的 nuPlan 场景数据。
from hdp_nuplan.utils.dataset import DiffusionPlannerData

# 带有 warm-up 和余弦退火重启机制的学习率调度器。
from hdp_nuplan.utils.lr_schedule import CosineAnnealingWarmUpRestarts

# ObservationNormalizer：
# 对地图、邻车、静态物体等场景输入进行归一化。
#
# StateNormalizer：
# 对自车轨迹、邻车轨迹等状态变量进行归一化。
from hdp_nuplan.utils.normalizer import ObservationNormalizer, StateNormalizer

# TensorBoard 日志记录器，同时可以根据参数选择是否使用 WandB。
from hdp_nuplan.utils.tb_log import TensorBoardLogger as Logger

# save_model 用于保存模型、优化器、调度器和 EMA 参数。
#
# set_seed 用于设置随机种子，尽可能保证实验可复现。
from hdp_nuplan.utils.train_utils import save_model, set_seed


# argparse 默认不能直接可靠地将命令行字符串转换为布尔值。
#
# 例如：
#
# bool("false")
#
# 在 Python 中仍然是 True，因为非空字符串都被视为 True。
#
# 因此，这里显式定义字符串到布尔值的转换规则。
def boolean(value):
    # 如果传入值本身已经是 bool 类型，直接返回。
    if isinstance(value, bool):
        return value

    # 将 yes、true、t、y、1 等字符串解释为 True。
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True

    # 将 no、false、f、n、0 等字符串解释为 False。
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False

    # 如果字符串不属于支持的范围，则报告命令行参数错误。
    raise argparse.ArgumentTypeError("Boolean value expected.")


# 定义并解析所有训练参数。
def get_args():
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="HDP-nuPlan diffusion RL")

    # ------------------------------------------------------------------
    # 实验名称、数据路径和预训练模型路径
    # ------------------------------------------------------------------

    # 当前实验名称，用于日志目录命名。
    parser.add_argument("--name", default="hdp-nuplan-rl", type=str)

    # 训练日志、检查点等输出文件的根目录。
    parser.add_argument("--save_dir", default=".", type=str)

    # 预处理后的 nuPlan 训练数据根目录。
    parser.add_argument("--train_set", required=True, type=str)

    # 训练样本文件列表。
    #
    # 数据集通常不会直接扫描整个目录，而是根据该列表加载指定场景。
    parser.add_argument("--train_set_list", required=True, type=str)

    # 模仿学习预训练模型的检查点路径。
    #
    # 强化学习不是从随机初始化开始，而是在已经具备基础驾驶能力的
    # HDP 模型上继续微调。
    parser.add_argument("--pretrained_model_path", required=True, type=str)

    # 状态和观测归一化参数文件。
    parser.add_argument("--normalization_file_path", default="normalization.json")

    # ------------------------------------------------------------------
    # 轨迹和场景输入维度
    # ------------------------------------------------------------------

    # 未来轨迹长度。
    #
    # 默认 future_len=80。
    # 如果数据采样频率为 10 Hz，则表示预测未来 8 秒：
    #
    # 80 / 10 = 8 秒
    parser.add_argument("--future_len", default=80, type=int)

    # 历史序列长度。
    #
    # 默认 time_len=21，通常表示包含当前帧在内的 21 个历史状态。
    parser.add_argument("--time_len", default=21, type=int)

    # 每个动态交通参与者的状态特征维度。
    parser.add_argument("--agent_state_dim", default=11, type=int)

    # 输入场景中最多保留的动态交通参与者数量。
    parser.add_argument("--agent_num", default=32, type=int)

    # 扩散模型联合预测未来轨迹的邻车数量。
    #
    # 该参数是否在 RL 微调中直接参与损失，
    # 取决于 rollout_epoch 和 update_epoch 的具体实现。
    parser.add_argument("--predicted_neighbor_num", default=10, type=int)

    # 每个静态物体的状态特征维度。
    parser.add_argument("--static_objects_state_dim", default=10, type=int)

    # 每个场景最多保留的静态物体数量。
    parser.add_argument("--static_objects_num", default=5, type=int)

    # 每条车道折线包含的采样点数量。
    parser.add_argument("--lane_len", default=20, type=int)

    # 单个车道采样点的状态特征维度。
    parser.add_argument("--lane_state_dim", default=12, type=int)

    # 场景中最多使用的普通车道段数量。
    parser.add_argument("--lane_num", default=70, type=int)

    # 每条导航路线车道折线包含的采样点数量。
    parser.add_argument("--route_len", default=20, type=int)

    # 导航路线中单个采样点的状态维度。
    parser.add_argument("--route_state_dim", default=12, type=int)

    # 场景中最多使用的导航路线车道段数量。
    parser.add_argument("--route_num", default=25, type=int)

    # ------------------------------------------------------------------
    # HDP 模型结构参数
    # ------------------------------------------------------------------

    # 场景编码器的网络深度。
    parser.add_argument("--encoder_depth", default=3, type=int)

    # 扩散解码器的网络深度。
    parser.add_argument("--decoder_depth", default=3, type=int)

    # Transformer 多头注意力中的注意力头数量。
    parser.add_argument("--num_heads", default=6, type=int)

    # 模型隐藏特征维度。
    parser.add_argument("--hidden_dim", default=192, type=int)

    # 扩散模型直接输出的变量类型。
    #
    # 可选值：
    #
    # v：
    # 预测扩散参数化中的 velocity，不是车辆物理速度。
    #
    # x_start：
    # 直接预测干净数据 x_0。
    #
    # noise：
    # 预测前向扩散中加入的高斯噪声 epsilon。
    #
    # score：
    # 直接预测分布的 score function。
    parser.add_argument(
        "--diffusion_model_type",
        choices=["v", "x_start", "noise", "score"],
        default="x_start",
    )

    # 扩散训练损失所在的监督空间。
    #
    # 模型直接输出什么，与最终在哪个变量空间计算损失，可以不同。
    #
    # 例如：
    #
    # diffusion_model_type="noise"
    # diffusion_supervision_type="x_start"
    #
    # 表示模型直接预测噪声，但先将噪声转换为 x_0，
    # 然后在干净轨迹空间计算损失。
    #
    # 根据 HDP 论文实验，默认使用：
    #
    # x_start prediction + x_start supervision
    parser.add_argument(
        "--diffusion_supervision_type",
        choices=["v", "x_start", "noise", "score"],
        default="x_start",
    )

    # 场景编码器中的 stochastic depth 比例。
    parser.add_argument("--encoder_drop_path_rate", default=0.1, type=float)

    # 扩散解码器中的 stochastic depth 比例。
    parser.add_argument("--decoder_drop_path_rate", default=0.1, type=float)

    # 规划混合损失中 waypoint 积分损失的权重。
    #
    # HDP 的混合损失一般可以表示为：
    #
    # L_hybrid
    # =
    # L_velocity
    # +
    # omega * L_waypoints
    #
    # 此处 planning_hybrid_loss 对应平衡系数 omega。
    # 【论文 HDP-RL】Table 6 使用 omega=0.1；监督训练和 RL 必须保持一致。
    parser.add_argument("--planning_hybrid_loss", default=0.1, type=float)

    # ------------------------------------------------------------------
    # 通用训练参数
    # ------------------------------------------------------------------

    # 总训练 epoch 数。
    parser.add_argument("--train_epochs", default=100, type=int)

    # 全局 batch size。
    #
    # 在 DDP 模式下，每个进程实际使用的 batch size 为：
    #
    # batch_size / world_size
    #
    # 例如：
    #
    # batch_size = 32
    # world_size = 4
    #
    # 则每张 GPU 每个训练步骤处理：
    #
    # 32 / 4 = 8 个样本
    parser.add_argument("--batch_size", default=32, type=int)

    # AdamW 优化器的初始学习率。
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    # 学习率 warm-up 的 epoch 数。
    parser.add_argument("--warm_up_epoch", default=5, type=int)

    # 每隔多少个 update epoch 保存一次模型。
    parser.add_argument("--save_utd", default=10, type=int)

    # DataLoader 使用的工作进程数量。
    parser.add_argument("--num_workers", default=4, type=int)

    # 是否将 DataLoader 返回的 CPU 张量放入页锁定内存。
    #
    # 当使用 GPU 训练时，pin_memory=True 通常可以加快
    # CPU 到 GPU 的异步数据传输。
    parser.add_argument("--pin_mem", default=True, type=boolean)

    # 训练设备。
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    # 基础随机种子。
    parser.add_argument("--seed", default=3407, type=int)

    # 是否使用 WandB 记录实验。
    parser.add_argument("--use_wandb", default=False, type=boolean)

    # 实验备注。
    parser.add_argument("--notes", default="", type=str)

    # ------------------------------------------------------------------
    # 强化学习和经验回放参数
    # ------------------------------------------------------------------

    # 每个场景生成的候选轨迹数量。
    #
    # 【论文 HDP-RL】Table 6 的 group size 为 32。
    #
    # tau_1, tau_2, ..., tau_8
    #
    # 然后计算这 8 条轨迹的组内相对奖励或优势。
    parser.add_argument("--rl_group_size", default=32, type=int)

    # rollout 阶段使用的扩散采样步骤数。
    #
    # 【论文 HDP-RL】Appendix D.4 使用 6 次去噪更新。
    #
    # 该参数的具体使用方式由 rollout_epoch 内部实现决定。
    parser.add_argument("--rl_rollout_steps", default=6, type=int)

    # 论文未报告额外放大初始噪声，默认与普通采样保持一致。
    parser.add_argument("--rl_sampling_noise_scale", default=0.1, type=float)

    # 【迁移自 HDP-NAVSIM】对每条 rollout 轨迹施加的局部纵向/横向高斯偏移标准差（米）。
    # 该偏移对整条轨迹共享，航向保持不变；0 表示关闭。
    parser.add_argument("--rl_trajectory_augmentation_std", default=0.0, type=float)

    # 轨迹扰动只在零基 epoch 小于该值时启用；默认 5 对应 NAVSIM 的 epoch < 5。
    parser.add_argument("--rl_trajectory_augmentation_epochs", default=0, type=int)

    # 每隔多少个 epoch 重新生成一次 rollout buffer。
    #
    # 论文式短周期策略迭代默认值为 2：
    #
    # epoch 0：rollout
    # epoch 1：update
    # epoch 2：使用更新后的 EMA 重新 rollout
    #
    # 因此每份 replay 只用于紧随其后的一个 update epoch。
    parser.add_argument("--rl_buffer_update_epoch", default=2, type=int)

    # replay buffer 最多保存的样本数量。
    parser.add_argument("--rl_buffer_size", default=4096, type=int)

    # 【论文 HDP-RL】Table 6 记录 EMA=0.05。本文将其明确解释为每次更新时
    # 当前策略写入 EMA 的比例：ema <- 0.95*ema + 0.05*model。
    # 若后续确认论文的 0.05 指 decay，可通过该参数做对照，不再硬编码。
    parser.add_argument("--rl_ema_update_rate", default=0.05, type=float)

    # 奖励转化为样本权重时使用的温度参数。
    #
    # 奖励加权通常可能具有如下形式：
    #
    # w_i
    # =
    # exp(A_i / temperature)
    #
    # 或：
    #
    # w_i
    # =
    # exp(temperature * A_i)
    #
    # 具体形式需要查看 update_epoch。
    #
    # 一般来说，该参数控制高奖励轨迹和低奖励轨迹之间的权重差异。
    parser.add_argument("--rl_reward_temperature", default=1.0, type=float)

    # 对组内标准化优势进行裁剪的绝对值上限。
    #
    # 例如 advantage_clip=5 时，可能执行：
    #
    # A_i
    # =
    # clip(A_i, -5, 5)
    #
    # 作用是防止极端奖励产生过大的训练权重和梯度。
    parser.add_argument("--rl_advantage_clip", default=5.0, type=float)

    # 【论文 HDP-RL】只丢弃所有 action 奖励完全相同的组。使用 1e-6
    # 吸收浮点误差；旧 reward v2 的 0.01 会误删多数自然 diffusion 候选组。
    parser.add_argument("--rl_min_reward_std", default=1e-6, type=float)

    # 将每个有效候选组的指数权重归一化到均值 1，固定有效学习率。
    parser.add_argument("--rl_normalize_weights", default=True, type=boolean)

    # 【NuPlan 小数据适配，非论文原式】使用 (w-1) 作为 control-variate，
    # 消除有限 replay 上与 reward 无关的无权自蒸馏梯度。默认关闭可复现论文
    # Eq. (9)；小数据实验需显式开启。
    parser.add_argument("--rl_center_reward_weights", default=False, type=boolean)

    # rollout reward-weighted 自蒸馏损失的权重；0 用于 expert-anchor-only 控制实验。
    parser.add_argument("--rl_rollout_loss_weight", default=1.0, type=float)

    # 真实专家轨迹监督 anchor 的权重，用于抑制 rollout 自蒸馏漂移。
    # 论文目标不包含额外 expert anchor；旧 checkpoint 兼容实验可显式传 0.1。
    parser.add_argument("--rl_expert_anchor_weight", default=0.0, type=float)

    # 0 表示完整遍历；正数用于短步门禁，例如 20 表示每个 update epoch 最多 20 步。
    parser.add_argument("--rl_max_update_steps_per_epoch", default=0, type=int)

    # HDP 混合损失中 Detached Integral 的反向传播时间窗口。
    #
    # 速度积分得到位置时，较早速度会影响多个未来位置，
    # 从而积累非常大的梯度。
    #
    # detach_window_size=10 表示位置损失最多向前传播到
    # 最近 10 个速度时间步，更早部分使用 stop-gradient。
    parser.add_argument("--rl_detach_window_size", default=10, type=int)

    # 强化学习更新时的梯度范数裁剪阈值。
    #
    # 典型实现为：
    #
    # clip_grad_norm_(parameters, max_norm=5.0)
    #
    # 作用是避免奖励加权导致梯度突然增大。
    parser.add_argument("--rl_grad_clip", default=5.0, type=float)

    # 是否在强化学习微调阶段冻结场景编码器。
    #
    # True 表示只更新扩散解码器等规划相关模块，
    # 不更新感知和场景编码部分。
    #
    # 这样通常可以：
    # 1. 减少显存和计算量；
    # 2. 避免 RL 奖励破坏已经学好的场景特征；
    # 3. 提高训练稳定性。
    parser.add_argument("--rl_freeze_encoder", default=True, type=boolean)
    # Reward-weighted self-distillation 的 target 来自 eval 模式旧策略；更新时也
    # 关闭 Dropout/DropPath，避免随机删层产生与 reward 无关的漂移。eval 模式
    # 不会关闭 autograd，模型参数仍会正常反向传播和更新。
    parser.add_argument(
        "--rl_deterministic_update", default=True, type=boolean
    )

    # ------------------------------------------------------------------
    # 奖励函数参数
    # ------------------------------------------------------------------

    # 轨迹前进进度的奖励权重。
    #
    # 一般用于鼓励车辆沿导航方向向前行驶，
    # 防止模型通过停车获得较高安全奖励。
    parser.add_argument("--reward_progress_weight", default=1.0, type=float)

    # 碰撞惩罚权重。
    #
    # 默认值为 10，明显大于其他奖励项，
    # 表明碰撞属于需要重点惩罚的安全事件。
    parser.add_argument("--reward_collision_weight", default=10.0, type=float)

    # 路线一致性奖励权重。
    #
    # 用于鼓励轨迹靠近导航路线或可通行区域。
    parser.add_argument("--reward_route_weight", default=1.0, type=float)

    # 舒适性奖励或惩罚的权重。
    #
    # v2 默认使用 pilot 已验证的 0.01，避免 jerk 数值主导总奖励。
    parser.add_argument("--reward_comfort_weight", default=0.01, type=float)

    # 倒车或向后运动的惩罚权重。
    parser.add_argument("--reward_backward_weight", default=1.0, type=float)

    # 模仿专家轨迹的奖励权重。
    #
    # 默认值为 0，表示奖励主要由安全、进度、路线和舒适性决定，
    # 不直接要求候选轨迹接近专家轨迹。
    #
    # 但更新时是否仍然包含普通扩散模仿损失，
    # 需要查看 update_epoch 的具体实现。
    parser.add_argument("--reward_imitation_weight", default=0.0, type=float)

    # reward v2 的 OBB 几何安全余量，单位为米；0 表示仅惩罚包围盒重叠。
    parser.add_argument("--reward_collision_distance", default=0.5, type=float)

    # 【论文 HDP-RL / Table 6】multi-reward 的三个固定权重。
    parser.add_argument("--reward_risk_weight", default=1.0, type=float)
    parser.add_argument("--reward_follow_weight", default=3.0, type=float)
    parser.add_argument("--reward_lane_weight", default=2.5, type=float)
    # 【NuPlan 小数据适配，非论文原式】默认 0 保持论文奖励；显式开启后，
    # 使用相对专家进度的有界奖励抑制弱 IL 起点上的 stop hacking。
    parser.add_argument("--reward_progress_guard_weight", default=0.0, type=float)
    parser.add_argument(
        "--reward_progress_guard_stop_tolerance", default=0.2, type=float
    )

    # 【NuPlan 适配】论文没有公开 TTC/THW/OCC shaping 的具体阈值，
    # 因此作为命令行参数写入 args.json，保证实验可以准确复现。
    parser.add_argument("--reward_risk_speed_reference", default=15.0, type=float)
    parser.add_argument("--reward_ttc_safe_low_speed", default=2.0, type=float)
    parser.add_argument("--reward_ttc_safe_high_speed", default=4.0, type=float)
    parser.add_argument("--reward_thw_critical", default=0.5, type=float)
    parser.add_argument("--reward_thw_safe_low_speed", default=1.0, type=float)
    parser.add_argument("--reward_thw_safe_high_speed", default=2.0, type=float)
    parser.add_argument("--reward_occupancy_safe_min", default=0.5, type=float)
    parser.add_argument("--reward_occupancy_safe_max", default=3.0, type=float)
    parser.add_argument("--reward_occupancy_time_headway", default=0.2, type=float)
    parser.add_argument("--reward_rear_end_collision_penalty", default=0.3, type=float)

    # 【NuPlan 适配】论文跟车奖励的 ACC 风格四个分项所需参数。
    parser.add_argument("--reward_follow_time_gap_low_speed", default=1.0, type=float)
    parser.add_argument("--reward_follow_time_gap_high_speed", default=2.0, type=float)
    parser.add_argument("--reward_follow_min_spacing", default=2.0, type=float)
    parser.add_argument("--reward_follow_speed_tolerance", default=2.0, type=float)
    parser.add_argument("--reward_follow_comfort_acceleration", default=2.0, type=float)
    parser.add_argument("--reward_follow_comfort_deceleration", default=3.0, type=float)
    parser.add_argument("--reward_leader_lateral_margin", default=0.5, type=float)
    parser.add_argument("--reward_lane_half_width_fallback", default=1.75, type=float)
    parser.add_argument("--reward_lane_change_ratio", default=0.5, type=float)

    # ------------------------------------------------------------------
    # 分布式训练参数
    # ------------------------------------------------------------------

    # 是否启用 DDP。
    parser.add_argument("--ddp", default=True, type=boolean)

    # PyTorch 分布式进程组使用的通信端口。
    parser.add_argument("--port", default="22324", type=str)

    # 解析命令行参数。
    args = parser.parse_args()

    # 组内相对优势至少需要两个候选样本才能定义。
    #
    # 如果每组只有一条轨迹：
    #
    # reward_mean = reward
    #
    # reward - reward_mean = 0
    #
    # 因而无法比较同一场景下不同候选轨迹的相对优劣。
    if args.rl_group_size < 2:
        raise ValueError("rl_group_size 至少为 2，组内优势才有意义")
    if args.rl_sampling_noise_scale <= 0:
        raise ValueError("rl_sampling_noise_scale 必须大于 0")
    if args.rl_trajectory_augmentation_std < 0:
        raise ValueError("rl_trajectory_augmentation_std 不能为负数")
    if args.rl_trajectory_augmentation_epochs < 0:
        raise ValueError("rl_trajectory_augmentation_epochs 不能为负数")
    if args.rl_min_reward_std < 0:
        raise ValueError("rl_min_reward_std 不能为负数")
    if args.rl_rollout_loss_weight < 0:
        raise ValueError("rl_rollout_loss_weight 不能为负数")
    if args.rl_expert_anchor_weight < 0:
        raise ValueError("rl_expert_anchor_weight 不能为负数")
    if args.rl_rollout_loss_weight == 0 and args.rl_expert_anchor_weight == 0:
        raise ValueError("rl_rollout_loss_weight 和 rl_expert_anchor_weight 不能同时为 0")
    if args.rl_max_update_steps_per_epoch < 0:
        raise ValueError("rl_max_update_steps_per_epoch 不能为负数")
    if not 0 < args.rl_ema_update_rate <= 1:
        raise ValueError("rl_ema_update_rate 必须在 (0,1]")
    if min(
        args.reward_risk_weight,
        args.reward_follow_weight,
        args.reward_lane_weight,
        args.reward_progress_guard_weight,
    ) < 0:
        raise ValueError("multi-reward 及 progress guard 权重不能为负数")
    if args.reward_progress_guard_stop_tolerance <= 0:
        raise ValueError("reward_progress_guard_stop_tolerance 必须大于 0")
    if args.reward_risk_speed_reference <= 0:
        raise ValueError("reward_risk_speed_reference 必须大于 0")
    if not 0 <= args.reward_rear_end_collision_penalty <= 1:
        raise ValueError("reward_rear_end_collision_penalty 必须在 [0,1]")
    if args.reward_thw_safe_low_speed <= args.reward_thw_critical:
        raise ValueError("低速安全 THW 必须大于临界 THW")
    if args.reward_thw_safe_high_speed <= args.reward_thw_critical:
        raise ValueError("高速安全 THW 必须大于临界 THW")

    # 从 normalization_file_path 指定的 JSON 文件中
    # 读取轨迹状态归一化参数。
    args.state_normalizer = StateNormalizer.from_json(args)

    # 从同一个归一化文件中读取场景观测归一化参数。
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    # 返回完整训练参数。
    return args


# 从模仿学习预训练检查点中加载模型参数。
def _load_pretrained(model, checkpoint_path, device):
    # 将检查点加载到指定设备。
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 优先加载 EMA 模型参数。
    #
    # EMA 参数通常比训练最后一步的普通参数更加稳定，
    # 因此如果检查点包含 ema_state_dict，则优先使用。
    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]

    # 如果没有 EMA 参数，则尝试读取键名 model。
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    # 如果检查点本身就是 state_dict，则直接使用。
    else:
        state_dict = checkpoint

    # 如果模型在 DDP 模式下保存，参数名称通常带有：
    #
    # module.encoder...
    # module.decoder...
    #
    # 而当前未包装 DDP 的模型参数名称通常是：
    #
    # encoder...
    # decoder...
    #
    # 因此需要删除参数名前缀 module.。
    state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    # strict=False 表示允许预训练检查点与当前模型之间存在少量键不匹配。
    #
    # missing：
    # 当前模型需要，但检查点中不存在的参数。
    #
    # unexpected：
    # 检查点中存在，但当前模型不需要的参数。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 输出加载情况，帮助检查是否存在严重的模型结构不匹配。
    print(
        f"Loaded pretrained model: missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )


def ema_decay_from_update_rate(update_rate: float) -> float:
    """把论文式 EMA 更新比例转换为 timm.ModelEma 使用的 decay。"""
    if not 0 < update_rate <= 1:
        raise ValueError("EMA update_rate 必须在 (0,1]")
    return 1.0 - update_rate


# 强化学习训练主函数。
def model_training(args):
    # 初始化分布式训练环境。
    #
    # global_rank：
    # 当前进程在所有训练进程中的全局编号。
    #
    # local_rank：
    # 当前进程在本机中的 GPU 编号。
    #
    # 第三个返回值在此处未使用。
    global_rank, local_rank, _ = ddp.ddp_setup_universal(True, args)

    # 未通过 torchrun 启动时 ddp_setup 会回退单进程，此时同步修正标志，
    # 避免后续 get_model() 误以为模型一定带有 .module。
    args.ddp = args.ddp and ddp.is_dist_avail_and_initialized()

    # 根据 local_rank 为当前进程绑定对应 GPU。
    #
    # 例如：
    #
    # local_rank=0 → cuda:0
    # local_rank=1 → cuda:1
    if args.device == "cuda":
        device = torch.device("cuda", local_rank)

    # CPU 训练模式。
    else:
        device = torch.device("cpu")

    # 不同 DDP 进程使用不同随机种子。
    #
    # 这样可以避免不同 GPU 产生完全相同的随机扩散噪声和候选轨迹。
    #
    # 例如：
    #
    # rank 0：seed = 3407
    # rank 1：seed = 3408
    # rank 2：seed = 3409
    set_seed(args.seed + global_rank)

    # 创建 nuPlan 缓存数据集。
    # Dataset长度：100
    # DataLoader每批：4个场景
    # DataLoader批数：25
    dataset = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,

        # 返回场景标识、文件信息等元数据。
        #
        # RL rollout 和 replay buffer 通常需要使用元数据区分场景。
        return_metadata=True,
    )

    # 将数据集划分给不同 DDP 进程。
    sampler = DistributedSampler(
        dataset,

        # 当前参与训练的总进程数。
        num_replicas=ddp.get_world_size(),

        # 当前进程的全局编号。
        rank=global_rank,

        # 每个 epoch 重新打乱样本顺序。
        shuffle=True,
    )

    # 创建批量数据加载器。
    data_loader = DataLoader(
        dataset,

        # 使用 DistributedSampler 控制样本划分，
        # 因此这里不再单独设置 shuffle=True。
        sampler=sampler,

        # args.batch_size 表示全局 batch size。
        #
        # 每个 DDP 进程实际 batch size 为：
        #
        # max(1, global_batch_size / world_size)
        batch_size=max(1, args.batch_size // ddp.get_world_size()),

        # DataLoader 子进程数量。
        num_workers=args.num_workers,

        # 是否使用页锁定内存。
        pin_memory=args.pin_mem,

        # 不丢弃最后一个不足完整 batch 的小批次。
        drop_last=False,
    )

    # 创建 Hyper Diffusion Planner 模型，并移动到当前训练设备。
    model = Hyper_Diffusion_Planner(args).to(device)

    # 加载模仿学习预训练模型参数。
    _load_pretrained(model, args.pretrained_model_path, device)

    # 如果启用 rl_freeze_encoder，则冻结场景编码器。
    if args.rl_freeze_encoder:
        for parameter in model.encoder.parameters():
            # requires_grad=False 后：
            # 1. 该参数不计算梯度；
            # 2. 优化器不会更新该参数；
            # 3. 可以减少显存和反向计算量。
            parameter.requires_grad_(False)

    # 如果已经成功初始化分布式环境，则使用 DDP 包装模型。
    if args.ddp and ddp.is_dist_avail_and_initialized():
        model = DDP(
            model,

            # CUDA 模式下，每个进程只控制一张 GPU。
            device_ids=[local_rank] if args.device == "cuda" else None,

            # 当编码器被冻结时，部分参数不会参与反向传播，
            # 因此允许 DDP 检测未使用参数。
            find_unused_parameters=args.rl_freeze_encoder,
        )

    # 只收集 requires_grad=True 的模型参数。
    #
    # 如果编码器被冻结，则优化器中不会包含编码器参数。
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    # 使用 AdamW 优化可训练参数。
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate)

    # 创建带 warm-up 的余弦退火学习率调度器。
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,

        # 总训练 epoch 数。
        args.train_epochs,

        # warm-up epoch 数。
        args.warm_up_epoch,
    )

    # 创建 EMA 模型。
    #
    # timm 的定义为 ema <- decay*ema + (1-decay)*model。
    # 默认 update_rate=0.05，因此传入 decay=0.95。
    ema = ModelEma(
        model,
        decay=ema_decay_from_update_rate(args.rl_ema_update_rate),
        device=str(device),
    )

    # 创建经验回放缓冲区。
    #
    # 它保存 rollout 阶段生成的候选轨迹及相关训练信息，
    # 供后续 update_epoch 重复使用。
    replay_buffer = NuPlanReplayBuffer(max_size=args.rl_buffer_size)

    # 创建轨迹奖励计算器。
    reward_scorer = NuPlanTensorRewardScorer(
        NuPlanRewardConfig(
            # 沿路线向前行驶的奖励权重。
            progress_weight=args.reward_progress_weight,

            # 碰撞惩罚权重。
            collision_weight=args.reward_collision_weight,

            # 路线一致性奖励权重。
            route_weight=args.reward_route_weight,

            # 舒适性奖励权重。
            comfort_weight=args.reward_comfort_weight,

            # 向后运动惩罚权重。
            backward_weight=args.reward_backward_weight,

            # 接近专家轨迹的奖励权重。
            imitation_weight=args.reward_imitation_weight,

            # 碰撞或危险接近距离阈值。
            collision_distance=args.reward_collision_distance,

            # 【论文 HDP-RL】risk/follow/lane 权重。
            risk_weight=args.reward_risk_weight,
            follow_weight=args.reward_follow_weight,
            lane_weight=args.reward_lane_weight,
            progress_guard_weight=args.reward_progress_guard_weight,
            progress_guard_stop_tolerance=(
                args.reward_progress_guard_stop_tolerance
            ),

            # 【NuPlan 适配】论文未公开的 shaping 阈值。
            risk_speed_reference=args.reward_risk_speed_reference,
            ttc_safe_low_speed=args.reward_ttc_safe_low_speed,
            ttc_safe_high_speed=args.reward_ttc_safe_high_speed,
            thw_critical=args.reward_thw_critical,
            thw_safe_low_speed=args.reward_thw_safe_low_speed,
            thw_safe_high_speed=args.reward_thw_safe_high_speed,
            occupancy_safe_min=args.reward_occupancy_safe_min,
            occupancy_safe_max=args.reward_occupancy_safe_max,
            occupancy_time_headway=args.reward_occupancy_time_headway,
            rear_end_collision_penalty=args.reward_rear_end_collision_penalty,
            follow_time_gap_low_speed=args.reward_follow_time_gap_low_speed,
            follow_time_gap_high_speed=args.reward_follow_time_gap_high_speed,
            follow_min_spacing=args.reward_follow_min_spacing,
            follow_speed_tolerance=args.reward_follow_speed_tolerance,
            follow_comfort_acceleration=args.reward_follow_comfort_acceleration,
            follow_comfort_deceleration=args.reward_follow_comfort_deceleration,
            leader_lateral_margin=args.reward_leader_lateral_margin,
            lane_half_width_fallback=args.reward_lane_half_width_fallback,
            lane_change_ratio=args.reward_lane_change_ratio,
        )
    )

    # 获取当前时间，用于创建独立实验目录。
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # 构造日志和模型保存路径：
    #
    # save_dir/
    #   training_log/
    #     experiment_name/
    #       timestamp/
    save_path = os.path.join(
        args.save_dir,
        "training_log",
        args.name,
        timestamp,
    )

    # 只有主进程负责创建文件夹和写入参数文件，
    # 避免多个 DDP 进程同时操作同一文件。
    if global_rank == 0:
        os.makedirs(save_path, exist_ok=True)

        # 将 argparse Namespace 转换为可写入 JSON 的字典。
        serializable_args = {
            key: (
                # Normalizer 对象不能直接序列化为 JSON，
                # 因此先转换为普通字典。
                value.to_dict()
                if isinstance(value, (StateNormalizer, ObservationNormalizer))

                # 其他普通参数直接保存。
                else value
            )
            for key, value in vars(args).items()
        }

        # 保存本次训练的所有参数。
        dump(
            serializable_args,
            os.path.join(save_path, "args.json"),
            file_format="json",
            indent=4,
        )

    # 在所有进程到达这里之前进行同步。
    #
    # 确保主进程已经创建好保存目录，
    # 然后其他进程才继续执行。
    if args.ddp and ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # 创建 TensorBoard/WandB 日志记录器。
    logger = Logger(
        args.name,
        args.notes,
        args,

        # 当前不从已有 WandB 运行恢复。
        wandb_resume_id=None,

        # 日志保存路径。
        save_path=save_path,

        # 当前进程编号。
        rank=global_rank,
    )

    # 开始逐 epoch 训练。
    for epoch in range(args.train_epochs):
        # 将 epoch 传给 DistributedSampler。
        #
        # DistributedSampler 会利用 epoch 改变随机种子，
        # 从而保证每个 epoch 的数据打乱顺序不同，
        # 同时所有 DDP 进程仍保持一致的数据划分规则。
        sampler.set_epoch(epoch)

        # 判断当前 epoch 是 rollout 阶段还是 update 阶段。
        #
        # 默认 rl_buffer_update_epoch=2 时：
        #
        # 偶数 epoch 执行 rollout；奇数 epoch 执行一次 update。
        is_rollout_epoch = epoch % args.rl_buffer_update_epoch == 0

        # --------------------------------------------------------------
        # Rollout 阶段
        # --------------------------------------------------------------
        if is_rollout_epoch:
            # 每次重新 rollout 前清空旧经验。
            #
            # 这样 replay buffer 中只保存当前策略最新生成的数据，
            # 避免过旧策略的数据长期影响训练。
            replay_buffer.clear()

            # 【论文 HDP-RL】使用冻结的 EMA 旧策略生成候选轨迹。
            # update 阶段只修改 model，并在 optimizer step 后平滑更新 ema；
            # 因而本轮 replay 中的 action 始终来自固定的 pi_{k-1}。
            #
            # rollout_epoch 内部通常会执行：
            #
            # 1. 编码当前 nuPlan 场景；
            # 2. 对同一场景复制 rl_group_size 份；
            # 3. 为每份样本采样不同扩散噪声；
            # 4. 使用 rl_rollout_steps 次去噪生成候选轨迹；
            # 5. 使用 reward_scorer 计算每条轨迹的奖励；
            # 6. 将候选轨迹、奖励和场景条件写入 replay_buffer；
            # 7. 汇总 rollout 的平均奖励、碰撞率等指标。
            metrics = rollout_epoch(
                data_loader,
                ema.ema,
                replay_buffer,
                reward_scorer,
                args,
                device,
                epoch=epoch,
            )

            # 标记当前阶段名称。
            phase = "rollout"

        # --------------------------------------------------------------
        # Update 阶段
        # --------------------------------------------------------------
        else:
            # 从 replay buffer 中读取 rollout 数据，
            # 计算奖励加权扩散损失并更新模型参数。
            #
            # update_epoch 内部通常会执行：
            #
            # 1. 读取同一场景下的一组候选轨迹及其奖励；
            # 2. 对组内奖励进行均值和标准差归一化；
            # 3. 得到组内相对优势；
            # 4. 对优势进行裁剪；
            # 5. 根据温度参数将优势转换为样本权重；
            # 6. 计算奖励加权 HDP hybrid diffusion loss；
            # 7. 反向传播并裁剪梯度；
            # 8. 更新模型和 EMA 参数。
            metrics = update_epoch(
                data_loader,
                model,
                optimizer,
                ema,
                replay_buffer,
                args,
                device,
            )

            # 每个 update epoch 后更新一次学习率。
            #
            # rollout epoch 不进行模型参数更新，
            # 因此也不推进学习率调度器。
            scheduler.step()

            # 标记当前阶段名称。
            phase = "update"

        # 只有主进程负责打印、记录日志和保存模型。
        if global_rank == 0:
            # 输出当前 epoch 的阶段和统计指标。
            print(f"Epoch {epoch + 1}: phase={phase}, metrics={metrics}")

            # 将指标写入 TensorBoard 或 WandB。
            #
            # 例如：
            #
            # rl_rollout/reward_mean
            # rl_rollout/collision_rate
            #
            # 或：
            #
            # rl_update/loss
            # rl_update/advantage_mean
            logger.log_metrics(
                {f"rl_{phase}/{key}": value for key, value in metrics.items()},
                step=epoch + 1,
            )

            # 仅在 update 阶段保存模型。
            #
            # rollout 阶段没有更新模型参数，因此没有保存必要。
            if (
                not is_rollout_epoch
                and (epoch + 1) % args.save_utd == 0
            ):
                # 保存：
                # 1. 当前模型参数；
                # 2. 优化器状态；
                # 3. 学习率调度器状态；
                # 4. 当前 epoch；
                # 5. 当前损失；
                # 6. 日志运行 ID；
                # 7. EMA 模型参数。
                save_model(
                    model,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    metrics["loss"],
                    logger.id,
                    ema.ema,
                )

    # 训练完成后关闭日志记录器并刷新剩余日志。
    logger.finish()


# 当该 Python 文件被直接运行时，启动参数解析和模型训练。
#
# 如果该文件只是被其他模块 import，
# 则不会自动开始训练。
if __name__ == "__main__":
    model_training(get_args())
