# 导入类型注解工具。
#
# Dict[str, float]：
# 表示键为字符串、值为浮点数的字典，主要用于保存训练指标。
#
# Iterable：
# 表示任意可迭代对象。
#
# List：
# 表示列表类型。
from typing import Dict, Iterable, List

# NumPy 用于计算多个 batch 指标的平均值。
import numpy as np

# PyTorch 核心库，用于张量运算、自动求导和模型训练。
import torch

# clip_grad_norm_ 用于执行梯度范数裁剪。
#
# 假设所有可训练参数的梯度整体范数为：
#
# ||g||_2
#
# 当：
#
# ||g||_2 > max_norm
#
# 时，会按照比例缩放所有梯度，使缩放后的整体范数不超过 max_norm。
#
# 其主要作用是抑制奖励加权产生的异常大梯度，提高训练稳定性。
from torch.nn.utils import clip_grad_norm_

# default_collate 是 PyTorch DataLoader 默认使用的 batch 拼接函数。
#
# 它可以把多个单场景样本：
#
# sample_1, sample_2, ..., sample_B
#
# 沿 batch 维度组合为一个批次。
#
# 当前代码从 Replay Buffer 中取出场景名称后，
# 需要重新到 Dataset 中加载对应场景，再使用该函数拼成 batch。
from torch.utils.data._utils.collate import default_collate

# tqdm 用于显示 rollout 和 update 阶段的进度条。
from tqdm import tqdm

# reward_weighted_diffusion_loss：
# 负责计算奖励加权的扩散训练损失。
#
# 从当前函数传入的参数可以确定，它会综合使用：
#
# 1. 当前扩散模型；
# 2. 场景输入；
# 3. Replay Buffer 中保存的候选轨迹；
# 4. 每条候选轨迹的奖励；
# 5. 扩散 SDE；
# 6. 扩散预测目标和监督空间；
# 7. HDP 的速度—航路点混合损失；
# 8. Detached Integral 时间窗口；
# 9. 奖励温度；
# 10. 优势裁剪阈值。
#
# 具体的奖励标准化、奖励权重公式和扩散损失计算，
# 由 hdp_nuplan.rl.loss 中的实现决定。
from hdp_nuplan.rl.loss import reward_weighted_diffusion_loss
from hdp_nuplan.rl.trajectory_augmentation import augment_trajectory_batch
from hdp_nuplan.loss import diffusion_loss_func

# 项目封装的分布式训练工具。
#
# 当前文件主要使用 ddp.get_model：
#
# 1. 在普通单卡模式下，直接返回模型本身；
# 2. 在 DDP 模式下，从 DDP 包装器中取出 model.module。
from hdp_nuplan.utils import ddp


# 把两个训练目标的组合单独封装，便于做“只用 rollout / 只用 expert anchor”控制实验。
def combine_update_losses(
    rl_loss,
    expert_anchor_loss,
    rollout_loss_weight,
    expert_anchor_weight,
    reference_anchor_loss=None,
    reference_anchor_weight=0.0,
):
    """按显式权重组合 rollout、expert 和 reference 三类损失。"""

    if (
        rollout_loss_weight < 0
        or expert_anchor_weight < 0
        or reference_anchor_weight < 0
    ):
        raise ValueError("update loss weights must be non-negative")
    total = (
        rollout_loss_weight * rl_loss
        + expert_anchor_weight * expert_anchor_loss
    )
    if reference_anchor_loss is not None:
        total = total + reference_anchor_weight * reference_anchor_loss
    return total


# 对多个 batch 记录的指标求平均。
#
# 输入示例：
#
# records = [
#     {
#         "loss": 1.2,
#         "reward_mean": 0.5,
#     },
#     {
#         "loss": 0.8,
#         "reward_mean": 0.7,
#     },
# ]
#
# 输出为：
#
# {
#     "loss": 1.0,
#     "reward_mean": 0.6,
# }
def _mean_metrics(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    # grouped 用于按照指标名称收集所有 batch 的数值。
    #
    # 例如：
    #
    # grouped = {
    #     "loss": [1.2, 0.8],
    #     "reward_mean": [0.5, 0.7],
    # }
    grouped: Dict[str, List[float]] = {}

    # 遍历每一个 batch 的指标字典。
    for record in records:
        # 遍历该 batch 中的每一个指标。
        for key, value in record.items():
            # 如果 grouped 中还不存在该指标，
            # setdefault 会先创建一个空列表。
            #
            # 随后把当前指标值转换为 Python float 并加入列表。
            grouped.setdefault(key, []).append(float(value))

    # 对每种指标的所有 batch 数值求平均。
    #
    # np.mean(values) 返回 NumPy 标量，
    # 再用 float 转换为普通 Python 浮点数，
    # 便于日志记录和 JSON 序列化。
    return {key: float(np.mean(values)) for key, values in grouped.items()}


# 将 DiffusionPlannerData 返回的原始 batch 整理为：
#
# 1. 经过归一化、可直接输入模型的 model_inputs；
# 2. 未归一化的 raw_inputs，供物理奖励计算使用；
# 3. 自车未来真实轨迹 ego_future；
# 4. 邻车未来真实轨迹 neighbors_future；
# 5. 邻车未来状态的无效位置掩码 neighbor_mask；
# 6. 场景名称 scene_names。
def prepare_nuplan_batch(batch, device, observation_normalizer, with_metadata=True):
    """把 Dataset batch 转成模型输入、物理空间监督和场景标识。"""

    # 当 with_metadata=True 时，Dataset 返回的最后一项是场景名称。
    #
    # batch[-1] 通常是长度为 B 的字符串序列：
    #
    # [
    #     scene_name_1,
    #     scene_name_2,
    #     ...,
    #     scene_name_B,
    # ]
    #
    # list(...) 将其转换为普通 Python 列表。
    scene_names = list(batch[-1]) if with_metadata else None

    # 如果包含元数据，则去掉 batch 最后一项，只保留张量。
    #
    # 如果不包含元数据，则整个 batch 都是张量数据。
    tensors = batch[:-1] if with_metadata else batch

    # 根据 DiffusionPlannerData 的固定字段顺序组织模型输入。
    #
    # 这些输入仍处在原始物理表示中，
    # 随后会交给 observation_normalizer 进行归一化。
    inputs = {
        # tensors[0]：
        # 当前自车状态。
        "ego_current_state": tensors[0].to(device),

        # tensors[2]：
        # 周围动态交通参与者的历史状态。
        "neighbor_agents_past": tensors[2].to(device),

        # tensors[4]：
        # 普通车道几何与属性。
        "lanes": tensors[4].to(device),

        # tensors[5]：
        # 普通车道的限速值。
        "lanes_speed_limit": tensors[5].to(device),

        # tensors[6]：
        # 标记对应车道是否存在有效限速。
        "lanes_has_speed_limit": tensors[6].to(device),

        # tensors[7]：
        # 导航路线对应的车道信息。
        "route_lanes": tensors[7].to(device),

        # tensors[8]：
        # 导航路线车道的限速值。
        "route_lanes_speed_limit": tensors[8].to(device),

        # tensors[9]：
        # 标记导航路线车道是否存在有效限速。
        "route_lanes_has_speed_limit": tensors[9].to(device),

        # tensors[10]：
        # 静态障碍物信息。
        "static_objects": tensors[10].to(device),
    }

    # tensors[1] 是自车未来真实轨迹。
    #
    # 从后续代码可以看出，其最后一个维度至少包含：
    #
    # [
    #     x,
    #     y,
    #     heading,
    # ]
    #
    # 这里保留原始物理空间表示，
    # 尚未转换航向角表示。
    ego_future_raw = tensors[1].to(device)

    # tensors[3] 是邻车未来真实轨迹。
    #
    # 其最后一个维度同样至少包含：
    #
    # [
    #     x,
    #     y,
    #     heading,
    # ]
    neighbors_future_raw = tensors[3].to(device)

    # 构造邻车未来状态的无效位置掩码。
    #
    # torch.ne(neighbors_future_raw[..., :3], 0)：
    # 检查每个未来状态的 x、y、heading 是否不等于 0。
    #
    # 假设某个未来状态为：
    #
    # [0, 0, 0]
    #
    # 则 torch.ne 的结果为：
    #
    # [False, False, False]
    #
    # 转换为数值后求和为 0。
    #
    # 如果某个状态至少有一个分量非零，求和就大于 0。
    neighbor_mask = torch.sum(
        torch.ne(neighbors_future_raw[..., :3], 0),

        # 只在状态特征维度上求和，
        # 保留 batch、邻车和未来时间维度。
        dim=-1,
    ) == 0

    # 将自车未来轨迹从：
    #
    # [x, y, heading]
    #
    # 转换为：
    #
    # [x, y, cos(heading), sin(heading)]
    #
    # 使用 cos 和 sin 表示航向角，可以避免角度在 -pi 和 pi
    # 附近产生不连续跳变。
    ego_future = torch.cat(
        [
            # 保留未来位置坐标 x、y。
            ego_future_raw[..., :2],

            # 将 heading 转换为单位方向向量。
            torch.stack(
                [ego_future_raw[..., 2].cos(), ego_future_raw[..., 2].sin()],

                # 在最后新建的维度上堆叠：
                #
                # [..., 2]
                #
                # 对应 [cos(heading), sin(heading)]。
                dim=-1,
            ),
        ],

        # 沿最后的状态特征维度拼接，
        # 得到 [..., 4]。
        dim=-1,
    )

    # 对邻车未来轨迹执行相同的航向角转换。
    neighbors_future = torch.cat(
        [
            # 邻车未来位置 x、y。
            neighbors_future_raw[..., :2],

            # 邻车未来航向角的 cos 和 sin 表示。
            torch.stack(
                [
                    neighbors_future_raw[..., 2].cos(),
                    neighbors_future_raw[..., 2].sin(),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    # 对原始数据中使用 [0,0,0] 填充的无效邻车未来状态，
    # 将转换后的四维状态重新设置为全 0。
    #
    # 这是必要的，因为：
    #
    # cos(0) = 1
    #
    # 如果不重新清零，无效填充值 [0,0,0] 会被转换成：
    #
    # [0,0,1,0]
    #
    # 从而被错误地识别为有效轨迹状态。
    neighbors_future[neighbor_mask] = 0

    # 对模型场景输入执行观测归一化。
    #
    # 归一化后的 model_inputs 用于神经网络前向计算。
    #
    # 原始 inputs 仍被保留，
    # 因为路线、静态障碍物等物理奖励通常需要真实尺度数据。
    model_inputs = observation_normalizer(inputs)

    # 返回模型输入、物理空间输入、监督轨迹、掩码和场景名称。
    return (
        model_inputs,
        inputs,
        ego_future,
        neighbors_future,
        neighbor_mask,
        scene_names,
    )


# rollout 阶段只负责采样轨迹和计算奖励，不更新模型参数。
#
# @torch.no_grad() 会关闭自动求导图构建，从而：
#
# 1. 减少显存占用；
# 2. 加快轨迹采样；
# 3. 防止 rollout 阶段误执行梯度更新。
@torch.no_grad()
def rollout_epoch(
    data_loader,
    model,
    replay_buffer,
    reward_scorer,
    args,
    device,
    epoch=0,
    reference_model=None,
):
    """对每个 NuPlan 场景分组采样并写入 Replay Buffer。"""

    # 如果 model 是 DDP 包装器，则取出内部真实模型 model.module。
    #
    # sample 通常是 Hyper_Diffusion_Planner 自定义的方法，
    # DDP 包装对象本身不一定直接暴露该方法，
    # 因此需要先获取基础模型。
    base_model = ddp.get_model(model, args.ddp)

    # 保存每个 batch 的 rollout 统计指标。
    records = []

    # 创建带进度条的数据迭代器。
    progress = tqdm(data_loader, desc="NuPlan RL rollout", unit="batch")

    # 遍历数据集中的所有场景 batch。
    for batch in progress:
        # 把原始 Dataset batch 转换成模型输入和奖励计算输入。
        (
            model_inputs,
            raw_inputs,
            ego_future,
            neighbors_future,
            neighbor_mask,
            scene_names,
        ) = prepare_nuplan_batch(
            batch,
            device,
            args.observation_normalizer,

            # rollout 阶段需要场景名称，
            # 因为 Replay Buffer 使用场景名称重新加载场景。
            with_metadata=True,
        )

        # 对同一个场景采样 rl_group_size 条候选轨迹。
        #
        # 如果 batch size 为 B，rl_group_size 为 K，
        # 则 trajectories 通常按以下逻辑组织：
        #
        # [
        #     场景1的K条轨迹,
        #     场景2的K条轨迹,
        #     ...,
        #     场景B的K条轨迹,
        # ]
        #
        # 即逻辑形状通常包含：
        #
        # [B, K, future_len, trajectory_dim]
        #
        # 具体张量维度由 base_model.sample 的实现决定。
        trajectories = base_model.sample(
            model_inputs,

            # 每个场景的候选轨迹数量 K。
            num_samples=args.rl_group_size,

            # 每条轨迹使用的反向扩散去噪步数。
            diffusion_steps=args.rl_rollout_steps,

            # 仅控制 RL 候选的初始噪声尺度；普通 planner 推理仍使用 sample() 默认 0.1。
            noise_scale=args.rl_sampling_noise_scale,
        )

        # 【迁移自 HDP-NAVSIM】前若干个 epoch 在每条候选自身的局部坐标系中，
        # 给整条轨迹加入共享的纵向/横向高斯偏移。它直接扩大候选几何差异，
        # 航向 cos/sin 保持不变；epoch 从 0 开始，所以默认 epoch < 5 生效。
        if (
            epoch < args.rl_trajectory_augmentation_epochs
            and args.rl_trajectory_augmentation_std > 0
        ):
            trajectories = augment_trajectory_batch(
                trajectories,
                std=args.rl_trajectory_augmentation_std,
            )

        # 为每条候选轨迹计算奖励。
        #
        # 奖励器接收：
        #
        # 1. 当前模型生成的自车候选轨迹；
        # 2. 邻车未来真实轨迹；
        # 3. 邻车未来有效性掩码；
        # 4. 导航路线；
        # 5. 静态障碍物；
        # 6. 自车未来专家轨迹。
        #
        # rewards：
        # 每个场景、每条候选轨迹的总奖励。
        #
        # details：
        # progress、collision、route、comfort 等分项奖励。
        # rewards : [B,G]
        # details : Dict[str, Tensor]，每个 Tensor 形状为 [B,G]
        # details = {
        #     "reward": rewards,
        #     "progress": progress,
        #     "no_collision": no_collision,
        #     "collision_cost": collision_cost,
        #     "route_cost": route_cost,
        #     "comfort_cost": comfort_cost,
        #     "backward_cost": backward_cost,
        #     "imitation_cost": imitation_cost,
        # }
        rewards, details = reward_scorer(
            trajectories=trajectories,
            neighbors_future=neighbors_future,
            neighbor_mask=neighbor_mask,

            # 奖励计算使用未归一化的物理路线数据。
            route_lanes=raw_inputs["route_lanes"],

            # 奖励计算使用未归一化的静态障碍物数据。
            static_objects=raw_inputs["static_objects"],

            # 可用于计算 imitation reward 或其他专家轨迹相关指标。
            ego_future=ego_future,

            # reward v2 使用邻车当前宽/长构造尺寸感知的 OBB 碰撞近似。
            neighbor_agents_past=raw_inputs["neighbor_agents_past"],

            # reward v2 使用当前速度/加速度补齐 comfort 有限差分的初始边界。
            ego_current_state=raw_inputs["ego_current_state"],

            # 【论文 HDP-RL】lane reward 使用全部邻近车道中心线及左右边界，
            # route_lanes 只继续用于 progress/route 退化诊断。
            lanes=raw_inputs["lanes"],
        )

        # 可选的 reference-relative reward：冻结参考策略通常是 B Epoch10。
        # 它只生成每个场景一条确定性候选，不写入 replay；其 reward 作为
        # 当前 G 条候选的 baseline，update 时只学习真正超过参考策略的候选。
        reference_rewards = None
        if getattr(args, "rl_relative_to_reference", False):
            if reference_model is None:
                raise RuntimeError(
                    "rl_relative_to_reference=true requires reference_model"
                )
            reference_trajectories = reference_model.sample(
                model_inputs,
                num_samples=1,
                diffusion_steps=args.rl_rollout_steps,
                noise_scale=args.rl_reference_noise_scale,
            )
            reference_rewards, _ = reward_scorer(
                trajectories=reference_trajectories,
                neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask,
                route_lanes=raw_inputs["route_lanes"],
                static_objects=raw_inputs["static_objects"],
                ego_future=ego_future,
                neighbor_agents_past=raw_inputs["neighbor_agents_past"],
                ego_current_state=raw_inputs["ego_current_state"],
                lanes=raw_inputs["lanes"],
            )
            # reward scorer 返回 [B,1]；ReplayItem 每个场景保存一个标量。
            reference_rewards = reference_rewards.squeeze(1)

        # 按场景把候选轨迹组和奖励组写入 Replay Buffer。
        #
        # zip 每次取出：
        #
        # scene_name：
        # 当前单个场景名称。
        #
        # scene_trajectories：
        # 当前场景对应的 K 条候选轨迹。
        #
        # scene_rewards：
        # 当前场景 K 条候选轨迹对应的 K 个奖励。
        # 把候选级安全资格和轨迹、奖励一起存入 Replay Buffer。旧实现只保存
        # 门控后的标量 reward，更新阶段无法区分“安全候选”和“被硬门降分的
        # 不安全候选”，因而 centered 权重仍可能对不安全候选产生负回归梯度。
        safety_candidate_masks = details["safety_gate_eligible"].to(torch.bool)
        candidate_masks = safety_candidate_masks
        if getattr(args, "rl_filter_progress_guard_candidates", False):
            # progress_guard_reward 是候选相对专家路线进度的 [0, 1] 质量分。
            # 这里取 safety mask 与 progress mask 的交集，避免把“没有碰撞但
            # 明显停滞/落后”的候选作为 rollout 目标。整组无候选时，update
            # 阶段会跳过该组 rollout loss，但仍保留 expert anchor。
            progress_guard_reward = details["progress_guard_reward"]
            progress_candidate_masks = progress_guard_reward >= float(
                getattr(args, "rl_min_progress_guard_reward", 0.9)
            )
            candidate_masks = safety_candidate_masks & progress_candidate_masks
        for (
            scene_name,
            scene_trajectories,
            scene_rewards,
            scene_candidate_mask,
            scene_reference_reward,
        ) in zip(
            scene_names,
            trajectories,
            rewards,
            candidate_masks,
            reference_rewards
            if reference_rewards is not None
            else [None] * len(scene_names),
        ):
            # Replay Buffer 保存：
            #
            # 场景名称 + 候选轨迹组 + 奖励组。
            #
            # 后续 update 阶段会根据 scene_name 重新加载场景输入，
            # 再用 scene_trajectories 和 scene_rewards 训练模型。
            replay_buffer.put(
                scene_name,
                scene_trajectories,
                scene_rewards,
                candidate_mask=scene_candidate_mask,
                reference_reward=scene_reference_reward,
            )

        # 对当前 batch 的每个奖励分项取平均。
        #
        # 例如 details 可能包含：
        #
        # {
        #     "total": Tensor,
        #     "progress": Tensor,
        #     "collision": Tensor,
        #     "route": Tensor,
        #     "comfort": Tensor,
        # }
        #
        # 转换后得到：
        #
        # {
        #     "reward/total": float,
        #     "reward/progress": float,
        #     ...
        # }
        record = {
            f"reward/{key}": value.mean().item()
            for key, value in details.items()
        }
        if getattr(args, "rl_filter_progress_guard_candidates", False):
            record["reward/progress_guard_eligible"] = (
                candidate_masks.float().mean().item()
            )
            record["reward/progress_guard_has_eligible"] = (
                candidate_masks.any(dim=1).float().mean().item()
            )

        # 记录处理完当前 batch 后 Replay Buffer 的即时容量。
        record["buffer_size"] = len(replay_buffer)

        # 保存该 batch 的指标。
        records.append(record)

        # 在进度条右侧动态显示：
        #
        # 1. 当前 batch 所有候选轨迹的平均奖励；
        # 2. 当前 Replay Buffer 大小。
        progress.set_postfix(
            reward=f"{rewards.mean().item():.3f}",
            buffer=len(replay_buffer),
        )

    # 对各个 batch 的指标求平均。
    metrics = _mean_metrics(records)

    # buffer_size 是一个 epoch 结束时的状态量，不应对每个 batch 后的
    # 递增容量求平均；否则 8, 16, ..., 1000 会被误报为 504。
    #
    # 例如各 batch 处理后的 Buffer 容量依次为：
    #
    # 8, 16, 24, ..., 1000
    #
    # 对这些状态值求平均无法表示最终 Buffer 大小。
    #
    # 因此这里用 epoch 结束后的实际长度覆盖平均结果。
    metrics["buffer_size"] = float(len(replay_buffer))

    # 返回整个 rollout epoch 的统计指标。
    return metrics


# 根据 Replay Buffer 中保存的场景名称，
# 重新从 Dataset 加载完整场景数据并组织训练 batch。
def _load_replay_batch(dataset, replay_items):
    # 遍历本次从 Replay Buffer 中抽出的所有条目。
    #
    # 每个 item 至少包含：
    #
    # 1. item.scene_name；
    # 2. item.trajectories；
    # 3. item.rewards。
    #
    # 根据 scene_name 到 Dataset 中重新获取原始场景张量。
    scene_tensors = [
        dataset.get_by_name(item.scene_name)
        for item in replay_items
    ]

    # 把多个单场景样本拼接为标准 batch。
    batch = default_collate(scene_tensors)

    # 把每个场景保存的候选轨迹组沿第 0 维堆叠。
    #
    # 假设每个 item.trajectories 的逻辑形状为：
    #
    # [K, T, D]
    #
    # 堆叠后 trajectories 的逻辑形状为：
    #
    # [B, K, T, D]
    trajectories = torch.stack(
        [item.trajectories for item in replay_items],
        dim=0,
    )

    # 把每个场景对应的 K 个奖励沿第 0 维堆叠。
    #
    # 假设单个 item.rewards 形状为：
    #
    # [K]
    #
    # 堆叠后 rewards 形状为：
    #
    # [B, K]
    rewards = torch.stack(
        [item.rewards for item in replay_items],
        dim=0,
    )

    # 旧 ReplayItem 没有候选掩码；为兼容已有调用，将其视为全部候选可用。
    candidate_masks = torch.stack(
        [
            item.candidate_mask
            if item.candidate_mask is not None
            else torch.ones_like(item.rewards, dtype=torch.bool)
            for item in replay_items
        ],
        dim=0,
    )

    # 返回重新加载的场景 batch、候选轨迹、奖励和候选级安全资格。
    reference_rewards = torch.stack(
        [
            item.reference_reward
            if item.reference_reward is not None
            else torch.tensor(float("nan"), dtype=item.rewards.dtype)
            for item in replay_items
        ],
        dim=0,
    )

    # 返回参考策略 reward；旧 replay 条目用 NaN 标记，调用方会在显式
    # reference-relative 模式下拒绝混用，避免静默退化为错误 baseline。
    return batch, trajectories, rewards, candidate_masks, reference_rewards


# 从 Replay Buffer 中采样数据，
# 使用奖励加权扩散损失执行一个完整的模型更新 epoch。
def update_epoch(
    data_loader,
    model,
    optimizer,
    ema,
    replay_buffer,
    args,
    device,
    reference_model=None,
):
    """从 Replay Buffer 采样，执行 reward-weighted diffusion 更新。"""

    # Update 阶段必须依赖前面 rollout 阶段生成的数据。
    #
    # 如果 Buffer 为空，则无法获得：
    #
    # 1. 当前策略生成的候选轨迹；
    # 2. 候选轨迹对应的奖励；
    # 3. 组内相对优劣信息。
    if len(replay_buffer) == 0:
        raise RuntimeError("Replay Buffer 为空，请先执行 rollout epoch")

    # Reward-weighted regression 的 target 来自 eval 模式旧策略。如果 update
    # 再启用 Dropout/DropPath，自蒸馏会混入与 reward 无关的随机删层梯度并造成
    # 系统漂移。eval() 只切换随机/统计层，不会关闭 autograd，因此默认在确定性
    # 模式下更新；保留开关便于复现实验和做消融。
    if getattr(args, "rl_deterministic_update", True):
        model.eval()
    else:
        model.train()

    # 保存每个 update iteration 的训练指标。
    records = []

    # 获取 DataLoader 对应的 Dataset。
    #
    # 后续通过 dataset.get_by_name(scene_name)
    # 重新读取 Replay Buffer 条目对应的场景。
    dataset = data_loader.dataset

    # 创建 update 进度条。
    progress = tqdm(data_loader, desc="NuPlan RL update", unit="batch")

    # 此处遍历 data_loader 的主要作用，是：
    #
    # 1. 决定一个 update epoch 有多少次更新迭代；
    # 2. 从 loader_batch 中读取当前 batch size。
    #
    # 实际用于模型训练的场景不是 loader_batch 本身，
    # 而是随后从 Replay Buffer 中重新采样的 replay_items。
    for update_step, loader_batch in enumerate(progress):
        max_steps = getattr(args, "rl_max_update_steps_per_epoch", 0)
        if max_steps > 0 and update_step >= max_steps:
            break
        # 获取当前 DataLoader batch 的样本数量。
        #
        # 最后一个 batch 可能小于配置的标准 batch size，
        # 因此不能始终直接使用 args.batch_size。
        batch_size = loader_batch[0].shape[0]

        # 从 Replay Buffer 中随机抽取 batch_size 个场景条目。
        #
        # 每个条目保存一个场景对应的：
        #
        # 1. 场景名称；
        # 2. K 条候选轨迹；
        # 3. K 个奖励。
        # loader_batch：决定本次更新抽多少个场景
        #                ↓
        # replay_buffer：真正随机提供4个训练场景
        replay_items = replay_buffer.sample(batch_size)

        # 根据场景名称重新加载场景输入，
        # 并堆叠对应的候选轨迹和奖励。
        (
            replay_batch,
            trajectories,
            rewards,
            candidate_masks,
            reference_rewards,
        ) = _load_replay_batch(
            dataset,
            replay_items,
        )

        # 把重新加载的场景 batch 转换为模型输入。
        #
        # update 阶段不再需要场景名称，因此：
        #
        # with_metadata=False
        (
            model_inputs,
            _,
            ego_future,
            neighbors_future,
            neighbor_mask,
            _,
        ) = prepare_nuplan_batch(
            replay_batch,
            device,
            args.observation_normalizer,
            with_metadata=False,
        )

        # 把候选轨迹移动到当前训练设备。
        trajectories = trajectories.to(device)

        # 把候选轨迹奖励移动到当前训练设备。
        rewards = rewards.to(device)
        reference_rewards = reference_rewards.to(device)

        # 保守更新显式开启时，只允许通过 rollout 安全门的候选成为回归目标。
        # 整组无合格候选时，rollout loss 对该场景为 0，后面的专家 anchor
        # 仍正常训练，避免“在一组不安全动作中挑一个相对没那么差的动作”。
        candidate_masks = candidate_masks.to(device)
        rollout_candidate_mask = (
            candidate_masks
            if args.rl_filter_safety_eligible_candidates
            else None
        )

        # 清空上一轮反向传播留下的梯度。
        #
        # set_to_none=True 不会把梯度张量填充为 0，
        # 而是直接将 parameter.grad 设置为 None。
        #
        # 相比 zero_grad() 的普通置零方式，
        # 这种方法通常可以减少内存写入并略微提高性能。
        optimizer.zero_grad(set_to_none=True)

        # 从 DDP 包装器中取出基础模型。
        #
        # 此处主要用于读取基础模型中的 sde 属性。
        #
        # 真正计算 loss 时仍把 model 传入，
        # 这样在 DDP 模式下反向传播可以正常同步梯度。
        base_model = ddp.get_model(model, args.ddp)

        # 计算奖励加权扩散损失。
        #
        # 从接口可以看出，该函数会综合执行以下过程：
        #
        # 1. 使用 trajectories 作为当前旧策略生成的数据样本；
        # 2. 使用 rewards 评价同一场景下各候选轨迹的相对质量；
        # 3. 根据 reward_temperature 将奖励信息转换为训练权重；
        # 4. 根据 advantage_clip 限制极端相对优势；
        # 5. 随机选择扩散时间并对轨迹加噪；
        # 6. 调用模型预测指定扩散变量；
        # 7. 在 supervision_type 对应的空间中计算监督损失；
        # 8. 使用 hybrid_loss_weight 加入积分航路点监督；
        # 9. 使用 detach_window_size 限制积分梯度传播范围；
        # 10. 对不同候选轨迹的扩散损失进行奖励加权。
        #
        # 奖励加权扩散训练的一般思想是：
        #
        # L_RL
        # =
        # E[
        #     w(s, tau)
        #     *
        #     L_diffusion(tau)
        # ]
        #
        # 其中：
        #
        # w(s, tau)
        #
        # 随轨迹奖励或相对优势增大而增大。
        #
        # 具体的权重公式必须以 reward_weighted_diffusion_loss
        # 函数内部实现为准。
        total_loss, metrics = reward_weighted_diffusion_loss(
            # 传入 DDP 包装后的模型，
            # 保证多卡训练时梯度可以同步。
            model=model,

            # 当前场景的归一化模型输入。
            inputs=model_inputs,

            # Replay Buffer 中当前策略生成的候选轨迹。
            trajectories=trajectories,

            # 每条候选轨迹的奖励。
            rewards=rewards,

            # 轨迹状态归一化器。
            #
            # 扩散模型通常在归一化轨迹空间中训练，
            # 而 Replay Buffer 中可能保存物理空间轨迹。
            state_normalizer=args.state_normalizer,

            # HDP 模型使用的随机微分方程和噪声调度。
            sde=base_model.sde,

            # 模型直接预测的扩散变量类型：
            #
            # v、x_start、noise 或 score。
            model_type=args.diffusion_model_type,

            # 最终计算监督损失的变量空间。
            supervision_type=args.diffusion_supervision_type,

            # HDP 混合损失中积分航路点损失的权重 omega。
            hybrid_loss_weight=args.planning_hybrid_loss,

            # Detached Integral 的时间窗口 W。
            detach_window_size=args.rl_detach_window_size,

            # 奖励或优势转换为样本权重时使用的温度参数。
            reward_temperature=args.rl_reward_temperature,

            # 组内优势或归一化奖励的裁剪阈值。
            advantage_clip=args.rl_advantage_clip,

            # reward 差异小于阈值的组不参与 rollout 自蒸馏。
            min_reward_std=args.rl_min_reward_std,

            # 有效组的权重均值固定为 1，避免隐式放大学习率。
            normalize_weights=args.rl_normalize_weights,

            # 可选 control-variate：只保留 reward-dependent 梯度。
            center_reward_weights=args.rl_center_reward_weights,

            # 新的 NuPlan 稳定目标使用正的 softmax advantage 权重；旧实验
            # 默认仍使用指数权重，便于复现实验和做消融。
            weighting_mode=args.rl_weighting_mode,

            # 将候选安全资格传入损失；None 保持旧实验完全兼容。
            candidate_mask=rollout_candidate_mask,

            # 可选的 B Epoch 10 reference policy 约束；默认 None 保持旧实验
            # 数值兼容，新 v7 实验通过显式权重开启。
            reference_model=reference_model,

            # 只有显式开启 reference-relative 模式时才把冻结策略 reward
            # 传给 advantage；旧实验保持组内均值 baseline。
            reference_rewards=(
                reference_rewards
                if args.rl_relative_to_reference
                else None
            ),
        )

        # 使用同一批场景的真实 ego future 作为监督 anchor，抑制仅拟合旧策略
        # rollout 样本导致的自蒸馏漂移。该损失与原监督训练入口使用同一实现。
        if args.rl_expert_anchor_weight > 0:
            anchor_loss_dict, _ = diffusion_loss_func(
                model,
                model_inputs,
                base_model.sde,
                (ego_future, neighbors_future, neighbor_mask),
                args.state_normalizer,
                {},
                args.diffusion_model_type,
                args.diffusion_supervision_type,
                detach_window_size=args.rl_detach_window_size,
            )
            expert_anchor_loss = (
                anchor_loss_dict["ego_planning_loss"]
                + args.planning_hybrid_loss
                * anchor_loss_dict["ego_planning_hybrid_loss"]
            )
        else:
            expert_anchor_loss = total_loss.new_zeros(())
        rl_loss = total_loss
        total_loss = combine_update_losses(
            rl_loss,
            expert_anchor_loss,
            args.rl_rollout_loss_weight,
            args.rl_expert_anchor_weight,
            metrics.get("reference_anchor_loss"),
            args.rl_reference_anchor_weight,
        )
        metrics["rl_loss"] = rl_loss
        metrics["expert_anchor_loss"] = expert_anchor_loss
        metrics["weighted_rl_loss"] = args.rl_rollout_loss_weight * rl_loss
        metrics["weighted_expert_anchor_loss"] = (
            args.rl_expert_anchor_weight * expert_anchor_loss
        )
        metrics["reference_anchor_loss"] = metrics.get(
            "reference_anchor_loss", total_loss.new_zeros(())
        )
        metrics["weighted_reference_anchor_loss"] = (
            args.rl_reference_anchor_weight * metrics["reference_anchor_loss"]
        )
        metrics["loss"] = total_loss

        # 对总损失执行反向传播。
        #
        # PyTorch 会计算所有 requires_grad=True 参数的梯度：
        #
        # d total_loss / d theta
        total_loss.backward()

        # 对所有可训练参数执行全局梯度范数裁剪。
        #
        # 冻结参数 requires_grad=False，
        # 不会被加入梯度裁剪参数列表。
        clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],

            # 最大允许梯度范数。
            args.rl_grad_clip,
        )

        # 根据裁剪后的梯度更新模型参数。
        optimizer.step()

        # 使用更新后的当前模型参数更新 EMA 模型。
        ema.update(model)

        # 将 reward_weighted_diffusion_loss 返回的指标
        # 从 Tensor 转换为普通 Python float。
        #
        # detach()：
        # 将指标从计算图中分离。
        #
        # item()：
        # 将单元素 Tensor 转换为 Python 标量。
        record = {
            key: value.detach().item()
            for key, value in metrics.items()
        }

        # 当前 update 阶段不会修改 Replay Buffer，
        # 因此该值通常在整个 epoch 中保持不变。
        record["buffer_size"] = len(replay_buffer)

        # 保存当前迭代指标。
        records.append(record)

        # 在进度条右侧显示：
        #
        # 1. 当前总损失；
        # 2. 当前采样 batch 的平均奖励。
        progress.set_postfix(
            loss=f"{total_loss.item():.4f}",
            reward=f"{metrics['reward_mean'].item():.3f}",
        )

    # 对一个 update epoch 内所有训练迭代的指标求平均并返回。
    summary = _mean_metrics(records)
    summary["update_steps"] = len(records)
    return summary
