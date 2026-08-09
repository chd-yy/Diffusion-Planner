# 导入类型注解。
#
# Dict：
# 表示 Python 字典。
#
# Tuple：
# 表示函数返回由多个对象组成的固定结构元组。
from typing import Dict, Tuple

# PyTorch：
# 用于张量计算、随机采样、自动求导以及扩散训练。
import torch

# torch.nn.functional：
# PyTorch 中的函数式神经网络接口。
#
# 当前代码使用：
#
# F.mse_loss(...)
#
# 来计算积分后 waypoint 的均方误差。
import torch.nn.functional as F

# 导入 HDP 中的 Detached Integral。
#
# detached_integral 的核心思想是：
#
# 前向数值：
#
# x_t
# =
# sum_{i=1}^{t} u_i
#
# 与普通累积积分完全一致。
#
# 但是反向传播时，只允许最近 W 个时间步传播梯度：
#
# x_t
# =
# sg(
#     sum_{i=1}^{t-W}u_i
# )
# +
# sum_{i=t-W+1}^{t}u_i
#
# 其中 sg 表示 stop-gradient。
#
# 这样可以缓解长时间积分导致的梯度累积问题。
from hdp_nuplan.utils.traj_kinematics import detached_integral


# ----------------------------------------------------------------------
# 将 waypoint 轨迹转换成 HDP 模型实际训练的 action 表示
# ----------------------------------------------------------------------
#
# 输入 trajectories 的逻辑形状：
#
# [B,T,4]
#
# 其中每个时间步为：
#
# [x_t,y_t,cos(theta_t),sin(theta_t)]
#
# 当前函数并不是直接把 [x,y] 当作扩散变量，
# 而是把绝对位置转换成相邻时间步的位置增量：
#
# Δx_t
# =
# x_t-x_{t-1}
#
# Δy_t
# =
# y_t-y_{t-1}
#
# 对第一个时间步，因为当前自车局部坐标原点被认为是：
#
# x_0=0
#
# y_0=0
#
# 所以：
#
# Δx_1=x_1
#
# Δy_1=y_1
#
# 最终 action 表示为：
#
# [Δx_t,Δy_t,cos(theta_t),sin(theta_t)]
#
# 注意：
#
# 这里严格来说是“逐帧位移 displacement”，
# 并不是除以 Δt 后的物理速度。
#
# 也就是说当前代码使用：
#
# displacement
# =
# p_t-p_{t-1}
#
# 而不是：
#
# velocity
# =
# (p_t-p_{t-1})/Δt
#
# 后面的 detached_integral 也是直接通过累加 displacement
# 恢复 waypoint，因此这里的实现内部是自洽的。
def waypoint_to_model_action(trajectories: torch.Tensor) -> torch.Tensor:
    """将 [x, y, cos, sin] 轨迹转换成 HDP 训练使用的逐帧位移表示。"""

    # 首先把轨迹第一个位置之前补一个局部坐标原点：
    #
    # [0,0]
    #
    # trajectories[:, :1, :2] 的形状：
    #
    # [B,1,2]
    #
    # torch.zeros_like(...)：
    #
    # 创建相同 shape、dtype、device 的全 0 张量。
    #
    # 假设原始位置序列：
    #
    # p_1,p_2,...,p_T
    #
    # 拼接后得到：
    #
    # 0,p_1,p_2,...,p_T
    #
    # 然后 torch.diff(...,dim=-2) 计算：
    #
    # p_1-0
    # p_2-p_1
    # ...
    # p_T-p_{T-1}
    #
    # 所以：
    #
    # displacement_t
    # =
    # p_t-p_{t-1}
    #
    # 输出 displacement 的形状仍然是：
    #
    # [B,T,2]
    displacement = torch.diff(
        torch.cat(
            [torch.zeros_like(trajectories[:, :1, :2]), trajectories[..., :2]],
            dim=-2,
        ),
        dim=-2,
    )

    # 将二维位移：
    #
    # [Δx,Δy]
    #
    # 与原始轨迹中的：
    #
    # [cos(theta),sin(theta)]
    #
    # 拼接起来。
    #
    # 最终模型 action：
    #
    # a_t
    # =
    # [Δx_t,Δy_t,cos(theta_t),sin(theta_t)]
    #
    # 输出形状仍然为：
    #
    # [B,T,4]
    return torch.cat([displacement, trajectories[..., 2:]], dim=-1)


# ----------------------------------------------------------------------
# 根据同一场景中 G 条候选轨迹的奖励计算组内 Advantage
# ----------------------------------------------------------------------
#
# 输入：
#
# rewards.shape
# =
# [B,G]
#
# 其中：
#
# B：
# batch 中的场景数量。
#
# G：
# 每个场景生成的候选轨迹数量。
#
# rewards[b,g]：
# 第 b 个场景、第 g 条候选轨迹的绝对奖励。
#
# 这里不会直接使用绝对 reward，
# 而是先在“同一个场景内部”进行标准化。
#
# 对第 b 个场景：
#
# μ_b
# =
# (1/G) sum_g r_{b,g}
#
# σ_b
# =
# sqrt(
#     (1/G) sum_g (r_{b,g}-μ_b)^2
# )
#
# Advantage：
#
# A_{b,g}
# =
# (r_{b,g}-μ_b)/(σ_b+1e-6)
#
# 然后进行裁剪：
#
# A_{b,g}
# =
# clip(A_{b,g},-c,c)
#
# 最终转换成奖励加权回归权重：
#
# w_{b,g}
# =
# exp(
#     temperature * A_{b,g}
# )
#
# 因此：
#
# A>0
# ->
# w>1
# ->
# 高于组内平均水平的轨迹得到更大的训练权重。
#
# A=0
# ->
# w=1
#
# A<0
# ->
# 0<w<1
# ->
# 对 reward 方差达到阈值的有效组，低于组内平均水平的轨迹仍参与训练，
# 但损失权重更小；低方差组整体跳过 rollout 自蒸馏。
def group_advantage_weights(
    rewards: torch.Tensor,
    temperature: float = 1.0,
    clip: float = 5.0,
    min_reward_std: float = 0.01,
    normalize_weights: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成稳定的组内权重；低方差组返回全零权重并跳过自蒸馏。"""

    # rewards 必须是二维矩阵：
    #
    # [B,G]
    #
    # 如果维度不是 2，说明上游 Replay Buffer
    # 或 reward scorer 的数据结构不符合预期。
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [B, G]")
    if min_reward_std < 0:
        raise ValueError("min_reward_std must be non-negative")

    # ------------------------------------------------------------------
    # 对每个场景的 G 条轨迹计算平均奖励
    # ------------------------------------------------------------------
    #
    # rewards.shape：
    #
    # [B,G]
    #
    # dim=1：
    #
    # 沿候选轨迹维 G 求平均。
    #
    # keepdim=True：
    #
    # 保留这一维，因此：
    #
    # mean.shape=[B,1]
    #
    # 对场景 b：
    #
    # mean[b,0]
    # =
    # (1/G)sum_g rewards[b,g]
    # 保存每个场景内部 G 条候选轨迹的平均奖励。
    mean = rewards.mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # 计算同一场景内部奖励的标准差
    # ------------------------------------------------------------------
    #
    # std.shape：
    #
    # [B,1]
    #
    # unbiased=False 表示使用总体标准差：
    #
    # σ
    # =
    # sqrt(
    #     (1/G)sum_g(r_g-μ)^2
    # )
    #
    # 而不是样本标准差：
    #
    # sqrt(
    #     1/(G-1) * ...
    # )
    #
    # 在这里 G 条候选轨迹本身就是当前组内全部样本，
    # 因此使用总体标准差。
    std = rewards.std(dim=1, keepdim=True, unbiased=False)

    # ------------------------------------------------------------------
    # 计算组内标准化优势
    # ------------------------------------------------------------------
    #
    # rewards：
    #
    # [B,G]
    #
    # mean/std：
    #
    # [B,1]
    #
    # PyTorch 自动广播到 [B,G]。
    #
    # 对每条轨迹：
    #
    # A_{b,g}
    # =
    # (r_{b,g}-μ_b)/(σ_b+1e-6)
    #
    # 加 1e-6 是为了防止：
    #
    # σ_b=0
    #
    # 时发生除零。
    #
    # 如果同一场景 G 条轨迹奖励完全一样：
    #
    # r_1=...=r_G
    #
    # 那么：
    #
    # r_g-μ=0
    #
    # 所以所有 advantage 都是 0。
    # advantages  [B,G],advantages[b,g]表示第 b 个场景中，第 g 条候选轨迹相对于该场景候选平均奖励的优势
    active_groups = std >= min_reward_std
    advantages = (rewards - mean) / (std + 1e-6)
    advantages = torch.where(active_groups, advantages, torch.zeros_like(advantages))

    # 对极端 advantage 进行裁剪：
    #
    # A
    # ->
    # clip(A,-clip,clip)
    #
    # 默认：
    #
    # clip=5
    #
    # 因此：
    #
    # -5<=A<=5
    #
    # 防止极端奖励经过指数函数后产生过大的权重。
    advantages = advantages.clamp(min=-clip, max=clip)

    # ------------------------------------------------------------------
    # Advantage -> reward weight
    # ------------------------------------------------------------------
    #
    # 当前代码使用：
    #
    # w
    # =
    # exp(
    #     temperature * A
    # )
    #
    # 默认 temperature=1：
    #
    # w=exp(A)
    #
    # 例如：
    #
    # A=2
    # ->
    # w≈7.39
    #
    # A=0
    # ->
    # w=1
    #
    # A=-2
    # ->
    # w≈0.135
    #
    # 注意当前代码中的 temperature 是直接乘在 A 上的。
    #
    # 因此 temperature 越大：
    #
    # 高低奖励轨迹权重差距越大。
    #
    # temperature 越小：
    #
    # 所有样本权重越接近 1。
    # 它把候选轨迹的相对奖励转换为“模型应该多大程度拟合该轨迹”
    # 【HDP-NAVSIM 主线】所有有效候选都参与训练，组内优势通过指数函数
    # 转换为相对权重；高 reward 候选权重大，低 reward 候选权重小。
    weights = torch.exp(temperature * advantages)

    # 有效组按组归一化到均值 1，避免 exp 的 Jensen 偏差改变有效学习率。
    if normalize_weights:
        weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-12)

    # 低方差组的候选差异主要是数值/采样噪声，不再用 rollout 轨迹自蒸馏。
    weights = torch.where(active_groups, weights, torch.zeros_like(weights))

    # 返回：
    #
    # advantages.shape=[B,G]
    #
    # weights.shape=[B,G]
    return advantages, weights


# ----------------------------------------------------------------------
# HDP 的奖励加权扩散训练损失
# ----------------------------------------------------------------------
#
# 这是整个 RL update 的核心。
#
# 输入：
#
# trajectories：
#
# [B,G,T,4]
#
# 表示旧策略 rollout 得到的候选轨迹。
#
# rewards：
#
# [B,G]
#
# 表示对应轨迹的奖励。
#
# 整体过程可以概括为：
#
# rollout waypoint
#       ↓
# 转换成 displacement action
#       ↓
# state normalization
#       ↓
# 随机选择 diffusion time t
#       ↓
# 前向加噪得到 x_t
#       ↓
# 模型预测
#       ↓
# 转换到 supervision space
#       ↓
# 计算每条候选轨迹的 diffusion loss
#       ↓
# reward -> advantage -> weight
#       ↓
# weighted diffusion loss
#       +
# weighted waypoint integral loss
#       ↓
# total RL loss
#
# 最终形式可以理解为：
#
# L_total
# =
# E[
#     w_{b,g} L_diff^{b,g}
# ]
# +
# λ_hybrid
# E[
#     w_{b,g} L_waypoint^{b,g}
# ]
def reward_weighted_diffusion_loss(
    model,
    inputs: Dict[str, torch.Tensor],
    trajectories: torch.Tensor,
    rewards: torch.Tensor,
    state_normalizer,
    sde,
    model_type: str,
    supervision_type: str,
    hybrid_loss_weight: float,
    detach_window_size: int = 10,
    reward_temperature: float = 1.0,
    advantage_clip: float = 5.0,
    min_reward_std: float = 0.01,
    normalize_weights: bool = True,
    center_reward_weights: bool = False,
    eps: float = 1e-3,
):
    """计算 NuPlan 分组奖励加权的 HDP 扩散损失。

    Args:
        trajectories: [B, G, T, 4] 的物理空间候选轨迹。
        rewards: [B, G] 的绝对场景奖励。
    """

    # 从轨迹张量读取四个维度：
    #
    # B：
    # batch size。
    #
    # G：
    # 每个场景候选轨迹数量。
    #
    # T：
    # 轨迹 horizon。
    #
    # D：
    # state dimension，当前通常为 4。
    batch_size, group_size, horizon, state_dim = trajectories.shape

    # ------------------------------------------------------------------
    # 将场景维 B 和候选组维 G 展平
    # ------------------------------------------------------------------
    #
    # 原来：
    #
    # trajectories.shape
    # =
    # [B,G,T,D]
    #
    # 变为：
    #
    # [B*G,T,D]
    #
    # 因为扩散模型训练时可以把每一条候选轨迹
    # 当成一个独立训练样本。
    #
    # 例如：
    #
    # B=4
    # G=8
    #
    # 则：
    #
    # B*G=32
    #
    # 相当于当前扩散训练 batch 中有 32 条轨迹。
    flat_trajectories = trajectories.reshape(
        batch_size * group_size, horizon, state_dim
    )

    # ------------------------------------------------------------------
    # waypoint -> displacement action
    # ------------------------------------------------------------------
    #
    # 原始轨迹：
    #
    # [x_t,y_t,cos(theta_t),sin(theta_t)]
    #
    # 转换为：
    #
    # [Δx_t,Δy_t,cos(theta_t),sin(theta_t)]
    #
    # shape：
    #
    # [B*G,T,4]
    model_actions = waypoint_to_model_action(flat_trajectories)

    # ------------------------------------------------------------------
    # 对 action 做状态归一化
    # ------------------------------------------------------------------
    #
    # 扩散模型通常不会直接在原始物理量尺度上训练，
    # 而是在标准化后的空间学习。
    #
    # 假设状态归一化形式类似：
    #
    # x_norm
    # =
    # (x-μ)/σ
    #
    # 那么：
    #
    # all_gt
    #
    # 就是当前扩散模型真正使用的 clean data x_0。
    #
    # 也就是说这里：
    #
    # x_0
    # =
    # normalized displacement action
    all_gt = state_normalizer(model_actions)

    # ------------------------------------------------------------------
    # 随机采样 diffusion time
    # ------------------------------------------------------------------
    #
    # 首先：
    #
    # torch.rand(...)
    #
    # 产生：
    #
    # u~Uniform(0,1)
    #
    # 然后：
    #
    # t
    # =
    # u*(1-eps)+eps
    #
    # 所以：
    #
    # t∈[eps,1]
    #
    # 默认：
    #
    # eps=0.001
    #
    # 这样避免 t=0 附近可能出现的数值问题。
    #
    # 每一条候选轨迹独立采样一个 t：
    #
    # t.shape=[B*G]
    t = torch.rand(
        batch_size * group_size,
        device=all_gt.device,
        dtype=all_gt.dtype,
    ) * (1 - eps) + eps

    # 为每条 clean trajectory 采样与其完全相同 shape 的高斯噪声：
    #
    # ε~N(0,I)
    #
    # noise.shape：
    #
    # [B*G,T,D]
    noise = torch.randn_like(all_gt)

    # ------------------------------------------------------------------
    # 前向扩散
    # ------------------------------------------------------------------
    #
    # sde.marginal_prob(x_0,t)
    #
    # 返回前向扩散边缘分布：
    #
    # q_t(x_t|x_0)
    #
    # 的均值和标准差。
    #
    # 一般形式为：
    #
    # q_t(x_t|x_0)
    # =
    # N(
    #     α_t x_0,
    #     σ_t^2 I
    # )
    #
    # 所以通常：
    #
    # mean
    # =
    # α_t x_0
    #
    # std
    # =
    # σ_t
    #
    # 具体 α_t、σ_t 的实现由 sde 决定。
    mean, std = sde.marginal_prob(all_gt, t)

    # std 通常只沿 batch 维保存：
    #
    # [B*G]
    #
    # 为了能够与：
    #
    # noise.shape=[B*G,T,D]
    #
    # 广播相乘，需要 reshape 为：
    #
    # [B*G,1,1]
    #
    # 如果 all_gt 有更多维度，
    # 这里也会自动插入对应数量的 1。
    std_view = std.reshape(-1, *([1] * (all_gt.ndim - 1)))

    # 根据高斯重参数化公式采样带噪数据：
    #
    # x_t
    # =
    # mean
    # +
    # std * ε
    #
    # 也就是一般形式：
    #
    # x_t
    # =
    # α_t x_0
    # +
    # σ_t ε
    #
    # x_t.shape：
    #
    # [B*G,T,D]
    x_t = mean + std_view * noise

    # ------------------------------------------------------------------
    # 构造 v-prediction 的监督目标
    # ------------------------------------------------------------------
    #
    # 当前系统允许：
    #
    # model_type:
    #
    # x_start / noise / v / score
    #
    # supervision_type:
    #
    # x_start / noise / v / score
    #
    # 因此即使当前 supervision 不是 v，
    # 也提前计算出 v_target。
    #
    # sde.transform("noise->v", ...)
    #
    # 根据当前 SDE 参数化关系，
    # 将真实噪声 ε 转换成对应 diffusion velocity target。
    #
    # 具体 v 的数学表达式由 sde.transform 的实现决定，
    # 当前文件本身没有给出，所以不能仅根据这里进一步确定。
    v_target = sde.transform("noise->v", noise, t, x_t)

    # ------------------------------------------------------------------
    # 将原本每个场景一份的 condition 扩展成每条候选轨迹一份
    # ------------------------------------------------------------------
    #
    # inputs 中每个张量原本第一维：
    #
    # B
    #
    # 但是现在扩散轨迹已经展平成：
    #
    # B*G
    #
    # 因此每个场景条件必须重复 G 次。
    #
    # 假设：
    #
    # inputs["ego_current_state"]
    # =
    # [
    #     scene_A,
    #     scene_B
    # ]
    #
    # G=3
    #
    # repeat_interleave 后：
    #
    # [
    #     scene_A,
    #     scene_A,
    #     scene_A,
    #     scene_B,
    #     scene_B,
    #     scene_B
    # ]
    #
    # 从而与 flat_trajectories 的展平顺序一一对应。
    repeated_inputs = {
        key: value.repeat_interleave(group_size, dim=0)
        for key, value in inputs.items()
    }

    # 将当前带噪轨迹 x_t 放入模型输入。
    #
    # 扩散 decoder 将根据：
    #
    # scene condition
    # +
    # x_t
    # +
    # diffusion time
    #
    # 预测相应扩散变量。
    repeated_inputs["sampled_trajectories"] = x_t # [B*G,T,D]

    # 将每个样本对应的扩散时刻 t 放入模型输入。
    repeated_inputs["diffusion_time"] = t # [B*G]

    # ------------------------------------------------------------------
    # HDP 前向传播
    # ------------------------------------------------------------------
    #
    # model(...) 返回两个结果。
    #
    # 第一个结果当前没有使用，所以用 _ 接收。
    #
    # decoder_output：
    # 扩散 decoder 的输出字典。
    _, decoder_output = model(repeated_inputs)

    # 取 decoder 输出中的 "score" 字段。
    #
    # 注意：
    #
    # 这个字段名叫 "score"，并不意味着当前模型一定使用 score prediction。
    #
    # 从后面的代码看，它更像是 decoder 原始输出的统一字段名。
    #
    # 其实际语义由：
    #
    # model_type
    #
    # 决定。
    #
    # 例如：
    #
    # model_type="x_start"
    #
    # 那么 model_prediction 实际被解释为 x_start prediction。
    model_prediction = decoder_output["score"]

    # ------------------------------------------------------------------
    # 把模型原始输出转换到指定 supervision space
    # ------------------------------------------------------------------
    #
    # 例如：
    #
    # model_type="x_start"
    # supervision_type="noise"
    #
    # 那么：
    #
    # x_start prediction
    # ->
    # noise prediction
    #
    # 再在 noise space 中计算 loss。
    #
    # 当前默认配置通常是：
    #
    # x_start
    # ->
    # x_start
    #
    # 即直接预测 clean data，
    # 直接在 clean-data space 监督。
    prediction = sde.transform(
        f"{model_type}->{supervision_type}",
        model_prediction,
        t,
        x_t,
    )

    # ------------------------------------------------------------------
    # 根据 supervision_type 计算扩散监督损失
    # ------------------------------------------------------------------

    # 如果监督目标是 score。
    if supervision_type == "score":

        # 对标准 VP/VE 扩散中的 score：
        #
        # score target
        # ≈
        # -ε/σ_t
        #
        # 因此理想情况下：
        #
        # prediction
        # =
        # -noise/std
        #
        # 等价于：
        #
        # prediction*std + noise
        # =
        # 0
        #
        # 所以这里计算：
        #
        # ||σ_t s_theta + ε||^2
        #
        # 最后：
        #
        # torch.sum(...,dim=-1)
        #
        # 对状态维 D 求和。
        #
        # 因此结果逻辑 shape：
        #
        # [B*G,T]
        per_step_loss = torch.sum(
            (prediction * std_view + noise) ** 2,
            dim=-1,
        )

    # 如果监督目标是 clean data x_0。
    elif supervision_type == "x_start":

        # 直接计算：
        #
        # L_t
        # =
        # ||x_hat_0-x_0||_2^2
        #
        # all_gt 就是 normalized clean action。
        #
        # dim=-1：
        # 对状态维 D 求和: [16,80,4] → [16,80]
        per_step_loss = torch.sum((prediction - all_gt) ** 2, dim=-1)

    # 如果监督目标是 Gaussian noise。
    elif supervision_type == "noise":

        # 计算：
        #
        # L_t
        # =
        # ||ε_hat-ε||_2^2
        per_step_loss = torch.sum((prediction - noise) ** 2, dim=-1)

    # 如果监督目标是 diffusion velocity。
    elif supervision_type == "v":

        # 计算：
        #
        # L_t
        # =
        # ||v_hat-v_target||_2^2
        per_step_loss = torch.sum((prediction - v_target) ** 2, dim=-1)

    # 如果传入未知监督类型，则立即报错。
    else:
        raise ValueError(f"unknown supervision type: {supervision_type}")

    # ------------------------------------------------------------------
    # 每条完整轨迹得到一个 diffusion loss
    # ------------------------------------------------------------------
    #
    # per_step_loss：
    #
    # [B*G,T]
    #
    # mean(dim=-1)：
    #
    # 对 T 个未来时间步求平均：
    #
    # L_diff^{sample}
    # =
    # (1/T)
    # sum_t L_t
    #
    # 得到：
    #
    # [B*G]
    #
    # reshape(B,G) 后：
    #
    # [B,G]
    #
    # 因此：
    #
    # per_sample_diffusion[b,g]
    #
    # 就是第 b 个场景中第 g 条候选轨迹
    # 对应的普通 diffusion regression loss。
    per_sample_diffusion = per_step_loss.mean(dim=-1).reshape(
        batch_size, group_size
    )

    # ------------------------------------------------------------------
    # reward -> group advantage -> sample weight
    # ------------------------------------------------------------------
    #
    # 输入：
    #
    # rewards.shape=[B,G]
    #
    # 得到：
    #
    # advantages.shape=[B,G]
    #
    # weights.shape=[B,G]
    #
    # 数学形式：
    #
    # A_{b,g}
    # =
    # (r_{b,g}-μ_b)/(σ_b+eps)
    #
    # A
    # ->
    # clip(A,-c,c)
    #
    # w_{b,g}
    # =
    # exp(
    #     temperature*A_{b,g}
    # )
    advantages, weights = group_advantage_weights(
        rewards,
        temperature=reward_temperature,
        clip=advantage_clip,
        min_reward_std=min_reward_std,
        normalize_weights=normalize_weights,
    )

    # 【NuPlan 小数据适配，非论文原式】论文直接使用正权重 w。在有限 replay
    # 上，w 的常数 1 分量会产生明显的无权自蒸馏漂移；减去该 baseline 后，
    # beta=0 时梯度严格为 0，只保留 reward 与回归梯度的协方差信号。
    # 低方差无效组仍保持 0，避免 0-1 变成负权重。
    reward_std = rewards.std(dim=1, unbiased=False)
    active_groups = reward_std >= min_reward_std
    regression_weights = weights
    if center_reward_weights:
        regression_weights = torch.where(
            active_groups[:, None],
            weights - 1.0,
            torch.zeros_like(weights),
        )

    # ------------------------------------------------------------------
    # Reward-weighted diffusion loss
    # ------------------------------------------------------------------
    #
    # per_sample_diffusion：
    #
    # [B,G]
    #
    # weights：
    #
    # [B,G]
    #
    # 对应元素相乘：
    #
    # w_{b,g}
    # *
    # L_diff^{b,g}
    #
    # 最后对所有 B*G 条候选轨迹求平均：
    #
    # L_diffusion
    # =
    # (1/(B G))
    # sum_b sum_g
    # w_{b,g}
    # L_diff^{b,g}
    #
    # 因此：
    #
    # 高奖励轨迹：
    # w 较大
    # ->
    # 模型更强地拟合这条轨迹。
    #
    # 低奖励轨迹：
    # w 较小
    # ->
    # 仍然参与训练，但影响变弱。
    # diffusion_loss 是带计算图的标量 Tensor
    diffusion_loss = (regression_weights * per_sample_diffusion).mean()

    # ------------------------------------------------------------------
    # HDP Hybrid Waypoint Loss
    # ------------------------------------------------------------------
    #
    # 如果：
    #
    # hybrid_loss_weight>0
    #
    # 则除了 displacement/action diffusion loss，
    # 还要把预测 action 积分成 waypoint，
    # 再与原始 waypoint 进行监督。
    if hybrid_loss_weight > 0:

        # 无论当前模型直接预测：
        #
        # noise / score / v / x_start
        #
        # 都统一把模型输出转换成：
        #
        # x_start
        #
        # 因为只有 clean action 才适合转换回物理 displacement。
        predicted_action = sde.transform(
            f"{model_type}->x_start",
            model_prediction,
            t,
            x_t,
        )

        # ------------------------------------------------------------------
        # 从归一化 action 恢复到物理 action
        # ------------------------------------------------------------------
        #
        # predicted_action 当前仍在 normalizer 的归一化空间。
        #
        # inverse 后恢复：
        #
        # [Δx,Δy,cos(theta),sin(theta)]
        #
        # 的物理尺度表示。
        predicted_action = state_normalizer.inverse(predicted_action)

        # ------------------------------------------------------------------
        # 对预测 displacement 做 Detached Integral
        # ------------------------------------------------------------------
        #
        # predicted_action[..., :2]：
        #
        # [Δx_t,Δy_t]
        #
        # detached_integral：
        #
        # x_t
        # =
        # sum_{i=1}^{t}Δx_i
        #
        # y_t
        # =
        # sum_{i=1}^{t}Δy_i
        #
        # 从而恢复出预测 waypoint：
        #
        # predicted_xy
        #
        # shape：
        #
        # [B*G,T,2]
        #
        # detach_window_size：
        #
        # 控制积分反向传播最多回传多少个历史时间步。
        #
        # max(detach_window_size,1)：
        #
        # 确保窗口至少是 1。
        predicted_xy = detached_integral(
            predicted_action[..., :2],
            detach_window_size=max(detach_window_size, 1),
        )

        # ------------------------------------------------------------------
        # Waypoint reconstruction loss
        # ------------------------------------------------------------------
        #
        # predicted_xy：
        #
        # 积分恢复出来的位置。
        #
        # flat_trajectories[..., :2]：
        #
        # rollout 保存的原始 absolute waypoint：
        #
        # [x,y]
        #
        # F.mse_loss(...,reduction="none")：
        #
        # 每个元素分别计算：
        #
        # (x_hat-x)^2
        #
        # (y_hat-y)^2
        #
        # 得到 shape：
        #
        # [B*G,T,2]
        #
        # mean(dim=(-1,-2))：
        #
        # 同时对：
        #
        # 1. xy 两个维度；
        # 2. T 个未来时间步
        #
        # 求平均。
        #
        # 因此每条轨迹得到一个 waypoint MSE：
        #
        # L_wp^{sample}
        # =
        # (1/(2T))
        # sum_t
        # [
        #     (x_hat_t-x_t)^2
        #     +
        #     (y_hat_t-y_t)^2
        # ]
        #
        # reshape(B,G)：
        #
        # 恢复成每个场景、每条候选轨迹一个 loss。
        per_sample_waypoint = F.mse_loss(
            predicted_xy,
            flat_trajectories[..., :2],
            reduction="none",
        ).mean(dim=(-1, -2)).reshape(batch_size, group_size)

        # ------------------------------------------------------------------
        # Reward-weighted waypoint loss
        # ------------------------------------------------------------------
        #
        # 与 diffusion loss 使用完全相同的 reward weights：
        #
        # L_waypoint
        # =
        # (1/(BG))
        # sum_b sum_g
        # w_{b,g}
        # L_wp^{b,g}
        #
        # 所以高奖励轨迹不仅在 action/displacement 空间
        # 得到更强监督，
        # 在积分后的 waypoint 空间中也获得更强监督。
        waypoint_loss = (regression_weights * per_sample_waypoint).mean()

    # 如果不启用 hybrid waypoint loss。
    else:

        # 创建一个与 diffusion_loss：
        #
        # 1. device 相同；
        # 2. dtype 相同；
        #
        # 的标量 0 Tensor。
        waypoint_loss = diffusion_loss.new_zeros(())

    # ------------------------------------------------------------------
    # 最终总损失
    # ------------------------------------------------------------------
    #
    # 当前实现：
    #
    # L_total
    # =
    # L_diffusion
    # +
    # λ_hybrid L_waypoint
    #
    # 其中：
    #
    # L_diffusion
    # =
    # E[
    #     w L_diff
    # ]
    #
    # L_waypoint
    # =
    # E[
    #     w L_waypoint
    # ]
    #
    # 因此也可以写成：
    #
    # L_total
    # =
    # E[
    #     w(
    #         L_diff
    #         +
    #         λ_hybrid L_waypoint
    #     )
    # ]
    #
    # 这就是当前代码中的 reward-weighted hybrid diffusion loss。
    total_loss = diffusion_loss + hybrid_loss_weight * waypoint_loss

    # ------------------------------------------------------------------
    # 记录训练监控指标
    # ------------------------------------------------------------------
    active_candidate_count = (
        active_groups.sum() * group_size
    ).clamp_min(1)

    metrics = {

        # 最终反向传播使用的总 loss：
        #
        # L_diffusion
        # +
        # λ L_waypoint
        "loss": total_loss,

        # 奖励加权后的基础 diffusion regression loss。
        "reward_weighted_diffusion_loss": diffusion_loss,

        # 奖励加权后的 waypoint reconstruction loss。
        #
        # 注意这里记录的是 waypoint_loss 本身，
        # 还没有再次乘 hybrid_loss_weight。
        "reward_weighted_waypoint_loss": waypoint_loss,

        # 当前 replay batch 中所有：
        #
        # B*G
        #
        # 条候选轨迹的平均绝对奖励。
        "reward_mean": rewards.mean(),

        # 对每个场景先取：
        #
        # max_g r_{b,g}
        #
        # 得到该场景最好候选轨迹的奖励，
        # 然后再在 B 个场景之间求平均：
        #
        # reward_max
        # =
        # (1/B)
        # sum_b max_g r_{b,g}
        "reward_max": rewards.max(dim=1).values.mean(),

        # 对每个场景先取：
        #
        # min_g r_{b,g}
        #
        # 再对场景平均。
        #
        # 表示每个场景中最差候选轨迹奖励的平均水平。
        "reward_min": rewards.min(dim=1).values.mean(),

        # 所有组内标准化 advantage 的平均值。
        #
        # 理论上标准化前的组内 advantage 均值接近 0。
        #
        # 但因为后面执行了 clamp，
        # 最终 advantage_mean 不一定严格等于 0。
        "advantage_mean": advantages.mean(),

        # 所有场景、所有候选的平均权重。normalize_weights=True 时，有效组
        # 内部均值为 1、低方差组为 0，所以该值近似等于 active_group_fraction。
        "weight_mean": weights.mean(),
        "regression_weight_mean": regression_weights.mean(),

        # v2 稳定性门控：监控有多少场景具有足够大的组内 reward 差异。
        "reward_std_mean": reward_std.mean(),
        "active_group_fraction": active_groups.float().mean(),
        "active_weight_mean": weights.sum() / active_candidate_count,
    }

    # 返回：
    #
    # total_loss：
    # 用于：
    #
    # total_loss.backward()
    #
    # metrics：
    # 用于训练日志和进度监控。
    return total_loss, metrics
