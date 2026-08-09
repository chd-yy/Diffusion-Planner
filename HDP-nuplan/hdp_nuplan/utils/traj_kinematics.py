# 导入 PyTorch。
#
# 当前函数主要使用以下三个 PyTorch 操作：
#
# 1. Tensor.detach()：
#    保留张量的前向数值，但切断该张量与上游计算图之间的梯度联系。
#
# 2. torch.cumsum()：
#    沿指定维度计算累积和，用离散累加近似速度积分。
#
# 3. torch.roll()：
#    沿指定维度循环平移张量，用于构造向前错开 W 个时间步的前缀和。
import torch


# detached_integral 用于执行“前向完整积分、反向截断梯度”的累积求和。
#
# 假设输入序列为：
#
# u_0,u_1,u_2,...,u_{T-1}
#
# 普通完整累积积分为：
#
# x_t
# =
# sum_{i=0}^{t}u_i
#
# 普通 torch.cumsum 会使第 t 个输出 x_t 对所有历史输入
# u_0,...,u_t 都传播梯度。
#
# 因此：
#
# ∂x_t/∂u_i
# =
# 1, 当 i<=t
#
# 对长时间序列而言，早期输入会接收来自大量未来输出的梯度，
# 容易造成时间维度上的梯度不均衡。
#
# 当前函数保留完整累积和的前向数值，但只允许最近 W 个输入
# 对当前累积结果传播梯度。
#
# 令：
#
# W=detach_window_size
#
# 则希望得到：
#
# x_t
# =
# sg(
#     sum_{i=0}^{t-W}u_i
# )
# +
# sum_{i=max(0,t-W+1)}^{t}u_i
#
# 其中 sg 表示 stop-gradient：
#
# sg(z)=z
#
# 但：
#
# ∂sg(z)/∂z=0
#
# 因而该表达式在数值上仍然等于：
#
# x_t
# =
# sum_{i=0}^{t}u_i
#
# 但其梯度只会传播到最近 W 个时间步。
#
# 这与截断时间反向传播 Truncated BPTT 的思想相似。
def detached_integral(u, detach_window_size):
    # HDP 新增的截断梯度积分：数值上仍累计完整轨迹，但只让最近窗口的积分项
    # 反向传播，降低长时域累积造成的梯度爆炸和显存开销。
    # u: (B, T=80, D)

    # ------------------------------------------------------------------
    # 第一部分：计算一份完全切断梯度的完整累积和
    # ------------------------------------------------------------------
    #
    # u.detach() 的前向数值与 u 完全相同：
    #
    # u.detach()[t]=u[t]
    #
    # 但它不再向 u 传播梯度：
    #
    # ∂u.detach()/∂u=0
    #
    # torch.cumsum(..., dim=-2) 沿倒数第二维累积求和。
    #
    # 如果 u 的形状确实为：
    #
    # (B,T,D)
    #
    # 那么 dim=-2 对应时间维 T。
    #
    # 对于每个 batch 和每个特征维度，有：
    #
    # cum_detach[:,t,:]
    # =
    # u[:,0,:]
    # +
    # u[:,1,:]
    # +
    # ...
    # +
    # u[:,t,:]
    #
    # 这份累积和只用于提供“更早历史”的前向数值，
    # 不会把梯度传播回对应的历史输入。
    cum_detach = torch.cumsum(u.detach(), dim=-2)

    # ------------------------------------------------------------------
    # 第二部分：计算一份保留正常梯度的完整累积和
    # ------------------------------------------------------------------
    #
    # cum_normal 与 cum_detach 的前向数值完全相同：
    #
    # cum_normal[t]=cum_detach[t]
    #
    # 但 cum_normal 保留完整自动求导关系。
    #
    # 因而普通情况下：
    #
    # ∂cum_normal[t]/∂u[i]
    # =
    # 1, 当 i<=t
    #
    # 这份累积和随后用于构造“最近 W 个时间步”的可导累积部分。
    cum_normal = torch.cumsum(u, dim=-2)

    # number of gradient from previous timesteps contained in: 
    # shifted: [0, 1, 2, ..., window_size-1, window_size, ...., T] ->
    # shifted: [T-window_size+1, T-window_size+2, ...,T, 0, 1, 2, ...., T - window_size] ->
    # sum_recent: [0, 1, 2, ..., window_size-1, window_size, ...., window_size]

    # ------------------------------------------------------------------
    # 第三部分：将正常累积和沿时间维向后移动 W 个位置
    # ------------------------------------------------------------------
    #
    # 令：
    #
    # W=detach_window_size
    #
    # 原始累积和为：
    #
    # C_t
    # =
    # sum_{i=0}^{t}u_i
    #
    # torch.roll(...,shifts=W,dims=-2) 会把时间序列循环右移 W 位。
    #
    # 右移后的逻辑效果为：
    #
    # shifted[t]
    # =
    # C_{t-W}, 当 t>=W
    #
    # 但 torch.roll 是循环移动，因此前 W 个位置会被序列末尾的值填充。
    #
    # 例如：
    #
    # 原序列：
    #
    # [C_0,C_1,C_2,C_3,C_4]
    #
    # 当 W=2 时，roll 后为：
    #
    # [C_3,C_4,C_0,C_1,C_2]
    #
    # 前两个元素 C_3、C_4 是循环回来的无效值，
    # 因此下一行需要把这些位置设置为 0。
    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)

    # 将平移后前 W 个时间位置设置为 0。
    #
    # 理想目标是构造：
    #
    # shifted[t]
    # =
    # 0, 当 t<W
    #
    # shifted[t]
    # =
    # C_{t-W}, 当 t>=W
    #
    # 使用省略号保留任意数量的前导 batch/候选维，并明确将倒数第二维
    # 作为时间维，因此同时兼容 (B,T,D) 和 (B,K,T,D)。
    shifted[..., :detach_window_size, :] = 0

    # ------------------------------------------------------------------
    # 第四部分：利用两个前缀和之差得到最近 W 项的累积和
    # ------------------------------------------------------------------
    #
    # 当 t<W 时：
    #
    # shifted[t]=0
    #
    # 因而：
    #
    # sum_recent[t]
    # =
    # C_t
    # =
    # sum_{i=0}^{t}u_i
    #
    # 当 t>=W 时：
    #
    # shifted[t]=C_{t-W}
    #
    # 因而：
    #
    # sum_recent[t]
    # =
    # C_t-C_{t-W}
    #
    # =
    # sum_{i=t-W+1}^{t}u_i
    #
    # 所以 sum_recent 最多只包含最近 W 个输入项。
    #
    # 更重要的是，sum_recent 由 cum_normal 构造，
    # 因此这些最近 W 项都保留正常梯度。
    #
    # 其梯度为：
    #
    # ∂sum_recent[t]/∂u_i
    # =
    # 1,
    # 当 max(0,t-W+1)<=i<=t
    #
    # 其余更早输入的梯度为 0。
    sum_recent = cum_normal - shifted
        
    # ------------------------------------------------------------------
    # 第五部分：构造更早历史的无梯度前缀
    # ------------------------------------------------------------------
    #
    # cum_detach 是完整前缀和，但已经通过 detach 切断梯度。
    #
    # 将它向后平移 W 个时间步后，期望得到：
    #
    # cum_detach_shifted[t]
    # =
    # 0, 当 t<W
    #
    # cum_detach_shifted[t]
    # =
    # sg(C_{t-W}), 当 t>=W
    #
    # 它表示当前时刻最近 W 项以前的历史累积值。
    #
    # 由于它来自 cum_detach，因此只参与前向数值计算，
    # 不会向早期的 u 传播梯度。
    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)

    # 与 shifted 相同，torch.roll 会把序列末尾内容循环到前面，
    # 所以前 W 个位置需要清零。
    #
    # 与 shifted 使用相同的通用时间维切片，清除 torch.roll 循环回填到
    # 前 W 个时间位置的无效前缀。
    cum_detach_shifted[..., :detach_window_size, :] = 0
        
    # ------------------------------------------------------------------
    # 第六部分：组合无梯度的早期历史和有梯度的最近窗口
    # ------------------------------------------------------------------
    #
    # 当 t<W 时：
    #
    # cum_detach_shifted[t]=0
    #
    # sum_recent[t]=sum_{i=0}^{t}u_i
    #
    # 所以：
    #
    # cumulative_sum[t]
    # =
    # sum_{i=0}^{t}u_i
    #
    # 此时所有已有时间步都正常传播梯度。
    #
    # 当 t>=W 时：
    #
    # cum_detach_shifted[t]
    # =
    # sg(
    #     sum_{i=0}^{t-W}u_i
    # )
    #
    # sum_recent[t]
    # =
    # sum_{i=t-W+1}^{t}u_i
    #
    # 因而：
    #
    # cumulative_sum[t]
    # =
    # sg(
    #     sum_{i=0}^{t-W}u_i
    # )
    # +
    # sum_{i=t-W+1}^{t}u_i
    #
    # 从前向数值上看：
    #
    # cumulative_sum[t]
    # =
    # sum_{i=0}^{t}u_i
    #
    # 与普通完整 cumsum 完全相同。
    #
    # 但从反向梯度看，只有最近 W 项参与反向传播：
    #
    # ∂cumulative_sum[t]/∂u_i
    # =
    # 1, 当 t-W+1<=i<=t
    #
    # ∂cumulative_sum[t]/∂u_i
    # =
    # 0, 当 i<=t-W
    cumulative_sum = cum_detach_shifted + sum_recent

    # 返回截断梯度后的完整累积和。
    #
    # 返回值与 torch.cumsum(u,dim=-2) 在前向数值上应一致，
    # 差别仅在反向传播路径。
    #
    # 举例说明，假设：
    #
    # W=3
    #
    # 输入为：
    #
    # [u_0,u_1,u_2,u_3,u_4]
    #
    # 返回值的前向形式为：
    #
    # x_0=u_0
    #
    # x_1=u_0+u_1
    #
    # x_2=u_0+u_1+u_2
    #
    # x_3=sg(u_0)+u_1+u_2+u_3
    #
    # x_4=sg(u_0+u_1)+u_2+u_3+u_4
    #
    # 因为 stop-gradient 不改变前向数值，所以它们仍然等于：
    #
    # x_3=u_0+u_1+u_2+u_3
    #
    # x_4=u_0+u_1+u_2+u_3+u_4
    #
    # 但是 x_4 的梯度只会传播到：
    #
    # u_2,u_3,u_4
    #
    # 不会传播到：
    #
    # u_0,u_1
    return cumulative_sum
