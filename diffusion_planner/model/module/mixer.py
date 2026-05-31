# 导入 PyTorch 的神经网络模块，并将其命名为 nn。
#
# 当前文件使用：
# 1. nn.Module：定义神经网络模块；
# 2. nn.LayerNorm：对特征维度执行层归一化；
# 3. nn.GELU：作为 MLP 中的激活函数。
import torch.nn as nn

# 从 timm 库中导入封装好的 Mlp 模块。
#
# Mlp 通常由以下结构组成：
#
# Linear
# → 激活函数
# → Dropout
# → Linear
# → Dropout
#
# 当前模块分别使用两个 MLP：
# 1. tokens_mlp：混合不同 token 之间的信息；
# 2. channels_mlp：混合同一个 token 内部不同通道之间的信息。
from timm.models.layers import Mlp


# 定义一个 MLP-Mixer 风格的基础模块。
#
# MLP-Mixer 的核心思想是：
#
# 1. Token Mixing：
#    在 token 维度上进行信息交互；
#
# 2. Channel Mixing：
#    在通道维度上进行特征变换。
#
# 与 Transformer 不同，
# 该模块没有使用注意力机制，
# 而是仅通过 MLP 完成 token 之间和通道之间的信息融合。
#
# 输入张量 x 通常具有以下形状：
#
# [B, T, C]
#
# 其中：
# 1. B：batch size；
# 2. T：token 数量；
# 3. C：每个 token 的通道数或特征维度。
class MixerBlock(nn.Module):

    # 初始化 MixerBlock。
    #
    # 参数说明：
    #
    # tokens_mlp_dim：
    # token 数量。
    #
    # 在 Token Mixing 阶段，
    # 输入张量会经过维度置换，
    # 使 token 维度成为 MLP 的最后一个维度。
    #
    # channels_mlp_dim：
    # 每个 token 的通道数或特征维度。
    #
    # 在 Channel Mixing 阶段，
    # MLP 会直接沿最后一个通道维度进行变换。
    #
    # drop_path_rate：
    # 传递给 timm.models.layers.Mlp 的 drop 参数。
    #
    # 注意：
    # 虽然变量名包含 drop_path，
    # 但当前代码并没有显式使用 DropPath 或 Stochastic Depth。
    #
    # 实际行为是：
    # 将 drop_path_rate 作为 MLP 内部 Dropout 的概率。
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):

        # 调用父类 nn.Module 的初始化方法。
        super().__init__()

        # 定义第一个 LayerNorm。
        #
        # 该归一化层用于 Token Mixing 之前。
        #
        # nn.LayerNorm(channels_mlp_dim) 会沿最后一个维度执行归一化。
        #
        # 如果输入形状为：
        #
        # [B, T, C]
        #
        # 则会针对每个 token 的 C 个通道分别计算均值和方差。
        self.norm1 = nn.LayerNorm(channels_mlp_dim)

        # 定义通道混合 MLP。
        #
        # 输入维度和隐藏层维度均为 channels_mlp_dim。
        #
        # 在 forward(...) 中，
        # 该 MLP 接收形状为：
        #
        # [B, T, C]
        #
        # 的张量，并沿最后一个通道维度 C 进行变换。
        #
        # 因此，每个 token 都会独立地进行通道特征融合。
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)

        # 定义第二个 LayerNorm。
        #
        # 该归一化层用于 Channel Mixing 之前。
        #
        # 与 self.norm1 一样，
        # 它会沿最后一个特征维度进行归一化。
        self.norm2 = nn.LayerNorm(channels_mlp_dim)

        # 定义 token 混合 MLP。
        #
        # 输入维度和隐藏层维度均为 tokens_mlp_dim。
        #
        # 在 forward(...) 中，
        # 输入张量会先从：
        #
        # [B, T, C]
        #
        # 置换为：
        #
        # [B, C, T]
        #
        # 此时最后一个维度 T 表示 token 数量。
        #
        # 因此，tokens_mlp 会沿 token 维度进行变换，
        # 让不同 token 之间产生信息交互。
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    # 定义 MixerBlock 的前向传播过程。
    #
    # 输入：
    #
    # x：
    # 特征张量，通常形状为：
    #
    # [B, T, C]
    #
    # 输出：
    #
    # 与输入形状相同：
    #
    # [B, T, C]
    #
    # 整体流程为：
    #
    # 输入 x
    # → LayerNorm
    # → Token Mixing
    # → 残差连接
    # → LayerNorm
    # → Channel Mixing
    # → 残差连接
    # → 输出
    def forward(self, x):

        # 对输入张量执行第一次 LayerNorm。
        #
        # 输入和输出形状保持不变：
        #
        # [B, T, C]
        #
        # 此时 y 是用于 Token Mixing 的归一化特征。
        y = self.norm1(x)

        # 交换 token 维度和通道维度。
        #
        # 原始形状：
        #
        # [B, T, C]
        #
        # 置换后：
        #
        # [B, C, T]
        #
        # PyTorch 中 permute(0, 2, 1) 表示：
        # 1. 第 0 个维度 B 保持不变；
        # 2. 将原来的第 2 个维度 C 放到第 1 个位置；
        # 3. 将原来的第 1 个维度 T 放到最后一个位置。
        #
        # 这样，后续 MLP 就可以沿 token 维度 T 执行变换。
        y = y.permute(0, 2, 1)

        # 执行 Token Mixing。
        #
        # 当前 y 的形状为：
        #
        # [B, C, T]
        #
        # timm.models.layers.Mlp 默认沿最后一个维度进行全连接变换。
        #
        # 因此，该操作会对每一个通道分别处理所有 token，
        # 使不同 token 之间能够交换信息。
        #
        # 输出形状仍然为：
        #
        # [B, C, T]
        y = self.tokens_mlp(y)

        # 将 token 维度和通道维度交换回原始顺序。
        #
        # 输入形状：
        #
        # [B, C, T]
        #
        # 输出形状：
        #
        # [B, T, C]
        y = y.permute(0, 2, 1)

        # 将 Token Mixing 的结果与原始输入相加。
        #
        # 这是一个残差连接：
        #
        # x_new = x + TokenMixing(LayerNorm(x))
        #
        # 残差连接可以：
        # 1. 保留原始特征；
        # 2. 缓解深层网络中的梯度消失问题；
        # 3. 提高训练稳定性。
        x = x + y

        # 对经过 Token Mixing 更新后的特征执行第二次 LayerNorm。
        #
        # 输入和输出形状均为：
        #
        # [B, T, C]
        #
        # 此时 y 将用于后续的 Channel Mixing。
        y = self.norm2(x)

        # 执行 Channel Mixing，并通过残差连接返回最终结果。
        #
        # self.channels_mlp(y)：
        #
        # y 的形状为：
        #
        # [B, T, C]
        #
        # MLP 沿最后一个通道维度 C 进行变换。
        #
        # 因此，每个 token 都会独立地进行通道特征融合。
        #
        # 最终计算过程为：
        #
        # output = x + ChannelMixing(LayerNorm(x))
        #
        # 返回张量形状保持为：
        #
        # [B, T, C]
        return x + self.channels_mlp(y)