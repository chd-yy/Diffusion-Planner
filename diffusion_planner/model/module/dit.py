# 导入 Python 标准库中的 math 模块。
#
# 当前文件使用 math.log(...) 计算时间步正弦嵌入中的频率衰减系数。
import math

# 导入 PyTorch。
#
# 当前文件使用 PyTorch 完成：
# 1. 张量切片；
# 2. 张量拼接；
# 3. 三角函数计算；
# 4. 指数运算；
# 5. 条件调制；
# 6. 神经网络前向传播。
import torch

# 导入 PyTorch 神经网络模块，并命名为 nn。
#
# 当前文件使用了：
# 1. nn.Module：定义神经网络模块；
# 2. nn.Linear：构造全连接层；
# 3. nn.Sequential：按顺序组合多个网络层；
# 4. nn.SiLU：使用 SiLU 激活函数；
# 5. nn.LayerNorm：对特征维度进行层归一化；
# 6. nn.MultiheadAttention：构造多头注意力层；
# 7. nn.GELU：使用 GELU 激活函数。
import torch.nn as nn

# 从 timm 库中导入 Mlp 模块。
#
# Mlp 是一个已经封装好的多层感知机模块。
# 在当前文件中，它被用作 Transformer Block 中的前馈神经网络。
from timm.models.layers import Mlp


# 定义特征调制函数。
#
# 该函数实现类似 adaptive Layer Normalization 中的仿射变换：
#
# x_new = x * (1 + scale) + shift
#
# 参数说明：
#
# x：
# 需要调制的特征张量。
# 常见形状为：
#
# [B, P, D]
#
# 其中：
# 1. B 表示 batch size；
# 2. P 表示 token 数量；
# 3. D 表示每个 token 的特征维度。
#
# shift：
# 平移参数。
# 常见形状为：
#
# [B, D]
#
# scale：
# 缩放参数。
# 常见形状为：
#
# [B, D]
#
# only_first：
# 是否只调制第一个 token。
#
# 当 only_first=False 时：
# 对所有 token 进行相同形式的条件调制。
#
# 当 only_first=True 时：
# 只调制索引为 0 的第一个 token，
# 其余 token 保持不变。
def modulate(x, shift, scale, only_first=False):

    # 判断是否只处理第一个 token。
    if only_first:

        # 将输入张量拆分为两部分。
        #
        # x_first：
        # 第一个 token，形状为：
        #
        # [B, 1, D]
        #
        # x_rest：
        # 除第一个 token 之外的其余 token，形状为：
        #
        # [B, P - 1, D]
        x_first, x_rest = x[:, :1], x[:, 1:]

        # 只对第一个 token 执行仿射调制。
        #
        # scale 的原始形状通常为：
        #
        # [B, D]
        #
        # scale.unsqueeze(1) 后变为：
        #
        # [B, 1, D]
        #
        # 这样可以与 x_first 对齐并逐元素相乘。
        #
        # shift.unsqueeze(1) 同样将 shift 调整为：
        #
        # [B, 1, D]
        #
        # 调制完成后，再将第一个 token 与未修改的其余 token 拼接。
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)

    # 如果 only_first=False，则对所有 token 执行条件调制。
    else:

        # 对 x 中的每一个 token 使用相同的 shift 和 scale。
        #
        # x 的形状通常为：
        #
        # [B, P, D]
        #
        # scale.unsqueeze(1) 和 shift.unsqueeze(1) 的形状为：
        #
        # [B, 1, D]
        #
        # PyTorch 会沿着 token 维度 P 自动广播。
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    # 返回调制后的特征张量。
    return x


# 定义仅执行缩放、不执行平移的特征调制函数。
#
# 该函数实现：
#
# x_new = x * (1 + scale)
#
# 与 modulate(...) 相比：
# 1. 保留了缩放操作；
# 2. 不包含 shift 平移项。
#
# 参数 only_first 的作用与 modulate(...) 中相同。
def scale(x, scale, only_first=False):

    # 判断是否只对第一个 token 进行缩放。
    if only_first:

        # 将输入拆分为第一个 token 和剩余 token。
        x_first, x_rest = x[:, :1], x[:, 1:]

        # 只缩放第一个 token。
        #
        # scale.unsqueeze(1) 会在 token 维度增加一个轴，
        # 从而使 scale 可以与 x_first 逐元素相乘。
        #
        # 缩放完成后，将第一个 token 和其余未修改 token 重新拼接。
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)

    # 如果 only_first=False，则缩放所有 token。
    else:

        # 对每个 token 使用相同的缩放参数。
        x = x * (1 + scale.unsqueeze(1))

    # 返回缩放后的特征张量。
    return x


# 定义扩散时间步嵌入模块。
#
# 扩散模型中的时间步 t 通常是一个标量，
# 但神经网络需要使用向量形式的条件信息。
#
# 因此，该模块会将标量时间步转换为高维特征向量。
#
# 整体流程为：
#
# 标量时间步 t
# → 正弦和余弦频率编码
# → 两层 MLP
# → 时间步条件向量
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    # 初始化时间步嵌入模块。
    #
    # 参数说明：
    #
    # hidden_size：
    # 输出时间步嵌入向量的维度。
    #
    # frequency_embedding_size：
    # 正弦和余弦频率编码的维度。
    # 默认值为 256。
    def __init__(self, hidden_size, frequency_embedding_size=256):

        # 调用 nn.Module 的初始化方法。
        super().__init__()

        # 构造用于处理频率编码的 MLP。
        #
        # 网络结构为：
        #
        # frequency_embedding_size
        # → Linear
        # → hidden_size
        # → SiLU
        # → Linear
        # → hidden_size
        #
        # 最终输出可作为神经网络中的时间条件特征。
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # 保存正弦和余弦频率编码维度。
        self.frequency_embedding_size = frequency_embedding_size

    # 将 timestep_embedding 定义为静态方法。
    #
    # 静态方法不依赖类实例中的成员变量。
    #
    # 只要提供 t、dim 和 max_period，
    # 就可以独立计算时间步频率编码。
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        创建正弦和余弦时间步嵌入。
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
                t: 一个一维张量，包含 N 个时间步索引，每个索引对应一个批次元素。
                这些索引可以是小数。
        :param dim: the dimension of the output.
                    This should be an even number.
                dim: 输出嵌入的维度。
                    这个值应该是偶数。
        :param max_period: controls the minimum frequency of the embeddings.
                            Default is 10,000, which matches the maximum period
                            of the positional embeddings in the original Transformer paper.
                max_period: 控制嵌入的最低频率。
        :return: an (N, D) Tensor of positional embeddings.
                    返回一个形状为 (N, D) 的时间步嵌入张量。
        """

        # 该实现参考 OpenAI GLIDE 项目中的时间步嵌入方法。
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        # 计算频率数量。
        #
        # dim 表示最终嵌入维度。
        #
        # 一半维度用于 cos 编码，
        # 另一半维度用于 sin 编码。
        #
        # 例如：
        #
        # dim = 256
        #
        # 则：
        #
        # half = 128
        half = dim // 2

        # 构造按照指数规律递减的频率序列。
        #
        # torch.arange(start=0, end=half, dtype=torch.float32)
        #
        # 生成：
        #
        # [0, 1, 2, ..., half - 1]
        #
        # 再计算：
        #
        # exp(-log(max_period) * index / half)
        #
        # 等价于：
        #
        # max_period ^ (-index / half)
        #
        # 因此 freqs 中包含从高频到低频的一组频率。
        #
        # 最后通过：
        #
        # .to(device=t.device)
        #
        # 将频率张量移动到与输入 t 相同的设备。
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        # 将每一个时间步分别乘以所有频率。
        #
        # 假设：
        #
        # t.shape = [N]
        #
        # 则：
        #
        # t[:, None].shape = [N, 1]
        #
        # freqs[None].shape = [1, half]
        #
        # 经过广播相乘后：
        #
        # args.shape = [N, half]
        #
        # 每一行表示一个时间步在不同频率下的相位。
        args = t[:, None].float() * freqs[None]

        # 分别计算所有相位的余弦值和正弦值，
        # 再沿最后一个维度拼接。
        #
        # torch.cos(args) 的形状为：
        #
        # [N, half]
        #
        # torch.sin(args) 的形状为：
        #
        # [N, half]
        #
        # 拼接后 embedding 的形状为：
        #
        # [N, 2 * half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 判断目标维度 dim 是否为奇数。
        #
        # 当 dim 为偶数时：
        #
        # 2 * half = dim
        #
        # 不需要额外处理。
        #
        # 当 dim 为奇数时：
        #
        # 2 * half = dim - 1
        #
        # 当前编码会缺少一个维度。
        if dim % 2:

            # 在最后补充一列全零值，
            # 确保最终嵌入维度严格等于 dim。
            #
            # embedding[:, :1] 的形状为：
            #
            # [N, 1]
            #
            # torch.zeros_like(...) 会生成相同形状的全零张量。
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # 返回正弦和余弦时间步编码。
        return embedding

    # 定义时间步嵌入模块的前向传播过程。
    #
    # 输入：
    #
    # t：
    # 一维时间步张量，常见形状为：
    #
    # [B]
    #
    # 输出：
    #
    # t_emb：
    # 时间步嵌入向量，形状为：
    #
    # [B, hidden_size]
    def forward(self, t):

        # 将标量时间步转换为正弦和余弦频率编码。
        #
        # t_freq 的形状为：
        #
        # [B, frequency_embedding_size]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        # 使用 MLP 进一步映射频率编码。
        #
        # t_emb 的形状为：
        #
        # [B, hidden_size]
        t_emb = self.mlp(t_freq)

        # 返回时间步条件向量。
        return t_emb


# 定义 DiT Block。
#
# DiT 是 Diffusion Transformer 的缩写。
#
# 当前模块将 Transformer 结构用于扩散模型，
# 并结合：
# 1. 自注意力；
# 2. 多层感知机；
# 3. adaptive LayerNorm 调制；
# 4. Cross-Attention；
# 5. 条件门控。
#
# y 通常表示条件向量，例如时间步嵌入。
#
# cross_c 通常表示外部条件特征，
# 例如地图特征、场景特征或其他上下文信息。
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    # 初始化 DiT Block。
    #
    # 参数说明：
    #
    # dim：
    # 每个 token 的特征维度，默认值为 192。
    #
    # heads：
    # 多头注意力的头数，默认值为 6。
    #
    # dropout：
    # 注意力权重中的 dropout 概率，默认值为 0.1。
    #
    # mlp_ratio：
    # MLP 隐藏层维度相对于输入维度的倍率。
    #
    # 默认情况下：
    #
    # mlp_hidden_dim = 192 * 4 = 768
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):

        # 调用父类 nn.Module 的初始化方法。
        super().__init__()

        # 构造第一个 LayerNorm。
        #
        # self.norm1 用于在自注意力之前对 x 做归一化。
        self.norm1 = nn.LayerNorm(dim)

        # 构造多头自注意力层。
        #
        # 参数：
        #
        # dim：
        # 输入 token 的特征维度。
        #
        # heads：
        # 注意力头数量。
        #
        # dropout：
        # 注意力权重 dropout 概率。
        #
        # batch_first=True：
        # 输入和输出张量格式使用：
        #
        # [B, P, D]
        #
        # 而不是默认的：
        #
        # [P, B, D]
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # 构造第二个 LayerNorm。
        #
        # self.norm2 用于在第一个 MLP 之前归一化 x。
        self.norm2 = nn.LayerNorm(dim)

        # 计算 MLP 隐藏层维度。
        #
        # 例如：
        #
        # dim = 192
        # mlp_ratio = 4.0
        #
        # 则：
        #
        # mlp_hidden_dim = 768
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 定义用于创建 GELU 激活函数的匿名函数。
        #
        # approximate="tanh" 表示使用 tanh 近似形式的 GELU，
        # 通常可以在保持效果的同时提高计算效率。
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # 构造第一个 MLP。
        #
        # 输入维度为 dim，
        # 隐藏层维度为 mlp_hidden_dim，
        # 激活函数为近似 GELU，
        # drop=0 表示该 MLP 内部不启用 dropout。
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # 构造 adaptive LayerNorm 调制参数生成器。
        #
        # 输入：
        #
        # y，形状通常为：
        #
        # [B, dim]
        #
        # 输出：
        #
        # [B, 6 * dim]
        #
        # 输出会被拆分为六组参数：
        #
        # 1. shift_msa：
        #    自注意力之前的平移参数；
        #
        # 2. scale_msa：
        #    自注意力之前的缩放参数；
        #
        # 3. gate_msa：
        #    自注意力残差分支的门控参数；
        #
        # 4. shift_mlp：
        #    MLP 之前的平移参数；
        #
        # 5. scale_mlp：
        #    MLP 之前的缩放参数；
        #
        # 6. gate_mlp：
        #    MLP 残差分支的门控参数。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # 构造第三个 LayerNorm。
        #
        # self.norm3 用于在交叉注意力之前归一化 x。
        self.norm3 = nn.LayerNorm(dim)

        # 构造多头交叉注意力层。
        #
        # 与自注意力不同：
        #
        # 1. Query 来自 x；
        # 2. Key 来自 cross_c；
        # 3. Value 来自 cross_c。
        #
        # 这样，轨迹 token 可以从外部上下文特征中读取信息。
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # 构造第四个 LayerNorm。
        #
        # self.norm4 用于在第二个 MLP 之前归一化特征。
        self.norm4 = nn.LayerNorm(dim)

        # 构造第二个 MLP。
        #
        # 输入维度、隐藏层维度和激活函数
        # 与 self.mlp1 保持一致。
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    # 定义 DiT Block 的前向传播过程。
    #
    # 参数说明：
    #
    # x：
    # 需要处理的 token 特征。
    # 常见形状为：
    #
    # [B, P, D]
    #
    # cross_c：
    # 交叉注意力使用的上下文特征。
    # 常见形状为：
    #
    # [B, C, D]
    #
    # 其中 C 表示上下文 token 数量。
    #
    # y：
    # 条件向量。
    # 常见形状为：
    #
    # [B, D]
    #
    # 通常可以包含时间步嵌入或其他条件信息。
    #
    # attn_mask：
    # 传递给自注意力层的 key_padding_mask。
    # 常见形状为：
    #
    # [B, P]
    #
    # 在 PyTorch 的 MultiheadAttention 中，
    # key_padding_mask 中值为 True 的位置通常会被忽略。
    def forward(self, x, cross_c, y, attn_mask):

        # 使用条件向量 y 生成六组 adaLN 调制参数。
        #
        # self.adaLN_modulation(y) 的形状为：
        #
        # [B, 6 * D]
        #
        # .chunk(6, dim=1)：
        # 沿特征维度均匀拆分成六个张量。
        #
        # 每个张量的形状均为：
        #
        # [B, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        # 对 x 执行 LayerNorm，
        # 再使用 shift_msa 和 scale_msa 进行条件调制。
        #
        # self.norm1(x) 的形状为：
        #
        # [B, P, D]
        #
        # shift_msa 和 scale_msa 的形状为：
        #
        # [B, D]
        #
        # modulate(...) 内部会通过 unsqueeze(1)
        # 将其广播到所有 token。
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # 执行多头自注意力，并通过残差连接更新 x。
        #
        # 自注意力调用形式为：
        #
        # Query = modulated_x
        # Key = modulated_x
        # Value = modulated_x
        #
        # key_padding_mask=attn_mask：
        # 用于屏蔽无效 token。
        #
        # self.attn(...) 返回一个元组：
        #
        # 1. 注意力输出；
        # 2. 注意力权重。
        #
        # [0] 表示只取注意力输出。
        #
        # gate_msa.unsqueeze(1)：
        # 将门控参数形状从：
        #
        # [B, D]
        #
        # 变为：
        #
        # [B, 1, D]
        #
        # 再沿 token 维度广播。
        #
        # 更新形式为：
        #
        # x_new = x + gate_msa * Attention(modulated_x)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]

        # 对经过自注意力更新后的 x 执行第二次归一化和调制。
        #
        # shift_mlp 和 scale_mlp 用于控制
        # 第一个 MLP 分支的输入特征。
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)

        # 执行第一个 MLP，并通过残差连接更新 x。
        #
        # 更新形式为：
        #
        # x_new = x + gate_mlp * MLP(modulated_x)
        #
        # gate_mlp 用于控制 MLP 分支对主干特征的影响强度。
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # 执行交叉注意力。
        #
        # Query：
        #
        # self.norm3(x)
        #
        # Key：
        #
        # cross_c
        #
        # Value：
        #
        # cross_c
        #
        # 这使得 x 中的 token 可以读取 cross_c 中的上下文信息。
        #
        # 注意：
        # 当前代码会直接使用交叉注意力输出覆盖 x。
        #
        # 这里没有显式写成：
        #
        # x = x + cross_attn(...)
        #
        # 因此，该步骤不是残差连接形式。
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]

        # 对交叉注意力输出执行 LayerNorm 和第二个 MLP。
        #
        # 注意：
        # 当前代码同样直接使用 MLP 输出覆盖 x。
        #
        # 这里没有显式写成：
        #
        # x = x + self.mlp2(...)
        #
        # 因此，该步骤也不是残差连接形式。
        x = self.mlp2(self.norm4(x))

        # 返回 DiT Block 的输出特征。
        return x
    
    
# 定义 DiT 模型的最终输出层。
#
# 该模块用于将 Transformer 隐藏特征映射到目标输出空间。
#
# 同时，它会使用条件向量 y 生成 shift 和 scale，
# 对最终隐藏特征执行自适应调制。
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    # 初始化最终输出层。
    #
    # 参数说明：
    #
    # hidden_size：
    # 输入隐藏特征维度。
    #
    # output_size：
    # 最终输出特征维度。
    #
    # 例如：
    # 如果模型需要为每个 token 输出轨迹状态，
    # output_size 可以对应目标状态维度。
    def __init__(self, hidden_size, output_size):

        # 调用父类 nn.Module 的初始化方法。
        super().__init__()

        # 构造最终调制前的 LayerNorm。
        #
        # 输入和输出维度均为 hidden_size。
        self.norm_final = nn.LayerNorm(hidden_size)

        # 构造最终投影网络。
        #
        # 整体结构为：
        #
        # hidden_size
        # → LayerNorm
        # → Linear
        # → hidden_size * 4
        # → GELU
        # → LayerNorm
        # → Linear
        # → output_size
        #
        # 该结构先提升特征维度，
        # 再映射到目标输出维度。
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        # 构造最终层使用的 adaptive LayerNorm 调制参数生成器。
        #
        # 输入：
        #
        # y，形状通常为：
        #
        # [B, hidden_size]
        #
        # 输出：
        #
        # [B, 2 * hidden_size]
        #
        # 输出随后会被拆分为：
        #
        # 1. shift：
        #    平移参数；
        #
        # 2. scale：
        #    缩放参数。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    # 定义最终输出层的前向传播过程。
    #
    # 参数说明：
    #
    # x：
    # Transformer 输出特征。
    # 常见形状为：
    #
    # [B, P, hidden_size]
    #
    # y：
    # 条件向量。
    # 常见形状为：
    #
    # [B, hidden_size]
    def forward(self, x, y):

        # 获取输入特征的形状。
        #
        # B：
        # batch size。
        #
        # P：
        # token 数量。
        #
        # _：
        # 隐藏特征维度。
        #
        # 注意：
        # B 和 P 在后续代码中没有继续使用，
        # 但该语句仍然保留原始实现。
        B, P, _ = x.shape
        
        # 根据条件向量 y 生成 shift 和 scale。
        #
        # self.adaLN_modulation(y) 的形状为：
        #
        # [B, 2 * hidden_size]
        #
        # chunk(2, dim=1)：
        # 沿特征维度拆分成两个张量。
        #
        # shift 和 scale 的形状均为：
        #
        # [B, hidden_size]
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)

        # 对输入特征执行：
        #
        # 1. LayerNorm；
        # 2. 条件缩放；
        # 3. 条件平移。
        #
        # 调制公式为：
        #
        # x_new = norm_final(x) * (1 + scale) + shift
        #
        # shift 和 scale 会沿 token 维度自动广播。
        x = modulate(self.norm_final(x), shift, scale)

        # 将调制后的隐藏特征投影到目标输出维度。
        #
        # 输入形状为：
        #
        # [B, P, hidden_size]
        #
        # 输出形状为：
        #
        # [B, P, output_size]
        x = self.proj(x)

        # 返回最终输出。
        return x