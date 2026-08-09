import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp


# 与原 Diffusion-Planner 实现一致：执行 adaLN 的仿射调制
# x_new = x * (1 + scale) + shift。
# x 通常为 [B, P, D]，shift/scale 通常为 [B, D]；unsqueeze(1) 后会沿 token 维广播。
def modulate(x, shift, scale, only_first=False):
    # only_first=True 时只调制第一个 token，其余 token 原样保留。
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


# 与原 Diffusion-Planner 实现一致：只缩放、不平移，公式为 x_new = x * (1 + scale)。
def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    将标量扩散时间步编码为向量；本类的可执行实现与原 Diffusion-Planner 完全一致。

    流程：标量时间步 -> 正弦/余弦频率编码 -> 两层 MLP -> 时间条件向量。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # [B, frequency_embedding_size] -> [B, hidden_size]。
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦/余弦时间步嵌入。

        :param t: 一维时间步张量 [N]，元素可以是小数。
        :param dim: 输出嵌入维度 D。
        :param max_period: 控制最低频率。
        :return: 时间步嵌入 [N, D]。
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # 一半维度放 cos，另一半放 sin；频率按指数规律从高到低排列。
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # t[:, None] 和 freqs[None] 通过广播得到 [N, half] 的相位矩阵。
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # dim 为奇数时，cos/sin 拼接后少一维，因此在末尾补零。
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t_freq: [B, frequency_embedding_size]；t_emb: [B, hidden_size]。
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    使用 adaLN-Zero 条件调制的 Diffusion Transformer Block。

    【HDP 与原 Diffusion-Planner 的核心区别】
    HDP 按“自注意力 -> 条件交叉注意力 -> MLP”执行，并让三条分支都使用
    独立的 adaLN shift/scale/gate 和残差连接；原版则是“自注意力 -> MLP1
    -> 交叉注意力 -> MLP2”，只有前两条分支接受 adaLN 调制和门控，后两步
    会直接覆盖 x。
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        # 【区别：LayerNorm 数值参数】HDP 三个 LayerNorm 显式使用 eps=1e-6；
        # 原版四个 LayerNorm 未传 eps，采用 PyTorch 默认值 1e-5。
        # elementwise_affine=True 与原版默认行为相同，不构成功能差异。
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        # 【区别：自注意力 dropout】HDP 将概率固定为 0.1，没有使用形参 dropout；
        # 原版把构造函数形参 dropout 传入自注意力。默认调用时二者均为 0.1。
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        # MLP 隐藏维度为 int(dim * mlp_ratio)，默认是 192 * 4 = 768。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # lambda 是匿名函数；timm.Mlp 会调用它来创建 tanh 近似的 GELU 激活层。
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # 【区别：交叉注意力 dropout】HDP 未传 dropout，使用 MultiheadAttention
        # 默认值 0.0；原版传入形参 dropout（默认 0.1）。
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        # 【区别：MLP 数量与 dropout】HDP 只有一个 MLP，drop 固定为 0.1；
        # 原版有 mlp1、mlp2 两个 MLP，二者的 drop 都为 0。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # 【区别：adaLN 参数数量】HDP 输出 9 组参数，分别控制 self-attention、
            # cross-attention、MLP 的 shift、scale、gate；原版输出 6 组参数，
            # 只控制 self-attention 和 mlp1。
            nn.Linear(dim, 9 * dim, bias=True)
        )


    def forward(self, x, cond, y, cond_padding_mask=None):
        # 输入通常为：x=[B,P,D]，cond=[B,C,D]，y=[B,D]。
        # 【区别：接口和 mask 语义】原版参数名为 (x, cross_c, y, attn_mask)，
        # 必须传入 [B,P] 的 attn_mask，并用它屏蔽自注意力中的无效预测对象 token。
        # HDP 的 cond_padding_mask 是可选的 [B,C] 条件 token mask，仅传给交叉注意力；
        # HDP 自注意力不使用 mask，因为当前 x 表示全部有效的未来时间 token。
        # chunk(9, dim=-1) 沿最后一个特征维拆分；对二维 y 而言，原版的 dim=1
        # 与这里的 dim=-1 指向同一维，但 HDP 拆成 9 组而不是 6 组。
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp \
            = self.adaLN_modulation(y).chunk(9, dim=-1)
        # 自注意力：Q=K=V=调制后的 x；[0] 只取注意力输出，不取注意力权重。
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x)[0]
        # 【区别：交叉注意力更新】Q 来自 x，K/V 来自 cond；HDP 对 Q 做独立
        # adaLN 调制，并以 gate_mca 控制残差增量。原版不调制、无门控且直接覆盖 x。
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), cond, cond, key_padding_mask=cond_padding_mask)[0]
        # 【区别：MLP 更新】HDP 的唯一 MLP 是第三条门控残差分支；原版先执行
        # 门控残差 mlp1，交叉注意力后再由无残差的 mlp2 覆盖 x。
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

    
class FinalLayer(nn.Module):
    """
    DiT 最终输出层；本类的可执行实现与原 Diffusion-Planner 完全一致。

    先根据条件 y 调制隐藏特征，再通过投影网络映射到目标输出维度。
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        # [B,P,hidden_size] -> [B,P,4*hidden_size] -> [B,P,output_size]。
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # 生成最终 adaLN 使用的 shift 和 scale 两组参数。
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        # B、P 后续未使用；保留该语句以严格维持原实现。
        B, P, _ = x.shape
        
        # y: [B,hidden_size] -> shift/scale: 各 [B,hidden_size]。
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # 输出形状为 [B,P,output_size]。
        x = self.proj(x)
        return x
