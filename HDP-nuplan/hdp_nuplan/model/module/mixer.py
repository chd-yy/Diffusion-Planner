# 【实现核对：与原 Diffusion-Planner 完全一致】两个 mixer.py 的导入、类定义、
# __init__ 和 forward 可执行 AST 逐语句相同，没有包路径、参数、张量计算或返回值差异。
# 本文件只是缺少原版的详细注释，以下注释由原版迁移并做了精简。

# nn 提供 Module、LayerNorm 和 GELU。
import torch.nn as nn
# timm 的 Mlp 通常由 Linear、激活、Dropout、Linear、Dropout 组成，并沿最后一维计算。
from timm.models.layers import Mlp


# MLP-Mixer 风格的基础模块，不使用注意力，而是依次进行：
# token mixing -> 残差连接 -> channel mixing -> 残差连接。
# 输入和输出形状均为 [B,T,C]：B 是 batch，T 是 token 数量，C 是通道维度。
class MixerBlock(nn.Module):
    # tokens_mlp_dim 必须等于输入 token 数 T；channels_mlp_dim 必须等于通道数 C。
    # 注意：参数名虽然叫 drop_path_rate，但代码没有使用 DropPath；它被传给 Mlp 的
    # drop 参数，因此实际控制的是 MLP 内部 Dropout 概率。
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        # 初始化 nn.Module，确保下面的归一化层和两个 MLP 被正确注册。
        super().__init__()

        # Token Mixing 前先沿最后一个通道维 C 做 LayerNorm，形状保持 [B,T,C]。
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        # Channel Mixing MLP 沿最后一维 C 变换，每个 token 独立融合自身通道。
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        # Channel Mixing 前的第二个 LayerNorm，同样沿通道维 C 归一化。
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        # Token Mixing MLP 的输入维度是 T；forward 会先把张量置换成 [B,C,T]，
        # 使 Mlp 沿最后一个 token 维度执行变换，实现不同 token 之间的信息交互。
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    # 输入 x:[B,T,C]，返回相同形状；采用 Pre-Norm 和两次残差连接。
    def forward(self, x):
        # 第一次归一化，为 Token Mixing 准备特征，形状仍为 [B,T,C]。
        y = self.norm1(x)
        # [B,T,C] -> [B,C,T]，把 token 维 T 移到 MLP 使用的最后一维。
        y = y.permute(0, 2, 1)
        # 沿 T 维进行 Token Mixing，每个通道分别聚合全部 token。
        y = self.tokens_mlp(y)
        # [B,C,T] -> [B,T,C]，恢复原始维度顺序。
        y = y.permute(0, 2, 1)
        # 第一次残差：x = x + TokenMixing(LayerNorm(x))。
        x = x + y
        # 对 Token Mixing 后的特征做第二次归一化，为 Channel Mixing 准备输入。
        y = self.norm2(x)
        # 沿 C 维执行 Channel Mixing 并做第二次残差，输出仍为 [B,T,C]。
        return x + self.channels_mlp(y)
