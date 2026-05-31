# 导入 PyTorch。
# 当前文件主要使用 PyTorch 完成张量拼接、形状变换、掩码计算、
# 三角函数运算以及创建占位张量等操作。
import torch

# 导入 PyTorch 的神经网络模块，并命名为 nn。
# 当前文件会使用 nn.Module、nn.Linear、nn.LayerNorm、
# nn.MultiheadAttention、nn.ModuleList、nn.Identity 和 nn.Embedding。
import torch.nn as nn

# 从 timm 中导入封装好的 MLP 模块。
# Mlp 通常包含两层全连接层、激活函数和 Dropout。
from timm.models.layers import Mlp

# 从 timm 中导入 DropPath。
# DropPath 也称为 Stochastic Depth。
# 在训练过程中，它可以随机丢弃整个残差分支，从而起到正则化作用。
from timm.layers import DropPath

# 导入项目中定义的 MixerBlock。
# MixerBlock 使用 MLP-Mixer 风格的结构，
# 同时完成 token 维度和通道维度的信息融合。
from diffusion_planner.model.module.mixer import MixerBlock


# 定义整个场景的编码器。
#
# Encoder 是当前文件的入口模块。
# 它会分别编码三类信息：
#
# 1. 动态交通参与者：
#    例如周围车辆、行人或其他移动目标；
#
# 2. 静态物体：
#    例如道路障碍物、路障或其他静止目标；
#
# 3. 矢量化车道：
#    例如道路中心线、车道边界、限速信息和信号灯信息。
#
# 三类 token 完成独立编码后，会被拼接到一起，
# 再通过 FusionEncoder 执行全局自注意力融合。
class Encoder(nn.Module):
    # 初始化场景编码器。
    #
    # config 是配置对象，通常需要包含：
    #
    # config.hidden_dim：
    # 所有 token 最终统一使用的隐藏特征维度；
    #
    # config.agent_num：
    # 动态交通参与者数量；
    #
    # config.static_objects_num：
    # 静态物体数量；
    #
    # config.lane_num：
    # 车道数量；
    #
    # config.time_len：
    # 动态交通参与者历史轨迹的时间长度；
    #
    # config.lane_len：
    # 每条车道包含的离散采样点数量；
    #
    # config.encoder_drop_path_rate：
    # 编码器中 Dropout 或 DropPath 的比例；
    #
    # config.encoder_depth：
    # 编码器堆叠的模块层数；
    #
    # config.num_heads：
    # 多头注意力中的注意力头数量；
    #
    # config.device：
    # 模型运行设备，例如 "cuda" 或 "cpu"。
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法。
        super().__init__()

        # 保存统一隐藏特征维度。
        # 动态交通参与者、静态物体和车道最终都会被映射到该维度。
        self.hidden_dim = config.hidden_dim

        # 计算场景 token 的总数量。
        #
        # 每一个动态交通参与者对应一个 token；
        # 每一个静态物体对应一个 token；
        # 每一条车道对应一个 token。
        self.token_num = config.agent_num + config.static_objects_num + config.lane_num

        # 创建动态交通参与者编码器。
        #
        # 该编码器会：
        # 1. 读取动态交通参与者的历史状态；
        # 2. 使用 MLP-Mixer 融合时间序列信息；
        # 3. 添加目标类别嵌入；
        # 4. 将每个参与者编码成一个 hidden_dim 维 token。
        self.neighbor_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)

        # 创建静态物体编码器。
        #
        # 静态物体没有历史时间序列，
        # 因此只需要使用 MLP 将状态直接映射为隐藏向量。
        self.static_encoder = StaticFusionEncoder(config.static_objects_state_dim, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim)

        # 创建车道编码器。
        #
        # 每条车道由多个离散采样点表示。
        # 该编码器会使用 MLP-Mixer 融合车道点信息，
        # 并添加限速信息和信号灯信息。
        self.lane_encoder = LaneFusionEncoder(config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
    
        # 创建全局融合编码器。
        #
        # FusionEncoder 会在所有 token 之间执行多层自注意力，
        # 从而建模动态目标、静态目标和道路结构之间的相互关系。
        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, 
            drop_path_rate=config.encoder_drop_path_rate, 
            depth=config.encoder_depth, 
            device=config.device
        )

        # 原始注释说明：
        # 位置嵌入用于编码 x、y、cos、sin 和 token 类型。
        #
        # 输入维度为 7：
        #
        # [x, y, cos_h, sin_h, type_0, type_1, type_2]
        #
        # 其中，最后三个分量用于表示 token 类型：
        #
        # 动态交通参与者：[1, 0, 0]
        # 静态物体：[0, 1, 0]
        # 车道：[0, 0, 1]
        #
        # 输出维度为 hidden_dim，
        # 因而可以直接加到 token 的隐藏特征上。
        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(7, config.hidden_dim)

    # 定义 Encoder 的前向传播过程。
    #
    # inputs 是一个字典，至少需要包含：
    #
    # inputs['neighbor_agents_past']
    # inputs['static_objects']
    # inputs['lanes']
    # inputs['lanes_speed_limit']
    # inputs['lanes_has_speed_limit']
    def forward(self, inputs):

        # 创建输出字典。
        # 最终场景编码结果会写入 encoder_outputs['encoding']。
        encoder_outputs = {}

        # 读取周围动态交通参与者的历史状态。
        #
        # 常见形状：
        #
        # [B, P_agent, V_time, D_agent]
        #
        # B：
        # batch size；
        #
        # P_agent：
        # 动态交通参与者数量；
        #
        # V_time：
        # 历史时间点数量；
        #
        # D_agent：
        # 每个历史状态的特征维度。
        # agents
        neighbors = inputs['neighbor_agents_past']

        # 读取静态物体状态。
        #
        # 常见形状：
        #
        # [B, P_static, D_static]
        # static objects
        static = inputs['static_objects']

        # 读取矢量化车道数据。
        #
        # 常见形状：
        #
        # [B, P_lane, V_lane, D_lane]
        #
        # P_lane：
        # 车道数量；
        #
        # V_lane：
        # 每条车道中的离散采样点数量。
        # vector maps
        lanes = inputs['lanes']

        # 读取每条车道对应的限速。
        #
        # 常见形状：
        #
        # [B, P_lane, 1]
        lanes_speed_limit = inputs['lanes_speed_limit']

        # 读取每条车道是否存在已知限速。
        #
        # 常见形状：
        #
        # [B, P_lane, 1]
        #
        # 通常：
        # True 表示限速已知；
        # False 表示限速未知。
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        # 从动态交通参与者张量中读取 batch size。
        B = neighbors.shape[0]

        # 对动态交通参与者进行编码。
        #
        # 返回：
        #
        # encoding_neighbors：
        # 动态交通参与者 token；
        #
        # neighbors_mask：
        # 动态交通参与者无效掩码；
        #
        # neighbor_pos：
        # 动态交通参与者的位置、方向和类别信息。
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)

        # 对静态物体进行编码。
        #
        # 返回：
        #
        # encoding_static：
        # 静态物体 token；
        #
        # static_mask：
        # 静态物体无效掩码；
        #
        # static_pos：
        # 静态物体的位置、方向和类别信息。
        encoding_static, static_mask, static_pos = self.static_encoder(static)

        # 对车道进行编码。
        #
        # 返回：
        #
        # encoding_lanes：
        # 车道 token；
        #
        # lanes_mask：
        # 无效车道掩码；
        #
        # lane_pos：
        # 车道位置、方向和类别信息。
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        # 沿 token 维度拼接三类编码结果。
        #
        # 拼接顺序为：
        #
        # 1. 动态交通参与者 token；
        # 2. 静态物体 token；
        # 3. 车道 token。
        #
        # 输出形状：
        #
        # [B, token_num, hidden_dim]
        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        # 拼接三类 token 对应的位置、方向和类别信息。
        #
        # 拼接后的形状：
        #
        # [B, token_num, 7]
        #
        # view(B * self.token_num, -1) 会将 batch 维度和 token 维度展平：
        #
        # [B * token_num, 7]
        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)

        # 拼接三类无效 token 掩码。
        #
        # 拼接后的原始形状：
        #
        # [B, token_num]
        #
        # view(-1) 会将其展平为：
        #
        # [B * token_num]
        #
        # 通常：
        #
        # True 表示 token 无效；
        # False 表示 token 有效。
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)

        # 只对有效 token 计算位置嵌入。
        #
        # encoding_pos[~encoding_mask]：
        # 保留掩码为 False 的有效 token。
        #
        # self.pos_emb(...)：
        # 将 7 维位置和类别向量映射为 hidden_dim 维嵌入。
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])

        # 创建保存全部位置嵌入的全零张量。
        #
        # 形状：
        #
        # [B * token_num, hidden_dim]
        #
        # 无效 token 会保持为全零向量。
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)

        # 将有效 token 的位置嵌入填入对应位置。
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        # 将位置和类别嵌入加到 token 内容特征上。
        #
        # encoding_pos_result.view(B, self.token_num, -1)：
        # 恢复 batch 维度和 token 维度。
        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        # 使用全局融合编码器处理所有 token。
        #
        # encoding_mask.view(B, self.token_num)：
        # 将掩码恢复为：
        #
        # [B, token_num]
        #
        # 输出形状：
        #
        # [B, token_num, hidden_dim]
        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        # 返回编码结果字典。
        return encoder_outputs


# 定义一个基础自注意力模块。
#
# 该模块使用 Pre-Norm Transformer 风格结构：
#
# 输入
# → LayerNorm
# → Multi-Head Attention
# → DropPath
# → 残差连接
# → LayerNorm
# → MLP
# → DropPath
# → 残差连接
#
# 输出形状与输入形状保持一致。
class SelfAttentionBlock(nn.Module):
    # 初始化自注意力模块。
    #
    # dim：
    # token 特征维度；
    #
    # heads：
    # 注意力头数；
    #
    # dropout：
    # 注意力 Dropout、MLP Dropout 和 DropPath 的比例；
    #
    # mlp_ratio：
    # MLP 隐藏层相对于输入维度的倍率。
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        # 调用父类初始化方法。
        super().__init__()

        # 创建第一个 LayerNorm。
        # 该层用于自注意力之前的特征归一化。
        self.norm1 = nn.LayerNorm(dim)

        # 创建多头注意力层。
        #
        # batch_first=True 表示输入张量格式为：
        #
        # [B, token_num, dim]
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # 创建 DropPath 模块。
        #
        # 当 dropout > 0 时，使用 DropPath(dropout)；
        # 否则使用 nn.Identity()，即不改变输入。
        #
        # 注意：
        # 当前变量名 dropout 同时控制：
        #
        # 1. 注意力层内部 Dropout；
        # 2. MLP 内部 Dropout；
        # 3. 残差分支 DropPath。
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        # 创建第二个 LayerNorm。
        # 该层用于 MLP 之前的特征归一化。
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

        # 创建前馈神经网络。
        #
        # 输入和输出维度均为 dim；
        # 中间隐藏层维度为 mlp_hidden_dim。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    # 定义自注意力模块的前向传播。
    #
    # x 的形状：
    #
    # [B, token_num, dim]
    #
    # mask 的形状：
    #
    # [B, token_num]
    #
    # 通常：
    # True 表示忽略该 token；
    # False 表示保留该 token。
    def forward(self, x, mask):
        # 执行多头注意力，并通过残差连接更新 x。
        #
        # 当前代码中的注意力输入为：
        #
        # Query = self.norm1(x)
        # Key = x
        # Value = x
        #
        # 注意：
        # 只有 Query 执行了 LayerNorm；
        # Key 和 Value 使用原始 x。
        #
        # key_padding_mask=mask：
        # 屏蔽无效 token。
        #
        # self.attn(...) 返回：
        #
        # 1. 注意力输出；
        # 2. 注意力权重。
        #
        # [0] 表示只取注意力输出。
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])

        # 执行 MLP，并通过第二个残差连接更新 x。
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 返回更新后的 token 特征。
        return x


# 定义动态交通参与者编码器。
#
# 该模块会将每一个动态交通参与者的历史轨迹
# 编码为一个固定长度的隐藏向量。
#
# 处理流程：
#
# 历史状态序列
# → 时间点有效性判断
# → 拼接有效性特征
# → 通道投影
# → token 投影
# → 多层 MixerBlock
# → 时间维度平均池化
# → 添加目标类型嵌入
# → 映射到 hidden_dim
class AgentFusionEncoder(nn.Module):
    # 初始化动态交通参与者编码器。
    #
    # time_len：
    # 历史时间点数量；
    #
    # drop_path_rate：
    # MixerBlock 和 MLP 使用的 Dropout 比例；
    #
    # hidden_dim：
    # 最终输出隐藏维度；
    #
    # depth：
    # MixerBlock 层数；
    #
    # tokens_mlp_dim：
    # 时间 token 预投影后的维度；
    #
    # channels_mlp_dim：
    # 每个 token 的通道维度。
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        # 调用父类初始化方法。
        super().__init__()

        # 保存最终隐藏维度。
        self._hidden_dim = hidden_dim

        # 保存中间通道维度。
        self._channel = channels_mlp_dim

        # 创建动态交通参与者类型嵌入层。
        #
        # 输入维度为 3，
        # 对应类型 one-hot 或连续编码。
        #
        # 输出维度为 channels_mlp_dim。
        self.type_emb = nn.Linear(3, channels_mlp_dim)

        # 创建通道预投影 MLP。
        #
        # 输入维度为 8 + 1：
        #
        # 前 8 个分量：
        # [x, y, cos, sin, vx, vy, w, l]
        #
        # 额外 1 个分量：
        # 时间点是否有效。
        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建 token 预投影 MLP。
        #
        # 输入维度为历史时间点数量 time_len；
        # 输出维度为 tokens_mlp_dim。
        #
        # 在执行该层之前，
        # 输入会被置换为：
        #
        # [有效目标数量, channels_mlp_dim, time_len]
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建多个 MixerBlock。
        #
        # 每个 MixerBlock 会执行：
        #
        # 1. token mixing；
        # 2. channel mixing；
        # 3. 残差连接。
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        # 创建最终投影之前的 LayerNorm。
        self.norm = nn.LayerNorm(channels_mlp_dim)

        # 创建最终特征投影 MLP。
        #
        # 输出维度为统一的 hidden_dim。
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    # 定义动态交通参与者编码过程。
    #
    # 输入 x 的形状：
    #
    # [B, P, V, D]
    #
    # B：
    # batch size；
    #
    # P：
    # 动态交通参与者数量；
    #
    # V：
    # 历史时间点数量；
    #
    # D：
    # 每个时间点的特征维度。
    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        '''

        # 读取每个交通参与者最后一个历史时间点中的类型特征。
        #
        # 根据文档字符串，类型信息通常位于索引 8 之后，
        # 形状通常为：
        #
        # [B, P, 3]
        neighbor_type = x[:, :, -1, 8:]

        # 只保留前 8 个连续状态特征：
        #
        # [x, y, cos, sin, vx, vy, w, l]
        x = x[..., :8]

        # 读取每个交通参与者最后一个历史时间点的前 7 个分量，
        # 并复制一份独立张量。
        #
        # pos 最终会被转换为：
        #
        # [x, y, cos, sin, type_0, type_1, type_2]
        pos = x[:, :, -1, :7].clone() # x, y, cos, sin

        # 原始注释：
        # 动态交通参与者的类别标记为 [1, 0, 0]。
        # neighbor: [1,0,0]

        # 将 pos 的最后三个分量置零。
        pos[..., -3:] = 0.0

        # 将倒数第三个分量设置为 1。
        #
        # 最终类别标记为：
        #
        # [1, 0, 0]
        pos[..., -3] = 1.0
        
        # 读取张量形状。
        #
        # B：
        # batch size；
        #
        # P：
        # 动态交通参与者数量；
        #
        # V：
        # 历史时间点数量。
        B, P, V, _ = x.shape

        # 判断每一个历史时间点是否为空。
        #
        # torch.ne(x[..., :8], 0)：
        # 判断每个状态分量是否不等于 0。
        #
        # torch.sum(..., dim=-1) == 0：
        # 如果 8 个分量全部为 0，
        # 则认为该时间点无效。
        #
        # mask_v 的形状：
        #
        # [B, P, V]
        #
        # True 表示该时间点无效；
        # False 表示该时间点有效。
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0

        # 判断每一个动态交通参与者是否完全无效。
        #
        # ~mask_v：
        # 将无效掩码取反，得到有效标记。
        #
        # 如果一个目标的所有时间点均无效，
        # 则 mask_p 对应位置为 True。
        #
        # mask_p 的形状：
        #
        # [B, P]
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        # 为每一个时间点附加一个有效性特征。
        #
        # (~mask_v).float()：
        #
        # 有效时间点 → 1.0
        # 无效时间点 → 0.0
        #
        # 拼接后，特征维度从 8 变为 9。
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)

        # 合并 batch 维度和参与者维度。
        #
        # 输入形状：
        #
        # [B, P, V, 9]
        #
        # 输出形状：
        #
        # [B * P, V, 9]
        x = x.view(B * P, V, -1)

        # 将参与者掩码展平并取反。
        #
        # valid_indices 的形状：
        #
        # [B * P]
        #
        # True 表示该参与者至少存在一个有效历史时间点。
        valid_indices = ~mask_p.view(-1) 

        # 只保留有效动态交通参与者。
        x = x[valid_indices] 

        # 对每个历史时间点的 9 维状态执行通道投影。
        #
        # 输入形状：
        #
        # [P_valid, V, 9]
        #
        # 输出形状：
        #
        # [P_valid, V, channels_mlp_dim]
        x = self.channel_pre_project(x)

        # 交换 token 维度和通道维度。
        #
        # 输入形状：
        #
        # [P_valid, V, channels_mlp_dim]
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim, V]
        x = x.permute(0, 2, 1)

        # 沿时间 token 维度执行预投影。
        #
        # 输入形状：
        #
        # [P_valid, channels_mlp_dim, time_len]
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim, tokens_mlp_dim]
        x = self.token_pre_project(x)

        # 将 token 维度和通道维度交换回 MixerBlock 需要的顺序。
        #
        # 输出形状：
        #
        # [P_valid, tokens_mlp_dim, channels_mlp_dim]
        x = x.permute(0, 2, 1)

        # 依次执行多个 MixerBlock。
        #
        # 每一个 block 都会融合：
        #
        # 1. 不同时间 token 之间的信息；
        # 2. 不同通道之间的信息。
        for block in self.blocks:
            x = block(x)  

        # 原始注释：
        # 对 token 维度进行池化。
        # pooling

        # 沿 token 维度计算均值。
        #
        # 输入形状：
        #
        # [P_valid, tokens_mlp_dim, channels_mlp_dim]
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim]
        #
        # 每一个动态交通参与者最终被压缩为一个向量。
        x = torch.mean(x, dim=1)

        # 合并 batch 维度和参与者维度。
        #
        # 输入形状通常为：
        #
        # [B, P, 3]
        #
        # 输出形状：
        #
        # [B * P, 3]
        neighbor_type = neighbor_type.view(B * P, -1)

        # 只保留有效动态交通参与者的类别特征。
        neighbor_type = neighbor_type[valid_indices]

        # 将类型信息映射为 channels_mlp_dim 维嵌入。
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data

        # 将目标类型嵌入加到内容特征上。
        x = x + type_embedding

        # 对内容特征执行：
        #
        # 1. LayerNorm；
        # 2. MLP 投影。
        #
        # 输出维度变为 hidden_dim。
        x = self.emb_project(self.norm(x))

        # 创建用于保存全部动态交通参与者编码的全零张量。
        #
        # 无效目标会继续保持为全零向量。
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)

        # 将有效目标的编码结果写回对应位置。
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 返回：
        #
        # 1. 动态交通参与者 token：
        #    [B, P, hidden_dim]
        #
        # 2. 无效目标掩码：
        #    [B, P]
        #
        # 3. 位置、方向和类别信息：
        #    [B, P, 7]
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

    
# 定义静态物体编码器。
#
# 静态物体不包含历史时间序列，
# 因此不需要使用 MixerBlock。
#
# 每一个静态物体会被直接投影为一个 hidden_dim 维 token。
class StaticFusionEncoder(nn.Module):
    # 初始化静态物体编码器。
    #
    # dim：
    # 静态物体输入状态维度；
    #
    # drop_path_rate：
    # 传递给 MLP 的 Dropout 比例；
    #
    # hidden_dim：
    # 输出 token 的隐藏维度；
    #
    # device：
    # 设备字符串。
    #
    # 注意：
    # 当前实现没有直接使用 device 参数，
    # 但仍然保留该参数。
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        # 调用父类初始化方法。
        super().__init__()

        # 保存输出隐藏维度。
        self._hidden_dim = hidden_dim

        # 创建静态物体特征投影 MLP。
        #
        # 输入维度为 dim；
        # 这里的 dim 是静态物体输入特征的最后一维大小；在本项目常见就是 static_objects 的 10 维特征。
        # 输出维度为 hidden_dim。
        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    # 定义静态物体编码流程。
    #
    # 输入 x 的形状：
    #
    # [B, P, D]
    #
    # B：
    # batch size；
    #
    # P：
    # 静态物体数量；
    #
    # D：
    # 静态物体状态维度。
    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        ''' 

        # 读取 batch size 和静态物体数量。
        B, P, _ = x.shape

        # 复制每一个静态物体的前 7 个特征。
        #
        # pos 最终会被处理为：
        #
        # [x, y, cos, sin, type_0, type_1, type_2]
        pos = x[:, :, :7].clone() # x, y, cos, sin

        # 原始注释：
        # 静态物体类别标记为 [0, 1, 0]。
        # static: [0,1,0]

        # 清空 pos 的最后三个分量。
        pos[..., -3:] = 0.0

        # 将倒数第二个分量设置为 1。
        #
        # 最终类别标记为：
        #
        # [0, 1, 0]
        pos[..., -2] = 1.0

        # 创建保存全部静态物体编码结果的全零张量。
        #
        # 形状：
        #
        # [B * P, hidden_dim]
        #
        # 无效静态物体会保持为全零向量。
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        # 判断每一个静态物体是否为空。
        #
        # 如果前 10 个特征全部为 0，
        # 则认为该静态物体无效。
        #
        # mask_p 的形状：
        #
        # [B, P]
        #
        # True 表示静态物体无效；
        # False 表示静态物体有效。
        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        # 将掩码展平并取反。
        #
        # valid_indices 的形状：
        #
        # [B * P]
        #
        # True 表示静态物体有效。
        valid_indices = ~mask_p.view(-1) 

        # 只有在至少存在一个有效静态物体时，
        # 才执行后续 MLP 投影。
        #
        # 这样可以避免将空张量传入 MLP。
        if valid_indices.sum() > 0:
            # 合并 batch 维度和静态物体维度。
            #
            # 输入形状：
            #
            # [B, P, D]
            #
            # 输出形状：
            #
            # [B * P, D]
            x = x.view(B * P, -1)

            # 只保留有效静态物体。
            x = x[valid_indices]

            # 将有效静态物体投影到 hidden_dim 维空间。
            x = self.projection(x)

            # 将有效静态物体编码结果写回对应位置。
            x_result[valid_indices] = x

        # 返回：
        #
        # 1. 静态物体 token：
        #    [B, P, hidden_dim]
        #
        # 2. 无效静态物体掩码：
        #    [B, P]
        #
        # 3. 位置、方向和类别信息：
        #    [B, P, 7]
        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)
    

# 定义车道编码器。
#
# 每条车道由多个离散采样点构成。
#
# 该模块会：
#
# 1. 提取每个车道点的几何特征；
# 2. 使用 MLP-Mixer 融合车道点信息；
# 3. 提取车道中心附近的位置和方向；
# 4. 添加限速嵌入；
# 5. 添加交通信号灯嵌入；
# 6. 将每条车道编码为一个 hidden_dim 维 token。
class LaneFusionEncoder(nn.Module):
    # 初始化车道编码器。
    #
    # lane_len：
    # 每条车道中的离散采样点数量；
    #
    # drop_path_rate：
    # MixerBlock 和 MLP 使用的 Dropout 比例；
    #
    # hidden_dim：
    # 输出隐藏维度；
    #
    # depth：
    # MixerBlock 层数；
    #
    # tokens_mlp_dim：
    # token 预投影后的维度；
    #
    # channels_mlp_dim：
    # 每个 token 的通道维度。
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        # 调用父类初始化方法。
        super().__init__()

        # 保存每条车道中的离散采样点数量。
        self._lane_len = lane_len

        # 保存中间通道维度。
        self._channel = channels_mlp_dim

        # 创建已知限速嵌入层。
        #
        # 输入维度为 1：
        #
        # [speed_limit]
        #
        # 输出维度为 channels_mlp_dim。
        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)

        # 创建未知限速对应的可学习嵌入。
        #
        # nn.Embedding(1, channels_mlp_dim) 会保存一个可训练向量。
        #
        # 当车道没有已知限速时，
        # 使用该向量代替真实限速嵌入。
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)

        # 创建交通信号灯嵌入层。
        #
        # 输入维度为 4，
        # 通常用于表示信号灯状态。
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        # 创建车道几何特征的通道投影 MLP。
        #
        # 输入维度为 8：
        #
        # [x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y]
        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建车道点 token 的预投影 MLP。
        #
        # 输入维度为 lane_len；
        # 输出维度为 tokens_mlp_dim。
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建多个 MixerBlock。
        #
        # 每一层会同时融合：
        #
        # 1. 不同车道点之间的信息；
        # 2. 不同隐藏通道之间的信息。
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        # 创建最终投影前的 LayerNorm。
        self.norm = nn.LayerNorm(channels_mlp_dim)

        # 创建最终投影 MLP。
        #
        # 输出维度为统一 hidden_dim。
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    # 定义车道编码流程。
    #
    # x：
    # 车道离散点特征；
    #
    # speed_limit：
    # 每条车道的限速；
    #
    # has_speed_limit：
    # 每条车道是否存在已知限速。
    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''

        # 从每条车道的第 0 个采样点中读取信号灯信息。
        #
        # traffic 的形状通常为：
        #
        # [B, P, 4]
        traffic = x[:, :, 0, 8:]

        # 只保留前 8 个几何特征。
        #
        # x 的形状变为：
        #
        # [B, P, V, 8]
        x = x[..., :8]

        # 读取每条车道中间采样点附近的前 7 个特征。
        #
        # int(self._lane_len / 2)：
        # 取车道离散点序列中间位置的索引。
        #
        # pos 最终会被处理为：
        #
        # [x, y, cos_h, sin_h, type_0, type_1, type_2]
        pos = x[:, :, int(self._lane_len / 2), :7].clone() # x, y, x'-x, y'-y

        # 根据车道方向向量计算航向角。
        #
        # pos[..., 2]：
        # 车道方向的 x 分量；
        #
        # pos[..., 3]：
        # 车道方向的 y 分量。
        heading = torch.atan2(pos[..., 3], pos[..., 2])

        # 使用航向角余弦值替换方向向量 x 分量。
        pos[..., 2] = torch.cos(heading)

        # 使用航向角正弦值替换方向向量 y 分量。
        #
        # 替换后，方向表示被归一化为：
        #
        # [cos_h, sin_h]
        pos[..., 3] = torch.sin(heading)

        # 原始注释：
        # 车道类别标记为 [0, 0, 1]。
        # lane: [0,0,1]

        # 清空 pos 的最后三个分量。
        pos[..., -3:] = 0.0

        # 将最后一个分量设置为 1。
        #
        # 最终类别标记为：
        #
        # [0, 0, 1]
        pos[..., -1] = 1.0

        # 读取输入形状。
        #
        # B：
        # batch size；
        #
        # P：
        # 车道数量；
        #
        # V：
        # 每条车道的离散采样点数量。
        B, P, V, _ = x.shape

        # 判断每一个车道点是否为空。
        #
        # 如果前 8 个几何特征全部为 0，
        # 则认为该车道点无效。
        #
        # mask_v 的形状：
        #
        # [B, P, V]
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0

        # 判断每条车道是否完全为空。
        #
        # 如果一条车道没有任何有效采样点，
        # 则该车道无效。
        #
        # mask_p 的形状：
        #
        # [B, P]
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        # 合并 batch 维度和车道维度。
        #
        # 输入形状：
        #
        # [B, P, V, 8]
        #
        # 输出形状：
        #
        # [B * P, V, 8]
        x = x.view(B * P, V, -1)

        # 将车道掩码展平并取反。
        #
        # valid_indices 的形状：
        #
        # [B * P]
        #
        # True 表示车道有效。
        valid_indices = ~mask_p.view(-1) 

        # 只保留有效车道。
        x = x[valid_indices] 

        # 对每个车道点的 8 维几何特征执行通道投影。
        #
        # 输入形状：
        #
        # [P_valid, V, 8]
        #
        # 输出形状：
        #
        # [P_valid, V, channels_mlp_dim]
        x = self.channel_pre_project(x)

        # 交换 token 维度和通道维度。
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim, V]
        x = x.permute(0, 2, 1)

        # 沿车道点 token 维度执行预投影。
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim, tokens_mlp_dim]
        x = self.token_pre_project(x)

        # 将 token 维度和通道维度交换回 MixerBlock 需要的顺序。
        #
        # 输出形状：
        #
        # [P_valid, tokens_mlp_dim, channels_mlp_dim]
        x = x.permute(0, 2, 1)

        # 依次执行多个 MixerBlock。
        for block in self.blocks:
            x = block(x)  

        # 沿 token 维度计算均值。
        #
        # 输入形状：
        #
        # [P_valid, tokens_mlp_dim, channels_mlp_dim]
        #
        # 输出形状：
        #
        # [P_valid, channels_mlp_dim]
        #
        # 每条车道最终被压缩为一个向量。
        x = torch.mean(x, dim=1)

        # 原始注释：
        # 将限速、限速有效标记和交通信号灯状态
        # 变形为与展平后的车道维度一致的形式。
        # Reshape speed_limit and traffic to match flattened dimensions

        # 合并 batch 维度和车道维度。
        #
        # 输出形状：
        #
        # [B * P, 1]
        speed_limit = speed_limit.view(B * P, 1)

        # 合并 batch 维度和车道维度。
        #
        # 输出形状：
        #
        # [B * P, 1]
        has_speed_limit = has_speed_limit.view(B * P, 1)

        # 合并 batch 维度和车道维度。
        #
        # 输出形状通常为：
        #
        # [B * P, 4]
        traffic = traffic.view(B * P, -1)

        # 原始注释：
        # 只对有效车道处理限速信息。
        # Apply embedding directly to valid speed limit data

        # 只保留有效车道的限速有效标记。
        #
        # squeeze(-1) 会删除最后一个长度为 1 的维度。
        #
        # 输出形状：
        #
        # [P_valid]
        #
        # 注意：
        # 后续代码会使用 ~has_speed_limit，
        # 因此该张量通常应当为布尔类型。
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)

        # 只保留有效车道的限速数值。
        #
        # 输出形状：
        #
        # [P_valid]
        speed_limit = speed_limit[valid_indices].squeeze(-1)

        # 创建速度限制嵌入占位张量。
        #
        # 形状：
        #
        # [P_valid, channels_mlp_dim]
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        # 判断是否至少存在一条已知限速的有效车道。
        if has_speed_limit.sum() > 0:
            # 提取已知限速的车道，
            # 并将限速数值映射为 channels_mlp_dim 维嵌入。
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))

            # 将已知限速嵌入写入对应位置。
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        # 判断是否至少存在一条限速未知的有效车道。
        if (~has_speed_limit).sum() > 0:
            # 读取未知限速对应的可学习向量，
            # 并复制到所有限速未知车道。
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )

            # 将未知限速嵌入写入对应位置。
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # 原始注释：
        # 只处理有效车道的信号灯状态。
        # Process traffic lights directly for valid positions

        # 只保留有效车道对应的交通信号灯状态。
        traffic = traffic[valid_indices]

        # 将信号灯状态映射为 channels_mlp_dim 维嵌入。
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data


        # 融合：
        #
        # 1. 车道几何特征；
        # 2. 限速嵌入；
        # 3. 信号灯嵌入。
        x = x + speed_limit_embedding + traffic_light_embedding

        # 对融合后的车道特征执行：
        #
        # 1. LayerNorm；
        # 2. MLP 投影。
        #
        # 输出维度变为 hidden_dim。
        x = self.emb_project(self.norm(x))

        # 创建保存全部车道编码结果的全零张量。
        #
        # 无效车道会继续保持为全零向量。
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)

        # 将有效车道编码结果写回对应位置。
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 返回：
        #
        # 1. 车道 token：
        #    [B, P, hidden_dim]
        #
        # 2. 无效车道掩码：
        #    [B, P]
        #
        # 3. 位置、方向和类别信息：
        #    [B, P, 7]
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


# 定义全局融合编码器。
#
# 该模块会接收三类 token：
#
# 1. 动态交通参与者 token；
# 2. 静态物体 token；
# 3. 车道 token。
#
# 然后使用多层 SelfAttentionBlock，
# 建模不同场景元素之间的相互关系。
class FusionEncoder(nn.Module):
    # 初始化全局融合编码器。
    #
    # hidden_dim：
    # token 隐藏特征维度；
    #
    # num_heads：
    # 多头注意力的头数；
    #
    # drop_path_rate：
    # Dropout 和 DropPath 比例；
    #
    # depth：
    # 自注意力模块层数；
    #
    # device：
    # 设备字符串。
    #
    # 注意：
    # 当前实现没有直接使用 device 参数，
    # 但仍然保留该参数。
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        # 调用父类初始化方法。
        super().__init__()

        # 将 drop_path_rate 保存到局部变量 dpr。
        #
        # dpr 通常是 Drop Path Rate 的缩写。
        dpr = drop_path_rate

        # 创建多个 SelfAttentionBlock。
        #
        # 每一层都会：
        #
        # 1. 使用多头注意力融合 token 信息；
        # 2. 使用 MLP 进一步变换特征；
        # 3. 使用残差连接保留原始信息。
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        # 创建最终 LayerNorm。
        self.norm = nn.LayerNorm(hidden_dim)

    # 定义全局融合流程。
    #
    # x 的形状：
    #
    # [B, token_num, hidden_dim]
    #
    # mask 的形状：
    #
    # [B, token_num]
    #
    # 通常：
    # True 表示 token 无效；
    # False 表示 token 有效。
    def forward(self, x, mask):

        # 强制将每个 batch 中索引为 0 的 token 标记为有效。
        #
        # Encoder 中 token 的拼接顺序为：
        #
        # 1. 动态交通参与者；
        # 2. 静态物体；
        # 3. 车道。
        #
        # 因此，第 0 个 token 通常对应第一个动态交通参与者。
        #
        # 将其设置为 False 可以确保：
        # 至少存在一个不会被 key_padding_mask 屏蔽的 token。
        #
        # 这有助于避免某个 batch 的全部 token 均被屏蔽。
        mask[:, 0] = False

        # 依次执行所有 SelfAttentionBlock。
        #
        # 每执行一层，所有场景 token 都会进一步交换信息。
        for b in self.blocks:
            x = b(x, mask)

        # 对最终融合结果执行 LayerNorm 并返回。
        #
        # 输出形状：
        #
        # [B, token_num, hidden_dim]
        return self.norm(x)
