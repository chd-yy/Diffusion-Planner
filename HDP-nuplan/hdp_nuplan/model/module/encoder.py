# PyTorch 用于张量拼接、mask、三角函数和占位张量等操作。
import torch
# nn 提供 Module、Linear、LayerNorm、MultiheadAttention、Embedding 和 ModuleList。
import torch.nn as nn
# timm Mlp 用于历史轨迹、静态目标、lane 和注意力后的前馈投影。
from timm.models.layers import Mlp
# DropPath 对残差分支执行随机深度正则化。
from timm.layers import DropPath

# 【HDP 与原 Diffusion-Planner 的唯一区别：包命名空间】MixerBlock 的导入路径由
# diffusion_planner.model.module.mixer 改为 hdp_nuplan.model.module.mixer。
# 【实现核对：其余部分完全一致】将该包路径统一后，两个 encoder.py 的六个类、
# 所有方法、参数、张量操作和返回值的可执行 AST 完全相同。
from hdp_nuplan.model.module.mixer import MixerBlock


# 场景编码总入口：分别编码动态交通参与者、静态物体和 lane，再通过全局自注意力融合。
# 输出 encoder_outputs["encoding"]，形状为 [B,token_num,hidden_dim]。
class Encoder(nn.Module):
    # config 提供各类 token 数量、输入长度、隐藏维度、层数、注意力头和正则化比例。
    def __init__(self, config):
        # 初始化父类并注册后续子模块。
        super().__init__()

        # 三类场景 token 最终统一投影到该隐藏维度。
        self.hidden_dim = config.hidden_dim

        # token 总数 = 动态目标数 + 静态目标数 + lane 数；route 由 Decoder 单独编码。
        self.token_num = config.agent_num + config.static_objects_num + config.lane_num

        # 每个动态目标的历史状态序列编码为一个 token。
        self.neighbor_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        # 每个静态物体直接投影为一个 token。
        self.static_encoder = StaticFusionEncoder(config.static_objects_state_dim, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim)
        # 每条 lane 的离散几何、限速和交通灯编码为一个 token。
        self.lane_encoder = LaneFusionEncoder(config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
    
        # 在全部场景 token 之间执行多层自注意力，建模跨类型交互。
        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, 
            drop_path_rate=config.encoder_drop_path_rate, 
            depth=config.encoder_depth, 
            device=config.device
        )

        # 位置/类型输入为 7 维：[x,y,cos(yaw),sin(yaw),type_agent,type_static,type_lane]。
        self.pos_emb = nn.Linear(7, config.hidden_dim)

    # inputs 至少包含 neighbor_agents_past、static_objects、lanes 及 lane 限速字段。
    def forward(self, inputs):

        # 创建输出字典，保持 Decoder 和训练入口约定的接口。
        encoder_outputs = {}

        # 动态交通参与者历史：[B,P_agent,V_history,D_agent]。
        neighbors = inputs['neighbor_agents_past']

        # 静态物体：[B,P_static,D_static]。
        static = inputs['static_objects']

        # lane 几何：[B,P_lane,V_lane,D_lane]，以及每条 lane 的限速值/是否已知。
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        # 从动态目标输入读取 batch size。
        B = neighbors.shape[0]

        # 三个子编码器均返回：token 特征、padding mask、7维位置/类型特征。
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        # 按“动态目标 -> 静态目标 -> lane”拼接为 [B,token_num,hidden_dim]。
        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        # 同顺序拼接位置特征和 mask，并展平 batch/token 以仅投影有效位置。
        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        # 只对有效 token 计算 Linear，避免其 bias 让 padding 位置产生非零位置 embedding。
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        # 建立全零结果并把有效 embedding 写回原 token 位置。
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        # 为每个有效场景 token 加上位置和类型 embedding。
        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        # 使用 padding mask 做全局自注意力融合。
        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        # 返回统一场景上下文字典。
        return encoder_outputs


# 标准的“注意力 + MLP”残差块，输入/输出均为 [B,token_num,dim]。
class SelfAttentionBlock(nn.Module):
    # dropout 同时用于注意力 Dropout、MLP Dropout 和残差 DropPath。
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        # 初始化父类。
        super().__init__()

        # 注意力前归一化；batch_first=True 约定输入为 [B,T,D]。
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        # dropout>0 时随机丢弃整条残差分支，否则为恒等映射。
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        # MLP 前归一化，隐藏维度为 dim*mlp_ratio。
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    # mask:[B,T]，True 表示该 token 是 padding，应作为 key/value 被忽略。
    def forward(self, x, mask):
        # 注意：当前实现只对 Query 使用 norm1(x)，Key/Value 仍是原始 x；[0] 取注意力输出。
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        # 第二条残差分支执行 Pre-Norm MLP。
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 形状保持 [B,T,D]。
        return x


# 将每个动态交通参与者的历史序列编码为一个固定长度 token：
# 有效性特征 -> 通道投影 -> 时间 token 投影 -> Mixer -> 平均池化 -> 类型融合。
class AgentFusionEncoder(nn.Module):
    # time_len 是历史点数；depth 是 MixerBlock 数；hidden_dim 是最终 token 维度。
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        # 初始化父类。
        super().__init__()

        # 保存输出维度和 Mixer 中间通道数。
        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        # 输入最后 3 维是目标类型 one-hot，通过 Linear 映射到通道维。
        self.type_emb = nn.Linear(3, channels_mlp_dim)

        # 每个历史点使用 8 维运动/尺寸状态 + 1 维有效性标记，共 9 维。
        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        # 沿 time_len 历史 token 维投影到 tokens_mlp_dim。
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 多层 MLP-Mixer 融合时间点与通道信息。
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        # 池化后先归一化，再映射到统一 hidden_dim。
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        '''
        # 类型在同一目标历史中保持不变，这里从最后一个时间点读取 3 维 one-hot。
        neighbor_type = x[:, :, -1, 8:]
        # 几何/运动主分支只保留前 8 维。
        x = x[..., :8]

        # 使用最后历史点构造位置 embedding 基础：[x,y,cos,sin,agent_type_one_hot]。
        pos = x[:, :, -1, :7].clone() # x, y, cos, sin
        # 最后 3 维覆盖为动态目标类型 [1,0,0]。
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0
        
        # B:batch，P:目标数，V:历史点数。
        B, P, V, _ = x.shape
        # mask_v:[B,P,V]，前 8 维全零的历史点视为 padding。
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        # mask_p:[B,P]，没有任何有效历史点的目标视为 padding token。
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        # 追加 1 维有效性特征，使网络显式区分真实零值和 padding。
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        # 合并 batch/agent，得到 [B*P,V,9]。
        x = x.view(B * P, V, -1)

        # 只编码有效目标，减少 padding 计算。
        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        # 投影通道后交换为 [N_valid,C,V]，沿历史 token 维做预投影，再恢复维度顺序。
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        # 多层 MixerBlock 融合时间和通道。
        for block in self.blocks:
            x = block(x)  

        # 沿 Mixer token 维平均池化为每个目标一个 channels_mlp_dim 向量。
        x = torch.mean(x, dim=1)

        # 展平并筛选有效目标，计算类型 embedding 后与轨迹特征相加。
        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        # 归一化并映射到 hidden_dim。
        x = self.emb_project(self.norm(x))

        # padding 目标保持全零，有效目标写回原位置。
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 返回 token:[B,P,H]、mask:[B,P]、位置/类型:[B,P,7]。
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

    
# 静态物体没有历史时间序列，因此直接用 MLP 把每个对象状态映射成 hidden_dim token。
class StaticFusionEncoder(nn.Module):
    # dim 是静态状态维度；device 参数为兼容保留，当前实现没有直接使用。
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        # 初始化父类。
        super().__init__()

        # 保存输出维度，用于创建全零占位结果。
        self._hidden_dim = hidden_dim

        # 将每个静态对象的 dim 维状态直接映射到 hidden_dim。
        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        ''' 
        # B 是 batch，P 是最多静态对象数。
        B, P, _ = x.shape

        # 构造位置 embedding 基础：[x,y,cos,sin,static_type_one_hot]。
        pos = x[:, :, :7].clone() # x, y, cos, sin
        # 最后 3 维覆盖为静态目标类型 [0,1,0]。
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        # 先创建 [B*P,H] 全零结果，padding 静态目标将一直保持零。
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        # 前 10 维全零表示该静态目标无效，mask_p:[B,P]。
        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        # 展平 mask，True 表示有效目标。
        valid_indices = ~mask_p.view(-1) 

        # 只有 batch 内至少存在一个有效静态目标时才执行 MLP，避免空张量路径。
        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        # 返回 token、padding mask 和 7 维位置/类型特征。
        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)
    

# 将每条 lane 的离散几何序列编码为一个 token，并融合限速是否已知、限速值和交通灯状态。
class LaneFusionEncoder(nn.Module):
    # lane_len 是每条 lane 的固定采样点数；depth 是 MixerBlock 层数。
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        # 初始化父类。
        super().__init__()

        # 保存 lane 点数和中间通道维度。
        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        # 已知限速通过 Linear 编码；未知限速使用一个可学习 embedding；交通灯为 4 维输入。
        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        # 每个 lane 点使用中心线/左右边界相对几何的前 8 维。
        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        # 沿 lane_len 点序列投影到 tokens_mlp_dim。
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 多层 MixerBlock 融合 lane 点和通道。
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        # 池化后归一化并投影到统一 hidden_dim。
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        # 交通灯状态在同一 lane 的点间共享，这里从第一个采样点读取后 4 维。
        traffic = x[:, :, 0, 8:]
        # 几何主分支只保留前 8 维。
        x = x[..., :8]

        # 用 lane 中间点构造位置 embedding 的几何基准。
        pos = x[:, :, int(self._lane_len / 2), :7].clone() # x, y, x'-x, y'-y
        # 根据中心线方向向量 (dx,dy) 计算 heading，并转换为连续 cos/sin 表示。
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # 最后 3 维覆盖为 lane 类型 [0,0,1]，最终位置特征为 7 维。
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        # B:batch，P:lane 数，V:每条 lane 点数。
        B, P, V, _ = x.shape
        # mask_v:[B,P,V]，8 维几何全零表示 padding 点。
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        # mask_p:[B,P]，没有有效点的整条 lane 是 padding token。
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        # 合并 batch/lane，得到 [B*P,V,8]。
        x = x.view(B * P, V, -1)

        # 只对有效 lane 执行后续编码。
        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        # 通道投影 -> 维度置换 -> lane token 投影 -> 恢复维度。
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        # 多层 MixerBlock 融合 lane 几何。
        for block in self.blocks:
            x = block(x)  

        # 沿 Mixer token 维平均池化为每条 lane 一个向量。
        x = torch.mean(x, dim=1)

        # 将限速、是否已知和交通灯展平到与 B*P lane 顺序一致。
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # 只保留有效 lane；squeeze 后 has_speed_limit 作为一维布尔索引。
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        # 为所有有效 lane 创建限速 embedding 占位结果。
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        # 当前 batch 至少有一条已知限速 lane 时，才使用 speed_limit_emb。
        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        # 至少有一条未知限速 lane 时，才读取 unknown_speed_emb；这两个数据依赖分支会让
        # 某个 batch 条件性不使用其中一组参数，因此训练 DDP 需要允许 unused parameters。
        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # 筛选有效 lane，并把 4 维交通灯状态映射到通道维。
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data


        # 融合 lane 几何、限速和交通灯三类特征，再映射到 hidden_dim。
        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        # padding lane 保持零，有效 lane 写回原位置。
        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 返回 token:[B,P,H]、mask:[B,P]、位置/类型:[B,P,7]。
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


# 全局融合编码器：在动态目标、静态目标和 lane token 之间执行多层自注意力。
class FusionEncoder(nn.Module):
    # device 参数为接口兼容保留，当前实现没有直接使用。
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        # 初始化父类。
        super().__init__()

        # dpr 同时传给 SelfAttentionBlock 的注意力 Dropout、MLP Dropout 和 DropPath。
        dpr = drop_path_rate

        # 堆叠 depth 个全局自注意力块。
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        # 所有融合层完成后的最终 LayerNorm。
        self.norm = nn.LayerNorm(hidden_dim)

    # x:[B,token_num,H]；mask:[B,token_num]，True 表示 padding token。
    def forward(self, x, mask):

        # 强制第 0 个动态目标 token 有效，确保每个场景至少有一个未屏蔽 key，
        # 避免 MultiheadAttention 在全部 token 都被 mask 时产生异常数值。
        # 这是原版和 HDP 完全相同的原地 mask 修改。
        mask[:, 0] = False

        # 依次融合全部场景 token。
        for b in self.blocks:
            x = b(x, mask)

        # 返回最终归一化的场景上下文，形状不变。
        return self.norm(x)
