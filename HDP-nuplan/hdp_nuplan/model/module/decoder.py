import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from hdp_nuplan.model.diffusion_utils.sampling import dpm_sampler
from hdp_nuplan.model.diffusion_utils.sde import SDE, VPSDE_linear
from hdp_nuplan.utils.normalizer import ObservationNormalizer, StateNormalizer
from hdp_nuplan.model.module.mixer import MixerBlock
from hdp_nuplan.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer


# 扩散解码器：训练时预测给定带噪轨迹的扩散目标；推理/RL rollout 时从噪声采样轨迹。
class Decoder(nn.Module):
    """连接场景 Encoder、轨迹 DiT 与扩散采样器的上层解码模块。

    整体数据流：
        训练/验证：带噪未来轨迹 + 扩散时间 + 场景条件 -> DiT -> 扩散监督输出。
        普通推理：随机噪声 + 场景条件 -> DPM-Solver -> 1 条自车轨迹。
        RL rollout：随机噪声 + 场景条件 -> DPM-Solver -> G 条候选自车轨迹。

    【HDP 与原 Diffusion-Planner 的总体区别】原版联合生成自车与邻车，并可使用
    classifier guidance；HDP 只生成自车，将多候选采样暴露为独立的 sample 接口。
    """
    def __init__(self, config):
        super().__init__()

        # decoder_drop_path_rate 在当前实现中作为 DiTBlock 的 dropout 参数传入。
        dpr = config.decoder_drop_path_rate
        # 【HDP 与原 Diffusion-Planner 的区别：预测对象】HDP 只预测自车未来轨迹，
        # 因此不再保存原版的 predicted_neighbor_num。
        self._future_len = config.future_len
        # 与原版一致，使用线性 beta 调度的 VP-SDE。
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            # 【区别：DiT 单 token 输出】HDP 每个未来时间 token 输出 4 维；
            # 原版每个 agent token 一次输出“当前+全部未来”的 (future_len+1)*4 维。
            output_dim=4, # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
            # 【区别：时间位置 embedding】HDP 额外传入未来长度，用于为每个未来点建表。
            future_length=config.future_len
        )
        # 与当前 Diffusion-Planner 的 (自车+邻车) 联合输出不同，HDP 的输出维度固定为
        # 单个自车未来点 [x/y 或增量、cos、sin]，邻车预测在 HDP 中已弃用。
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        # 【区别：无 guidance_fn】原版还保存 config.guidance_fn 供 classifier guidance；
        # HDP 的采样器调用不使用该配置，因此这里没有 self._guidance_fn。
        
    @property
    def sde(self):
        # @property 让外部通过 decoder.sde 只读访问底层 SDE。
        return self._sde

    @torch.no_grad()
    def sample(
        self,
        encoder_outputs,
        inputs,
        num_samples=1,
        diffusion_steps=10,
        noise_scale=0.1,
    ):
        """显式生成多条自车候选轨迹，供普通推理和 NuPlan RL rollout 共用。

        【HDP 与原 Diffusion-Planner 的区别】这是 HDP-nuPlan 新增的统一采样入口。
        原版把单次采样直接写在 forward 的推理分支中，不能显式指定候选数 G。

        Returns:
            [B, G, T, 4]，最后一维为 [x, y, cos(yaw), sin(yaw)]。

        其中 B 是场景数，G 是每场景候选数，T 是未来时间点数。前两维在
        normalizer.inverse 后仍是逐帧位移，需要通过 cumsum 恢复累计位置。
        """
        # 防止空候选集在 repeat/reshape 时产生难以定位的错误。
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        # encoding: [B,N,D]；route_lanes: [B,P,V,D_route]；ego_v: [B,2]。
        encoding = encoder_outputs["encoding"]
        route_lanes = inputs["route_lanes"]
        # 当前状态第 4、5 维是 vx、vy，只取速度作为 DiT 的自车状态条件。
        ego_v = inputs["ego_current_state"][:, 4:6]
        batch_size = encoding.shape[0]

        # 每个场景复制 G 份条件，并一次性展平到 batch 维执行扩散采样。
        # repeat_interleave(..., dim=0) 的排列是 scene0 重复 G 次，再到 scene1。
        encoding = encoding.repeat_interleave(num_samples, dim=0)
        route_lanes = route_lanes.repeat_interleave(num_samples, dim=0)
        ego_v = ego_v.repeat_interleave(num_samples, dim=0)
        sample_batch = batch_size * num_samples
        # 【区别：初始噪声】HDP 直接采样 [B*G,T,4] 的纯噪声；普通推理默认缩放为 0.1，
        # RL rollout 可显式传入更大 noise_scale 以提高同场景候选多样性；
        # 原版采样 [B,P,T,4]、缩放为 0.5，并把当前状态拼在第一个时间位置。
        xT = torch.randn(
            sample_batch,
            self._future_len,
            4,
            device=encoding.device,
            dtype=encoding.dtype,
        ) * noise_scale

        # DPM-Solver 从终点噪声 xT 逐步求解到预测的干净轨迹 x0。
        # 【区别：无 classifier guidance/初态纠正】HDP 不传原版的 guidance_fn，
        # 也不使用 correcting_xt_fn 把第一个位置反复固定为当前状态。
        # x0 : [sample_batch, future_len, 4] = [B × G, T, 4]
        x0 = dpm_sampler(
            self.dit,
            xT,
            # sampling.py 会把这些条件按关键字转发给 DiT.forward。
            other_model_params={
                "cross_c": encoding,
                "route_lanes": route_lanes,
                "ego_current_states": ego_v,
            },
            diffusion_steps=diffusion_steps,
            dpm_solver_params={},
            model_wrapper_params={},
        )
        # 将训练使用的标准化轨迹恢复到真实量纲。
        x0 = self._state_normalizer.inverse(x0.reshape(sample_batch, -1, 4))
        # HDP 预测前两维的逐帧位移，通过积分恢复 NuPlan 自车局部坐标轨迹。
        x0 = torch.cat(
            [torch.cumsum(x0[..., :2], dim=-2), x0[..., 2:]],
            dim=-1,
        )
        # 恢复被展平的候选维，得到 [B,G,T,4]。
        return x0.reshape(batch_size, num_samples, self._future_len, 4)
    
    def forward(self, encoder_outputs, inputs):
        """
        扩散解码流程。

        Args:
            encoder_outputs["encoding"]: 场景上下文编码 [B,N,D]。
            inputs["ego_current_state"]: 当前自车状态。
            inputs["route_lanes"]: 路线折线特征。
            inputs["sampled_trajectories"]: 仅训练/验证扩散目标时存在，[B,T,4]。
            inputs["diffusion_time"]: 扩散时间 t，形状 [B]。

        Returns:
            有扩散训练输入时返回 score: [B,T,4]；否则返回 prediction: [B,1,T,4]。
        """
        # 取 Encoder 输出的 agent、静态物体和地图上下文编码。
        ego_neighbor_encoding = encoder_outputs['encoding']
        B = ego_neighbor_encoding.shape[0]
        route_lanes = inputs['route_lanes']
        # HDP 将当前自车速度 (vx, vy) 作为 DiT 条件；原实现主要使用邻车 mask 和当前状态序列。
        ego_v = inputs['ego_current_state'][:, 4:6]

        # 输出契约由输入是否包含扩散训练状态决定，而不是由 module.training 决定。
        # 这样 model.eval() 可以关闭 dropout，同时仍计算验证集 diffusion loss。
        # Python 的 `in` 用于判断字典中是否存在对应 key；and 要求两个 key 同时存在。
        has_diffusion_state = (
            'sampled_trajectories' in inputs
            and 'diffusion_time' in inputs
        )
        if has_diffusion_state:
            # 【区别：训练张量布局】HDP 是 [B,T,4] 的单自车序列；原版 reshape 成
            # [B,P,(T+1)*4]，P=1+predicted_neighbor_num，并联合预测自车和邻车。
            sampled_trajectories = inputs['sampled_trajectories'] # [B, V_future, 4]
            diffusion_time = inputs['diffusion_time']
            # 保留迁移接口变量；HDP 的未来时间 token 全部有效，实际不会把该 mask 传入 DiT。
            neighbor_current_mask = None

            return {
                    # self.dit 输出 [B,T,4]；reshape 只显式整理最后一个状态维。
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        ego_neighbor_encoding,
                        route_lanes,
                        ego_v,
                    ).reshape(B, -1, 4)
                }
        else:
            return {
                    # 保留候选轨迹维，使 NuPlan planner 的 [0, 0] 索引与输出一致。
                    # 【区别：推理复用 sample】原版在此内联构造噪声、初态约束和 guidance；
                    # HDP 调用统一采样入口，普通推理默认每场景生成 1 条候选轨迹。
                    "prediction": self.sample(encoder_outputs, inputs, num_samples=1)
                }

        
class RouteEncoder(nn.Module):
    """把整组 route lane 折线压缩为每个场景一个全局路线条件向量。

    本类的可执行实现与原 Diffusion-Planner 一致，下面迁移其张量形状和 mask 注释。

    这里使用 MLP-Mixer，而不是 attention：先在每个点的特征 channel 上投影，
    再把 P*V 个路线点作为 token 混合，最终平均池化成 [B,hidden_dim]。
    """
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        # 先逐点把 4 维路线特征投影到 channels_mlp_dim。
        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        # 再沿 route_num*lane_len 个 token 做 token-mixing 投影。
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # MixerBlock 依次混合 token 维和 channel 维的信息。
        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        # 把聚合后的 channels_mlp_dim 特征映射到 DiT hidden_dim。
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: [B,P,V,D]，P 为 route lane 数，V 为每条 lane 的采样点数。
        '''
        # HDP 的路线条件只取每个点前 4 维；边界、限速和红绿灯不进入该路线编码器。
        x = x[..., :4]

        B, P, V, _ = x.shape
        # mask_v: [B,P,V]，某点前4维全为0时为 True，表示 padding 点。
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        # mask_p: [B,P]，一条 lane 的所有点都无效时为 True。
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        # mask_b: [B]，一个场景的所有 route lane 都无效时为 True。
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        # torch.ne(..., 0) 逐元素判断非零；~ 对 bool mask 取反；sum==0 判断是否全无效。
        # 将 lane 维 P 和点维 V 合并为 token 维：[B,P,V,4] -> [B,P*V,4]。
        x = x.view(B, P * V, -1)

        # 只让至少包含一条有效路线的场景进入 MLP-Mixer，避免全 padding 干扰网络。
        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        # channel projection: [B_valid,P*V,4] -> [B_valid,P*V,C]。
        x = self.channel_pre_project(x)
        # permute 交换 token/channel 维，让 token_pre_project 在 P*V 维上工作。
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        # 沿 token 维平均池化，得到每个有效场景一个路线向量。
        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        # 为全 padding 场景补回全零向量，并恢复原 batch 顺序。
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 输出 [B,hidden_dim]，作为所有 DiTBlock 共享的全局条件。
        return x_result.view(B, -1)


class DiT(nn.Module):
    """HDP 的轨迹级 Diffusion Transformer 主干。

    每个未来时间点对应一个 token。DiT 先叠加时间位置和当前速度条件，再通过
    多层 DiTBlock 读取 Encoder 场景上下文，最后为每个时间点输出4维扩散结果。
    """
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start", future_length=80):
        super().__init__()
        
        # 【HDP 与原 Diffusion-Planner 的区别：预测参数化】HDP 支持 noise、score、
        # x_start、v 四种参数化；原版这里只允许 score 和 x_start。
        assert model_type in ["noise", "score", "x_start", "v"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        # 这里的 embedding 索引表示未来时间位置，而不是原实现中的 agent 类型。
        # 【区别：embedding 表含义和大小】HDP 建立 future_length 个时间位置 embedding；
        # 原版只建立 2 个类型 embedding，分别表示 ego 和 neighbor。
        self.agent_embedding = nn.Embedding(future_length, hidden_dim)
        # 【区别：输入布局】HDP 每个 token 只有一个未来点的4维状态；原版每个
        # agent token 输入的是展平后的“当前+全部未来”状态。
        # Mlp 是 timm 封装的两层前馈网络，这里完成 4 -> 512 -> hidden_dim 的映射。
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        # 新增当前自车速度条件投影层，为每个未来 token 注入 vx/vy 信息。
        self.ego_state_proj = nn.Linear(2, hidden_dim)
        # 把标量扩散时间 t 编码成 [B,hidden_dim] 条件向量。
        self.t_embedder = TimestepEmbedder(hidden_dim)
        # DiT 主干会同时接收 token 级条件、全局条件和场景上下文。
        #
        # 条件来源可以分为三类：
        # 1. token 级条件：未来时间位置 embedding；
        # 2. token 级条件：当前自车 vx/vy embedding；
        # 3. 全局条件：route encoding 与 diffusion timestep embedding 之和。
        #
        # 带噪轨迹 x：
        # 每个未来时间点是一个 token，形状为 [B,T,hidden_dim]。
        #
        # 扩散时间 t：
        # 每个场景一个标量，经 TimestepEmbedder 变成 [B,hidden_dim]。
        #
        # 场景上下文 cross_c：
        # Encoder 输出的 N 个上下文 token，通过交叉注意力提供地图和动态目标信息。
        #
        # 路线条件 route_lanes：
        # 先由 RouteEncoder 压缩成每场景一个 [B,hidden_dim] 向量。
        #
        # 当前速度 ego_current_states：
        # 先由 ego_state_proj 映射，再广播到全部 T 个未来 token。
        #
        # 每个 DiTBlock 内部依次执行自注意力、条件交叉注意力和 MLP。
        # 自注意力建模未来时间点之间的轨迹一致性；交叉注意力读取场景上下文；
        # MLP 在每个 token 内进行非线性特征变换。
        #
        # nn.ModuleList 会把所有 block 注册为子模块，确保 optimizer、to(device)、
        # state_dict 和 checkpoint 能自动遍历这些层。
        # depth 决定重复堆叠多少个 DiTBlock，但不会改变张量的 [B,T,hidden_dim] 形状。
        # dropout 与 mlp_ratio 会继续传给每一个 DiTBlock。
        # block 数量由配置项 decoder_depth 决定。
        # ModuleList 会注册 depth 个 DiTBlock，使其参数参与训练和 checkpoint 保存。
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        # 所有 block 都保持 hidden_dim 不变，便于通过残差连接稳定堆叠。
        # block 的输出仍然是 [B,T,hidden_dim]。
        #
        # 最终层不负责扩散采样，只负责把隐藏特征转换成指定的模型参数化输出。
        # score/noise/x_start/v 的语义由 model_type 和 loss.py 共同决定。
        # 采样阶段再由 sampling.py 根据 model_type 将网络输出解释为求解器需要的形式。
        # 将每个未来 token 的 hidden_dim 特征投影回 output_dim=4。
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, ego_current_states):
        """
        DiT 前向传播。
        x: [B,T,output_dim]，扩散过程中的带噪自车未来轨迹。
        t: [B]，每个样本的扩散时间。
        cross_c: [B,N,D]，Encoder 产生的交叉注意力上下文。
        """
        # 【HDP 与原 Diffusion-Planner 的区别：第二维含义】这里 T 是未来时间点数；
        # 原版对应位置是预测对象数 P=1+predicted_neighbor_num。
        B, T, _ = x.shape
        
        # 将每个时间点的 4 维带噪状态映射到 hidden_dim。
        x = self.preproj(x)

        # 取出所有未来位置 embedding 并扩展 batch，形状为 [B,future_length,hidden_dim]。
        # 正常配置要求输入 T 与构造时 future_length 一致，否则下面相加会形状不匹配。
        x_embedding = self.agent_embedding.weight[None, :, :].expand(B, -1, -1) # (B, T, D) 
        # [B,2] 的 vx/vy 投影成 [B,hidden_dim]，再增加 token 轴用于广播。
        ego_state_embedding = self.ego_state_proj(ego_current_states)
        ego_state_embedding = ego_state_embedding[:, None, :]
        # 【HDP 与原 Diffusion-Planner 的区别：token 输入条件】HDP 加“未来位置 embedding
        # + 当前速度 embedding”；原版只加 ego/neighbor 类型 embedding。
        x = x + x_embedding + ego_state_embedding

        # RouteEncoder 输出 [B,hidden_dim]，与扩散时间 embedding 相加得到全局条件 y。
        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)
        
        # 【HDP 与原 Diffusion-Planner 的区别：无邻车 attention mask】原版构造 [B,P]
        # mask 屏蔽 padding 邻车并传给每个 DiTBlock；HDP 全部 T 个未来 token 都有效，
        # 因此不创建 attn_mask，调用 block 时也少一个 mask 参数。
        for block in self.blocks:
            x = block(x, cross_c, y)
        
        # 把每个未来 token 投影回 4 维扩散输出，形状为 [B,T,4]。
        x = self.final_layer(x, y)
        
        # 与原版一致，score 输出除以 sigma(t)；1e-6 防止小 t 时除零。
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        # 【HDP 与原 Diffusion-Planner 的区别：直接返回类型】原版仅直接返回 x_start；
        # HDP 对 x_start、noise、v 都直接返回网络原始输出，loss 再按监督类型转换。
        elif self._model_type == "x_start" or self._model_type == "noise" or self._model_type == 'v':
            return x
        # 理论上非法类型已在 __init__ 的 assert 被拦截，这里保留防御性异常。
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
