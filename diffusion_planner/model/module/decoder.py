# 导入 Python 标准库中的 math 模块。
#
# 当前文件本身没有直接调用 math 中的函数，
# 但仍然保留原始导入语句。
import math

# 导入 PyTorch。
#
# 当前文件主要使用 PyTorch 完成：
# 1. 张量拼接；
# 2. 张量变形；
# 3. 随机噪声生成；
# 4. 掩码计算；
# 5. 占位张量创建；
# 6. 模型前向传播。
import torch

# 导入 PyTorch 神经网络模块，并命名为 nn。
#
# 当前文件会使用：
# 1. nn.Module；
# 2. nn.Embedding；
# 3. nn.LayerNorm；
# 4. nn.ModuleList；
# 5. nn.GELU。
import torch.nn as nn

# 从 timm 中导入封装好的 MLP 模块。
#
# Mlp 通常包含：
#
# Linear
# → 激活函数
# → Dropout
# → Linear
# → Dropout
from timm.models.layers import Mlp

# 从 timm 中导入 DropPath。
#
# 注意：
# 当前文件中并没有直接使用 DropPath，
# 但仍然保留原始导入语句。
from timm.layers import DropPath

# 导入 DPM-Solver 采样器。
#
# 在推理阶段，Decoder 会调用 dpm_sampler(...)，
# 从随机噪声出发逐步生成未来轨迹。
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler

# 导入随机微分方程接口和线性 VP-SDE。
#
# SDE：
# 随机微分方程抽象基类。
#
# VPSDE_linear：
# 使用线性 beta 调度的 Variance Preserving SDE。
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear

# 导入观测量归一化器和状态归一化器。
#
# ObservationNormalizer：
# 用于处理场景输入，例如邻居车辆、车道和静态目标。
#
# StateNormalizer：
# 用于处理模型预测轨迹中的状态量。
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

# 导入项目中定义的 MixerBlock。
#
# MixerBlock 使用 MLP-Mixer 风格的结构，
# 用于融合 token 维度和通道维度的信息。
from diffusion_planner.model.module.mixer import MixerBlock

# 导入 DiT 相关基础模块。
#
# TimestepEmbedder：
# 将扩散时间步 t 转换为隐藏向量。
#
# DiTBlock：
# 包含自注意力、交叉注意力和 adaLN 条件调制的基础模块。
#
# FinalLayer：
# 将 DiT 隐藏特征映射回轨迹状态空间。
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer


# 定义扩散模型解码器。
#
# Decoder 的作用是：
#
# 训练阶段：
# 1. 读取带噪轨迹；
# 2. 调用 DiT；
# 3. 返回模型预测结果。
#
# 推理阶段：
# 1. 构造包含当前状态和未来随机噪声的初始轨迹；
# 2. 使用 DPM-Solver 执行反向扩散采样；
# 3. 可选地使用 classifier guidance 调整生成过程；
# 4. 返回未来轨迹预测结果。
class Decoder(nn.Module):
    # 初始化 Decoder。
    #
    # config 通常需要包含：
    #
    # config.decoder_drop_path_rate：
    # Decoder 中使用的 Dropout 比例；
    #
    # config.predicted_neighbor_num：
    # 需要联合预测的邻居车辆数量；
    #
    # config.future_len：
    # 未来轨迹长度；
    #
    # config.route_num：
    # 路线中包含的车道段数量；
    #
    # config.lane_len：
    # 每条车道包含的离散采样点数量；
    #
    # config.encoder_drop_path_rate：
    # RouteEncoder 中使用的 Dropout 比例；
    #
    # config.hidden_dim：
    # 隐藏特征维度；
    #
    # config.decoder_depth：
    # DiTBlock 堆叠层数；
    #
    # config.num_heads：
    # 多头注意力头数；
    #
    # config.diffusion_model_type：
    # 扩散模型输出类型；
    #
    # config.state_normalizer：
    # 状态归一化器；
    #
    # config.observation_normalizer：
    # 观测量归一化器；
    #
    # config.guidance_fn：
    # 可选的 guidance 函数。
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法。
        super().__init__()

        # 读取 Decoder 中使用的 Dropout 比例。
        #
        # dpr 通常是 Drop Path Rate 的缩写。
        #
        # 在当前实现中，它会作为 dropout 参数传递给 DiTBlock。
        dpr = config.decoder_drop_path_rate

        # 保存需要预测的邻居车辆数量。
        #
        # 总预测目标数量为：
        #
        # 1 个自车 + predicted_neighbor_num 个邻居车辆。
        self._predicted_neighbor_num = config.predicted_neighbor_num

        # 保存未来轨迹长度。
        #
        # 每个目标最终需要预测 future_len 个未来时间点。
        self._future_len = config.future_len

        # 创建线性 VP-SDE。
        #
        # 该对象用于：
        # 1. 描述正向扩散过程；
        # 2. 计算不同扩散时间下的噪声标准差；
        # 3. 为反向扩散采样提供噪声调度信息。
        self._sde = VPSDE_linear()

        # 创建 DiT 模型。
        #
        # DiT 会：
        # 1. 编码当前带噪轨迹；
        # 2. 编码路线信息；
        # 3. 编码扩散时间步；
        # 4. 使用多个 DiTBlock 融合场景上下文；
        # 5. 输出轨迹预测结果。
        self.dit = DiT(
            # 传入线性 VP-SDE。
            sde=self._sde, 

            # 创建路线编码器。
            #
            # RouteEncoder 会将 route_lanes 编码为一个隐藏向量，
            # 并与扩散时间步嵌入相加后作为 DiT 条件向量。
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),

            # DiTBlock 的层数。
            depth=config.decoder_depth, 

            # 输出维度。
            #
            # 每个目标的轨迹包含：
            #
            # future_len + 1
            #
            # 个时间点。
            #
            # 多出的 1 个时间点表示当前时刻。
            #
            # 每个时间点包含 4 个状态量：
            #
            # [x, y, cos, sin]
            output_dim= (config.future_len + 1) * 4, # x, y, cos, sin

            # 隐藏特征维度。
            hidden_dim=config.hidden_dim, 

            # 多头注意力头数。
            heads=config.num_heads, 

            # Dropout 比例。
            dropout=dpr,

            # 模型输出类型。
            #
            # 可选值：
            #
            # "score"
            # 或
            # "x_start"
            model_type=config.diffusion_model_type
        )
        
        # 保存状态归一化器。
        #
        # 该对象用于将模型输出从归一化空间
        # 恢复到真实物理空间。
        self._state_normalizer: StateNormalizer = config.state_normalizer

        # 保存观测量归一化器。
        #
        # guidance 函数可能需要读取真实尺度下的输入场景，
        # 因此需要使用该对象执行逆归一化。
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
        # 保存可选的 guidance 函数。
        #
        # 如果该值为 None，
        # 则推理阶段使用无条件采样。
        #
        # 如果该值不为 None，
        # 则 DPM-Solver 会在采样过程中应用 classifier guidance。
        # 训练入口的 argparse 配置没有 guidance_fn；没有显式提供时，
        # 采用 None，保持原始无 guidance 的训练/采样路径。
        self._guidance_fn = getattr(config, "guidance_fn", None)
        
    # 将 sde 暴露为只读属性。
    #
    # 外部代码可以通过 decoder.sde 访问线性 VP-SDE。
    @property
    def sde(self):
        # 返回内部保存的 SDE 对象。
        return self._sde
    
    # 定义 Decoder 的前向传播。
    #
    # 该方法包含两个分支：
    #
    # 1. self.training=True：
    #    执行训练阶段前向传播；
    #
    # 2. self.training=False：
    #    执行推理阶段 DPM-Solver 采样。
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """

        # 原始注释：
        # 提取自车和邻居车辆的当前状态。
        # Extract ego & neighbor current states

        # 读取自车当前状态的前 4 个分量。
        #
        # inputs['ego_current_state'] 的常见形状为：
        #
        # [B, D]
        #
        # [:, None, :4] 会：
        # 1. 读取前 4 个状态量；
        # 2. 在目标维度增加一个轴。
        #
        # 输出形状为：
        #
        # [B, 1, 4]
        #
        # 其中 1 表示自车 token。
        ego_current = inputs['ego_current_state'][:, None, :4]

        # 读取需要联合预测的邻居车辆当前状态。
        #
        # inputs["neighbor_agents_past"] 的常见形状为：
        #
        # [B, P_neighbor, V_history, D]
        #
        # [:, :self._predicted_neighbor_num, -1, :4] 表示：
        # 1. 只读取前 predicted_neighbor_num 个邻居；
        # 2. 只读取最后一个历史时间点；
        # 3. 只读取前 4 个状态量。
        #
        # 输出形状为：
        #
        # [B, predicted_neighbor_num, 4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]

        # 判断每一个邻居车辆是否为空。
        #
        # torch.ne(neighbors_current[..., :4], 0)：
        # 判断前 4 个状态量是否不等于 0。
        #
        # torch.sum(..., dim=-1) == 0：
        # 如果一个邻居车辆的 4 个状态量全部为 0，
        # 则认为该车辆无效。
        #
        # neighbor_current_mask 的形状为：
        #
        # [B, predicted_neighbor_num]
        #
        # 通常：
        #
        # True 表示邻居车辆无效；
        # False 表示邻居车辆有效。
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

        # 将邻居车辆有效性掩码写回 inputs。
        #
        # 后续 guidance 函数也可以读取该掩码。
        inputs["neighbor_current_mask"] = neighbor_current_mask

        # 拼接自车和邻居车辆当前状态。
        #
        # 拼接顺序：
        #
        # 1. 自车；
        # 2. 邻居车辆。
        #
        # 输出形状：
        #
        # [B, 1 + predicted_neighbor_num, 4]
        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        # 读取 batch size 和联合预测目标数量。
        B, P, _ = current_states.shape

        # 检查目标总数是否符合预期。
        #
        # P 应当等于：
        #
        # 1 个自车 + predicted_neighbor_num 个邻居车辆。
        assert P == (1 + self._predicted_neighbor_num)

        # 原始注释：
        # 提取上下文编码。
        # Extract context encoding

        # 读取 Encoder 输出的场景上下文编码。
        #
        # 该张量通常包含：
        # 1. 动态交通参与者 token；
        # 2. 静态物体 token；
        # 3. 车道 token。
        #
        # 常见形状：
        #
        # [B, N_context, hidden_dim]
        ego_neighbor_encoding = encoder_outputs['encoding']

        # 读取路线车道数据。
        #
        # 该数据会被 RouteEncoder 编码，
        # 并作为 DiT 的条件信息。
        route_lanes = inputs['route_lanes']

        # 判断当前是否处于训练阶段。
        if self.training:
            # 读取训练阶段使用的带噪轨迹。
            #
            # inputs['sampled_trajectories'] 的常见形状为：
            #
            # [B, P, 1 + V_future, 4]
            #
            # reshape(B, P, -1) 会将时间维度和状态维度压平：
            #
            # [B, P, (1 + V_future) * 4]
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]

            # 读取当前扩散时间。
            #
            # diffusion_time 的形状通常为：
            #
            # [B]
            #
            # 每个样本可以对应不同的扩散时间 t。
            diffusion_time = inputs['diffusion_time']

            # 调用 DiT 预测轨迹。
            #
            # 返回值字典中的键名为 "score"。
            #
            # 注意：
            # 即使 config.diffusion_model_type 为 "x_start"，
            # 当前训练分支仍然统一使用键名 "score"。
            #
            # DiT 输出形状：
            #
            # [B, P, (1 + V_future) * 4]
            #
            # reshape(B, P, -1, 4) 后：
            #
            # [B, P, 1 + V_future, 4]
            return {
                    "score": self.dit(
                        # 输入带噪轨迹。
                        sampled_trajectories, 

                        # 输入扩散时间。
                        diffusion_time,

                        # 输入场景上下文编码。
                        ego_neighbor_encoding,

                        # 输入路线车道。
                        route_lanes,

                        # 输入邻居车辆有效性掩码。
                        neighbor_current_mask
                    ).reshape(B, P, -1, 4)
                }

        # 如果不处于训练阶段，则执行推理采样。
        else:
            # 构造 DPM-Solver 的初始状态 xT。
            #
            # 初始轨迹由两部分拼接得到：
            #
            # 第一部分：
            #
            # current_states[:, :, None]
            #
            # 形状为：
            #
            # [B, P, 1, 4]
            #
            # 表示每个目标的当前状态。
            #
            # 第二部分：
            #
            # torch.randn(B, P, self._future_len, 4)
            #
            # 形状为：
            #
            # [B, P, future_len, 4]
            #
            # 表示未来状态的随机高斯噪声。
            #
            # * 0.5：
            # 将随机噪声幅值缩放为原来的一半。
            #
            # 拼接后形状为：
            #
            # [B, P, 1 + future_len, 4]
            #
            # reshape(B, P, -1) 后形状为：
            #
            # [B, P, (1 + future_len) * 4]
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = torch.cat([current_states[:, :, None], torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5], dim=2).reshape(B, P, -1)

            # 定义初始状态约束函数。
            #
            # DPM-Solver 在每一步更新轨迹后，
            # 都可以调用该函数修正 xt。
            #
            # 该约束的目的为：
            #
            # 强制保持第 0 个时间点不变。
            #
            # 即：
            # 1. 当前自车状态不能被采样过程修改；
            # 2. 当前邻居车辆状态不能被采样过程修改；
            # 3. 只允许未来轨迹发生变化。
            def initial_state_constraint(xt, t, step):
                # 将压平后的轨迹恢复为：
                #
                # [B, P, 1 + future_len, 4]
                xt = xt.reshape(B, P, -1, 4)

                # 将第 0 个时间点覆盖为真实当前状态。
                #
                # 这一步会在每次采样更新后重新执行。
                xt[:, :, 0, :] = current_states

                # 将轨迹重新压平为：
                #
                # [B, P, (1 + future_len) * 4]
                return xt.reshape(B, P, -1)
            
            # 使用 DPM-Solver 执行反向扩散采样。
            #
            # 输入：
            #
            # xT：
            # 包含当前状态和未来随机噪声的初始轨迹；
            #
            # 输出：
            #
            # x0：
            # 反向扩散完成后的预测轨迹。
            x0 = dpm_sampler(
                        # 传入 DiT 模型。
                        self.dit,

                        # 传入终止时刻噪声状态。
                        xT,

                        # 向 DiT.forward(...) 传递额外模型参数。
                        other_model_params={
                            # 场景上下文编码。
                            "cross_c": ego_neighbor_encoding, 

                            # 路线车道信息。
                            "route_lanes": route_lanes,

                            # 邻居车辆掩码。
                            "neighbor_current_mask": neighbor_current_mask                            
                        },

                        # 向 DPM_Solver 构造函数传递额外参数。
                        dpm_solver_params={
                            # 每一步采样后执行 initial_state_constraint，
                            # 强制固定第 0 个时间点。
                            "correcting_xt_fn":initial_state_constraint,
                        },

                        # 向 model_wrapper(...) 传递额外参数。
                        model_wrapper_params={
                            # 可选 classifier guidance 函数。
                            #
                            # 如果 self._guidance_fn 为 None，
                            # 则不使用 guidance。
                            "classifier_fn": self._guidance_fn,

                            # 向 classifier guidance 函数传递额外参数。
                            "classifier_kwargs": {
                                # 传入 DiT 模型。
                                #
                                # guidance 函数可能需要再次调用模型。
                                "model": self.dit,

                                # 保存模型调用时需要的条件参数。
                                "model_condition": {
                                    # 场景上下文编码。
                                    "cross_c": ego_neighbor_encoding, 

                                    # 路线车道信息。
                                    "route_lanes": route_lanes,

                                    # 邻居车辆掩码。
                                    "neighbor_current_mask": neighbor_current_mask                            
                                },

                                # 传入完整 inputs。
                                #
                                # guidance 函数可能需要读取：
                                # 1. 邻居车辆状态；
                                # 2. 车辆尺寸；
                                # 3. 掩码；
                                # 4. 其他场景信息。
                                "inputs": inputs,

                                # 传入观测量归一化器。
                                "observation_normalizer": self._observation_normalizer,

                                # 传入状态归一化器。
                                "state_normalizer": self._state_normalizer
                            },

                            # 设置 guidance 强度。
                            #
                            # 数值越大，guidance 对采样轨迹的影响通常越强。
                            "guidance_scale": 0.5,

                            # 根据是否存在 guidance 函数，
                            # 动态选择采样模式。
                            #
                            # self._guidance_fn 不为 None：
                            #
                            # "classifier"
                            #
                            # self._guidance_fn 为 None：
                            #
                            # "uncond"
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )

            # 将采样结果恢复为四维轨迹形式，
            # 再执行逆归一化。
            #
            # x0.reshape(B, P, -1, 4)：
            #
            # [B, P, 1 + future_len, 4]
            #
            # self._state_normalizer.inverse(...)：
            # 将状态恢复到真实物理尺度。
            #
            # [:, :, 1:]：
            # 删除第 0 个当前状态，
            # 只保留未来预测轨迹。
            #
            # 最终形状：
            #
            # [B, P, future_len, 4]
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            # 返回推理阶段轨迹预测结果。
            return {
                    "prediction": x0
                }

        
# 定义路线编码器。
#
# RouteEncoder 用于处理规划路线上的车道数据。
#
# 与 Encoder 中的 LaneFusionEncoder 不同：
#
# 1. LaneFusionEncoder：
#    面向整个场景中的车道；
#
# 2. RouteEncoder：
#    只面向规划路线中的车道。
#
# RouteEncoder 会将整条路线压缩为一个隐藏向量，
# 并作为 DiT 的条件信息。
class RouteEncoder(nn.Module):
    # 初始化路线编码器。
    #
    # route_num：
    # 路线包含的车道段数量；
    #
    # lane_len：
    # 每个车道段包含的离散点数量；
    #
    # drop_path_rate：
    # MixerBlock 和 MLP 中使用的 Dropout 比例；
    #
    # hidden_dim：
    # 输出隐藏维度；
    #
    # tokens_mlp_dim：
    # token 预投影后的维度；
    #
    # channels_mlp_dim：
    # 中间通道维度。
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        # 调用父类初始化方法。
        super().__init__()

        # 保存中间通道维度。
        self._channel = channels_mlp_dim

        # 创建几何特征通道投影 MLP。
        #
        # 每个路线点只使用前 4 个特征：
        #
        # [x, y, x'-x, y'-y]
        #
        # 输出维度为 channels_mlp_dim。
        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建路线 token 预投影 MLP。
        #
        # 路线中总离散点数量为：
        #
        # route_num * lane_len
        #
        # 经过投影后，
        # token 数量变为 tokens_mlp_dim。
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        # 创建一个 MixerBlock。
        #
        # 该模块会融合：
        #
        # 1. 路线离散点 token；
        # 2. 隐藏通道。
        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        # 创建最终投影前的 LayerNorm。
        self.norm = nn.LayerNorm(channels_mlp_dim)

        # 创建最终投影 MLP。
        #
        # 输出维度为 hidden_dim，
        # 以便与时间步嵌入直接相加。
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    # 定义 RouteEncoder 的前向传播。
    #
    # 输入 x 的形状：
    #
    # [B, P, V, D]
    #
    # B：
    # batch size；
    #
    # P：
    # 路线中的车道段数量；
    #
    # V：
    # 每条车道段中的离散点数量；
    #
    # D：
    # 每个离散点的特征维度。
    def forward(self, x):
        '''
        x: B, P, V, D
        '''

        # 原始注释：
        # 只使用 x、y 和从 x 指向 x' 的方向向量。
        #
        # 不使用：
        # 1. 车道边界；
        # 2. 限速；
        # 3. 信号灯。
        # only x and x->x' vector, no boundary, no speed limit, no traffic light

        # 只保留前 4 个特征：
        #
        # [x, y, x'-x, y'-y]
        x = x[..., :4]

        # 读取输入形状。
        B, P, V, _ = x.shape

        # 判断每一个路线离散点是否为空。
        #
        # 如果前 4 个特征全部为 0，
        # 则认为该离散点无效。
        #
        # mask_v 的形状：
        #
        # [B, P, V]
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0

        # 判断每一个路线车道段是否为空。
        #
        # 如果一个车道段没有任何有效离散点，
        # 则 mask_p 对应位置为 True。
        #
        # mask_p 的形状：
        #
        # [B, P]
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        # 判断每一个 batch 中的整条路线是否为空。
        #
        # 如果一个 batch 中没有任何有效车道段，
        # 则 mask_b 对应位置为 True。
        #
        # mask_b 的形状：
        #
        # [B]
        mask_b = torch.sum(~mask_p, dim=-1) == 0

        # 合并车道段维度和离散点维度。
        #
        # 输入形状：
        #
        # [B, P, V, 4]
        #
        # 输出形状：
        #
        # [B, P * V, 4]
        #
        # 这样，一整条路线会被视为一个长 token 序列。
        x = x.view(B, P * V, -1)

        # 将 batch 有效性掩码取反。
        #
        # valid_indices 的形状：
        #
        # [B]
        #
        # True 表示该 batch 中存在至少一个有效路线点。
        valid_indices = ~mask_b.view(-1) 

        # 只保留存在有效路线的 batch。
        x = x[valid_indices] 

        # 将每个路线离散点的 4 维几何特征
        # 映射为 channels_mlp_dim 维特征。
        #
        # 输入形状：
        #
        # [B_valid, P * V, 4]
        #
        # 输出形状：
        #
        # [B_valid, P * V, channels_mlp_dim]
        x = self.channel_pre_project(x)

        # 交换 token 维度和通道维度。
        #
        # 输出形状：
        #
        # [B_valid, channels_mlp_dim, P * V]
        x = x.permute(0, 2, 1)

        # 沿路线点 token 维度执行预投影。
        #
        # 输出形状：
        #
        # [B_valid, channels_mlp_dim, tokens_mlp_dim]
        x = self.token_pre_project(x)

        # 将 token 维度和通道维度交换回 MixerBlock 所需顺序。
        #
        # 输出形状：
        #
        # [B_valid, tokens_mlp_dim, channels_mlp_dim]
        x = x.permute(0, 2, 1)

        # 使用 MixerBlock 融合路线信息。
        x = self.Mixer(x)

        # 沿 token 维度计算均值。
        #
        # 输入形状：
        #
        # [B_valid, tokens_mlp_dim, channels_mlp_dim]
        #
        # 输出形状：
        #
        # [B_valid, channels_mlp_dim]
        #
        # 每个 batch 的整条路线被压缩成一个向量。
        x = torch.mean(x, dim=1)

        # 对路线编码执行：
        #
        # 1. LayerNorm；
        # 2. MLP 投影。
        #
        # 输出形状：
        #
        # [B_valid, hidden_dim]
        x = self.emb_project(self.norm(x))

        # 创建保存所有 batch 路线编码的全零张量。
        #
        # 形状：
        #
        # [B, hidden_dim]
        #
        # 没有有效路线的 batch 会保持为全零向量。
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)

        # 将有效路线编码写回对应 batch。
        x_result[valid_indices] = x  # Fill in valid parts
        
        # 返回路线编码。
        #
        # 输出形状：
        #
        # [B, hidden_dim]
        return x_result.view(B, -1)


# 定义 Diffusion Transformer。
#
# DiT 会接收：
#
# 1. 当前带噪轨迹 x；
# 2. 扩散时间 t；
# 3. 场景上下文 cross_c；
# 4. 路线车道 route_lanes；
# 5. 邻居车辆有效性掩码。
#
# 然后执行：
#
# 带噪轨迹投影
# → 目标类型嵌入
# → 路线编码
# → 时间步嵌入
# → 多层 DiTBlock
# → FinalLayer
# → 根据模型类型返回 score 或 x_start
class DiT(nn.Module):
    # 初始化 DiT。
    #
    # sde：
    # 随机微分方程对象；
    #
    # route_encoder：
    # 路线编码器；
    #
    # depth：
    # DiTBlock 数量；
    #
    # output_dim：
    # 每个目标轨迹压平后的维度；
    #
    # hidden_dim：
    # 隐藏特征维度；
    #
    # heads：
    # 多头注意力头数；
    #
    # dropout：
    # 注意力中的 Dropout 比例；
    #
    # mlp_ratio：
    # DiTBlock 内部 MLP 隐藏层倍率；
    #
    # model_type：
    # 输出类型。
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        # 调用父类初始化方法。
        super().__init__()
        
        # 检查模型输出类型是否合法。
        #
        # 当前仅支持：
        #
        # 1. "score"；
        # 2. "x_start"。
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"

        # 保存模型输出类型。
        self._model_type = model_type

        # 保存路线编码器。
        self.route_encoder = route_encoder

        # 创建目标类型嵌入。
        #
        # nn.Embedding(2, hidden_dim) 表示保存两个可学习向量：
        #
        # 1. 索引 0：
        #    自车 embedding；
        #
        # 2. 索引 1：
        #    邻居车辆 embedding。
        self.agent_embedding = nn.Embedding(2, hidden_dim)

        # 创建输入轨迹预投影 MLP。
        #
        # 输入维度：
        #
        # output_dim
        #
        # 即：
        #
        # (future_len + 1) * 4
        #
        # 输出维度：
        #
        # hidden_dim
        #
        # 中间隐藏层维度固定为 512。
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)

        # 创建时间步嵌入器。
        #
        # 该模块会将标量扩散时间 t
        # 转换为 hidden_dim 维条件向量。
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # 创建多个 DiTBlock。
        #
        # 每个 DiTBlock 会执行：
        #
        # 1. 条件调制；
        # 2. 自注意力；
        # 3. MLP；
        # 4. 交叉注意力；
        # 5. 第二个 MLP。
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])

        # 创建最终输出层。
        #
        # 该层会将 hidden_dim 维隐藏特征
        # 映射回 output_dim 维轨迹状态。
        self.final_layer = FinalLayer(hidden_dim, output_dim)

        # 保存 SDE。
        self._sde = sde

        # 保存边缘分布标准差计算函数。
        #
        # 后续当模型类型为 "score" 时，
        # 会使用该函数对模型输出执行尺度变换。
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    # 暴露模型输出类型。
    #
    # dpm_sampler 中的 model_wrapper(...)
    # 会读取 model.model_type。
    @property
    def model_type(self):
        # 返回模型输出类型。
        return self._model_type

    # 定义 DiT 前向传播。
    #
    # 输入：
    #
    # x：
    # 带噪轨迹；
    #
    # t：
    # 扩散时间；
    #
    # cross_c：
    # 场景上下文；
    #
    # route_lanes：
    # 路线车道；
    #
    # neighbor_current_mask：
    # 邻居车辆无效掩码。
    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """

        # 读取 batch size 和联合预测目标数量。
        #
        # x 的形状：
        #
        # [B, P, output_dim]
        B, P, _ = x.shape
        
        # 将压平后的轨迹状态投影到 hidden_dim 维隐藏空间。
        #
        # 输入形状：
        #
        # [B, P, output_dim]
        #
        # 输出形状：
        #
        # [B, P, hidden_dim]
        x = self.preproj(x)

        # 构造目标类型嵌入。
        #
        # self.agent_embedding.weight[0][None, :]：
        # 读取自车 embedding，
        # 并增加 token 维度。
        #
        # 形状：
        #
        # [1, hidden_dim]
        #
        # self.agent_embedding.weight[1][None, :].expand(P - 1, -1)：
        # 读取邻居车辆 embedding，
        # 并复制 P - 1 次。
        #
        # 形状：
        #
        # [P - 1, hidden_dim]
        #
        # 拼接后：
        #
        # 第 0 个 token 使用自车 embedding；
        # 后续 token 使用邻居车辆 embedding。
        #
        # 输出形状：
        #
        # [P, hidden_dim]
        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (P, D)

        # 将目标类型嵌入复制到每个 batch。
        #
        # 输出形状：
        #
        # [B, P, hidden_dim]
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)

        # 将目标类型嵌入加到轨迹隐藏特征上。
        x = x + x_embedding     

        # 使用 RouteEncoder 编码路线。
        #
        # route_encoding 的形状：
        #
        # [B, hidden_dim]
        route_encoding = self.route_encoder(route_lanes)

        # 将路线编码作为条件向量初始值。
        y = route_encoding

        # 将时间步嵌入加到路线编码上。
        #
        # self.t_embedder(t) 的形状：
        #
        # [B, hidden_dim]
        #
        # 最终 y 同时包含：
        #
        # 1. 路线条件；
        # 2. 当前扩散时间条件。
        y = y + self.t_embedder(t)      

        # 创建自注意力掩码。
        #
        # 初始值全部为 False，
        # 表示所有目标 token 默认有效。
        #
        # attn_mask 的形状：
        #
        # [B, P]
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)

        # 将邻居车辆掩码写入索引 1 之后的位置。
        #
        # 第 0 个 token 为自车，
        # 始终保持有效。
        #
        # 后续 token 对应邻居车辆。
        #
        # neighbor_current_mask 中：
        #
        # True 表示邻居车辆无效；
        # False 表示邻居车辆有效。
        attn_mask[:, 1:] = neighbor_current_mask
        
        # 依次执行多个 DiTBlock。
        #
        # 每个 block 接收：
        #
        # x：
        # 当前轨迹 token；
        #
        # cross_c：
        # 场景上下文；
        #
        # y：
        # 路线和时间步条件；
        #
        # attn_mask：
        # 无效邻居车辆掩码。
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)  
            
        # 使用 FinalLayer 将隐藏特征映射回轨迹状态空间。
        #
        # 输出形状：
        #
        # [B, P, output_dim]
        x = self.final_layer(x, y)
        
        # 如果模型类型为 "score"：
        #
        # 将模型输出除以当前扩散时间对应的边缘分布标准差。
        #
        # self.marginal_prob_std(t) 的形状：
        #
        # [B]
        #
        # [:, None, None]：
        # 调整为：
        #
        # [B, 1, 1]
        #
        # 这样可以沿目标维度和状态维度广播。
        #
        # + 1e-6：
        # 避免标准差过小时发生除零问题。
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)

        # 如果模型类型为 "x_start"：
        #
        # 直接返回模型预测的原始无噪轨迹。
        elif self._model_type == "x_start":
            return x

        # 如果模型类型既不是 "score" 也不是 "x_start"，
        # 则抛出异常。
        #
        # 正常情况下，该分支不会触发，
        # 因为 __init__ 中已经执行过合法性检查。
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
