# 导入 PyTorch，用于张量计算、权重初始化，以及 no_grad 推理上下文。
import torch
# nn.Module 是所有模型包装类的基类；nn.Linear、LayerNorm、Embedding 用于识别并初始化子模块。
import torch.nn as nn

# 【HDP 与原 Diffusion-Planner 的区别：包命名空间】Encoder/Decoder 的导入路径由
# diffusion_planner.model.* 切换为 hdp_nuplan.model.*。
# 【实现核对：底层 Encoder 一致】将包路径统一后，两边 encoder.py 的可执行 AST 完全一致；
# 即场景 token、融合网络、张量形状和 forward 计算相同，差别仅是命名空间和注释。
# 【HDP 与原 Diffusion-Planner 的区别：底层 Decoder 架构】两边导入的 Decoder 同名但实现不同：
# 1. 原版联合建模 1 个自车和 predicted_neighbor_num 个邻车；HDP 只生成自车未来；
# 2. 原版扩散状态是绝对位置且包含当前时刻；HDP 训练目标是未来 T 帧运动增量，不拼当前时刻；
# 3. 原版 DiT 每个目标一次输出 (future_len+1)*4；HDP 按时间序列输出每帧 4 维；
# 4. 原版使用邻车有效 mask 作为条件；HDP 改用当前自车速度 (vx, vy)；
# 5. 原版 forward 按 self.training 分训练/推理；HDP 按输入中是否有扩散状态字段分支；
# 6. HDP 新增多候选 sample()，把预测 x/y 增量累加成位置；原版没有该显式接口；
# 7. 原版 DiT 的 token 轴是“自车/邻车”，使用 2 类 agent embedding；HDP 的 token 轴是
#    “未来时间位置”，使用 future_len 个位置 embedding，并新增自车 vx/vy 条件投影；
# 8. 原版模型输出参数化只接受 score/x_start；HDP 还接受 noise/v；
# 9. 原版保存并使用 guidance_fn；HDP Decoder 不读取该配置，当前采样不走该 guidance 路径；
# 10. 两边虽然都使用 VP-SDE 和 DPM 类采样器，但采样张量形状、条件参数和返回契约不同。
from hdp_nuplan.model.module.encoder import Encoder
from hdp_nuplan.model.module.decoder import Decoder


# 【HDP 与原 Diffusion-Planner 的区别：顶层模型名称】原版顶层类为 Diffusion_Planner；
# HDP 使用独立的 Hyper_Diffusion_Planner，避免训练入口和 checkpoint 架构相互混淆。
# 该类继承 nn.Module，因此支持参数注册、to(device)、train/eval、state_dict 以及 model(inputs)。
# 整体数据流仍然是：inputs -> Encoder -> encoder_outputs -> Decoder -> decoder_outputs。
class Hyper_Diffusion_Planner(nn.Module):
    # config 集中保存输入形状、隐藏维度、Encoder/Decoder 深度、注意力头数和扩散参数化等配置。
    def __init__(self, config):
        # 初始化 nn.Module，确保下面创建的 Encoder 和 Decoder 被正确注册为子模块。
        super().__init__()

        # 【HDP 与原 Diffusion-Planner 的区别：包装类名称】原版使用
        # Diffusion_Planner_Encoder/Decoder；HDP 改为 Hyper_Diffusion_Planner_Encoder/Decoder。
        # Encoder 的场景编码职责不变，Decoder 使用 HDP 独立实现。
        self.encoder = Hyper_Diffusion_Planner_Encoder(config)
        self.decoder = Hyper_Diffusion_Planner_Decoder(config)

    # 【实现核对：与原版一致】@property、包装层级和返回语句完全相同；外部通过
    # model.sde 访问底层 Decoder 的 SDE，而不需要写成 model.sde()。
    @property
    def sde(self):
        # 包装层级为 self.decoder -> wrapper.decoder -> HDP Decoder -> sde。
        # 这一只读出口与原版一致，供 loss、采样器和扩散参数化转换复用同一个 SDE。
        return self.decoder.decoder.sde
    
    # 【实现核对：与原版一致】除包装类名称外，标准 forward 的三条可执行语句完全相同：
    # 先调用 Encoder，再调用 Decoder，最后同时返回两部分输出。
    def forward(self, inputs):

        # Encoder 将邻车历史、静态目标和矢量地图等观测编码成场景上下文 token。
        encoder_outputs = self.encoder(inputs)
        # Decoder 同时接收编码结果和原始 inputs，以读取 route、扩散时间和带噪轨迹等条件。
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        # 与原版接口一致，同时返回中间场景编码和 Decoder 输出，便于训练、调试和推理。
        return encoder_outputs, decoder_outputs

    # 【HDP 与原 Diffusion-Planner 的区别：显式采样入口】原版顶层模型没有 sample()；
    # HDP 新增不依赖 forward 的多候选采样接口，供普通推理和离线 RL rollout 直接调用。
    # @torch.no_grad() 禁用梯度记录，减少采样阶段显存和计算开销。
    @torch.no_grad()
    def sample(self, inputs, num_samples=1, diffusion_steps=10, noise_scale=0.1):
        """不依赖 train/eval 分支的显式采样接口，主要供 RL rollout 使用。"""
        # 返回形状为 [B, num_samples, future_len, 4]。
        # 记录调用前模式，避免采样永久改变外部模型的 train/eval 状态。
        was_training = self.training
        # 临时切换到 eval，关闭 Dropout/DropPath 等训练随机性。
        self.eval()
        # try/finally 保证采样成功或抛出异常时都能恢复原模式。
        try:
            # 每次采样先编码一次场景，再由底层 HDP Decoder 生成指定数量的候选轨迹。
            encoder_outputs = self.encoder(inputs)
            return self.decoder.decoder.sample(
                encoder_outputs,
                inputs,
                num_samples=num_samples,
                diffusion_steps=diffusion_steps,
                noise_scale=noise_scale,
            )
        finally:
            # train(bool) 会递归恢复整个模型；原来是 eval 时仍恢复为 eval。
            self.train(was_training)


# Encoder 包装层负责创建真正的场景 Encoder、统一初始化参数并转发 forward 调用。
# 【HDP 与原 Diffusion-Planner 的区别：Encoder 包装类名称】类名增加 Hyper 前缀；
# 其 __init__、initialize_weights() 和 forward() 可执行语句均与原版完全一致，
# 便于加载名称和 shape 匹配的预训练 Encoder。
class Hyper_Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        # 初始化父类后，子模块和参数才能被 PyTorch 正确注册。
        super().__init__()

        # 创建实际执行动态目标、静态目标和 lane 场景编码的 Encoder。
        self.encoder = Encoder(config)
        # 创建完成后立即应用与原版一致的显式初始化策略。
        self.initialize_weights()

    # 对 Encoder 内的通用层和特定 embedding 执行初始化。
    def initialize_weights(self):
        # Initialize transformer layers:
        # 局部函数会由 self.apply() 递归应用到当前包装层的所有子模块。
        def _basic_init(m):
            # Linear 权重使用 Xavier 均匀初始化，偏置初始化为 0。
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm 初始为单位缩放、零偏移。
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # Embedding 使用均值 0、标准差 0.02 的正态分布初始化。
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # 递归遍历 Encoder 中的 Linear、LayerNorm 和 Embedding。
        self.apply(_basic_init)

        # Initialize embedding MLP:
        # 对位置、交通参与者类型、lane 限速和交通灯 embedding 显式使用 0.02 标准差。
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    # 将输入交给实际 Encoder，并原样返回场景编码字典。
    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

# Decoder 包装层负责创建底层 HDP Decoder，并将场景编码和原始输入转发给它。
# 【HDP 与原 Diffusion-Planner 的区别：Decoder 包装类名称与实现】原版包装
# Diffusion Planner 的联合自车/邻车 Decoder；当前包装的是仅规划自车的 HDP Decoder。
class Hyper_Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        # 初始化父类，随后注册底层 Decoder。
        super().__init__()

        # 底层 Decoder 持有 VP-SDE、DiT、route encoder、训练前向和显式采样实现。
        self.decoder = Decoder(config)
        # 【HDP 与原 Diffusion-Planner 的区别：Decoder 初始化未自动调用】原版在创建
        # Decoder 后立即调用 self.initialize_weights()，执行 Xavier、adaLN 零初始化和
        # 输出层零初始化；HDP 当前注释掉该调用，因此使用各底层模块构造时的默认初始化。
        # initialize_weights() 方法仍保留，但除非外部显式调用，否则下面的规则不会执行。
        # self.initialize_weights()

    # 【实现核对：方法体与原版一致】Decoder 特殊初始化方法的每条可执行语句都相同；
    # 差别不在方法内部，而在 HDP 构造函数没有调用它。
    def initialize_weights(self):
        # Initialize transformer layers:
        # 递归初始化 Linear、LayerNorm 和 Embedding。
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                # Xavier 初始化根据输入/输出维度控制初始权重方差。
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        # 扩散时间 embedding MLP 的两个 Linear 权重使用标准差 0.02 的正态初始化。
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # 若该方法被调用，将每个 DiT block 的 adaLN 最后一层置零，使初始调制接近关闭。
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # 若该方法被调用，将最终 adaLN 和投影层置零，使 Decoder 初始输出接近 0。
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    # 【实现核对：包装 forward 与原版一致】两边都把 encoder_outputs 和 inputs 传给
    # 底层 Decoder，再原样返回结果；实际行为差异来自两边导入的 Decoder 实现不同。
    def forward(self, encoder_outputs, inputs):

        # HDP Decoder 根据输入中是否存在 sampled_trajectories/diffusion_time，返回训练用
        # "score" 或推理用 "prediction"；原版主要依据 module.training 选择分支。
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        # 将底层 Decoder 的输出字典原样返回给顶层模型。
        return decoder_outputs
