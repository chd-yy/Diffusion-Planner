
# 导入 PyTorch。
#
# PyTorch 是该项目使用的深度学习框架。
# 它负责：
# 1. 张量运算；
# 2. 神经网络参数管理；
# 3. 权重初始化；
# 4. 前向传播；
# 5. 反向传播；
# 6. GPU 加速。
import torch

# 导入 PyTorch 的神经网络模块，并将其命名为 nn。
#
# nn 中包含大量构建神经网络时常用的组件，例如：
# 1. nn.Module：所有神经网络模块的基类；
# 2. nn.Linear：全连接层；
# 3. nn.LayerNorm：层归一化；
# 4. nn.Embedding：嵌入层。
import torch.nn as nn


# 导入 Encoder。
#
# Encoder 是 Diffusion Planner 中真正负责场景编码的模块。
#
# 它通常会处理：
# 1. 自车历史状态；
# 2. 周围交通参与者的历史状态；
# 3. 车道线等地图元素；
# 4. 限速信息；
# 5. 红绿灯状态；
# 6. 导航路线信息。
#
# Encoder 会将原始场景信息转换为高维特征，
# 供后续 Decoder 生成未来轨迹。
from diffusion_planner.model.module.encoder import Encoder

# 导入 Decoder。
#
# Decoder 是 Diffusion Planner 中真正执行轨迹生成的模块。
#
# 它通常会接收：
# 1. Encoder 提取出的场景特征；
# 2. 原始输入中的部分辅助信息；
#
# 然后通过扩散模型的采样过程生成未来轨迹。
from diffusion_planner.model.module.decoder import Decoder


# 定义完整的 Diffusion Planner 模型。
#
# 该类继承 nn.Module，因此可以：
# 1. 注册神经网络参数；
# 2. 调用 model.parameters() 获取可训练参数；
# 3. 使用 model.to(device) 将模型移动到 CPU 或 GPU；
# 4. 使用 model.eval() 切换到推理模式；
# 5. 使用 model(inputs) 自动调用 forward()；
# 6. 保存和加载 state_dict。
#
# 从整体结构上看，该模型由两部分组成：
#
# inputs
#   ↓
# Diffusion_Planner_Encoder
#   ↓
# encoder_outputs
#   ↓
# Diffusion_Planner_Decoder
#   ↓
# decoder_outputs
#
# Encoder 负责理解场景。
# Decoder 负责根据场景信息生成规划轨迹。
class Diffusion_Planner(nn.Module):

    # 初始化完整的 Diffusion Planner。
    #
    # config：
    #   项目的配置对象。
    #
    # config 中通常保存：
    # 1. 模型隐藏层维度；
    # 2. Transformer 层数；
    # 3. 注意力头数量；
    # 4. 轨迹预测长度；
    # 5. 扩散模型参数；
    # 6. SDE 配置；
    # 7. 数据特征维度。
    def __init__(self, config):

        # 调用父类 nn.Module 的初始化函数。
        #
        # 这一步非常重要。
        # 只有调用 super().__init__() 后，
        # PyTorch 才能正确注册当前模块内部的子模块和参数。
        super().__init__()

        # 创建 Encoder 包装模块。
        #
        # Diffusion_Planner_Encoder 内部还会进一步创建真正的 Encoder，
        # 并负责初始化 Encoder 中的权重。
        self.encoder = Diffusion_Planner_Encoder(config)

        # 创建 Decoder 包装模块。
        #
        # Diffusion_Planner_Decoder 内部还会进一步创建真正的 Decoder，
        # 并负责初始化 Decoder 中的权重。
        self.decoder = Diffusion_Planner_Decoder(config)


    # 将 sde 暴露为 Diffusion_Planner 的只读属性。
    #
    # @property 的作用是：
    # 允许外部像访问普通成员变量一样访问该函数的返回值。
    #
    # 外部可以写：
    #
    # model.sde
    #
    # 而不需要写：
    #
    # model.sde()
    #
    # sde 通常表示 Stochastic Differential Equation，
    # 即随机微分方程。
    #
    # 在扩散模型中，SDE 会定义：
    # 1. 前向加噪过程；
    # 2. 反向去噪过程；
    # 3. 噪声随时间的变化方式；
    # 4. 采样过程中需要使用的扩散系数。
    @property
    def sde(self):

        # 当前对象的层级关系为：
        #
        # self
        #   ↓
        # self.decoder
        #   ↓
        # Diffusion_Planner_Decoder
        #   ↓
        # self.decoder.decoder
        #   ↓
        # Decoder
        #   ↓
        # self.decoder.decoder.sde
        #
        # 因此，这一行将内部 Decoder 保存的 sde 对象返回给外部。
        return self.decoder.decoder.sde
    

    # 定义完整模型的前向传播过程。
    #
    # 当外部执行：
    #
    # model(inputs)
    #
    # PyTorch 会自动调用：
    #
    # model.forward(inputs)
    #
    # inputs：
    #   经过数据处理和归一化后的模型输入。
    #
    # inputs 通常是一个字典，
    # 其中可能包含自车、周围交通参与者、车道线、
    # 导航路线和交通灯等特征张量。
    def forward(self, inputs):

        # 将输入传递给 Encoder。
        #
        # Encoder 会提取场景中的高维语义特征。
        #
        # encoder_outputs 通常会包含：
        # 1. 自车特征；
        # 2. 周围交通参与者特征；
        # 3. 地图特征；
        # 4. 场景上下文特征；
        # 5. 有效性掩码；
        # 6. Decoder 后续需要使用的条件信息。
        #
        # 具体包含哪些键，需要结合 Encoder 的实现进一步确认。
        encoder_outputs = self.encoder(inputs)

        # 将 Encoder 输出以及原始输入传递给 Decoder。
        #
        # Decoder 不仅接收 encoder_outputs，
        # 还接收原始 inputs。
        #
        # 这说明 Decoder 的轨迹生成过程除了使用编码后的场景特征，
        # 还可能直接使用原始输入中的某些数据，
        # 例如掩码、当前状态或其他辅助变量。
        #
        # decoder_outputs 通常是一个字典，
        # 其中可能包含预测轨迹、训练损失所需的中间结果或采样结果。
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        # 同时返回 Encoder 输出和 Decoder 输出。
        #
        # 返回 encoder_outputs 的好处是：
        # 外部不仅可以获得最终轨迹，
        # 还可以访问 Encoder 产生的中间特征。
        #
        # 在训练、调试或可视化时，
        # 这些中间特征可能很有用。
        return encoder_outputs, decoder_outputs


# 定义 Encoder 的包装模块。
#
# 这一层并不是具体的 Encoder 网络本身。
# 它主要负责：
# 1. 创建 Encoder；
# 2. 对 Encoder 中的参数执行统一初始化；
# 3. 在 forward() 中调用真正的 Encoder。
#
# 这种写法可以将“模型结构定义”和“初始化策略”分离开。
class Diffusion_Planner_Encoder(nn.Module):

    # 初始化 Encoder 包装模块。
    #
    # config：
    #   用于创建 Encoder 的配置对象。
    def __init__(self, config):

        # 调用父类 nn.Module 的初始化函数。
        super().__init__()

        # 创建真正执行场景特征提取的 Encoder。
        self.encoder = Encoder(config)

        # 初始化 Encoder 中的参数。
        #
        # 这一行会调用当前类中定义的 initialize_weights()。
        self.initialize_weights()


    # 定义 Encoder 的权重初始化策略。
    #
    # 神经网络刚刚创建时，其参数必须设置为合理的初始值。
    #
    # 初始化策略会影响：
    # 1. 训练是否稳定；
    # 2. 梯度是否容易消失；
    # 3. 梯度是否容易爆炸；
    # 4. 模型收敛速度；
    # 5. 最终训练效果。
    def initialize_weights(self):

        # Initialize transformer layers:

        # 定义一个局部函数 _basic_init()。
        #
        # 参数 m 表示当前正在处理的神经网络子模块。
        #
        # 后续 self.apply(_basic_init) 会递归访问 Encoder 中的所有子模块，
        # 并对每一个子模块调用该函数。
        #
        # 因此，该函数相当于统一的基础初始化规则。
        def _basic_init(m):

            # 如果当前子模块是全连接层 nn.Linear，
            # 则对其权重和偏置进行初始化。
            if isinstance(m, nn.Linear):

                # 使用 Xavier 均匀分布初始化全连接层的权重。
                #
                # Xavier 初始化也称为 Glorot 初始化。
                #
                # 它会根据输入维度和输出维度控制权重的初始范围，
                # 使神经网络中信号的方差尽量保持稳定。
                #
                # 这样可以降低梯度消失或梯度爆炸的风险。
                torch.nn.init.xavier_uniform_(m.weight)

                # 检查当前模块是否为 nn.Linear，
                # 并且是否具有偏置项。
                #
                # 外层已经判断过 isinstance(m, nn.Linear)，
                # 因此内层再次判断 nn.Linear 在逻辑上是重复的。
                #
                # 这里严格保留原始代码，没有进行修改。
                if isinstance(m, nn.Linear) and m.bias is not None:

                    # 将全连接层偏置初始化为 0。
                    nn.init.constant_(m.bias, 0)

            # 如果当前子模块是 LayerNorm，
            # 则初始化其缩放参数和偏移参数。
            elif isinstance(m, nn.LayerNorm):

                # 将 LayerNorm 的偏置初始化为 0。
                #
                # LayerNorm 的输出通常可以理解为：
                #
                # output = normalized_input * weight + bias
                #
                # 当 bias = 0 时，不会额外引入平移。
                nn.init.constant_(m.bias, 0)

                # 将 LayerNorm 的缩放参数初始化为 1。
                #
                # 当 weight = 1 时，
                # LayerNorm 初始化阶段不会额外放大或缩小归一化结果。
                nn.init.constant_(m.weight, 1.0)

            # 如果当前子模块是嵌入层 nn.Embedding，
            # 则使用正态分布初始化嵌入向量。
            elif isinstance(m, nn.Embedding):

                # 从均值为 0、标准差为 0.02 的正态分布中
                # 随机采样嵌入层权重。
                #
                # 这种初始化方式在 Transformer 模型中较为常见。
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 递归遍历当前包装模块中的所有子模块，
        # 并对每个子模块应用 _basic_init()。
        #
        # 因为 self.encoder 是当前模块的子模块，
        # 所以 Encoder 内部的 Linear、LayerNorm 和 Embedding
        # 都会被统一初始化。
        self.apply(_basic_init)

        # Initialize embedding MLP:

        # 对 Encoder 的位置嵌入权重再次执行正态分布初始化。
        #
        # pos_emb 通常用于表示不同 token、轨迹点或地图元素的位置。
        #
        # 即使 self.apply(_basic_init) 已经处理过部分嵌入层，
        # 这里仍然针对特定嵌入权重显式初始化，
        # 以确保它们采用预期的初始化方式。
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)

        # 初始化周围交通参与者类型嵌入。
        #
        # neighbor_encoder.type_emb 可能用于区分不同类型的动态目标，
        # 例如：
        # 1. 车辆；
        # 2. 行人；
        # 3. 自行车；
        # 4. 其他交通参与者。
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)

        # 初始化车道限速嵌入。
        #
        # lane_encoder.speed_limit_emb 可能用于将离散化后的限速信息
        # 转换为可学习的高维向量。
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)

        # 初始化车道交通灯状态嵌入。
        #
        # lane_encoder.traffic_emb 可能用于区分：
        # 1. 红灯；
        # 2. 黄灯；
        # 3. 绿灯；
        # 4. 未知状态；
        # 5. 无交通灯。
        #
        # 具体类别数量需要结合 Lane Encoder 的实现确认。
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)


    # 定义 Encoder 包装模块的前向传播。
    #
    # inputs：
    #   模型输入字典。
    def forward(self, inputs):

        # 调用真正的 Encoder。
        #
        # Encoder 会将原始场景数据编码为高维特征。
        encoder_outputs = self.encoder(inputs)

        # 将 Encoder 产生的结果原样返回。
        return encoder_outputs
    

# 定义 Decoder 的包装模块。
#
# 与 Diffusion_Planner_Encoder 类似，
# 这一层主要负责：
# 1. 创建真正的 Decoder；
# 2. 执行 Decoder 的权重初始化；
# 3. 在 forward() 中调用真正的 Decoder。
#
# Decoder 通常承担扩散模型中最核心的轨迹生成任务。
class Diffusion_Planner_Decoder(nn.Module):

    # 初始化 Decoder 包装模块。
    #
    # config：
    #   用于创建 Decoder 的配置对象。
    def __init__(self, config):

        # 调用父类 nn.Module 的初始化函数。
        super().__init__()

        # 创建真正的 Decoder。
        #
        # Decoder 内部可能包含：
        # 1. SDE；
        # 2. 扩散采样器；
        # 3. DiT 网络；
        # 4. 时间步嵌入；
        # 5. 条件特征融合模块；
        # 6. 轨迹输出层。
        #
        # 具体结构需要结合 Decoder 的实现进一步确认。
        self.decoder = Decoder(config)

        # 初始化 Decoder 中的参数。
        self.initialize_weights()


    # 定义 Decoder 的权重初始化策略。
    #
    # 该方法包含两类初始化：
    #
    # 第一类：
    #   对 Linear、LayerNorm 和 Embedding 执行基础初始化。
    #
    # 第二类：
    #   针对 DiT 中的时间嵌入、adaLN 调制层和输出层
    #   执行更加特殊的初始化。
    def initialize_weights(self):

        # Initialize transformer layers:

        # 定义统一的基础初始化函数。
        #
        # self.apply(_basic_init) 会递归访问 Decoder 中的所有子模块，
        # 并对每一个子模块调用该函数。
        def _basic_init(m):

            # 如果当前模块是全连接层，则初始化其权重和偏置。
            if isinstance(m, nn.Linear):

                # 使用 Xavier 均匀分布初始化权重。
                torch.nn.init.xavier_uniform_(m.weight)

                # 如果该线性层具有偏置项，
                # 则将偏置初始化为 0。
                #
                # 外层已经确认 m 是 nn.Linear，
                # 内层再次执行类型判断在逻辑上是重复的。
                #
                # 这里严格保留原始代码。
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 如果当前模块是 LayerNorm，
            # 则将偏置初始化为 0，缩放参数初始化为 1。
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            # 如果当前模块是 Embedding，
            # 则使用均值为 0、标准差为 0.02 的正态分布初始化。
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 对当前 Decoder 包装模块内部的所有子模块
        # 递归执行基础初始化。
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:

        # 初始化 DiT 中时间步嵌入器的第一层全连接权重。
        #
        # 扩散模型在不同时间步的噪声强度不同。
        # 因此，网络需要知道当前正在处理哪个扩散时间步。
        #
        # t_embedder 会将时间步 t 转换为高维向量，
        # 使神经网络可以根据不同噪声等级采取不同的去噪策略。
        #
        # mlp[0] 通常表示时间嵌入 MLP 中的第一层线性层。
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)

        # 初始化时间步嵌入器中的另一层全连接权重。
        #
        # mlp[2] 通常表示时间嵌入 MLP 中后续的线性层。
        #
        # 中间的 mlp[1] 可能是激活函数，
        # 具体结构需要结合 t_embedder 的定义确认。
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:

        # 遍历 DiT 中的每一个 Transformer block。
        #
        # DiT 是 Diffusion Transformer 的缩写。
        #
        # 与传统扩散模型常用的 U-Net 不同，
        # DiT 使用 Transformer 结构完成去噪预测。
        #
        # 每个 block 通常包含：
        # 1. 自注意力；
        # 2. 前馈网络；
        # 3. LayerNorm；
        # 4. 条件调制模块；
        # 5. 残差连接。
        for block in self.decoder.dit.blocks:

            # 将当前 DiT block 中 adaLN 调制模块最后一层的权重初始化为 0。
            #
            # adaLN 通常表示 Adaptive Layer Normalization，
            # 即自适应层归一化。
            #
            # 它可以根据条件信息调整 LayerNorm 后的特征，
            # 例如根据扩散时间步、场景编码或其他条件信息，
            # 动态产生缩放和平移参数。
            #
            # 将最后一层权重初始化为 0，
            # 可以让调制模块在训练初期从接近“无调制”的状态开始。
            #
            # 这样通常有利于稳定深层网络的训练。
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

            # 将当前 DiT block 中 adaLN 调制模块最后一层的偏置初始化为 0。
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:

        # 将 DiT 最终输出层中的 adaLN 调制模块
        # 最后一层权重初始化为 0。
        #
        # 这使最终输出层中的条件调制在训练初期保持较弱影响。
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)

        # 将 DiT 最终输出层中的 adaLN 调制模块
        # 最后一层偏置初始化为 0。
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)

        # 将 DiT 最终投影模块的最后一层权重初始化为 0。
        #
        # proj 通常用于将 DiT 隐藏特征映射为最终预测结果，
        # 例如轨迹噪声、score 或其他参数化形式。
        #
        # 将最后一层初始化为 0 后，
        # 网络刚开始训练时的输出会接近 0。
        #
        # 这种初始化方式在扩散 Transformer 中较为常见，
        # 有助于让训练过程更加稳定。
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)

        # 将 DiT 最终投影模块的最后一层偏置初始化为 0。
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)


    # 定义 Decoder 包装模块的前向传播过程。
    #
    # encoder_outputs：
    #   Encoder 输出的场景特征。
    #
    # inputs：
    #   原始模型输入。
    #
    # Decoder 同时接收两者，
    # 说明轨迹生成过程不仅依赖编码后的场景语义，
    # 还可能直接使用输入中的部分辅助信息。
    def forward(self, encoder_outputs, inputs):

        # 调用真正的 Decoder。
        #
        # Decoder 会根据场景特征执行轨迹生成。
        #
        # 在扩散模型中，这通常涉及：
        # 1. 构造初始噪声轨迹；
        # 2. 在多个扩散时间步中执行迭代去噪；
        # 3. 使用 DiT 预测噪声、score 或其他等价参数；
        # 4. 逐步恢复合理的未来轨迹；
        # 5. 返回最终规划结果。
        #
        # 具体采样步骤需要结合 Decoder 的内部实现进一步分析。
        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        # 将 Decoder 输出原样返回。
        return decoder_outputs

