# 导入 Python 标准库中的 abc 模块。
# abc 是 Abstract Base Classes 的缩写，即“抽象基类”。
# 它用于定义统一的接口，并要求子类实现指定的方法。
import abc

# 导入 PyTorch。
# 本文件中的张量运算、指数运算、平方根运算以及形状变换均依赖 PyTorch。
import torch


# 设置标准差的最小值。
#
# 在扩散过程接近初始时刻 t=0 时，噪声标准差可能接近 0。
# 如果后续代码需要除以标准差，过小的标准差可能导致数值不稳定。
# 因此，可以通过该常量为标准差设置一个下限。
STD_MIN = 1e-6


# 定义随机微分方程 SDE 的抽象基类。
#
# SDE 是 Stochastic Differential Equation 的缩写，即“随机微分方程”。
#
# 扩散模型中的正向加噪过程通常可以写成：
#
# dx = f(x,t)dt + g(t)dW_t
#
# 其中：
# 1. x 表示当前时刻的样本；
# 2. t 表示扩散时间；
# 3. f(x,t) 表示漂移系数 drift；
# 4. g(t) 表示扩散系数 diffusion；
# 5. dW_t 表示标准布朗运动的增量。
#
# 该抽象类只规定接口，不直接实现某一种具体的 SDE。
# 后续可以通过继承该类，实现 VP-SDE、VE-SDE 或 sub-VP-SDE 等不同形式。
class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    # 构造 SDE 抽象基类。
    #
    # 当前基类没有额外的成员变量。
    # 调用 super().__init__() 可以确保父类的初始化逻辑正常执行。
    def __init__(self):
        """Construct an SDE.
        """
        super().__init__()

    # 将 T 声明为只读属性。
    #
    # @property 允许使用 obj.T 的形式访问终止时间，
    # 而不是调用 obj.T()。
    #
    # @abc.abstractmethod 表示该属性必须由子类实现。
    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""

        # 抽象属性本身不提供具体实现。
        pass

    # 定义 SDE 系数计算接口。
    #
    # 输入：
    # 1. x：当前时刻的样本张量；
    # 2. t：当前扩散时间。
    #
    # 输出：
    # 1. drift：漂移项 f(x,t)；
    # 2. diffusion：扩散项 g(t)。
    #
    # 子类必须根据自身对应的随机微分方程实现该方法。
    @abc.abstractmethod
    def sde(self, x, t):
        """
        sde: A function that returns the drift and diffusion coefficients of the SDE.

        return (drift $f(x,t)$, diffusion $g(t)$)
        """

        # 抽象方法本身不提供具体实现。
        pass

    # 定义边缘分布参数的计算接口。
    #
    # 对于给定的初始样本 x 和扩散时间 t，
    # 该方法用于计算条件边缘分布 p_t(x_t | x_0) 的参数。
    #
    # 对许多常见扩散过程，该条件分布可以写为高斯分布：
    #
    # x_t = mean + std * epsilon
    #
    # 其中 epsilon 通常服从标准高斯分布。
    #
    # 返回：
    # 1. mean：边缘分布的均值；
    # 2. std：边缘分布的标准差。
    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return mean, std
        """

        # 抽象方法本身不提供具体实现。
        pass

    # 定义扩散系数计算接口。
    #
    # 输入：
    # t：扩散时间。
    #
    # 返回：
    # g(t)：随机微分方程中的扩散系数。
    @abc.abstractmethod
    def diffusion_coeff(self, t):
        """
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.

        return $g(t)$
        """

        # 抽象方法本身不提供具体实现。
        pass

    # 定义边缘分布标准差的计算接口。
    #
    # 与 marginal_prob(...) 不同，
    # 该方法只返回边缘分布中的标准差 std，
    # 不计算均值。

    # marginal_prob_std(t) 返回的是前向扩散到时间 t 后，x_t 条件分布中的噪声标准差。
    # 它主要由时间 t、SDE 的噪声调度参数，比如 beta_min/beta_max，以及具体 SDE 类型决定。
    # 在 VP-SDE 中，它通常可以理解为 sigma(t)=sqrt(1-alpha(t)^2)，
    # 表示当前时间步注入噪声的强度。
    @abc.abstractmethod
    def marginal_prob_std(self, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return std
        """

        # 抽象方法本身不提供具体实现。
        pass


# 定义采用线性 beta 调度的 Variance Preserving SDE。
#
# VPSDE 是 Variance Preserving Stochastic Differential Equation 的缩写，
# 即“方差保持随机微分方程”。
#
# 该 SDE 的形式为：
#
# dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dW_t
#
# 其中 beta(t) 随时间线性变化：
#
# beta(t) = beta_min + (beta_max - beta_min) * t
#
# 在扩散过程中：
# 1. 原始数据 x_0 的信息逐渐衰减；
# 2. 高斯噪声逐渐增加；
# 3. 在终止时刻附近，样本逐渐接近高斯噪声。
class VPSDE_linear(SDE):

    # 初始化线性 VP-SDE。
    #
    # 参数：
    # 1. beta_max：终止时刻附近的最大噪声增长率，默认值为 20.0；
    # 2. beta_min：初始时刻附近的最小噪声增长率，默认值为 0.1。
    #
    # beta(t) 会在 beta_min 和 beta_max 之间线性变化。
    def __init__(self, beta_max=20.0, beta_min=0.1):
        """
        VP SDE

        SDE:
        $ \mathrm{d}x = -\frac{\beta(t)}{2} x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W_t $
        """

        # 调用父类 SDE 的初始化方法。
        super().__init__()

        # 保存 beta(t) 的最大值。
        self._beta_max = beta_max

        # 保存 beta(t) 的最小值。
        self._beta_min = beta_min

    # 定义该 SDE 的终止时间。
    #
    # 当前实现将连续时间范围设置为 [0, 1]。
    # t=0 对应原始数据附近；
    # t=1 对应噪声较强的终止状态。
    @property
    def T(self):
        return 1.0

    # 计算线性 VP-SDE 在给定状态 x 和时间 t 下的漂移项与扩散项。
    #
    # 对应公式：
    #
    # drift = -0.5 * beta(t) * x
    # diffusion = sqrt(beta(t))
    #
    # 输入 x 通常是一个批量张量，例如：
    #
    # [batch_size, channels, height, width]
    #
    # 或轨迹预测任务中的：
    #
    # [batch_size, horizon, feature_dim]
    def sde(self, x, t):
        """
        SDE of diffusion process

        drift = $-\frac{\beta(t)}{2} x$
        diffusion = $\sqrt{\beta(t)}$
        """

        # 获取输入张量 x 的形状。
        #
        # 例如：
        # x.shape = [batch_size, horizon, feature_dim]
        shape = x.shape

        # 构造用于广播的目标形状。
        #
        # 假设 x 的形状为：
        #
        # [batch_size, horizon, feature_dim]
        #
        # 那么 reshape 将变为：
        #
        # [-1, 1, 1]
        #
        # 这样，每个样本对应的时间 t 都可以沿着其余维度广播。
        #
        # -1 表示由 PyTorch 自动推断批量维度大小。
        reshape = [-1] + [1, ] * (len(shape) - 1)

        # 将时间张量 t 调整为可与 x 广播相乘的形状。
        #
        # 例如：
        #
        # 原始 t.shape = [batch_size]
        # 调整后 t.shape = [batch_size, 1, 1]
        t = t.reshape(reshape)

        # 根据线性调度公式计算当前时刻的 beta(t)。
        #
        # beta(t) = (beta_max - beta_min) * t + beta_min
        #
        # 当 t=0 时：
        #
        # beta(0) = beta_min
        #
        # 当 t=1 时：
        #
        # beta(1) = beta_max
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min

        # 计算确定性的漂移项。
        #
        # drift = -0.5 * beta(t) * x
        #
        # 负号表示样本幅值会随着扩散过程逐渐衰减。
        drift = - 0.5 * beta_t * x

        # 计算随机噪声项前面的扩散系数。
        #
        # diffusion = sqrt(beta(t))
        #
        # beta(t) 越大，每个单位时间内注入的随机噪声越强。
        diffusion = torch.sqrt(beta_t)

        # 返回：
        # 1. 漂移项 drift；
        # 2. 扩散项 diffusion。
        return drift, diffusion

    # 计算线性 VP-SDE 的条件边缘分布参数。
    #
    # 给定初始样本 x 和扩散时间 t，
    # 正向扩散后的样本通常可以表示为：
    #
    # x_t = alpha(t) * x_0 + sigma(t) * epsilon
    #
    # 其中：
    # 1. epsilon 是标准高斯噪声；
    # 2. alpha(t) 表示原始信号保留比例；
    # 3. sigma(t) 表示噪声标准差。
    #
    # 该方法返回：
    #
    # mean = alpha(t) * x
    # std = sigma(t)
    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
        """

        # 获取输入样本 x 的形状。
        shape = x.shape

        # 构造用于广播的目标形状。
        #
        # 其作用与 sde(...) 方法中的 reshape 相同：
        # 让每个批次样本对应的 t 能够广播到 x 的其他维度。
        reshape = [-1] + [1, ] * (len(shape) - 1)

        # 将时间张量 t 调整为可广播的形状。
        t = t.reshape(reshape)

        # 计算均值衰减系数 alpha(t) 的对数。
        #
        # 对于线性 beta 调度：
        #
        # beta(t) = beta_min + (beta_max - beta_min) * t
        #
        # alpha(t) 可以写为：
        #
        # alpha(t) = exp(-0.5 * integral_0^t beta(s) ds)
        #
        # 对 beta(s) 积分后：
        #
        # integral_0^t beta(s) ds
        # = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
        #
        # 因此：
        #
        # log(alpha(t))
        # = -0.5 * beta_min * t
        #   -0.25 * (beta_max - beta_min) * t^2
        #
        # 这里先计算 log(alpha(t))，
        # 再通过指数函数恢复 alpha(t)。
        mean_log_coeff = -0.25 * t ** 2 * \
            (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t

        # 计算边缘分布的均值。
        #
        # mean = alpha(t) * x
        #
        # 随着 t 增大，alpha(t) 逐渐减小，
        # 原始样本 x 中的信息逐渐衰减。
        mean = torch.exp(mean_log_coeff) * x

        # 计算边缘分布的标准差。
        #
        # std = sqrt(1 - alpha(t)^2)
        #
        # 因为：
        #
        # alpha(t) = exp(mean_log_coeff)
        #
        # 所以：
        #
        # alpha(t)^2 = exp(2 * mean_log_coeff)
        #
        # 当 t 接近 0 时，std 接近 0；
        # 当 t 增大时，std 逐渐增大。
        std = torch.sqrt(1 - torch.exp(2. * mean_log_coeff))

        # 返回边缘分布的均值和标准差。
        return mean, std

    # 单独计算线性 VP-SDE 的扩散系数 g(t)。
    #
    # 对应公式：
    #
    # g(t) = sqrt(beta(t))
    #
    # 当只需要扩散系数而不需要漂移项时，
    # 可以调用该方法，避免额外计算。
    def diffusion_coeff(self, t):

        # 计算当前时刻的线性噪声增长率 beta(t)。
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min

        # 计算扩散系数。
        diffusion = torch.sqrt(beta_t)

        # 返回扩散系数 g(t)。
        return diffusion

    # 单独计算线性 VP-SDE 边缘分布的标准差。
    #
    # 对于线性 VP-SDE：
    #
    # std(t) = sqrt(1 - alpha(t)^2)
    #
    # 其中：
    #
    # alpha(t)^2
    # = exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t^2)
    def marginal_prob_std(self, t):

        # 计算 alpha(t)^2。
        #
        # 变量名 discount 表示信号能量随扩散时间增加而逐渐衰减。
        #
        # discount
        # = exp(-0.5 * t^2 * (beta_max - beta_min) - beta_min * t)
        discount = torch.exp(
            -0.5 * t ** 2 * (self._beta_max - self._beta_min) - self._beta_min * t)

        # 根据 std(t) = sqrt(1 - alpha(t)^2) 计算标准差。
        std = torch.sqrt(1 - discount)

        # 返回边缘分布标准差。
        return std


# 定义采用指数 beta 调度的 sub-VP-SDE。
#
# sub-VP-SDE 是 Variance Preserving SDE 的一种变体。
# 当前类将 beta(t) 定义为指数形式：
#
# beta(t) = sigma^t
#
# 注意：
#
# 当前类的构造函数开头包含：
#
# raise NotImplementedError
#
# 因此，该类目前处于“未实现”状态。
# 只要尝试实例化 subVPSDE_exp，就会立即抛出异常。
#
# 下面的方法虽然已经写出，但在当前状态下无法通过正常实例化调用。
class subVPSDE_exp(SDE):

    # 初始化指数调度的 sub-VP-SDE。
    #
    # 参数：
    # sigma：指数增长底数，默认值为 25.0。
    #
    # 根据当前代码：
    #
    # beta(t) = sigma^t
    def __init__(self, sigma=25.0):
        """
        subVPSDE

        $beta(t) = sigma^t$
        """

        # 主动抛出异常，表示该类尚未完成实现或尚未启用。
        #
        # 由于该语句位于构造函数最前面，
        # 后续 super().__init__() 和 self._sigma = sigma 均不会执行。
        raise NotImplementedError

        # 调用父类初始化方法。
        #
        # 注意：由于前面已经抛出了异常，
        # 当前代码实际执行时不会运行到这一行。
        super().__init__()

        # 保存指数调度中的 sigma。
        #
        # 注意：由于前面已经抛出了异常，
        # 当前代码实际执行时不会运行到这一行。
        self._sigma = sigma

    # 定义该 SDE 的终止时间。
    #
    # 当前实现同样使用连续时间范围 [0, 1]。
    @property
    def T(self):
        return 1.0

    # 计算指数 sub-VP-SDE 的漂移项和扩散项。
    #
    # 根据当前代码：
    #
    # beta(t) = sigma^t
    #
    # drift = -0.5 * beta(t) * x
    #
    # diffusion
    # = sqrt(beta(t) * (1 - discount))
    #
    # 其中 discount 是一个与时间有关的衰减项。
    def sde(self, x, t):

        # 获取输入张量 x 的形状。
        shape = x.shape

        # 构造用于广播的目标形状。
        #
        # 例如：
        #
        # x.shape = [batch_size, horizon, feature_dim]
        #
        # 则：
        #
        # reshape = [-1, 1, 1]
        reshape = [-1] + [1, ] * (len(shape) - 1)

        # 将时间张量 t 调整为可与 x 广播运算的形状。
        t = t.reshape(reshape)

        # 根据指数调度计算当前时刻的 beta(t)。
        #
        # beta(t) = sigma^t
        beta_t = self._sigma ** t

        # 计算漂移项。
        #
        # drift = -0.5 * beta(t) * x
        drift = - 0.5 * beta_t * x

        # 计算扩散系数中使用的衰减项。
        #
        # 根据当前代码：
        #
        # discount
        # = exp(-2 * (beta(t) - 1) / log(sigma))
        #
        # 注意：
        #
        # 当前 self._sigma 在构造函数中被设置为普通 Python 浮点数。
        # 如果移除 raise NotImplementedError 并直接运行此代码，
        # torch.log(self._sigma) 可能需要进一步确认类型兼容性。
        # 这里仅解释原始代码，不对实现进行改动。
        discount = torch.exp(- 2 * (beta_t - 1) / torch.log(self._sigma))

        # 计算扩散系数。
        #
        # diffusion = sqrt(beta(t) * (1 - discount))
        diffusion = torch.sqrt(beta_t * (1.0 - discount))

        # 返回漂移项和扩散项。
        return drift, diffusion

    # 计算指数 sub-VP-SDE 的边缘分布参数。
    #
    # 根据当前代码：
    #
    # discount
    # = exp(-(sigma^t - 1) / log(sigma))
    #
    # mean = discount * x
    #
    # std = clamp(1 - discount, min=STD_MIN)
    #
    # 注意：
    #
    # 这里变量名为 std，但代码中并没有对 1 - discount 开平方。
    # 当前注释仅说明原始实现的实际计算逻辑，不对其进行修改。
    def marginal_prob(self, x, t):

        # 获取输入样本 x 的形状。
        shape = x.shape

        # 构造广播所需的形状。
        reshape = [-1] + [1, ] * (len(shape) - 1)

        # 将时间张量 t 调整为可广播的形状。
        t = t.reshape(reshape)

        # 计算信号衰减系数。
        #
        # discount
        # = exp(-(sigma^t - 1) / log(sigma))
        #
        # 注意：
        #
        # 与 sde(...) 方法相同，
        # torch.log(self._sigma) 的输入类型需要在启用该类前进一步确认。
        discount = torch.exp(-(self._sigma ** t - 1) / torch.log(self._sigma))

        # 计算边缘分布均值。
        #
        # mean = discount * x
        mean = discount * x

        # 按照原始代码计算名为 std 的量。
        #
        # std = max(1 - discount, STD_MIN)
        #
        # torch.clamp(..., min=STD_MIN) 用于防止结果过小，
        # 从而降低后续除法或对数计算中的数值不稳定风险。
        #
        # 注意：
        #
        # 这里严格保留了原始代码，
        # 没有额外添加平方根操作。
        std = torch.clamp(1 - discount, min=STD_MIN)

        # 返回均值和名为 std 的量。
        return mean, std

    # 单独计算指数 sub-VP-SDE 的扩散系数。
    #
    # 该方法与 sde(...) 中扩散项的计算逻辑一致，
    # 但不需要输入样本 x，也不计算漂移项。
    def diffusion_coeff(self, t):

        # 计算指数调度：
        #
        # beta(t) = sigma^t
        beta_t = self._sigma ** t

        # 计算衰减项：
        #
        # discount
        # = exp(-2 * (beta(t) - 1) / log(sigma))
        discount = torch.exp(- 2 * (beta_t - 1) / torch.log(self._sigma))

        # 计算扩散系数：
        #
        # diffusion
        # = sqrt(beta(t) * (1 - discount))
        diffusion = torch.sqrt(beta_t * (1.0 - discount))

        # 返回扩散系数。
        return diffusion

    # 单独计算指数 sub-VP-SDE 边缘分布中名为 std 的量。
    #
    # 根据当前代码：
    #
    # discount
    # = exp(-(sigma^t - 1) / log(sigma))
    #
    # std
    # = clamp(1 - discount, min=STD_MIN)
    #
    # 注意：
    #
    # 当前实现没有对 1 - discount 开平方。
    # 注释仅解释代码，不修改其计算方式。
    def marginal_prob_std(self, t):

        # 计算信号衰减系数。
        discount = torch.exp(-(self._sigma ** t - 1) / torch.log(self._sigma))

        # 设置结果下限，避免数值过小。
        std = torch.clamp(1 - discount, min=STD_MIN)

        # 返回名为 std 的量。
        return std