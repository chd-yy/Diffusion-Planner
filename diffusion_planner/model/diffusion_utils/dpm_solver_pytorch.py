# 本文件实现 DPM-Solver / DPM-Solver++ 的核心采样逻辑。
# 本注释版仅新增以 # 开头的中文解释；原始代码、原始注释、缩进和执行逻辑均保持不变。
# 阅读顺序建议：NoiseScheduleVP -> model_wrapper -> DPM_Solver.sample -> 各阶更新公式 -> 工具函数。
#
import torch


# NoiseScheduleVP：统一封装 VP（Variance Preserving，方差保持）类型前向扩散过程。
# 它把离散 DDPM 的训练步或连续 VPSDE 的时间映射为 alpha_t、sigma_t 和 lambda_t。
# lambda_t = log(alpha_t) - log(sigma_t)，也称 half-logSNR，是求解器常用的时间坐标。
class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).
            创建一个用于表示前向 SDE 的包装类，其中 SDE 的类型为 VP,也就是方差保持型 SDE。
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        更新：为了支持离散时间扩散模型，我们对 log_alpha_t:log(alpha_t) 实现了分段线性插值。

        对于离散时间扩散模型，推荐使用：

        schedule='discrete'

        尤其是在处理高分辨率图片时。
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        前向 SDE 保证条件分布满足 q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I );
        其中 alpha_t 和 sigma_t 是时间 t 的函数.

        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        这个量称为 half-logSNR,即信噪比对数的一半,定义为 lambda_t = log(alpha_t) - log(sigma_t)。
        DPM-Solver 的求解算法通常以 lambda_t 作为时间坐标。

        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
        因此，该类实现了用于计算 $\alpha_t$、$\sigma_t$ 和 $\lambda_t$ 的函数。对于 t in [0, T]，我们有：
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
        此外，由于 $\lambda(t)$ 是一个可逆函数，因此该类还支持其反函数：
            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        该类同时支持两种类型的扩散概率模型：离散时间扩散模型, 连续时间扩散模型。
        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            对于训练在 n = 0, 1, ..., N-1 上的离散时间 DPM, 我们通过 t_i = (i + 1) / N 将离散步转换为连续时间步。

            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            例如对于 N = 1000, 我们有 t_0 = 1e-3 和 T = t_{N-1} = 1。我们从时间 T = 1 走到时间 t_0 = 1e-3 来求解对应的扩散 ODE。

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                betas 是一个 torch.Tensor, 表示离散时间扩散模型中的噪声强度数组, beta_n 控制第 n 步加入多少噪声。

                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod 是一个 torch.Tensor, 表示离散时间扩散模型中的 alpha 累积乘积数组, 

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            注意我们总是有 alphas_cumprod = cumprod(1 - betas)（意思是：先把每一步的 beta 转成“保留比例” 1 - beta ,然后做累乘)。
            因此，我们只需要设置 `betas` 和 `alphas_cumprod` 中的一个。

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    **alpha_{t_n} = \sqrt{\hat{alpha_n}}**,
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:
            对于连续时间模型，该类支持线性 VP-SDE。
            线性 VP-SDE 的噪声调度超参数是 Yang Song 的 ScoreSDE 中的默认设置：
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.
            通常使用线性函数：
                beta(t) = beta_min + (beta_max - beta_min) * t
            随着时间增大，噪声强度逐渐增加。
        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
            返回一个用于表示 VP 类型前向 SDE 的包装对象。
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        如果已有 DDPM 中每一步的噪声强度数组 betas,可以写成：
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        如果已经计算出累计乘积数组 alphas_cumprod,可以写成:
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        对于连续时间 VP-SDE,可以设置线性噪声调度:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        # 只允许两类调度：离散训练步的插值版本，以及连续线性 beta(t) 版本。
        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        # 保存调度类型，后续所有边缘量计算都会根据该字段选择离散或连续公式。
        self.schedule = schedule
        # 离散分支：从 betas 或 alphas_cumprod 构造每个训练步对应的 log(alpha_t)。
        if schedule == 'discrete':
            # 若传入 betas，则先使用 alpha_bar_n = prod_{i<=n}(1 - beta_i)，再取 0.5 * log(alpha_bar_n)。
            if betas is not None:
                # cumsum(log(1 - beta)) 等价于 log(cumprod(1 - beta))；乘 0.5 是因为 alpha_t = sqrt(alpha_bar_t)。
                # alphas_cumprod = torch.cumprod(1. - betas, dim=0)
                # log_alphas = 0.5 * torch.log(alphas_cumprod)
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            # 若未传入 betas，则要求直接提供累计乘积 alphas_cumprod。
            else:
                assert alphas_cumprod is not None
                # 同样取 0.5 * log，是为了得到 DPM-Solver 记号中的 log(alpha_t)。
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            # 扩散终点统一设为 T = 1。
            self.T = 1.
            # 裁剪末端可能数值不稳定的点，然后整理为 [1, N]，便于批量插值。
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            # 裁剪后实际保留的离散时间步数量。
            self.total_N = self.log_alpha_array.shape[1]
            # 把离散训练步 i = 0,...,N-1 映射到连续标签 (i + 1) / N；因此不包含 t = 0。
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)

            # self.t_array          = [[1/N, 2/N, 3/N, ..., 1]]
            # self.log_alpha_array  = [[log_alpha_0, log_alpha_1, ..., log_alpha_{N-1}]]
        # 连续线性 VPSDE 分支：beta(t) 在 beta_0 与 beta_1 之间线性变化。
        else:
            self.T = 1.
            # total_N 在连续分支中主要用于提供默认最小采样时间 1 / total_N。
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    # 对离散 log(alpha) 数组执行数值稳定性裁剪。
    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        # 由 VP 关系 alpha_t^2 + sigma_t^2 = 1，反推出 log(sigma_t)。
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        # 计算 half-logSNR：lambda_t = log(alpha_t) - log(sigma_t)。
        lambs = log_alphas - log_sigmas  
        # lambda 随时间通常单调下降。翻转后用 searchsorted 查找低于阈值的末端区域。
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        # idx > 0 表示扩散末端存在需要删除的极低 logSNR 点。
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    # 计算 log(alpha_t)：这是 alpha、sigma、lambda 等边缘量的基础。
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        # 离散调度没有闭式表达式，因此在预存关键点之间做可微分的分段线性插值。
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        # 线性 VPSDE 有闭式积分：log(alpha_t) = -0.5 * integral_0^t beta(s) ds。
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    # 由 log(alpha_t) 取指数得到均值缩放系数 alpha_t。
    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    # 由 VP 约束 alpha_t^2 + sigma_t^2 = 1 得到标准差 sigma_t。
    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    # 计算 half-logSNR。lambda 越大，信号相对噪声越强；反向采样时通常从小 lambda 走向大 lambda。
    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        # 先得到 log(alpha_t)。
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        # 再由 VP 约束计算 log(sigma_t)，避免先求 sigma 再取对数。
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    # 把 half-logSNR 反解为时间 t，供 logSNR 均匀步长和高阶中间节点使用。
    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        # 连续线性 VPSDE 可以解析反解。logaddexp 用于稳定计算 log(1 + exp(-2 * lambda))。
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        # 离散调度先从 lambda 恢复 log(alpha)，再对反向排列的关键点做插值求 t。
        elif self.schedule == 'discrete':
            # 利用 alpha^2 = sigmoid(2 * lambda)，以稳定形式计算 log(alpha)。
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            # 由于 log(alpha) 和 t 的原始数组方向相反，先翻转再调用同一个插值函数。
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


# model_wrapper：把不同训练参数化方式、不同引导方式统一包装成“噪声预测函数”。
# DPM-Solver 的内部接口最终只需要 model_fn(x, t_continuous) -> predicted_noise。
def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.
        创建一个用于噪声预测模型的包装函数。
    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.
    DPM-Solver 需要求解连续时间的扩散 ODE。
    对于训练在离散时间标签上的 DPM,我们首先需要把模型函数包装成一个接受连续时间作为输入的噪声预测模型。
    We support four types of the diffusion model by setting `model_type`:
    我们支持四种类型的扩散模型，通过设置 `model_type` 来区分：
        1. "noise": noise prediction model. (Trained by predicting noise).
                模型直接预测噪声 epsilon,训练时的损失通常是 MSE(epsilon_pred, epsilon_true)。
        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).
            模型直接预测干净数据 x_0,训练时的损失通常是 MSE(x_0_pred, x_0_true)。
        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].
            模型预测 velocity, 这是 [1] 的附录 D 中推导出的一个参数化方式，在 Imagen-Video [2] 中使用。
            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).
    
        4. "score": marginal score function. (Trained by denoising score matching).
            “score” 参数化的模型直接预测边缘 score 函数，即 score(x_t, t) = grad_{x_t} log q_{t|0}(x_t | x_0)。
            Note that the score function and the noise prediction model follows a simple relationship:
            注意 score 函数和噪声预测模型之间有一个简单的关系：
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
                在扩散模型中，“预测噪声”和“预测 score”本质上是两种等价的参数化方式。
                只要知道噪声强度 $\sigma_t$，就可以在二者之间相互转换。
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        我们支持三种类型的 DPM 引导采样，通过设置 `guidance_type` 来区分：
        1. "uncond": unconditional sampling by DPMs.
            无条件采样：直接使用 DPM 模型的输出作为噪声预测。
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            分类器引导采样：在 DPM 模型的基础上，使用一个额外的分类器提供的梯度信息来调整采样方向。
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            `` 

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
                返回一个标量，表示条件 cond 在当前带噪样本 x 和时间 t 下的对数概率或与其等价的可求梯度的标量函数。
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            无分类器引导采样：使用同一个条件 DPM 模型分别进行有条件和无条件的前向预测，然后在两者之间进行外推。
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).
        

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).
    "t_input" 是模型使用的时间标签，可能是离散时间标签（例如 0 到 999)或连续时间标签(例如 epsilon 到 T)。

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    我们把模型函数包装成一个只接受 `x` 和 `t_continuous` 作为输入，并输出预测噪声的函数：
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)         
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.
    其中 `t_continuous` 是连续时间标签（例如 epsilon 到 T)。我们把 `model_fn` 作为 DPM-Solver 的输入模型函数。
    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        模型：一个扩散模型，格式如上所述。
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        噪声调度：一个噪声调度对象，例如 NoiseScheduleVP。
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
                    模型类型：一个字符串，表示扩散模型的参数化类型。
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
                    模型其他输入：一个字典，包含模型函数的其他输入。
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
                    引导类型：一个字符串，表示采样的引导类型。
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
                    条件：一个 PyTorch 张量，表示引导采样的条件。仅用于 "classifier" 或 "classifier-free" 引导类型。
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
                    无条件条件：一个 PyTorch 张量，表示无条件采样的条件。仅用于 "classifier-free" 引导类型。
        guidance_scale: A `float`. The scale for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
                    引导强度：一个浮点数，表示引导采样的强度。仅用于 "classifier" 或 "classifier-free" 引导类型。
        classifier_fn: A classifier function. Only used for the classifier guidance.
                    "classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)"
                    分类器函数：一个分类器函数，仅用于分类器引导。格式如上所示。
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
        返回一个噪声预测模型，接受带噪数据和连续时间作为输入。
    """

    # 将求解器使用的连续时间标签转换成底层模型实际接收的时间标签。
    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        将求解器使用的连续时间 `t_continuous`（在 [epsilon, T] 范围内）转换为模型输入时间。
        对于离散时间 DPM,我们把 `t_continuous` 从 [1 / N, 1] 映射到 [0, 1000 * (N - 1) / N]，以对应模型训练时的离散标签。
        对于连续时间 DPM,我们直接使用 `t_continuous` 作为模型输入时间。
        """
        # 离散模型通常按 0 到 999 的标签训练，因此把 [1/N, 1] 线性映射到对应标签范围。
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    # 调用底层模型，并把其输出统一转换为噪声 epsilon 的预测。
    def noise_pred_fn(x, t_continuous, cond=None):
        # 先将连续时间转换为模型训练时使用的时间格式。
        t_input = get_model_input_time(t_continuous)
        # 无条件模型不需要额外条件张量；有条件模型将 cond 作为第三个位置参数传入。
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        # noise 参数化：模型输出本身就是 epsilon，无需转换。
        if model_type == "noise":
            return output
        # x_start 参数化：模型预测 x_0，根据 x_t = alpha_t * x_0 + sigma_t * epsilon 反解 epsilon。
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(sigma_t, x.dim())
        # v 参数化：根据 velocity 参数化与 epsilon 的线性关系恢复噪声预测。
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return expand_dims(alpha_t, x.dim()) * output + expand_dims(sigma_t, x.dim()) * x
        # score 参数化：使用 epsilon = -sigma_t * score 将 score 转为噪声预测。
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    # 分类器引导需要计算条件对当前带噪样本的梯度 grad_x log p_t(condition | x_t)。
    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        # 采样整体常在 no_grad 环境中运行；此处临时开启梯度以计算分类器引导项。
        with torch.enable_grad():
            # detach 避免连接到此前计算图；requires_grad_(True) 只对当前 x_t 求梯度。
            x_in = x.detach().requires_grad_(True)
            # 分类器输出条件对数概率或与其等价的可求梯度标量。
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            # 对 batch 内标量求和后求梯度，得到与 x_in 同形状的引导方向。
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    # 对外暴露的统一模型函数：根据 guidance_type 选择无条件、分类器引导或 classifier-free guidance。
    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        # 无条件采样：直接返回噪声预测。
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        # 分类器引导：在噪声预测上减去与分类器梯度成比例的修正项。
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            # sigma_t 用于把 score 空间中的分类器梯度换算到噪声参数化空间。
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        # Classifier-Free Guidance：使用同一个扩散模型分别得到无条件与有条件预测。
        elif guidance_type == "classifier-free":
            # guidance_scale = 1 或未提供无条件条件时，直接使用有条件预测，避免多做一次前向。
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                # x_in = torch.cat([x] * 2)
                # t_in = torch.cat([t_continuous] * 2)
                # c_in = torch.cat([unconditional_condition, condition])
                # noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                # 分别计算无条件噪声预测与有条件噪声预测。上方保留的注释代码展示了可选的 batch 拼接优化。
                noise_uncond = noise_pred_fn(x, t_continuous, cond=unconditional_condition)
                noise = noise_pred_fn(x, t_continuous, cond=condition)
                # 沿着“有条件 - 无条件”的方向外推；guidance_scale 越大，条件约束通常越强。
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    # 尽早校验配置，防止错误参数在采样循环深处才暴露。
    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


# DPM_Solver：执行确定性扩散 ODE 的数值积分。
# 它同时支持原始 DPM-Solver（噪声预测形式）和 DPM-Solver++（数据预测形式）。
class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver. 
        创建一个 DPM-Solver 实例。
        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).
        我们同时支持 DPM-Solver(`algorithm_type="dpmsolver"`）和 DPM-Solver++(`algorithm_type="dpmsolver++"`）。
        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).
        我们还支持 Imagen[1] 中的“动态阈值”方法。对于像素空间的扩散模型，您可以同时设置 `algorithm_type="dpmsolver++"` 和 `correcting_x0_fn="dynamic_thresholding"` 来使用动态阈值。
        对于具有大引导尺度的像素空间 DPM,动态阈值可以大大提高样本质量。请注意，该阈值方法**不适用于**潜空间 DPM(例如 stable-diffusion)。
        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.
        为了支持图像到图像应用中的高级算法，我们还支持 x0 和 xt 的修正函数。
        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
            model_fn: 一个噪声预测模型函数，接受连续时间输入（t 在 [epsilon, T] 范围内）：
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            noise_schedule: 一个噪声调度对象，例如 NoiseScheduleVP。
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            algorithm_type: 一个字符串，表示算法类型，"dpmsolver" 或 "dpmsolver++"。
            correcting_x0_fn: A `str` or a function with the following format:
            correcting_x0_fn: 一个字符串或一个函数，格式如下：
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                这个函数用于修正数据预测模型在每个采样步骤的输出，例如：
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                如果 `correcting_xt_fn` 不为 None,则在每个采样步骤后对中间状态 xt 进行额外修正。函数格式如下：
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                这个函数用于修正每个采样步骤的中间样本 xt，例如：
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            thresholding_max_val: 一个浮点数，表示阈值的最大值。仅在使用 `dpmsolver++` 和 `correcting_x0_fn="dynamic_thresholding"` 时有效。
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: 一个浮点数，表示动态阈值的比例（详见 Imagen[1]）。仅在使用 `dpmsolver++` 和 `correcting_x0_fn="dynamic_thresholding"` 时有效。

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        # 内部模型总是接收 batch 大小匹配的时间向量。若外部传入单元素 t，则 expand 到 batch 维。
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        # 保存噪声调度对象，所有 alpha、sigma、lambda 的计算均由它提供。
        self.noise_schedule = noise_schedule
        # 限定算法类型，避免混入未实现的更新公式。
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        # 若传入特殊字符串，则把 x_0 修正器替换为内置动态阈值函数。
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        # correcting_xt_fn 可在每个采样步后对中间状态 x_t 进行额外修正。
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    # Imagen 动态阈值：按样本自适应压缩过大的 x_0 预测，常用于像素空间模型的大引导尺度。
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        # 记录 x0 的总维度，之后将每个 batch 的阈值扩展回可广播形状。
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        # 对每个样本取 |x0| 的高分位数，得到自适应阈值 s。
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        # 阈值至少为 thresholding_max_val，避免阈值过小导致不必要的放大。
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        # 先裁剪到 [-s, s]，再除以 s，将结果压缩到 [-1, 1]。
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    # 直接调用统一包装后的噪声预测模型。
    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    # 把噪声预测转换为数据预测 x_0，并按需应用 x_0 修正器。
    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        # 取得 epsilon_theta(x_t, t)。
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        # 由 x_t = alpha_t * x_0 + sigma_t * epsilon 反解 x_0。
        x0 = (x - sigma_t * noise) / alpha_t
        # 像素空间模型可在此处应用动态阈值或自定义后处理。
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    # 根据算法类型决定高阶更新公式实际使用的数据项：DPM-Solver++ 使用 x_0，原始 DPM-Solver 使用 epsilon。
    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    # 生成从起点 t_T 到终点 t_0 的 N + 1 个时间节点。
    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
            计算采样的中间时间步。
        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
            skip_type: 一个字符串，表示时间步的间隔类型。我们支持三种类型：
                - 'logSNR': uniform logSNR for the time steps.
                logSNR 均匀：先在线性 lambda 空间取点，再通过 inverse_lambda 映射回时间。
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                time_uniform 时间均匀：直接在 t 空间线性取点。(**推荐用于高分辨率数据**。)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
                time_quadratic 二次时间步：先对 sqrt(t) 均匀取点，再平方；会改变节点在时间轴上的密度。 (DDIM 在低分辨率数据中使用。)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_T: 一个浮点数，表示采样的起始时间（默认是 T）。
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            t_0: 一个浮点数，表示采样的结束时间（默认是 epsilon）。
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
            一个包含时间步的 PyTorch 张量，形状为 (N + 1,)。
        """
        # logSNR 均匀：先在线性 lambda 空间取点，再通过 inverse_lambda 映射回时间。
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        # 时间均匀：直接在 t 空间线性取点。
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        # 二次时间步：先对 sqrt(t) 均匀取点，再平方；会改变节点在时间轴上的密度。
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    # 为 singlestep 的 DPM-Solver-fast 分配每个外层区间使用的一阶、二阶或三阶求解器。
    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        得到单步 DPM-Solver 每个采样步骤的阶数分配。
        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        我们结合 DPM-Solver-1,2,3 来使用所有的函数评估，这被称为 "DPM-Solver-fast"。
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
        已知固定的函数评估次数 `steps`, DPM-Solver-fast 的采样过程如下：
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
                我们使用 `steps` 个 DPM-Solver-1 步（即 DDIM）。
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                令 K = (steps // 2). 我们取 K 或 (K + 1) 个中间时间步进行采样。
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                如果 steps % 2 == 0,我们使用 K 个 DPM-Solver-2 步。
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
                如果 steps % 2 == 1,我们使用 K 个 DPM-Solver-2 步和 1 个 DPM-Solver-1 步。
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                令 K = (steps // 3 + 1). 我们取 K 个中间时间步进行采样。
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================   
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
            skip_type: 一个字符串，表示时间步的间隔类型。我们支持三种类型：
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
            orders: 每个采样步骤的求解器阶数列表。
            timesteps_outer: A pytorch tensor of the time steps for the outer loop, with the shape (K,).
            timesteps_outer: 外层循环使用的时间步的 PyTorch 张量，形状为 (K,)。
        """
        # 三阶模式尽量使用三阶步，并根据 NFE 除以 3 的余数补一个二阶或一阶步。
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        # 二阶模式尽量使用二阶步；若 NFE 为奇数，最后补一个一阶步。
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        # 一阶模式中每次函数评估对应一个 DDIM 等价更新。
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        # logSNR 间隔需要直接按外层步数 K 划分，以复现实验设定。
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        # 其他间隔先生成细粒度节点，再按每段消耗的阶数累计索引抽取外层节点。
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    # 最终去噪到零：直接输出当前状态对应的 x_0 预测，相当于额外进行一次一阶终止处理。
    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    # 一阶单步更新：DPM-Solver-1；在常见设定下与 DDIM 更新等价。
    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.
        DPM-Solver-1（等价于 DDIM）从时间 `s` 到时间 `t` 的更新。
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
            x_t: 一个 PyTorch 张量，表示时间 `t` 的近似解。
             If `return_intermediate` is true, also return a dict of the intermediate values, which can be used for the higher-order updates to avoid redundant model evaluations.
             如果 `return_intermediate` 为真，还返回一个包含中间值的字典，这些值可用于高阶更新以避免冗余的模型评估。
        """
        # 使用局部别名 ns 简化后续噪声调度调用。
        ns = self.noise_schedule
        dims = x.dim()
        # 将起止时间变换到 lambda 坐标，并定义当前步长 h = lambda_t - lambda_s。
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        # 预先计算更新公式所需的 log(alpha)、sigma 和 alpha_t。
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        # DPM-Solver++ 分支使用数据预测 x_0 作为模型值。
        if self.algorithm_type == "dpmsolver++":
            # torch.expm1(z) 稳定计算 exp(z) - 1，尤其适合 |z| 较小时避免消减误差。
            phi_1 = torch.expm1(-h)
            # 允许调用者复用已计算的 model_s，减少一次模型前向。
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 闭式积分的一阶近似：传播旧状态，并加入基于模型值的修正。
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        # 原始 DPM-Solver 分支使用噪声预测 epsilon 作为模型值。
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    # 二阶单步更新：在区间 [s, t] 内增加一个中间节点 s1，以估计模型随 lambda 的变化。
    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 高阶公式提供 dpmsolver 与 taylor 两种离散化形式。
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        # 未显式给出 r1 时，中间节点默认位于 lambda 区间的中点。
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        # 在 lambda 坐标计算总步长 h。
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        # lambda_s1 = lambda_s + r1 * h；随后反解出真实时间 s1。
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        # 批量取得起点、中间点、终点处的噪声调度系数。
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        # DPM-Solver++：模型值是 x_0 预测。
        if self.algorithm_type == "dpmsolver++":
            # phi_11 对应子区间 s -> s1，phi_1 对应完整区间 s -> t。
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            # 先用一阶公式预测中间状态 x_s1。
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            # 在中间状态重新评估模型，从而获取高阶修正所需的变化量。
            model_s1 = self.model_fn(x_s1, s1)
            # dpmsolver 与 taylor 的差异体现在对 model_s1 - model_s 的系数选择。
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        # 原始 DPM-Solver：模型值改为噪声 epsilon，传播系数相应改用 alpha 与 sigma 的另一种组合。
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )
        # 自适应求解器可请求返回中间模型值，以便高阶嵌套更新复用。
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    # 三阶单步更新：使用两个中间节点 s1、s2，进一步估计模型的一阶与二阶变化。
    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 只接受两种已实现的高阶离散化形式。
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        # 默认在 lambda 区间的 1/3 与 2/3 位置布置两个中间节点。
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        # 计算 lambda 空间中的总步长。
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        # 构造两个中间 lambda，并通过 inverse_lambda 转回真实时间。
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        # 一次性计算各节点所需的 log(alpha)、sigma 和 alpha，减少重复调用。
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        # DPM-Solver++ 三阶分支：以下 phi 项来自闭式积分核的稳定展开。
        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            # phi_22、phi_2、phi_3 用于刻画更高阶积分修正。
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            # 起点模型值可由外部复用；若不存在才执行模型前向。
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 第一中间节点模型值同样支持复用。
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            # 利用 s 与 s1 的信息构造第二中间状态 x_s2。
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            # 在第二中间节点评估模型，得到三阶终点更新所需的信息。
            model_s2 = self.model_fn(x_s2, s2)
            # dpmsolver 形式直接使用 model_s2 - model_s 进行高阶修正。
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            # taylor 形式显式构造离散一阶导数 D1 与二阶导数 D2。
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        # 原始 DPM-Solver 三阶分支：整体结构相同，但以噪声预测形式书写。
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        # 按需返回中间模型值，便于嵌套求解或调试。
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    # 二阶多步更新：复用前两个时间点已计算的模型值，不再额外引入区间内中间节点。
    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        # 取最近两个历史模型值与对应时间。
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        # 将历史时间与目标时间映射到 lambda 坐标。
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        # h_0 是上一段 lambda 步长，h 是当前段 lambda 步长；r0 用于处理非均匀步长。
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        # D1_0 是根据最近两个模型值构造的归一化一阶差分。
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        # 根据算法类型选择 x_0 形式或 epsilon 形式的二阶多步公式。
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        return x_t

    # 三阶多步更新：复用最近三个时间点的模型值，构造更高阶差分修正。
    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        # 按从旧到新的顺序取出三个历史模型值和三个历史时间。
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        # 计算前两段历史步长与当前目标步长。
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        # D1_0、D1_1 是两个相邻区间的一阶差分。
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        # 组合一阶差分得到 D1，并用差分之差得到二阶修正 D2。
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        # DPM-Solver++ 采用数据预测形式的三阶多步更新。
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        # 原始 DPM-Solver 采用噪声预测形式的三阶多步更新。
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    # 单步求解器分发器：按照 order 调用对应的一阶、二阶或三阶实现。
    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # order 决定每个外层区间内部进行多少阶的近似。
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    # 多步求解器分发器：按照 order 调用对应实现，并复用历史模型值。
    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 一阶多步退化为一阶单步更新，但直接复用最后一个历史模型值。
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    # 自适应步长求解器：通过低阶与高阶结果的差估计局部误差，并动态调整 lambda 步长。
    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        # 初始化起点时间、终点 lambda、初始步长与误差估计所需的前一状态。
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        # 二阶模式嵌套一阶与二阶更新，形成 embedded 1/2 阶误差估计。
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        # 三阶模式嵌套二阶与三阶更新，形成 embedded 2/3 阶误差估计。
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        # 持续推进，直到真实时间 s 与目标时间 t_0 的差足够小。
        while torch.abs((s - t_0)).mean() > t_err:
            # 在 lambda 空间提出下一步，并反解成真实时间 t。
            t = ns.inverse_lambda(lambda_s + h)
            # 分别计算低阶和高阶近似；高阶计算复用低阶阶段产生的模型值。
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            # delta 为逐元素容许误差尺度，取绝对误差阈值与相对误差阈值中的较大者。
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            # 将每个样本的误差压平后计算 RMS 范数。
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            # E 是归一化局部误差；E <= 1 表示当前候选步满足容差。
            E = norm_fn((x_higher - x_lower) / delta).max()
            # 只有误差可接受时才提交高阶结果并推进当前时间。
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            # 根据误差大小缩放下一步 h，并确保不越过终点 lambda_0。
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            # 每轮消耗约 order 次模型函数评估，累计统计 NFE。
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    # 前向加噪工具：按给定时间 t 生成 x_t = alpha_t * x + sigma_t * noise。
    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        # 分别计算所有目标时间对应的信号系数与噪声系数。
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        # 未提供噪声时，为每个时间点和每个 batch 样本独立生成标准高斯噪声。
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        # 在最前面增加时间维，以便一次广播计算多个 t。
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        # 只有一个时间点时去掉额外时间维，使返回形状与常见调用习惯一致。
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    # inverse：沿正向时间方向调用同一 sample 逻辑，可将较干净样本推进到更噪的状态。
    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        # inverse 中默认从最小时间 1/N 出发，到扩散终点 T；随后交换为 sample 的起止参数。
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        # 复用统一采样器，避免维护另一套积分流程。
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    # sample：DPM-Solver 的主入口。它负责选择时间节点、选择求解模式并推进 x。
    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        # 默认从扩散终点 T 反向积分到最小可用时间 1 / total_N。
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        # 自适应模式的步数不固定，因此不能同时保存预定义步序列的中间结果。
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        # 自适应模式也不支持逐步 correcting_xt_fn，因为其步接受与拒绝机制更复杂。
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        # 后续生成的时间张量必须与输入 x 位于同一设备。
        device = x.device
        intermediates = []
        # 采样阶段不训练模型，关闭 autograd 可显著减少显存与计算开销。
        with torch.no_grad():
            # adaptive：忽略固定 steps，交给误差控制器决定实际步长。
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            # multistep：缓存历史模型值，用较少模型评估完成高阶更新。
            elif method == 'multistep':
                assert steps >= order
                # 生成 steps + 1 个时间节点，对应 steps 个积分区间。
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # 初始化第一个时间点及其模型值缓存。
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                # 可选的 xt 修正器在每次状态更新后执行；初始状态也可修正。
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                # 高阶多步法启动时历史不足，因此先用较低阶公式逐步填充缓存。
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    # 把新时间和新模型值加入历史缓存。
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                # 缓存充足后，进入常规 order 阶多步更新。
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    # 步数很少时，末端降低阶数有助于稳定采样。
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    # 滑动历史窗口：丢弃最旧项，为当前时间腾出位置。
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    # 最终时间点之后不再继续积分，因此无需额外计算最终模型值。
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            # singlestep / singlestep_fixed：每个外层区间独立完成，不维护跨区间模型缓存。
            elif method in ['singlestep', 'singlestep_fixed']:
                # singlestep 会自动混合不同阶数，以用满指定 NFE。
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                # singlestep_fixed 只使用固定阶数；不能整除的剩余 NFE 会被舍弃。
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                # 逐个处理外层积分区间。
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    # 为当前外层区间生成内部节点，以确定高阶公式中的相对位置 r1、r2。
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    # 将内部时间节点转为 lambda，后续比例均在 lambda 空间计算。
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    # r1、r2 表示中间节点位于完整 lambda 步长中的相对位置。
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    # 调用统一单步分发器完成当前区间更新。
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            # 可选的 denoise_to_zero 会额外执行一次 x_0 预测，因此增加一次 NFE。
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        # 根据 return_intermediate 决定是否同时返回每一步的中间状态。
        if return_intermediate:
            return x, intermediates
        else:
            return x



#############################################################
# other utility functions
#############################################################

# 分段线性插值工具：离散噪声调度与 inverse_lambda 都依赖它。
# 实现方式保持可微分，因此在需要时可以参与 autograd。
def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    # N 是待查询点数量，K 是每个通道的关键点数量。
    N, K = x.shape[0], xp.shape[1]
    # 把查询点 x 与关键点 xp 拼到同一轴，便于通过排序确定 x 落在哪个区间。
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    # 排序后的索引记录每个元素原先来自查询点还是关键点。
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    # 查询点在排序结果中的位置决定其左右邻接关键点。
    x_idx = torch.argmin(x_indices, dim=2)
    # 候选左端索引通常是查询点位置减一。
    cand_start_idx = x_idx - 1
    # 边界处理：若 x 位于最左或最右侧，选择最外侧两个关键点做线性外推。
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    # 根据边界情形确定右端点在排序数组中的位置。
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    # start_idx2 是对应到 yp 关键点数组中的左端索引。
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    # 将 yp 扩展出 batch 维，以便为每个查询点 gather 对应的左右 y 值。
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # 应用标准线性插值 / 外推公式。
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


# 广播辅助工具：把形状 [N] 的时间系数扩展成 [N, 1, ..., 1]。
def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    # 使用 None 索引追加 dims - 1 个单例维度，使系数可与图像或特征张量广播相乘。
    return v[(...,) + (None,)*(dims - 1)]
