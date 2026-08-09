import abc
import torch
# 【HDP 与原 Diffusion-Planner 的区别：参数化转换类型注解】HDP 为新增的
# expand_dim()/transform() 引入 Tensor、Optional、List 和 Union；原版没有这些导入。
from torch import Tensor
from typing import Optional, List, Union


STD_MIN = 1e-6
# 【HDP 与原 Diffusion-Planner 的区别：sub-VP 实现移除】原版还定义了
# subVPSDE_exp，但其 __init__ 首行直接 raise NotImplementedError，实际无法实例化；
# HDP 删除了该类，因此当前 STD_MIN 在本文件内保留但没有被可执行代码使用。


# 【HDP 与原 Diffusion-Planner 的区别：广播辅助函数】HDP 新增该函数，为统一的
# x_start/noise/v/score 转换把 batch 级 alpha/sigma 扩展到参考轨迹张量的维数。
def expand_dim(x: torch.Tensor, ref: torch.Tensor):
    # 将按 batch 给出的 alpha/sigma 扩展为可与轨迹张量广播的形状。
    return x.reshape(*x.shape, *([1] * (ref.ndim - x.ndim)))


# 【实现核对：公共 SDE 逻辑一致】SDE 抽象类全部接口，以及 VPSDE_linear 原有的
# __init__、T、sde、marginal_prob、diffusion_coeff、marginal_prob_std 均与原版逐语句相同。
class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):
        """Construct an SDE.
        """
        super().__init__()

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """
        sde: A function that returns the drift and diffusion coefficients of the SDE.

        return (drift $f(x,t)$, diffusion $g(t)$)
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return mean, std
        """
        pass

    @abc.abstractmethod
    def diffusion_coeff(self, t):
        """
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.

        return $g(t)$
        """
        pass

    @abc.abstractmethod
    def marginal_prob_std(self, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return std
        """
        pass


class VPSDE_linear(SDE):
    def __init__(self, beta_max=20.0, beta_min=0.1):
        """
        VP SDE

        SDE:
        $ \mathrm{d}x = -\frac{\beta(t)}{2} x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W_t $
        """
        super().__init__()

        self._beta_max = beta_max
        self._beta_min = beta_min

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        """
        SDE of diffusion process

        drift = $-\frac{\beta(t)}{2} x$
        diffusion = $\sqrt{\beta(t)}$
        """
        shape = x.shape
        reshape = [-1] + [1, ] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = - 0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
        """
        shape = x.shape
        reshape = [-1] + [1, ] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t ** 2 * \
            (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t

        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2. * mean_log_coeff))
        return mean, std
    
    def marginal_alpha(self, t):
        # 【HDP 与原 Diffusion-Planner 的区别：显式 alpha 接口】原版只在
        # marginal_prob 内部计算 alpha(t)；HDP 单独暴露它，供 transform() 参数化转换。
        mean_log_coeff = -0.25 * t ** 2 * \
            (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t
        return torch.exp(mean_log_coeff)

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(
            -0.5 * t ** 2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std

    def transform(self, pattern, input: Tensor, t: Tensor, x_t: Optional[Tensor]) -> Union[Tensor, List[Tensor]]:
        # 【HDP 与原 Diffusion-Planner 的区别：统一参数化转换】HDP 新增转换入口，
        # 支持以 "source->target" 在 x_start、noise、v、score 之间转换；原版 SDE
        # 没有该方法，只在 loss/model_wrapper 的局部逻辑中处理部分参数化。

        src, tgt = pattern.split("->")

        # 输入和目标参数化类型相同时不需要执行中间转换，直接返回原 Tensor。
        # 例如 x_start->x_start 若先绕行 x_start->noise->x_start，两个分母中的
        # 1e-6 会引入不必要的数值误差，也会增加额外计算。
        if src == tgt:
            return input

        alpha_t = expand_dim(self.marginal_alpha(t), input)
        sigma_t = expand_dim(self.marginal_prob_std(t), input)

        # ---- src -> noise (epsilon) ----
        if src == "noise":
            noise = input
        elif src == "score":
            noise = -input * expand_dim(sigma_t, input)
        elif src == "x_start":
            noise = (x_t - alpha_t * input) / (sigma_t + 1e-6)
        elif src == "v":
            # noise = sigma * x_t + alpha * v
            noise = expand_dim(sigma_t, input) * x_t + alpha_t * input
        else:
            raise ValueError(f"Unknown src: {src}")

        # ---- noise (epsilon) -> tgt ----
        if tgt == "noise":
            output = noise
        elif tgt == "score":
            output = -noise / (sigma_t + 1e-6)
        elif tgt == "x_start":
            output = (x_t - sigma_t * noise) / (alpha_t + 1e-6)
        elif tgt == "v":
            # v = (noise - sigma * x_t) / alpha
            output = (noise - sigma_t * x_t) / (alpha_t + 1e-6)
        else:
            raise ValueError(f"Unknown tgt: {tgt}")

        return output
