"""HDP-NAVSIM 风格的 rollout 轨迹扰动。"""

from __future__ import annotations

import torch


def augment_trajectory_batch(
    trajectories: torch.Tensor,
    std: float = 0.5,
) -> torch.Tensor:
    """在轨迹自身朝向坐标系中加入整条轨迹共享的纵向/横向偏移。

    Args:
        trajectories: 形状为 ``[..., T, 4]`` 的 ``[x, y, cos_yaw, sin_yaw]``。
        std: 纵向和横向偏移的高斯标准差，单位为米。

    Returns:
        与输入同形状的新张量；航向 ``cos_yaw/sin_yaw`` 保持不变。
    """

    if trajectories.ndim < 3 or trajectories.shape[-1] != 4:
        raise ValueError("trajectories must have shape [..., T, 4]")
    if std < 0:
        raise ValueError("std must be non-negative")
    if std == 0:
        return trajectories

    # 每条候选只采样一对偏移，并沿所有未来时刻广播；这与 HDP-NAVSIM
    # augment_trajectory_batch() 的语义一致，而不是给每个轨迹点独立加抖动。
    offset_shape = (*trajectories.shape[:-2], 1)
    longitudinal = torch.randn(
        offset_shape,
        device=trajectories.device,
        dtype=trajectories.dtype,
    ) * std
    lateral = torch.randn(
        offset_shape,
        device=trajectories.device,
        dtype=trajectories.dtype,
    ) * std

    x = trajectories[..., 0]
    y = trajectories[..., 1]
    cos_yaw = trajectories[..., 2]
    sin_yaw = trajectories[..., 3]
    x_new = x + longitudinal * cos_yaw - lateral * sin_yaw
    y_new = y + longitudinal * sin_yaw + lateral * cos_yaw
    return torch.stack((x_new, y_new, cos_yaw, sin_yaw), dim=-1)
