"""Open-loop trajectory metrics for HDP-nuPlan experiments.

The HDP sampler returns ego-local absolute positions after integrating its
internal displacement representation.  This module therefore accepts
``[x, y, cos(yaw), sin(yaw)]`` trajectories and never applies ``cumsum`` a
second time.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch


def _validate_trajectory_inputs(
    trajectories: torch.Tensor,
    targets: torch.Tensor,
    ego_current_state: torch.Tensor,
    dt: float,
) -> None:
    if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
        raise ValueError("trajectories must have shape [B,G,T,4]")
    if targets.ndim != 3 or targets.shape[-1] != 4:
        raise ValueError("targets must have shape [B,T,4]")
    if ego_current_state.ndim != 2 or ego_current_state.shape[-1] < 8:
        raise ValueError("ego_current_state must have shape [B,D] with D >= 8")
    if trajectories.shape[0] != targets.shape[0]:
        raise ValueError("trajectories and targets must have the same batch size")
    if trajectories.shape[0] != ego_current_state.shape[0]:
        raise ValueError("trajectories and ego_current_state must share batch size")
    if trajectories.shape[-2] != targets.shape[-2]:
        raise ValueError("trajectories and targets must have the same horizon")
    if trajectories.shape[-2] < 1:
        raise ValueError("trajectory horizon must be non-empty")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not trajectories.is_floating_point() or not targets.is_floating_point():
        raise ValueError("trajectories and targets must be floating-point tensors")


def _normalized_direction(states: torch.Tensor) -> torch.Tensor:
    """Return normalized ``[cos(yaw), sin(yaw)]`` with a stable fallback."""

    direction = states[..., 2:4]
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-8, direction / norm.clamp_min(1e-8), fallback)


def _motion_descriptors(
    xy: torch.Tensor,
    ego_current_state: torch.Tensor,
    dt: float,
) -> Dict[str, torch.Tensor]:
    """Compute descriptive kinematics for ``xy`` with shape ``[B,G,T,2]``."""

    batch_size, group_size = xy.shape[:2]
    origin = torch.zeros_like(xy[..., :1, :])
    displacement = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
    velocity = displacement / dt
    speed = torch.linalg.vector_norm(velocity, dim=-1)

    current_velocity = ego_current_state[:, None, None, 4:6].expand(
        batch_size, group_size, 1, 2
    )
    current_acceleration = ego_current_state[:, None, None, 6:8].expand(
        batch_size, group_size, 1, 2
    )
    acceleration = torch.diff(
        torch.cat([current_velocity, velocity], dim=-2), dim=-2
    ) / dt
    jerk = torch.diff(
        torch.cat([current_acceleration, acceleration], dim=-2), dim=-2
    ) / dt

    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    jerk_norm = torch.linalg.vector_norm(jerk, dim=-1)

    # Discrete signed curvature from consecutive displacement vectors.  The
    # first step has no predecessor and is defined as zero.  Very short or
    # stationary segments are clamped to avoid reporting numerical spikes.
    if displacement.shape[-2] > 1:
        previous = displacement[..., :-1, :]
        following = displacement[..., 1:, :]
        cross = previous[..., 0] * following[..., 1] - previous[..., 1] * following[..., 0]
        dot = (previous * following).sum(dim=-1)
        heading_change = torch.atan2(cross, dot)
        arc_length = 0.5 * (
            torch.linalg.vector_norm(previous, dim=-1)
            + torch.linalg.vector_norm(following, dim=-1)
        )
        curvature = torch.where(
            arc_length > 1e-4,
            heading_change / arc_length.clamp_min(1e-4),
            torch.zeros_like(heading_change),
        )
        curvature = torch.cat(
            [torch.zeros_like(curvature[..., :1]), curvature], dim=-1
        )
    else:
        curvature = torch.zeros_like(speed)

    return {
        "mean_speed_mps": speed.mean(dim=-1),
        "max_speed_mps": speed.max(dim=-1).values,
        "mean_acceleration_mps2": acceleration_norm.mean(dim=-1),
        "max_acceleration_mps2": acceleration_norm.max(dim=-1).values,
        "mean_jerk_mps3": jerk_norm.mean(dim=-1),
        "max_jerk_mps3": jerk_norm.max(dim=-1).values,
        "mean_abs_curvature_1pm": curvature.abs().mean(dim=-1),
        "max_abs_curvature_1pm": curvature.abs().max(dim=-1).values,
    }


def trajectory_open_loop_metrics(
    trajectories: torch.Tensor,
    targets: torch.Tensor,
    ego_current_state: torch.Tensor,
    *,
    dt: float = 0.1,
    horizons_seconds: Iterable[float] = (1.0, 3.0, 5.0, 8.0),
) -> Dict[str, torch.Tensor]:
    """Compute per-candidate open-loop metrics.

    Args:
        trajectories: Sampled trajectories ``[B,G,T,4]`` in physical units.
        targets: Expert trajectories ``[B,T,4]`` in physical units.
        ego_current_state: Current state ``[B,D]``.  Indices 4:6 and 6:8 are
            interpreted as current velocity and acceleration.
        dt: Seconds between adjacent future points.
        horizons_seconds: Reporting horizons.  Values beyond the available
            trajectory duration are rejected instead of silently truncated.

    Returns:
        A dictionary whose tensors all have shape ``[B,G]``.
    """

    _validate_trajectory_inputs(trajectories, targets, ego_current_state, dt)
    horizon_steps = trajectories.shape[-2]
    requested_horizons = tuple(float(value) for value in horizons_seconds)
    if len(set(requested_horizons)) != len(requested_horizons):
        raise ValueError("horizons_seconds must not contain duplicates")
    if any(value <= 0 for value in requested_horizons):
        raise ValueError("horizons_seconds must be positive")

    xy = trajectories[..., :2]
    target_xy = targets[:, None, :, :2]
    position_error = torch.linalg.vector_norm(xy - target_xy, dim=-1)

    prediction_direction = _normalized_direction(trajectories)
    target_direction = _normalized_direction(targets)[:, None]
    cosine = (prediction_direction * target_direction).sum(dim=-1).clamp(-1.0, 1.0)
    heading_error = torch.acos(cosine)

    metrics: Dict[str, torch.Tensor] = {
        "ade_m": position_error.mean(dim=-1),
        "fde_m": position_error[..., -1],
        "heading_mae_rad": heading_error.mean(dim=-1),
        "heading_fde_rad": heading_error[..., -1],
    }

    for horizon_seconds in requested_horizons:
        steps_float = horizon_seconds / dt
        steps = int(round(steps_float))
        if abs(steps_float - steps) > 1e-6:
            raise ValueError(
                f"horizon {horizon_seconds} seconds is not aligned to dt={dt}"
            )
        if steps > horizon_steps:
            raise ValueError(
                f"horizon {horizon_seconds} seconds exceeds available "
                f"duration {horizon_steps * dt} seconds"
            )
        label = f"{horizon_seconds:g}s"
        metrics[f"ade_{label}_m"] = position_error[..., :steps].mean(dim=-1)
        metrics[f"fde_{label}_m"] = position_error[..., steps - 1]
        metrics[f"heading_mae_{label}_rad"] = heading_error[..., :steps].mean(dim=-1)
        metrics[f"heading_fde_{label}_rad"] = heading_error[..., steps - 1]

    metrics.update(_motion_descriptors(xy, ego_current_state, dt))
    return metrics
