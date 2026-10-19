"""Longitudinal guided-denoising energy used only while collecting RL rollout.

The guide is deliberately proposal-only: normal planner inference never passes a
``guidance_fn``.  Unlike the first A6 prototype, the longitudinal axis follows
the expert path at every future step instead of freezing the current velocity
direction for the complete horizon.  This avoids interpreting a turn as a
longitudinal speed error.
"""

from typing import Callable, Tuple

import torch


def _normalise(vector: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def expert_frenet_targets(
    ego_future: torch.Tensor,
    dt: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-step expert tangent and non-negative longitudinal speed.

    Args:
        ego_future: ``[B,T,4]`` ego-local absolute waypoints.
    Returns:
        tangent ``[B,T,2]`` and speed ``[B,T]``.
    """
    if ego_future.ndim != 3 or ego_future.shape[-1] < 2:
        raise ValueError("ego_future must have shape [B,T,D>=2]")
    if dt <= 0:
        raise ValueError("dt must be positive")
    origin = torch.zeros_like(ego_future[:, :1, :2])
    displacement = torch.diff(
        torch.cat([origin, ego_future[..., :2]], dim=1), dim=1
    )
    speed = torch.linalg.vector_norm(displacement, dim=-1) / dt
    tangent = _normalise(displacement)

    # Stopped/near-zero steps have no direction.  Use the closest previous
    # valid expert tangent, then fall back to the ego-local +x axis.
    valid = torch.linalg.vector_norm(displacement, dim=-1) > 1e-4
    fallback = torch.zeros_like(tangent)
    fallback[..., 0] = 1.0
    for step in range(tangent.shape[1]):
        if step:
            fallback[:, step] = torch.where(
                valid[:, step - 1, None],
                tangent[:, step - 1],
                fallback[:, step - 1],
            )
        tangent[:, step] = torch.where(
            valid[:, step, None], tangent[:, step], fallback[:, step]
        )
    return tangent, speed


def make_longitudinal_guidance_energy(
    dt: float = 0.1,
    trajectory_representation: str = "increment",
    cost_clip: float = 0.0,
) -> Callable:
    """Build the classifier function consumed by the DPM-Solver wrapper.

    ``deviation`` is assigned per proposal.  A value of ``+0.25`` asks for
    125% of the expert longitudinal speed and ``-0.25`` asks for 75%.
    ``gate`` is scene-derived and makes the energy exactly zero when guidance
    is not allowed.  The positive cost convention matches the vendored solver
    used by HDP and the independently verified A6 implementation.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if trajectory_representation not in {"increment", "absolute"}:
        raise ValueError("guided rollout supports increment or absolute representation")

    def energy(
        x: torch.Tensor,
        t,
        cond,
        *,
        state_normalizer,
        expert_tangent: torch.Tensor,
        expert_speed: torch.Tensor,
        deviation: torch.Tensor,
        gate: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del t, cond, kwargs
        physical = state_normalizer.inverse(x)
        if trajectory_representation == "increment":
            displacement = physical[..., :2]
        else:
            origin = torch.zeros_like(physical[:, :1, :2])
            displacement = torch.diff(
                torch.cat([origin, physical[..., :2]], dim=1), dim=1
            )
        velocity = displacement / dt
        longitudinal_speed = (velocity * expert_tangent).sum(dim=-1)
        target = expert_speed * (1.0 + deviation[:, None])
        cost = (longitudinal_speed - target).square().mean(dim=-1)
        if cost_clip > 0:
            cost = cost.clamp(max=float(cost_clip))
        return cost * gate

    return energy
