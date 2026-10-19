"""Safety-first subset selection for offline diffusion rollout candidates."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def safety_reward_spread_indices(
    reward: torch.Tensor,
    eligible: torch.Tensor,
    candidate_noise_scales: Sequence[float],
    target_size: int,
) -> torch.Tensor:
    """Select a fixed-size group while preserving safe reward variation.

    All eligible candidates are retained when they fit.  If there are more
    eligible candidates than slots, evenly spaced reward ranks are selected
    so group-relative learning observes both weak and strong safe behavior.
    Remaining slots are filled by low-temperature candidates first, then by
    higher reward.  The function never relabels an unsafe candidate as safe.
    """
    if reward.ndim != 2 or eligible.shape != reward.shape:
        raise ValueError("reward and eligible must have matching [B,P] shapes")
    batch_size, pool_size = reward.shape
    if target_size < 2 or target_size > pool_size:
        raise ValueError("target_size must be in [2, pool_size]")
    if len(candidate_noise_scales) != pool_size:
        raise ValueError("candidate_noise_scales must match the pool size")

    scales = torch.as_tensor(
        candidate_noise_scales,
        device=reward.device,
        dtype=reward.dtype,
    )
    selected = []
    for batch_index in range(batch_size):
        safe = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
        unsafe = torch.nonzero(~eligible[batch_index], as_tuple=False).flatten()
        if safe.numel() >= target_size:
            safe_by_reward = safe[
                torch.argsort(reward[batch_index, safe], stable=True)
            ]
            rank_positions = torch.linspace(
                0,
                safe.numel() - 1,
                target_size,
                device=reward.device,
            ).round().long()
            chosen = safe_by_reward[rank_positions]
        else:
            chosen = safe
            slots = target_size - safe.numel()
            if slots:
                # Stable two-stage sorting: reward descending inside each
                # scale, then temperature ascending across scale groups.
                unsafe_by_reward = unsafe[
                    torch.argsort(
                        reward[batch_index, unsafe],
                        descending=True,
                        stable=True,
                    )
                ]
                fill_order = unsafe_by_reward[
                    torch.argsort(scales[unsafe_by_reward], stable=True)
                ]
                chosen = torch.cat([chosen, fill_order[:slots]])
        selected.append(chosen)
    return torch.stack(selected, dim=0)


def safety_diverse_reward_indices(
    trajectories: torch.Tensor,
    reward: torch.Tensor,
    eligible: torch.Tensor,
    candidate_noise_scales: Sequence[float],
    target_size: int,
) -> torch.Tensor:
    """Safety-first farthest-point selection with reward-extreme seeds.

    Eligible reward minima and maxima seed the subset, preserving a usable
    group-relative signal.  Farthest-point sampling over the complete XY
    trajectory then retains geometric coverage.  Unsafe candidates are used
    only to fill otherwise empty slots; their eligibility remains unchanged.
    """
    if trajectories.ndim != 4 or trajectories.shape[:2] != reward.shape:
        raise ValueError("trajectories must be [B,P,T,D] matching reward")
    if eligible.shape != reward.shape:
        raise ValueError("eligible must match reward")
    batch_size, pool_size = reward.shape
    if target_size < 2 or target_size > pool_size:
        raise ValueError("target_size must be in [2, pool_size]")
    if len(candidate_noise_scales) != pool_size:
        raise ValueError("candidate_noise_scales must match the pool size")

    scales = torch.as_tensor(
        candidate_noise_scales,
        device=reward.device,
        dtype=reward.dtype,
    )
    selected_batches = []
    for batch_index in range(batch_size):
        features = trajectories[batch_index, ..., :2].reshape(pool_size, -1)
        pairwise = torch.cdist(features, features)
        safe = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
        chosen = []
        if safe.numel():
            safe_reward = reward[batch_index, safe]
            chosen.append(int(safe[safe_reward.argmin()].item()))
            maximum = int(safe[safe_reward.argmax()].item())
            if maximum != chosen[0]:
                chosen.append(maximum)
        else:
            minimum_scale = scales.min()
            low_temperature = torch.nonzero(
                scales == minimum_scale,
                as_tuple=False,
            ).flatten()
            seed = low_temperature[
                reward[batch_index, low_temperature].argmax()
            ]
            chosen.append(int(seed.item()))

        def extend_farthest(candidates: torch.Tensor, limit: int) -> None:
            remaining = set(int(index) for index in candidates.tolist()) - set(chosen)
            while remaining and len(chosen) < limit:
                remaining_tensor = torch.as_tensor(
                    sorted(remaining),
                    device=reward.device,
                    dtype=torch.long,
                )
                chosen_tensor = torch.as_tensor(
                    chosen,
                    device=reward.device,
                    dtype=torch.long,
                )
                minimum_distance = pairwise[remaining_tensor][:, chosen_tensor].min(
                    dim=1
                ).values
                next_index = int(
                    remaining_tensor[minimum_distance.argmax()].item()
                )
                chosen.append(next_index)
                remaining.remove(next_index)

        extend_farthest(safe, min(target_size, int(safe.numel())))
        if len(chosen) < target_size:
            extend_farthest(torch.arange(pool_size, device=reward.device), target_size)
        selected_batches.append(
            torch.as_tensor(chosen, device=reward.device, dtype=torch.long)
        )
    return torch.stack(selected_batches, dim=0)


def stratified_safety_diverse_indices(
    trajectories: torch.Tensor,
    reward: torch.Tensor,
    eligible: torch.Tensor,
    candidate_noise_scales: Sequence[float],
    target_size: int,
    target_noise_scales: Sequence[float] = None,
) -> torch.Tensor:
    """Select safety-diverse candidates with fixed per-temperature quotas.

    Quotas follow the proposal-pool proportions using largest-remainder
    allocation.  This prevents unconstrained farthest-point selection from
    filling the final group with high-temperature trajectories and degrading
    average quality.
    """
    pool_size = reward.shape[1]
    if target_size > pool_size:
        raise ValueError("target_size cannot exceed the proposal pool")
    scales = torch.as_tensor(
        candidate_noise_scales,
        device=reward.device,
        dtype=reward.dtype,
    )
    unique_scales, counts = torch.unique(scales, sorted=True, return_counts=True)
    if target_noise_scales is None:
        exact_quotas = counts.to(torch.float64) * float(target_size) / float(pool_size)
        quotas = torch.floor(exact_quotas).long()
        remainder = target_size - int(quotas.sum().item())
        if remainder:
            fractional = exact_quotas - quotas.to(exact_quotas.dtype)
            quota_order = torch.argsort(fractional, descending=True, stable=True)
            quotas[quota_order[:remainder]] += 1
    else:
        if len(target_noise_scales) != target_size:
            raise ValueError("target_noise_scales must contain target_size values")
        target_scales = torch.as_tensor(
            target_noise_scales,
            device=reward.device,
            dtype=reward.dtype,
        )
        target_unique, quotas = torch.unique(
            target_scales,
            sorted=True,
            return_counts=True,
        )
        if not torch.equal(target_unique, unique_scales):
            raise ValueError(
                "target_noise_scales must use every and only pool temperature"
            )
        if torch.any(quotas > counts):
            raise ValueError("target quota cannot exceed its proposal-pool count")

    per_scale_indices = []
    for scale, quota in zip(unique_scales, quotas):
        quota_value = int(quota.item())
        if quota_value == 0:
            continue
        pool_indices = torch.nonzero(scales == scale, as_tuple=False).flatten()
        if quota_value == 1:
            chosen_rows = []
            for batch_index in range(reward.shape[0]):
                safe_local = torch.nonzero(
                    eligible[batch_index, pool_indices], as_tuple=False
                ).flatten()
                candidates = safe_local if safe_local.numel() else torch.arange(
                    pool_indices.numel(), device=reward.device
                )
                best = candidates[
                    reward[batch_index, pool_indices[candidates]].argmax()
                ]
                chosen_rows.append(best.reshape(1))
            local_indices = torch.stack(chosen_rows, dim=0)
        else:
            local_indices = safety_diverse_reward_indices(
                trajectories[:, pool_indices],
                reward[:, pool_indices],
                eligible[:, pool_indices],
                (float(scale.item()),) * int(pool_indices.numel()),
                quota_value,
            )
        per_scale_indices.append(pool_indices[local_indices])
    selected = torch.cat(per_scale_indices, dim=1)
    if selected.shape[1] != target_size:
        raise RuntimeError("stratified candidate selection produced wrong size")
    return selected


def quality_diverse_indices(
    quality: torch.Tensor,
    reward: torch.Tensor,
    eligible: torch.Tensor,
    candidate_noise_scales: Sequence[float],
    target_noise_scales: Sequence[float],
) -> torch.Tensor:
    """Compress a proposal pool in objective-quality space.

    The selector is safety first.  Within every noise-scale quota it keeps the
    safe reward extremes and objective-component extrema, then fills remaining
    safe slots by farthest-point sampling in quality space.  Unsafe candidates
    are used only to keep the tensor size fixed and remain ineligible downstream.
    """
    if quality.ndim != 3 or quality.shape[:2] != reward.shape:
        raise ValueError("quality must have shape [B,P,K] matching reward")
    if eligible.shape != reward.shape:
        raise ValueError("eligible must match reward")
    pool_size = reward.shape[1]
    if len(candidate_noise_scales) != pool_size:
        raise ValueError("candidate_noise_scales must match the pool size")
    if not target_noise_scales:
        raise ValueError("target_noise_scales cannot be empty")

    scales = torch.as_tensor(
        candidate_noise_scales, device=reward.device, dtype=reward.dtype
    )
    targets = torch.as_tensor(
        target_noise_scales, device=reward.device, dtype=reward.dtype
    )
    unique_scales, quotas = torch.unique(targets, sorted=True, return_counts=True)
    if not torch.equal(unique_scales, torch.unique(scales, sorted=True)):
        raise ValueError("target scales must use every and only pool scale")

    selected_by_scale = []
    for scale, quota_tensor in zip(unique_scales, quotas):
        quota = int(quota_tensor.item())
        pool_indices = torch.nonzero(scales == scale, as_tuple=False).flatten()
        if quota > pool_indices.numel():
            raise ValueError("target quota cannot exceed its proposal-pool count")
        rows = []
        for batch_index in range(reward.shape[0]):
            local_quality = quality[batch_index, pool_indices]
            local_reward = reward[batch_index, pool_indices]
            safe = torch.nonzero(
                eligible[batch_index, pool_indices], as_tuple=False
            ).flatten()
            unsafe = torch.nonzero(
                ~eligible[batch_index, pool_indices], as_tuple=False
            ).flatten()
            chosen = []

            def append_unique(index):
                index = int(index)
                if index not in chosen and len(chosen) < quota:
                    chosen.append(index)

            if safe.numel():
                safe_reward = local_reward[safe]
                if quota == 1:
                    append_unique(safe[safe_reward.argmax()].item())
                else:
                    append_unique(safe[safe_reward.argmin()].item())
                    append_unique(safe[safe_reward.argmax()].item())
                for component in range(local_quality.shape[-1]):
                    values = local_quality[safe, component]
                    append_unique(safe[values.argmin()].item())
                    append_unique(safe[values.argmax()].item())
            else:
                append_unique(local_reward.argmax().item())

            remaining = set(int(index) for index in safe.tolist()) - set(chosen)
            while remaining and len(chosen) < quota:
                candidates = torch.as_tensor(
                    sorted(remaining), device=reward.device, dtype=torch.long
                )
                seeds = torch.as_tensor(
                    chosen, device=reward.device, dtype=torch.long
                )
                distances = torch.cdist(
                    local_quality[candidates], local_quality[seeds]
                )
                next_index = int(
                    candidates[distances.min(dim=1).values.argmax()].item()
                )
                chosen.append(next_index)
                remaining.remove(next_index)

            if len(chosen) < quota:
                fill = unsafe[
                    torch.argsort(
                        local_reward[unsafe], descending=True, stable=True
                    )
                ]
                for index in fill.tolist():
                    append_unique(index)
            if len(chosen) != quota:
                raise RuntimeError("quality selection could not fill target quota")
            rows.append(
                torch.as_tensor(chosen, device=reward.device, dtype=torch.long)
            )
        selected_by_scale.append(pool_indices[torch.stack(rows, dim=0)])
    return torch.cat(selected_by_scale, dim=1)


def weighted_quality_dispersion(
    quality: torch.Tensor,
    eligible: torch.Tensor,
    component_weights: Sequence[float],
) -> torch.Tensor:
    """Return per-scene weighted population dispersion in quality space."""
    if quality.ndim != 3 or eligible.shape != quality.shape[:2]:
        raise ValueError("quality and eligible must have shapes [B,G,K] and [B,G]")
    if len(component_weights) != quality.shape[-1]:
        raise ValueError("component_weights must match the quality dimension")
    weights = torch.as_tensor(
        component_weights, device=quality.device, dtype=quality.dtype
    )
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("component_weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("component_weights must have positive sum")
    weights = weights / weights.sum()
    mask = eligible.to(device=quality.device, dtype=torch.bool)
    counts = mask.sum(dim=1, keepdim=True)
    denominator = counts.clamp_min(1).to(quality.dtype)
    mean = (quality * mask[..., None]).sum(dim=1) / denominator
    centered = torch.where(
        mask[..., None], quality - mean[:, None, :], torch.zeros_like(quality)
    )
    variance = centered.square().sum(dim=1) / denominator
    dispersion = torch.sqrt((variance * weights).sum(dim=-1))
    return torch.where(counts.squeeze(1) >= 2, dispersion, torch.zeros_like(dispersion))


def gather_candidate_tensor(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather the candidate dimension (dimension 1) from a batched tensor."""
    if value.ndim < 2 or value.shape[0] != indices.shape[0]:
        raise ValueError("candidate tensor must have batch and candidate dimensions")
    view_shape = list(indices.shape) + [1] * (value.ndim - 2)
    expanded_shape = list(indices.shape) + list(value.shape[2:])
    gather_index = indices.reshape(view_shape).expand(expanded_shape)
    return value.gather(1, gather_index)


def gather_candidate_details(
    details: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    pool_size: int,
):
    """Gather candidate-indexed scorer details and preserve scene tensors."""
    gathered = {}
    for name, value in details.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == indices.shape[0]
            and value.shape[1] == pool_size
        ):
            gathered[name] = gather_candidate_tensor(value, indices)
        else:
            gathered[name] = value
    return gathered
