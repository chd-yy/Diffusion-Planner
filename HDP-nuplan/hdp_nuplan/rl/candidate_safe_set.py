"""Safe-set qualification and source-aware candidate compression.

The chapter-4 protocol intentionally uses only trajectory-level information
that is available in the current reward cache.  Collision and active
drivable-area violations are hard exclusions.  Unsafe candidates may remain
in a fixed-shape tensor for diagnostics, but they are never selected ahead of
safe candidates and never become diffusion-regression targets.
"""

from dataclasses import dataclass
from typing import Iterable

import torch


SOURCE_REFERENCE_NATURAL = 0
SOURCE_POLICY_NATURAL = 1
SOURCE_POLICY_MEDIUM = 2
SOURCE_GUIDED = 3


@dataclass(frozen=True)
class CandidateEligibility:
    valid: torch.Tensor
    hard_safe: torch.Tensor


def classify_candidate_eligibility(
    trajectories: torch.Tensor,
    details,
) -> CandidateEligibility:
    """Build the trajectory-level feasible set from available cache fields."""
    if trajectories.ndim != 4:
        raise ValueError("trajectories must have shape [B,P,T,D]")
    expected = trajectories.shape[:2]
    valid = torch.isfinite(trajectories).flatten(2).all(dim=-1)

    def value(name, default):
        tensor = details.get(name)
        if tensor is None:
            return torch.full(
                expected,
                default,
                device=trajectories.device,
                dtype=trajectories.dtype,
            )
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape [B,P]")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
        return tensor

    no_collision = value("no_collision", 1.0) >= 1.0
    da_compliant = value("drivable_area_compliance", 1.0) >= 1.0
    da_active = value("drivable_area_gate_active", 0.0) >= 1.0
    hard_safe = valid & no_collision & (~da_active | da_compliant)
    return CandidateEligibility(valid=valid, hard_safe=hard_safe)


def _append_unique(target, candidates: Iterable[int], limit: int):
    for index in candidates:
        if len(target) >= limit:
            break
        index = int(index)
        if index not in target:
            target.append(index)


def _interleave_sources(indices, sources, source_values):
    ordered = []
    per_source = {
        source: indices[sources[indices] == source]
        for source in source_values
    }
    maximum = max((group.numel() for group in per_source.values()), default=0)
    for offset in range(maximum):
        for source in source_values:
            group = per_source[source]
            if offset < group.numel():
                ordered.append(int(group[offset]))
    return ordered


def _farthest_indices(
    quality: torch.Tensor,
    candidates: torch.Tensor,
    seeds,
    count: int,
):
    chosen = []
    remaining = set(int(index) for index in candidates.tolist()) - set(seeds)
    seed_list = list(seeds)
    while remaining and len(chosen) < count:
        ordered = torch.as_tensor(sorted(remaining), device=quality.device)
        if seed_list or chosen:
            anchors = torch.as_tensor(seed_list + chosen, device=quality.device)
            distance = torch.cdist(quality[ordered], quality[anchors])
            pick = int(ordered[distance.min(dim=1).values.argmax()].item())
        else:
            centered = quality[ordered] - quality[ordered].mean(dim=0, keepdim=True)
            pick = int(ordered[centered.square().sum(dim=-1).argmax()].item())
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def safe_set_compression_indices(
    quality: torch.Tensor,
    reward: torch.Tensor,
    eligibility: CandidateEligibility,
    source_ids: torch.Tensor,
    *,
    target_size: int = 32,
    natural_slots: int = 8,
    medium_elite_slots: int = 8,
    medium_diverse_slots: int = 8,
    global_elite_slots: int = 8,
    return_padding_mask: bool = False,
) -> torch.Tensor:
    """Compress the 88-proposal pool without creating an unsafe quota.

    Safe natural candidates preserve the reference/current-policy support.
    Two explicit medium-noise quotas then retain reward elites and quality-
    diverse samples, so final-group diversity cannot disappear merely because
    low-noise or inactive-guidance candidates rank slightly higher.  The last
    quota admits global safe elites from every source, including guidance.

    Safe candidates always backfill missing quotas before any merely valid
    candidate.  Merely valid candidates are used only to preserve a fixed
    tensor shape and remain excluded by the downstream update mask.
    """
    if quality.ndim != 3 or reward.shape != quality.shape[:2]:
        raise ValueError("quality/reward must have shapes [B,P,K] and [B,P]")
    if source_ids.ndim == 1:
        source_ids = source_ids[None].expand(reward.shape[0], -1)
    if source_ids.shape != reward.shape:
        raise ValueError("source_ids must have shape [P] or [B,P]")
    slot_counts = (
        natural_slots,
        medium_elite_slots,
        medium_diverse_slots,
        global_elite_slots,
    )
    if sum(slot_counts) != target_size:
        raise ValueError("safe-set compression slots must sum to target_size")
    if min(slot_counts) < 0:
        raise ValueError("safe-set compression slots must be non-negative")
    if reward.shape[1] < target_size:
        raise ValueError("proposal pool cannot be smaller than target_size")

    rows = []
    padding_rows = []
    natural_sources = (SOURCE_REFERENCE_NATURAL, SOURCE_POLICY_NATURAL)
    natural_source_tensor = torch.tensor(
        natural_sources, device=reward.device, dtype=source_ids.dtype
    )
    for row in range(reward.shape[0]):
        chosen = []
        valid = eligibility.valid[row]
        hard_safe = eligibility.hard_safe[row]
        sources = source_ids[row]

        safe_natural = torch.nonzero(
            hard_safe & torch.isin(sources, natural_source_tensor),
            as_tuple=False,
        ).flatten()
        _append_unique(
            chosen,
            _interleave_sources(safe_natural, sources, natural_sources),
            natural_slots,
        )

        safe_medium = torch.nonzero(
            hard_safe & (sources == SOURCE_POLICY_MEDIUM), as_tuple=False
        ).flatten()
        medium_elite_order = (
            safe_medium[
                torch.argsort(reward[row, safe_medium], descending=True, stable=True)
            ]
            if safe_medium.numel()
            else safe_medium
        )
        medium_elite_limit = natural_slots + medium_elite_slots
        _append_unique(chosen, medium_elite_order.tolist(), medium_elite_limit)

        medium_diversity = _farthest_indices(
            quality[row], safe_medium, chosen, medium_diverse_slots
        )
        medium_diverse_limit = medium_elite_limit + medium_diverse_slots
        _append_unique(chosen, medium_diversity, medium_diverse_limit)

        safe_indices = torch.nonzero(hard_safe, as_tuple=False).flatten()
        global_elite_order = (
            safe_indices[
                torch.argsort(reward[row, safe_indices], descending=True, stable=True)
            ]
            if safe_indices.numel()
            else safe_indices
        )
        _append_unique(chosen, global_elite_order.tolist(), target_size)

        # Safe candidates always have priority.  If fewer than target_size are
        # available, valid-but-unsafe entries only pad storage; update_mask
        # keeps them out of the regression objective.
        _append_unique(chosen, global_elite_order.tolist(), target_size)
        finite_indices = torch.nonzero(valid, as_tuple=False).flatten()
        _append_unique(chosen, finite_indices.tolist(), target_size)
        unique_count = len(chosen)
        if unique_count < target_size:
            # A small number of real mini-train scenes can make the policy emit
            # fewer than 32 finite proposals.  Keep the fixed tensor shape by
            # repeating a finite storage value, but explicitly mark every
            # repeated slot as padding so it can never enter safety/advantage
            # masks.  A row with zero finite proposals is represented by index
            # zero and is sanitized by the rollout caller into an anchor-only
            # replay group.
            fallback = chosen[-1] if chosen else 0
            chosen.extend([fallback] * (target_size - unique_count))
        rows.append(torch.tensor(chosen, device=reward.device, dtype=torch.long))
        padding = torch.zeros(target_size, device=reward.device, dtype=torch.bool)
        padding[unique_count:] = True
        padding_rows.append(padding)
    indices = torch.stack(rows)
    padding_mask = torch.stack(padding_rows)
    if return_padding_mask:
        return indices, padding_mask
    return indices
