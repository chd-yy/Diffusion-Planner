"""Fail-closed contracts for explicitly safety-filtered RL updates."""

import torch


def candidate_filter_enabled(args):
    return getattr(args, "rl_guided_safe_set_protocol", False) or any(
        getattr(args, name, False) for name in (
        "rl_filter_safety_eligible_candidates",
        "rl_filter_progress_guard_candidates",
        "rl_filter_collision_candidates",
        )
    )


def validate_safety_update(args):
    if not candidate_filter_enabled(args):
        return  # Explicit legacy runs retain their historical objective.
    if getattr(args, "rl_center_reward_weights", False):
        raise ValueError("Filtered safety updates require rl_center_reward_weights=false; negative regression weights are not permitted")
    if getattr(args, "rl_guided_safe_set_protocol", False):
        if getattr(args, "rl_weighting_mode", "") != "constraint_uniform_positive":
            raise ValueError(
                "guided safe-set protocol requires constraint_uniform_positive"
            )
        if not getattr(args, "rl_component_dispersion_gate", False):
            raise ValueError("guided safe-set protocol requires component dispersion gate")
        if getattr(args, "reward_objective_mode", "") != "nuplan_aligned":
            raise ValueError("guided safe-set protocol requires nuplan_aligned reward")
        if not getattr(args, "reward_safety_gate_require_drivable_area", False):
            raise ValueError("guided safe-set protocol requires a drivable-area hard gate")
    if not (getattr(args, "reward_safety_gate_threshold", 0) > 0
            or getattr(args, "reward_safety_gate_min_ttc_seconds", 0) > 0
            or getattr(args, "reward_safety_gate_require_drivable_area", False)
            or getattr(args, "rl_filter_collision_candidates", False)):
        raise ValueError("Filtered safety updates require an enabled safety gate or collision filter")
    if getattr(args, "rl_weighting_mode", "") == "constraint_uniform_positive":
        if not (
            getattr(args, "rl_task_preserving_update_mask", False)
            or getattr(args, "rl_guided_safe_set_protocol", False)
        ):
            raise ValueError(
                "constraint_uniform_positive requires a task-preserving update mask"
            )
        if not getattr(args, "rl_component_dispersion_gate", False):
            raise ValueError(
                "constraint_uniform_positive requires the component-dispersion gate"
            )


def build_candidate_mask(details, args):
    """Compose candidate qualifications; each enabled filter must reach loss."""
    qualification = details["safety_gate_eligible"]
    if not torch.isfinite(qualification).all():
        raise ValueError("Non-finite safety qualification")
    mask = qualification >= 1
    if getattr(args, "rl_filter_collision_candidates", False):
        # Independent collision exclusion: risk shaping can give nonzero rewards
        # to collisions (e.g. rear-end weighting). This is a proxy, not official
        # NuPlan at-fault classification or a closed-loop guarantee.
        collision = details["no_collision"]
        if not torch.isfinite(collision).all():
            raise ValueError("Non-finite collision qualification")
        mask = mask & (collision >= 1.0)
    if getattr(args, "rl_filter_progress_guard_candidates", False):
        progress = details["progress_guard_reward"]
        if not torch.isfinite(progress).all():
            raise ValueError("Non-finite progress qualification")
        mask = mask & (progress >= args.rl_min_progress_guard_reward)
    return mask


def validate_replay_target_coverage(
    replay_buffer,
    args,
    *,
    return_target_flags=False,
    audit_batch_size=1024,
):
    """Fail before optimizer steps when rollout target support is insufficient."""
    minimum_active = float(getattr(args, "rl_min_active_group_fraction", 0.0))
    minimum_retention = float(
        getattr(args, "rl_min_target_group_retention", 0.0)
    )
    minimum_targets = float(
        getattr(args, "rl_min_targets_per_retained_group", 0.0)
    )
    minimum_target_groups = int(
        getattr(args, "rl_min_target_group_count", 0)
    )
    for name, value in (
        ("rl_min_active_group_fraction", minimum_active),
        ("rl_min_target_group_retention", minimum_retention),
    ):
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if minimum_targets < 0:
        raise ValueError("rl_min_targets_per_retained_group must be non-negative")
    if minimum_target_groups < 0:
        raise ValueError("rl_min_target_group_count must be non-negative")
    if not any((minimum_active, minimum_retention, minimum_targets, minimum_target_groups)) and not return_target_flags:
        return {}

    if len(replay_buffer) == 0:
        raise RuntimeError("Replay target coverage gate requires a non-empty buffer")
    active_count = 0
    retained_count = 0
    target_candidate_count = 0
    audited_count = 0
    target_flags = [] if return_target_flags else None
    for items in replay_buffer.iter_item_batches(audit_batch_size):
        target_state = replay_target_state(items, args)
        active = target_state["active"]
        retained = target_state["target_groups"]
        targets = target_state["targets"]
        active_count += int(active.sum())
        retained_count += int(retained.sum())
        target_candidate_count += int(targets[retained].sum())
        audited_count += len(items)
        if return_target_flags:
            target_flags.extend(bool(value) for value in retained.tolist())
    if audited_count != len(replay_buffer):
        raise RuntimeError(
            "Replay target audit count does not match replay length: "
            f"{audited_count} != {len(replay_buffer)}"
        )
    active_fraction = active_count / audited_count
    retention = retained_count / max(active_count, 1)
    targets_per_retained = (
        target_candidate_count / retained_count if retained_count else 0.0
    )
    metrics = {
        "replay_preflight_active_group_fraction": active_fraction,
        "replay_preflight_target_group_retention": retention,
        "replay_preflight_targets_per_retained_group": targets_per_retained,
        "replay_preflight_active_group_count": float(active_count),
        "replay_preflight_retained_group_count": float(retained_count),
    }
    failures = []
    if active_fraction < minimum_active:
        failures.append(
            f"active_group_fraction={active_fraction:.6f} < {minimum_active:.6f}"
        )
    if retention < minimum_retention:
        failures.append(
            f"target_group_retention={retention:.6f} < {minimum_retention:.6f}"
        )
    if targets_per_retained < minimum_targets:
        failures.append(
            f"targets_per_retained_group={targets_per_retained:.6f} < {minimum_targets:.6f}"
        )
    if retained_count < minimum_target_groups:
        failures.append(
            f"target_group_count={retained_count} < {minimum_target_groups}"
        )
    if failures:
        raise RuntimeError("Replay target coverage gate failed: " + "; ".join(failures))
    if return_target_flags:
        return metrics, target_flags
    return metrics


def replay_target_state(items, args):
    """Reconstruct the exact group-level target contract used by the loss."""
    items = tuple(items)
    if not items:
        raise RuntimeError("Replay target state requires at least one item")
    for item in items:
        if item.candidate_mask is None or item.update_mask is None or item.active_group is None:
            raise RuntimeError(
                "Replay target state requires candidate_mask, update_mask and active_group"
            )
    rewards = torch.stack([item.rewards for item in items])
    candidate_mask = torch.stack([item.candidate_mask for item in items])
    update_mask = torch.stack([item.update_mask for item in items])
    external_active = torch.stack([item.active_group for item in items]).bool()
    counts = candidate_mask.sum(dim=1)
    denominator = counts.clamp_min(1).to(rewards.dtype)
    mean = torch.where(candidate_mask, rewards, torch.zeros_like(rewards)).sum(dim=1)
    mean = mean / denominator
    centered = torch.where(
        candidate_mask, rewards - mean[:, None], torch.zeros_like(rewards)
    )
    std = torch.sqrt(centered.square().sum(dim=1) / denominator)
    active = (
        external_active
        & (counts >= 2)
        & (std >= float(getattr(args, "rl_min_reward_std", 0.0)))
    )
    if getattr(args, "rl_relative_to_reference", False):
        if any(item.reference_reward is None for item in items):
            raise RuntimeError("Reference-relative preflight requires reference rewards")
        reference = torch.stack([item.reference_reward for item in items])
        baseline = reference[:, None]
    else:
        baseline = mean[:, None]
    advantage = (rewards - baseline) / (std[:, None] + 1e-6)
    margin = float(getattr(args, "rl_positive_advantage_margin", 0.0))
    targets = (
        update_mask
        & candidate_mask
        & active[:, None]
        & (advantage >= margin)
    )
    return {
        "rewards": rewards,
        "candidate_mask": candidate_mask,
        "update_mask": update_mask,
        "active": active,
        "targets": targets,
        "target_groups": active & targets.any(dim=1),
    }


def apply_update(total_loss, model, optimizer, ema, args, metrics):
    """Do not let AdamW momentum/decay move a policy with no learning target.

    In DDP all ranks make the same decision. If any rank has a target, every
    rank participates in backward, including ranks whose local loss is zero.
    """
    has_target = (
        (args.rl_rollout_loss_weight > 0 and bool(metrics["has_regression_targets"].item()))
        or args.rl_expert_anchor_weight > 0
        or getattr(args, "rl_reference_anchor_weight", 0) > 0
    )
    status = torch.tensor([int(has_target), int(not torch.isfinite(total_loss).item())],
                          device=total_loss.device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    if status[1].item():
        raise FloatingPointError("Non-finite RL loss; optimizer and EMA were not updated")
    if not status[0].item():
        return False
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], args.rl_grad_clip,
        error_if_nonfinite=True,
    )
    optimizer.step()
    ema.update(model)
    return True
