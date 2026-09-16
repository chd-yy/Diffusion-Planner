"""Fail-closed contracts for explicitly safety-filtered RL updates."""

import torch


def candidate_filter_enabled(args):
    return any(getattr(args, name, False) for name in (
        "rl_filter_safety_eligible_candidates",
        "rl_filter_progress_guard_candidates",
        "rl_filter_collision_candidates",
    ))


def validate_safety_update(args):
    if not candidate_filter_enabled(args):
        return  # Explicit legacy runs retain their historical objective.
    if getattr(args, "rl_center_reward_weights", False):
        raise ValueError("Filtered safety updates require rl_center_reward_weights=false; negative regression weights are not permitted")
    if not (getattr(args, "reward_safety_gate_threshold", 0) > 0
            or getattr(args, "reward_safety_gate_min_ttc_seconds", 0) > 0
            or getattr(args, "reward_safety_gate_require_drivable_area", False)
            or getattr(args, "rl_filter_collision_candidates", False)):
        raise ValueError("Filtered safety updates require an enabled safety gate or collision filter")


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
