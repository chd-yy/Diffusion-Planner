#!/usr/bin/env python3
"""Three-arm audit of when DA eligibility enters Repair candidate selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.model.diffusion_utils.candidate_noise import expand_scale_counts
from hdp_nuplan.rl.candidate_selection import (
    gather_candidate_details,
    gather_candidate_tensor,
    stratified_safety_diverse_indices,
)
from hdp_nuplan.rl.reward import NuPlanTensorRewardScorer
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed
from scripts.compare_checkpoint_behavior import load_checkpoint_model
from scripts.diagnose_b10_candidate_distribution import (
    balanced_indices,
    candidate_eligible_mask,
    reward_config_from_json,
    sha256,
    summarize,
)


POOL_SPEC = "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
SELECT_SPEC = "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"
ARMS = (
    "A_current_repair",
    "B_da_eligibility",
    "C_da_eligibility_aligned_reward",
    "D_quality_fixed_quota",
    "E_quality_pool_quota",
    "F25_quality_interpolated_quota",
    "F50_quality_interpolated_quota",
    "F75_quality_interpolated_quota",
    "G_f25_task_preserving_gate",
)


def quality_diverse_indices(
    quality: torch.Tensor,
    reward: torch.Tensor,
    eligible: torch.Tensor,
    candidate_noise_scales,
    target_noise_scales,
) -> torch.Tensor:
    """Select safe reward/component extremes, then quality-space farthest points."""
    if quality.ndim != 3 or quality.shape[:2] != reward.shape:
        raise ValueError("quality must be [B,P,K] matching reward")
    scales_tensor = torch.as_tensor(
        candidate_noise_scales, device=reward.device, dtype=reward.dtype
    )
    target_tensor = torch.as_tensor(
        target_noise_scales, device=reward.device, dtype=reward.dtype
    )
    unique_scales, quotas = torch.unique(target_tensor, sorted=True, return_counts=True)
    selected_by_scale = []
    for scale, quota_tensor in zip(unique_scales, quotas):
        quota = int(quota_tensor.item())
        pool_indices = torch.nonzero(scales_tensor == scale, as_tuple=False).flatten()
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
                # With one slot, keep the best safe candidate. With two or
                # more slots, retain both ends of the safe reward range.
                if quota == 1:
                    append_unique(safe[safe_reward.argmax()].item())
                else:
                    append_unique(safe[safe_reward.argmin()].item())
                    append_unique(safe[safe_reward.argmax()].item())
                # Preserve extrema of every normalized objective component.
                for component in range(local_quality.shape[-1]):
                    component_values = local_quality[safe, component]
                    append_unique(safe[component_values.argmin()].item())
                    append_unique(safe[component_values.argmax()].item())
            else:
                append_unique(local_reward.argmax().item())

            # Fill safe slots by farthest-point sampling in normalized quality space.
            remaining = set(int(index) for index in safe.tolist()) - set(chosen)
            while remaining and len(chosen) < quota:
                candidates = torch.as_tensor(
                    sorted(remaining), device=reward.device, dtype=torch.long
                )
                seeds = torch.as_tensor(chosen, device=reward.device, dtype=torch.long)
                distance = torch.cdist(local_quality[candidates], local_quality[seeds])
                next_index = int(candidates[distance.min(dim=1).values.argmax()].item())
                chosen.append(next_index)
                remaining.remove(next_index)

            # Keep fixed group size; unsafe fills remain marked ineligible.
            if len(chosen) < quota:
                fill = unsafe[torch.argsort(local_reward[unsafe], descending=True, stable=True)]
                for index in fill.tolist():
                    append_unique(index)
            if len(chosen) != quota:
                raise RuntimeError("quality selection could not fill target quota")
            rows.append(torch.as_tensor(chosen, device=reward.device, dtype=torch.long))
        selected_by_scale.append(pool_indices[torch.stack(rows, dim=0)])
    return torch.cat(selected_by_scale, dim=1)


def scales(spec):
    return expand_scale_counts([
        (float(part.split("x", 1)[0]), int(part.split("x", 1)[1]))
        for part in spec.split("+")
    ])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-args", "reward-args", "checkpoint", "data-dir", "data-list", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--balanced-scenes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    return parser.parse_args()


def paired(candidate, baseline):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return {
        **summarize(delta.tolist()),
        "wins": int(np.sum(delta > 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pool_scales, target_scales = scales(POOL_SPEC), scales(SELECT_SPEC)
    proportional_target_scales = scales("0.1x7+0.5x10+1.0x5+1.5x5+2.0x5")
    interpolated_target_scales = {
        "F25_quality_interpolated_quota": scales("0.1x10+0.5x14+1.0x3+1.5x3+2.0x2"),
        "F50_quality_interpolated_quota": scales("0.1x9+0.5x13+1.0x4+1.5x3+2.0x3"),
        "F75_quality_interpolated_quota": scales("0.1x8+0.5x12+1.0x4+1.5x4+2.0x4"),
    }
    device = torch.device("cuda")
    config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        config, args.checkpoint, device, requested_source="ema_state_dict"
    )
    base, reward_args = reward_config_from_json(args.reward_args)
    legacy = NuPlanTensorRewardScorer(replace(
        base, objective_mode="legacy", safety_gate_require_drivable_area=False
    ))
    aligned_da = NuPlanTensorRewardScorer(replace(
        base, objective_mode="nuplan_aligned", safety_gate_require_drivable_area=True
    ))
    weights = np.asarray([
        base.aligned_safety_exponent, base.aligned_progress_exponent,
        base.aligned_route_exponent, base.aligned_comfort_exponent,
        base.aligned_follow_exponent,
    ], dtype=np.float64)
    weights /= weights.sum()
    dataset = DiffusionPlannerData(
        str(args.data_dir), str(args.data_list), config.agent_num,
        config.predicted_neighbor_num, config.future_len, return_metadata=True,
    )
    indices, counts_by_log = balanced_indices(dataset, args.balanced_scenes, args.seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    storage = {arm: defaultdict(list) for arm in ARMS}
    pool_storage = defaultdict(list)
    advantage_mask_storage = defaultdict(list)
    margin_specs = {
        "margin_0": 0.0,
        "margin_0p1": 0.1,
        "margin_0p25": 0.25,
        "margin_0p5": 0.5,
    }
    margin_storage = {name: defaultdict(list) for name in margin_specs}
    per_scene = []
    pool_scale_array = np.asarray(pool_scales, dtype=np.float64)
    unique_pool_scales = sorted(set(float(value) for value in pool_scales))

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, raw, ego_future, neighbors_future, neighbor_mask, names = prepare_nuplan_batch(
                batch, device, config.observation_normalizer, with_metadata=True
            )
            set_seed(args.seed + batch_index)
            pool = model.sample(
                inputs, num_samples=len(pool_scales),
                diffusion_steps=args.diffusion_steps, candidate_noise_scales=pool_scales,
            )
            scorer_kwargs = dict(
                trajectories=pool, neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask, route_lanes=raw["route_lanes"],
                static_objects=raw["static_objects"], ego_future=ego_future,
                neighbor_agents_past=raw["neighbor_agents_past"],
                ego_current_state=raw["ego_current_state"], lanes=raw["lanes"],
            )
            _, legacy_pool = legacy(**scorer_kwargs)
            _, aligned_da_pool = aligned_da(**scorer_kwargs)
            legacy_eligible = candidate_eligible_mask(legacy_pool, reward_args)
            da_eligible = candidate_eligible_mask(aligned_da_pool, reward_args)
            pool_quality = torch.stack([
                aligned_da_pool["aligned_safety_quality"],
                aligned_da_pool["aligned_progress_quality"],
                aligned_da_pool["aligned_route_quality"],
                aligned_da_pool["aligned_comfort_quality"],
                aligned_da_pool["aligned_follow_quality"],
            ], dim=-1)
            arm_specs = {
                "A_current_repair": (legacy_eligible, legacy_pool["reward"]),
                "B_da_eligibility": (da_eligible, legacy_pool["reward"]),
                "C_da_eligibility_aligned_reward": (da_eligible, aligned_da_pool["reward"]),
            }
            arm_batches = {}
            arm_selected = {}
            arm_custom_eligibility = {}
            for arm, (eligible, selection_reward) in arm_specs.items():
                selected = stratified_safety_diverse_indices(
                    pool, selection_reward, eligible, pool_scales,
                    len(target_scales), target_noise_scales=target_scales,
                )
                if any(torch.unique(row).numel() != selected.shape[1] for row in selected):
                    raise RuntimeError(f"{arm} produced duplicate candidate indices")
                trajectories = gather_candidate_tensor(pool, selected)
                if not torch.isfinite(trajectories).all():
                    raise RuntimeError(f"{arm} produced non-finite trajectories")
                # Gather the already-computed aligned-DA details for exact same
                # pool candidates; no second scorer pass can alter values.
                evaluated = gather_candidate_details(
                    aligned_da_pool, selected, len(pool_scales)
                )
                raw_evaluated = gather_candidate_details(
                    legacy_pool, selected, len(pool_scales)
                )
                arm_batches[arm] = (trajectories, evaluated, raw_evaluated)
                arm_selected[arm] = selected
            quality_profiles = {
                "D_quality_fixed_quota": target_scales,
                "E_quality_pool_quota": proportional_target_scales,
            }
            quality_profiles.update(interpolated_target_scales)
            for arm, arm_target_scales in quality_profiles.items():
                selected = quality_diverse_indices(
                    pool_quality, aligned_da_pool["reward"], da_eligible,
                    pool_scales, arm_target_scales,
                )
                if arm.startswith("F"):
                    # Groups with no eligible candidate produce no regression
                    # target. Reuse B's fallback so quota search cannot create
                    # a meaningless unsafe-fallback difference.
                    no_eligible = ~da_eligible.any(dim=1)
                    selected[no_eligible] = arm_selected["B_da_eligibility"][no_eligible]
                if any(torch.unique(row).numel() != selected.shape[1] for row in selected):
                    raise RuntimeError(f"{arm} produced duplicate candidate indices")
                selected_trajectories = gather_candidate_tensor(pool, selected)
                if not torch.isfinite(selected_trajectories).all():
                    raise RuntimeError(f"{arm} produced non-finite trajectories")
                arm_batches[arm] = (
                    selected_trajectories,
                    gather_candidate_details(aligned_da_pool, selected, len(pool_scales)),
                    gather_candidate_details(legacy_pool, selected, len(pool_scales)),
                )
                arm_selected[arm] = selected

            # G keeps the F25 quota but makes route/progress preservation
            # non-compensatory relative to B's best eligible candidate.
            b_details = arm_batches["B_da_eligibility"][1]
            b_eligible = candidate_eligible_mask(b_details, reward_args)
            b_reward = b_details["reward"]
            b_top = torch.where(
                b_eligible, b_reward, torch.full_like(b_reward, -torch.inf)
            ).argmax(dim=1)
            batch_rows = torch.arange(b_top.shape[0], device=b_top.device)
            b_route = b_details["route_cost"][batch_rows, b_top]
            b_progress = b_details["progress_guard_reward"][batch_rows, b_top]
            task_eligible = (
                da_eligible
                & (aligned_da_pool["route_cost"] <= b_route[:, None] + 1e-4 + 1e-12)
                & (aligned_da_pool["progress_guard_reward"] >= b_progress[:, None] - 0.005 - 1e-12)
            )
            g_selected = quality_diverse_indices(
                pool_quality, aligned_da_pool["reward"], task_eligible,
                pool_scales, interpolated_target_scales["F25_quality_interpolated_quota"],
            )
            no_task_eligible = ~task_eligible.any(dim=1)
            g_selected[no_task_eligible] = arm_selected["B_da_eligibility"][no_task_eligible]
            if any(torch.unique(row).numel() != g_selected.shape[1] for row in g_selected):
                raise RuntimeError("G task-preserving selector produced duplicate indices")
            g_trajectories = gather_candidate_tensor(pool, g_selected)
            if not torch.isfinite(g_trajectories).all():
                raise RuntimeError("G task-preserving selector produced non-finite trajectories")
            arm_batches["G_f25_task_preserving_gate"] = (
                g_trajectories,
                gather_candidate_details(aligned_da_pool, g_selected, len(pool_scales)),
                gather_candidate_details(legacy_pool, g_selected, len(pool_scales)),
            )
            arm_selected["G_f25_task_preserving_gate"] = g_selected
            arm_custom_eligibility["G_f25_task_preserving_gate"] = gather_candidate_tensor(
                task_eligible, g_selected
            )

            for row, name in enumerate(names):
                pool_row_eligible = da_eligible[row]
                pool_count = int(pool_row_eligible.sum())
                if pool_count >= 2:
                    pool_q = torch.stack([
                        aligned_da_pool["aligned_safety_quality"][row],
                        aligned_da_pool["aligned_progress_quality"][row],
                        aligned_da_pool["aligned_route_quality"][row],
                        aligned_da_pool["aligned_comfort_quality"][row],
                        aligned_da_pool["aligned_follow_quality"][row],
                    ], dim=-1)[pool_row_eligible].detach().float().cpu().numpy()
                    pool_dispersion = float(np.sqrt(np.sum(weights * pool_q.var(axis=0))))
                else:
                    pool_dispersion = 0.0
                pool_metrics = {
                    "eligible_count": float(pool_count),
                    "has_eligible": float(pool_count >= 1),
                    "has_two": float(pool_count >= 2),
                    "component_dispersion": pool_dispersion,
                    "active_at_0p01": float(pool_count >= 2 and pool_dispersion >= 0.01),
                    "active_at_0p005": float(pool_count >= 2 and pool_dispersion >= 0.005),
                }
                for scale in unique_pool_scales:
                    positions = torch.as_tensor(
                        np.isclose(pool_scale_array, scale), device=pool_row_eligible.device
                    )
                    scale_count = int(pool_row_eligible[positions].sum())
                    scale_name = str(scale).replace(".", "p")
                    pool_metrics[f"eligible_count_scale_{scale_name}"] = float(scale_count)
                    pool_metrics[f"has_two_scale_{scale_name}"] = float(scale_count >= 2)
                for key, value in pool_metrics.items():
                    pool_storage[key].append(value)
                record = {"scene": str(name), "pool": pool_metrics, "arms": {}}
                for arm in ARMS:
                    trajectories, details, raw_details = arm_batches[arm]
                    eligible = (
                        arm_custom_eligibility[arm][row]
                        if arm in arm_custom_eligibility
                        else candidate_eligible_mask(details, reward_args)[row]
                    )
                    count = int(eligible.sum())
                    reward = details["reward"][row]
                    top_index = int(torch.where(
                        eligible, reward, torch.full_like(reward, -torch.inf)
                    ).argmax()) if count else int(reward.argmax())
                    if count >= 2:
                        q = torch.stack([
                            details["aligned_safety_quality"][row],
                            details["aligned_progress_quality"][row],
                            details["aligned_route_quality"][row],
                            details["aligned_comfort_quality"][row],
                            details["aligned_follow_quality"][row],
                        ], dim=-1)[eligible].detach().float().cpu().numpy()
                        dispersion = float(np.sqrt(np.sum(weights * q.var(axis=0))))
                    else:
                        dispersion = 0.0
                    horizon = min(trajectories.shape[-2], ego_future.shape[-2])
                    errors = torch.linalg.vector_norm(
                        trajectories[row, :, :horizon, :2] - ego_future[row, None, :horizon, :2], dim=-1
                    )
                    metrics = {
                        "eligible_count": float(count),
                        "has_eligible": float(count >= 1),
                        "has_two": float(count >= 2),
                        "component_dispersion": dispersion,
                        "active_at_0p01": float(count >= 2 and dispersion >= 0.01),
                        "active_at_0p005": float(count >= 2 and dispersion >= 0.005),
                        "raw_da_top": float(raw_details["drivable_area_compliance"][row, top_index]),
                        "no_collision_top": float(details["no_collision"][row, top_index]),
                        "risk_top": float(details["risk_reward"][row, top_index]),
                        "ttc_top": float(details["min_ttc_seconds"][row, top_index].clamp(max=8.0)),
                        "progress_guard_top": float(details["progress_guard_reward"][row, top_index]),
                        "route_cost_top": float(details["route_cost"][row, top_index]),
                        "comfort_cost_top": float(details["comfort_cost"][row, top_index]),
                        "mean_candidate_ade_m": float(errors.mean()),
                        "oracle_ade_m": float(errors.mean(dim=-1).min()),
                    }
                    for key, value in metrics.items():
                        storage[arm][key].append(value)
                    record["arms"][arm] = metrics

                # H audit: keep F25 safety-group statistics intact, then mask
                # only positive update weights that violate route/progress
                # preservation relative to B's top eligible candidate.
                f_details = arm_batches["F25_quality_interpolated_quota"][1]
                f_eligible = candidate_eligible_mask(f_details, reward_args)[row]
                f_rewards = f_details["reward"][row]
                if int(f_eligible.sum()) >= 2:
                    f_reward_values = f_rewards[f_eligible]
                    f_mean = f_reward_values.mean()
                    f_std = f_reward_values.std(unbiased=False)
                    positive_advantage = torch.relu(
                        (f_rewards - f_mean) / (f_std + 1e-6)
                    ) * f_eligible.to(f_rewards.dtype)
                else:
                    positive_advantage = torch.zeros_like(f_rewards)
                active_group = bool(record["arms"]["F25_quality_interpolated_quota"]["active_at_0p01"])
                if not active_group:
                    positive_advantage.zero_()
                task_allowed = (
                    f_eligible
                    & (f_details["route_cost"][row] <= b_route[row] + 1e-4 + 1e-12)
                    & (f_details["progress_guard_reward"][row] >= b_progress[row] - 0.005 - 1e-12)
                )
                route_allowed = f_eligible & (
                    f_details["route_cost"][row] <= b_route[row] + 1e-4 + 1e-12
                )
                progress_allowed = f_eligible & (
                    f_details["progress_guard_reward"][row] >= b_progress[row] - 0.005 - 1e-12
                )
                retained = positive_advantage * task_allowed.to(f_rewards.dtype)
                raw_mass = float(positive_advantage.sum())
                retained_mass = float(retained.sum())
                h_metrics = {
                    "active_group": float(active_group),
                    "positive_count": float((positive_advantage > 0).sum()),
                    "task_positive_count": float((retained > 0).sum()),
                    "active_group_retained": float(active_group and bool((retained > 0).any())),
                    "raw_positive_mass": raw_mass,
                    "retained_positive_mass": retained_mass,
                    "retained_mass_fraction": retained_mass / raw_mass if raw_mass > 0 else 0.0,
                    "route_violation_mass": float(
                        (positive_advantage * (~route_allowed).to(f_rewards.dtype)).sum()
                    ),
                    "progress_violation_mass": float(
                        (positive_advantage * (~progress_allowed).to(f_rewards.dtype)).sum()
                    ),
                }
                for key, value in h_metrics.items():
                    advantage_mask_storage[key].append(value)
                record["advantage_update_mask"] = h_metrics
                record["constraint_positive_target_sweep"] = {}
                for margin_name, margin in margin_specs.items():
                    if margin == 0.0:
                        target_mask = task_allowed & (positive_advantage > 0)
                    else:
                        target_mask = task_allowed & (positive_advantage >= margin)
                    target_count = int(target_mask.sum()) if active_group else 0
                    sweep_metrics = {
                        "active_group": float(active_group),
                        "retained_group": float(active_group and target_count >= 1),
                        "target_count": float(target_count),
                        # Uniform weights inside the retained target set.
                        "uniform_weight_max_to_mean": 1.0 if target_count else 0.0,
                    }
                    for key, value in sweep_metrics.items():
                        margin_storage[margin_name][key].append(value)
                    record["constraint_positive_target_sweep"][margin_name] = sweep_metrics
                per_scene.append(record)
            if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"batch={batch_index + 1}/{len(loader)}", flush=True
                )

    summaries = {
        arm: {key: summarize(values) for key, values in metrics.items()}
        for arm, metrics in storage.items()
    }
    pool_summary = {key: summarize(values) for key, values in pool_storage.items()}
    advantage_mask_summary = {
        key: summarize(values) for key, values in advantage_mask_storage.items()
    }
    active_count = int(sum(advantage_mask_storage["active_group"]))
    retained_group_count = int(sum(advantage_mask_storage["active_group_retained"]))
    raw_mass_total = float(sum(advantage_mask_storage["raw_positive_mass"]))
    retained_mass_total = float(sum(advantage_mask_storage["retained_positive_mass"]))
    route_violation_mass_total = float(sum(advantage_mask_storage["route_violation_mass"]))
    progress_violation_mass_total = float(sum(advantage_mask_storage["progress_violation_mass"]))
    advantage_mask_summary["aggregate"] = {
        "active_group_count": active_count,
        "retained_active_group_count": retained_group_count,
        "active_group_retention_rate": retained_group_count / max(active_count, 1),
        "raw_positive_mass": raw_mass_total,
        "retained_positive_mass": retained_mass_total,
        "positive_mass_retention_rate": retained_mass_total / max(raw_mass_total, 1e-12),
        "route_violation_mass_fraction": route_violation_mass_total / max(raw_mass_total, 1e-12),
        "progress_violation_mass_fraction": progress_violation_mass_total / max(raw_mass_total, 1e-12),
        "all_retained_weights_task_compliant": True,
    }
    advantage_mask_summary["pre_registered_gate"] = {
        "active_group_retention_ge_0p90": (
            retained_group_count / max(active_count, 1) >= 0.90
        ),
        "positive_mass_retention_ge_0p50": (
            retained_mass_total / max(raw_mass_total, 1e-12) >= 0.50
        ),
        "all_retained_weights_task_compliant": True,
    }
    advantage_mask_summary["pre_registered_gate"]["passed"] = all(
        advantage_mask_summary["pre_registered_gate"].values()
    )
    margin_sweep_summary = {}
    for margin_name, metrics in margin_storage.items():
        active = int(sum(metrics["active_group"]))
        retained = int(sum(metrics["retained_group"]))
        retained_counts = [
            count for count, keep in zip(metrics["target_count"], metrics["retained_group"])
            if keep
        ]
        count_summary = summarize(retained_counts) if retained_counts else summarize([0.0])
        retention_rate = retained / max(active, 1)
        passed = retention_rate >= 0.90 and count_summary["mean"] >= 2.0
        margin_sweep_summary[margin_name] = {
            "margin_std": margin_specs[margin_name],
            "active_group_count": active,
            "retained_group_count": retained,
            "active_group_retention_rate": retention_rate,
            "total_target_count": int(sum(metrics["target_count"])),
            "target_count_among_retained": count_summary,
            "single_target_group_fraction": (
                sum(count == 1 for count in retained_counts) / max(retained, 1)
            ),
            "uniform_weight_max_to_mean": 1.0,
            "passed": passed,
        }
    passing_margins = [
        value for value in margin_sweep_summary.values() if value["passed"]
    ]
    margin_sweep_summary["selection"] = {
        "rule": "largest margin with >=90% active-group retention and mean >=2 targets",
        "selected_margin_std": (
            max(value["margin_std"] for value in passing_margins)
            if passing_margins else None
        ),
    }
    comparisons = {}
    gates = {}
    for arm in ARMS[1:]:
        baseline = (
            "A_current_repair"
            if arm in ("B_da_eligibility", "C_da_eligibility_aligned_reward")
            else "B_da_eligibility"
        )
        delta = {
            key: paired(storage[arm][key], storage[baseline][key])
            for key in storage[baseline]
        }
        comparisons[f"{arm}_minus_{baseline}"] = delta
        eligible_tolerance = -0.02 if arm == "G_f25_task_preserving_gate" else -0.01
        if arm == "G_f25_task_preserving_gate":
            f25 = summaries["F25_quality_interpolated_quota"]
            has_two_check_name = "has_two_drop_vs_f25_ge_minus_0p03"
            has_two_check = summaries[arm]["has_two"]["mean"] - f25["has_two"]["mean"] >= -0.03 - 1e-12
        else:
            has_two_check_name = (
                "has_two_gain_ge_0p03"
                if baseline == "A_current_repair"
                else "has_two_non_decreasing"
            )
            has_two_check = (
                delta["has_two"]["mean"] >= 0.03 - 1e-12
                if baseline == "A_current_repair"
                else delta["has_two"]["mean"] >= -1e-12
            )
        checks = {
            "has_eligible_within_tolerance": delta["has_eligible"]["mean"] >= eligible_tolerance - 1e-12,
            (
                has_two_check_name
            ): has_two_check,
            "active_at_0p01_ge_0p40": summaries[arm]["active_at_0p01"]["mean"] >= 0.40,
            "raw_da_top_non_decreasing": delta["raw_da_top"]["mean"] >= 0.0,
            "no_collision_top_non_decreasing": delta["no_collision_top"]["mean"] >= 0.0,
            "risk_top_non_decreasing": delta["risk_top"]["mean"] >= 0.0,
            "ttc_top_delta_ge_minus_0p01": delta["ttc_top"]["mean"] >= -0.01,
            "progress_guard_delta_ge_minus_0p005": delta["progress_guard_top"]["mean"] >= -0.005,
            "route_cost_delta_le_1e-4": delta["route_cost_top"]["mean"] <= 1e-4,
            "comfort_cost_delta_le_1e-4": delta["comfort_cost_top"]["mean"] <= 1e-4,
            "mean_ade_relative_increase_le_1pct": (
                summaries[arm]["mean_candidate_ade_m"]["mean"]
                <= summaries[baseline]["mean_candidate_ade_m"]["mean"] * 1.01
            ),
        }
        gates[arm] = {"passed": all(checks.values()), "checks": checks}
    result = {
        "schema_version": 3,
        "experiment": "Chapter 4 DA-aware candidate selection timing ablation",
        "created_at": datetime.now().isoformat(),
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "selected_scenes": len(indices), "counts_by_log": counts_by_log,
            "seed": args.seed, "pool_profile": POOL_SPEC,
            "selection_profile": SELECT_SPEC, "arms": list(ARMS),
            "proportional_quality_profile": "0.1x7+0.5x10+1.0x5+1.5x5+2.0x5",
            "interpolated_quality_profiles": {
                "F25": "0.1x10+0.5x14+1.0x3+1.5x3+2.0x2",
                "F50": "0.1x9+0.5x13+1.0x4+1.5x3+2.0x3",
                "F75": "0.1x8+0.5x12+1.0x4+1.5x4+2.0x4",
            },
            "task_preserving_gate": {
                "base_profile": "F25",
                "route_cost_vs_b_top_max_increase": 1e-4,
                "progress_guard_vs_b_top_max_drop": 0.005,
                "fallback": "reuse B indices when no task-eligible pool candidate",
            },
            "same_pool_across_arms": True,
        },
        "identity": {
            "checkpoint": checkpoint_metadata,
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_args_sha256": sha256(args.model_args),
            "reward_args_sha256": sha256(args.reward_args),
            "data_list_sha256": sha256(args.data_list),
        },
        "pool_summary": pool_summary,
        "advantage_update_mask_audit": advantage_mask_summary,
        "constraint_positive_target_sweep": margin_sweep_summary,
        "summary_by_arm": summaries,
        "paired_vs_current_repair": comparisons,
        "pre_registered_gates": gates,
        "per_scene": per_scene,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "gates": gates,
        "key_summary": {
            "pool": {
                "has_two": pool_summary["has_two"]["mean"],
                "active_at_0p01": pool_summary["active_at_0p01"]["mean"],
            },
            **{arm: {
                "has_two": summaries[arm]["has_two"]["mean"],
                "active_at_0p01": summaries[arm]["active_at_0p01"]["mean"],
                "raw_da_top": summaries[arm]["raw_da_top"]["mean"],
            } for arm in ARMS},
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
