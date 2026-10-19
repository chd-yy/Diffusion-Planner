#!/usr/bin/env python3
"""Minimal paired pre-training gate for the guided safe-set protocol.

The script never updates model weights.  It compares B10 natural sampling
(32 proposals at noise 0.1) against the exact 88-to-32 guided safe-set
protocol on the same deterministic, motion-stratified validation scenes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.rl.candidate_safe_set import (
    classify_candidate_eligibility,
    safe_set_compression_indices,
)
from hdp_nuplan.rl.candidate_selection import (
    gather_candidate_details,
    gather_candidate_tensor,
    weighted_quality_dispersion,
)
from hdp_nuplan.rl.reward import NuPlanTensorRewardScorer
from hdp_nuplan.rl.train_epoch_rl import (
    _sample_guided_safe_set_pool,
    prepare_nuplan_batch,
)
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed
from scripts.compare_checkpoint_behavior import load_checkpoint_model
from scripts.diagnose_b10_candidate_distribution import (
    reward_config_from_json,
    sha256,
    summarize,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-args", required=True, type=Path)
    parser.add_argument("--reward-args", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--scenes", default=40, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--diffusion-steps", default=6, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _path_length(ego_future):
    xy = np.asarray(ego_future, dtype=np.float64)[..., :2]
    origin = np.zeros((1, 2), dtype=np.float64)
    return float(np.linalg.norm(np.diff(np.concatenate([origin, xy]), axis=0), axis=1).sum())


def _round_robin_by_log(records, quota, rng):
    by_log = defaultdict(list)
    for record in records:
        by_log[record["log_name"]].append(record)
    for values in by_log.values():
        rng.shuffle(values)
    logs = sorted(by_log)
    selected = []
    while len(selected) < quota:
        progressed = False
        for log_name in logs:
            if by_log[log_name] and len(selected) < quota:
                selected.append(by_log[log_name].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != quota:
        raise RuntimeError(f"stratum has only {len(selected)} of {quota} required scenes")
    return selected


def motion_stratified_indices(dataset, total, seed):
    if total < 10 or total > len(dataset):
        raise ValueError("minimal gate requires 10 <= scenes <= dataset size")
    records = []
    for index, name in enumerate(dataset.data_list):
        with np.load(Path(dataset.data_dir) / name, allow_pickle=False) as item:
            records.append({
                "index": index,
                "name": name,
                "log_name": str(item["log_name"].item()),
                "scenario_type": str(item["scenario_type"].item()),
                "expert_path_length_m": _path_length(item["ego_agent_future"]),
            })
    lengths = np.asarray([item["expert_path_length_m"] for item in records])
    q25, q75 = np.quantile(lengths, [0.25, 0.75])
    strata = {
        "high_motion": [item for item in records if item["expert_path_length_m"] >= q75],
        "medium_motion": [item for item in records if q25 < item["expert_path_length_m"] < q75],
        "low_motion": [item for item in records if item["expert_path_length_m"] <= q25],
    }
    if total == len(dataset):
        # The formal validation-1000 protocol evaluates every entry in the
        # already-balanced list.  Do not resample it into the 50/30/20 pilot
        # mixture; doing so would change the scene population relative to the
        # historical Repair experiment that this script is meant to replace.
        selected = [
            {
                **item,
                "motion_stratum": (
                    "high_motion"
                    if item["expert_path_length_m"] >= q75
                    else (
                        "low_motion"
                        if item["expert_path_length_m"] <= q25
                        else "medium_motion"
                    )
                ),
            }
            for item in records
        ]
        metadata = {
            "selection_mode": "all_scenes_from_balanced_validation_list",
            "q25_expert_path_length_m": float(q25),
            "q75_expert_path_length_m": float(q75),
            "selected_by_stratum": dict(
                Counter(item["motion_stratum"] for item in selected)
            ),
        }
        return [item["index"] for item in selected], metadata
    high = total // 2
    low = total // 5
    quotas = {"high_motion": high, "medium_motion": total - high - low, "low_motion": low}
    rng = np.random.default_rng(seed)
    selected = []
    for name in ("high_motion", "medium_motion", "low_motion"):
        for item in _round_robin_by_log(strata[name], quotas[name], rng):
            selected.append({**item, "motion_stratum": name})
    selected.sort(key=lambda item: item["index"])
    metadata = {
        "q25_expert_path_length_m": float(q25),
        "q75_expert_path_length_m": float(q75),
        "counts_by_stratum": dict(Counter(item["motion_stratum"] for item in selected)),
        "counts_by_log": dict(Counter(item["log_name"] for item in selected)),
        "counts_by_scenario_type": dict(Counter(item["scenario_type"] for item in selected)),
        "selected": selected,
    }
    return [item["index"] for item in selected], metadata


def _quality(details):
    return torch.stack(
        [
            details["aligned_safety_quality"],
            details["aligned_progress_quality"],
            details["aligned_route_quality"],
            details["aligned_comfort_quality"],
            details["aligned_follow_quality"],
        ],
        dim=-1,
    )


def _pairwise_mean(xy):
    count = xy.shape[0]
    if count < 2:
        return None, None
    # Keep the formal validation-1000 metric bitwise aligned with the
    # historical Repair experiment: average the ordered off-diagonal cdist
    # matrix rather than using the numerically different pdist CUDA kernel.
    endpoint_distance = torch.cdist(xy[:, -1, :], xy[:, -1, :])
    endpoint = endpoint_distance.sum() / (count * (count - 1))
    delta = xy[:, None] - xy[None, :]
    distance = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)
    upper = torch.triu_indices(count, count, offset=1, device=xy.device)
    trajectory = distance[upper[0], upper[1]].mean()
    return float(endpoint), float(trajectory)


def scene_metrics(trajectories, details, eligibility, ego_future):
    batch_metrics = []
    for row in range(trajectories.shape[0]):
        valid = eligibility.valid[row]
        safe = eligibility.hard_safe[row]
        reward = details["reward"][row]
        xy = trajectories[row, :, :, :2]
        all_endpoint, all_trajectory = _pairwise_mean(xy[valid])
        safe_endpoint, safe_trajectory = _pairwise_mean(xy[safe])
        safe_rewards = reward[safe]
        horizon = min(xy.shape[1], ego_future.shape[1])
        ade = torch.linalg.vector_norm(
            xy[:, :horizon] - ego_future[row, None, :horizon, :2], dim=-1
        ).mean(dim=-1)
        safe_ade = ade[safe]
        origin = torch.zeros_like(ego_future[row, :1, :2])
        expert_path_length = torch.linalg.vector_norm(
            torch.diff(
                torch.cat([origin, ego_future[row, :horizon, :2]], dim=0),
                dim=0,
            ),
            dim=-1,
        ).sum()
        if safe.any():
            safe_indices = torch.nonzero(safe, as_tuple=False).flatten()
            best_safe_index = safe_indices[reward[safe_indices].argmax()]
            best_safe_reward_ade = float(ade[best_safe_index])
        else:
            # Match the historical validation-1000 Repair table: when a scene
            # has no eligible candidate, fall back to candidate zero so the
            # aggregate still covers the same complete scene population.
            best_safe_reward_ade = float(ade[0])
        batch_metrics.append({
            "finite_candidate_fraction": float(valid.float().mean()),
            "safe_candidate_fraction": float(safe.float().mean()),
            "safe_candidate_count": int(safe.sum()),
            "has_two_safe_candidates": float(safe.sum() >= 2),
            "no_collision_candidate_fraction": float((details["no_collision"][row] >= 1).float().mean()),
            "drivable_candidate_fraction": float((details["drivable_area_compliance"][row] >= 1).float().mean()),
            "endpoint_pairwise_distance_m": all_endpoint,
            "trajectory_pairwise_distance_m": all_trajectory,
            "safe_endpoint_pairwise_distance_m": safe_endpoint,
            "safe_trajectory_pairwise_distance_m": safe_trajectory,
            "safe_reward_std": float(safe_rewards.std(unbiased=False)) if safe_rewards.numel() else 0.0,
            "safe_reward_range": float(safe_rewards.max() - safe_rewards.min()) if safe_rewards.numel() else 0.0,
            "active_safe_reward_std_0_01": float(safe_rewards.numel() >= 2 and safe_rewards.std(unbiased=False) >= 0.01),
            "expert_path_length_m": float(expert_path_length),
            "mean_candidate_ade_m": float(ade.mean()),
            "oracle_best_of_group_ade_m": float(ade.min()),
            "best_safe_reward_candidate_ade_m": best_safe_reward_ade,
            "single_candidate_ade_m": float(ade[0]),
            "mean_safe_candidate_ade_m": float(safe_ade.mean()) if safe_ade.numel() else None,
            "oracle_safe_ade_m": float(safe_ade.min()) if safe_ade.numel() else None,
        })
    return batch_metrics


def _summary(per_scene):
    keys = sorted(
        key
        for key in {key for item in per_scene for key in item}
        if key != "scene"
    )
    return {
        key: summarize([item[key] for item in per_scene if item.get(key) is not None])
        for key in keys
    }


def _paired_bootstrap(baseline, proposed, key, seed, draws=5000):
    pairs = [
        (left[key], right[key])
        for left, right in zip(baseline, proposed)
        if left.get(key) is not None and right.get(key) is not None
    ]
    if not pairs:
        return {"count": 0, "mean_delta": None, "ci95": [None, None]}
    delta = np.asarray([right - left for left, right in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    return {
        "count": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-12))


def _moving_has_two_safe_fraction(rows, threshold_m=2.0):
    moving = [
        item for item in rows
        if item["expert_path_length_m"] > float(threshold_m)
    ]
    if not moving:
        return None
    return float(np.mean([item["has_two_safe_candidates"] for item in moving]))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("minimal guided safe-set validation requires CUDA")
    if args.batch_size < 1 or args.diffusion_steps < 1:
        raise ValueError("batch size and diffusion steps must be positive")

    device = torch.device("cuda")
    model_config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        model_config, args.checkpoint, device, requested_source="ema_state_dict"
    )
    reward_config, reward_args = reward_config_from_json(args.reward_args)
    reward_config = replace(
        reward_config,
        objective_mode="nuplan_aligned",
        safety_gate_require_drivable_area=True,
        safety_gate_threshold=0.0,
        safety_gate_min_ttc_seconds=0.0,
    )
    scorer = NuPlanTensorRewardScorer(reward_config)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        model_config.agent_num,
        model_config.predicted_neighbor_num,
        model_config.future_len,
        return_metadata=True,
    )
    indices, selection = motion_stratified_indices(dataset, args.scenes, args.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    protocol_args = SimpleNamespace(
        rl_guided_reference_count=16,
        rl_guided_policy_count=16,
        rl_guided_medium_count=24,
        rl_guided_count=32,
        rl_guided_low_noise_scale=0.1,
        rl_guided_medium_noise_scale=0.5,
        rl_guidance_scale=0.01,
        rl_guidance_deviations="-0.5,-0.25,-0.125,0,0.125,0.25,0.5",
        rl_guidance_progress_threshold=5.358,
        rl_guidance_corridor_half_width=3.5,
        rl_guidance_corridor_range=15.0,
        rl_rollout_steps=args.diffusion_steps,
        trajectory_representation=getattr(model_config, "trajectory_representation", "increment"),
        rl_group_size=32,
        rl_safe_set_natural_slots=8,
        rl_safe_set_medium_elite_slots=8,
        rl_safe_set_medium_diverse_slots=8,
        rl_safe_set_global_elite_slots=8,
        rl_route_cost_tolerance=1e-4,
        rl_progress_guard_tolerance=0.005,
        rl_reference_reward_margin=0.0,
        rl_min_component_dispersion=0.01,
    )

    baseline_rows, proposed_rows = [], []
    readiness_rows = []
    guidance_gate_values = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            (
                model_inputs,
                raw_inputs,
                ego_future,
                neighbors_future,
                neighbor_mask,
                scene_names,
            ) = prepare_nuplan_batch(
                batch,
                device,
                model_config.observation_normalizer,
                with_metadata=True,
            )
            scorer_kwargs = dict(
                neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask,
                route_lanes=raw_inputs["route_lanes"],
                static_objects=raw_inputs["static_objects"],
                ego_future=ego_future,
                neighbor_agents_past=raw_inputs["neighbor_agents_past"],
                ego_current_state=raw_inputs["ego_current_state"],
                lanes=raw_inputs["lanes"],
            )

            set_seed(args.seed + batch_index)
            baseline = model.sample(
                model_inputs,
                num_samples=32,
                diffusion_steps=args.diffusion_steps,
                noise_scale=0.1,
            )
            _, baseline_details = scorer(trajectories=baseline, **scorer_kwargs)
            baseline_eligibility = classify_candidate_eligibility(
                baseline, baseline_details
            )

            set_seed(args.seed + batch_index)
            pool, source_ids, guidance_stats = _sample_guided_safe_set_pool(
                model,
                model,
                model_inputs,
                raw_inputs,
                ego_future,
                neighbors_future,
                neighbor_mask,
                scorer,
                protocol_args,
            )
            _, pool_details = scorer(trajectories=pool, **scorer_kwargs)
            pool_eligibility = classify_candidate_eligibility(pool, pool_details)
            quality = _quality(pool_details)
            selected_indices = safe_set_compression_indices(
                quality,
                pool_details["reward"],
                pool_eligibility,
                source_ids,
                target_size=32,
                natural_slots=8,
                medium_elite_slots=8,
                medium_diverse_slots=8,
                global_elite_slots=8,
            )
            selected = gather_candidate_tensor(pool, selected_indices)
            selected_details = gather_candidate_details(
                pool_details, selected_indices, pool.shape[1]
            )
            selected_eligibility = classify_candidate_eligibility(
                selected, selected_details
            )

            baseline_batch = scene_metrics(
                baseline, baseline_details, baseline_eligibility, ego_future
            )
            proposed_batch = scene_metrics(
                selected, selected_details, selected_eligibility, ego_future
            )

            reference_count = protocol_args.rl_guided_reference_count
            reference_safe = pool_eligibility.hard_safe[:, :reference_count]
            reference_values = pool_details["reward"][:, :reference_count]
            reference_top = torch.where(
                reference_safe,
                reference_values,
                torch.full_like(reference_values, -torch.inf),
            ).argmax(dim=1)
            rows = torch.arange(reference_top.shape[0], device=device)
            reference_has_safe = reference_safe.any(dim=1)
            reference_reward = reference_values[rows, reference_top]
            route_bound = pool_details["route_cost"][:, :reference_count][rows, reference_top]
            progress_bound = pool_details["progress_guard_reward"][:, :reference_count][rows, reference_top]
            update_mask = (
                selected_eligibility.hard_safe
                & reference_has_safe[:, None]
                & (
                    selected_details["route_cost"]
                    <= route_bound[:, None] + protocol_args.rl_route_cost_tolerance + 1e-12
                )
                & (
                    selected_details["progress_guard_reward"]
                    >= progress_bound[:, None] - protocol_args.rl_progress_guard_tolerance - 1e-12
                )
                & (
                    selected_details["reward"]
                    >= reference_reward[:, None] + protocol_args.rl_reference_reward_margin
                )
            )
            config = scorer.config
            dispersion = weighted_quality_dispersion(
                _quality(selected_details),
                selected_eligibility.hard_safe,
                (
                    config.aligned_safety_exponent,
                    config.aligned_progress_exponent,
                    config.aligned_route_exponent,
                    config.aligned_comfort_exponent,
                    config.aligned_follow_exponent,
                ),
            )
            active_group = (
                (selected_eligibility.hard_safe.sum(dim=1) >= 2)
                & (dispersion >= protocol_args.rl_min_component_dispersion)
            )
            for row, scene_name in enumerate(scene_names):
                baseline_batch[row]["scene"] = scene_name
                proposed_batch[row]["scene"] = scene_name
                baseline_rows.append(baseline_batch[row])
                proposed_rows.append(proposed_batch[row])
                target_count = int(update_mask[row].sum())
                retained = bool(active_group[row].item()) and target_count > 0
                readiness_rows.append({
                    "scene": scene_name,
                    "quality_dispersion": float(dispersion[row]),
                    "active_quality_group": float(active_group[row]),
                    "update_target_count": target_count,
                    "has_update_target": float(retained),
                    "retained_update_target_count": target_count if retained else None,
                })
            guidance_gate_values.extend(
                [float(guidance_stats["guidance/gate_fraction"])] * len(scene_names)
            )
            print(f"batch={batch_index + 1}/{len(loader)} scenes={len(baseline_rows)}/{args.scenes}")

    baseline_summary = _summary(baseline_rows)
    proposed_summary = _summary(proposed_rows)
    readiness_summary = _summary(readiness_rows)
    baseline_moving_safe = _moving_has_two_safe_fraction(baseline_rows)
    proposed_moving_safe = _moving_has_two_safe_fraction(proposed_rows)
    paired_keys = (
        "endpoint_pairwise_distance_m",
        "trajectory_pairwise_distance_m",
        "safe_endpoint_pairwise_distance_m",
        "safe_trajectory_pairwise_distance_m",
        "safe_reward_std",
        "safe_candidate_fraction",
        "no_collision_candidate_fraction",
        "drivable_candidate_fraction",
        "mean_candidate_ade_m",
    )
    paired = {
        key: _paired_bootstrap(baseline_rows, proposed_rows, key, args.seed + index)
        for index, key in enumerate(paired_keys)
    }

    b_endpoint = baseline_summary["endpoint_pairwise_distance_m"]["p50"]
    p_endpoint = proposed_summary["endpoint_pairwise_distance_m"]["p50"]
    b_trajectory = baseline_summary["trajectory_pairwise_distance_m"]["p50"]
    p_trajectory = proposed_summary["trajectory_pairwise_distance_m"]["p50"]
    b_active = baseline_summary["active_safe_reward_std_0_01"]["mean"]
    p_active = proposed_summary["active_safe_reward_std_0_01"]["mean"]
    diversity_checks = {
        "endpoint_p50_at_least_0_25m": p_endpoint >= 0.25,
        "trajectory_p50_at_least_0_10m": p_trajectory >= 0.10,
        "endpoint_p50_at_least_4x_baseline": _ratio(p_endpoint, b_endpoint) >= 4.0,
        "trajectory_p50_at_least_4x_baseline": _ratio(p_trajectory, b_trajectory) >= 4.0,
        "active_reward_std_fraction_at_least_0_30": p_active >= 0.30,
        "active_reward_std_gain_at_least_0_15": p_active - b_active >= 0.15,
    }
    preservation_checks = {
        "all_candidates_finite": proposed_summary["finite_candidate_fraction"]["mean"] == 1.0,
        "has_two_safe_not_worse_by_0_02": (
            proposed_summary["has_two_safe_candidates"]["mean"]
            >= baseline_summary["has_two_safe_candidates"]["mean"] - 0.02
        ),
        "collision_fraction_not_worse_by_0_01": (
            proposed_summary["no_collision_candidate_fraction"]["mean"]
            >= baseline_summary["no_collision_candidate_fraction"]["mean"] - 0.01
        ),
        "da_fraction_not_worse_by_0_01": (
            proposed_summary["drivable_candidate_fraction"]["mean"]
            >= baseline_summary["drivable_candidate_fraction"]["mean"] - 0.01
        ),
        "mean_ade_not_worse_by_0_05": (
            proposed_summary["mean_candidate_ade_m"]["mean"]
            <= baseline_summary["mean_candidate_ade_m"]["mean"] * 1.05
        ),
    }
    active_quality_fraction = readiness_summary["active_quality_group"]["mean"]
    update_group_fraction = readiness_summary["has_update_target"]["mean"]
    retained_targets = readiness_summary["retained_update_target_count"]["mean"]
    readiness_checks = {
        "quality_active_group_fraction_at_least_0_25": active_quality_fraction >= 0.25,
        "update_target_group_fraction_at_least_0_20": update_group_fraction >= 0.20,
        "targets_per_retained_group_at_least_2": (
            retained_targets is not None and retained_targets >= 2.0
        ),
    }
    verdict = {
        "candidate_differentiation_passed": all(diversity_checks.values()),
        "safety_quality_preservation_passed": all(preservation_checks.values()),
        "training_readiness_passed": all(readiness_checks.values()),
    }
    verdict["overall_pretraining_gate_passed"] = all(verdict.values())

    result = {
        "schema_version": 1,
        "experiment": "minimal paired guided safe-set pre-training gate",
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "selected_scenes": args.scenes,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "baseline": "B10 0.1x32 natural proposals",
            "proposed_pool": "B10-reference 0.1x16 + B10-policy 0.1x16 + medium 0.5x24 + guided 0.1x32",
            "proposed_compression": (
                "8 safe-first natural + 8 safe medium elites + "
                "8 safe medium quality-diverse + 8 global safe elites"
            ),
            "motion_stratification": selection,
            "guidance_gate_fraction": float(np.mean(guidance_gate_values)),
        },
        "identity": {
            "checkpoint": checkpoint_metadata,
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_args": str(args.model_args.resolve()),
            "model_args_sha256": sha256(args.model_args),
            "reward_args": str(args.reward_args.resolve()),
            "reward_args_sha256": sha256(args.reward_args),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
        },
        "pre_registered_thresholds": {
            "endpoint_pairwise_distance_p50_m_min": 0.25,
            "trajectory_pairwise_distance_p50_m_min": 0.10,
            "pairwise_distance_ratio_vs_baseline_min": 4.0,
            "active_safe_reward_std_0_01_fraction_min": 0.30,
            "active_safe_reward_std_0_01_gain_min": 0.15,
            "has_two_safe_fraction_max_drop": 0.02,
            "collision_candidate_fraction_max_drop": 0.01,
            "drivable_candidate_fraction_max_drop": 0.01,
            "mean_candidate_ade_max_relative_increase": 0.05,
            "quality_active_group_fraction_min": 0.25,
            "update_target_group_fraction_min": 0.20,
            "targets_per_retained_group_min": 2.0,
        },
        "acceptance_checks": {
            "diversity": diversity_checks,
            "preservation": preservation_checks,
            "training_readiness": readiness_checks,
        },
        "verdict": verdict,
        "summary": {
            "baseline": baseline_summary,
            "guided_safe_set": proposed_summary,
            "training_readiness": readiness_summary,
        },
        "formal_validation1000_table": {
            "endpoint_pairwise_distance_p50_m": {
                "baseline": b_endpoint,
                "guided_safe_set": p_endpoint,
                "ratio": _ratio(p_endpoint, b_endpoint),
            },
            "trajectory_pairwise_distance_p50_m": {
                "baseline": b_trajectory,
                "guided_safe_set": p_trajectory,
                "ratio": _ratio(p_trajectory, b_trajectory),
            },
            "active_safe_reward_std_0_01_fraction": {
                "baseline": b_active,
                "guided_safe_set": p_active,
                "percentage_point_delta": 100.0 * (p_active - b_active),
            },
            "has_two_safe_candidates_fraction": {
                "baseline": baseline_summary["has_two_safe_candidates"]["mean"],
                "guided_safe_set": proposed_summary["has_two_safe_candidates"]["mean"],
                "percentage_point_delta": 100.0 * (
                    proposed_summary["has_two_safe_candidates"]["mean"]
                    - baseline_summary["has_two_safe_candidates"]["mean"]
                ),
            },
            "moving_has_two_safe_candidates_fraction": {
                "expert_path_length_threshold_m": 2.0,
                "baseline": baseline_moving_safe,
                "guided_safe_set": proposed_moving_safe,
                "percentage_point_delta": (
                    100.0 * (proposed_moving_safe - baseline_moving_safe)
                    if baseline_moving_safe is not None
                    and proposed_moving_safe is not None
                    else None
                ),
            },
            "mean_candidate_ade_m": {
                "baseline": baseline_summary["mean_candidate_ade_m"]["mean"],
                "guided_safe_set": proposed_summary["mean_candidate_ade_m"]["mean"],
                "relative_change_percent": 100.0 * (
                    _ratio(
                        proposed_summary["mean_candidate_ade_m"]["mean"],
                        baseline_summary["mean_candidate_ade_m"]["mean"],
                    )
                    - 1.0
                ),
            },
            "oracle_best_of_group_ade_m": {
                "baseline": baseline_summary["oracle_best_of_group_ade_m"]["mean"],
                "guided_safe_set": proposed_summary["oracle_best_of_group_ade_m"]["mean"],
                "delta_m": (
                    proposed_summary["oracle_best_of_group_ade_m"]["mean"]
                    - baseline_summary["oracle_best_of_group_ade_m"]["mean"]
                ),
            },
            "best_safe_reward_candidate_ade_m": {
                "baseline": baseline_summary["best_safe_reward_candidate_ade_m"]["mean"],
                "guided_safe_set": proposed_summary["best_safe_reward_candidate_ade_m"]["mean"],
                "delta_m": (
                    proposed_summary["best_safe_reward_candidate_ade_m"]["mean"]
                    - baseline_summary["best_safe_reward_candidate_ade_m"]["mean"]
                ),
            },
            "no_collision_candidate_fraction": {
                "baseline": baseline_summary["no_collision_candidate_fraction"]["mean"],
                "guided_safe_set": proposed_summary["no_collision_candidate_fraction"]["mean"],
                "percentage_point_delta": 100.0 * (
                    proposed_summary["no_collision_candidate_fraction"]["mean"]
                    - baseline_summary["no_collision_candidate_fraction"]["mean"]
                ),
            },
        },
        "paired_bootstrap": paired,
        "per_scene": {
            "baseline": baseline_rows,
            "guided_safe_set": proposed_rows,
            "training_readiness": readiness_rows,
        },
        "limitations": [
            "40-scene one-seed pre-training diagnostic, not final statistical evidence",
            "No model weights are updated",
            "Collision and DA use trajectory-level reward-cache proxies",
            "Speeding is evaluated only by later official closed-loop acceptance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(
        f"endpoint_p50: {b_endpoint:.6f} -> {p_endpoint:.6f} m; "
        f"trajectory_p50: {b_trajectory:.6f} -> {p_trajectory:.6f} m; "
        f"active@0.01: {b_active:.3f} -> {p_active:.3f}"
    )
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
