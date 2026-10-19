#!/usr/bin/env python3
"""Audit reward ranking on one frozen Repair candidate set per scene.

The B10 sampler is executed once for each batch.  The original Repair
pool-200 -> group-32 selection is then frozen, and every reward objective
scores exactly those same trajectories.  No model parameter is updated.
"""

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


MODES = (
    "legacy",
    "nuplan_aligned",
    "nuplan_score_proxy_v2",
    "nuplan_score_proxy_v3",
)
POOL_PROFILE = "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
SELECTION_PROFILE = "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"
HIGHER_IS_BETTER = {
    "progress": True,
    "progress_guard_reward": True,
    "risk_reward": True,
    "min_ttc_capped_s": True,
    "no_collision": True,
    "drivable_area_compliance": True,
    "collision_cost": False,
    "route_cost": False,
    "comfort_cost": False,
    "backward_cost": False,
    "ade_m": False,
    "fde_m": False,
}


def parse_profile(specification: str) -> list[float]:
    return expand_scale_counts(
        [
            (float(segment.split("x", 1)[0]), int(segment.split("x", 1)[1]))
            for segment in specification.split("+")
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-args", required=True, type=Path)
    parser.add_argument("--reward-args", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--balanced-scenes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--pool-profile", default=POOL_PROFILE)
    parser.add_argument("--selection-profile", default=SELECTION_PROFILE)
    parser.add_argument("--weight-source", default="ema_state_dict")
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if np.isfinite(value) else None


def is_dominated(qualities: np.ndarray, chosen: int, tolerance: float = 1e-9) -> bool:
    selected = qualities[chosen]
    no_worse = np.all(qualities >= selected[None] - tolerance, axis=1)
    strictly_better = np.any(qualities > selected[None] + tolerance, axis=1)
    return bool(np.any(no_worse & strictly_better))


def finite_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite audit value: {result}")
    return result


def paired_summary(candidate: list[float], baseline: list[float]) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    tolerance = 1e-12
    return {
        **summarize(delta.tolist()),
        "wins": int(np.sum(delta > tolerance)),
        "losses": int(np.sum(delta < -tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("reward ranking audit requires CUDA")
    pool_scales = parse_profile(args.pool_profile)
    selected_scales = parse_profile(args.selection_profile)
    if len(selected_scales) < 2 or len(pool_scales) < len(selected_scales):
        raise ValueError("invalid pool/selection profiles")

    device = torch.device("cuda")
    model_config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        model_config,
        args.checkpoint,
        device,
        requested_source=args.weight_source,
    )
    base_reward_config, reward_args = reward_config_from_json(args.reward_args)
    scorers = {
        mode: NuPlanTensorRewardScorer(replace(base_reward_config, objective_mode=mode))
        for mode in MODES
    }
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        model_config.agent_num,
        model_config.predicted_neighbor_num,
        model_config.future_len,
        return_metadata=True,
    )
    indices, counts_by_log = balanced_indices(dataset, args.balanced_scenes, args.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    top_values: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in MODES
    }
    reward_stats: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in MODES
    }
    correlations: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in MODES
    }
    dominated: dict[str, list[float]] = defaultdict(list)
    top_indices: dict[str, list[int]] = defaultdict(list)
    pairwise_rank_corr: dict[str, list[float]] = defaultdict(list)
    per_scene: list[dict] = []
    eligibility_reference: list[list[bool]] = []
    eligibility_mismatch = 0

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
            set_seed(args.seed + batch_index)
            pool = model.sample(
                model_inputs,
                num_samples=len(pool_scales),
                diffusion_steps=args.diffusion_steps,
                candidate_noise_scales=pool_scales,
            )
            _, pool_legacy = scorers["legacy"](
                trajectories=pool,
                neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask,
                route_lanes=raw_inputs["route_lanes"],
                static_objects=raw_inputs["static_objects"],
                ego_future=ego_future,
                neighbor_agents_past=raw_inputs["neighbor_agents_past"],
                ego_current_state=raw_inputs["ego_current_state"],
                lanes=raw_inputs["lanes"],
            )
            pool_eligible = candidate_eligible_mask(pool_legacy, reward_args)
            chosen = stratified_safety_diverse_indices(
                pool,
                pool_legacy["reward"],
                pool_eligible,
                pool_scales,
                len(selected_scales),
                target_noise_scales=selected_scales,
            )
            trajectories = gather_candidate_tensor(pool, chosen)

            details_by_mode = {}
            for mode, scorer in scorers.items():
                _, details_by_mode[mode] = scorer(
                    trajectories=trajectories,
                    neighbors_future=neighbors_future,
                    neighbor_mask=neighbor_mask,
                    route_lanes=raw_inputs["route_lanes"],
                    static_objects=raw_inputs["static_objects"],
                    ego_future=ego_future,
                    neighbor_agents_past=raw_inputs["neighbor_agents_past"],
                    ego_current_state=raw_inputs["ego_current_state"],
                    lanes=raw_inputs["lanes"],
                )

            horizon = min(trajectories.shape[-2], ego_future.shape[-2])
            ade = torch.linalg.vector_norm(
                trajectories[..., :horizon, :2] - ego_future[:, None, :horizon, :2],
                dim=-1,
            ).mean(dim=-1)
            fde = torch.linalg.vector_norm(
                trajectories[..., horizon - 1, :2] - ego_future[:, None, horizon - 1, :2],
                dim=-1,
            )

            for scene_index, scene_name in enumerate(scene_names):
                scene_record = {"scene": str(scene_name), "modes": {}}
                legacy_eligible = candidate_eligible_mask(
                    details_by_mode["legacy"], reward_args
                )[scene_index]
                eligibility_reference.append(legacy_eligible.cpu().tolist())
                eligible_count = int(legacy_eligible.sum())
                scene_record["eligible_count"] = eligible_count
                scene_record["has_eligible"] = eligible_count > 0

                mode_rewards = {}
                for mode in MODES:
                    details = details_by_mode[mode]
                    eligible = candidate_eligible_mask(details, reward_args)[scene_index]
                    if not torch.equal(eligible, legacy_eligible):
                        eligibility_mismatch += 1
                    reward = details["reward"][scene_index]
                    mode_rewards[mode] = reward.detach().float().cpu().numpy()
                    if eligible_count > 0:
                        masked = torch.where(
                            eligible,
                            reward,
                            torch.full_like(reward, -torch.inf),
                        )
                        top_index = int(masked.argmax())
                        eligible_rewards = reward[eligible]
                        reward_std = finite_float(
                            eligible_rewards.std(unbiased=False)
                            if eligible_count > 1
                            else torch.zeros((), device=device)
                        )
                        reward_range = finite_float(
                            eligible_rewards.max() - eligible_rewards.min()
                        )
                        if eligible_count > 1 and reward_range > 1e-12:
                            sorted_reward = eligible_rewards.sort(descending=True).values
                            normalized_margin = finite_float(
                                (sorted_reward[0] - sorted_reward[1]) / reward_range
                            )
                        else:
                            normalized_margin = 0.0
                    else:
                        top_index = int(reward.argmax())
                        reward_std = 0.0
                        reward_range = 0.0
                        normalized_margin = 0.0

                    top_indices[mode].append(top_index)
                    reward_stats[mode]["eligible_count"].append(float(eligible_count))
                    reward_stats[mode]["reward_std_eligible"].append(reward_std)
                    reward_stats[mode]["reward_range_eligible"].append(reward_range)
                    reward_stats[mode]["normalized_top2_margin"].append(normalized_margin)
                    reward_stats[mode]["active_at_0_01"].append(
                        float(eligible_count >= 2 and reward_std >= 0.01)
                    )

                    metric_tensors = {
                        "progress": details["progress"][scene_index],
                        "progress_guard_reward": details["progress_guard_reward"][scene_index],
                        "risk_reward": details["risk_reward"][scene_index],
                        "min_ttc_capped_s": details["min_ttc_seconds"][scene_index].clamp(max=8.0),
                        "no_collision": details["no_collision"][scene_index].float(),
                        "drivable_area_compliance": details["drivable_area_compliance"][scene_index].float(),
                        "collision_cost": details["collision_cost"][scene_index],
                        "route_cost": details["route_cost"][scene_index],
                        "comfort_cost": details["comfort_cost"][scene_index],
                        "backward_cost": details["backward_cost"][scene_index],
                        "ade_m": ade[scene_index],
                        "fde_m": fde[scene_index],
                    }
                    top_record = {}
                    for metric_name, tensor in metric_tensors.items():
                        value = finite_float(tensor[top_index])
                        top_record[metric_name] = value
                        if eligible_count > 0:
                            top_values[mode][metric_name].append(value)

                    if eligible_count >= 3:
                        eligible_np = legacy_eligible.detach().cpu().numpy().astype(bool)
                        reward_np = mode_rewards[mode][eligible_np]
                        quality_columns = []
                        for metric_name, tensor in metric_tensors.items():
                            values = tensor.detach().float().cpu().numpy()[eligible_np]
                            oriented = values if HIGHER_IS_BETTER[metric_name] else -values
                            correlation = spearman(reward_np, oriented)
                            if correlation is not None:
                                correlations[mode][metric_name].append(correlation)
                            if metric_name in {
                                "progress_guard_reward",
                                "risk_reward",
                                "min_ttc_capped_s",
                                "no_collision",
                                "drivable_area_compliance",
                                "collision_cost",
                                "route_cost",
                                "comfort_cost",
                                "backward_cost",
                            }:
                                quality_columns.append(oriented)
                        qualities = np.stack(quality_columns, axis=1)
                        eligible_positions = np.flatnonzero(eligible_np)
                        chosen_position = int(np.flatnonzero(eligible_positions == top_index)[0])
                        dominated[mode].append(float(is_dominated(qualities, chosen_position)))

                    scene_record["modes"][mode] = {
                        "top_index": top_index,
                        "reward_std_eligible": reward_std,
                        "reward_range_eligible": reward_range,
                        "normalized_top2_margin": normalized_margin,
                        "top": top_record,
                    }

                if eligible_count >= 3:
                    mask = legacy_eligible.detach().cpu().numpy().astype(bool)
                    for left_index, left in enumerate(MODES):
                        for right in MODES[left_index + 1 :]:
                            value = spearman(mode_rewards[left][mask], mode_rewards[right][mask])
                            if value is not None:
                                pairwise_rank_corr[f"{left}__vs__{right}"].append(value)
                per_scene.append(scene_record)

            if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"batch={batch_index + 1}/{len(loader)} scenes={len(per_scene)}",
                    flush=True,
                )

    summary_by_mode = {}
    for mode in MODES:
        summary_by_mode[mode] = {
            "reward_statistics": {
                key: summarize(values) for key, values in reward_stats[mode].items()
            },
            "top_eligible_candidate": {
                key: summarize(values) for key, values in top_values[mode].items()
            },
            "eligible_rank_correlation_reward_vs_oriented_metric": {
                key: summarize(values) for key, values in correlations[mode].items()
            },
            "pareto_dominated_top_fraction": (
                float(np.mean(dominated[mode])) if dominated[mode] else None
            ),
            "top_same_as_legacy_fraction": float(
                np.mean(np.asarray(top_indices[mode]) == np.asarray(top_indices["legacy"]))
            ),
        }

    paired_vs_legacy = {}
    for mode in MODES[1:]:
        paired_vs_legacy[mode] = {
            metric: paired_summary(top_values[mode][metric], top_values["legacy"][metric])
            for metric in top_values["legacy"]
        }
        delta = paired_vs_legacy[mode]
        paired_vs_legacy[mode]["pre_registered_training_candidate_gate"] = {
            "passed": bool(
                delta["collision_cost"]["mean"] <= 1e-4
                and delta["min_ttc_capped_s"]["mean"] >= -0.01
                and delta["route_cost"]["mean"] <= 1e-4
                and delta["comfort_cost"]["mean"] <= 1e-4
                and delta["progress_guard_reward"]["mean"] >= -0.005
                and delta["no_collision"]["mean"] >= 0.0
            ),
            "checks": {
                "collision_cost_delta_le_1e-4": delta["collision_cost"]["mean"] <= 1e-4,
                "min_ttc_delta_ge_minus_0p01s": delta["min_ttc_capped_s"]["mean"] >= -0.01,
                "route_cost_delta_le_1e-4": delta["route_cost"]["mean"] <= 1e-4,
                "comfort_cost_delta_le_1e-4": delta["comfort_cost"]["mean"] <= 1e-4,
                "progress_guard_delta_ge_minus_0p005": delta["progress_guard_reward"]["mean"] >= -0.005,
                "no_collision_delta_ge_0": delta["no_collision"]["mean"] >= 0.0,
            },
        }

    result = {
        "schema_version": 1,
        "experiment": "Chapter 4 frozen-candidate reward ranking audit",
        "created_at": datetime.now().isoformat(),
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "selected_scenes": len(per_scene),
            "counts_by_log": counts_by_log,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "pool_profile": args.pool_profile,
            "selection_profile": args.selection_profile,
            "candidate_pool_generated_once_per_scene": True,
            "same_final_32_candidates_for_all_objectives": True,
            "selection_objective": "legacy (matches Repair-100 training)",
            "objective_modes": list(MODES),
            "eligibility_mismatch_count": eligibility_mismatch,
            "eligible_scene_count": len(top_values["legacy"]["progress"]),
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
        "pre_registered_gate_thresholds": {
            "collision_cost_mean_delta_max": 1e-4,
            "min_ttc_capped_mean_delta_min_s": -0.01,
            "route_cost_mean_delta_max": 1e-4,
            "comfort_cost_mean_delta_max": 1e-4,
            "progress_guard_mean_delta_min": -0.005,
            "no_collision_mean_delta_min": 0.0,
        },
        "summary_by_mode": summary_by_mode,
        "paired_top_candidate_vs_legacy": paired_vs_legacy,
        "pairwise_eligible_reward_rank_correlations": {
            key: summarize(values) for key, values in pairwise_rank_corr.items()
        },
        "per_scene": per_scene,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output),
        "sha256": digest,
        "scene_count": len(per_scene),
        "eligible_scene_count": len(top_values["legacy"]["progress"]),
        "eligibility_mismatch_count": eligibility_mismatch,
        "training_candidate_gates": {
            mode: paired_vs_legacy[mode]["pre_registered_training_candidate_gate"]["passed"]
            for mode in MODES[1:]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
