#!/usr/bin/env python3
"""Evaluate structured B10 candidate-noise profiles without updating weights."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.model.diffusion_utils.candidate_noise import expand_scale_counts
from hdp_nuplan.rl.candidate_selection import (
    gather_candidate_details,
    gather_candidate_tensor,
    safety_diverse_reward_indices,
    safety_reward_spread_indices,
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
    group_metrics,
    reward_config_from_json,
    sha256,
    summarize,
)


DEFAULT_PROFILES = (
    "baseline_01=0.1x32;"
    "uniform_05=0.5x32;"
    "uniform_10=1.0x32;"
    "anchor16_mid16=0.1x16+0.5x16;"
    "anchor8_mid16_wide8=0.1x8+0.5x16+1.0x8;"
    "anchor20_wide12=0.1x20+1.0x12;"
    "pool64_anchor_mid_wide=0.1x16+0.5x32+1.0x16;"
    "pool96_anchor_mid_wide=0.1x24+0.5x48+1.0x24"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-args", required=True, type=Path)
    parser.add_argument("--reward-args", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--weight-source",
        default="ema_state_dict",
        choices=["auto", "ema_state_dict", "model"],
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--balanced-scenes", default=100, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--group-size", default=32, type=int)
    parser.add_argument("--diffusion-steps", default=6, type=int)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument(
        "--selection-profile",
        default="",
        help=(
            "Optional final-group scale quotas, e.g. "
            "0.1x10+0.5x16+1.0x4+1.5x2. Applied to oversampled pools."
        ),
    )
    parser.add_argument("--baseline-profile", default="baseline_01")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_profiles(specification: str, group_size: int):
    profiles = OrderedDict()
    for raw_profile in specification.split(";"):
        raw_profile = raw_profile.strip()
        if not raw_profile:
            continue
        if "=" not in raw_profile:
            raise ValueError(f"profile lacks '=': {raw_profile}")
        name, raw_segments = raw_profile.split("=", 1)
        name = name.strip()
        if not name or name in profiles:
            raise ValueError(f"profile name must be unique and nonempty: {name!r}")
        scale_counts = []
        for raw_segment in raw_segments.split("+"):
            if "x" not in raw_segment:
                raise ValueError(f"profile segment lacks 'x': {raw_segment}")
            raw_scale, raw_count = raw_segment.split("x", 1)
            scale_counts.append((float(raw_scale), int(raw_count)))
        scales = expand_scale_counts(scale_counts)
        if len(scales) < group_size:
            raise ValueError(
                f"profile {name} contains {len(scales)} candidates; "
                f"expected at least group-size {group_size}"
            )
        profiles[name] = scales
    if not profiles:
        raise ValueError("at least one candidate profile is required")
    return profiles


def conditional_summary(values, mask):
    values_array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=np.float64) > 0.5
    return summarize(values_array[mask_array].tolist())


def acceptance_record(summary, metrics, baseline_summary, baseline_metrics):
    moving = np.asarray(metrics["expert_path_length_m"]) > 2.0
    baseline_moving = np.asarray(baseline_metrics["expert_path_length_m"]) > 2.0
    moving_coverage = float(
        np.asarray(metrics["has_two_eligible_candidates"])[moving].mean()
    )
    baseline_moving_coverage = float(
        np.asarray(baseline_metrics["has_two_eligible_candidates"])[
            baseline_moving
        ].mean()
    )
    eligible_corr = conditional_summary(
        metrics["eligible_reward_progress_correlation"],
        metrics["has_two_eligible_candidates"],
    )
    baseline_single_ade = baseline_summary["single_candidate_ade_m"]["mean"]
    checks = OrderedDict(
        finite_candidates=(summary["finite_candidate_fraction"]["mean"] == 1.0),
        endpoint_diversity=(summary["endpoint_pairwise_distance_m"]["p50"] >= 0.25),
        trajectory_diversity=(summary["trajectory_pairwise_distance_m"]["p50"] >= 0.10),
        active_group_fraction=(summary["active_group_at_0_01"]["mean"] >= 0.40),
        eligible_coverage_gain=(
            summary["has_two_eligible_candidates"]["mean"]
            - baseline_summary["has_two_eligible_candidates"]["mean"]
            + 1e-12
            >= 0.05
        ),
        moving_coverage_gain=(
            moving_coverage - baseline_moving_coverage + 1e-12 >= 0.05
        ),
        mean_ade_preservation=(
            summary["mean_candidate_ade_m"]["mean"]
            <= baseline_summary["mean_candidate_ade_m"]["mean"] * 1.01
        ),
        oracle_ade_gain=(
            baseline_single_ade - summary["oracle_best_ade_m"]["mean"] >= 0.10
        ),
        safe_reward_best_ade=(
            summary["best_eligible_reward_candidate_ade_m"]["mean"]
            <= baseline_single_ade
        ),
        collision_preservation=(
            summary["no_collision_candidate_fraction"]["mean"]
            >= baseline_summary["no_collision_candidate_fraction"]["mean"] - 0.005
        ),
        reward_progress_alignment=(
            eligible_corr["p50"] is not None and eligible_corr["p50"] >= 0.5
        ),
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "moving_has_two_eligible_fraction": moving_coverage,
        "baseline_moving_has_two_eligible_fraction": baseline_moving_coverage,
        "eligible_reward_progress_correlation": eligible_corr,
    }


def main() -> None:
    args = parse_args()
    if args.group_size < 2 or args.repeats < 1 or args.diffusion_steps < 1:
        raise ValueError("group-size >=2, repeats >=1 and diffusion-steps >=1 required")
    profiles = parse_profiles(args.profiles, args.group_size)
    selection_scales = None
    if args.selection_profile:
        selection_scales = expand_scale_counts(
            [
                (float(segment.split("x", 1)[0]), int(segment.split("x", 1)[1]))
                for segment in args.selection_profile.split("+")
            ]
        )
        if len(selection_scales) != args.group_size:
            raise ValueError("selection-profile must contain group-size candidates")
    if args.baseline_profile not in profiles:
        raise ValueError("baseline-profile must be one of the configured profiles")
    if not torch.cuda.is_available():
        raise RuntimeError("candidate profile validation requires CUDA")

    device = torch.device("cuda")
    model_config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        model_config,
        args.checkpoint,
        device,
        requested_source=args.weight_source,
    )
    reward_config, reward_args = reward_config_from_json(args.reward_args)
    scorer = NuPlanTensorRewardScorer(reward_config)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        model_config.agent_num,
        model_config.predicted_neighbor_num,
        model_config.future_len,
        return_metadata=True,
    )
    indices, counts_by_log = balanced_indices(
        dataset, args.balanced_scenes, args.seed
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    storage = {name: defaultdict(list) for name in profiles}
    with torch.no_grad():
        for repeat in range(args.repeats):
            for batch_index, batch in enumerate(loader):
                (
                    model_inputs,
                    raw_inputs,
                    ego_future,
                    neighbors_future,
                    neighbor_mask,
                    _,
                ) = prepare_nuplan_batch(
                    batch,
                    device,
                    model_config.observation_normalizer,
                    with_metadata=True,
                )
                sample_seed = args.seed + repeat * 100_000 + batch_index
                for profile_name, scales in profiles.items():
                    set_seed(sample_seed)
                    trajectories = model.sample(
                        model_inputs,
                        num_samples=len(scales),
                        diffusion_steps=args.diffusion_steps,
                        candidate_noise_scales=scales,
                    )
                    _, details = scorer(
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
                    pool_size = len(scales)
                    if pool_size > args.group_size:
                        eligible = candidate_eligible_mask(details, reward_args)
                        selected_indices = stratified_safety_diverse_indices(
                            trajectories,
                            details["reward"],
                            eligible,
                            scales,
                            args.group_size,
                            target_noise_scales=selection_scales,
                        )
                        trajectories = gather_candidate_tensor(
                            trajectories,
                            selected_indices,
                        )
                        details = gather_candidate_details(
                            details,
                            selected_indices,
                            pool_size,
                        )
                    metrics = group_metrics(
                        trajectories,
                        details,
                        reward_args,
                        ego_future,
                    )
                    for metric_name, tensor in metrics.items():
                        storage[profile_name][metric_name].extend(
                            tensor.detach().float().cpu().tolist()
                        )
                if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                    print(
                        f"repeat={repeat + 1}/{args.repeats} "
                        f"batch={batch_index + 1}/{len(loader)}"
                    )

    summaries = {
        profile: {
            metric: summarize(values)
            for metric, values in profile_metrics.items()
        }
        for profile, profile_metrics in storage.items()
    }
    baseline_summary = summaries[args.baseline_profile]
    baseline_metrics = storage[args.baseline_profile]
    acceptance = {
        profile: acceptance_record(
            summaries[profile],
            storage[profile],
            baseline_summary,
            baseline_metrics,
        )
        for profile in profiles
    }
    result = {
        "schema_version": 1,
        "experiment": "B10 structured candidate-generation repair",
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "dataset_total": len(dataset),
            "selected_scenes": len(indices),
            "counts_by_log": counts_by_log,
            "group_size": args.group_size,
            "diffusion_steps": args.diffusion_steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "common_base_noise_across_profiles": True,
            "baseline_profile": args.baseline_profile,
            "profiles": {
                name: list(scales) for name, scales in profiles.items()
            },
            "selection_profile": (
                list(selection_scales) if selection_scales is not None else None
            ),
        },
        "identity": {
            "checkpoint": checkpoint_metadata,
            "model_args": str(args.model_args.resolve()),
            "model_args_sha256": sha256(args.model_args),
            "reward_args": str(args.reward_args.resolve()),
            "reward_args_sha256": sha256(args.reward_args),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
        },
        "acceptance_thresholds": {
            "endpoint_pairwise_distance_p50_m": 0.25,
            "trajectory_pairwise_distance_p50_m": 0.10,
            "active_group_at_reward_std_0_01": 0.40,
            "has_two_eligible_gain": 0.05,
            "moving_has_two_eligible_gain": 0.05,
            "mean_candidate_ade_max_relative_increase": 0.01,
            "oracle_ade_min_gain_m": 0.10,
            "no_collision_fraction_max_drop": 0.005,
            "eligible_reward_progress_correlation_p50": 0.5,
        },
        "summary_by_profile": summaries,
        "acceptance_by_profile": acceptance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved candidate-generation validation to {args.output}")
    for name, record in acceptance.items():
        summary = summaries[name]
        passed = sum(record["checks"].values())
        print(
            f"{name}: passed={record['passed']} checks={passed}/{len(record['checks'])} "
            f"endpoint_p50={summary['endpoint_pairwise_distance_m']['p50']:.4f}m "
            f"active={summary['active_group_at_0_01']['mean']:.3f} "
            f"has_two={summary['has_two_eligible_candidates']['mean']:.3f} "
            f"ADE={summary['mean_candidate_ade_m']['mean']:.4f}m"
        )


if __name__ == "__main__":
    main()
