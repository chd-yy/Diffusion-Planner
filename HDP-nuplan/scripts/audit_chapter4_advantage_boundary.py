#!/usr/bin/env python3
"""Audit physical meaning and perturbation stability of Ds boundary groups."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.model.diffusion_utils.candidate_noise import expand_scale_counts
from hdp_nuplan.rl.candidate_selection import gather_candidate_tensor, stratified_safety_diverse_indices
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


POOL = "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
SELECT = "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"


def scales(spec: str) -> list[float]:
    return expand_scale_counts([
        (float(part.split("x", 1)[0]), int(part.split("x", 1)[1]))
        for part in spec.split("+")
    ])


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
    parser.add_argument("--lower", type=float, default=0.005)
    parser.add_argument("--upper", type=float, default=0.01)
    parser.add_argument("--perturbation-std", type=float, default=0.001)
    parser.add_argument("--perturbation-repeats", type=int, default=256)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def percentile_span(values: torch.Tensor) -> float:
    array = values.detach().float().cpu().numpy()
    return float(np.percentile(array, 90) - np.percentile(array, 10))


def main() -> None:
    args = parse_args()
    if not (0 < args.lower < args.upper):
        raise ValueError("expected 0 < lower < upper")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pool_scales, selected_scales = scales(POOL), scales(SELECT)
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
        base.aligned_safety_exponent,
        base.aligned_progress_exponent,
        base.aligned_route_exponent,
        base.aligned_comfort_exponent,
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

    boundary = []
    active_at_lower = 0
    active_at_upper = 0
    global_scene_index = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            model_inputs, raw, ego_future, neighbors_future, neighbor_mask, names = prepare_nuplan_batch(
                batch, device, config.observation_normalizer, with_metadata=True
            )
            set_seed(args.seed + batch_index)
            pool = model.sample(
                model_inputs, num_samples=len(pool_scales),
                diffusion_steps=args.diffusion_steps, candidate_noise_scales=pool_scales,
            )
            _, pool_details = legacy(
                trajectories=pool, neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask, route_lanes=raw["route_lanes"],
                static_objects=raw["static_objects"], ego_future=ego_future,
                neighbor_agents_past=raw["neighbor_agents_past"],
                ego_current_state=raw["ego_current_state"], lanes=raw["lanes"],
            )
            chosen = stratified_safety_diverse_indices(
                pool, pool_details["reward"],
                candidate_eligible_mask(pool_details, reward_args), pool_scales,
                len(selected_scales), target_noise_scales=selected_scales,
            )
            candidates = gather_candidate_tensor(pool, chosen)
            _, details = aligned_da(
                trajectories=candidates, neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask, route_lanes=raw["route_lanes"],
                static_objects=raw["static_objects"], ego_future=ego_future,
                neighbor_agents_past=raw["neighbor_agents_past"],
                ego_current_state=raw["ego_current_state"], lanes=raw["lanes"],
            )
            eligible_all = candidate_eligible_mask(details, reward_args)
            origin = torch.zeros_like(ego_future[:, :1, :2])
            expert_path = torch.linalg.vector_norm(
                torch.diff(torch.cat([origin, ego_future[..., :2]], dim=1), dim=1), dim=-1
            ).sum(dim=1)

            for row, name in enumerate(names):
                eligible = eligible_all[row]
                count = int(eligible.sum())
                if count < 2:
                    global_scene_index += 1
                    continue
                quality = torch.stack([
                    details["aligned_safety_quality"][row],
                    details["aligned_progress_quality"][row],
                    details["aligned_route_quality"][row],
                    details["aligned_comfort_quality"][row],
                    details["aligned_follow_quality"][row],
                ], dim=-1)[eligible]
                quality_np = quality.detach().float().cpu().numpy().astype(np.float64)
                dispersion = float(np.sqrt(np.sum(weights * quality_np.var(axis=0))))
                active_at_lower += int(dispersion >= args.lower)
                active_at_upper += int(dispersion >= args.upper)
                if not (args.lower <= dispersion < args.upper):
                    global_scene_index += 1
                    continue

                nominal = np.exp(np.sum(weights * np.log(np.clip(quality_np, 1e-4, 1)), axis=1))
                nominal_top = int(np.argmax(nominal))
                nominal_ranks = rankdata(nominal)
                rng = np.random.default_rng(args.seed + 1_000_000 + global_scene_index)
                stable = 0
                rank_correlations = []
                for _ in range(args.perturbation_repeats):
                    perturbed_quality = np.clip(
                        quality_np + rng.normal(0.0, args.perturbation_std, quality_np.shape),
                        1e-4,
                        1.0,
                    )
                    perturbed = np.exp(np.sum(
                        weights * np.log(perturbed_quality), axis=1
                    ))
                    stable += int(int(np.argmax(perturbed)) == nominal_top)
                    perturbed_ranks = rankdata(perturbed)
                    corr = float(np.corrcoef(nominal_ranks, perturbed_ranks)[0, 1])
                    if np.isfinite(corr):
                        rank_correlations.append(corr)

                metric_spans = {
                    "progress_m": percentile_span(details["progress"][row][eligible]),
                    "ttc_capped_s": percentile_span(details["min_ttc_seconds"][row][eligible].clamp(max=8.0)),
                    "risk_reward": percentile_span(details["risk_reward"][row][eligible]),
                    "route_cost": percentile_span(details["route_cost"][row][eligible]),
                    "comfort_cost": percentile_span(details["comfort_cost"][row][eligible]),
                    "follow_reward": percentile_span(details["follow_reward"][row][eligible]),
                }
                material_checks = {
                    "progress_span_ge_0p10m": metric_spans["progress_m"] >= 0.10,
                    "ttc_span_ge_0p10s": metric_spans["ttc_capped_s"] >= 0.10,
                    "risk_span_ge_0p01": metric_spans["risk_reward"] >= 0.01,
                    "route_span_ge_0p01": metric_spans["route_cost"] >= 0.01,
                    "comfort_span_ge_0p01": metric_spans["comfort_cost"] >= 0.01,
                    "follow_span_ge_0p01": metric_spans["follow_reward"] >= 0.01,
                }
                sorted_nominal = np.sort(nominal)[::-1]
                boundary.append({
                    "scene": str(name),
                    "eligible_count": count,
                    "dispersion": dispersion,
                    "expert_path_length_m": float(expert_path[row]),
                    "moving_expert_gt_2m": bool(expert_path[row] > 2.0),
                    "top1_stability": stable / args.perturbation_repeats,
                    "mean_rank_spearman": float(np.mean(rank_correlations)),
                    "nominal_top2_gap": float(sorted_nominal[0] - sorted_nominal[1]),
                    "metric_p90_minus_p10": metric_spans,
                    "material_checks": material_checks,
                    "material": any(material_checks.values()),
                })
                global_scene_index += 1

            if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"batch={batch_index + 1}/{len(loader)} boundary={len(boundary)}",
                    flush=True,
                )

    stable_fraction = float(np.mean([row["top1_stability"] >= 0.8 for row in boundary])) if boundary else 0.0
    material_fraction = float(np.mean([row["material"] for row in boundary])) if boundary else 0.0
    mean_rank_spearman = float(np.mean([row["mean_rank_spearman"] for row in boundary])) if boundary else 0.0
    active_fraction_lower = active_at_lower / len(indices)
    active_fraction_upper = active_at_upper / len(indices)
    checks = {
        "boundary_count_matches_93": len(boundary) == 93,
        "stable_group_fraction_ge_0p80": stable_fraction >= 0.80,
        "material_group_fraction_ge_0p80": material_fraction >= 0.80,
        "mean_rank_spearman_ge_0p90": mean_rank_spearman >= 0.90,
        "active_fraction_at_lower_ge_0p40": active_fraction_lower >= 0.40,
    }
    result = {
        "schema_version": 1,
        "experiment": "Chapter 4 scale-invariant advantage boundary audit",
        "created_at": datetime.now().isoformat(),
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "selected_scenes": len(indices),
            "counts_by_log": counts_by_log,
            "seed": args.seed,
            "pool_profile": POOL,
            "selection_profile": SELECT,
            "lower_inclusive": args.lower,
            "upper_exclusive": args.upper,
            "perturbation_std": args.perturbation_std,
            "perturbation_repeats": args.perturbation_repeats,
            "component_weights": weights.tolist(),
        },
        "identity": {
            "checkpoint": checkpoint_metadata,
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_args_sha256": sha256(args.model_args),
            "reward_args_sha256": sha256(args.reward_args),
            "data_list_sha256": sha256(args.data_list),
        },
        "summary": {
            "boundary_count": len(boundary),
            "stable_group_fraction": stable_fraction,
            "material_group_fraction": material_fraction,
            "mean_rank_spearman": mean_rank_spearman,
            "active_fraction_at_0p005": active_fraction_lower,
            "active_fraction_at_0p01": active_fraction_upper,
            "moving_group_fraction": float(np.mean([row["moving_expert_gt_2m"] for row in boundary])) if boundary else 0.0,
            "top1_stability": summarize([row["top1_stability"] for row in boundary]),
            "rank_spearman": summarize([row["mean_rank_spearman"] for row in boundary]),
            "dispersion": summarize([row["dispersion"] for row in boundary]),
            "top2_gap": summarize([row["nominal_top2_gap"] for row in boundary]),
        },
        "pre_registered_gate": {"passed": all(checks.values()), "checks": checks},
        "boundary_scenes": boundary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "summary": result["summary"],
        "gate": result["pre_registered_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
