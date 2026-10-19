#!/usr/bin/env python3
"""Audit aligned reward with an expert-conditioned drivable-area hard gate."""

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
MODES = ("legacy", "aligned", "aligned_da")


def profile(spec: str) -> list[float]:
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
    parser.add_argument("--component-dispersion-threshold", type=float, default=0.01)
    return parser.parse_args()


def paired(candidate: list[float], baseline: list[float]) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return {
        **summarize(delta.tolist()),
        "wins": int(np.sum(delta > 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pool_scales, selected_scales = profile(POOL), profile(SELECT)
    device = torch.device("cuda")
    config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        config, args.checkpoint, device, requested_source="ema_state_dict"
    )
    base, reward_args = reward_config_from_json(args.reward_args)
    scorers = {
        "legacy": NuPlanTensorRewardScorer(replace(base, objective_mode="legacy", safety_gate_require_drivable_area=False)),
        "aligned": NuPlanTensorRewardScorer(replace(base, objective_mode="nuplan_aligned", safety_gate_require_drivable_area=False)),
        "aligned_da": NuPlanTensorRewardScorer(replace(base, objective_mode="nuplan_aligned", safety_gate_require_drivable_area=True)),
    }
    dataset = DiffusionPlannerData(
        str(args.data_dir), str(args.data_list), config.agent_num,
        config.predicted_neighbor_num, config.future_len, return_metadata=True,
    )
    indices, counts_by_log = balanced_indices(dataset, args.balanced_scenes, args.seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    coverage = {mode: defaultdict(list) for mode in MODES}
    top = {mode: defaultdict(list) for mode in MODES}
    common_top = {mode: defaultdict(list) for mode in MODES}
    component_dispersion = []
    per_scene = []
    eligibility_mismatch_legacy_aligned = 0

    weights = np.asarray([
        base.aligned_safety_exponent,
        base.aligned_progress_exponent,
        base.aligned_route_exponent,
        base.aligned_comfort_exponent,
        base.aligned_follow_exponent,
    ], dtype=np.float64)
    weights /= weights.sum()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            model_inputs, raw, ego_future, neighbors_future, neighbor_mask, scene_names = prepare_nuplan_batch(
                batch, device, config.observation_normalizer, with_metadata=True
            )
            set_seed(args.seed + batch_index)
            candidates = model.sample(
                model_inputs, num_samples=len(pool_scales),
                diffusion_steps=args.diffusion_steps, candidate_noise_scales=pool_scales,
            )
            _, pool_details = scorers["legacy"](
                trajectories=candidates, neighbors_future=neighbors_future,
                neighbor_mask=neighbor_mask, route_lanes=raw["route_lanes"],
                static_objects=raw["static_objects"], ego_future=ego_future,
                neighbor_agents_past=raw["neighbor_agents_past"],
                ego_current_state=raw["ego_current_state"], lanes=raw["lanes"],
            )
            selected = stratified_safety_diverse_indices(
                candidates, pool_details["reward"],
                candidate_eligible_mask(pool_details, reward_args), pool_scales,
                len(selected_scales), target_noise_scales=selected_scales,
            )
            candidates = gather_candidate_tensor(candidates, selected)
            details = {}
            for mode, scorer in scorers.items():
                _, details[mode] = scorer(
                    trajectories=candidates, neighbors_future=neighbors_future,
                    neighbor_mask=neighbor_mask, route_lanes=raw["route_lanes"],
                    static_objects=raw["static_objects"], ego_future=ego_future,
                    neighbor_agents_past=raw["neighbor_agents_past"],
                    ego_current_state=raw["ego_current_state"], lanes=raw["lanes"],
                )
            horizon = min(candidates.shape[-2], ego_future.shape[-2])
            ade = torch.linalg.vector_norm(
                candidates[..., :horizon, :2] - ego_future[:, None, :horizon, :2], dim=-1
            ).mean(dim=-1)
            fde = torch.linalg.vector_norm(
                candidates[..., horizon - 1, :2] - ego_future[:, None, horizon - 1, :2], dim=-1
            )

            for row, scene_name in enumerate(scene_names):
                eligible = {
                    mode: candidate_eligible_mask(details[mode], reward_args)[row]
                    for mode in MODES
                }
                if not torch.equal(eligible["legacy"], eligible["aligned"]):
                    eligibility_mismatch_legacy_aligned += 1
                record = {"scene": str(scene_name), "modes": {}}
                legacy_has = bool(eligible["legacy"].any())
                da_has = bool(eligible["aligned_da"].any())
                common = legacy_has and da_has

                for mode in MODES:
                    count = int(eligible[mode].sum())
                    coverage[mode]["eligible_count"].append(float(count))
                    coverage[mode]["has_eligible"].append(float(count >= 1))
                    coverage[mode]["has_two"].append(float(count >= 2))
                    reward = details[mode]["reward"][row]
                    if count:
                        chosen = int(torch.where(
                            eligible[mode], reward, torch.full_like(reward, -torch.inf)
                        ).argmax())
                    else:
                        chosen = int(reward.argmax())

                    # Raw DA is always taken from the no-DA scorer.  The gated
                    # scorer intentionally maps expert-invalid corridor scenes
                    # to compliant so they are not falsely rejected.
                    metric_source = details["aligned"]
                    metrics = {
                        "progress": metric_source["progress"][row, chosen],
                        "progress_guard_reward": metric_source["progress_guard_reward"][row, chosen],
                        "risk_reward": metric_source["risk_reward"][row, chosen],
                        "min_ttc_capped_s": metric_source["min_ttc_seconds"][row, chosen].clamp(max=8.0),
                        "no_collision": metric_source["no_collision"][row, chosen].float(),
                        "raw_drivable_area_compliance": metric_source["drivable_area_compliance"][row, chosen].float(),
                        "collision_cost": metric_source["collision_cost"][row, chosen],
                        "route_cost": metric_source["route_cost"][row, chosen],
                        "comfort_cost": metric_source["comfort_cost"][row, chosen],
                        "backward_cost": metric_source["backward_cost"][row, chosen],
                        "ade_m": ade[row, chosen],
                        "fde_m": fde[row, chosen],
                    }
                    values = {key: float(value) for key, value in metrics.items()}
                    if count:
                        for key, value in values.items():
                            top[mode][key].append(value)
                    if common:
                        for key, value in values.items():
                            common_top[mode][key].append(value)
                    record["modes"][mode] = {
                        "eligible_count": count,
                        "top_index": chosen,
                        "top": values,
                    }

                da_eligible = eligible["aligned_da"]
                da_count = int(da_eligible.sum())
                if da_count >= 2:
                    q = torch.stack([
                        details["aligned_da"]["aligned_safety_quality"][row],
                        details["aligned_da"]["aligned_progress_quality"][row],
                        details["aligned_da"]["aligned_route_quality"][row],
                        details["aligned_da"]["aligned_comfort_quality"][row],
                        details["aligned_da"]["aligned_follow_quality"][row],
                    ], dim=-1)[da_eligible]
                    var = q.var(dim=0, unbiased=False).detach().cpu().numpy()
                    dispersion = float(np.sqrt(np.sum(weights * var)))
                else:
                    dispersion = 0.0
                component_dispersion.append(dispersion)
                record["aligned_da_component_dispersion"] = dispersion
                record["aligned_da_component_active"] = bool(
                    da_count >= 2 and dispersion >= args.component_dispersion_threshold
                )
                record["da_gate_active"] = bool(
                    details["aligned_da"]["drivable_area_gate_active"][row, 0]
                )
                per_scene.append(record)

            if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"batch={batch_index + 1}/{len(loader)} scenes={len(per_scene)}",
                    flush=True,
                )

    coverage_summary = {
        mode: {key: summarize(values) for key, values in metrics.items()}
        for mode, metrics in coverage.items()
    }
    top_summary = {
        mode: {key: summarize(values) for key, values in metrics.items()}
        for mode, metrics in top.items()
    }
    paired_common = {
        comparison: {
            key: paired(common_top[mode][key], common_top[baseline][key])
            for key in common_top[baseline]
        }
        for comparison, mode, baseline in [
            ("aligned_minus_legacy", "aligned", "legacy"),
            ("aligned_da_minus_legacy", "aligned_da", "legacy"),
            ("aligned_da_minus_aligned", "aligned_da", "aligned"),
        ]
    }
    da_vs_legacy = paired_common["aligned_da_minus_legacy"]
    coverage_drop_has = coverage_summary["aligned_da"]["has_eligible"]["mean"] - coverage_summary["legacy"]["has_eligible"]["mean"]
    coverage_drop_two = coverage_summary["aligned_da"]["has_two"]["mean"] - coverage_summary["legacy"]["has_two"]["mean"]
    active_fraction = float(np.mean([
        row["aligned_da_component_active"] for row in per_scene
    ]))
    checks = {
        "raw_da_non_decreasing": da_vs_legacy["raw_drivable_area_compliance"]["mean"] >= 0.0,
        # Decimal proportions such as 0.597 - 0.647 can be represented as
        # -0.050000000000000044.  The tolerance prevents an exact boundary
        # result from being rejected solely by binary floating-point error.
        "has_eligible_drop_ge_minus_0p05": coverage_drop_has >= -0.05 - 1e-12,
        "has_two_drop_ge_minus_0p05": coverage_drop_two >= -0.05 - 1e-12,
        "component_active_fraction_ge_0p40": active_fraction >= 0.40,
        "risk_reward_non_decreasing": da_vs_legacy["risk_reward"]["mean"] >= 0.0,
        "min_ttc_delta_ge_minus_0p01s": da_vs_legacy["min_ttc_capped_s"]["mean"] >= -0.01,
        "no_collision_non_decreasing": da_vs_legacy["no_collision"]["mean"] >= 0.0,
        "route_cost_delta_le_1e-4": da_vs_legacy["route_cost"]["mean"] <= 1e-4,
        "comfort_cost_delta_le_1e-4": da_vs_legacy["comfort_cost"]["mean"] <= 1e-4,
        "progress_guard_delta_ge_minus_0p005": da_vs_legacy["progress_guard_reward"]["mean"] >= -0.005,
    }
    result = {
        "schema_version": 1,
        "experiment": "Chapter 4 aligned reward + DA gate + scale-invariant signal audit",
        "created_at": datetime.now().isoformat(),
        "evaluation_only": True,
        "weights_updated": False,
        "protocol": {
            "selected_scenes": len(per_scene),
            "counts_by_log": counts_by_log,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "pool_profile": POOL,
            "selection_profile": SELECT,
            "same_final_32_candidates_for_all_modes": True,
            "eligibility_mismatch_legacy_aligned": eligibility_mismatch_legacy_aligned,
            "component_dispersion_threshold": args.component_dispersion_threshold,
            "component_weights": weights.tolist(),
            "common_eligible_scene_count": len(common_top["legacy"]["progress"]),
            "da_gate_active_scene_count": sum(row["da_gate_active"] for row in per_scene),
        },
        "identity": {
            "checkpoint": checkpoint_metadata,
            "checkpoint_sha256": sha256(args.checkpoint),
            "model_args_sha256": sha256(args.model_args),
            "reward_args_sha256": sha256(args.reward_args),
            "data_list_sha256": sha256(args.data_list),
        },
        "coverage_summary": coverage_summary,
        "top_summary": top_summary,
        "paired_common_eligible": paired_common,
        "component_dispersion": summarize(component_dispersion),
        "component_active_fraction": active_fraction,
        "coverage_delta_aligned_da_minus_legacy": {
            "has_eligible": coverage_drop_has,
            "has_two": coverage_drop_two,
        },
        "pre_registered_gate": {"passed": all(checks.values()), "checks": checks},
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
        "gate": result["pre_registered_gate"],
        "coverage_delta": result["coverage_delta_aligned_da_minus_legacy"],
        "component_active_fraction": active_fraction,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
