#!/usr/bin/env python3
"""Diagnose whether B10 produces useful within-scene candidate variation.

This is an evaluation-only Chapter 4 experiment.  It never updates model
parameters.  Multiple noise scales reuse the same standard-normal base noise,
so changes can be attributed to the scale rather than a different random draw.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed
from scripts.compare_checkpoint_behavior import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-args", required=True, type=Path)
    parser.add_argument("--reward-args", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weight-source", default="ema_state_dict",
                        choices=["auto", "ema_state_dict", "model"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--balanced-scenes", default=0, type=int,
                        help="Use this many scenes, balanced across NPZ log_name; 0 uses all.")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--group-size", default=32, type=int)
    parser.add_argument("--diffusion-steps", default=6, type=int)
    parser.add_argument("--noise-scales", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reward_config_from_json(path: Path) -> tuple[NuPlanRewardConfig, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    kwargs = {}
    for field in fields(NuPlanRewardConfig):
        source = f"reward_{field.name}"
        if source in raw:
            kwargs[field.name] = raw[source]
    if raw.get("reward_use_nuplan_vehicle_geometry", False):
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

        vehicle = get_pacifica_parameters()
        kwargs.update(
            ego_width=vehicle.width,
            ego_length=vehicle.length,
            ego_rear_axle_to_center=vehicle.rear_axle_to_center,
        )
    return NuPlanRewardConfig(**kwargs), raw


def balanced_indices(dataset: DiffusionPlannerData, total: int, seed: int):
    if total <= 0 or total >= len(dataset):
        return list(range(len(dataset))), {"all": len(dataset)}
    by_log = defaultdict(list)
    for index, name in enumerate(dataset.data_list):
        with np.load(Path(dataset.data_dir) / name, allow_pickle=False) as item:
            by_log[str(item["log_name"].item())].append(index)
    rng = np.random.default_rng(seed)
    logs = sorted(by_log)
    quota, remainder = divmod(total, len(logs))
    selected = []
    counts = {}
    for rank, log_name in enumerate(logs):
        candidates = np.asarray(by_log[log_name], dtype=np.int64)
        rng.shuffle(candidates)
        take = min(len(candidates), quota + int(rank < remainder))
        chosen = candidates[:take].tolist()
        selected.extend(chosen)
        counts[log_name] = len(chosen)
    selected.sort()
    if len(selected) != total:
        raise RuntimeError(f"Requested {total} balanced scenes but selected {len(selected)}")
    return selected, counts


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "p10": None,
                "p50": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def candidate_eligible_mask(details, reward_args):
    no_collision = details["no_collision"] >= 1.0
    eligible = details["safety_gate_eligible"] >= 1.0
    if reward_args.get("rl_filter_collision_candidates", False):
        eligible = eligible & no_collision
    if reward_args.get("rl_filter_progress_guard_candidates", False):
        eligible = eligible & (
            details["progress_guard_reward"]
            >= float(reward_args.get("rl_min_progress_guard_reward", 0.9))
        )
    return eligible


def group_metrics(trajectories, details, reward_args, ego_future):
    reward = details["reward"]
    progress = details["progress"]
    no_collision = details["no_collision"] >= 1.0
    eligible = candidate_eligible_mask(details, reward_args)

    counts = eligible.sum(dim=1)
    denominator = counts.clamp_min(1).to(reward.dtype)
    eligible_reward_sum = torch.where(eligible, reward, 0.0).sum(dim=1)
    eligible_mean = eligible_reward_sum / denominator
    centered = torch.where(eligible, reward - eligible_mean[:, None], 0.0)
    eligible_std = torch.sqrt(centered.square().sum(dim=1) / denominator)
    eligible_progress_mean = (
        torch.where(eligible, progress, 0.0).sum(dim=1) / denominator
    )
    eligible_progress_centered = torch.where(
        eligible,
        progress - eligible_progress_mean[:, None],
        0.0,
    )
    eligible_correlation = (
        (centered * eligible_progress_centered).sum(dim=1) / denominator
    ) / (
        torch.sqrt(centered.square().sum(dim=1) / denominator)
        * torch.sqrt(
            eligible_progress_centered.square().sum(dim=1) / denominator
        )
        + 1e-12
    )
    eligible_correlation = torch.where(
        counts >= 2,
        eligible_correlation,
        torch.zeros_like(eligible_correlation),
    )
    masked_max = torch.where(eligible, reward, torch.full_like(reward, -torch.inf)).max(dim=1).values
    masked_min = torch.where(eligible, reward, torch.full_like(reward, torch.inf)).min(dim=1).values
    eligible_range = torch.where(counts > 0, masked_max - masked_min, 0.0)

    xy = trajectories[..., :2]
    endpoint_distance = torch.cdist(xy[..., -1, :], xy[..., -1, :])
    trajectory_distance = torch.linalg.vector_norm(
        xy[:, :, None, :, :] - xy[:, None, :, :, :], dim=-1
    ).mean(dim=-1)
    pair_count = trajectories.shape[1] * (trajectories.shape[1] - 1)

    reward_centered = reward - reward.mean(dim=1, keepdim=True)
    progress_centered = progress - progress.mean(dim=1, keepdim=True)
    correlation = (reward_centered * progress_centered).mean(dim=1) / (
        reward_centered.square().mean(dim=1).sqrt()
        * progress_centered.square().mean(dim=1).sqrt()
        + 1e-12
    )
    best = reward.argmax(dim=1, keepdim=True)
    best_eligible = eligible.gather(1, best).squeeze(1)
    best_reward = reward.gather(1, best).squeeze(1)
    best_eligible_index = torch.where(
        eligible,
        reward,
        torch.full_like(reward, -torch.inf),
    ).argmax(dim=1, keepdim=True)
    horizon = min(trajectories.shape[-2], ego_future.shape[-2])
    position_error = torch.linalg.vector_norm(
        xy[..., :horizon, :] - ego_future[:, None, :horizon, :2], dim=-1
    )
    ade = position_error.mean(dim=-1)
    fde = position_error[..., -1]
    safe_best_ade = ade.gather(1, best_eligible_index).squeeze(1)
    safe_best_fde = fde.gather(1, best_eligible_index).squeeze(1)
    safe_best_progress = progress.gather(1, best_eligible_index).squeeze(1)
    safe_best_ade = torch.where(counts > 0, safe_best_ade, ade[:, 0])
    safe_best_fde = torch.where(counts > 0, safe_best_fde, fde[:, 0])
    safe_best_progress = torch.where(
        counts > 0,
        safe_best_progress,
        progress[:, 0],
    )
    origin = torch.zeros_like(ego_future[:, :1, :2])
    expert_path_length = torch.linalg.vector_norm(
        torch.diff(
            torch.cat([origin, ego_future[..., :2]], dim=1),
            dim=1,
        ),
        dim=-1,
    ).sum(dim=1)

    min_std = float(reward_args.get("rl_min_reward_std", 1e-6))
    return {
        "reward_std_all": reward.std(dim=1, unbiased=False),
        "reward_range_all": reward.max(dim=1).values - reward.min(dim=1).values,
        "reward_std_eligible": eligible_std,
        "reward_range_eligible": eligible_range,
        "eligible_candidates": counts.float(),
        "eligible_candidate_fraction": eligible.float().mean(dim=1),
        "has_eligible_candidate": (counts > 0).float(),
        "has_two_eligible_candidates": (counts >= 2).float(),
        "active_group_at_config_threshold": ((counts >= 2) & (eligible_std >= min_std)).float(),
        "active_group_at_0_01": ((counts >= 2) & (eligible_std >= 0.01)).float(),
        "no_collision_candidate_fraction": no_collision.float().mean(dim=1),
        "drivable_candidate_fraction": details["drivable_area_compliance"].float().mean(dim=1),
        "ttc_eligible_candidate_fraction": details["safety_gate_ttc_eligible"].float().mean(dim=1),
        "endpoint_pairwise_distance_m": endpoint_distance.sum(dim=(1, 2)) / pair_count,
        "trajectory_pairwise_distance_m": trajectory_distance.sum(dim=(1, 2)) / pair_count,
        "final_longitudinal_std_m": xy[..., -1, 0].std(dim=1, unbiased=False),
        "final_lateral_std_m": xy[..., -1, 1].std(dim=1, unbiased=False),
        "reward_progress_correlation": correlation,
        "eligible_reward_progress_correlation": eligible_correlation,
        "expert_path_length_m": expert_path_length,
        "mean_candidate_reward": reward.mean(dim=1),
        "mean_candidate_progress": progress.mean(dim=1),
        "mean_candidate_ade_m": ade.mean(dim=1),
        "mean_candidate_fde_m": fde.mean(dim=1),
        "oracle_best_ade_m": ade.min(dim=1).values,
        "oracle_best_fde_m": fde.min(dim=1).values,
        "single_candidate_ade_m": ade[:, 0],
        "single_candidate_fde_m": fde[:, 0],
        "best_reward_candidate_ade_m": ade.gather(1, best).squeeze(1),
        "best_reward_candidate_fde_m": fde.gather(1, best).squeeze(1),
        "best_eligible_reward_candidate_ade_m": safe_best_ade,
        "best_eligible_reward_candidate_fde_m": safe_best_fde,
        "best_eligible_reward_candidate_progress": safe_best_progress,
        "best_reward_progress_delta": progress.gather(1, best).squeeze(1) - progress.mean(dim=1),
        "best_reward_gain_over_mean": best_reward - reward.mean(dim=1),
        "best_reward_gain_over_single": best_reward - reward[:, 0],
        "best_reward_candidate_eligible": best_eligible.float(),
        "finite_candidate_fraction": torch.isfinite(trajectories).flatten(-2).all(dim=-1).float().mean(dim=1),
    }


def main() -> None:
    args = parse_args()
    noise_scales = [float(item) for item in args.noise_scales.split(",")]
    if not noise_scales or any(value <= 0 for value in noise_scales):
        raise ValueError("noise scales must be positive")
    if args.group_size < 2 or args.repeats < 1 or args.diffusion_steps < 1:
        raise ValueError("group-size >= 2, repeats >= 1, diffusion-steps >= 1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA")

    device = torch.device("cuda")
    model_config = Config(args.model_args)
    model, checkpoint_metadata = load_checkpoint_model(
        model_config, args.checkpoint, device, requested_source=args.weight_source
    )
    reward_config, reward_args = reward_config_from_json(args.reward_args)
    scorer = NuPlanTensorRewardScorer(reward_config)
    dataset = DiffusionPlannerData(
        str(args.data_dir), str(args.data_list), model_config.agent_num,
        model_config.predicted_neighbor_num, model_config.future_len,
        return_metadata=True,
    )
    indices, counts_by_log = balanced_indices(dataset, args.balanced_scenes, args.seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    storage = {str(scale): defaultdict(list) for scale in noise_scales}
    per_scene = []
    with torch.no_grad():
        for repeat in range(args.repeats):
            for batch_index, batch in enumerate(loader):
                (model_inputs, raw_inputs, ego_future, neighbors_future,
                 neighbor_mask, scene_names) = prepare_nuplan_batch(
                    batch, device, model_config.observation_normalizer, with_metadata=True
                )
                sample_seed = args.seed + repeat * 100_000 + batch_index
                for scale in noise_scales:
                    set_seed(sample_seed)
                    trajectories = model.sample(
                        model_inputs, num_samples=args.group_size,
                        diffusion_steps=args.diffusion_steps, noise_scale=scale,
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
                    metrics = group_metrics(
                        trajectories, details, reward_args, ego_future
                    )
                    for name, tensor in metrics.items():
                        storage[str(scale)][name].extend(
                            tensor.detach().float().cpu().tolist()
                        )
                    for scene_index, scene_name in enumerate(scene_names):
                        per_scene.append({
                            "scene": scene_name,
                            "repeat": repeat,
                            "noise_scale": scale,
                            **{
                                name: float(tensor[scene_index].detach().cpu())
                                for name, tensor in metrics.items()
                            },
                        })
                if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
                    print(
                        f"repeat={repeat + 1}/{args.repeats} "
                        f"batch={batch_index + 1}/{len(loader)}"
                    )

    result = {
        "schema_version": 1,
        "experiment": "B10 candidate distribution and effective advantage diagnosis",
        "evaluation_only": True,
        "protocol": {
            "dataset_total": len(dataset),
            "selected_scenes": len(indices),
            "counts_by_log": counts_by_log,
            "group_size": args.group_size,
            "diffusion_steps": args.diffusion_steps,
            "noise_scales": noise_scales,
            "repeats": args.repeats,
            "seed": args.seed,
            "common_base_noise_across_scales": True,
            "configured_min_reward_std": float(reward_args.get("rl_min_reward_std", 1e-6)),
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
        "reward_config": reward_config.__dict__,
        "summary_by_noise_scale": {
            scale: {name: summarize(values) for name, values in metrics.items()}
            for scale, metrics in storage.items()
        },
        "per_scene": per_scene,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved diagnostic to {args.output}")
    for scale, summary in result["summary_by_noise_scale"].items():
        print(
            f"noise={scale}: endpoint_p50="
            f"{summary['endpoint_pairwise_distance_m']['p50']:.6f}m "
            f"reward_std_eligible_p50={summary['reward_std_eligible']['p50']:.6f} "
            f"eligible_mean={summary['eligible_candidate_fraction']['mean']:.6f} "
            f"active@0.01={summary['active_group_at_0_01']['mean']:.6f}"
        )


if __name__ == "__main__":
    main()
