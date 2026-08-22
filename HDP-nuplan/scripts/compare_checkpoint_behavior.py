#!/usr/bin/env python3
"""在相同 NuPlan 场景和扩散噪声下比较监督与 RL checkpoint 的轨迹行为。"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner
from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.rl.trajectory_augmentation import augment_trajectory_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed


HIGHER_IS_BETTER = {
    "reward",
    "progress",
    "no_collision",
}
LOWER_IS_BETTER = {
    "collision_cost",
    "route_cost",
    "comfort_cost",
    "backward_cost",
    "imitation_cost",
    "ade_m",
    "fde_m",
    "heading_error_rad",
    "low_speed_step_fraction",
    "stationary_trajectory_fraction",
    "mean_acceleration_mps2",
    "max_acceleration_mps2",
    "mean_jerk_mps3",
    "max_jerk_mps3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", required=True, type=Path)
    parser.add_argument("--supervised-checkpoint", required=True, type=Path)
    parser.add_argument("--rl-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--supervised-weight-source",
        default="auto",
        choices=["auto", "ema_state_dict", "model"],
    )
    parser.add_argument(
        "--rl-weight-source",
        default="auto",
        choices=["auto", "ema_state_dict", "model"],
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--diffusion-steps", default=10, type=int)
    parser.add_argument("--sampling-noise-scale", default=0.1, type=float)
    parser.add_argument("--trajectory-augmentation-std", default=0.0, type=float)
    parser.add_argument(
        "--reward-progress-guard-weight",
        default=None,
        type=float,
        help="Override the args.json progress guard weight for reward diagnostics.",
    )
    parser.add_argument("--low-speed-threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--top-regressions", default=20, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(
    config: Config,
    checkpoint_path: Path,
    device: torch.device,
    requested_source: str = "auto",
) -> tuple[Hyper_Diffusion_Planner, dict]:
    """优先加载 EMA 权重并严格检查模型结构。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if requested_source != "auto":
        if requested_source not in checkpoint:
            raise KeyError(
                f"{checkpoint_path} does not contain {requested_source!r}"
            )
        source = requested_source
        state_dict = checkpoint[source]
    elif "ema_state_dict" in checkpoint:
        source = "ema_state_dict"
        state_dict = checkpoint[source]
    elif "model" in checkpoint:
        source = "model"
        state_dict = checkpoint[source]
    else:
        source = "checkpoint_root"
        state_dict = checkpoint
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model = Hyper_Diffusion_Planner(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "weight_source": source,
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("loss"),
    }
    return model, metadata


def reward_config_from_args(config: Config) -> NuPlanRewardConfig:
    """从训练 args.json 恢复 reward 配置，并兼容没有新字段的旧文件。"""
    return NuPlanRewardConfig(
        progress_weight=getattr(config, "reward_progress_weight", 1.0),
        collision_weight=getattr(config, "reward_collision_weight", 10.0),
        route_weight=getattr(config, "reward_route_weight", 1.0),
        comfort_weight=getattr(config, "reward_comfort_weight", 0.01),
        backward_weight=getattr(config, "reward_backward_weight", 1.0),
        imitation_weight=getattr(config, "reward_imitation_weight", 0.0),
        collision_distance=getattr(config, "reward_collision_distance", 0.5),
        risk_weight=getattr(config, "reward_risk_weight", 1.0),
        follow_weight=getattr(config, "reward_follow_weight", 3.0),
        lane_weight=getattr(config, "reward_lane_weight", 2.5),
        progress_guard_weight=getattr(
            config, "reward_progress_guard_weight", 0.0
        ),
        progress_guard_stop_tolerance=getattr(
            config, "reward_progress_guard_stop_tolerance", 0.2
        ),
        safety_gate_threshold=getattr(
            config, "reward_safety_gate_threshold", 0.0
        ),
        safety_gate_margin=getattr(config, "reward_safety_gate_margin", 1.0),
        safety_gate_min_ttc_seconds=getattr(
            config, "reward_safety_gate_min_ttc_seconds", 0.0
        ),
        risk_speed_reference=getattr(config, "reward_risk_speed_reference", 15.0),
        ttc_safe_low_speed=getattr(config, "reward_ttc_safe_low_speed", 2.0),
        ttc_safe_high_speed=getattr(config, "reward_ttc_safe_high_speed", 4.0),
        thw_critical=getattr(config, "reward_thw_critical", 0.5),
        thw_safe_low_speed=getattr(config, "reward_thw_safe_low_speed", 1.0),
        thw_safe_high_speed=getattr(config, "reward_thw_safe_high_speed", 2.0),
        occupancy_safe_min=getattr(config, "reward_occupancy_safe_min", 0.5),
        occupancy_safe_max=getattr(config, "reward_occupancy_safe_max", 3.0),
        occupancy_time_headway=getattr(config, "reward_occupancy_time_headway", 0.2),
        rear_end_collision_penalty=getattr(
            config, "reward_rear_end_collision_penalty", 0.3
        ),
        follow_time_gap_low_speed=getattr(
            config, "reward_follow_time_gap_low_speed", 1.0
        ),
        follow_time_gap_high_speed=getattr(
            config, "reward_follow_time_gap_high_speed", 2.0
        ),
        follow_min_spacing=getattr(config, "reward_follow_min_spacing", 2.0),
        follow_speed_tolerance=getattr(
            config, "reward_follow_speed_tolerance", 2.0
        ),
        follow_comfort_acceleration=getattr(
            config, "reward_follow_comfort_acceleration", 2.0
        ),
        follow_comfort_deceleration=getattr(
            config, "reward_follow_comfort_deceleration", 3.0
        ),
        leader_lateral_margin=getattr(
            config, "reward_leader_lateral_margin", 0.5
        ),
        lane_half_width_fallback=getattr(
            config, "reward_lane_half_width_fallback", 1.75
        ),
        lane_change_ratio=getattr(config, "reward_lane_change_ratio", 0.5),
    )


def trajectory_behavior_metrics(
    trajectories: torch.Tensor,
    ego_future: torch.Tensor,
    ego_current_state: torch.Tensor,
    dt: float = 0.1,
    low_speed_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """计算不依赖 reward 权重的物理行为指标，所有输出形状均为 [B,G]。"""
    if trajectories.ndim != 4:
        raise ValueError("trajectories must have shape [B,G,T,4]")
    if dt <= 0:
        raise ValueError("dt must be positive")

    xy = trajectories[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    displacement = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
    step_distance = torch.linalg.vector_norm(displacement, dim=-1)
    velocity = displacement / dt
    speed = torch.linalg.vector_norm(velocity, dim=-1)

    current_velocity = ego_current_state[:, None, None, 4:6].expand(
        -1, trajectories.shape[1], -1, -1
    )
    current_acceleration = ego_current_state[:, None, None, 6:8].expand(
        -1, trajectories.shape[1], -1, -1
    )
    acceleration = torch.diff(
        torch.cat([current_velocity, velocity], dim=-2), dim=-2
    ) / dt
    jerk = torch.diff(
        torch.cat([current_acceleration, acceleration], dim=-2), dim=-2
    ) / dt
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    jerk_norm = torch.linalg.vector_norm(jerk, dim=-1)

    horizon = min(trajectories.shape[-2], ego_future.shape[-2])
    position_error = torch.linalg.vector_norm(
        xy[..., :horizon, :] - ego_future[:, None, :horizon, :2], dim=-1
    )
    candidate_direction = torch.nn.functional.normalize(
        trajectories[..., :horizon, 2:4], dim=-1, eps=1e-6
    )
    expert_direction = torch.nn.functional.normalize(
        ego_future[:, None, :horizon, 2:4], dim=-1, eps=1e-6
    )
    heading_cosine = torch.sum(candidate_direction * expert_direction, dim=-1)
    heading_error = torch.acos(heading_cosine.clamp(-1.0, 1.0))

    path_length = step_distance.sum(dim=-1)
    final_displacement = torch.linalg.vector_norm(xy[..., -1, :], dim=-1)
    return {
        "final_displacement_m": final_displacement,
        "forward_x_m": xy[..., -1, 0],
        "path_length_m": path_length,
        "path_efficiency": final_displacement / path_length.clamp_min(1e-6),
        "mean_speed_mps": speed.mean(dim=-1),
        "low_speed_step_fraction": (speed < low_speed_threshold).float().mean(dim=-1),
        "stationary_trajectory_fraction": (
            speed.mean(dim=-1) < low_speed_threshold
        ).float(),
        "mean_acceleration_mps2": acceleration_norm.mean(dim=-1),
        "max_acceleration_mps2": acceleration_norm.max(dim=-1).values,
        "mean_jerk_mps3": jerk_norm.mean(dim=-1),
        "max_jerk_mps3": jerk_norm.max(dim=-1).values,
        "ade_m": position_error.mean(dim=-1),
        "fde_m": position_error[..., -1],
        "heading_error_rad": heading_error.mean(dim=-1),
    }


def group_diversity_metrics(
    trajectories: torch.Tensor,
    metrics: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """量化每个场景候选组的 reward 差异和几何多样性，输出形状为 [B]。"""
    group_size = trajectories.shape[1]
    reward = metrics["reward"]
    progress = metrics["progress"]
    if group_size == 1:
        zeros = reward[:, 0] * 0
        return {
            "group_reward_std": zeros,
            "group_reward_range": zeros,
            "group_progress_std": zeros,
            "group_endpoint_diversity_m": zeros,
            "group_reward_progress_correlation": zeros,
            "best_reward_gain": zeros,
            "best_reward_progress_delta": zeros,
            "best_reward_path_length_delta_m": zeros,
            "best_reward_collision_cost_delta": zeros,
        }

    endpoint_distance = torch.cdist(
        trajectories[..., -1, :2], trajectories[..., -1, :2]
    )
    pair_count = group_size * (group_size - 1)
    reward_centered = reward - reward.mean(dim=1, keepdim=True)
    progress_centered = progress - progress.mean(dim=1, keepdim=True)
    correlation = (reward_centered * progress_centered).mean(dim=1) / (
        reward_centered.square().mean(dim=1).sqrt()
        * progress_centered.square().mean(dim=1).sqrt()
        + 1e-12
    )
    best_index = reward.argmax(dim=1, keepdim=True)

    def best_minus_mean(name: str) -> torch.Tensor:
        value = metrics[name]
        return value.gather(1, best_index).squeeze(1) - value.mean(dim=1)

    return {
        "group_reward_std": reward.std(dim=1, unbiased=False),
        "group_reward_range": reward.max(dim=1).values - reward.min(dim=1).values,
        "group_progress_std": progress.std(dim=1, unbiased=False),
        "group_endpoint_diversity_m": endpoint_distance.sum(dim=(1, 2)) / pair_count,
        "group_reward_progress_correlation": correlation,
        "best_reward_gain": best_minus_mean("reward"),
        "best_reward_progress_delta": best_minus_mean("progress"),
        "best_reward_path_length_delta_m": best_minus_mean("path_length_m"),
        "best_reward_collision_cost_delta": best_minus_mean("collision_cost"),
    }


def summarize_values(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _correlation(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    # 候选级指标有 B*G 个值，group 指标只有 B 个值；不同统计层级不能直接相关。
    if (
        x_array.size != y_array.size
        or x_array.size < 2
        or x_array.std() == 0
        or y_array.std() == 0
    ):
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def paired_summary(
    supervised: Dict[str, list], rl: Dict[str, list]
) -> Dict[str, dict]:
    result = {}
    for metric in supervised.keys() & rl.keys():
        supervised_values = np.asarray(supervised[metric], dtype=np.float64)
        rl_values = np.asarray(rl[metric], dtype=np.float64)
        delta = rl_values - supervised_values
        baseline_mean = supervised_values.mean()
        record = summarize_values(delta)
        record["supervised_mean"] = float(baseline_mean)
        record["rl_mean"] = float(rl_values.mean())
        record["relative_change_percent"] = float(
            100.0 * delta.mean() / max(abs(baseline_mean), 1e-12)
        )
        record["rl_higher_fraction"] = float((delta > 0).mean())
        if metric in HIGHER_IS_BETTER:
            record["improved_fraction"] = record["rl_higher_fraction"]
        elif metric in LOWER_IS_BETTER:
            record["improved_fraction"] = float((delta < 0).mean())
        result[metric] = record
    return result


def _append_metrics(storage: Dict[str, list], metrics: Dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        storage[key].extend(value.detach().float().cpu().reshape(-1).tolist())


@torch.no_grad()
def score_trajectories(
    trajectories: torch.Tensor,
    raw_inputs: dict,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_mask: torch.Tensor,
    scorer: NuPlanTensorRewardScorer,
    low_speed_threshold: float,
) -> Dict[str, torch.Tensor]:
    _, reward_details = scorer(
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
    physical = trajectory_behavior_metrics(
        trajectories,
        ego_future,
        raw_inputs["ego_current_state"],
        dt=scorer.config.dt,
        low_speed_threshold=low_speed_threshold,
    )
    combined = {**reward_details, **physical}
    return {**combined, **group_diversity_metrics(trajectories, combined)}


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.num_samples < 1 or args.diffusion_steps < 1:
        raise ValueError("repeats, num_samples and diffusion_steps must be positive")
    if args.sampling_noise_scale <= 0:
        raise ValueError("sampling_noise_scale must be positive")
    if args.trajectory_augmentation_std < 0:
        raise ValueError("trajectory_augmentation_std must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    device = torch.device(args.device)
    config = Config(args.args_file)
    supervised_model, supervised_checkpoint = load_checkpoint_model(
        config,
        args.supervised_checkpoint,
        device,
        requested_source=args.supervised_weight_source,
    )
    rl_model, rl_checkpoint = load_checkpoint_model(
        config,
        args.rl_checkpoint,
        device,
        requested_source=args.rl_weight_source,
    )
    reward_config = reward_config_from_args(config)
    if args.reward_progress_guard_weight is not None:
        if args.reward_progress_guard_weight < 0:
            raise ValueError("reward progress guard weight must be non-negative")
        reward_config.progress_guard_weight = args.reward_progress_guard_weight
    scorer = NuPlanTensorRewardScorer(reward_config)

    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        config.agent_num,
        config.predicted_neighbor_num,
        config.future_len,
        return_metadata=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    values = {
        "supervised": defaultdict(list),
        "rl": defaultdict(list),
    }
    per_scene = defaultdict(
        lambda: {
            "supervised": defaultdict(list),
            "rl": defaultdict(list),
        }
    )

    for repeat in range(args.repeats):
        for batch_index, batch in enumerate(data_loader):
            (
                model_inputs,
                raw_inputs,
                ego_future,
                neighbors_future,
                neighbor_mask,
                scene_names,
            ) = prepare_nuplan_batch(
                batch, device, config.observation_normalizer, with_metadata=True
            )
            sample_seed = args.seed + repeat * 100_000 + batch_index

            # 两个模型在每个 batch 前重置同一个 RNG 状态，因此接收完全相同的 xT。
            set_seed(sample_seed)
            supervised_trajectories = supervised_model.sample(
                model_inputs,
                num_samples=args.num_samples,
                diffusion_steps=args.diffusion_steps,
                noise_scale=args.sampling_noise_scale,
            )
            supervised_trajectories = augment_trajectory_batch(
                supervised_trajectories,
                std=args.trajectory_augmentation_std,
            )
            set_seed(sample_seed)
            rl_trajectories = rl_model.sample(
                model_inputs,
                num_samples=args.num_samples,
                diffusion_steps=args.diffusion_steps,
                noise_scale=args.sampling_noise_scale,
            )
            rl_trajectories = augment_trajectory_batch(
                rl_trajectories,
                std=args.trajectory_augmentation_std,
            )

            batch_metrics = {
                "supervised": score_trajectories(
                    supervised_trajectories,
                    raw_inputs,
                    ego_future,
                    neighbors_future,
                    neighbor_mask,
                    scorer,
                    args.low_speed_threshold,
                ),
                "rl": score_trajectories(
                    rl_trajectories,
                    raw_inputs,
                    ego_future,
                    neighbors_future,
                    neighbor_mask,
                    scorer,
                    args.low_speed_threshold,
                ),
            }
            for label, metrics in batch_metrics.items():
                _append_metrics(values[label], metrics)
                for scene_index, scene_name in enumerate(scene_names):
                    for metric, tensor in metrics.items():
                        per_scene[scene_name][label][metric].append(
                            float(tensor[scene_index].mean().cpu())
                        )

            if (batch_index + 1) % 20 == 0 or batch_index + 1 == len(data_loader):
                print(
                    f"repeat={repeat + 1}/{args.repeats} "
                    f"batch={batch_index + 1}/{len(data_loader)}"
                )

    model_summary = {
        label: {metric: summarize_values(metric_values) for metric, metric_values in data.items()}
        for label, data in values.items()
    }
    paired = paired_summary(values["supervised"], values["rl"])
    correlations = {
        label: {
            metric: _correlation(data["reward"], metric_values)
            for metric, metric_values in data.items()
            if metric != "reward"
        }
        for label, data in values.items()
    }

    scene_deltas = []
    for scene_name, scene_data in per_scene.items():
        record = {"scene": scene_name}
        for metric in ("reward", "progress", "path_length_m", "mean_speed_mps", "low_speed_step_fraction"):
            supervised_mean = float(np.mean(scene_data["supervised"][metric]))
            rl_mean = float(np.mean(scene_data["rl"][metric]))
            record[f"supervised/{metric}"] = supervised_mean
            record[f"rl/{metric}"] = rl_mean
            record[f"delta/{metric}"] = rl_mean - supervised_mean
        scene_deltas.append(record)
    scene_deltas.sort(key=lambda record: record["delta/progress"])

    result = {
        "schema_version": 1,
        "comparison_protocol": {
            "common_random_numbers": True,
            "seed_formula": "seed + repeat * 100000 + batch_index",
            "dataset_samples": len(dataset),
            "generated_trajectories_per_model": len(dataset)
            * args.repeats
            * args.num_samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "diffusion_steps": args.diffusion_steps,
            "sampling_noise_scale": args.sampling_noise_scale,
            "trajectory_augmentation_std_m": args.trajectory_augmentation_std,
            "low_speed_threshold_mps": args.low_speed_threshold,
        },
        "args_file": str(args.args_file.resolve()),
        "data_list": str(args.data_list.resolve()),
        "checkpoints": {
            "supervised": supervised_checkpoint,
            "rl": rl_checkpoint,
        },
        "reward_config": reward_config.__dict__,
        "models": model_summary,
        "paired_delta_rl_minus_supervised": paired,
        "reward_correlations": correlations,
        "worst_progress_regressions": scene_deltas[: args.top_regressions],
        "best_progress_changes": list(reversed(scene_deltas[-args.top_regressions :])),
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved behavior comparison to {args.output}")
    for metric in (
        "reward",
        "progress",
        "path_length_m",
        "mean_speed_mps",
        "low_speed_step_fraction",
        "collision_cost",
        "route_cost",
        "comfort_cost",
        "ade_m",
    ):
        record = paired[metric]
        print(
            f"{metric}: supervised={record['supervised_mean']:.6f} "
            f"rl={record['rl_mean']:.6f} delta={record['mean']:+.6f}"
        )


if __name__ == "__main__":
    main()
