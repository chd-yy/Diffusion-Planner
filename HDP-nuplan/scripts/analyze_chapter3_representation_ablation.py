#!/usr/bin/env python3
"""Create paired R0/C0 trajectory-representation diagnostics and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.open_loop_metrics import trajectory_open_loop_metrics
from hdp_nuplan.utils.train_utils import set_seed
from scripts.evaluate_supervised_open_loop import load_model, sha256


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-args", required=True, type=Path)
    parser.add_argument("--r0-checkpoint", required=True, type=Path)
    parser.add_argument("--c0-args", required=True, type=Path)
    parser.add_argument("--c0-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--diffusion-steps", default=10, type=int)
    parser.add_argument("--sampling-noise-scale", default=0.1, type=float)
    parser.add_argument("--dt", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    name: str,
    args_path: Path,
    checkpoint_path: Path,
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    config = Config(args_path)
    model, checkpoint_metadata = load_model(config, checkpoint_path, device)
    repeat_trajectories = []
    repeat_metrics: dict[str, list[np.ndarray]] = {}
    targets = None
    current_states = None
    scenes: list[str] | None = None

    for repeat in range(args.repeats):
        trajectories = []
        target_chunks = []
        state_chunks = []
        repeat_scenes: list[str] = []
        metric_chunks: dict[str, list[np.ndarray]] = {}
        for batch_index, batch in enumerate(data_loader):
            model_inputs, raw_inputs, ego_future, _, _, scene_names = prepare_nuplan_batch(
                batch,
                device,
                config.observation_normalizer,
                with_metadata=True,
            )
            set_seed(args.seed + repeat * 1_000_000 + batch_index)
            prediction = model.sample(
                model_inputs,
                num_samples=1,
                diffusion_steps=args.diffusion_steps,
                noise_scale=args.sampling_noise_scale,
            )
            metrics = trajectory_open_loop_metrics(
                prediction,
                ego_future,
                raw_inputs["ego_current_state"],
                dt=args.dt,
                horizons_seconds=(1.0, 3.0, 5.0, 8.0),
            )
            trajectories.append(prediction[:, 0].float().cpu().numpy())
            target_chunks.append(ego_future.float().cpu().numpy())
            state_chunks.append(
                raw_inputs["ego_current_state"].float().cpu().numpy()
            )
            repeat_scenes.extend(str(value) for value in scene_names)
            for metric, values in metrics.items():
                metric_chunks.setdefault(metric, []).append(
                    values[:, 0].float().cpu().numpy()
                )
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == len(data_loader):
                print(
                    f"{name}: repeat {repeat + 1}/{args.repeats} "
                    f"batch {batch_index + 1}/{len(data_loader)}"
                )
        repeat_trajectories.append(np.concatenate(trajectories))
        current_targets = np.concatenate(target_chunks)
        current_states_array = np.concatenate(state_chunks)
        if targets is None:
            targets = current_targets
            current_states = current_states_array
            scenes = repeat_scenes
        elif (
            scenes != repeat_scenes
            or not np.array_equal(targets, current_targets)
            or not np.array_equal(current_states, current_states_array)
        ):
            raise RuntimeError(f"{name} dataset order changed between repeats")
        for metric, chunks in metric_chunks.items():
            repeat_metrics.setdefault(metric, []).append(np.concatenate(chunks))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": name,
        "config": config,
        "checkpoint": checkpoint_metadata,
        "trajectories": np.stack(repeat_trajectories),
        "targets": targets,
        "current_states": current_states,
        "scenes": scenes,
        "metrics": {
            metric: np.stack(chunks)
            for metric, chunks in repeat_metrics.items()
        },
    }


def finite_difference_profiles(
    xy: np.ndarray,
    current_state: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.zeros((1, 2), dtype=xy.dtype)
    velocity = np.diff(np.concatenate([origin, xy], axis=0), axis=0) / dt
    acceleration = np.diff(
        np.concatenate([current_state[None, 4:6], velocity], axis=0), axis=0
    ) / dt
    jerk = np.diff(
        np.concatenate([current_state[None, 6:8], acceleration], axis=0), axis=0
    ) / dt
    return (
        np.linalg.norm(velocity, axis=-1),
        np.linalg.norm(acceleration, axis=-1),
        np.linalg.norm(jerk, axis=-1),
    )


def load_log_names(data_dir: Path, scenes: list[str]) -> list[str]:
    log_names = []
    for scene in scenes:
        with np.load(data_dir / scene, allow_pickle=False) as data:
            log_names.append(str(data["log_name"].item()))
    return log_names


def clustered_interval(
    per_scene_delta: np.ndarray,
    log_names: list[str],
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict:
    unique_logs = sorted(set(log_names))
    if len(unique_logs) < 2:
        raise ValueError("cluster bootstrap requires at least two logs")
    names = np.asarray(log_names)
    log_means = np.asarray(
        [per_scene_delta[names == log_name].mean() for log_name in unique_logs]
    )
    rng = np.random.default_rng(seed)
    samples = log_means[
        rng.integers(0, len(unique_logs), size=(iterations, len(unique_logs)))
    ].mean(axis=1)
    return {
        "cluster_unit": "nuPlan log_name",
        "cluster_count": len(unique_logs),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "mean_of_log_means": float(log_means.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "min_log_mean": float(log_means.min()),
        "max_log_mean": float(log_means.max()),
    }


def select_examples(r0: dict, c0: dict) -> dict[str, int]:
    ade_delta = (r0["metrics"]["ade_8s_m"] - c0["metrics"]["ade_8s_m"]).mean(axis=0)
    fde_delta = (r0["metrics"]["fde_8s_m"] - c0["metrics"]["fde_8s_m"]).mean(axis=0)
    jerk_delta = (
        r0["metrics"]["mean_jerk_mps3"]
        - c0["metrics"]["mean_jerk_mps3"]
    ).mean(axis=0)
    selected = {
        "typical_ade_delta": int(np.argmin(np.abs(ade_delta - np.median(ade_delta)))),
        "largest_r0_ade_degradation": int(np.argmax(ade_delta)),
        "largest_r0_fde_improvement": int(np.argmin(fde_delta)),
        "largest_r0_jerk_degradation": int(np.argmax(jerk_delta)),
    }
    return selected


def plot_error_summary(r0: dict, c0: dict, output: Path) -> None:
    horizons = (1, 3, 5, 8)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "ade", "平均位移误差（ADE）"),
        (axes[1], "fde", "终点位移误差（FDE）"),
    ):
        for result, label, marker in (
            (r0, "R0绝对轨迹", "o"),
            (c0, "C0增量轨迹", "s"),
        ):
            values = [result["metrics"][f"{prefix}_{h}s_m"].mean() for h in horizons]
            axis.plot(horizons, values, marker=marker, linewidth=2, label=label)
        axis.set_title(title)
        axis.set_xlabel("预测时域（s）")
        axis.set_ylabel("误差（m，越低越好）")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_paired_distributions(r0: dict, c0: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    pairs = (
        ("ade_8s_m", "R0 - C0 8 s ADE (m)"),
        ("mean_jerk_mps3", "R0 - C0 mean jerk proxy (m/s^3)"),
    )
    for axis, (metric, label) in zip(axes, pairs):
        delta = (r0["metrics"][metric] - c0["metrics"][metric]).reshape(-1)
        axis.hist(delta, bins=50, color="#4c78a8", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(delta.mean(), color="#e45756", linestyle="--", label=f"mean={delta.mean():.3f}")
        axis.set_xlabel(label)
        axis.set_ylabel("Scenes")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_trajectories(
    r0: dict,
    c0: dict,
    selected: dict[str, int],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), constrained_layout=True)
    for axis, (reason, index) in zip(axes.flat, selected.items()):
        target = r0["targets"][index]
        axis.plot(target[:, 0], target[:, 1], color="black", linewidth=2.2, label="Expert")
        axis.plot(
            r0["trajectories"][0, index, :, 0],
            r0["trajectories"][0, index, :, 1],
            color="#e45756",
            linewidth=1.6,
            label="R0 absolute",
        )
        axis.plot(
            c0["trajectories"][0, index, :, 0],
            c0["trajectories"][0, index, :, 1],
            color="#4c78a8",
            linewidth=1.6,
            label="C0 increment",
        )
        axis.scatter([0], [0], marker="x", color="black", s=35)
        axis.set_title(f"{reason}\n{Path(r0['scenes'][index]).stem}", fontsize=9)
        axis.set_xlabel("Longitudinal x (m)")
        axis.set_ylabel("Lateral y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_kinematic_profiles(
    r0: dict,
    c0: dict,
    index: int,
    dt: float,
    output: Path,
) -> None:
    current_state = r0["current_states"][index]
    profiles = {
        "Expert": finite_difference_profiles(r0["targets"][index, :, :2], current_state, dt),
        "R0 absolute": finite_difference_profiles(r0["trajectories"][0, index, :, :2], current_state, dt),
        "C0 increment": finite_difference_profiles(c0["trajectories"][0, index, :, :2], current_state, dt),
    }
    colors = {"Expert": "black", "R0 absolute": "#e45756", "C0 increment": "#4c78a8"}
    time = np.arange(1, r0["targets"].shape[1] + 1) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True, constrained_layout=True)
    labels = ("Speed (m/s)", "Acceleration proxy (m/s^2)", "Jerk proxy (m/s^3)")
    for profile_index, (axis, ylabel) in enumerate(zip(axes, labels)):
        for name, values in profiles.items():
            axis.plot(time, values[profile_index], label=name, color=colors[name], linewidth=1.5)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Future time (s)")
    axes[0].set_title(f"Kinematic sanity check: {Path(r0['scenes'][index]).stem}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    c0_config = Config(args.c0_args)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        c0_config.agent_num,
        c0_config.predicted_neighbor_num,
        c0_config.future_len,
        return_metadata=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    r0 = evaluate("R0_absolute", args.r0_args, args.r0_checkpoint, data_loader, device, args)
    c0 = evaluate("C0_increment", args.c0_args, args.c0_checkpoint, data_loader, device, args)
    if r0["scenes"] != c0["scenes"]:
        raise RuntimeError("paired scene order changed between R0 and C0")
    if not np.array_equal(r0["targets"], c0["targets"]):
        raise RuntimeError("paired expert targets changed between R0 and C0")
    for result in (r0, c0):
        if not np.isfinite(result["trajectories"]).all():
            raise RuntimeError(f"{result['name']} contains non-finite trajectories")

    log_names = load_log_names(args.data_dir, r0["scenes"])
    selected = select_examples(r0, c0)
    plot_error_summary(r0, c0, args.output_dir / "horizon_error_comparison.png")
    plot_paired_distributions(r0, c0, args.output_dir / "paired_delta_distributions.png")
    plot_trajectories(r0, c0, selected, args.output_dir / "representative_trajectories.png")
    plot_kinematic_profiles(
        r0,
        c0,
        selected["largest_r0_jerk_degradation"],
        args.dt,
        args.output_dir / "worst_jerk_kinematic_profiles.png",
    )

    npz_payload = {
        "r0_trajectories": r0["trajectories"],
        "c0_trajectories": c0["trajectories"],
        "expert_trajectories": r0["targets"],
        "ego_current_states": r0["current_states"],
    }
    for metric, values in r0["metrics"].items():
        npz_payload[f"r0__{metric}"] = values
        npz_payload[f"c0__{metric}"] = c0["metrics"][metric]
    np.savez_compressed(args.output_dir / "paired_per_scene_data.npz", **npz_payload)

    selected_details = {}
    for reason, index in selected.items():
        r0_metrics = {
            name: float(values[:, index].mean())
            for name, values in r0["metrics"].items()
        }
        c0_metrics = {
            name: float(values[:, index].mean())
            for name, values in c0["metrics"].items()
        }
        selected_details[reason] = {
            "index": index,
            "scene": r0["scenes"][index],
            "log_name": log_names[index],
            "metrics_are_repeat_means": True,
            "r0_ade_8s_m": r0_metrics["ade_8s_m"],
            "c0_ade_8s_m": c0_metrics["ade_8s_m"],
            "r0_fde_8s_m": r0_metrics["fde_8s_m"],
            "c0_fde_8s_m": c0_metrics["fde_8s_m"],
            "r0_mean_acceleration_mps2": r0_metrics["mean_acceleration_mps2"],
            "c0_mean_acceleration_mps2": c0_metrics["mean_acceleration_mps2"],
            "r0_mean_jerk_mps3": r0_metrics["mean_jerk_mps3"],
            "c0_mean_jerk_mps3": c0_metrics["mean_jerk_mps3"],
        }

    summary = {
        "protocol": {
            "dataset_samples": len(dataset),
            "repeats": args.repeats,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "sampling_noise_scale": args.sampling_noise_scale,
            "paired_common_random_numbers": True,
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
        },
        "inputs": {
            "r0_args": str(args.r0_args.resolve()),
            "r0_checkpoint": r0["checkpoint"],
            "r0_representation": r0["config"].trajectory_representation,
            "r0_state_normalizer": r0["config"].state_normalizer.to_dict(),
            "c0_args": str(args.c0_args.resolve()),
            "c0_checkpoint": c0["checkpoint"],
            "c0_representation": c0["config"].trajectory_representation,
            "c0_state_normalizer": c0["config"].state_normalizer.to_dict(),
        },
        "selected_examples": selected_details,
        "paired": {},
    }
    for metric in r0["metrics"]:
        delta = r0["metrics"][metric] - c0["metrics"][metric]
        per_scene_delta = delta.mean(axis=0)
        summary["paired"][metric] = {
            "r0_mean": float(r0["metrics"][metric].mean()),
            "c0_mean": float(c0["metrics"][metric].mean()),
            "r0_minus_c0_mean": float(delta.mean()),
            "r0_lower_fraction": float((delta < 0).mean()),
            "r0_higher_fraction": float((delta > 0).mean()),
            "delta_p10": float(np.percentile(delta, 10)),
            "delta_p50": float(np.percentile(delta, 50)),
            "delta_p90": float(np.percentile(delta, 90)),
            "log_cluster_bootstrap95": clustered_interval(
                per_scene_delta,
                log_names,
                seed=args.seed,
            ),
        }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "scene_order.json").write_text(
        json.dumps(r0["scenes"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved representation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
