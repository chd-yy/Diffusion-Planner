#!/usr/bin/env python3
"""Audit gradient scale and temporal bias of the HDP hybrid waypoint loss.

The audit never updates model parameters.  It replays a deterministic subset of
the validation set, computes the diffusion increment loss and the physical-space
waypoint loss from the same decoder output, and compares their gradients in both
decoder-output space and parameter space.  Multiple detached-integral windows
can be evaluated without changing the forward waypoint values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed
from hdp_nuplan.utils.traj_kinematics import (
    decode_trajectory_representation,
    encode_trajectory_representation,
)
from scripts.evaluate_supervised_open_loop import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Repeat for each checkpoint to audit.",
    )
    parser.add_argument(
        "--model-args",
        action="append",
        default=[],
        metavar="NAME=ARGS_FILE",
        help="Optional per-model args.json override.",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-batches", default=16, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--eps", default=1e-3, type=float)
    parser.add_argument("--hybrid-weight", default=0.01, type=float)
    parser.add_argument(
        "--window-sizes",
        default=(0, 10),
        nargs="+",
        type=int,
        help="Detached-integral gradient windows to compare; 0 is full cumsum.",
    )
    parser.add_argument(
        "--parameter-scope",
        default="all",
        choices=("all", "decoder"),
        help="Parameters used for model-space norm and cosine diagnostics.",
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_paths(specs: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected NAME=PATH, got {spec!r}")
        name, value = spec.split("=", 1)
        name = name.strip()
        path = Path(value).expanduser().resolve()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate name in {spec!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def vector_norm(grads: Iterable[torch.Tensor | None]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            squared += grad.detach().double().square().sum().cpu()
    return math.sqrt(float(squared))


def vector_cosine(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> float:
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for left_grad, right_grad in zip(left, right):
        if left_grad is None or right_grad is None:
            continue
        left_cpu = left_grad.detach().double().cpu()
        right_cpu = right_grad.detach().double().cpu()
        dot += (left_cpu * right_cpu).sum()
        left_sq += left_cpu.square().sum()
        right_sq += right_cpu.square().sum()
    denominator = math.sqrt(float(left_sq) * float(right_sq))
    return float(dot) / denominator if denominator > 0 else float("nan")


def summarize(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def temporal_gradient_norm(gradient: torch.Tensor) -> np.ndarray:
    """Return the batch-mean XY gradient norm for every future time index."""
    return (
        torch.linalg.vector_norm(gradient[..., :2], dim=-1)
        .mean(dim=0)
        .detach()
        .double()
        .cpu()
        .numpy()
    )


def forward_losses(
    model: Hyper_Diffusion_Planner,
    model_inputs: dict[str, torch.Tensor],
    ego_future: torch.Tensor,
    config: Config,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one training-style noisy forward pass and return reusable tensors."""
    target = encode_trajectory_representation(
        ego_future,
        getattr(config, "trajectory_representation", "increment"),
    )
    normalized_target = config.state_normalizer(target)
    batch_size = normalized_target.shape[0]
    diffusion_time = (
        torch.rand(batch_size, device=normalized_target.device) * (1 - eps) + eps
    )
    noise = torch.randn_like(normalized_target)
    mean, std = model.sde.marginal_prob(normalized_target, diffusion_time)
    std = std.view(-1, *([1] * (normalized_target.ndim - 1)))
    noisy_state = mean + std * noise
    _, decoder_output = model(
        {
            **model_inputs,
            "sampled_trajectories": noisy_state,
            "diffusion_time": diffusion_time,
        }
    )
    model_prediction = decoder_output["score"]
    prediction = model.sde.transform(
        f"{config.diffusion_model_type}->{config.diffusion_supervision_type}",
        model_prediction,
        diffusion_time,
        noisy_state,
    )
    if config.diffusion_supervision_type != "x_start":
        raise ValueError(
            "this Chapter-3 audit currently requires x_start supervision, got "
            f"{config.diffusion_supervision_type!r}"
        )
    increment_loss = torch.sum((prediction - normalized_target).square(), dim=-1).mean()
    predicted_x_start = model.sde.transform(
        f"{config.diffusion_model_type}->x_start",
        model_prediction,
        diffusion_time,
        noisy_state,
    )
    predicted_state = config.state_normalizer.inverse(predicted_x_start)
    return increment_loss, model_prediction, predicted_state, diffusion_time


def audit_model(
    model: Hyper_Diffusion_Planner,
    data_loader: DataLoader,
    config: Config,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    if args.parameter_scope == "decoder":
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.") and parameter.requires_grad
        ]
    else:
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    parameters = [parameter for _, parameter in named_parameters]
    records: list[dict] = []
    temporal: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()

    for batch_index, batch in enumerate(data_loader):
        if batch_index >= args.max_batches:
            break
        set_seed(args.seed + batch_index)
        model_inputs, _, ego_future, _, _, scene_names = prepare_nuplan_batch(
            batch,
            device,
            config.observation_normalizer,
            with_metadata=True,
        )
        increment_loss, model_prediction, predicted_state, diffusion_time = forward_losses(
            model,
            model_inputs,
            ego_future,
            config,
            args.eps,
        )
        increment_grads = torch.autograd.grad(
            increment_loss,
            [model_prediction, *parameters],
            retain_graph=True,
            allow_unused=True,
        )
        increment_output_grad = increment_grads[0]
        increment_parameter_grads = increment_grads[1:]
        increment_output_norm = vector_norm([increment_output_grad])
        increment_parameter_norm = vector_norm(increment_parameter_grads)
        temporal["increment"].append(temporal_gradient_norm(increment_output_grad))

        record = {
            "batch_index": batch_index,
            "scene_count": len(scene_names),
            "scene_names": scene_names,
            "diffusion_time_mean": float(diffusion_time.detach().mean().cpu()),
            "increment_loss": float(increment_loss.detach().cpu()),
            "increment_output_gradient_norm": increment_output_norm,
            "increment_parameter_gradient_norm": increment_parameter_norm,
            "windows": {},
        }
        for window_size in args.window_sizes:
            predicted_waypoints = decode_trajectory_representation(
                predicted_state,
                getattr(config, "trajectory_representation", "increment"),
                detach_window_size=window_size,
            )
            waypoint_loss = torch.sum(
                (predicted_waypoints[..., :2] - ego_future[..., :2]).square(),
                dim=-1,
            ).mean()
            waypoint_grads = torch.autograd.grad(
                waypoint_loss,
                [model_prediction, *parameters],
                retain_graph=window_size != args.window_sizes[-1],
                allow_unused=True,
            )
            waypoint_output_grad = waypoint_grads[0]
            waypoint_parameter_grads = waypoint_grads[1:]
            raw_output_norm = vector_norm([waypoint_output_grad])
            raw_parameter_norm = vector_norm(waypoint_parameter_grads)
            weighted_output_norm = args.hybrid_weight * raw_output_norm
            weighted_parameter_norm = args.hybrid_weight * raw_parameter_norm
            output_cosine = vector_cosine(
                [increment_output_grad],
                [waypoint_output_grad],
            )
            parameter_cosine = vector_cosine(
                increment_parameter_grads,
                waypoint_parameter_grads,
            )
            key = str(window_size)
            temporal[f"waypoint_weighted_window_{key}"].append(
                args.hybrid_weight * temporal_gradient_norm(waypoint_output_grad)
            )
            record["windows"][key] = {
                "waypoint_loss": float(waypoint_loss.detach().cpu()),
                "raw_output_gradient_norm": raw_output_norm,
                "weighted_output_gradient_norm": weighted_output_norm,
                "weighted_to_increment_output_gradient_ratio": (
                    weighted_output_norm / increment_output_norm
                    if increment_output_norm > 0
                    else float("nan")
                ),
                "output_gradient_cosine": output_cosine,
                "raw_parameter_gradient_norm": raw_parameter_norm,
                "weighted_parameter_gradient_norm": weighted_parameter_norm,
                "weighted_to_increment_parameter_gradient_ratio": (
                    weighted_parameter_norm / increment_parameter_norm
                    if increment_parameter_norm > 0
                    else float("nan")
                ),
                "parameter_gradient_cosine": parameter_cosine,
            }
            del waypoint_grads
        records.append(record)
        del increment_grads
        print(
            f"batch={batch_index + 1}/{min(args.max_batches, len(data_loader))} "
            f"scenes={len(scene_names)} increment_loss={record['increment_loss']:.6f}"
        )

    if not records:
        raise RuntimeError("no batches were audited")

    aggregate = {
        "batch_count": len(records),
        "scene_observations": int(sum(record["scene_count"] for record in records)),
        "increment_loss": summarize(record["increment_loss"] for record in records),
        "increment_output_gradient_norm": summarize(
            record["increment_output_gradient_norm"] for record in records
        ),
        "increment_parameter_gradient_norm": summarize(
            record["increment_parameter_gradient_norm"] for record in records
        ),
        "windows": {},
    }
    for window_size in args.window_sizes:
        key = str(window_size)
        fields = records[0]["windows"][key].keys()
        aggregate["windows"][key] = {
            field: summarize(record["windows"][key][field] for record in records)
            for field in fields
        }

    temporal_summary = {}
    for name, arrays in temporal.items():
        stacked = np.stack(arrays, axis=0)
        mean = stacked.mean(axis=0)
        temporal_summary[name] = {
            "mean_by_timestep": mean.tolist(),
            "first_10_mean": float(mean[:10].mean()),
            "middle_10_mean": float(mean[35:45].mean()),
            "last_10_mean": float(mean[-10:].mean()),
            "first_to_last_ratio": (
                float(mean[:10].mean() / mean[-10:].mean())
                if mean[-10:].mean() > 0
                else float("nan")
            ),
        }
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "aggregate": aggregate,
        "temporal_output_gradient_xy": temporal_summary,
        "batch_records": records,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.num_workers < 0 or args.max_batches < 1:
        raise ValueError("batch-size/max-batches must be positive; num-workers non-negative")
    if args.hybrid_weight < 0 or not 0 < args.eps < 1:
        raise ValueError("hybrid-weight must be non-negative and eps must be in (0, 1)")
    if len(set(args.window_sizes)) != len(args.window_sizes):
        raise ValueError("window-sizes must be unique")
    if any(window < 0 for window in args.window_sizes):
        raise ValueError("window-sizes must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_paths = parse_named_paths(args.model)
    model_args_paths = parse_named_paths(args.model_args)
    unknown = set(model_args_paths) - set(model_paths)
    if unknown:
        raise ValueError(f"--model-args contains unknown models: {sorted(unknown)}")

    default_config = Config(args.args_file)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        default_config.agent_num,
        default_config.predicted_neighbor_num,
        default_config.future_len,
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
    device = torch.device(args.device)
    results = {}
    checkpoint_metadata = {}
    args_metadata = {}
    for name, checkpoint_path in model_paths.items():
        args_path = model_args_paths.get(name, args.args_file).resolve()
        config = Config(args_path)
        model, metadata = load_model(config, checkpoint_path, device)
        results[name] = audit_model(model, data_loader, config, args, device)
        checkpoint_metadata[name] = metadata
        args_metadata[name] = {
            "path": str(args_path),
            "sha256": sha256(args_path),
            "trajectory_representation": getattr(
                config, "trajectory_representation", "increment"
            ),
            "state_normalizer": config.state_normalizer.to_dict(),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "purpose": (
            "diagnose hybrid-loss gradient scale, direction conflict, and temporal "
            "imbalance without parameter updates"
        ),
        "protocol": {
            "dataset_size": len(dataset),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "seed_formula": "seed + batch_index",
            "seed": args.seed,
            "hybrid_weight": args.hybrid_weight,
            "window_sizes": args.window_sizes,
            "parameter_scope": args.parameter_scope,
            "no_parameter_updates": True,
        },
        "inputs": {
            "args_file": str(args.args_file.resolve()),
            "args_file_sha256": sha256(args.args_file),
            "per_model_args": args_metadata,
            "data_dir": str(args.data_dir.resolve()),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
            "checkpoints": checkpoint_metadata,
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved audit to {args.output}")


if __name__ == "__main__":
    main()
