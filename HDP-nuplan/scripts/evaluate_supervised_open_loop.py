#!/usr/bin/env python3
"""Evaluate one or more HDP checkpoints with paired open-loop metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.open_loop_metrics import trajectory_open_loop_metrics
from hdp_nuplan.utils.train_utils import set_seed


ERROR_METRICS = ("ade_", "fde_", "heading_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        metavar="NAME=CHECKPOINT",
        help="Repeat for every checkpoint; the first model is the paired reference.",
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
    parser.add_argument("--dt", default=0.1, type=float)
    parser.add_argument(
        "--horizons-seconds",
        default=(1.0, 3.0, 5.0, 8.0),
        nargs="+",
        type=float,
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed = []
    names = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --model {spec!r}; expected NAME=CHECKPOINT")
        name, path_text = spec.split("=", 1)
        name = name.strip()
        path = Path(path_text).expanduser().resolve()
        if not name:
            raise ValueError("model name must not be empty")
        if name in names:
            raise ValueError(f"duplicate model name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        parsed.append((name, path))
    return parsed


def load_model(
    config: Config, checkpoint_path: Path, device: torch.device
) -> tuple[Hyper_Diffusion_Planner, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema_state_dict" in checkpoint:
        source = "ema_state_dict"
    elif "model" in checkpoint:
        source = "model"
    else:
        source = "checkpoint_root"
    state_dict = checkpoint if source == "checkpoint_root" else checkpoint[source]
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model = Hyper_Diffusion_Planner(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256(checkpoint_path),
        "weight_source": source,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "train_loss": checkpoint.get("loss") if isinstance(checkpoint, dict) else None,
    }
    return model, metadata


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def paired_delta_summary(reference: np.ndarray, candidate: np.ndarray, metric: str) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"paired arrays for {metric} have different shapes")
    delta = candidate - reference
    result = {
        **summarize(delta),
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "candidate_higher_fraction": float((delta > 0).mean()),
    }
    if metric.startswith(ERROR_METRICS):
        result["candidate_improved_fraction"] = float((delta < 0).mean())
    return result


@torch.no_grad()
def evaluate_model(
    model: Hyper_Diffusion_Planner,
    data_loader: DataLoader,
    config: Config,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Dict[str, np.ndarray], list[dict]]:
    values: Dict[str, list[float]] = defaultdict(list)
    observations = []
    for repeat in range(args.repeats):
        for batch_index, batch in enumerate(data_loader):
            model_inputs, raw_inputs, ego_future, _, _, scene_names = prepare_nuplan_batch(
                batch,
                device,
                config.observation_normalizer,
                with_metadata=True,
            )
            sample_seed = args.seed + repeat * 1_000_000 + batch_index
            set_seed(sample_seed)
            trajectories = model.sample(
                model_inputs,
                num_samples=args.num_samples,
                diffusion_steps=args.diffusion_steps,
                noise_scale=args.sampling_noise_scale,
            )
            metrics = trajectory_open_loop_metrics(
                trajectories,
                ego_future,
                raw_inputs["ego_current_state"],
                dt=args.dt,
                horizons_seconds=args.horizons_seconds,
            )
            batch_size, group_size = trajectories.shape[:2]
            for metric, tensor in metrics.items():
                values[metric].extend(tensor.detach().float().cpu().reshape(-1).tolist())
            for scene_index, scene_name in enumerate(scene_names):
                for candidate_index in range(group_size):
                    observations.append(
                        {
                            "repeat": repeat,
                            "batch_index": batch_index,
                            "scene_index": scene_index,
                            "candidate_index": candidate_index,
                            "scene": scene_name,
                            "sample_seed": sample_seed,
                        }
                    )
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == len(data_loader):
                print(
                    f"repeat={repeat + 1}/{args.repeats} "
                    f"batch={batch_index + 1}/{len(data_loader)}"
                )
    return {
        metric: np.asarray(metric_values, dtype=np.float64)
        for metric, metric_values in values.items()
    }, observations


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.repeats < 1 or args.num_samples < 1 or args.diffusion_steps < 1:
        raise ValueError("repeats, num-samples and diffusion-steps must be positive")
    if args.sampling_noise_scale <= 0 or args.dt <= 0:
        raise ValueError("sampling-noise-scale and dt must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    for path in (args.args_file, args.data_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_specs = parse_model_specs(args.model)
    config = Config(args.args_file)
    device = torch.device(args.device)

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

    arrays_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    metadata_by_model = {}
    reference_observations = None
    for model_name, checkpoint_path in model_specs:
        print(f"Evaluating {model_name}: {checkpoint_path}")
        model, checkpoint_metadata = load_model(config, checkpoint_path, device)
        arrays, observations = evaluate_model(model, data_loader, config, device, args)
        if reference_observations is None:
            reference_observations = observations
        elif observations != reference_observations:
            raise RuntimeError("observation order changed between paired model evaluations")
        arrays_by_model[model_name] = arrays
        metadata_by_model[model_name] = checkpoint_metadata
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_summaries = {
        model_name: {metric: summarize(values) for metric, values in arrays.items()}
        for model_name, arrays in arrays_by_model.items()
    }
    reference_name = model_specs[0][0]
    paired = {}
    for model_name, _ in model_specs[1:]:
        paired[model_name] = {
            metric: paired_delta_summary(
                arrays_by_model[reference_name][metric],
                arrays_by_model[model_name][metric],
                metric,
            )
            for metric in arrays_by_model[reference_name]
        }

    result = {
        "schema_version": 1,
        "metric_contract": {
            "trajectory_input": "[x,y,cos(yaw),sin(yaw)] in ego-local physical units",
            "sampler_output_is_already_integrated": True,
            "ade": "mean Euclidean xy error through the named horizon",
            "fde": "Euclidean xy error at the named horizon",
            "heading": "wrapped angular error from normalized cos/sin direction",
            "kinematics": "descriptive finite-difference quantities, not feasibility guarantees",
        },
        "protocol": {
            "paired_reference": reference_name,
            "common_random_numbers": True,
            "seed_formula": "seed + repeat * 1000000 + batch_index",
            "dataset_samples": len(dataset),
            "observations_per_model": len(reference_observations or []),
            "repeats": args.repeats,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "diffusion_steps": args.diffusion_steps,
            "sampling_noise_scale": args.sampling_noise_scale,
            "dt_seconds": args.dt,
            "horizons_seconds": args.horizons_seconds,
        },
        "inputs": {
            "args_file": str(args.args_file.resolve()),
            "args_file_sha256": sha256(args.args_file),
            "data_dir": str(args.data_dir.resolve()),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
        },
        "checkpoints": metadata_by_model,
        "models": model_summaries,
        "paired_delta_candidate_minus_reference": paired,
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved open-loop evaluation to {args.output}")


if __name__ == "__main__":
    main()
