#!/usr/bin/env python3
"""Recompute paired C0/B10 open-loop metrics with log-clustered intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from scripts.analyze_chapter3_representation_ablation import (
    clustered_interval,
    evaluate,
    load_log_names,
)
from scripts.evaluate_supervised_open_loop import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", required=True, type=Path)
    parser.add_argument("--c0-checkpoint", required=True, type=Path)
    parser.add_argument("--b10-checkpoint", required=True, type=Path)
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
    parser.add_argument("--bootstrap-iterations", default=10_000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = Config(args.args_file)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        config.agent_num,
        config.predicted_neighbor_num,
        config.future_len,
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    b10 = evaluate(
        "historical_B10",
        args.args_file,
        args.b10_checkpoint,
        loader,
        device,
        args,
    )
    c0 = evaluate(
        "C0",
        args.args_file,
        args.c0_checkpoint,
        loader,
        device,
        args,
    )
    if b10["scenes"] != c0["scenes"]:
        raise RuntimeError("paired scene order changed between B10 and C0")
    if not np.array_equal(b10["targets"], c0["targets"]):
        raise RuntimeError("paired expert targets changed between B10 and C0")

    log_names = load_log_names(args.data_dir, b10["scenes"])
    summary = {
        "schema_version": 1,
        "delta_definition": "historical_B10 minus C0",
        "protocol": {
            "dataset_samples": len(dataset),
            "repeats": args.repeats,
            "observations_per_model": len(dataset) * args.repeats,
            "seed": args.seed,
            "diffusion_steps": args.diffusion_steps,
            "sampling_noise_scale": args.sampling_noise_scale,
            "paired_common_random_numbers": True,
            "cluster_unit": "nuPlan log_name",
            "cluster_count": len(set(log_names)),
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "inputs": {
            "args_file": str(args.args_file.resolve()),
            "args_file_sha256": sha256(args.args_file),
            "c0_checkpoint": c0["checkpoint"],
            "b10_checkpoint": b10["checkpoint"],
            "data_dir": str(args.data_dir.resolve()),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
        },
        "metrics": {},
    }
    npz_payload: dict[str, np.ndarray] = {
        "scene_names": np.asarray(b10["scenes"]),
        "log_names": np.asarray(log_names),
        "historical_b10_trajectories": b10["trajectories"],
        "c0_trajectories": c0["trajectories"],
        "expert_trajectories": b10["targets"],
        "ego_current_states": b10["current_states"],
    }
    for metric in sorted(b10["metrics"]):
        b10_values = b10["metrics"][metric]
        c0_values = c0["metrics"][metric]
        if b10_values.shape != c0_values.shape:
            raise RuntimeError(f"shape mismatch for {metric}")
        delta = b10_values - c0_values
        per_scene_delta = delta.mean(axis=0)
        summary["metrics"][metric] = {
            "c0_mean": float(c0_values.mean()),
            "historical_b10_mean": float(b10_values.mean()),
            "b10_minus_c0_mean": float(delta.mean()),
            "log_cluster_bootstrap95": clustered_interval(
                per_scene_delta,
                log_names,
                seed=args.seed,
                iterations=args.bootstrap_iterations,
            ),
        }
        npz_payload[f"c0__{metric}"] = c0_values
        npz_payload[f"historical_b10__{metric}"] = b10_values

    np.savez_compressed(args.output_dir / "paired_per_scene_metrics.npz", **npz_payload)
    (args.output_dir / "clustered_intervals.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved C0/B10 clustered intervals to {args.output_dir}")


if __name__ == "__main__":
    main()
