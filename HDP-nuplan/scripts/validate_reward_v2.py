#!/usr/bin/env python3
"""在真实 NuPlan NPZ 上验证 reward v2 对典型轨迹扰动的排序。"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer


CANDIDATE_NAMES = ("expert", "stop", "lateral", "reverse", "jitter", "collision")


def _trajectory_from_pose(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [pose[..., :2], pose[..., 2:3].cos(), pose[..., 2:3].sin()], dim=-1
    )


def _build_candidates(
    ego_future: torch.Tensor, neighbors_future: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    expert = _trajectory_from_pose(ego_future)
    horizon = expert.shape[0]
    ramp = torch.linspace(0.0, 1.0, horizon, dtype=expert.dtype)[:, None]

    stop = torch.zeros_like(expert)
    stop[:, 2] = 1.0

    lateral = expert.clone()
    lateral[:, 1:2] += 3.0 * ramp

    reverse = expert.clone()
    reverse[:, :2] *= -1
    reverse[:, 2:] *= -1

    jitter = expert.clone()
    alternating = torch.where(
        torch.arange(horizon) % 2 == 0,
        torch.ones(horizon),
        -torch.ones(horizon),
    ).to(expert)
    jitter[:, 1] += 0.25 * alternating

    collision = expert.clone()
    collision_available = False
    if neighbors_future.shape[0] > 0:
        valid = torch.any(neighbors_future[..., :3] != 0, dim=-1)
        best_neighbor = valid.sum(dim=-1).argmax()
        best_valid = valid[best_neighbor]
        if torch.any(best_valid):
            neighbor_pose = neighbors_future[best_neighbor]
            neighbor_trajectory = _trajectory_from_pose(neighbor_pose)
            collision[best_valid] = neighbor_trajectory[best_valid]
            collision_available = True

    return torch.stack([expert, stop, lateral, reverse, jitter, collision]), collision_available


def _score_npz(path: Path, scorer: NuPlanTensorRewardScorer) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ego_future_raw = torch.from_numpy(data["ego_agent_future"]).float()
        neighbors_raw = torch.from_numpy(data["neighbor_agents_future"][:10]).float()
        neighbor_past = torch.from_numpy(data["neighbor_agents_past"][:10]).float()[None]
        route_lanes = torch.from_numpy(data["route_lanes"]).float()[None]
        static_objects = torch.from_numpy(data["static_objects"]).float()[None]
        ego_current_state = torch.from_numpy(data["ego_current_state"]).float()[None]

    candidates, collision_available = _build_candidates(ego_future_raw, neighbors_raw)
    neighbors_future = _trajectory_from_pose(neighbors_raw)[None]
    neighbor_mask = torch.sum(torch.ne(neighbors_raw[..., :3], 0), dim=-1) == 0
    neighbors_future[0, neighbor_mask] = 0

    rewards, details = scorer(
        trajectories=candidates[None],
        neighbors_future=neighbors_future,
        neighbor_mask=neighbor_mask[None],
        route_lanes=route_lanes,
        static_objects=static_objects,
        ego_future=candidates[None, 0],
        neighbor_agents_past=neighbor_past,
        ego_current_state=ego_current_state,
    )

    values = {
        key: {name: float(value[0, idx]) for idx, name in enumerate(CANDIDATE_NAMES)}
        for key, value in details.items()
    }
    # 近乎静止的专家轨迹可能对应红灯、拥堵或停车场景，不应强制要求其
    # 优于停车，也无法用坐标取反构造有意义的逆行对照。
    moving_scene = bool(torch.linalg.vector_norm(candidates[0, -1, :2]) >= 1.0)
    checks = {
        "lateral_increases_route_cost": (
            values["route_cost"]["lateral"] > values["route_cost"]["expert"] + 1e-6
        ),
        "jitter_increases_comfort_cost": (
            values["comfort_cost"]["jitter"] > values["comfort_cost"]["expert"] + 1e-6
        ),
        "direction_metrics_are_finite_and_bounded": bool(
            torch.isfinite(details["direction_cost"]).all()
            and torch.isfinite(details["motion_alignment"]).all()
            and torch.isfinite(details["heading_alignment"]).all()
            and torch.isfinite(details["reverse_fraction"]).all()
            and torch.isfinite(details["min_progress_in_1s"]).all()
            and torch.isfinite(details["direction_compliance_score_approx"]).all()
            and torch.all(
                (details["direction_cost"] >= 0)
                & (details["direction_cost"] <= 1)
            )
            and torch.all(
                (details["reverse_fraction"] >= 0)
                & (details["reverse_fraction"] <= 1)
            )
        ),
    }
    if moving_scene:
        checks["reverse_reduces_progress"] = (
            values["progress"]["reverse"] < values["progress"]["expert"] - 1e-6
        )
        checks["reverse_increases_backward_cost"] = (
            values["backward_cost"]["reverse"]
            > values["backward_cost"]["expert"] + 1e-6
        )
        checks["reverse_increases_direction_cost"] = (
            values["direction_cost"]["reverse"]
            > values["direction_cost"]["expert"] + 1e-6
        )
        checks["reverse_reduces_motion_alignment"] = (
            values["motion_alignment"]["reverse"]
            < values["motion_alignment"]["expert"] - 1e-6
        )
        checks["reverse_reduces_heading_alignment"] = (
            values["heading_alignment"]["reverse"]
            < values["heading_alignment"]["expert"] - 1e-6
        )
        checks["reverse_reduces_min_progress_in_1s"] = (
            values["min_progress_in_1s"]["reverse"]
            < values["min_progress_in_1s"]["expert"] - 1e-6
        )
        checks["reverse_reduces_direction_compliance_score_approx"] = (
            values["direction_compliance_score_approx"]["reverse"]
            < values["direction_compliance_score_approx"]["expert"] - 1e-6
        )
        checks["moving_expert_beats_stop"] = (
            values["reward"]["expert"] > values["reward"]["stop"]
        )
    if collision_available:
        checks["collision_increases_collision_cost"] = (
            values["collision_cost"]["collision"]
            > values["collision_cost"]["expert"] + 1e-6
        )

    return {
        "file": path.name,
        "moving_scene": moving_scene,
        "collision_available": collision_available,
        "values": values,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-scenes", default=100, type=int)
    parser.add_argument("--minimum-pass-rate", default=0.8, type=float)
    parser.add_argument("--progress-guard-weight", default=0.0, type=float)
    parser.add_argument("--direction-guard-weight", default=0.0, type=float)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    names: List[str] = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = names[: args.max_scenes]
    scorer = NuPlanTensorRewardScorer(
        NuPlanRewardConfig(
            progress_guard_weight=args.progress_guard_weight,
            direction_guard_weight=args.direction_guard_weight,
        )
    )
    scene_results = [_score_npz(args.cache_dir / name, scorer) for name in names]

    check_values: Dict[str, List[bool]] = {}
    metric_values: Dict[str, Dict[str, List[float]]] = {}
    for result in scene_results:
        for key, passed in result["checks"].items():
            check_values.setdefault(key, []).append(bool(passed))
        for metric, candidates in result["values"].items():
            for candidate, value in candidates.items():
                metric_values.setdefault(metric, {}).setdefault(candidate, []).append(value)

    check_summary = {
        key: {
            "passed": sum(values),
            "total": len(values),
            "pass_rate": sum(values) / len(values),
        }
        for key, values in check_values.items()
    }
    metric_summary = {
        metric: {
            candidate: float(np.mean(values))
            for candidate, values in candidates.items()
        }
        for metric, candidates in metric_values.items()
    }
    accepted = all(
        summary["pass_rate"] >= args.minimum_pass_rate
        for summary in check_summary.values()
    )
    report = {
        "schema_version": 1,
        "scene_count": len(scene_results),
        "minimum_pass_rate": args.minimum_pass_rate,
        "progress_guard_weight": args.progress_guard_weight,
        "direction_guard_weight": args.direction_guard_weight,
        "accepted": accepted,
        "checks": check_summary,
        "metric_means": metric_summary,
        "scenes": scene_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"accepted": accepted, "checks": check_summary}, indent=2))
    if args.strict and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
