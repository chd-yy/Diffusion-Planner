#!/usr/bin/env python3
"""在真实 NuPlan NPZ 上验证论文 risk/follow/lane reward 的数值和排序。"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer


CANDIDATE_NAMES = ("expert", "stop", "lateral", "slow", "jitter", "collision")
BOUNDED_METRICS = (
    "risk_reward",
    "safety_reward",
    "ttc_reward",
    "thw_reward",
    "occupancy_reward",
    "follow_reward",
    "lane_reward",
)


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

    slow = expert.clone()
    slow[:, :2] *= 0.7

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
            collision[best_valid] = _trajectory_from_pose(
                neighbors_future[best_neighbor]
            )[best_valid]
            collision_available = True

    return torch.stack([expert, stop, lateral, slow, jitter, collision]), collision_available


def _score_npz(path: Path, scorer: NuPlanTensorRewardScorer) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ego_future_raw = torch.from_numpy(data["ego_agent_future"]).float()
        neighbors_raw = torch.from_numpy(data["neighbor_agents_future"]).float()
        neighbor_past = torch.from_numpy(data["neighbor_agents_past"]).float()[None]
        lanes = torch.from_numpy(data["lanes"]).float()[None]
        route_lanes = torch.from_numpy(data["route_lanes"]).float()[None]
        static_objects = torch.from_numpy(data["static_objects"]).float()[None]
        ego_current_state = torch.from_numpy(data["ego_current_state"]).float()[None]

    candidates, collision_available = _build_candidates(
        ego_future_raw, neighbors_raw
    )
    neighbors_future = _trajectory_from_pose(neighbors_raw)[None]
    neighbor_mask = torch.sum(torch.ne(neighbors_raw[..., :3], 0), dim=-1) == 0
    neighbors_future[0, neighbor_mask] = 0

    _, details = scorer(
        trajectories=candidates[None],
        neighbors_future=neighbors_future,
        neighbor_mask=neighbor_mask[None],
        route_lanes=route_lanes,
        static_objects=static_objects,
        ego_future=candidates[None, 0],
        neighbor_agents_past=neighbor_past,
        ego_current_state=ego_current_state,
        lanes=lanes,
    )
    values = {
        key: {name: float(value[0, idx]) for idx, name in enumerate(CANDIDATE_NAMES)}
        for key, value in details.items()
    }

    moving_scene = bool(torch.linalg.vector_norm(candidates[0, -1, :2]) >= 1.0)
    lane_active = bool(details["lane_reward_mask"][0, 0] > 0.5)
    checks = {
        "all_reward_components_finite": all(
            torch.isfinite(details[key]).all().item() for key in BOUNDED_METRICS
        ),
        "all_reward_components_bounded": all(
            ((details[key] >= 0) & (details[key] <= 1)).all().item()
            for key in BOUNDED_METRICS
        ),
    }
    if lane_active:
        checks["lateral_reduces_lane_reward"] = (
            values["lane_reward"]["lateral"]
            < values["lane_reward"]["expert"] - 1e-6
        )
    if collision_available:
        checks["collision_reduces_risk_reward"] = (
            values["risk_reward"]["collision"]
            < values["risk_reward"]["expert"] - 1e-6
        )
    if moving_scene:
        # 这是防止停车 reward hacking 的诊断门禁，不是论文声明的数学性质。
        checks["moving_expert_beats_stop"] = (
            values["reward"]["expert"] > values["reward"]["stop"]
        )

    return {
        "file": path.name,
        "moving_scene": moving_scene,
        "lane_active": lane_active,
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
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    names: List[str] = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = names[: args.max_scenes]
    scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
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
