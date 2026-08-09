from pathlib import Path
import sys

import pytest
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from compare_checkpoint_behavior import (  # noqa: E402
    _correlation,
    group_diversity_metrics,
    paired_summary,
    trajectory_behavior_metrics,
)


def test_trajectory_behavior_metrics_distinguish_moving_and_stopped_paths():
    horizon = 4
    trajectories = torch.zeros(1, 2, horizon, 4)
    trajectories[..., 2] = 1.0
    trajectories[0, 0, :, 0] = torch.arange(1, horizon + 1) * 0.1
    ego_future = trajectories[:, 0].clone()
    ego_state = torch.zeros(1, 10)
    ego_state[:, 4] = 1.0

    metrics = trajectory_behavior_metrics(
        trajectories,
        ego_future,
        ego_state,
        dt=0.1,
        low_speed_threshold=0.5,
    )

    assert metrics["mean_speed_mps"][0, 0] == pytest.approx(1.0)
    assert metrics["mean_speed_mps"][0, 1] == pytest.approx(0.0)
    assert metrics["stationary_trajectory_fraction"][0, 0] == 0
    assert metrics["stationary_trajectory_fraction"][0, 1] == 1
    assert metrics["ade_m"][0, 0] == pytest.approx(0.0)
    assert metrics["fde_m"][0, 1] == pytest.approx(0.4)


def test_paired_summary_uses_metric_optimization_direction():
    supervised = {
        "progress": [1.0, 2.0],
        "route_cost": [0.2, 0.4],
    }
    rl = {
        "progress": [2.0, 1.0],
        "route_cost": [0.1, 0.5],
    }

    summary = paired_summary(supervised, rl)

    assert summary["progress"]["mean"] == pytest.approx(0.0)
    assert summary["progress"]["improved_fraction"] == pytest.approx(0.5)
    assert summary["route_cost"]["improved_fraction"] == pytest.approx(0.5)


def test_group_diversity_metrics_measure_reward_and_endpoint_spread():
    trajectories = torch.zeros(1, 2, 3, 4)
    trajectories[0, 1, -1, 0] = 2.0
    metrics = {
        "reward": torch.tensor([[1.0, 3.0]]),
        "progress": torch.tensor([[0.5, 1.5]]),
        "path_length_m": torch.tensor([[1.0, 3.0]]),
        "collision_cost": torch.tensor([[0.2, 0.1]]),
    }

    diversity = group_diversity_metrics(trajectories, metrics)

    assert diversity["group_reward_std"] == pytest.approx(torch.tensor([1.0]))
    assert diversity["group_reward_range"] == pytest.approx(torch.tensor([2.0]))
    assert diversity["group_progress_std"] == pytest.approx(torch.tensor([0.5]))
    assert diversity["group_endpoint_diversity_m"] == pytest.approx(
        torch.tensor([2.0])
    )
    assert diversity["group_reward_progress_correlation"] == pytest.approx(
        torch.tensor([1.0])
    )
    assert diversity["best_reward_progress_delta"] == pytest.approx(
        torch.tensor([0.5])
    )
    assert diversity["best_reward_path_length_delta_m"] == pytest.approx(
        torch.tensor([1.0])
    )
    assert diversity["best_reward_collision_cost_delta"] == pytest.approx(
        torch.tensor([-0.05])
    )


def test_correlation_rejects_different_statistical_levels():
    assert _correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0]) is None
