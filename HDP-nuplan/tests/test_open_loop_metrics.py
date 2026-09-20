import math

import pytest
import torch

from hdp_nuplan.utils.open_loop_metrics import trajectory_open_loop_metrics


def _straight_trajectory(batch=1, group=1, steps=80, step_distance=0.1):
    trajectory = torch.zeros(batch, group, steps, 4)
    trajectory[..., 0] = (
        torch.arange(1, steps + 1, dtype=torch.float32) * step_distance
    )
    trajectory[..., 2] = 1.0
    return trajectory


def test_open_loop_metrics_report_full_and_named_horizons():
    prediction = _straight_trajectory()
    target = prediction[:, 0].clone()
    prediction[..., 1] = 1.0
    ego_state = torch.zeros(1, 10)
    ego_state[:, 4] = 1.0

    metrics = trajectory_open_loop_metrics(
        prediction,
        target,
        ego_state,
        dt=0.1,
        horizons_seconds=(1.0, 3.0, 5.0, 8.0),
    )

    for horizon in ("1s", "3s", "5s", "8s"):
        assert metrics[f"ade_{horizon}_m"] == pytest.approx(torch.tensor([[1.0]]))
        assert metrics[f"fde_{horizon}_m"] == pytest.approx(torch.tensor([[1.0]]))
    assert metrics["ade_m"] == pytest.approx(torch.tensor([[1.0]]))
    assert metrics["fde_m"] == pytest.approx(torch.tensor([[1.0]]))

    # Evaluate kinematics on the unshifted path: a constant lateral offset at
    # the first future point correctly creates a large initial displacement.
    motion_metrics = trajectory_open_loop_metrics(
        target[:, None],
        target,
        ego_state,
        dt=0.1,
        horizons_seconds=(1.0,),
    )
    assert motion_metrics["mean_speed_mps"] == pytest.approx(torch.tensor([[1.0]]))


def test_heading_error_uses_wrapped_direction_difference():
    prediction = _straight_trajectory(steps=10)
    target = prediction[:, 0].clone()
    angle = math.radians(179.0)
    target[..., 2] = math.cos(angle)
    target[..., 3] = math.sin(angle)
    prediction[..., 2] = math.cos(-angle)
    prediction[..., 3] = math.sin(-angle)
    ego_state = torch.zeros(1, 10)

    metrics = trajectory_open_loop_metrics(
        prediction,
        target,
        ego_state,
        dt=0.1,
        horizons_seconds=(1.0,),
    )

    expected = math.radians(2.0)
    assert metrics["heading_mae_rad"].item() == pytest.approx(expected, abs=1e-5)
    assert metrics["heading_fde_rad"].item() == pytest.approx(expected, abs=1e-5)


def test_open_loop_metrics_reject_horizon_beyond_available_trajectory():
    prediction = _straight_trajectory(steps=20)
    target = prediction[:, 0].clone()
    ego_state = torch.zeros(1, 10)

    with pytest.raises(ValueError, match="exceeds available duration"):
        trajectory_open_loop_metrics(
            prediction,
            target,
            ego_state,
            dt=0.1,
            horizons_seconds=(3.0,),
        )


def test_open_loop_metrics_keep_candidate_dimension():
    prediction = _straight_trajectory(group=2, steps=10)
    prediction[:, 1, :, 1] = 2.0
    target = prediction[:, 0].clone()
    ego_state = torch.zeros(1, 10)

    metrics = trajectory_open_loop_metrics(
        prediction,
        target,
        ego_state,
        dt=0.1,
        horizons_seconds=(1.0,),
    )

    assert all(value.shape == (1, 2) for value in metrics.values())
    assert metrics["ade_m"][0, 0] == pytest.approx(0.0)
    assert metrics["ade_m"][0, 1] == pytest.approx(2.0)
