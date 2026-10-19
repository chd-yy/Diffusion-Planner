import pytest
import torch

from hdp_nuplan.utils.traj_kinematics import (
    decode_trajectory_representation,
    encode_trajectory_representation,
)


def example_trajectory():
    return torch.tensor(
        [[[1.0, 2.0, 1.0, 0.0], [3.5, 1.0, 0.0, 1.0], [4.0, 4.0, -1.0, 0.0]]]
    )


def test_absolute_representation_is_identity():
    trajectory = example_trajectory()
    encoded = encode_trajectory_representation(trajectory, "absolute")
    decoded = decode_trajectory_representation(encoded, "absolute")
    assert torch.equal(encoded, trajectory)
    assert torch.equal(decoded, trajectory)


def test_increment_representation_round_trip_preserves_heading():
    trajectory = example_trajectory()
    encoded = encode_trajectory_representation(trajectory, "increment")
    expected_xy = torch.tensor([[[1.0, 2.0], [2.5, -1.0], [0.5, 3.0]]])
    assert torch.equal(encoded[..., :2], expected_xy)
    assert torch.equal(encoded[..., 2:], trajectory[..., 2:])
    decoded = decode_trajectory_representation(encoded, "increment")
    assert torch.allclose(decoded, trajectory)


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="trajectory_representation"):
        encode_trajectory_representation(example_trajectory(), "velocity")
