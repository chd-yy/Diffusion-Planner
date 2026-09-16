import math

import numpy as np
import pytest
import torch

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer


def scorer():
    v = get_pacifica_parameters()
    return NuPlanTensorRewardScorer(NuPlanRewardConfig(
        ego_width=v.width, ego_length=v.length, ego_rear_axle_to_center=v.rear_axle_to_center))


def separation(reward, ego, obj, width=.6, length=.6):
    return reward._rectangle_signed_separation(
        torch.tensor([[[[ego.x, ego.y]]]], dtype=torch.float64),
        torch.tensor([[[[math.cos(ego.heading), math.sin(ego.heading)]]]], dtype=torch.float64),
        torch.tensor([[[[obj.x, obj.y]]]], dtype=torch.float64),
        torch.tensor([[[[math.cos(obj.heading), math.sin(obj.heading)]]]], dtype=torch.float64),
        torch.tensor([[width]], dtype=torch.float64), torch.tensor([[length]], dtype=torch.float64),
    ).item()


def test_front_collision_missed_by_legacy_center_assumption():
    ego, pedestrian = StateSE2(0, 0, 0), StateSE2(3.5, 0, 0)
    assert separation(NuPlanTensorRewardScorer(NuPlanRewardConfig()), ego, pedestrian) > 0
    assert separation(scorer(), ego, pedestrian) < 0
    assert CarFootprint.build_from_rear_axle(ego, get_pacifica_parameters()).geometry.intersects(
        OrientedBox(pedestrian, .6, .6, 1.7).geometry)


def test_rear_false_positive_is_removed_not_just_inflated():
    ego, pedestrian = StateSE2(0, 0, 0), StateSE2(-2, 0, 0)
    assert separation(NuPlanTensorRewardScorer(NuPlanRewardConfig()), ego, pedestrian) < 0
    assert separation(scorer(), ego, pedestrian) > 0


def test_geometry_matches_official_boxes_for_rotated_random_poses():
    rng = np.random.default_rng(42)
    reward = scorer()
    for _ in range(200):
        ego = StateSE2(*rng.uniform(-4, 4, 3))
        obj = StateSE2(*rng.uniform(-4, 4, 3))
        width, length = rng.uniform(.3, 5, 2)
        official = CarFootprint.build_from_rear_axle(ego, get_pacifica_parameters()).geometry.intersects(
            OrientedBox(obj, length, width, 1.7).geometry)
        assert (separation(reward, ego, obj, width, length) <= 0) == official


def test_collision_risk_and_following_share_corrected_frame():
    reward = scorer()
    ego = torch.zeros(1, 1, 2, 4)
    ego[..., 2] = 1
    neighbors = torch.zeros(1, 1, 2, 4)
    neighbors[..., 0] = 3.5
    neighbors[..., 2] = 1
    past = torch.zeros(1, 1, 1, 11)
    past[..., 0] = 3.5
    past[..., 2] = 1
    past[..., 6:8] = .6
    mask = torch.zeros(1, 1, 2, dtype=torch.bool)
    geometry = reward._neighbor_geometry(ego, neighbors, mask, past)
    assert geometry["longitudinal"][0, 0, 0, 0].item() == pytest.approx(3.5 - 1.461)
    _, no_collision = reward._collision_cost(ego, neighbors, mask, None, past)
    assert no_collision.item() == 0
    risk = reward._risk_reward(ego, neighbors, mask, None, past, None)
    assert risk["min_ttc_seconds"].item() == 0
    assert risk["risk_reward"].item() == 0


def test_static_collision_uses_same_rear_axle_offset():
    reward = scorer()
    ego = torch.zeros(1, 1, 2, 4)
    ego[..., 2] = 1
    static = torch.zeros(1, 1, 10)
    static[..., 0] = 3.5
    static[..., 2] = 1
    static[..., 4:6] = .6
    neighbors = torch.zeros(1, 0, 2, 4)
    mask = torch.zeros(1, 0, 2, dtype=torch.bool)
    _, no_collision = reward._collision_cost(ego, neighbors, mask, static)
    assert no_collision.item() == 0
    assert reward._risk_reward(ego, neighbors, mask, static, None, None)["risk_reward"].item() == 0
