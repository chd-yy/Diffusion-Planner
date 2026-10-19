import torch

from hdp_nuplan.rl.candidate_selection import (
    gather_candidate_details,
    gather_candidate_tensor,
    quality_diverse_indices,
    safety_diverse_reward_indices,
    safety_reward_spread_indices,
    stratified_safety_diverse_indices,
    weighted_quality_dispersion,
)


def test_selection_keeps_all_safe_candidates_and_fills_low_noise_first():
    reward = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.7]])
    eligible = torch.tensor([[False, True, False, True, False]])
    scales = (0.1, 1.0, 0.1, 1.0, 0.5)
    indices = safety_reward_spread_indices(reward, eligible, scales, 4)
    assert set(indices[0].tolist()) >= {1, 3}
    assert 0 in indices[0].tolist()
    assert 2 in indices[0].tolist()


def test_selection_spans_safe_reward_ranks_when_pool_is_full():
    reward = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    eligible = torch.ones_like(reward, dtype=torch.bool)
    indices = safety_reward_spread_indices(reward, eligible, (0.5,) * 8, 4)
    assert indices.tolist() == [[0, 2, 5, 7]]


def test_candidate_gather_handles_trajectory_and_reward_tensors():
    trajectories = torch.arange(2 * 4 * 3 * 2).reshape(2, 4, 3, 2)
    indices = torch.tensor([[0, 2], [1, 3]])
    selected = gather_candidate_tensor(trajectories, indices)
    assert selected.shape == (2, 2, 3, 2)
    torch.testing.assert_close(selected[0, 1], trajectories[0, 2])
    details = {
        "reward": torch.arange(8).reshape(2, 4),
        "scene_value": torch.tensor([1.0, 2.0]),
    }
    gathered = gather_candidate_details(details, indices, pool_size=4)
    assert gathered["reward"].tolist() == [[0, 2], [5, 7]]
    torch.testing.assert_close(gathered["scene_value"], details["scene_value"])


def test_safety_diverse_selection_keeps_reward_extremes_and_geometry():
    trajectories = torch.zeros(1, 5, 2, 4)
    trajectories[0, :, :, 0] = torch.tensor([0.0, 1.0, 2.0, 8.0, 9.0])[:, None]
    reward = torch.tensor([[0.0, 1.0, 2.0, 8.0, 9.0]])
    eligible = torch.ones_like(reward, dtype=torch.bool)
    indices = safety_diverse_reward_indices(
        trajectories,
        reward,
        eligible,
        (0.1, 0.1, 0.5, 1.0, 1.0),
        3,
    )
    assert 0 in indices[0].tolist()
    assert 4 in indices[0].tolist()
    assert len(set(indices[0].tolist())) == 3


def test_stratified_selection_preserves_temperature_proportions():
    trajectories = torch.zeros(1, 8, 2, 4)
    trajectories[0, :, :, 0] = torch.arange(8, dtype=torch.float32)[:, None]
    reward = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    eligible = torch.ones_like(reward, dtype=torch.bool)
    scales = (0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0)
    indices = stratified_safety_diverse_indices(
        trajectories,
        reward,
        eligible,
        scales,
        target_size=4,
    )
    chosen_scales = [scales[index] for index in indices[0].tolist()]
    assert chosen_scales.count(0.1) == 2
    assert chosen_scales.count(0.5) == 1
    assert chosen_scales.count(1.0) == 1


def test_stratified_selection_accepts_explicit_final_quotas():
    trajectories = torch.zeros(1, 10, 2, 4)
    trajectories[0, :, :, 0] = torch.arange(10, dtype=torch.float32)[:, None]
    reward = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    eligible = torch.ones_like(reward, dtype=torch.bool)
    scales = (0.1,) * 4 + (0.5,) * 4 + (1.5,) * 2
    indices = stratified_safety_diverse_indices(
        trajectories,
        reward,
        eligible,
        scales,
        target_size=5,
        target_noise_scales=(0.1, 0.1, 0.5, 0.5, 1.5),
    )
    chosen_scales = [scales[index] for index in indices[0].tolist()]
    assert chosen_scales.count(0.1) == 2
    assert chosen_scales.count(0.5) == 2
    assert chosen_scales.count(1.5) == 1


def test_quality_selection_preserves_safe_component_extremes_and_quota():
    quality = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [1., 1.], [9., 9.]]])
    reward = torch.tensor([[0., 1., 2., 3., 100.]])
    eligible = torch.tensor([[True, True, True, True, False]])
    indices = quality_diverse_indices(
        quality,
        reward,
        eligible,
        candidate_noise_scales=(0.1,) * 5,
        target_noise_scales=(0.1,) * 3,
    )
    chosen = indices[0].tolist()
    assert len(set(chosen)) == 3
    assert 4 not in chosen  # unsafe high reward cannot displace safe candidates
    assert 0 in chosen and 3 in chosen  # reward/quality extremes


def test_weighted_quality_dispersion_uses_only_eligible_candidates():
    quality = torch.tensor(
        [
            [[0., 0.], [2., 0.], [100., 100.]],
            [[1., 1.], [9., 9.], [0., 0.]],
        ]
    )
    eligible = torch.tensor([[True, True, False], [True, False, False]])
    dispersion = weighted_quality_dispersion(quality, eligible, (1., 1.))
    torch.testing.assert_close(dispersion[0], torch.tensor(2.0 ** -0.5))
    torch.testing.assert_close(dispersion[1], torch.tensor(0.0))
