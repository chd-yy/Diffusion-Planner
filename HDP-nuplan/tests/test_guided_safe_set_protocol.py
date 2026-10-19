import torch
from types import SimpleNamespace

from hdp_nuplan.model.guidance.longitudinal import (
    expert_frenet_targets,
    make_longitudinal_guidance_energy,
)
from hdp_nuplan.rl.candidate_safe_set import (
    CandidateEligibility,
    SOURCE_GUIDED,
    SOURCE_POLICY_MEDIUM,
    SOURCE_POLICY_NATURAL,
    SOURCE_REFERENCE_NATURAL,
    classify_candidate_eligibility,
    safe_set_compression_indices,
)
from hdp_nuplan.rl.candidate_safe_set import _append_unique
from hdp_nuplan.rl.replay_buffer import NuPlanReplayBuffer
from hdp_nuplan.rl.loss import reward_weighted_diffusion_loss
from hdp_nuplan.rl.safety_update import validate_safety_update
from hdp_nuplan.rl.train_epoch_rl import (
    _scene_adaptive_guidance_deviations,
    _temporal_forward_conflict_mask,
)
from hdp_nuplan.model.diffusion_utils.sde import VPSDE_linear


class IdentityNormalizer:
    def __call__(self, value):
        return value

    def inverse(self, value):
        return value


class TinyDiffusionModel(torch.nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, inputs):
        return {}, {"score": inputs["sampled_trajectories"] * self.scale}


def test_guidance_uses_per_step_expert_tangent_and_respects_gate():
    expert = torch.zeros(1, 3, 4)
    expert[0, :, :2] = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    tangent, speed = expert_frenet_targets(expert, dt=1.0)
    torch.testing.assert_close(tangent[0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(tangent[0, 1], torch.tensor([0.0, 1.0]))
    energy = make_longitudinal_guidance_energy(dt=1.0)
    action = torch.zeros(2, 3, 4)
    action[:, :, 0] = 1.0
    cost = energy(
        action,
        None,
        None,
        state_normalizer=IdentityNormalizer(),
        expert_tangent=tangent.repeat_interleave(2, dim=0),
        expert_speed=speed.repeat_interleave(2, dim=0),
        deviation=torch.zeros(2),
        gate=torch.tensor([1.0, 0.0]),
    )
    assert cost[0] > 0
    assert cost[1] == 0


def test_scene_adaptive_guidance_uses_deceleration_for_forward_conflict():
    deviation, gate = _scene_adaptive_guidance_deviations(
        [-0.5, -0.25, 0.0, 0.25, 0.5],
        forward_conflict=torch.tensor([False, True, False]),
        moving=torch.tensor([True, True, False]),
        guided_count=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert deviation[0].min() < 0 and deviation[0].max() > 0
    assert deviation[1].min() < 0 and deviation[1].max() <= 0
    torch.testing.assert_close(deviation[2], torch.zeros(10))
    torch.testing.assert_close(gate, torch.tensor([1.0, 1.0, 0.0]))


def test_temporal_forward_conflict_ignores_rear_vehicle():
    ego = torch.zeros(1, 3, 4)
    ego[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    neighbours = torch.zeros(1, 2, 3, 3)
    neighbours[0, 0, :, 0] = ego[0, :, 0] + 5.0
    neighbours[0, 1, :, 0] = ego[0, :, 0] - 2.0
    valid = torch.zeros(1, 2, 3, dtype=torch.bool)
    assert _temporal_forward_conflict_mask(ego, neighbours, valid).item()
    neighbours[0, 0, :, 1] = 10.0
    assert not _temporal_forward_conflict_mask(ego, neighbours, valid).item()


def test_safe_set_uses_only_collision_da_and_finite_trajectory():
    trajectories = torch.zeros(1, 4, 2, 4)
    trajectories[0, 3, 0, 0] = float("nan")
    details = {
        "no_collision": torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        "drivable_area_compliance": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "drivable_area_gate_active": torch.ones(1, 4),
        # Deliberately poor TTC is not a cache-independent hard event in this
        # protocol; it remains part of aligned reward/quality diagnostics.
        "min_ttc_seconds": torch.tensor([[0.1, 2.0, 2.0, 2.0]]),
    }
    eligibility = classify_candidate_eligibility(trajectories, details)
    assert eligibility.valid.tolist() == [[True, True, True, False]]
    assert eligibility.hard_safe.tolist() == [[True, False, False, False]]


def test_safe_set_compression_enforces_medium_quota_without_unsafe_targets():
    batch, pool, dims = 1, 48, 5
    quality = torch.arange(pool, dtype=torch.float32).reshape(
        1, pool, 1
    ).repeat(1, 1, dims)
    reward = torch.linspace(0, 1, pool).reshape(1, pool)
    valid = torch.ones(batch, pool, dtype=torch.bool)
    hard_safe = torch.ones_like(valid)
    hard_safe[:, -4:] = False
    eligibility = CandidateEligibility(valid=valid, hard_safe=hard_safe)
    sources = torch.tensor(
        [SOURCE_REFERENCE_NATURAL] * 8
        + [SOURCE_POLICY_NATURAL] * 8
        + [SOURCE_POLICY_MEDIUM] * 24
        + [SOURCE_GUIDED] * 8,
        dtype=torch.long,
    )
    indices = safe_set_compression_indices(
        quality,
        reward,
        eligibility,
        sources,
        target_size=32,
        natural_slots=8,
        medium_elite_slots=8,
        medium_diverse_slots=8,
        global_elite_slots=8,
    )
    assert indices.shape == (1, 32)
    assert len(set(indices[0].tolist())) == 32
    assert not set(range(44, 48)).intersection(indices[0].tolist())
    selected_sources = sources[indices[0]]
    assert int((selected_sources == SOURCE_POLICY_MEDIUM).sum()) >= 16
    assert int(sources.max()) == SOURCE_GUIDED


def test_safe_set_compression_marks_repeated_storage_padding():
    quality = torch.zeros(2, 40, 5)
    reward = torch.zeros(2, 40)
    valid = torch.zeros(2, 40, dtype=torch.bool)
    valid[0, :3] = True
    hard_safe = valid.clone()
    eligibility = CandidateEligibility(valid=valid, hard_safe=hard_safe)
    sources = torch.full((40,), SOURCE_POLICY_MEDIUM, dtype=torch.long)
    indices, padding = safe_set_compression_indices(
        quality,
        reward,
        eligibility,
        sources,
        target_size=32,
        natural_slots=0,
        medium_elite_slots=8,
        medium_diverse_slots=8,
        global_elite_slots=16,
        return_padding_mask=True,
    )
    assert indices.shape == padding.shape == (2, 32)
    assert int((~padding[0]).sum()) == 3
    assert padding[1].all()
    assert set(indices[0, :3].tolist()) == {0, 1, 2}


def test_append_unique_never_exceeds_an_already_reached_limit():
    selected = [0, 1, 2]
    _append_unique(selected, [3, 4], 3)
    assert selected == [0, 1, 2]


def test_replay_preserves_safe_set_masks_and_sources():
    buffer = NuPlanReplayBuffer()
    trajectories = torch.zeros(4, 2, 4)
    rewards = torch.arange(4, dtype=torch.float32)
    candidate_mask = torch.tensor([True, True, False, False])
    update_mask = torch.tensor([False, True, False, False])
    sources = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    buffer.put(
        "scene",
        trajectories,
        rewards,
        candidate_mask=candidate_mask,
        update_mask=update_mask,
        active_group=torch.tensor(True),
        reference_reward=torch.tensor(0.5),
        source_ids=sources,
    )
    item = buffer.items()[0]
    torch.testing.assert_close(item.candidate_mask, candidate_mask)
    torch.testing.assert_close(item.update_mask, update_mask)
    torch.testing.assert_close(item.source_ids, sources)


def test_safe_set_loss_has_positive_targets_only():
    model = TinyDiffusionModel()
    reference = TinyDiffusionModel()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trajectories = torch.zeros(1, 4, 5, 4)
    trajectories[0, :, :, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])[:, None]
    trajectories[..., 2] = 1.0
    loss, metrics = reward_weighted_diffusion_loss(
        model=model,
        inputs={"condition": torch.zeros(1, 1)},
        trajectories=trajectories,
        rewards=torch.tensor([[0.9, 0.3, 0.6, 1.0]]),
        state_normalizer=IdentityNormalizer(),
        sde=VPSDE_linear(),
        model_type="x_start",
        supervision_type="x_start",
        hybrid_loss_weight=0.0,
        min_reward_std=1e-6,
        normalize_weights=False,
        weighting_mode="constraint_uniform_positive",
        candidate_mask=torch.tensor([[False, True, True, True]]),
        update_mask=torch.tensor([[False, False, False, True]]),
        active_group_mask=torch.tensor([True]),
        positive_margin=0.1,
        reference_model=reference,
        reference_rewards=torch.tensor([0.5]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["regression_targets_per_group"].item() == 1.0
    assert metrics["task_update_candidate_fraction"].item() == 0.25
    assert "event_preference_loss" not in metrics
    assert torch.isfinite(model.scale.grad)


def test_safe_set_protocol_requires_explicit_da_hard_gate():
    args = SimpleNamespace(
        rl_guided_safe_set_protocol=True,
        rl_filter_safety_eligible_candidates=False,
        rl_filter_progress_guard_candidates=False,
        rl_filter_collision_candidates=False,
        rl_center_reward_weights=False,
        rl_weighting_mode="constraint_uniform_positive",
        rl_component_dispersion_gate=True,
        rl_task_preserving_update_mask=False,
        reward_objective_mode="nuplan_aligned",
        reward_safety_gate_threshold=0.0,
        reward_safety_gate_min_ttc_seconds=0.0,
        reward_safety_gate_require_drivable_area=True,
    )
    validate_safety_update(args)
    args.reward_safety_gate_require_drivable_area = False
    try:
        validate_safety_update(args)
    except ValueError as error:
        assert "drivable-area" in str(error)
    else:
        raise AssertionError("safe-set protocol accepted a missing DA hard gate")
