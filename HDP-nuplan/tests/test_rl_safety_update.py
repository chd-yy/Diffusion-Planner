from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hdp_nuplan.rl.loss import group_advantage_weights, reward_weighted_diffusion_loss
from hdp_nuplan.rl.replay_buffer import (
    NuPlanReplayBuffer,
    ReplayTargetBatchSampler,
    ShardedNuPlanReplayBuffer,
)
from hdp_nuplan.rl.safety_update import (
    apply_update, build_candidate_mask, candidate_filter_enabled, validate_safety_update,
    validate_replay_target_coverage,
)
from hdp_nuplan.rl.train_epoch_rl import _load_replay_batch, update_epoch
from hdp_nuplan.model.diffusion_utils.sde import VPSDE_linear


def config(**overrides):
    args = dict(
        rl_filter_safety_eligible_candidates=False, rl_filter_progress_guard_candidates=False,
        rl_filter_collision_candidates=False, rl_center_reward_weights=False,
        reward_safety_gate_min_ttc_seconds=1.0, rl_min_progress_guard_reward=0.9,
        rl_rollout_loss_weight=1., rl_expert_anchor_weight=0., rl_reference_anchor_weight=0.,
        rl_grad_clip=5., rl_relative_to_reference=False, ddp=False,
        rl_deterministic_update=True, rl_max_update_steps_per_epoch=1,
        observation_normalizer=lambda value: value, state_normalizer=Identity(),
        diffusion_model_type="x_start", diffusion_supervision_type="x_start",
        planning_hybrid_loss=.01, rl_detach_window_size=0, rl_reward_temperature=1.,
        rl_advantage_clip=5., rl_min_reward_std=1e-6, rl_normalize_weights=True,
        rl_weighting_mode="positive_advantage",
        rl_task_preserving_update_mask=False, rl_component_dispersion_gate=False,
        rl_min_active_group_fraction=0., rl_min_target_group_retention=0.,
        rl_min_targets_per_retained_group=0., rl_min_target_group_count=0,
        rl_positive_advantage_margin=.1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class Identity:
    def __call__(self, value):
        return value

    def inverse(self, value):
        return value


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.5))
        self.sde = VPSDE_linear()

    def forward(self, inputs):
        return {}, {"score": inputs["sampled_trajectories"] * self.scale}


def buffer_with(mask):
    buffer = NuPlanReplayBuffer(1)
    trajectories = torch.zeros(3, 5, 4)
    trajectories[..., 0] = torch.arange(5).float()
    trajectories[..., 2] = 1.
    buffer.put("scene", trajectories, torch.tensor([0., 100., 1.]), candidate_mask=mask)
    return buffer


def test_filtered_replay_rejects_unknown_mask_before_reading_data():
    dataset = Mock()
    with pytest.raises(ValueError, match="Missing candidate_mask"):
        _load_replay_batch(dataset, buffer_with(None).sample(1), require_candidate_mask=True)
    dataset.get_by_name.assert_not_called()


def test_reference_relative_replay_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="reference_reward"):
        _load_replay_batch(Mock(), buffer_with(None).sample(1), require_reference_reward=True)


def test_constraint_replay_rejects_missing_update_contract():
    with pytest.raises(ValueError, match="update_mask/active_group"):
        _load_replay_batch(
            Mock(), buffer_with(torch.ones(3, dtype=torch.bool)).sample(1),
            require_update_contract=True,
        )


def test_replay_rejects_update_mask_outside_safety_set():
    buffer = NuPlanReplayBuffer(1)
    with pytest.raises(ValueError, match="ineligible"):
        buffer.put(
            "scene", torch.zeros(2, 3, 4), torch.tensor([0., 1.]),
            candidate_mask=torch.tensor([True, False]),
            update_mask=torch.tensor([True, True]),
            active_group=torch.tensor(True),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_replay_accepts_cuda_masks_and_stores_them_on_cpu():
    buffer = NuPlanReplayBuffer(1)
    buffer.put(
        "scene", torch.zeros(2, 3, 4, device="cuda"),
        torch.tensor([0., 1.], device="cuda"),
        candidate_mask=torch.tensor([True, False], device="cuda"),
        update_mask=torch.tensor([True, False], device="cuda"),
        active_group=torch.tensor(True, device="cuda"),
    )
    item = buffer.sample(1)[0]
    assert item.candidate_mask.device.type == "cpu"
    assert item.update_mask.device.type == "cpu"
    assert item.active_group.device.type == "cpu"


def test_target_aware_replay_sampling_guarantees_requested_slots():
    buffer = NuPlanReplayBuffer(4)
    for index in range(4):
        buffer.put(
            f"scene-{index}",
            torch.zeros(2, 3, 4),
            torch.tensor([0.0, 1.0]),
            candidate_mask=torch.ones(2, dtype=torch.bool),
            update_mask=torch.ones(2, dtype=torch.bool),
            active_group=torch.tensor(index == 0),
        )
    sampled = buffer.sample_target_aware(2, [True, False, False, False], 0.5)
    assert len(sampled) == 2
    assert any(item.scene_name == "scene-0" for item in sampled)


def test_target_aware_replay_sampling_fails_without_target_groups():
    buffer = NuPlanReplayBuffer(1)
    buffer.put("scene", torch.zeros(2, 3, 4), torch.tensor([0.0, 1.0]))
    with pytest.raises(RuntimeError, match="at least one target-bearing"):
        buffer.sample_target_aware(2, [False], 0.5)


def test_disk_sharded_replay_roundtrip_and_streaming_audit(tmp_path):
    buffer = ShardedNuPlanReplayBuffer(
        tmp_path / "replay", max_size=5, shard_size=2, cache_shards=1
    )
    for index in range(5):
        buffer.put(
            f"scene-{index}",
            torch.full((3, 5, 4), float(index)),
            torch.tensor([0.0, 1.0, 2.0]),
            candidate_mask=torch.ones(3, dtype=torch.bool),
            update_mask=torch.ones(3, dtype=torch.bool),
            active_group=torch.tensor(index != 4),
            reference_reward=torch.tensor(0.5),
            source_ids=torch.tensor([0, 1, 2]),
        )
    buffer.finalize()
    assert len(list((tmp_path / "replay").glob("shard_*.pt"))) == 3
    items = buffer.get_items([4, 0, 2])
    assert [item.scene_name for item in items] == ["scene-4", "scene-0", "scene-2"]
    torch.testing.assert_close(items[0].trajectories, torch.full((3, 5, 4), 4.0))
    streamed = [item.scene_name for batch in buffer.iter_item_batches(1) for item in batch]
    assert streamed == [f"scene-{index}" for index in range(5)]
    metrics, flags = validate_replay_target_coverage(
        buffer,
        config(rl_min_active_group_fraction=.7),
        return_target_flags=True,
        audit_batch_size=2,
    )
    assert metrics["replay_preflight_active_group_fraction"] == .8
    assert flags == [True, True, True, True, False]


def test_disk_sharded_replay_refuses_silent_capacity_truncation(tmp_path):
    buffer = ShardedNuPlanReplayBuffer(tmp_path / "replay", max_size=1)
    buffer.put("scene-0", torch.zeros(2, 3, 4), torch.tensor([0.0, 1.0]))
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        buffer.put("scene-1", torch.zeros(2, 3, 4), torch.tensor([0.0, 1.0]))


def test_target_batch_sampler_covers_targets_once_before_exhaustion():
    buffer = NuPlanReplayBuffer(6)
    for index in range(6):
        buffer.put(
            f"scene-{index}", torch.zeros(3, 5, 4), torch.tensor([0., 1., 2.]),
            candidate_mask=torch.ones(3, dtype=torch.bool),
            update_mask=torch.ones(3, dtype=torch.bool),
            active_group=torch.tensor(index < 3),
        )
    sampler = ReplayTargetBatchSampler(
        buffer, [True, True, True, False, False, False],
        batch_size=2, target_fraction=.5, seed=7,
    )
    seen = []
    for _ in range(sampler.steps_for_target_coverage):
        batch = sampler.next_batch()
        seen.extend(item.scene_name for item in batch if int(item.scene_name.split("-")[-1]) < 3)
    assert sorted(seen) == ["scene-0", "scene-1", "scene-2"]
    with pytest.raises(StopIteration):
        sampler.next_batch()


def test_replay_target_coverage_fails_before_update_on_low_active_fraction():
    buffer = NuPlanReplayBuffer(4)
    for index in range(4):
        buffer.put(
            f"scene-{index}", torch.zeros(3, 5, 4), torch.tensor([0., 1., 2.]),
            candidate_mask=torch.ones(3, dtype=torch.bool),
            update_mask=torch.ones(3, dtype=torch.bool),
            active_group=torch.tensor(index == 0),
        )
    with pytest.raises(RuntimeError, match="active_group_fraction"):
        validate_replay_target_coverage(
            buffer, config(rl_min_active_group_fraction=.4)
        )


def test_replay_target_coverage_reports_passing_target_contract():
    buffer = NuPlanReplayBuffer(2)
    for index in range(2):
        buffer.put(
            f"scene-{index}", torch.zeros(3, 5, 4), torch.tensor([0., 1., 2.]),
            candidate_mask=torch.ones(3, dtype=torch.bool),
            update_mask=torch.ones(3, dtype=torch.bool),
            active_group=torch.tensor(True),
        )
    metrics = validate_replay_target_coverage(
        buffer,
        config(
            rl_min_active_group_fraction=.4,
            rl_min_target_group_retention=.9,
            rl_min_targets_per_retained_group=1.,
        ),
    )
    assert metrics["replay_preflight_active_group_fraction"] == 1.
    assert metrics["replay_preflight_target_group_retention"] == 1.
    assert metrics["replay_preflight_targets_per_retained_group"] == 1.


def test_replay_target_coverage_supports_absolute_full_data_gate():
    buffer = NuPlanReplayBuffer(3)
    for index in range(3):
        buffer.put(
            f"scene-{index}", torch.zeros(3, 5, 4), torch.tensor([0., 1., 2.]),
            candidate_mask=torch.ones(3, dtype=torch.bool),
            update_mask=torch.ones(3, dtype=torch.bool),
            active_group=torch.tensor(index < 2),
        )
    validate_replay_target_coverage(
        buffer, config(rl_min_target_group_count=2)
    )
    with pytest.raises(RuntimeError, match="target_group_count"):
        validate_replay_target_coverage(
            buffer, config(rl_min_target_group_count=3)
        )


def test_safety_filter_rejects_negative_weights_but_keeps_explicit_legacy():
    validate_safety_update(config(rl_center_reward_weights=True))
    with pytest.raises(ValueError, match="negative"):
        validate_safety_update(config(rl_center_reward_weights=True, rl_filter_safety_eligible_candidates=True))


def test_direct_loss_cannot_bypass_nonnegative_contract():
    with pytest.raises(ValueError, match="nonnegative"):
        reward_weighted_diffusion_loss(None, {}, torch.zeros(1, 2, 3, 4), torch.zeros(1, 2),
            Identity(), VPSDE_linear(), "x_start", "x_start", .01,
            candidate_mask=torch.ones(1, 2, dtype=torch.bool), center_reward_weights=True)


def test_collision_filter_excludes_collision_even_if_risk_gate_passes():
    args = config(rl_filter_collision_candidates=True)
    mask = build_candidate_mask({"safety_gate_eligible": torch.ones(1, 3),
                                 "no_collision": torch.tensor([[1., 0., 1.]])}, args)
    assert candidate_filter_enabled(args)
    torch.testing.assert_close(mask, torch.tensor([[True, False, True]]))


def test_nonfinite_qualifications_and_rewards_are_rejected():
    with pytest.raises(ValueError, match="qualification"):
        build_candidate_mask({"safety_gate_eligible": torch.tensor([[float('nan')]])}, config())
    with pytest.raises(ValueError, match="finite"):
        group_advantage_weights(torch.tensor([[0., float('nan')]]))
    with pytest.raises(ValueError, match="reference_rewards"):
        group_advantage_weights(torch.tensor([[0., 1.]]), reference_rewards=torch.tensor([float('nan')]))


def test_zero_target_does_not_apply_adamw_momentum_decay_or_ema():
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    model.scale.square().backward()
    optimizer.step()  # Nonzero momentum makes an accidental zero-loss step observable.
    optimizer.zero_grad(set_to_none=True)
    before = model.scale.detach().clone()
    step = optimizer.state[model.scale]["step"].clone()
    ema = Mock()
    assert not apply_update(model.scale * 0, model, optimizer, ema, config(),
                            {"has_regression_targets": torch.tensor(0.)})
    torch.testing.assert_close(model.scale, before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[model.scale]["step"], step)
    ema.update.assert_not_called()


def test_anchor_still_updates_when_rollout_has_no_targets():
    model, ema = Model(), Mock()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    assert apply_update(model.scale.square(), model, optimizer, ema,
                        config(rl_expert_anchor_weight=.1), {"has_regression_targets": torch.tensor(0.)})
    assert model.scale.item() < .5
    ema.update.assert_called_once()


def test_nonfinite_loss_does_not_touch_optimizer_or_ema():
    model, optimizer, ema = Model(), Mock(), Mock()
    with pytest.raises(FloatingPointError):
        apply_update(model.scale * float('nan'), model, optimizer, ema, config(),
                     {"has_regression_targets": torch.tensor(1.)})
    optimizer.step.assert_not_called()
    ema.update.assert_not_called()


@pytest.mark.parametrize("flag", ["rl_filter_progress_guard_candidates", "rl_filter_collision_candidates"])
def test_filter_alone_reaches_real_update_loss(flag):
    # Integration: Replay -> loader -> update_epoch -> real diffusion loss.
    # The highest-reward (100) candidate must not re-enter via a missing safety flag.
    class Loader(list):
        dataset = Mock()
    loader = Loader([(torch.zeros(1, 10),)])
    loader.dataset.get_by_name.return_value = (torch.zeros(10),)
    prepared = ({}, {}, torch.zeros(1, 5, 4), torch.zeros(1, 1, 5, 4),
                torch.ones(1, 1, 5, dtype=torch.bool), None)
    args = config(**{flag: True})
    model, ema = Model(), Mock()
    with patch("hdp_nuplan.rl.train_epoch_rl.prepare_nuplan_batch", return_value=prepared):
        summary = update_epoch(loader, model, torch.optim.AdamW(model.parameters()), ema,
                               buffer_with(torch.tensor([True, False, True])), args, "cpu")
    assert summary["eligible_candidate_fraction"] == pytest.approx(2 / 3)
    assert summary["update_steps"] == 1


def test_update_reports_skipped_optimizer_steps_for_all_unsafe_batch():
    class Loader(list):
        dataset = Mock()
    loader = Loader([(torch.zeros(1, 10),)])
    loader.dataset.get_by_name.return_value = (torch.zeros(10),)
    prepared = ({}, {}, torch.zeros(1, 5, 4), torch.zeros(1, 1, 5, 4),
                torch.ones(1, 1, 5, dtype=torch.bool), None)
    model, ema = Model(), Mock()
    with patch("hdp_nuplan.rl.train_epoch_rl.prepare_nuplan_batch", return_value=prepared):
        summary = update_epoch(loader, model, torch.optim.AdamW(model.parameters()), ema,
            buffer_with(torch.zeros(3, dtype=torch.bool)), config(rl_filter_safety_eligible_candidates=True), "cpu")
    assert summary["update_steps"] == 0
    assert summary["skipped_update_steps"] == 1
    assert summary["sampled_batches"] == 1
    ema.update.assert_not_called()


def test_update_checkpoint_callback_counts_actual_optimizer_steps_only():
    class Loader(list):
        dataset = Mock()
    loader = Loader([(torch.zeros(1, 10),), (torch.zeros(1, 10),)])
    loader.dataset.get_by_name.return_value = (torch.zeros(10),)
    prepared = ({}, {}, torch.zeros(1, 5, 4), torch.zeros(1, 1, 5, 4),
                torch.ones(1, 1, 5, dtype=torch.bool), None)
    model, ema, callback = Model(), Mock(), Mock()
    args = config(rl_max_update_steps_per_epoch=2)
    with patch("hdp_nuplan.rl.train_epoch_rl.prepare_nuplan_batch", return_value=prepared):
        summary = update_epoch(
            loader,
            model,
            torch.optim.AdamW(model.parameters()),
            ema,
            buffer_with(torch.tensor([True, False, True])),
            args,
            "cpu",
            update_checkpoint_callback=callback,
        )
    assert summary["update_steps"] == 2
    assert [call.args[0] for call in callback.call_args_list] == [1, 2]


def _ddp_worker(rank, rendezvous):
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        ema = Mock()
        # Only one rank has targets, followed by a global skip, then another
        # update. This also checks the reducer remains usable after a skip.
        for any_target in (True, False, True):
            optimizer.zero_grad(set_to_none=True)
            before = next(model.parameters()).detach().clone()
            local_target = any_target and rank == 0
            loss = model(torch.ones(1, 1)).square().mean() * int(local_target)
            applied = apply_update(loss, model, optimizer, ema, config(),
                                   {"has_regression_targets": torch.tensor(float(local_target))})
            assert applied == any_target
            if not any_target:
                torch.testing.assert_close(next(model.parameters()), before, rtol=0, atol=0)
            values = [torch.zeros_like(before) for _ in range(2)]
            torch.distributed.all_gather(values, next(model.parameters()).detach())
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert ema.update.call_count == 2
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_mixed_targets_and_global_skip(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    torch.multiprocessing.spawn(_ddp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
