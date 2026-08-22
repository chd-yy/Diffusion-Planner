import unittest

import torch
from timm.utils import ModelEma

from hdp_nuplan.model.diffusion_utils.sde import VPSDE_linear
from hdp_nuplan.rl.loss import (
    group_advantage_weights,
    reward_weighted_diffusion_loss,
)
from hdp_nuplan.rl.replay_buffer import NuPlanReplayBuffer
from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer
from hdp_nuplan.rl.trajectory_augmentation import augment_trajectory_batch
from hdp_nuplan.rl.train_epoch_rl import combine_update_losses
from hdp_nuplan.utils.traj_kinematics import detached_integral
from train_predictor_rl import ema_decay_from_update_rate


class _Normalizer:
    def __init__(self):
        self.mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.std = torch.tensor([0.5, 0.5, 1.0, 1.0])

    def __call__(self, value):
        return (value - self.mean.to(value)) / self.std.to(value)

    def inverse(self, value):
        return value * self.std.to(value) + self.mean.to(value)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs):
        return {}, {"score": inputs["sampled_trajectories"] * self.scale}


class RlComponentTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.group_size = 2
        self.horizon = 20
        x = torch.linspace(0.5, 10.0, self.horizon)

        self.trajectories = torch.zeros(
            self.batch_size,
            self.group_size,
            self.horizon,
            4,
        )
        self.trajectories[..., 0] = x
        self.trajectories[..., 2] = 1.0
        self.trajectories[:, 1, :, 1] = 4.0

        self.neighbors = torch.zeros(self.batch_size, 1, self.horizon, 4)
        self.neighbors[..., 0] = x
        self.neighbors[..., 2] = 1.0
        self.neighbor_mask = torch.zeros(
            self.batch_size,
            1,
            self.horizon,
            dtype=torch.bool,
        )
        self.route = torch.zeros(self.batch_size, 1, self.horizon, 4)
        self.route[..., 0] = x
        self.route[..., 2] = 1.0

    def test_reward_penalizes_collision(self):
        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(comfort_weight=0.0)
        )
        rewards, details = scorer(
            self.trajectories,
            self.neighbors,
            self.neighbor_mask,
            self.route,
            static_objects=torch.zeros(self.batch_size, 2, 10),
        )
        self.assertGreater(
            details["collision_cost"][0, 0],
            details["collision_cost"][0, 1],
        )
        # 论文 multi-reward 不是 safety 的硬约束；该构造中第二条轨迹同时离开车道，
        # 因此只检查 risk 分项确实把安全候选排在碰撞候选之前。
        self.assertGreater(
            details["risk_reward"][0, 1], details["risk_reward"][0, 0]
        )

    def test_paper_rewards_are_bounded_and_use_table6_weights(self):
        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        rewards, details = scorer(
            self.trajectories,
            self.neighbors,
            self.neighbor_mask,
            self.route,
            static_objects=torch.zeros(self.batch_size, 2, 10),
        )

        for key in ("risk_reward", "follow_reward", "lane_reward"):
            self.assertTrue(torch.all(details[key] >= 0))
            self.assertTrue(torch.all(details[key] <= 1))
        expected = (
            details["risk_reward"]
            + 3.0 * details["follow_reward"]
            + 2.5 * details["lane_reward"]
        )
        torch.testing.assert_close(rewards, expected)

    def test_progress_guard_is_bounded_and_does_not_reward_overspeed(self):
        trajectories = torch.zeros(1, 3, 4, 4)
        trajectories[..., 2] = 1.0
        trajectories[0, 0, :, 0] = torch.tensor([0.5, 1.0, 1.5, 2.0])
        trajectories[0, 1, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        trajectories[0, 2, :, 0] = torch.tensor([1.5, 3.0, 4.5, 6.0])
        ego_future = trajectories[:, 1]
        route_lanes = torch.zeros(1, 1, 1, 12)

        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(progress_guard_weight=0.5)
        )
        progress, _ = scorer._route_motion_metrics(trajectories, route_lanes)
        guard = scorer._progress_guard_reward(
            trajectories, ego_future, route_lanes, progress
        )

        self.assertTrue(torch.all((guard >= 0) & (guard <= 1)))
        self.assertAlmostEqual(guard[0, 0].item(), 0.5, places=6)
        self.assertAlmostEqual(guard[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(guard[0, 2].item(), 1.0, places=6)

    def test_safety_gate_prevents_progress_from_overriding_safe_candidate(self):
        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(
                safety_gate_threshold=0.4,
                safety_gate_margin=1.0,
            )
        )
        # 候选 0 的原奖励更高，但 risk 不达标；候选 1 安全达标。
        base_reward = torch.tensor([[11.0, 2.0, 3.0]])
        risk_reward = torch.tensor([[0.3, 0.4, 0.8]])

        gated, eligible, has_eligible = scorer._apply_safety_gate(
            base_reward, risk_reward
        )

        self.assertTrue(has_eligible.item())
        self.assertFalse(eligible[0, 0].item())
        self.assertLess(gated[0, 0], gated[0, 1:].min())
        torch.testing.assert_close(gated[0, 1:], base_reward[0, 1:])

    def test_safety_gate_enforces_min_ttc_seconds(self):
        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(
                safety_gate_min_ttc_seconds=1.0,
                safety_gate_margin=1.0,
            )
        )
        # 两个候选的连续 risk 相同；只有候选 1 通过 1 秒 TTC 硬门。
        base_reward = torch.tensor([[10.0, 1.0]])
        risk_reward = torch.tensor([[0.8, 0.8]])
        min_ttc_seconds = torch.tensor([[0.9, 1.1]])

        gated, eligible, has_eligible = scorer._apply_safety_gate(
            base_reward,
            risk_reward,
            min_ttc_seconds,
        )

        self.assertTrue(has_eligible.item())
        self.assertFalse(eligible[0, 0].item())
        self.assertTrue(eligible[0, 1].item())
        self.assertLess(gated[0, 0], gated[0, 1])

    def test_reward_returns_physical_min_ttc_seconds(self):
        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        _, details = scorer(
            self.trajectories,
            self.neighbors,
            self.neighbor_mask,
            self.route,
            static_objects=torch.zeros(self.batch_size, 2, 10),
        )

        # 候选 0 与邻车重叠，预计 TTC 为 0；候选 1 横向错开且无 closing，
        # 在当前恒速近似下没有预计碰撞，返回 inf。
        self.assertAlmostEqual(details["min_ttc_seconds"][0, 0].item(), 0.0)
        self.assertTrue(torch.isinf(details["min_ttc_seconds"][0, 1]))

    def test_safety_gate_all_unsafe_group_prefers_highest_risk(self):
        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(safety_gate_threshold=0.4)
        )
        # 整组都不达标时，更高的原始 utility 不能压过更好的 risk。
        base_reward = torch.tensor([[11.0, 1.0]])
        risk_reward = torch.tensor([[0.1, 0.3]])

        gated, eligible, has_eligible = scorer._apply_safety_gate(
            base_reward, risk_reward
        )

        self.assertFalse(has_eligible.item())
        self.assertFalse(eligible.any().item())
        self.assertGreater(gated[0, 1], gated[0, 0])

    def test_disabled_safety_gate_preserves_paper_reward(self):
        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(safety_gate_threshold=0.0)
        )
        base_reward = torch.tensor([[1.0, 2.0]])
        risk_reward = torch.tensor([[0.1, 0.9]])

        gated, _, _ = scorer._apply_safety_gate(base_reward, risk_reward)

        torch.testing.assert_close(gated, base_reward)

    def test_paper_follow_reward_prefers_safe_gap_to_tailgating(self):
        horizon = 20
        time = torch.arange(1, horizon + 1, dtype=torch.float32) * 0.1
        trajectories = torch.zeros(1, 2, horizon, 4)
        trajectories[0, 0, :, 0] = 5.0 * time
        trajectories[0, 1, :, 0] = 5.0 * time + 5.0
        trajectories[..., 2] = 1.0

        neighbors = torch.zeros(1, 1, horizon, 4)
        neighbors[0, 0, :, 0] = 10.0 + 5.0 * time
        neighbors[..., 2] = 1.0
        neighbor_mask = torch.zeros(1, 1, horizon, dtype=torch.bool)
        neighbor_past = torch.zeros(1, 1, 21, 11)
        neighbor_past[0, 0, -1, 0] = 10.0
        neighbor_past[0, 0, -1, 2] = 1.0
        neighbor_past[0, 0, -1, 6:8] = torch.tensor([2.0, 4.8])
        neighbor_past[0, 0, -1, 8] = 1.0
        ego_current_state = torch.zeros(1, 10)
        ego_current_state[0, 4] = 5.0

        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        follow, leader_fraction = scorer._following_reward(
            trajectories,
            neighbors,
            neighbor_mask,
            neighbor_past,
            ego_current_state,
        )

        self.assertGreater(follow[0, 0], follow[0, 1])
        torch.testing.assert_close(leader_fraction, torch.ones_like(leader_fraction))

    def test_paper_lane_reward_prefers_centerline(self):
        horizon = 20
        x = torch.arange(1, horizon + 1, dtype=torch.float32)
        trajectories = torch.zeros(1, 2, horizon, 4)
        trajectories[..., 0] = x
        trajectories[..., 2] = 1.0
        trajectories[0, 1, :, 1] = 1.5

        lanes = torch.zeros(1, 1, horizon, 12)
        lanes[..., 0] = x
        lanes[..., 2] = 1.0
        lanes[..., 5] = 1.75
        lanes[..., 7] = -1.75
        ego_future = trajectories[:, 0]

        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        lane_reward, lane_mask = scorer._lane_reward(
            trajectories, lanes, ego_future
        )

        self.assertGreater(lane_reward[0, 0], lane_reward[0, 1])
        torch.testing.assert_close(lane_mask, torch.ones_like(lane_mask))

    def test_paper_ema_value_is_interpreted_as_update_rate(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        ema = ModelEma(model, decay=ema_decay_from_update_rate(0.05))
        model.weight.data.fill_(2.0)
        ema.update(model)

        # ema = 0.95 * 1.0 + 0.05 * 2.0 = 1.05
        torch.testing.assert_close(ema.ema.weight, torch.tensor([[1.05]]))

    def test_reward_v2_detects_size_aware_box_overlap(self):
        horizon = 6
        trajectories = torch.zeros(1, 2, horizon, 4)
        trajectories[..., 2] = 1.0
        trajectories[:, 1, :, 1] = 5.0

        neighbors = torch.zeros(1, 1, horizon, 4)
        neighbors[..., 0] = 3.5
        neighbors[..., 2] = 1.0
        neighbor_mask = torch.zeros(1, 1, horizon, dtype=torch.bool)
        neighbor_past = torch.zeros(1, 1, 21, 11)
        neighbor_past[..., 6] = 2.0
        neighbor_past[..., 7] = 4.8

        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        collision_cost, no_collision = scorer._collision_cost(
            trajectories,
            neighbors,
            neighbor_mask,
            static_objects=None,
            neighbor_agents_past=neighbor_past,
        )

        # 3.5 m 大于旧中心阈值 2.5 m，但两辆 4.8 m 长车辆实际重叠。
        self.assertGreater(collision_cost[0, 0], 0)
        self.assertEqual(no_collision[0, 0], 0)
        self.assertEqual(no_collision[0, 1], 1)

    def test_reward_v2_uses_route_aligned_progress(self):
        horizon = 8
        route = torch.zeros(1, 1, horizon, 12)
        route[0, 0, :, 1] = torch.arange(1, horizon + 1, dtype=torch.float32)
        route[0, 0, :, 3] = 1.0

        trajectories = torch.zeros(1, 2, horizon, 4)
        trajectories[0, 0, :, 1] = torch.arange(
            1, horizon + 1, dtype=torch.float32
        )
        trajectories[0, 0, :, 3] = 1.0
        trajectories[0, 1, :, 0] = torch.arange(
            1, horizon + 1, dtype=torch.float32
        )
        trajectories[0, 1, :, 2] = 1.0

        scorer = NuPlanTensorRewardScorer(
            NuPlanRewardConfig(collision_weight=0.0, comfort_weight=0.0)
        )
        rewards, details = scorer(
            trajectories,
            neighbors_future=torch.zeros(1, 0, horizon, 4),
            neighbor_mask=torch.zeros(1, 0, horizon, dtype=torch.bool),
            route_lanes=route,
            static_objects=torch.zeros(1, 0, 10),
        )

        self.assertGreater(details["progress"][0, 0], 0)
        self.assertAlmostEqual(details["progress"][0, 1].item(), 0.0, places=6)
        self.assertLess(details["route_cost"][0, 0], details["route_cost"][0, 1])
        self.assertGreater(rewards[0, 0], rewards[0, 1])

    def test_reward_v2_comfort_uses_current_state_and_clips_outliers(self):
        horizon = 20
        time = torch.arange(1, horizon + 1, dtype=torch.float32) * 0.1
        trajectories = torch.zeros(1, 2, horizon, 4)
        trajectories[0, 0, :, 0] = time
        trajectories[0, 1, :, 0] = time + 0.2 * torch.where(
            torch.arange(horizon) % 2 == 0,
            torch.ones(horizon),
            -torch.ones(horizon),
        )
        trajectories[..., 2] = 1.0
        ego_current_state = torch.zeros(1, 10)
        ego_current_state[:, 4] = 1.0

        scorer = NuPlanTensorRewardScorer(NuPlanRewardConfig())
        comfort = scorer._comfort_cost(trajectories, ego_current_state)

        self.assertAlmostEqual(comfort[0, 0].item(), 0.0, places=6)
        self.assertGreater(comfort[0, 1], comfort[0, 0])
        self.assertLessEqual(
            comfort.max().item(), 2 * scorer.config.comfort_violation_clip
        )

    def test_replay_buffer_round_trip(self):
        buffer = NuPlanReplayBuffer(max_size=2)
        rewards = torch.tensor([0.0, 1.0])
        buffer.put("scene.npz", self.trajectories[0], rewards)
        item = buffer.sample(1)[0]
        self.assertEqual(item.scene_name, "scene.npz")
        self.assertEqual(tuple(item.trajectories.shape), (2, 20, 4))

    def test_reward_weighted_loss_has_gradient(self):
        model = _DummyModel()
        rewards = torch.tensor([[0.0, 1.0]])
        loss, _ = reward_weighted_diffusion_loss(
            model=model,
            inputs={"condition": torch.zeros(self.batch_size, 3)},
            trajectories=self.trajectories,
            rewards=rewards,
            state_normalizer=_Normalizer(),
            sde=VPSDE_linear(),
            model_type="x_start",
            supervision_type="x_start",
            hybrid_loss_weight=0.01,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.scale.grad))

    def test_centered_weights_remove_beta_zero_self_distillation_gradient(self):
        model = _DummyModel()
        trajectories = self.trajectories.expand(2, -1, -1, -1).clone()
        loss, metrics = reward_weighted_diffusion_loss(
            model=model,
            inputs={"condition": torch.zeros(2, 3)},
            trajectories=trajectories,
            rewards=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            state_normalizer=_Normalizer(),
            sde=VPSDE_linear(),
            model_type="x_start",
            supervision_type="x_start",
            hybrid_loss_weight=0.01,
            reward_temperature=0.0,
            center_reward_weights=True,
        )
        loss.backward()

        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(
            metrics["regression_weight_mean"],
            torch.zeros_like(metrics["regression_weight_mean"]),
        )
        torch.testing.assert_close(model.scale.grad, torch.zeros_like(model.scale.grad))

    def test_group_weights_skip_low_variance_and_normalize_active_group(self):
        rewards = torch.tensor(
            [
                [1.000, 1.002, 1.004, 1.006],
                [0.0, 0.5, 1.0, 1.5],
            ]
        )
        advantages, weights = group_advantage_weights(
            rewards,
            min_reward_std=0.01,
            normalize_weights=True,
        )

        torch.testing.assert_close(advantages[0], torch.zeros(4))
        torch.testing.assert_close(weights[0], torch.zeros(4))
        self.assertAlmostEqual(weights[1].mean().item(), 1.0, places=6)
        self.assertGreater(weights[1, -1], weights[1, 0])

    def test_navsim_trajectory_augmentation_uses_local_frame(self):
        trajectories = torch.zeros(1, 2, 3, 4)
        trajectories[0, 0, :, 2] = 1.0
        trajectories[0, 1, :, 3] = 1.0

        torch.manual_seed(7)
        augmented = augment_trajectory_batch(trajectories, std=0.5)

        # 每条候选的局部偏移沿时间共享，不会给每个轨迹点分别增加 jitter。
        torch.testing.assert_close(
            augmented[..., :2],
            augmented[..., :1, :2].expand_as(augmented[..., :2]),
        )
        # 航向必须原样保留。
        torch.testing.assert_close(augmented[..., 2:4], trajectories[..., 2:4])
        # 两种朝向都应得到非零的全局平移。
        self.assertTrue(torch.all(augmented[..., :2].abs().sum(dim=-1) > 0))

    def test_trajectory_augmentation_zero_is_noop(self):
        augmented = augment_trajectory_batch(self.trajectories, std=0.0)
        self.assertIs(augmented, self.trajectories)

    def test_update_loss_weights_can_isolate_expert_anchor(self):
        rl_loss = torch.tensor(3.0, requires_grad=True)
        anchor_loss = torch.tensor(2.0, requires_grad=True)
        total = combine_update_losses(rl_loss, anchor_loss, 0.0, 1.0)
        total.backward()

        self.assertEqual(total.item(), 2.0)
        self.assertEqual(rl_loss.grad.item(), 0.0)
        self.assertEqual(anchor_loss.grad.item(), 1.0)

    def test_detached_integral_preserves_values_and_truncates_gradient(self):
        window_size = 3
        values = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
        values.requires_grad_()

        integrated = detached_integral(values, window_size)
        torch.testing.assert_close(integrated, torch.cumsum(values, dim=-2))

        integrated[:, -1, :].sum().backward()
        expected_gradient = torch.zeros_like(values)
        expected_gradient[:, -window_size:, :] = 1
        torch.testing.assert_close(values.grad, expected_gradient)

    def test_detached_integral_zero_uses_full_cumsum_gradient(self):
        values = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
        values.requires_grad_()

        integrated = detached_integral(values, detach_window_size=0)
        torch.testing.assert_close(integrated, torch.cumsum(values, dim=-2))

        integrated[:, -1, :].sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_detached_integral_rejects_negative_window(self):
        values = torch.ones(1, 2, 2)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            detached_integral(values, detach_window_size=-1)


if __name__ == "__main__":
    unittest.main()
