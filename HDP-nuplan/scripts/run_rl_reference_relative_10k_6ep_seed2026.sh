#!/usr/bin/env bash
set -euo pipefail

# 这是 reference-relative RL 的独立对照实验，不覆盖历史 v8/v9 结果。
# 训练目标：候选 reward - 冻结 B Epoch10 reward，而不是候选组均值。

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1"
CACHE="$ROOT/cache"
MANIFEST="$ROOT/diffusion_planner_training.json"
PRETRAINED="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
NORMALIZATION="$PROJECT_ROOT/HDP-nuplan/normalization.json"
RUN_TAG="reference_relative_v1_10k_6ep_seed2026"
RUN_ROOT="$ROOT/rl_${RUN_TAG}_from_b10"

NPZ_COUNT="$(find "$CACHE" -maxdepth 1 -type f -name '*.npz' | wc -l)"
[[ "$NPZ_COUNT" == "10000" ]] || {
    echo "Expected 10000 NPZ files, found $NPZ_COUNT" >&2
    exit 1
}
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$PRETRAINED" ]] || { echo "Missing checkpoint: $PRETRAINED" >&2; exit 1; }

exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 \
    --nproc-per-node 1 \
    --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor_rl.py" \
    --name "hdp-rl-$RUN_TAG-from-full-mini-b10" \
    --save_dir "$RUN_ROOT" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --pretrained_model_path "$PRETRAINED" \
    --normalization_file_path "$NORMALIZATION" \
    --train_epochs 6 \
    --batch_size 2 \
    --learning_rate 4e-7 \
    --warm_up_epoch 1 \
    --save_utd 1 \
    --num_workers 2 \
    --planning_hybrid_loss 0.01 \
    --rl_group_size 32 \
    --rl_rollout_steps 6 \
    --rl_sampling_noise_scale 0.1 \
    --rl_reference_noise_scale 0.0 \
    --rl_trajectory_augmentation_std 0 \
    --rl_trajectory_augmentation_epochs 0 \
    --rl_buffer_update_epoch 2 \
    --rl_buffer_size 10000 \
    --rl_ema_update_rate 0.05 \
    --rl_reward_temperature 1.0 \
    --rl_advantage_clip 5.0 \
    --rl_min_reward_std 1e-6 \
    --rl_normalize_weights true \
    --rl_center_reward_weights false \
    --rl_weighting_mode positive_advantage \
    --rl_relative_to_reference true \
    --rl_filter_safety_eligible_candidates false \
    --rl_filter_progress_guard_candidates false \
    --rl_rollout_loss_weight 1.0 \
    --rl_expert_anchor_weight 0.1 \
    --rl_reference_anchor_weight 20.0 \
    --rl_max_update_steps_per_epoch 500 \
    --rl_detach_window_size 0 \
    --rl_grad_clip 5.0 \
    --rl_freeze_encoder true \
    --rl_deterministic_update true \
    --reward_progress_weight 1.0 \
    --reward_collision_weight 10.0 \
    --reward_route_weight 1.0 \
    --reward_comfort_weight 0.01 \
    --reward_backward_weight 1.0 \
    --reward_imitation_weight 0.0 \
    --reward_collision_distance 0.5 \
    --reward_risk_weight 1.0 \
    --reward_follow_weight 3.0 \
    --reward_lane_weight 2.5 \
    --reward_progress_guard_weight 5.0 \
    --reward_progress_guard_stop_tolerance 0.2 \
    --reward_safety_gate_threshold 0.3 \
    --reward_safety_gate_margin 1.0 \
    --reward_safety_gate_min_ttc_seconds 1.0 \
    --reward_safety_gate_require_drivable_area false \
    --reward_objective_mode nuplan_score_proxy_v3 \
    --seed 2026 \
    --use_wandb false
