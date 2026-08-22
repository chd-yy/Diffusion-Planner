#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
PILOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1"
TRAIN_OUT="$PILOT/rl_from_full_mini_b_epoch10_omega001_nodetach"
TRAIN_LOG="$PILOT/rl_from_full_mini_b10_pilot1000.log"
CACHE="$PILOT/cache"
MANIFEST="$PILOT/diffusion_planner_training.json"
NORMALIZATION="$PROJECT_ROOT/HDP-nuplan/normalization.json"
PRETRAINED="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
CLOSED_ROOT="$PILOT/rl_from_full_mini_b_epoch10_omega001_nodetach_closed_loop20"
VAL_FILTER="mini-val-closed-loop-20"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$TRAIN_LOG"
}

find_rl_checkpoint() {
    find "$TRAIN_OUT/training_log" -type f -name 'model_epoch_2_trainloss_*.pth' -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

mkdir -p "$TRAIN_OUT" "$CLOSED_ROOT"

"$PYTHON_BIN" - "$PRETRAINED" "$CACHE" "$MANIFEST" <<'PY' >> "$TRAIN_LOG" 2>&1
import json
import sys
from pathlib import Path
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
required = {"model", "ema_state_dict"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"Pretrained checkpoint missing keys: {sorted(missing)}")
npz_count = len(list(Path(sys.argv[2]).glob("*.npz")))
if npz_count != 1000:
    raise SystemExit(f"Expected 1000 NPZ files, got {npz_count}")
manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if len(manifest) != 1000:
    raise SystemExit(f"Expected 1000 manifest entries, got {len(manifest)}")
print(f"Validated B Epoch 10 checkpoint: {sys.argv[1]}")
print(f"Validated pilot cache and manifest: {npz_count} NPZ")
PY

cd "$PROJECT_ROOT"
log "Starting RL pilot from HDP B Epoch 10"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor_rl.py" \
    --name hdp-rl-from-full-mini-b10-pilot1000-omega001-nodetach \
    --save_dir "$TRAIN_OUT" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --pretrained_model_path "$PRETRAINED" \
    --normalization_file_path "$NORMALIZATION" \
    --train_epochs 2 --batch_size 2 --learning_rate 4e-7 \
    --warm_up_epoch 1 --save_utd 1 --num_workers 2 \
    --planning_hybrid_loss 0.01 \
    --rl_group_size 32 --rl_rollout_steps 6 \
    --rl_sampling_noise_scale 0.1 \
    --rl_trajectory_augmentation_std 0 \
    --rl_trajectory_augmentation_epochs 0 \
    --rl_buffer_update_epoch 2 --rl_buffer_size 1024 \
    --rl_ema_update_rate 0.05 --rl_reward_temperature 1.0 \
    --rl_advantage_clip 5.0 --rl_min_reward_std 1e-6 \
    --rl_normalize_weights true --rl_center_reward_weights true \
    --rl_rollout_loss_weight 1.0 --rl_expert_anchor_weight 0.0 \
    --rl_max_update_steps_per_epoch 500 \
    --rl_detach_window_size 0 --rl_grad_clip 5.0 \
    --rl_freeze_encoder true --rl_deterministic_update true \
    --reward_progress_guard_weight 5.0 \
    --reward_progress_guard_stop_tolerance 0.2 \
    --seed 3407 --use_wandb false \
    >> "$TRAIN_LOG" 2>&1
log "RL pilot training process exited successfully"

RL_CHECKPOINT="$(find_rl_checkpoint)"
if [[ -z "$RL_CHECKPOINT" || ! -f "$RL_CHECKPOINT" ]]; then
    log "ERROR: RL epoch 2 checkpoint not found"
    exit 1
fi
log "RL checkpoint selected: $RL_CHECKPOINT"

export NUPLAN_EXP_ROOT="$CLOSED_ROOT"
export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

log "Starting RL pilot closed-loop on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    rl-from-full-mini-b10-pilot1000 \
    "$(dirname "$RL_CHECKPOINT")/args.json" \
    "$RL_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$TRAIN_LOG" 2>&1
log "RL pilot closed-loop complete"

B10_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch10-constant5e5"
DP_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20_vs_diffusion/exp/simulation/closed_loop_nonreactive_agents/original-diffusion-planner-epoch-unknown"
RL_RUN="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents/rl-from-full-mini-b10-pilot1000"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run original_diffusion_planner="$DP_RUN" \
    --run hdp_b_epoch10="$B10_RUN" \
    --run hdp_rl_pilot1000="$RL_RUN" \
    --output "$CLOSED_ROOT/closed_loop_dp_b10_rlpilot1000.json" \
    >> "$TRAIN_LOG" 2>&1
log "DP/B10/RL pilot comparison complete: $CLOSED_ROOT/closed_loop_dp_b10_rlpilot1000.json"
