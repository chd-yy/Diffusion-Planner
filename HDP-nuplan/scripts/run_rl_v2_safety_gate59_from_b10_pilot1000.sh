#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
PILOT="${RL_PILOT_ROOT:-$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1}"
BASELINE_ROOT="${RL_BASELINE_ROOT:-$PILOT}"
EXPECTED_NPZ="${RL_EXPECTED_NPZ:-1000}"
VARIANT_TAG="${RL_VARIANT_TAG:-safetygate04_anchor01}"
SAFETY_GATE_THRESHOLD="${RL_SAFETY_GATE_THRESHOLD:-0.4}"
EXPERT_ANCHOR_WEIGHT="${RL_EXPERT_ANCHOR_WEIGHT:-0.1}"
SEED="${RL_SEED:-3407}"
RUN_ROOT="$PILOT/rl_${VARIANT_TAG}_from_b10"
TRAIN_LOG="$RUN_ROOT/rl_v2_train_and_gate59.log"
CACHE="${RL_CACHE:-$PILOT/cache}"
MANIFEST="$PILOT/diffusion_planner_training.json"
NORMALIZATION="$PROJECT_ROOT/HDP-nuplan/normalization.json"
PRETRAINED="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
B10_ARGS="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/args.json"
B10_RUN20="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch10-constant5e5"
B10_RUN39="$BASELINE_ROOT/rl_vs_b10_closed_loop100/exp/simulation/closed_loop_nonreactive_agents/b10-closed-loop100"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$TRAIN_LOG"
}

find_rl_checkpoint() {
    find "$RUN_ROOT/training_log" -type f -name 'model_epoch_2_trainloss_*.pth' -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

mkdir -p "$RUN_ROOT"

if [[ "${RL_SKIP_TRAINING:-0}" == "1" ]]; then
    RL_CHECKPOINT="$(find_rl_checkpoint)"
    log "Skipping RL training and reusing checkpoint: $RL_CHECKPOINT"
else
"$PYTHON_BIN" - "$PRETRAINED" "$CACHE" "$MANIFEST" "$B10_RUN20" "$B10_RUN39" "$EXPECTED_NPZ" <<'PY' >> "$TRAIN_LOG" 2>&1
import json
import sys
from pathlib import Path
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
missing = {"model", "ema_state_dict"}.difference(checkpoint)
if checkpoint.get("epoch") != 10 or missing:
    raise SystemExit(
        f"Invalid B10 checkpoint: epoch={checkpoint.get('epoch')}, missing={sorted(missing)}"
    )
npz_count = len(list(Path(sys.argv[2]).glob("*.npz")))
manifest_count = len(json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")))
expected = int(sys.argv[6])
if npz_count != expected or manifest_count != expected:
    raise SystemExit(f"Pilot data mismatch: npz={npz_count}, manifest={manifest_count}")
for run_dir in map(Path, sys.argv[4:-1]):
    if not (run_dir / "runner_report.parquet").is_file():
        raise SystemExit(f"Missing fixed-gate baseline: {run_dir}")
print(f"Validated B Epoch 10 checkpoint: {sys.argv[1]}")
print(f"Validated pilot data: {npz_count} NPZ")
print("Validated fixed B10 baselines: 20 + 39 disjoint scenarios")
PY

cd "$PROJECT_ROOT"
log "Starting RL variant=$VARIANT_TAG: safety gate=$SAFETY_GATE_THRESHOLD, expert anchor=$EXPERT_ANCHOR_WEIGHT, seed=$SEED"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor_rl.py" \
    --name "hdp-rl-${VARIANT_TAG}-from-full-mini-b10" \
    --save_dir "$RUN_ROOT" \
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
    --rl_rollout_loss_weight 1.0 --rl_expert_anchor_weight "$EXPERT_ANCHOR_WEIGHT" \
    --rl_max_update_steps_per_epoch 500 \
    --rl_detach_window_size 0 --rl_grad_clip 5.0 \
    --rl_freeze_encoder true --rl_deterministic_update true \
    --reward_progress_guard_weight 5.0 \
    --reward_progress_guard_stop_tolerance 0.2 \
    --reward_safety_gate_threshold "$SAFETY_GATE_THRESHOLD" \
    --reward_safety_gate_margin 1.0 \
    --reward_safety_gate_min_ttc_seconds 1.0 \
    --seed "$SEED" --use_wandb false \
    >> "$TRAIN_LOG" 2>&1
log "RL v2 training completed"

RL_CHECKPOINT="$(find_rl_checkpoint)"
fi
if [[ -z "$RL_CHECKPOINT" || ! -f "$RL_CHECKPOINT" ]]; then
    log "ERROR: RL v2 epoch 2 checkpoint not found"
    exit 1
fi
RL_ARGS="$(dirname "$RL_CHECKPOINT")/args.json"
log "RL v2 checkpoint: $RL_CHECKPOINT"

export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

for spec in "20:mini-val-closed-loop-20:$B10_RUN20" "39:mini-val-closed-loop-100:$B10_RUN39"; do
    IFS=: read -r gate_size scenario_filter b10_run <<< "$spec"
    gate_root="$RUN_ROOT/closed_loop${gate_size}"
    if [[ "$gate_size" == "20" && "${RL_SKIP_GATE20:-0}" == "1" ]]; then
        rl_run="$gate_root/exp/simulation/closed_loop_nonreactive_agents/rl-${VARIANT_TAG}-gate${gate_size}"
        "$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
            --run "hdp_b_epoch10=$b10_run" \
            --run "hdp_rl_variant=$rl_run" \
            --output "$gate_root/b10_vs_rl_${VARIANT_TAG}_gate${gate_size}.json" \
            >> "$TRAIN_LOG" 2>&1
        log "Reused existing RL variant closed-loop gate${gate_size}"
        continue
    fi
    export NUPLAN_EXP_ROOT="$gate_root"
    log "Starting RL v2 closed-loop gate${gate_size}: $scenario_filter"
    bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
        "rl-${VARIANT_TAG}-gate${gate_size}" \
        "$RL_ARGS" \
        "$RL_CHECKPOINT" \
        "$scenario_filter" \
        hdp \
        >> "$TRAIN_LOG" 2>&1
    rl_run="$gate_root/exp/simulation/closed_loop_nonreactive_agents/rl-${VARIANT_TAG}-gate${gate_size}"
    "$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
        --run "hdp_b_epoch10=$b10_run" \
        --run "hdp_rl_variant=$rl_run" \
        --output "$gate_root/b10_vs_rl_${VARIANT_TAG}_gate${gate_size}.json" \
        >> "$TRAIN_LOG" 2>&1
    log "Completed RL v2 closed-loop gate${gate_size}"
    if [[ "$gate_size" == "20" && "${RL_GATE20_ONLY:-0}" == "1" ]]; then
        log "Stopping after gate20 because RL_GATE20_ONLY=1"
        exit 0
    fi
done

"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/merge_paired_closed_loop_summaries.py" \
    --input "$RUN_ROOT/closed_loop20/b10_vs_rl_${VARIANT_TAG}_gate20.json" \
    --input "$RUN_ROOT/closed_loop39/b10_vs_rl_${VARIANT_TAG}_gate39.json" \
    --baseline-label hdp_b_epoch10 \
    --candidate-label hdp_rl_variant \
    --output "$RUN_ROOT/b10_vs_rl_${VARIANT_TAG}_fixed59.json" \
    >> "$TRAIN_LOG" 2>&1
log "RL variant=$VARIANT_TAG fixed 59-scenario gate complete: $RUN_ROOT/b10_vs_rl_${VARIANT_TAG}_fixed59.json"
