#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
CACHE="$OUT/cache"
MANIFEST="$OUT/diffusion_planner_training.json"
A_RUN="$OUT/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
B_RUN="$OUT/experiment_b_constant5e5_from_epoch4"
A_LOG="$OUT/experiment_a_epoch10_resume.log"
B_LOG="$OUT/experiment_b_epoch10_resume.log"
PIPELINE_LOG="$OUT/train_ab_epoch10_then_closed_loop.log"
CLOSED_ROOT="$OUT/closed_loop_epoch10_a_vs_b"
VAL_FILTER="mini-val-closed-loop-20"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$PIPELINE_LOG"
}

validate_epoch_checkpoint() {
    local checkpoint="$1"
    local expected_epoch="$2"
    "$PYTHON_BIN" - "$checkpoint" "$expected_epoch" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
expected_epoch = int(sys.argv[2])
if checkpoint.get("epoch") != expected_epoch:
    raise SystemExit(f"expected epoch {expected_epoch}, got {checkpoint.get('epoch')}")
required = {"model", "ema_state_dict", "optimizer", "schedule"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"checkpoint missing keys: {sorted(missing)}")
print(f"validated epoch={expected_epoch} checkpoint={sys.argv[1]}")
PY
}

find_one_epoch10() {
    local directory="$1"
    find "$directory" -maxdepth 1 -type f -name 'model_epoch_10_trainloss_*.pth' -print -quit
}

cd "$PROJECT_ROOT"
mkdir -p "$CLOSED_ROOT"

log "starting A Epoch 10 from $A_RUN"
env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-full-mini-experiment-a-epoch10 \
    --save_dir "$A_RUN" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --resume_model_path "$A_RUN" \
    --freeze_encoder_epochs 3 \
    --train_epochs 10 --batch_size 8 --learning_rate 5e-4 \
    --warm_up_epoch 2 --save_utd 1 --num_workers 0 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    > "$A_LOG" 2>&1

A_CHECKPOINT="$(find_one_epoch10 "$A_RUN")"
[[ -n "$A_CHECKPOINT" ]]
validate_epoch_checkpoint "$A_CHECKPOINT" 10 >> "$PIPELINE_LOG"
log "A Epoch 10 complete: $A_CHECKPOINT"

log "starting B Epoch 10 at constant lr=5e-5 from $B_RUN"
env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-full-mini-experiment-b-epoch10-constant5e5 \
    --save_dir "$B_RUN" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --resume_model_path "$B_RUN" \
    --freeze_encoder_epochs 3 \
    --train_epochs 10 --batch_size 8 --learning_rate 5e-5 \
    --warm_up_epoch 1 --reset_lr_schedule_on_resume true \
    --save_utd 1 --num_workers 0 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    > "$B_LOG" 2>&1

B_CHECKPOINT="$(find_one_epoch10 "$B_RUN")"
[[ -n "$B_CHECKPOINT" ]]
validate_epoch_checkpoint "$B_CHECKPOINT" 10 >> "$PIPELINE_LOG"
log "B Epoch 10 complete: $B_CHECKPOINT"

export NUPLAN_EXP_ROOT="$CLOSED_ROOT"
export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

log "starting closed-loop A on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    full-mini-a-epoch10 \
    "$A_RUN/args.json" \
    "$A_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$PIPELINE_LOG" 2>&1
log "closed-loop A complete"

log "starting closed-loop B on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    full-mini-b-epoch10-constant5e5 \
    "$B_RUN/args.json" \
    "$B_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$PIPELINE_LOG" 2>&1
log "closed-loop B complete"

ROOT="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run experiment_a_epoch10="$ROOT/full-mini-a-epoch10" \
    --run experiment_b_epoch10_constant5e5="$ROOT/full-mini-b-epoch10-constant5e5" \
    --output "$CLOSED_ROOT/closed_loop_a_vs_b_epoch10.json" \
    >> "$PIPELINE_LOG" 2>&1
log "closed-loop A/B summary complete: $CLOSED_ROOT/closed_loop_a_vs_b_epoch10.json"
