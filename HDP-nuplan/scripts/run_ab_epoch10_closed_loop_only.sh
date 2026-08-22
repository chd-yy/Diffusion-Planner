#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
A_RUN="$OUT/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
B_RUN="$OUT/experiment_b_constant5e5_from_epoch4"
A_CHECKPOINT="$A_RUN/model_epoch_10_trainloss_0.0502.pth"
B_CHECKPOINT="$B_RUN/model_epoch_10_trainloss_0.0091.pth"
CLOSED_ROOT="$OUT/closed_loop_epoch10_a_vs_b"
LOG="$CLOSED_ROOT/closed_loop_only.log"
VAL_FILTER="mini-val-closed-loop-20"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

validate_checkpoint() {
    local checkpoint="$1"
    "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
if checkpoint.get("epoch") != 10:
    raise SystemExit(f"expected epoch 10, got {checkpoint.get('epoch')}: {path}")
required = {"model", "ema_state_dict", "optimizer", "schedule"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"checkpoint missing keys {sorted(missing)}: {path}")
print(f"validated epoch=10 checkpoint={path}")
PY
}

cd "$PROJECT_ROOT"
mkdir -p "$CLOSED_ROOT"

validate_checkpoint "$A_CHECKPOINT" >> "$LOG" 2>&1
validate_checkpoint "$B_CHECKPOINT" >> "$LOG" 2>&1
log "A/B Epoch 10 checkpoints validated; resuming at closed-loop only"

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
    >> "$LOG" 2>&1
log "closed-loop A complete"

log "starting closed-loop B on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    full-mini-b-epoch10-constant5e5 \
    "$B_RUN/args.json" \
    "$B_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$LOG" 2>&1
log "closed-loop B complete"

RUN_ROOT="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run experiment_a_epoch10="$RUN_ROOT/full-mini-a-epoch10" \
    --run experiment_b_epoch10_constant5e5="$RUN_ROOT/full-mini-b-epoch10-constant5e5" \
    --output "$CLOSED_ROOT/closed_loop_a_vs_b_epoch10.json" \
    >> "$LOG" 2>&1
log "closed-loop A/B summary complete: $CLOSED_ROOT/closed_loop_a_vs_b_epoch10.json"
