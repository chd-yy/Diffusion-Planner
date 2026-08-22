#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
B_RUN="$OUT/experiment_b_constant5e5_from_epoch4"
B20_CHECKPOINT="$B_RUN/model_epoch_20_trainloss_0.0055.pth"
CLOSED_ROOT="$OUT/closed_loop_b_epoch10_vs20"
LOG="$CLOSED_ROOT/closed_loop_b20.log"
VAL_FILTER="mini-val-closed-loop-20"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

mkdir -p "$CLOSED_ROOT"

"$PYTHON_BIN" - "$B20_CHECKPOINT" <<'PY' >> "$LOG" 2>&1
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
if checkpoint.get("epoch") != 20:
    raise SystemExit(f"Expected epoch 20, got {checkpoint.get('epoch')}")
required = {"model", "ema_state_dict", "optimizer", "schedule"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"Checkpoint missing keys: {sorted(missing)}")
print(f"Validated epoch 20 checkpoint: {path}")
PY

export NUPLAN_EXP_ROOT="$CLOSED_ROOT"
export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

log "Starting B Epoch 20 closed-loop on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    full-mini-b-epoch20-constant5e5 \
    "$B_RUN/args.json" \
    "$B20_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$LOG" 2>&1
log "B Epoch 20 closed-loop complete"

B10_RUN="$OUT/closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch10-constant5e5"
B20_RUN="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch20-constant5e5"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run experiment_b_epoch10="$B10_RUN" \
    --run experiment_b_epoch20="$B20_RUN" \
    --output "$CLOSED_ROOT/closed_loop_b_epoch10_vs20.json" \
    >> "$LOG" 2>&1
log "B Epoch 10/20 comparison complete: $CLOSED_ROOT/closed_loop_b_epoch10_vs20.json"
