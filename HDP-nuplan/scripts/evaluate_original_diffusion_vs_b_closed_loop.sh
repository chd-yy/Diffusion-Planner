#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
CLOSED_ROOT="$OUT/closed_loop_b_epoch10_vs20_vs_diffusion"
LOG="$CLOSED_ROOT/closed_loop_diffusion.log"
VAL_FILTER="mini-val-closed-loop-20"
DP_ARGS="$PROJECT_ROOT/checkpoints/args.json"
DP_CHECKPOINT="$PROJECT_ROOT/checkpoints/model.pth"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

mkdir -p "$CLOSED_ROOT"

"$PYTHON_BIN" - "$DP_CHECKPOINT" "$DP_ARGS" <<'PY' >> "$LOG" 2>&1
import json
import sys
from pathlib import Path
import torch

checkpoint_path, args_path = map(Path, sys.argv[1:])
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if not {"model", "ema_state_dict"}.issubset(checkpoint):
    raise SystemExit(f"Original Diffusion Planner checkpoint is incomplete: {checkpoint_path}")
args = json.loads(args_path.read_text())
for key in ("future_len", "agent_num", "predicted_neighbor_num"):
    if key not in args:
        raise SystemExit(f"Original planner args missing {key}: {args_path}")
print(f"Validated original Diffusion Planner checkpoint: {checkpoint_path}")
print(f"Validated original planner args: {args_path}")
PY

export NUPLAN_EXP_ROOT="$CLOSED_ROOT"
export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

log "Starting original Diffusion Planner on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    original-diffusion-planner-epoch-unknown \
    "$DP_ARGS" \
    "$DP_CHECKPOINT" \
    "$VAL_FILTER" \
    diffusion \
    >> "$LOG" 2>&1
log "Original Diffusion Planner closed-loop complete"

B10_RUN="$OUT/closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch10-constant5e5"
B20_RUN="$OUT/closed_loop_b_epoch10_vs20/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch20-constant5e5"
DP_RUN="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents/original-diffusion-planner-epoch-unknown"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run original_diffusion_planner="$DP_RUN" \
    --run hdp_b_epoch10="$B10_RUN" \
    --run hdp_b_epoch20="$B20_RUN" \
    --output "$CLOSED_ROOT/closed_loop_diffusion_vs_b_epoch10_epoch20.json" \
    >> "$LOG" 2>&1
log "Three-model comparison complete: $CLOSED_ROOT/closed_loop_diffusion_vs_b_epoch10_epoch20.json"
