#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
PILOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1"
RL_RUN="$PILOT/training_log/hdp-rl-from-full-mini-b10-pilot1000-omega001-nodetach/2026-08-21-12:05:28"
RL_CHECKPOINT="$RL_RUN/model_epoch_2_trainloss_-0.0000.pth"
CLOSED_ROOT="$PILOT/rl_from_full_mini_b_epoch10_omega001_nodetach_closed_loop20"
LOG="$PILOT/rl_from_full_mini_b10_closed_loop_only.log"
VAL_FILTER="mini-val-closed-loop-20"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

mkdir -p "$CLOSED_ROOT"
"$PYTHON_BIN" - "$RL_CHECKPOINT" "$RL_RUN/args.json" <<'PY' >> "$LOG" 2>&1
import json
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
required = {"model", "ema_state_dict", "optimizer", "schedule"}
missing = required.difference(checkpoint)
if checkpoint.get("epoch") != 2 or missing:
    raise SystemExit(f"Invalid RL checkpoint: epoch={checkpoint.get('epoch')}, missing={sorted(missing)}")
args = json.load(open(sys.argv[2]))
if args.get("rl_detach_window_size") != 0 or args.get("planning_hybrid_loss") != 0.01:
    raise SystemExit("RL args do not match nodetach omega=0.01 pilot")
print(f"Validated RL checkpoint: {sys.argv[1]}")
print(f"Validated RL args: {sys.argv[2]}")
PY

export NUPLAN_EXP_ROOT="$CLOSED_ROOT"
export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"

log "Starting RL pilot closed-loop on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    rl-from-full-mini-b10-pilot1000 \
    "$RL_RUN/args.json" \
    "$RL_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$LOG" 2>&1
log "RL pilot closed-loop complete"

B10_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_epoch10_a_vs_b/exp/simulation/closed_loop_nonreactive_agents/full-mini-b-epoch10-constant5e5"
DP_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/closed_loop_b_epoch10_vs20_vs_diffusion/exp/simulation/closed_loop_nonreactive_agents/original-diffusion-planner-epoch-unknown"
RL_CLOSED_RUN="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents/rl-from-full-mini-b10-pilot1000"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run original_diffusion_planner="$DP_RUN" \
    --run hdp_b_epoch10="$B10_RUN" \
    --run hdp_rl_pilot1000="$RL_CLOSED_RUN" \
    --output "$CLOSED_ROOT/closed_loop_dp_b10_rlpilot1000.json" \
    >> "$LOG" 2>&1
log "DP/B10/RL pilot comparison complete: $CLOSED_ROOT/closed_loop_dp_b10_rlpilot1000.json"
