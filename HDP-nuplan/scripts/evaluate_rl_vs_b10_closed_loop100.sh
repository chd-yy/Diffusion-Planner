#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1"
CLOSED_ROOT="$OUT/rl_vs_b10_closed_loop100"
LOG="$CLOSED_ROOT/closed_loop100.log"
VAL_FILTER="mini-val-closed-loop-100"
B10_ARGS="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/args.json"
B10_CHECKPOINT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
RL_RUN="$OUT/training_log/hdp-rl-from-full-mini-b10-pilot1000-omega001-nodetach/2026-08-21-12:05:28"
RL_ARGS="$RL_RUN/args.json"
RL_CHECKPOINT="$RL_RUN/model_epoch_2_trainloss_-0.0000.pth"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

mkdir -p "$CLOSED_ROOT"

"$PYTHON_BIN" - "$B10_CHECKPOINT" "$RL_CHECKPOINT" <<'PY' >> "$LOG" 2>&1
import sys
import torch

for path, epoch in ((sys.argv[1], 10), (sys.argv[2], 2)):
    checkpoint = torch.load(path, map_location="cpu")
    required = {"model", "ema_state_dict"}
    missing = required.difference(checkpoint)
    if checkpoint.get("epoch") != epoch or missing:
        raise SystemExit(f"Invalid checkpoint {path}: epoch={checkpoint.get('epoch')}, missing={sorted(missing)}")
    print(f"Validated checkpoint epoch={epoch}: {path}")
PY

export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"
export NUPLAN_EXP_ROOT="$CLOSED_ROOT"

log "Starting B Epoch 10 on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    b10-closed-loop100 \
    "$B10_ARGS" \
    "$B10_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$LOG" 2>&1
log "B Epoch 10 closed-loop100 complete"

log "Starting B Epoch 10 + RL pilot on $VAL_FILTER"
bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
    rl-pilot1000-closed-loop100 \
    "$RL_ARGS" \
    "$RL_CHECKPOINT" \
    "$VAL_FILTER" \
    hdp \
    >> "$LOG" 2>&1
log "RL pilot closed-loop100 complete"

ROOT="$CLOSED_ROOT/exp/simulation/closed_loop_nonreactive_agents"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run hdp_b_epoch10="$ROOT/b10-closed-loop100" \
    --run hdp_rl_pilot1000="$ROOT/rl-pilot1000-closed-loop100" \
    --output "$CLOSED_ROOT/b10_vs_rl_closed_loop100.json" \
    >> "$LOG" 2>&1
log "B10/RL closed-loop100 comparison complete: $CLOSED_ROOT/b10_vs_rl_closed_loop100.json"
