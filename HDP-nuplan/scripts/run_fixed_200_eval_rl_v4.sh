#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
MAP_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
OUT_ROOT="${RL_FIXED200_ROOT:-$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_fixed200_eval}"
FILTER="mini-val-fixed-200-rl-v4"

B10_ARGS="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42/args.json"
B10_CHECKPOINT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
SEED42_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed42_from_b10"
SEED2026_RUN="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10"

find_checkpoint() {
    find "$1/training_log" -type f -name 'model_epoch_2_trainloss_*.pth' -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_eval() {
    local uid="$1"
    local args="$2"
    local checkpoint="$3"
    log "Starting $uid on $FILTER"
    bash "$PROJECT_ROOT/HDP-nuplan/scripts/run_mini_closed_loop.sh" \
        "$uid" "$args" "$checkpoint" "$FILTER" hdp >> "$LOG" 2>&1
    log "Completed $uid"
}

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/fixed200_eval.log"
SEED42_CHECKPOINT="$(find_checkpoint "$SEED42_RUN")"
SEED2026_CHECKPOINT="$(find_checkpoint "$SEED2026_RUN")"
SEED42_ARGS="$(dirname "$SEED42_CHECKPOINT")/args.json"
SEED2026_ARGS="$(dirname "$SEED2026_CHECKPOINT")/args.json"

for path in "$B10_ARGS" "$B10_CHECKPOINT" "$SEED42_ARGS" "$SEED42_CHECKPOINT" "$SEED2026_ARGS" "$SEED2026_CHECKPOINT"; do
    [[ -f "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 1; }
done

export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$MAP_ROOT"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_MINI_DB_ROOT="$MINI_DB_ROOT"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log "Fixed 200 paired evaluation started"
log "Filter: $PROJECT_ROOT/HDP-nuplan/hdp_nuplan/config/scenario_filter/mini-val-fixed-200-rl-v4.yaml"
run_eval b10-closed-loop200 "$B10_ARGS" "$B10_CHECKPOINT"
run_eval rl-seed42-closed-loop200 "$SEED42_ARGS" "$SEED42_CHECKPOINT"
run_eval rl-seed2026-closed-loop200 "$SEED2026_ARGS" "$SEED2026_CHECKPOINT"

ROOT="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run hdp_b_epoch10="$ROOT/b10-closed-loop200" \
    --run hdp_rl_seed42="$ROOT/rl-seed42-closed-loop200" \
    --run hdp_rl_seed2026="$ROOT/rl-seed2026-closed-loop200" \
    --output "$OUT_ROOT/b10_vs_rl_fixed200.json" >> "$LOG" 2>&1
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/analyze_fixed200_paired.py" \
    --summary "$OUT_ROOT/b10_vs_rl_fixed200.json" \
    --output "$OUT_ROOT/b10_vs_rl_fixed200_analysis.json" \
    --markdown "$OUT_ROOT/b10_vs_rl_fixed200_analysis.md" >> "$LOG" 2>&1
log "Fixed 200 paired evaluation and analysis completed: $OUT_ROOT/b10_vs_rl_fixed200_analysis.md"
