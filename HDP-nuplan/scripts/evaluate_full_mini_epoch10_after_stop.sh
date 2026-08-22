#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
RUN_DIR="$OUT/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
STOP_MARKER="$OUT/stopped_after_epoch10"
VAL_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1"
OUTPUT="$VAL_ROOT/checkpoint_evaluation_full_mini306801_epoch10_repeat3.json"
EVAL_LOG="$VAL_ROOT/checkpoint_evaluation_full_mini306801_epoch10_repeat3.log"
WATCH_LOG="$OUT/evaluate_epoch10_after_stop_monitor.log"

echo "[$(date '+%F %T %Z')] waiting for the Epoch 10 stop marker" >> "$WATCH_LOG"
while [[ ! -f "$STOP_MARKER" ]]; do
    sleep 15
done

checkpoint="$(<"$STOP_MARKER")"
if [[ ! -f "$checkpoint" ]]; then
    echo "[$(date '+%F %T %Z')] checkpoint from stop marker does not exist: $checkpoint" >> "$WATCH_LOG"
    exit 2
fi

echo "[$(date '+%F %T %Z')] starting fixed val1k repeat-3 evaluation: $checkpoint" >> "$WATCH_LOG"
cd "$PROJECT_ROOT/HDP-nuplan"
"$PYTHON_BIN" evaluate_checkpoints.py \
    --args_file "$RUN_DIR/args.json" \
    --checkpoint_dir "$RUN_DIR" \
    --pattern "$(basename "$checkpoint")" \
    --data_dir "$VAL_ROOT/cache" \
    --data_list "$VAL_ROOT/diffusion_planner_validation.json" \
    --batch_size 16 \
    --num_workers 2 \
    --repeats 3 \
    --seed 3407 \
    --device cuda \
    --output "$OUTPUT" \
    > "$EVAL_LOG" 2>&1
echo "[$(date '+%F %T %Z')] evaluation complete: $OUTPUT" >> "$WATCH_LOG"
