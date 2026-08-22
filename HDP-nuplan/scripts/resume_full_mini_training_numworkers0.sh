#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
CACHE="$OUT/cache"
MANIFEST="$OUT/diffusion_planner_training.json"
TRAIN_ROOT="$OUT/supervised_training_full_mini_omega001_nodetach"
RESUME_DIR="$TRAIN_ROOT/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
RESUME_LOG="$OUT/supervised_full_mini_omega001_nodetach_resume_numworkers0.log"
PID_FILE="$OUT/supervised_training_resume.pid"

cd "$PROJECT_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
printf '[%s] resuming with num_workers=0 from %s\n' "$(date '+%F %T %Z')" "$RESUME_DIR" >> "$RESUME_LOG"

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-paper-supervised-full-mini-omega001-nodetach \
    --save_dir "$TRAIN_ROOT" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --resume_model_path "$RESUME_DIR" \
    --freeze_encoder_epochs 3 \
    --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
    --warm_up_epoch 2 --save_utd 1 --num_workers 0 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    >> "$RESUME_LOG" 2>&1
