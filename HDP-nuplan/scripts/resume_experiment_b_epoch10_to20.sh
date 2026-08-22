#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
B_RUN="$OUT/experiment_b_constant5e5_from_epoch4"
CACHE="$OUT/cache"
MANIFEST="$OUT/diffusion_planner_training.json"
LOG="$OUT/experiment_b_epoch10_to20.log"

echo "[$(date '+%F %T %Z')] Starting B resume: epoch 10 -> epoch 20" | tee -a "$LOG"

"$PYTHON_BIN" - "$B_RUN/model_epoch_10_trainloss_0.0091.pth" <<'PY' >> "$LOG" 2>&1
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
if checkpoint.get("epoch") != 10:
    raise SystemExit(f"Expected epoch 10, got {checkpoint.get('epoch')}")
required = {"model", "ema_state_dict", "optimizer", "schedule"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"Checkpoint missing keys: {sorted(missing)}")
print(f"Validated resume checkpoint: {path}")
PY

cd "$PROJECT_ROOT"
env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-full-mini-experiment-b-epoch20-constant5e5 \
    --save_dir "$B_RUN" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --resume_model_path "$B_RUN" \
    --freeze_encoder_epochs 3 \
    --train_epochs 20 --batch_size 8 --learning_rate 5e-5 \
    --warm_up_epoch 1 --reset_lr_schedule_on_resume true \
    --save_utd 1 --num_workers 0 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    >> "$LOG" 2>&1

echo "[$(date '+%F %T %Z')] B epoch 20 training process exited successfully" | tee -a "$LOG"
