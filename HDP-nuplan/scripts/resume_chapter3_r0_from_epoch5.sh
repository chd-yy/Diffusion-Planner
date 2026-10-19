#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
RUN_DIR="$HDP_ROOT/tmp/chapter3_representation_ablation_v1/R0_absolute/training_log/chapter3-R0-absolute-hybrid0/2026-10-02-16:52:32"
DATA_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1"
RESUME_LOG="$HDP_ROOT/tmp/chapter3_representation_ablation_v1/R0_absolute_resume_epoch5.log"

"$PYTHON_BIN" - "$RUN_DIR/latest.pth" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("epoch") != 5:
    raise SystemExit(f"expected epoch-5 checkpoint, got {checkpoint.get('epoch')!r}")
for key in ("model", "ema_state_dict", "optimizer", "schedule"):
    if key not in checkpoint:
        raise SystemExit(f"resume checkpoint is missing {key}")
print("validated complete R0 epoch-5 resume checkpoint")
PY

if pgrep -af "train_predictor.py.*chapter3-R0-absolute-hybrid0" >/dev/null; then
    echo "R0 training process already exists; refusing duplicate launch" >&2
    exit 8
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$HDP_ROOT/train_predictor.py" \
    --name chapter3-R0-absolute-hybrid0 \
    --save_dir "$HDP_ROOT/tmp/chapter3_representation_ablation_v1/R0_absolute" \
    --train_set "$DATA_ROOT/cache" \
    --train_set_list "$DATA_ROOT/diffusion_planner_training.json" \
    --normalization_file_path "$PROJECT_ROOT/normalization.json" \
    --resume_model_path "$RUN_DIR" \
    --freeze_encoder_epochs 3 \
    --train_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --warm_up_epoch 1 \
    --save_utd 1 \
    --num_workers 0 \
    --use_data_augment true \
    --augment_prob 0.5 \
    --trajectory_representation absolute \
    --planning_hybrid_loss 0 \
    --planning_detach_window_size 0 \
    --diffusion_model_type x_start \
    --diffusion_supervision_type x_start \
    --seed 3407 \
    --use_ema true \
    --use_wandb false \
    --ddp true \
    --device cuda \
    --notes "Chapter3 R0 exact resume from complete epoch-5 checkpoint" \
    2>&1 | tee -a "$RESUME_LOG"
