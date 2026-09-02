#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
DATA_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
OUT_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10"
PHASE1_ROOT="$OUT_ROOT/phase1_pretrain"
PHASE2_ROOT="$OUT_ROOT/phase2_finetune"
PHASE1_NAME="original-diffusion-aligned-b10-phase1"
PHASE2_NAME="original-diffusion-aligned-b10-phase2"
PHASE1_LOG="$OUT_ROOT/phase1.log"
PHASE2_LOG="$OUT_ROOT/phase2.log"

mkdir -p "$OUT_ROOT" "$PHASE1_ROOT" "$PHASE2_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

# 第一阶段对应 B 的 Epoch1～4：从同一个原始 DP checkpoint warm-start，
# 前 3 个 epoch 冻结 Encoder，学习率为 5e-4，warm-up 为 2 epoch。
"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node=1 \
    --standalone \
    train_predictor.py \
    --name "$PHASE1_NAME" \
    --save_dir "$PHASE1_ROOT" \
    --train_set "$DATA_ROOT/cache" \
    --train_set_list "$DATA_ROOT/diffusion_planner_training.json" \
    --normalization_file_path "$PROJECT_ROOT/normalization.json" \
    --encoder_pretrained_model_path "$PROJECT_ROOT/checkpoints/model.pth" \
    --freeze_encoder_epochs 3 \
    --future_len 80 \
    --time_len 21 \
    --agent_state_dim 11 \
    --agent_num 32 \
    --predicted_neighbor_num 10 \
    --static_objects_state_dim 10 \
    --static_objects_num 5 \
    --lane_len 20 \
    --lane_state_dim 12 \
    --lane_num 70 \
    --route_len 20 \
    --route_state_dim 12 \
    --route_num 25 \
    --augment_prob 0.5 \
    --use_data_augment true \
    --num_workers 0 \
    --pin-mem \
    --seed 3407 \
    --train_epochs 4 \
    --save_utd 1 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --warm_up_epoch 2 \
    --encoder_drop_path_rate 0.1 \
    --decoder_drop_path_rate 0.1 \
    --alpha_planning_loss 1.0 \
    --device cuda \
    --use_ema true \
    --encoder_depth 3 \
    --decoder_depth 3 \
    --num_heads 6 \
    --hidden_dim 192 \
    --diffusion_model_type x_start \
    --ddp true \
    --use_wandb false \
    > "$PHASE1_LOG" 2>&1

PHASE1_CKPT_DIR=$(find "$PHASE1_ROOT/training_log/$PHASE1_NAME" \
    -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)

if [[ -z "$PHASE1_CKPT_DIR" || ! -f "$PHASE1_CKPT_DIR/latest.pth" ]]; then
    echo "Phase1 checkpoint directory was not found" >&2
    exit 1
fi

echo "[$(date '+%F %T %Z')] Phase1 completed: $PHASE1_CKPT_DIR" | tee -a "$PHASE1_LOG"

# 第二阶段对应 B 从 Epoch4 分叉后的 Epoch5～10：恢复 Epoch4 模型和 EMA，
# 重置为 5e-5/1 epoch warm-up，并训练到总 Epoch10。Encoder 从 Epoch4 起解冻。
"$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node=1 \
    --standalone \
    train_predictor.py \
    --name "$PHASE2_NAME" \
    --save_dir "$PHASE2_ROOT" \
    --train_set "$DATA_ROOT/cache" \
    --train_set_list "$DATA_ROOT/diffusion_planner_training.json" \
    --normalization_file_path "$PROJECT_ROOT/normalization.json" \
    --resume_model_path "$PHASE1_CKPT_DIR" \
    --freeze_encoder_epochs 3 \
    --future_len 80 \
    --time_len 21 \
    --agent_state_dim 11 \
    --agent_num 32 \
    --predicted_neighbor_num 10 \
    --static_objects_state_dim 10 \
    --static_objects_num 5 \
    --lane_len 20 \
    --lane_state_dim 12 \
    --lane_num 70 \
    --route_len 20 \
    --route_state_dim 12 \
    --route_num 25 \
    --augment_prob 0.5 \
    --use_data_augment true \
    --num_workers 0 \
    --pin-mem \
    --seed 3407 \
    --train_epochs 10 \
    --save_utd 1 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --warm_up_epoch 1 \
    --reset_lr_schedule_on_resume true \
    --encoder_drop_path_rate 0.1 \
    --decoder_drop_path_rate 0.1 \
    --alpha_planning_loss 1.0 \
    --device cuda \
    --use_ema true \
    --encoder_depth 3 \
    --decoder_depth 3 \
    --num_heads 6 \
    --hidden_dim 192 \
    --diffusion_model_type x_start \
    --ddp true \
    --use_wandb false \
    > "$PHASE2_LOG" 2>&1

echo "[$(date '+%F %T %Z')] Aligned same-data original Diffusion Planner training completed" | tee -a "$PHASE2_LOG"
