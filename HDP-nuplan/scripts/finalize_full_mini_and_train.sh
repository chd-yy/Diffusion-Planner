#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$(realpath "${1:?usage: finalize_full_mini_and_train.sh OUTPUT_ROOT}")"
CACHE="$OUT/cache"
WORKERS="$OUT/parallel_workers_4"
MANIFEST="$OUT/diffusion_planner_training.json"
MERGED_REPORT="$OUT/merged_processing_report.json"
VALIDATION_REPORT="$OUT/cache_validation_report.json"
TRAIN_ROOT="$OUT/supervised_training_full_mini_omega001_nodetach"
TRAIN_LOG="$OUT/supervised_full_mini_omega001_nodetach_train.log"

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T %Z')] waiting for all preprocessing shards"
while true; do
    complete=1
    for index in 0 1 2 3; do
        report="$WORKERS/$(printf 'shard_%05d' "$index")/processing_report.json"
        if [[ ! -f "$report" ]] || ! jq -e '.status == "complete" and .failed == 0' "$report" >/dev/null; then
            complete=0
            break
        fi
    done
    if [[ "$complete" -eq 1 ]]; then
        break
    fi
    pid_file="$WORKERS/shard_00003/runner_resume.pid"
    if [[ -f "$pid_file" ]] && ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
        echo "[$(date '+%F %T %Z')] shard_00003 stopped before completion" >&2
        exit 2
    fi
    sleep 30
done

echo "[$(date '+%F %T %Z')] all shards complete; merging"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/merge_preprocessing_shards.py" \
    --shards_root "$WORKERS" \
    --shared_cache "$CACHE" \
    --output_manifest "$MANIFEST" \
    --output_report "$MERGED_REPORT"

echo "[$(date '+%F %T %Z')] validating 306801 NPZ files"
"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/validate_processed_cache.py" \
    --cache_dir "$CACHE" \
    --manifest "$MANIFEST" \
    --sampling_report "$OUT/sampling_report.json" \
    --expected_count 306801 \
    --expected_log_count 44 \
    --output "$VALIDATION_REPORT"
jq -e '.status == "passed" and .manifest_count == 306801 and .npz_count == 306801' "$VALIDATION_REPORT" >/dev/null

mkdir -p "$TRAIN_ROOT"
echo "[$(date '+%F %T %Z')] validation passed; starting training"
setsid env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-paper-supervised-full-mini-omega001-nodetach \
    --save_dir "$TRAIN_ROOT" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --encoder_pretrained_model_path "$PROJECT_ROOT/checkpoints/model.pth" \
    --freeze_encoder_epochs 3 \
    --train_epochs 20 --batch_size 8 --learning_rate 5e-4 \
    --warm_up_epoch 2 --save_utd 1 --num_workers 4 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$OUT/supervised_training.pid"
echo "[$(date '+%F %T %Z')] training launcher PID=$TRAIN_PID"
