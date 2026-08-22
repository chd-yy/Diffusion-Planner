#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
CACHE="$OUT/cache"
MANIFEST="$OUT/diffusion_planner_training.json"
A_RUN="$OUT/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
A_PID_FILE="$OUT/supervised_training_resume.pid"
B_RUN="$OUT/experiment_b_constant5e5_from_epoch4"
B_LOG="$OUT/experiment_b_constant5e5_from_epoch4.log"
B_PID_FILE="$OUT/experiment_b_constant5e5_from_epoch4.pid"
SWITCH_LOG="$OUT/switch_epoch9_a_to_b.log"
VAL_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_val_balanced_1000_seed3407_v1"
A_EVAL="$VAL_ROOT/checkpoint_evaluation_full_mini306801_experiment_a_epoch9_repeat3.json"
B_EVAL="$VAL_ROOT/checkpoint_evaluation_full_mini306801_experiment_b_constant5e5_epoch9_repeat3.json"
COMPARISON="$VAL_ROOT/checkpoint_comparison_full_mini306801_a_vs_b_epoch9_repeat3.json"

launcher_pid="$(<"$A_PID_FILE")"
echo "[$(date '+%F %T %Z')] waiting for a valid Experiment A Epoch 9 checkpoint; launcher PID=$launcher_pid" >> "$SWITCH_LOG"

while kill -0 "$launcher_pid" 2>/dev/null; do
    if ! tr '\0' ' ' < "/proc/$launcher_pid/cmdline" | grep -Fq "$PROJECT_ROOT/HDP-nuplan/train_predictor.py"; then
        echo "[$(date '+%F %T %Z')] A PID no longer belongs to the expected training command" >> "$SWITCH_LOG"
        exit 2
    fi

    shopt -s nullglob
    a_epoch9_checkpoints=("$A_RUN"/model_epoch_9_trainloss_*.pth)
    shopt -u nullglob
    if (( ${#a_epoch9_checkpoints[@]} > 0 )); then
        a_checkpoint="${a_epoch9_checkpoints[0]}"
        if "$PYTHON_BIN" - "$a_checkpoint" <<'PY' >/dev/null 2>&1
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
if checkpoint.get("epoch") != 9:
    raise SystemExit(1)
required = {"model", "ema_state_dict", "optimizer", "schedule"}
if not required.issubset(checkpoint):
    raise SystemExit(1)
PY
        then
            echo "[$(date '+%F %T %Z')] verified A Epoch 9 checkpoint: $a_checkpoint" >> "$SWITCH_LOG"
            kill -TERM "$launcher_pid"
            for _ in $(seq 1 60); do
                if ! kill -0 "$launcher_pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$launcher_pid" 2>/dev/null; then
                echo "[$(date '+%F %T %Z')] A launcher did not stop within 60 seconds" >> "$SWITCH_LOG"
                exit 3
            fi
            printf '%s\n' "$a_checkpoint" > "$OUT/experiment_a_stopped_after_epoch9"
            break
        fi
    fi
    sleep 15
done

if [[ ! -f "$OUT/experiment_a_stopped_after_epoch9" ]]; then
    echo "[$(date '+%F %T %Z')] Experiment A exited before Epoch 9 checkpoint was verified" >> "$SWITCH_LOG"
    exit 4
fi

# B_RUN/latest.pth 已提前固定为与 A Epoch 4 完全相同的 checkpoint，并通过 SHA-256 校验。
echo "[$(date '+%F %T %Z')] starting Experiment B at constant lr=5e-5 from Epoch 4" >> "$SWITCH_LOG"
cd "$PROJECT_ROOT"
env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 1 --standalone \
    "$PROJECT_ROOT/HDP-nuplan/train_predictor.py" \
    --name hdp-full-mini-experiment-b-constant5e5-from-epoch4 \
    --save_dir "$B_RUN" \
    --train_set "$CACHE" \
    --train_set_list "$MANIFEST" \
    --normalization_file_path "$PROJECT_ROOT/HDP-nuplan/normalization.json" \
    --resume_model_path "$B_RUN" \
    --freeze_encoder_epochs 3 \
    --train_epochs 9 --batch_size 8 --learning_rate 5e-5 \
    --warm_up_epoch 1 --reset_lr_schedule_on_resume true \
    --save_utd 1 --num_workers 0 \
    --planning_hybrid_loss 0.01 --planning_detach_window_size 0 \
    --seed 3407 --use_wandb false \
    >> "$B_LOG" 2>&1 &
b_pid=$!
printf '%s\n' "$b_pid" > "$B_PID_FILE"
echo "[$(date '+%F %T %Z')] Experiment B launcher PID=$b_pid" >> "$SWITCH_LOG"
wait "$b_pid"

shopt -s nullglob
b_epoch9_checkpoints=("$B_RUN"/model_epoch_9_trainloss_*.pth)
shopt -u nullglob
if (( ${#b_epoch9_checkpoints[@]} != 1 )); then
    echo "[$(date '+%F %T %Z')] expected exactly one B Epoch 9 checkpoint" >> "$SWITCH_LOG"
    exit 5
fi
b_checkpoint="${b_epoch9_checkpoints[0]}"
echo "[$(date '+%F %T %Z')] Experiment B complete: $b_checkpoint" >> "$SWITCH_LOG"

cd "$PROJECT_ROOT/HDP-nuplan"
"$PYTHON_BIN" evaluate_checkpoints.py \
    --args_file "$A_RUN/args.json" \
    --checkpoint_dir "$A_RUN" \
    --pattern "$(basename "$a_checkpoint")" \
    --data_dir "$VAL_ROOT/cache" \
    --data_list "$VAL_ROOT/diffusion_planner_validation.json" \
    --batch_size 16 --num_workers 2 --repeats 3 --seed 3407 --device cuda \
    --output "$A_EVAL" \
    > "${A_EVAL%.json}.log" 2>&1

"$PYTHON_BIN" evaluate_checkpoints.py \
    --args_file "$B_RUN/args.json" \
    --checkpoint_dir "$B_RUN" \
    --pattern "$(basename "$b_checkpoint")" \
    --data_dir "$VAL_ROOT/cache" \
    --data_list "$VAL_ROOT/diffusion_planner_validation.json" \
    --batch_size 16 --num_workers 2 --repeats 3 --seed 3407 --device cuda \
    --output "$B_EVAL" \
    > "${B_EVAL%.json}.log" 2>&1

"$PYTHON_BIN" - "$A_EVAL" "$B_EVAL" "$COMPARISON" <<'PY'
import json
import sys
from pathlib import Path

a_path, b_path, output_path = map(Path, sys.argv[1:])
a = json.loads(a_path.read_text())
b = json.loads(b_path.read_text())
a_metrics = a["best_metrics"]
b_metrics = b["best_metrics"]
common = sorted(set(a_metrics) & set(b_metrics))
result = {
    "experiment_a": {"description": "Epoch 5-9 at lr=5e-4", "result": str(a_path), "metrics": a_metrics},
    "experiment_b": {"description": "Epoch 5-9 at constant lr=5e-5", "result": str(b_path), "metrics": b_metrics},
    "b_minus_a": {key: b_metrics[key] - a_metrics[key] for key in common},
    "lower_is_better": True,
}
output_path.write_text(json.dumps(result, indent=4) + "\n")
PY
echo "[$(date '+%F %T %Z')] A/B val1k repeat-3 comparison complete: $COMPARISON" >> "$SWITCH_LOG"
