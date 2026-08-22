#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
OUT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1"
PID_FILE="$OUT/supervised_training_resume.pid"
RUN_DIR="$OUT/supervised_training_full_mini_omega001_nodetach/training_log/hdp-paper-supervised-full-mini-omega001-nodetach/2026-08-20-04:08:42"
MONITOR_LOG="$OUT/stop_at_epoch10_monitor.log"
STOP_MARKER="$OUT/stopped_after_epoch10"

launcher_pid="$(<"$PID_FILE")"
echo "[$(date '+%F %T %Z')] waiting for a valid Epoch 10 checkpoint; launcher PID=$launcher_pid" >> "$MONITOR_LOG"

while kill -0 "$launcher_pid" 2>/dev/null; do
    # 防止 PID 被复用后误杀无关进程。
    if ! tr '\0' ' ' < "/proc/$launcher_pid/cmdline" | grep -Fq "$PROJECT_ROOT/HDP-nuplan/train_predictor.py"; then
        echo "[$(date '+%F %T %Z')] PID no longer belongs to the expected training command" >> "$MONITOR_LOG"
        exit 2
    fi

    shopt -s nullglob
    epoch10_checkpoints=("$RUN_DIR"/model_epoch_10_trainloss_*.pth)
    shopt -u nullglob
    if (( ${#epoch10_checkpoints[@]} > 0 )); then
        checkpoint="${epoch10_checkpoints[0]}"
        # 只有文件已完整写入且确实记录 epoch=10 时，才允许终止训练。
        if "$PYTHON_BIN" - "$checkpoint" <<'PY' >/dev/null 2>&1
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
if checkpoint.get("epoch") != 10:
    raise SystemExit(1)
required = {"model", "ema_state_dict", "optimizer", "schedule"}
if not required.issubset(checkpoint):
    raise SystemExit(1)
PY
        then
            echo "[$(date '+%F %T %Z')] verified $checkpoint; sending SIGTERM to launcher" >> "$MONITOR_LOG"
            kill -TERM "$launcher_pid"
            for _ in $(seq 1 60); do
                if ! kill -0 "$launcher_pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$launcher_pid" 2>/dev/null; then
                echo "[$(date '+%F %T %Z')] launcher did not stop within 60 seconds" >> "$MONITOR_LOG"
                exit 3
            fi
            printf '%s\n' "$checkpoint" > "$STOP_MARKER"
            echo "[$(date '+%F %T %Z')] training stopped safely after Epoch 10" >> "$MONITOR_LOG"
            exit 0
        fi
    fi
    sleep 15
done

echo "[$(date '+%F %T %Z')] training exited before a valid Epoch 10 checkpoint appeared" >> "$MONITOR_LOG"
exit 1
