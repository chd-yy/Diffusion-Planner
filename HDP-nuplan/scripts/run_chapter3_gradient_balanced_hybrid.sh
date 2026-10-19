#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
fi

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
SOURCE_CHECKPOINT="$PROJECT_ROOT/checkpoints/model.pth"
NORMALIZATION="$HDP_ROOT/normalization.json"
TRAIN_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1"
TRAIN_CACHE="$TRAIN_ROOT/cache"
TRAIN_MANIFEST="$TRAIN_ROOT/diffusion_planner_training.json"
TRAIN_REPORT="$TRAIN_ROOT/cache_validation_report.json"
VAL_ROOT="$HDP_ROOT/tmp/mini_val_balanced_1000_seed3407_v1"
C0_RUN="$HDP_ROOT/tmp/chapter3_hybrid_ablation_v1/C0/training_log/chapter3-C0-hybrid0/2026-09-17-22:47:24"
B10_RUN="$TRAIN_ROOT/experiment_b_constant5e5_from_epoch4"
# The external PSSD is exFAT and rejects the colon-containing timestamp
# directory emitted by train_predictor.py.  Checkpoints therefore stay on the
# local experiment filesystem; later simulation caches can still use the PSSD.
ARTIFACT_ROOT="$HDP_ROOT/tmp/chapter3_hybrid_c2_w10_lambda003_v1"
RESULT_ROOT="$HDP_ROOT/doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH3_hybrid_gradient_balanced_C2"

EXPECTED_SOURCE_SHA="7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"
EXPECTED_NORMALIZATION_SHA="c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a"
EXPECTED_MANIFEST_SHA="a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0"
EXPECTED_REPORT_SHA="48f10e6afdc670448979bab066f12b68fe40c0254b0af8f29b2fe18ee502b8e9"
EXPECTED_VAL_MANIFEST_SHA="bf025308d6b2da624836033773f06a95c43930fe247d2a5729571fa0707f61ab"
EXPECTED_C0_CHECKPOINT_SHA="afb07593c2c989372ad5d11e39b13b532f65a457121b302f11c4b00e5c02aeb0"
EXPECTED_B10_CHECKPOINT_SHA="22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"
EXPECTED_INITIAL_MODEL_SHA="0df19230ccd06ae4a25cf37013a662bcc1aa716f05a9150183b1224705262586"

validate_sha() {
    local path="$1"
    local expected="$2"
    [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 3; }
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "SHA256 mismatch for $path: expected=$expected actual=$actual" >&2
        exit 4
    }
}

validate_sha "$SOURCE_CHECKPOINT" "$EXPECTED_SOURCE_SHA"
validate_sha "$NORMALIZATION" "$EXPECTED_NORMALIZATION_SHA"
validate_sha "$TRAIN_MANIFEST" "$EXPECTED_MANIFEST_SHA"
validate_sha "$TRAIN_REPORT" "$EXPECTED_REPORT_SHA"
validate_sha "$VAL_ROOT/diffusion_planner_validation.json" "$EXPECTED_VAL_MANIFEST_SHA"
validate_sha "$C0_RUN/model_epoch_10_trainloss_0.0031.pth" "$EXPECTED_C0_CHECKPOINT_SHA"
validate_sha "$B10_RUN/model_epoch_10_trainloss_0.0091.pth" "$EXPECTED_B10_CHECKPOINT_SHA"
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE" >&2; exit 5; }
[[ -d "$VAL_ROOT/cache" ]] || { echo "missing validation cache: $VAL_ROOT/cache" >&2; exit 5; }

train_command=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nnodes 1 --nproc-per-node 1 --standalone
    "$HDP_ROOT/train_predictor.py"
    --train_set "$TRAIN_CACHE"
    --train_set_list "$TRAIN_MANIFEST"
    --normalization_file_path "$NORMALIZATION"
    --encoder_pretrained_model_path "$SOURCE_CHECKPOINT"
    --freeze_encoder_epochs 3
    --train_epochs 10
    --batch_size 8
    --learning_rate 5e-5
    --warm_up_epoch 1
    --save_utd 1
    --num_workers 0
    --use_data_augment true
    --augment_prob 0.5
    --planning_hybrid_loss 0.003
    --planning_detach_window_size 10
    --diffusion_model_type x_start
    --diffusion_supervision_type x_start
    --trajectory_representation increment
    --seed 3407
    --use_ema true
    --use_wandb false
    --ddp true
    --device cuda
    --name chapter3-C2-gradient-balanced-hybrid-w10-lambda003
    --save_dir "$ARTIFACT_ROOT"
    --notes "Chapter3 C2: gradient-audited hybrid, lambda=0.003, detach window=10; C0-matched protocol"
)

printf 'training command: '
printf '%q ' "${train_command[@]}"
printf '\n'
if [[ "$MODE" == "--dry-run" ]]; then
    exit 0
fi

if [[ -e "$ARTIFACT_ROOT" ]]; then
    echo "refusing to overwrite existing artifact root: $ARTIFACT_ROOT" >&2
    exit 6
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    active_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -n "$active_compute" ]]; then
        echo "GPU already has compute processes: $active_compute" >&2
        exit 7
    fi
fi
available_kb="$(df -Pk "$(dirname "$ARTIFACT_ROOT")" | awk 'NR==2 {print $4}')"
if (( available_kb < 20 * 1024 * 1024 )); then
    echo "less than 20 GiB free on training artifact filesystem" >&2
    exit 8
fi

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"
printf '%q ' "${train_command[@]}" > "$ARTIFACT_ROOT/training_command.txt"
printf '\n' >> "$ARTIFACT_ROOT/training_command.txt"
env PYTHONUNBUFFERED=1 "${train_command[@]}" 2>&1 | tee "$ARTIFACT_ROOT/training.log"

mapfile -t args_matches < <(find "$ARTIFACT_ROOT/training_log" -type f -name args.json | sort)
mapfile -t init_matches < <(find "$ARTIFACT_ROOT/training_log" -type f -name initialization_report.json | sort)
mapfile -t checkpoint_matches < <(find "$ARTIFACT_ROOT/training_log" -type f -name 'model_epoch_10_*.pth' | sort)
if [[ "${#args_matches[@]}" -ne 1 || "${#init_matches[@]}" -ne 1 || "${#checkpoint_matches[@]}" -ne 1 ]]; then
    echo "expected exactly one args/init/Epoch10 artifact" >&2
    exit 9
fi
C2_ARGS="${args_matches[0]}"
C2_INIT="${init_matches[0]}"
C2_CHECKPOINT="${checkpoint_matches[0]}"

"$PYTHON_BIN" - "$C2_INIT" "$EXPECTED_INITIAL_MODEL_SHA" "$C2_CHECKPOINT" "$RESULT_ROOT/identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

initialization = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
actual = initialization["fingerprints"]["model"]["sha256"]
if actual != expected:
    raise SystemExit(f"C2 initial model differs from C0: expected={expected} actual={actual}")
checkpoint = Path(sys.argv[3])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
result = {
    "status": "passed",
    "C0_initial_model_sha256": expected,
    "C2_initial_model_sha256": actual,
    "C2_checkpoint": str(checkpoint.resolve()),
    "C2_checkpoint_sha256": digest,
}
Path(sys.argv[4]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
PY

PYTHONPATH="$HDP_ROOT" "$PYTHON_BIN" "$HDP_ROOT/scripts/evaluate_supervised_open_loop.py" \
    --args-file "$C0_RUN/args.json" \
    --model "C0=$C0_RUN/model_epoch_10_trainloss_0.0031.pth" \
    --model "historical_B10=$B10_RUN/model_epoch_10_trainloss_0.0091.pth" \
    --model "C2=$C2_CHECKPOINT" \
    --model-args "historical_B10=$B10_RUN/args.json" \
    --model-args "C2=$C2_ARGS" \
    --data-dir "$VAL_ROOT/cache" \
    --data-list "$VAL_ROOT/diffusion_planner_validation.json" \
    --batch-size 16 \
    --num-workers 2 \
    --repeats 3 \
    --seed 3407 \
    --num-samples 1 \
    --diffusion-steps 10 \
    --sampling-noise-scale 0.1 \
    --device cuda \
    --output "$RESULT_ROOT/open_loop_C0_B10_C2.json" \
    2>&1 | tee "$RESULT_ROOT/open_loop_evaluation.log"

PYTHONPATH="$HDP_ROOT" "$PYTHON_BIN" "$HDP_ROOT/scripts/audit_hybrid_loss_gradients.py" \
    --args-file "$C0_RUN/args.json" \
    --model "C2=$C2_CHECKPOINT" \
    --model-args "C2=$C2_ARGS" \
    --data-dir "$VAL_ROOT/cache" \
    --data-list "$VAL_ROOT/diffusion_planner_validation.json" \
    --batch-size 4 \
    --max-batches 64 \
    --num-workers 0 \
    --hybrid-weight 0.003 \
    --window-sizes 0 10 \
    --parameter-scope all \
    --device cuda \
    --output "$RESULT_ROOT/C2_posttraining_gradient_audit.json" \
    2>&1 | tee "$RESULT_ROOT/C2_posttraining_gradient_audit.log"

echo "C2 training, initialization identity check, open-loop evaluation, and post-training gradient audit completed."
