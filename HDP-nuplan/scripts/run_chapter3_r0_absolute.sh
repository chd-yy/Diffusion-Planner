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
ABSOLUTE_NORMALIZATION="$PROJECT_ROOT/normalization.json"
INCREMENT_NORMALIZATION="$HDP_ROOT/normalization.json"
DATA_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1"
CACHE="$DATA_ROOT/cache"
MANIFEST="$DATA_ROOT/diffusion_planner_training.json"
VALIDATION_REPORT="$DATA_ROOT/cache_validation_report.json"
OUTPUT_ROOT="$HDP_ROOT/tmp/chapter3_representation_ablation_v1/R0_absolute"
LOG_ROOT="$HDP_ROOT/tmp/chapter3_representation_ablation_v1"
TRAIN_LOG="$LOG_ROOT/R0_absolute.log"

EXPECTED_SOURCE_SHA="7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"
EXPECTED_ABSOLUTE_NORMALIZATION_SHA="2772eb0bacb57164aeb7ebb2c1a5e6c8410b9a6c3ee64acd78e767c3415f39c6"
EXPECTED_INCREMENT_NORMALIZATION_SHA="c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a"
EXPECTED_MANIFEST_SHA="a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0"
EXPECTED_VALIDATION_SHA="48f10e6afdc670448979bab066f12b68fe40c0254b0af8f29b2fe18ee502b8e9"

# Existing C0 initialization fingerprints.  R0 changes only the target
# representation and target normalizer; the parameter initialization must be
# byte-identical before the first optimizer step.
EXPECTED_MODEL_INIT_SHA="0df19230ccd06ae4a25cf37013a662bcc1aa716f05a9150183b1224705262586"
EXPECTED_ENCODER_INIT_SHA="359289e3b0d7d4bdd8fd3e9316ebc064a038b80a3a00d6cdadf602f60d95c94e"
EXPECTED_DECODER_INIT_SHA="5af7b545094a903f473d8c12da06210f93368c288a5c51753b330826824252e3"

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
validate_sha "$ABSOLUTE_NORMALIZATION" "$EXPECTED_ABSOLUTE_NORMALIZATION_SHA"
validate_sha "$INCREMENT_NORMALIZATION" "$EXPECTED_INCREMENT_NORMALIZATION_SHA"
validate_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA"
validate_sha "$VALIDATION_REPORT" "$EXPECTED_VALIDATION_SHA"
[[ -d "$CACHE" ]] || { echo "missing training cache: $CACHE" >&2; exit 5; }

"$PYTHON_BIN" - \
    "$VALIDATION_REPORT" \
    "$ABSOLUTE_NORMALIZATION" \
    "$INCREMENT_NORMALIZATION" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "status": "passed",
    "manifest_count": 306801,
    "unique_manifest_count": 306801,
    "npz_count": 306801,
    "log_count": 44,
}
for key, expected in required.items():
    actual = report.get(key)
    if actual != expected:
        raise SystemExit(
            f"validation report mismatch: {key}={actual!r}, expected {expected!r}"
        )

absolute = json.load(open(sys.argv[2], encoding="utf-8"))
increment = json.load(open(sys.argv[3], encoding="utf-8"))
if absolute["ego"] != {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]}:
    raise SystemExit(f"unexpected absolute ego normalizer: {absolute['ego']!r}")
if increment["ego"] != {"mean": [0, 0, 0, 0], "std": [0.5, 0.5, 1, 1]}:
    raise SystemExit(f"unexpected increment ego normalizer: {increment['ego']!r}")
for key in absolute:
    if key != "ego" and absolute[key] != increment[key]:
        raise SystemExit(f"observation normalizer differs for {key}")
print("validated R0 inputs and representation-specific target normalizers")
PY

command=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nnodes 1 --nproc-per-node 1 --standalone
    "$HDP_ROOT/train_predictor.py"
    --name chapter3-R0-absolute-hybrid0
    --save_dir "$OUTPUT_ROOT"
    --train_set "$CACHE"
    --train_set_list "$MANIFEST"
    --normalization_file_path "$ABSOLUTE_NORMALIZATION"
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
    --trajectory_representation absolute
    --planning_hybrid_loss 0
    --planning_detach_window_size 0
    --diffusion_model_type x_start
    --diffusion_supervision_type x_start
    --seed 3407
    --expected_initialization_sha256 "$EXPECTED_MODEL_INIT_SHA"
    --use_ema true
    --use_wandb false
    --ddp true
    --device cuda
    --notes "Chapter3 R0 absolute-waypoint representation; controlled against C0 increment representation"
)

printf 'R0 command: '
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$MODE" == "--dry-run" ]]; then
    exit 0
fi

if [[ -e "$OUTPUT_ROOT" || -e "$TRAIN_LOG" ]]; then
    echo "refusing to reuse existing R0 output: $OUTPUT_ROOT or $TRAIN_LOG" >&2
    exit 6
fi

available_kb="$(df -Pk "$HDP_ROOT/tmp" | awk 'NR==2 {print $4}')"
if (( available_kb < 10 * 1024 * 1024 )); then
    echo "less than 10 GiB free on experiment filesystem; refusing to start" >&2
    exit 7
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    active_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -n "$active_compute" ]]; then
        echo "GPU already has compute processes: $active_compute" >&2
        exit 8
    fi
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"${command[@]}" 2>&1 | tee "$TRAIN_LOG"

"$PYTHON_BIN" - \
    "$OUTPUT_ROOT" \
    "$EXPECTED_MODEL_INIT_SHA" \
    "$EXPECTED_ENCODER_INIT_SHA" \
    "$EXPECTED_DECODER_INIT_SHA" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
expected = {
    "model": sys.argv[2],
    "encoder": sys.argv[3],
    "decoder": sys.argv[4],
}
reports = list(root.glob("training_log/*/*/initialization_report.json"))
checkpoints = list(root.glob("training_log/*/*/model_epoch_10_*.pth"))
args_files = list(root.glob("training_log/*/*/args.json"))
if len(reports) != 1 or len(checkpoints) != 1 or len(args_files) != 1:
    raise SystemExit(
        f"expected one report/checkpoint/args, got {len(reports)}/{len(checkpoints)}/{len(args_files)}"
    )
report = json.loads(reports[0].read_text(encoding="utf-8"))
actual = {key: report["fingerprints"][key]["sha256"] for key in expected}
if actual != expected:
    raise SystemExit(f"R0 initialization mismatch: expected={expected}, actual={actual}")
args = json.loads(args_files[0].read_text(encoding="utf-8"))
required_args = {
    "trajectory_representation": "absolute",
    "planning_hybrid_loss": 0.0,
    "planning_detach_window_size": 0,
    "diffusion_model_type": "x_start",
    "diffusion_supervision_type": "x_start",
    "seed": 3407,
}
for key, expected_value in required_args.items():
    if args.get(key) != expected_value:
        raise SystemExit(f"R0 args mismatch: {key}={args.get(key)!r}")
checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
if checkpoint.get("epoch") != 10 or not math.isfinite(float(checkpoint.get("loss"))):
    raise SystemExit(
        f"invalid R0 checkpoint metadata: epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss')}"
    )
print(f"R0 training audit passed: {checkpoints[0]}")
PY
