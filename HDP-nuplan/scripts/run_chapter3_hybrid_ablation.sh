#!/usr/bin/env bash
set -euo pipefail

# Controlled Chapter 3 ablation.  The default mode only validates inputs and
# prints both commands.  Pass --execute explicitly to start the two full runs.

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
DATA_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1"
CACHE="$DATA_ROOT/cache"
MANIFEST="$DATA_ROOT/diffusion_planner_training.json"
VALIDATION_REPORT="$DATA_ROOT/cache_validation_report.json"
OUTPUT_ROOT="$HDP_ROOT/tmp/chapter3_hybrid_ablation_v1"

EXPECTED_SOURCE_SHA="7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45"
EXPECTED_NORMALIZATION_SHA="c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a"
EXPECTED_MANIFEST_SHA="a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0"
EXPECTED_VALIDATION_SHA="48f10e6afdc670448979bab066f12b68fe40c0254b0af8f29b2fe18ee502b8e9"

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
validate_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA"
validate_sha "$VALIDATION_REPORT" "$EXPECTED_VALIDATION_SHA"
[[ -d "$CACHE" ]] || { echo "missing training cache: $CACHE" >&2; exit 5; }

"$PYTHON_BIN" - "$VALIDATION_REPORT" <<'PY'
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
        raise SystemExit(f"validation report mismatch: {key}={actual!r}, expected {expected!r}")
print("validated frozen training inputs: 306801 unique NPZ from 44 logs")
PY

common_args=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nnodes 1 --nproc-per-node 1 --standalone
    "$HDP_ROOT/train_predictor.py"
    --train_set "$CACHE"
    --train_set_list "$MANIFEST"
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
    --planning_detach_window_size 0
    --diffusion_model_type x_start
    --diffusion_supervision_type x_start
    --seed 3407
    --use_ema true
    --use_wandb false
    --ddp true
    --device cuda
)

run_arm() {
    local arm_id="$1"
    local hybrid_weight="$2"
    local name="chapter3-${arm_id}-hybrid${hybrid_weight//./p}"
    local arm_root="$OUTPUT_ROOT/$arm_id"
    local log="$OUTPUT_ROOT/${arm_id}.log"
    local command=(
        "${common_args[@]}"
        --name "$name"
        --save_dir "$arm_root"
        --planning_hybrid_loss "$hybrid_weight"
        --notes "Chapter3 controlled hybrid ablation ${arm_id}; only hybrid weight differs"
    )

    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$MODE" == "--dry-run" ]]; then
        return
    fi

    if [[ -e "$arm_root" || -e "$log" ]]; then
        echo "refusing to reuse existing output for $arm_id: $arm_root or $log" >&2
        exit 6
    fi
    mkdir -p "$arm_root" "$OUTPUT_ROOT"
    env PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee "$log"
}

if [[ "$MODE" == "--execute" ]]; then
    available_kb="$(df -Pk "$HDP_ROOT/tmp" | awk 'NR==2 {print $4}')"
    if (( available_kb < 15 * 1024 * 1024 )); then
        echo "less than 15 GiB free on experiment filesystem; refusing to start" >&2
        exit 7
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        active_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
        if [[ -n "$active_compute" ]]; then
            echo "GPU already has compute processes: $active_compute" >&2
            exit 8
        fi
    fi
fi

echo "C0 command (hybrid=0):"
run_arm C0 0
echo "C1 command (hybrid=0.01):"
run_arm C1 0.01

if [[ "$MODE" == "--execute" ]]; then
    "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = {}
for arm in ("C0", "C1"):
    matches = list((root / arm).glob("training_log/*/*/initialization_report.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one initialization report for {arm}, found {len(matches)}")
    report = json.loads(matches[0].read_text(encoding="utf-8"))
    reports[arm] = {
        key: value["sha256"] for key, value in report["fingerprints"].items()
    }
if reports["C0"] != reports["C1"]:
    raise SystemExit(f"C0/C1 initialization mismatch: {reports}")
(root / "initialization_identity.json").write_text(
    json.dumps({"status": "passed", "fingerprints": reports}, indent=2) + "\n",
    encoding="utf-8",
)
print("C0/C1 initialization identity passed")
PY
fi
