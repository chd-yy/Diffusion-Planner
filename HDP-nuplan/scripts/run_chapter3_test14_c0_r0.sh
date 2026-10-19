#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner}"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
TEST_WORK_ROOT="$HDP_ROOT/tmp/test14_full_remote_subset"
TEST_DB_ROOT="$TEST_WORK_ROOT/data/cache/test14"
GENERATED_CONFIG_ROOT="$TEST_WORK_ROOT/config"
MANIFEST="$TEST_WORK_ROOT/eval_chunk_manifest.json"
RESULT_MANAGER="$HDP_ROOT/scripts/manage_test14_chunk_results.py"

OLD_EVAL_ROOT="$TEST_WORK_ROOT/three_model_eval"
OUT_ROOT="$HDP_ROOT/tmp/chapter3_test14_c0_r0_v1"
RESULT_ROOT="$HDP_ROOT/doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH3_Test14_C0_R0"
LOG="$OUT_ROOT/run_chapter3_test14_c0_r0.log"
SIMULATION_WORKERS=1

C0_RUN="$HDP_ROOT/tmp/chapter3_hybrid_ablation_v1/C0/training_log/chapter3-C0-hybrid0/2026-09-17-22:47:24"
C0_ARGS="$C0_RUN/args.json"
C0_CKPT="$C0_RUN/model_epoch_10_trainloss_0.0031.pth"
R0_RUN="$HDP_ROOT/tmp/chapter3_representation_ablation_v1/R0_absolute/training_log/chapter3-R0-absolute-hybrid0/2026-10-02-16:52:32"
R0_ARGS="$R0_RUN/args.json"
R0_CKPT="$R0_RUN/model_epoch_10_trainloss_0.0022.pth"

DP_HARD_RUN="$OLD_EVAL_ROOT/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard"
DP_RANDOM_RUN="$OLD_EVAL_ROOT/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-random"
B10_HARD_RUN="$OLD_EVAL_ROOT/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-hard"
B10_RANDOM_RUN="$OLD_EVAL_ROOT/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-random"

EXPECTED_MANIFEST_SHA="a6b495893945d719055d9841a91e1e637063da7f3d6b439c5e82077d5964230d"
EXPECTED_MANAGER_SHA="3ed4d57db3693a542d78b7f36b58fd2d85f87f9b93419b620b7a3006bdaea2cc"
EXPECTED_C0_ARGS_SHA="7eb8522eb8c87d3681d6d920aca4858b24babb614bd82ac8142fa960e4ef0948"
EXPECTED_C0_CKPT_SHA="afb07593c2c989372ad5d11e39b13b532f65a457121b302f11c4b00e5c02aeb0"
EXPECTED_R0_ARGS_SHA="de641ae7e3a647b3b48bd829cfe0a5521e3a075659b22c216997dbdb5c294882"
EXPECTED_R0_CKPT_SHA="0f4343025e0e2031b3c120bac36d46addee60eb86d6bb0976c64b6c513ae1e33"

mkdir -p "$OUT_ROOT" "$RESULT_ROOT"

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

check_sha256() {
    local expected=$1
    local path=$2
    local actual
    [[ -f "$path" ]] || { log "ERROR: missing file $path"; exit 3; }
    actual="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        log "ERROR: SHA256 mismatch for $path expected=$expected actual=$actual"
        exit 4
    }
}

check_sha256 "$EXPECTED_MANIFEST_SHA" "$MANIFEST"
check_sha256 "$EXPECTED_MANAGER_SHA" "$RESULT_MANAGER"
check_sha256 "$EXPECTED_C0_ARGS_SHA" "$C0_ARGS"
check_sha256 "$EXPECTED_C0_CKPT_SHA" "$C0_CKPT"
check_sha256 "$EXPECTED_R0_ARGS_SHA" "$R0_ARGS"
check_sha256 "$EXPECTED_R0_CKPT_SHA" "$R0_CKPT"

"$PYTHON_BIN" - "$MANIFEST" "$TEST_DB_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import yaml

manifest_path = Path(sys.argv[1])
db_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {"test14-hard": 272, "test14-random": 261}
for benchmark, count in expected.items():
    spec = manifest["benchmarks"][benchmark]
    tokens = spec["tokens"]
    chunks = spec["chunks"]
    if len(tokens) != count or len(set(tokens)) != count:
        raise SystemExit(f"invalid {benchmark} identity/count")
    flattened = []
    for chunk in chunks:
        config_path = Path(chunk["config"])
        if not config_path.is_file():
            raise SystemExit(f"missing chunk config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("scenario_tokens") != chunk["tokens"]:
            raise SystemExit(f"chunk config drift: {config_path}")
        if chunk["count"] != len(chunk["tokens"]):
            raise SystemExit(f"chunk count drift: {config_path}")
        flattened.extend(chunk["tokens"])
    if flattened != tokens:
        raise SystemExit(f"chunk order/coverage drift: {benchmark}")
db_count = len(list(db_root.glob("*.db")))
if db_count != 217:
    raise SystemExit(f"expected 217 Test14 DB files, got {db_count}")
print("validated frozen Test14 identity: hard=272 random=261 db=217")
PY

for item in \
    "test14-hard:$DP_HARD_RUN" \
    "test14-random:$DP_RANDOM_RUN" \
    "test14-hard:$B10_HARD_RUN" \
    "test14-random:$B10_RANDOM_RUN"; do
    benchmark="${item%%:*}"
    run_dir="${item#*:}"
    "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir"
done

if [[ "$MODE" == "--dry-run" ]]; then
    log "Dry-run passed: inputs, frozen chunks, DB count, DP and B10 comparator runs are valid"
    exit 0
fi

if [[ "$SIMULATION_WORKERS" -ne 1 ]]; then
    log "ERROR: Test14 requires one simulation worker for metric isolation"
    exit 5
fi
exec 9> "$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
    log "ERROR: another Chapter-3 Test14 pipeline holds the lock"
    exit 6
fi
active_compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$active_compute" ]]; then
    log "ERROR: GPU already has compute processes: $active_compute"
    exit 7
fi
available_kb="$(df -Pk "$OUT_ROOT" | awk 'NR==2 {print $4}')"
if (( available_kb < 10 * 1024 * 1024 )); then
    log "ERROR: less than 10 GiB free on evaluation filesystem"
    exit 8
fi

export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$MAPS_ROOT"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" - \
    "$RESULT_ROOT/protocol.json" "$MANIFEST" "$C0_ARGS" "$C0_CKPT" "$R0_ARGS" "$R0_CKPT" <<'PY'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

output, manifest, c0_args, c0_ckpt, r0_args, r0_ckpt = map(Path, sys.argv[1:])
data = {
    "created_at": datetime.now().astimezone().isoformat(),
    "evaluation_order": [
        "C0/test14-hard", "C0/test14-random",
        "R0/test14-hard", "R0/test14-random",
    ],
    "simulation": "closed_loop_nonreactive_agents",
    "seed": 0,
    "simulation_workers": 1,
    "simulation_log_callback": False,
    "chunk_size": 40,
    "manifest": {"path": str(manifest.resolve()), "sha256": sha(manifest)},
    "models": {
        "C0": {
            "args": str(c0_args.resolve()), "args_sha256": sha(c0_args),
            "checkpoint": str(c0_ckpt.resolve()), "checkpoint_sha256": sha(c0_ckpt),
        },
        "R0": {
            "args": str(r0_args.resolve()), "args_sha256": sha(r0_args),
            "checkpoint": str(r0_ckpt.resolve()), "checkpoint_sha256": sha(r0_ckpt),
        },
    },
    "existing_comparators": ["aligned_original_dp", "hdp_b_epoch10"],
}
output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

run_chunked() {
    local benchmark=$1
    local model_key=$2
    local args_file=$3
    local checkpoint=$4
    local uid="${model_key}-${benchmark}"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"
    local report_dir="$run_dir/chunk_runner_reports"

    if "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1; then
        log "Keeping completed run: $uid"
        return
    fi
    mkdir -p "$report_dir"
    while IFS=$'\t' read -r chunk_index filter_name expected_count; do
        if "$PYTHON_BIN" "$RESULT_MANAGER" validate-chunk-artifacts \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            >> "$LOG" 2>&1; then
            log "Keeping completed chunk: $uid chunk=$chunk_index count=$expected_count"
            continue
        fi
        log "Starting chunk: $uid chunk=$chunk_index count=$expected_count"
        "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=closed_loop_nonreactive_agents \
            planner=hyper_diffusion_planner \
            "planner.hyper_diffusion_planner.config.args_file=$args_file" \
            "planner.hyper_diffusion_planner.ckpt_path=$checkpoint" \
            scenario_builder=nuplan \
            scenario_builder.db_files="$TEST_DB_ROOT" \
            scenario_filter="$filter_name" \
            experiment_uid="$uid" \
            seed=0 \
            worker=single_machine_thread_pool \
            worker.max_workers="$SIMULATION_WORKERS" \
            worker.use_process_pool=false \
            number_of_gpus_allocated_per_simulation=1 \
            number_of_cpus_allocated_per_simulation=1 \
            max_callback_workers=1 \
            disable_callback_parallelization=true \
            enable_simulation_progress_bar=true \
            verbose=true \
            '~callback.simulation_log_callback' \
            "hydra.searchpath=[file://$GENERATED_CONFIG_ROOT,pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
            >> "$LOG" 2>&1

        "$PYTHON_BIN" "$RESULT_MANAGER" archive-chunk \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            --report "$run_dir/runner_report.parquet" \
            >> "$LOG" 2>&1
        log "Completed chunk: $uid chunk=$chunk_index count=$expected_count"
    done < <(
        "$PYTHON_BIN" "$RESULT_MANAGER" list-chunks \
            --manifest "$MANIFEST" --benchmark "$benchmark"
    )
    "$PYTHON_BIN" "$RESULT_MANAGER" merge \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1
    log "Merged and validated complete run: $uid"
}

summary_path() {
    local benchmark=$1
    local suffix=${benchmark#test14-}
    echo "$RESULT_ROOT/test14_${suffix}_four_models.json"
}

summarize_if_available() {
    local benchmark=$1
    local suffix=${benchmark#test14-}
    local c0_run="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/C0-${benchmark}"
    local r0_run="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/R0-${benchmark}"
    local dp_run b10_run output
    if [[ "$benchmark" == "test14-hard" ]]; then
        dp_run="$DP_HARD_RUN"
        b10_run="$B10_HARD_RUN"
    else
        dp_run="$DP_RANDOM_RUN"
        b10_run="$B10_RANDOM_RUN"
    fi
    output="$(summary_path "$benchmark")"
    if ! "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$c0_run" \
        >> "$LOG" 2>&1; then
        return
    fi
    if "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$r0_run" \
        >> "$LOG" 2>&1; then
        "$PYTHON_BIN" "$HDP_ROOT/scripts/summarize_closed_loop_metrics.py" \
            --run "aligned_original_dp=$dp_run" \
            --run "hdp_b_epoch10=$b10_run" \
            --run "C0=$c0_run" \
            --run "R0=$r0_run" \
            --output "$output" >> "$LOG" 2>&1
        "$PYTHON_BIN" "$HDP_ROOT/scripts/analyze_paired_closed_loop.py" \
            --summary "$output" --baseline C0 --candidate R0 \
            --output "$RESULT_ROOT/test14_${suffix}_C0_vs_R0.json" \
            --markdown "$RESULT_ROOT/test14_${suffix}_C0_vs_R0.md" \
            >> "$LOG" 2>&1
    else
        "$PYTHON_BIN" "$HDP_ROOT/scripts/summarize_closed_loop_metrics.py" \
            --run "aligned_original_dp=$dp_run" \
            --run "hdp_b_epoch10=$b10_run" \
            --run "C0=$c0_run" \
            --output "$RESULT_ROOT/test14_${suffix}_DP_B10_C0.json" \
            >> "$LOG" 2>&1
    fi
    log "Updated summary for $benchmark"
}

log "Starting frozen Chapter-3 Test14 sequence: C0 hard -> C0 random -> R0 hard -> R0 random"
run_chunked test14-hard C0 "$C0_ARGS" "$C0_CKPT"
summarize_if_available test14-hard
run_chunked test14-random C0 "$C0_ARGS" "$C0_CKPT"
summarize_if_available test14-random
run_chunked test14-hard R0 "$R0_ARGS" "$R0_CKPT"
summarize_if_available test14-hard
run_chunked test14-random R0 "$R0_ARGS" "$R0_CKPT"
summarize_if_available test14-random
log "Completed C0/R0 Test14 system-level closed-loop evaluation"
