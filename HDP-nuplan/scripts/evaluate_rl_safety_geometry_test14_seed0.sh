#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner}"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}"
WORK_ROOT="$HDP_ROOT/tmp/test14_full_remote_subset"
DB_ROOT="$WORK_ROOT/data/cache/test14"
CONFIG_ROOT="$WORK_ROOT/config"
MANIFEST="$WORK_ROOT/eval_chunk_manifest.json"
MANAGER="$HDP_ROOT/scripts/manage_test14_chunk_results.py"
TRAIN_ROOT="$HDP_ROOT/tmp/rl_safety_geometry_training_0915_2112"
MODEL_RUN="$TRAIN_ROOT/training_log/hdp-rl-safety-repair-controlled/2026-09-15-21:12:50"
MODEL_ARGS="$MODEL_RUN/args.json"
MODEL_CKPT="$MODEL_RUN/model_epoch_2_trainloss_0.0009.pth"
OUT_ROOT="${TEST14_SAFETY_GEOMETRY_EVAL_ROOT:-$HDP_ROOT/tmp/rl_safety_geometry_test14_seed0}"
LOG="$OUT_ROOT/evaluate.log"

mkdir -p "$OUT_ROOT"

check_sha256() {
    local expected=$1
    local path=$2
    local actual
    [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
        echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
        exit 1
    }
}

check_sha256 3a2348a12eb88f32018921514ffd4bce46720392370af9928aa0392ac4e56948 "$MODEL_CKPT"
[[ -f "$MODEL_ARGS" && -f "$MANIFEST" ]] || { echo "Missing args or Test14 manifest" >&2; exit 1; }

export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HDP_ROOT:$PROJECT_ROOT:$DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log() {
    echo "[$(date '+%m月%d日 %H:%M')] $*" | tee -a "$LOG"
}

run_benchmark() {
    local benchmark=$1
    local uid="hdp-rl-safety-geometry-${benchmark}-seed0"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"
    local benchmark_log="$OUT_ROOT/${benchmark}.log"

    if "$PYTHON_BIN" "$MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$benchmark_log" 2>&1; then
        log "Keeping completed run: $benchmark"
        return
    fi
    mkdir -p "$run_dir/chunk_runner_reports"
    while IFS=$'\t' read -r chunk_index filter_name expected_count; do
        if "$PYTHON_BIN" "$MANAGER" validate-chunk-artifacts \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            >> "$benchmark_log" 2>&1; then
            log "Keeping $benchmark chunk=$chunk_index count=$expected_count"
            continue
        fi
        log "Starting $benchmark chunk=$chunk_index count=$expected_count"
        "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=closed_loop_nonreactive_agents \
            planner=hyper_diffusion_planner \
            "planner.hyper_diffusion_planner.config.args_file=$MODEL_ARGS" \
            "planner.hyper_diffusion_planner.ckpt_path=$MODEL_CKPT" \
            scenario_builder=nuplan \
            scenario_builder.db_files="$DB_ROOT" \
            scenario_filter="$filter_name" \
            experiment_uid="$uid" \
            seed=0 \
            worker=single_machine_thread_pool \
            worker.max_workers=1 \
            worker.use_process_pool=false \
            number_of_gpus_allocated_per_simulation=1 \
            number_of_cpus_allocated_per_simulation=1 \
            max_callback_workers=1 \
            disable_callback_parallelization=true \
            enable_simulation_progress_bar=true \
            verbose=true \
            '~callback.simulation_log_callback' \
            "hydra.searchpath=[file://$CONFIG_ROOT,pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
            >> "$benchmark_log" 2>&1
        "$PYTHON_BIN" "$MANAGER" archive-chunk \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            --report "$run_dir/runner_report.parquet" \
            >> "$benchmark_log" 2>&1
        log "Completed $benchmark chunk=$chunk_index"
    done < <("$PYTHON_BIN" "$MANAGER" list-chunks --manifest "$MANIFEST" --benchmark "$benchmark")

    "$PYTHON_BIN" "$MANAGER" merge \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$benchmark_log" 2>&1
    "$PYTHON_BIN" "$HDP_ROOT/scripts/summarize_closed_loop_metrics.py" \
        --run "hdp_rl_safety_geometry=$run_dir" \
        --output "$OUT_ROOT/${benchmark}_candidate.json" \
        >> "$benchmark_log" 2>&1
    log "Merged complete benchmark: $benchmark"
}

log "Validating Test14 database coverage"
"$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_eval_chunks.py" \
    --chunk-size 40 --validate-db >> "$LOG" 2>&1

# Separate processes isolate stateful metric engines. Each process stays single-worker.
run_benchmark test14-hard &
hard_pid=$!
run_benchmark test14-random &
random_pid=$!
status=0
wait "$hard_pid" || status=1
wait "$random_pid" || status=1
[[ "$status" -eq 0 ]] || { log "ERROR: at least one benchmark failed"; exit 1; }

"$PYTHON_BIN" "$HDP_ROOT/scripts/assemble_test14_safety_geometry_comparison.py" \
    --historical-root "$WORK_ROOT/three_model_eval" \
    --candidate-root "$OUT_ROOT" \
    --output "$OUT_ROOT/full_comparison.json" \
    --markdown "$OUT_ROOT/full_comparison.md" >> "$LOG" 2>&1
log "Full Test14 seed0 candidate evaluation complete"
