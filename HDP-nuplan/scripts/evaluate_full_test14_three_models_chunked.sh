#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner}"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
TEST_WORK_ROOT="${TEST14_WORK_ROOT:-$HDP_ROOT/tmp/test14_full_remote_subset}"
TEST_DB_ROOT="${TEST14_DB_ROOT:-$TEST_WORK_ROOT/data/cache/test14}"
GENERATED_CONFIG_ROOT="$TEST_WORK_ROOT/config"
OUT_ROOT="${TEST14_EVAL_ROOT:-$TEST_WORK_ROOT/three_model_eval}"
LOG="$OUT_ROOT/evaluate_full_test14_three_models_chunked.log"
MANIFEST="$TEST_WORK_ROOT/eval_chunk_manifest.json"
RECOVERY_MANIFEST="$TEST_WORK_ROOT/aligned_original_hard_recovery_manifest.json"
RESULT_MANAGER="$HDP_ROOT/scripts/manage_test14_chunk_results.py"
SIMULATION_WORKERS=1

DP_RUN="$HDP_ROOT/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09"
DP_ARGS="${DP_ARGS:-$DP_RUN/args.json}"
DP_CKPT="${DP_CKPT:-$DP_RUN/model_epoch_10_trainloss_0.0618.pth}"
B10_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
B10_ARGS="${B10_ARGS:-$B10_ROOT/args.json}"
B10_CKPT="${B10_CKPT:-$B10_ROOT/model_epoch_10_trainloss_0.0091.pth}"
RL_RUN="$HDP_ROOT/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45"
RL_ARGS="${RL_ARGS:-$RL_RUN/args.json}"
RL_CKPT="${RL_CKPT:-$RL_RUN/model_epoch_2_trainloss_0.0003.pth}"

mkdir -p "$OUT_ROOT"

# NuPlan reuses one stateful MetricsEngine per scenario type. With the thread-pool
# worker, metric callbacks run synchronously inside each simulation thread, so two
# simulation workers can mutate the same metric objects concurrently. Keep this
# pipeline strictly single-worker unless the metric engine is isolated per runner.
if [[ "$SIMULATION_WORKERS" -ne 1 ]]; then
    echo "ERROR: Test14 evaluation requires SIMULATION_WORKERS=1 for metric isolation" >&2
    exit 1
fi

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

check_sha256() {
    local expected=$1
    local path=$2
    local actual
    [[ -f "$path" ]] || { log "ERROR: missing file $path"; exit 1; }
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
        log "ERROR: SHA256 mismatch for $path (expected=$expected actual=$actual)"
        exit 1
    }
}

check_sha256 5908f185cf61fdbd96f8384690da7a61ba2e55c1217009471c49eaa406850eec "$DP_ARGS"
check_sha256 37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786 "$DP_CKPT"
check_sha256 ba6848d19bf5ea486fcc9ff31299cadccd8778c927fdf7e961e65381b2caa39f "$B10_ARGS"
check_sha256 22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce "$B10_CKPT"
check_sha256 1c0736ad108ff79fe03711b14886c0a97e77e284b2d5ab252422fec0dcdeda9f "$RL_ARGS"
check_sha256 8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c "$RL_CKPT"

export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$MAPS_ROOT"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

run_chunked() {
    local benchmark=$1
    local model_key=$2
    local planner_kind=$3
    local args_file=$4
    local checkpoint=$5
    local run_manifest=${6:-$MANIFEST}
    local uid="${model_key}-${benchmark}"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"
    local report_dir="$run_dir/chunk_runner_reports"
    local planner_config
    local planner_key

    if "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$run_manifest" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1; then
        log "Keeping completed chunked run: $uid"
        return
    fi

    if [[ "$planner_kind" == "diffusion" ]]; then
        planner_config=diffusion_planner
        planner_key=diffusion_planner
    else
        planner_config=hyper_diffusion_planner
        planner_key=hyper_diffusion_planner
    fi
    mkdir -p "$report_dir"

    while IFS=$'\t' read -r chunk_index filter_name expected_count; do
        if "$PYTHON_BIN" "$RESULT_MANAGER" validate-chunk-artifacts \
            --manifest "$run_manifest" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" >> "$LOG" 2>&1; then
            log "Keeping completed chunk: $uid chunk=$chunk_index count=$expected_count"
            continue
        fi

        log "Starting chunk: $uid chunk=$chunk_index count=$expected_count"
        "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=closed_loop_nonreactive_agents \
            planner="$planner_config" \
            "planner.$planner_key.config.args_file=$args_file" \
            "planner.$planner_key.ckpt_path=$checkpoint" \
            scenario_builder=nuplan \
            scenario_builder.db_files="$TEST_DB_ROOT" \
            scenario_filter="$filter_name" \
            experiment_uid="$uid" \
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
            --manifest "$run_manifest" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            --report "$run_dir/runner_report.parquet" \
            >> "$LOG" 2>&1
        log "Completed chunk: $uid chunk=$chunk_index count=$expected_count"
    done < <(
        "$PYTHON_BIN" "$RESULT_MANAGER" list-chunks \
            --manifest "$run_manifest" --benchmark "$benchmark"
    )

    "$PYTHON_BIN" "$RESULT_MANAGER" merge \
        --manifest "$run_manifest" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1
    log "Merged and validated complete run: $uid"
}

summarize_benchmark() {
    local benchmark=$1
    local output="$OUT_ROOT/${benchmark}_three_models.json"
    "$PYTHON_BIN" "$HDP_ROOT/scripts/summarize_closed_loop_metrics.py" \
        --run "aligned_original_dp=$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-${benchmark}" \
        --run "hdp_b_epoch10=$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-${benchmark}" \
        --run "hdp_rl_epoch2=$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-${benchmark}" \
        --output "$output" >> "$LOG" 2>&1
    "$PYTHON_BIN" "$HDP_ROOT/scripts/analyze_paired_closed_loop.py" \
        --summary "$output" --baseline hdp_b_epoch10 --candidate hdp_rl_epoch2 \
        --output "$OUT_ROOT/${benchmark}_b10_vs_rl_epoch2_analysis.json" \
        --markdown "$OUT_ROOT/${benchmark}_b10_vs_rl_epoch2_analysis.md" \
        >> "$LOG" 2>&1
    log "Summarized benchmark: $benchmark"
}

log "Generating and validating 40-scenario Test14 chunks"
"$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_eval_chunks.py" \
    --chunk-size 40 --validate-db >> "$LOG" 2>&1

ORIGINAL_HARD_RUN="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard"
if [[ ! -f "$RECOVERY_MANIFEST" ]]; then
    "$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_metric_recovery_chunks.py" \
        --base-manifest "$MANIFEST" --benchmark test14-hard \
        --metrics-dir "$ORIGINAL_HARD_RUN/metrics" --planner-name diffusion_planner \
        --chunk-size 40 --manifest "$RECOVERY_MANIFEST" >> "$LOG" 2>&1
else
    log "Keeping existing aligned-original hard recovery manifest"
fi
run_chunked test14-hard aligned-original-dp diffusion "$DP_ARGS" "$DP_CKPT" "$RECOVERY_MANIFEST"

run_chunked test14-hard hdp-b-epoch10 hdp "$B10_ARGS" "$B10_CKPT"
run_chunked test14-hard hdp-rl-epoch2 hdp "$RL_ARGS" "$RL_CKPT"
summarize_benchmark test14-hard

run_chunked test14-random aligned-original-dp diffusion "$DP_ARGS" "$DP_CKPT"
run_chunked test14-random hdp-b-epoch10 hdp "$B10_ARGS" "$B10_CKPT"
run_chunked test14-random hdp-rl-epoch2 hdp "$RL_ARGS" "$RL_CKPT"
summarize_benchmark test14-random

log "Full Test14 three-model chunked evaluation completed"
