#!/usr/bin/env bash
set -euo pipefail

# Run only the full-306801 post-trained candidate on Test14.  The historical
# B10 arm is reused only after its identity and complete per-scenario outputs
# have passed strict validation against the frozen Test14 manifest.

HDP_ROOT="${HDP_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan}"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
TEST_WORK_ROOT="${TEST14_WORK_ROOT:-$HDP_ROOT/tmp/test14_full_remote_subset}"
TEST_DB_ROOT="${TEST14_DB_ROOT:-$TEST_WORK_ROOT/data/cache/test14}"
GENERATED_CONFIG_ROOT="$TEST_WORK_ROOT/config"
MANIFEST="$TEST_WORK_ROOT/eval_chunk_manifest.json"
RESULT_MANAGER="$HDP_ROOT/scripts/manage_test14_chunk_results.py"

OUT_ROOT="${TEST14_EVAL_ROOT:-/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006}"
LOG="$OUT_ROOT/evaluate_full306801_test14_single_arm.log"
UID_PREFIX="full306801-posttrain"

B10_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
B10_ARGS="$B10_ROOT/args.json"
B10_CKPT="$B10_ROOT/model_epoch_10_trainloss_0.0091.pth"
B10_EVAL_ROOT="$TEST_WORK_ROOT/three_model_eval/exp/simulation/closed_loop_nonreactive_agents"

NEW_SOURCE_RUN="/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/training_outputs/EXP_CH4_guided_safeset_full306801_seed2026_v3_1006/training_log/chapter4-safety_reference-noise0p1-std1em06-steps0-seed2026-guided-safeset88-full306801/2026-10-06-00-13-19"
# Hydra's override lexer rejects the Chinese mount-point characters when the
# paths are passed on the command line.  These aliases point to the exact same
# files and keep the override text ASCII-only; content identity is checked below.
MODEL_ALIAS_ROOT="$TEST_WORK_ROOT/model_aliases/full306801_posttrain_1006"
NEW_ARGS="${NEW_ARGS:-$MODEL_ALIAS_ROOT/args.json}"
NEW_CKPT="${NEW_CKPT:-$MODEL_ALIAS_ROOT/model_epoch_2_trainloss_0.0005.pth}"

mkdir -p "$OUT_ROOT"

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

validate_historical_b10() {
    local benchmark=$1
    local run_dir="$B10_EVAL_ROOT/hdp-b-epoch10-${benchmark}"
    "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1
    log "Qualified historical B10 reuse: $benchmark"
}

run_candidate() {
    local benchmark=$1
    local uid="${UID_PREFIX}-${benchmark}"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"

    if "$PYTHON_BIN" "$RESULT_MANAGER" validate-run \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1; then
        log "Keeping completed candidate run: $uid"
        return
    fi

    mkdir -p "$run_dir/chunk_runner_reports"
    while IFS=$'\t' read -r chunk_index filter_name expected_count; do
        if "$PYTHON_BIN" "$RESULT_MANAGER" validate-chunk-artifacts \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            >> "$LOG" 2>&1; then
            log "Keeping completed candidate chunk: $uid chunk=$chunk_index count=$expected_count"
            continue
        fi

        log "Starting candidate chunk: $uid chunk=$chunk_index count=$expected_count"
        "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=closed_loop_nonreactive_agents \
            planner=hyper_diffusion_planner \
            "planner.hyper_diffusion_planner.config.args_file=$NEW_ARGS" \
            "planner.hyper_diffusion_planner.ckpt_path=$NEW_CKPT" \
            scenario_builder=nuplan \
            scenario_builder.db_files="$TEST_DB_ROOT" \
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
            "hydra.searchpath=[file://$GENERATED_CONFIG_ROOT,pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
            >> "$LOG" 2>&1

        "$PYTHON_BIN" "$RESULT_MANAGER" archive-chunk \
            --manifest "$MANIFEST" --benchmark "$benchmark" \
            --chunk-index "$chunk_index" --run-dir "$run_dir" \
            --report "$run_dir/runner_report.parquet" \
            >> "$LOG" 2>&1
        log "Completed candidate chunk: $uid chunk=$chunk_index count=$expected_count"
    done < <(
        "$PYTHON_BIN" "$RESULT_MANAGER" list-chunks \
            --manifest "$MANIFEST" --benchmark "$benchmark"
    )

    "$PYTHON_BIN" "$RESULT_MANAGER" merge \
        --manifest "$MANIFEST" --benchmark "$benchmark" --run-dir "$run_dir" \
        >> "$LOG" 2>&1
    log "Merged and validated complete candidate run: $uid"
}

summarize_pair() {
    local benchmark=$1
    local b10_dir="$B10_EVAL_ROOT/hdp-b-epoch10-${benchmark}"
    local candidate_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/${UID_PREFIX}-${benchmark}"
    local summary="$OUT_ROOT/${benchmark}_historical_B10_vs_full306801.json"

    "$PYTHON_BIN" "$HDP_ROOT/scripts/summarize_closed_loop_metrics.py" \
        --run "historical_B10=$b10_dir" \
        --run "full306801_posttrain=$candidate_dir" \
        --output "$summary" >> "$LOG" 2>&1
    "$PYTHON_BIN" "$HDP_ROOT/scripts/analyze_paired_closed_loop.py" \
        --summary "$summary" \
        --baseline historical_B10 \
        --candidate full306801_posttrain \
        --output "$OUT_ROOT/${benchmark}_paired_analysis.json" \
        --markdown "$OUT_ROOT/${benchmark}_paired_analysis.md" \
        >> "$LOG" 2>&1
    log "Summarized historical-B10 paired comparison: $benchmark"
}

export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$MAPS_ROOT"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

log "Auditing frozen identities before Test14 candidate-only evaluation"
mkdir -p "$MODEL_ALIAS_ROOT"
ln -sfn "$NEW_SOURCE_RUN/args.json" "$MODEL_ALIAS_ROOT/args.json"
ln -sfn "$NEW_SOURCE_RUN/model_epoch_2_trainloss_0.0005.pth" \
    "$MODEL_ALIAS_ROOT/model_epoch_2_trainloss_0.0005.pth"
check_sha256 a6b495893945d719055d9841a91e1e637063da7f3d6b439c5e82077d5964230d "$MANIFEST"
check_sha256 ba6848d19bf5ea486fcc9ff31299cadccd8778c927fdf7e961e65381b2caa39f "$B10_ARGS"
check_sha256 22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce "$B10_CKPT"
check_sha256 e070b3ba08a51195017baadd52ffc4343b37ac4c602084801db5e40c5e0c404c "$NEW_ARGS"
check_sha256 41cf43c5a7b7a70663409b9cb1260e6d549814b128b2b91cc0e94af93cf58763 "$NEW_CKPT"

cp -f "$MANIFEST" "$OUT_ROOT/frozen_eval_chunk_manifest.json"
sha256sum "$MANIFEST" "$B10_ARGS" "$B10_CKPT" "$NEW_ARGS" "$NEW_CKPT" \
    > "$OUT_ROOT/artifact_sha256.txt"

validate_historical_b10 test14-hard
validate_historical_b10 test14-random

log "Starting Test14 candidate-only evaluation: hard=272 random=261 seed=0 workers=1"
run_candidate test14-hard
summarize_pair test14-hard
run_candidate test14-random
summarize_pair test14-random
log "Completed Test14 candidate-only evaluation and paired analysis"
