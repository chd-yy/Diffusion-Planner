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
LOG="$OUT_ROOT/evaluate_full_test14_three_models.log"

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

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "$DP_ARGS"
require_file "$DP_CKPT"
require_file "$B10_ARGS"
require_file "$B10_CKPT"
require_file "$RL_ARGS"
require_file "$RL_CKPT"
require_file "$TEST_WORK_ROOT/coverage_validation.json"
require_file "$GENERATED_CONFIG_ROOT/scenario_filter/test14-random-full.yaml"

"$PYTHON_BIN" - "$TEST_WORK_ROOT/coverage_validation.json" <<'PY'
import json
import sys

coverage = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"test14_hard": 272, "test14_random": 261}
for name, count in expected.items():
    actual = coverage[name]["count"]
    unique = coverage[name]["unique_token_count"]
    if actual != count or unique != count:
        raise SystemExit(f"Coverage gate failed for {name}: count={actual}, unique={unique}")
print("Coverage gate passed: test14-hard=272, test14-random=261")
PY

check_sha256() {
    local expected=$1
    local path=$2
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
        echo "SHA256 mismatch: $path" >&2
        echo "expected=$expected actual=$actual" >&2
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

run_one() {
    local benchmark=$1
    local filter=$2
    local model_key=$3
    local planner_kind=$4
    local args_file=$5
    local checkpoint=$6
    local uid="${model_key}-${benchmark}"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"
    if [[ -f "$run_dir/runner_report.parquet" ]]; then
        echo "[$(date '+%F %T %Z')] Keeping completed run: $uid" | tee -a "$LOG"
        return
    fi

    local planner_config
    local planner_key
    if [[ "$planner_kind" == "diffusion" ]]; then
        planner_config=diffusion_planner
        planner_key=diffusion_planner
    else
        planner_config=hyper_diffusion_planner
        planner_key=hyper_diffusion_planner
    fi

    echo "[$(date '+%F %T %Z')] Starting $uid" | tee -a "$LOG"
    "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation=closed_loop_nonreactive_agents \
        planner="$planner_config" \
        "planner.$planner_key.config.args_file=$args_file" \
        "planner.$planner_key.ckpt_path=$checkpoint" \
        scenario_builder=nuplan \
        scenario_builder.db_files="$TEST_DB_ROOT" \
        scenario_filter="$filter" \
        experiment_uid="$uid" \
        worker=single_machine_thread_pool \
        worker.max_workers=2 \
        worker.use_process_pool=false \
        number_of_gpus_allocated_per_simulation=0.5 \
        number_of_cpus_allocated_per_simulation=1 \
        max_callback_workers=2 \
        disable_callback_parallelization=false \
        enable_simulation_progress_bar=true \
        verbose=true \
        '~callback.simulation_log_callback' \
        "hydra.searchpath=[file://$GENERATED_CONFIG_ROOT,pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
        >> "$LOG" 2>&1
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
        --summary "$output" \
        --baseline hdp_b_epoch10 \
        --candidate hdp_rl_epoch2 \
        --output "$OUT_ROOT/${benchmark}_b10_vs_rl_epoch2_analysis.json" \
        --markdown "$OUT_ROOT/${benchmark}_b10_vs_rl_epoch2_analysis.md" \
        >> "$LOG" 2>&1
}

evaluate_benchmark() {
    local benchmark=$1
    local filter=$2
    run_one "$benchmark" "$filter" aligned-original-dp diffusion "$DP_ARGS" "$DP_CKPT"
    run_one "$benchmark" "$filter" hdp-b-epoch10 hdp "$B10_ARGS" "$B10_CKPT"
    run_one "$benchmark" "$filter" hdp-rl-epoch2 hdp "$RL_ARGS" "$RL_CKPT"
    summarize_benchmark "$benchmark"
}

evaluate_benchmark test14-hard test14-hard
evaluate_benchmark test14-random test14-random-full
echo "[$(date '+%F %T %Z')] Full Test14 three-model evaluation completed" | tee -a "$LOG"
