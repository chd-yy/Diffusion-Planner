#!/usr/bin/env bash
set -euo pipefail

# Re-run selected paired Test14-hard scenarios with NuPlan's simulation-log
# callback enabled. The original complete evaluations remain untouched.

HDP_ROOT="${HDP_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan}"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}"
TEST_WORK_ROOT="${TEST14_WORK_ROOT:-$HDP_ROOT/tmp/test14_full_remote_subset}"
FILTER_CONFIG_ROOT="$HDP_ROOT/doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_qualitative_replay/config"
OUT_ROOT="${CH4_REPLAY_OUT_ROOT:-/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/test14_outputs/EXP_CH4_qualitative_replay_1008}"
B10_ROOT="$HDP_ROOT/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
NEW_ALIAS_ROOT="$TEST_WORK_ROOT/model_aliases/full306801_posttrain_1006"
LOG="$OUT_ROOT/paired_replay.log"

check_sha256() {
    local expected="$1"
    local path="$2"
    local actual
    [[ -f "$path" ]] || { echo "Missing model artifact: $path" >&2; exit 1; }
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
        echo "Model artifact SHA256 mismatch: $path" >&2
        exit 1
    }
}

run_arm() {
    local uid="$1"
    local args="$2"
    local checkpoint="$3"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$uid"

    if [[ -d "$run_dir/simulation_log" ]] &&
       [[ $(find "$run_dir/simulation_log" -type f -name '*.msgpack.xz' | wc -l) -eq 7 ]]; then
        echo "Keeping complete replay arm: $uid" | tee -a "$LOG"
        return
    fi
    if [[ -e "$run_dir" ]]; then
        echo "Incomplete output exists; inspect it before re-running: $run_dir" >&2
        exit 1
    fi

    echo "Starting replay arm: $uid" | tee -a "$LOG"
    "$PYTHON_BIN" "$DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation=closed_loop_nonreactive_agents \
        planner=hyper_diffusion_planner \
        "planner.hyper_diffusion_planner.config.args_file=$args" \
        "planner.hyper_diffusion_planner.ckpt_path=$checkpoint" \
        scenario_builder=nuplan \
        scenario_builder.db_files="$TEST_WORK_ROOT/data/cache/test14" \
        scenario_filter=ch4-qualitative-replay-hard \
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
        "hydra.searchpath=[file://$FILTER_CONFIG_ROOT,file://$TEST_WORK_ROOT/config,pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
        >> "$LOG" 2>&1
    echo "Finished replay arm: $uid" | tee -a "$LOG"
}

mkdir -p "$OUT_ROOT"
export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NUPLAN_DEVKIT_ROOT="$DEVKIT_ROOT"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

check_sha256 ba6848d19bf5ea486fcc9ff31299cadccd8778c927fdf7e961e65381b2caa39f "$B10_ROOT/args.json"
check_sha256 22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce "$B10_ROOT/model_epoch_10_trainloss_0.0091.pth"
check_sha256 e070b3ba08a51195017baadd52ffc4343b37ac4c602084801db5e40c5e0c404c "$NEW_ALIAS_ROOT/args.json"
check_sha256 41cf43c5a7b7a70663409b9cb1260e6d549814b128b2b91cc0e94af93cf58763 "$NEW_ALIAS_ROOT/model_epoch_2_trainloss_0.0005.pth"

run_arm "B10-qualitative-hard" "$B10_ROOT/args.json" "$B10_ROOT/model_epoch_10_trainloss_0.0091.pth"
run_arm "full306801-qualitative-hard" "$NEW_ALIAS_ROOT/args.json" "$NEW_ALIAS_ROOT/model_epoch_2_trainloss_0.0005.pth"
