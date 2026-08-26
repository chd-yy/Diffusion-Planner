#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1"
TRAIN_ROOT="$ROOT/rl_reference_relative_v1_10k_6ep_seed2026_from_b10"
OUT_ROOT="$TRAIN_ROOT/fixed200_epoch2_epoch4_eval"
BASELINE_SUMMARY="$ROOT/rl_v4_fixed200_eval_retry/b10_vs_rl_fixed200.json"

export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_ROOT"

find_checkpoint() {
    local epoch="$1"
    find "$TRAIN_ROOT/training_log" -type f -name "model_epoch_${epoch}_trainloss_*.pth" \
        -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_one() {
    local epoch="$1"
    local label="hdp_rl_reference_relative_epoch${epoch}"
    local exp_uid="rl-reference-relative-v1-10k-epoch${epoch}-seed2026-fixed200"
    local run_dir="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$exp_uid"
    local log="$OUT_ROOT/epoch${epoch}_fixed200.log"
    local checkpoint
    checkpoint="$(find_checkpoint "$epoch")"
    [[ -n "$checkpoint" && -f "$checkpoint" ]] || {
        echo "Missing Epoch${epoch} checkpoint" >&2
        return 1
    }
    local args_file="$(dirname "$checkpoint")/args.json"
    [[ -f "$args_file" ]] || { echo "Missing args: $args_file" >&2; return 1; }

    if [[ ! -f "$run_dir/runner_report.parquet" ]]; then
        echo "[$(date '+%F %T %Z')] Starting Epoch${epoch} fixed200" | tee -a "$log"
        cd "$PROJECT_ROOT"
        "$PYTHON_BIN" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=closed_loop_nonreactive_agents \
            planner=hyper_diffusion_planner \
            "planner.hyper_diffusion_planner.config.args_file=$args_file" \
            "planner.hyper_diffusion_planner.ckpt_path=$checkpoint" \
            scenario_builder=nuplan \
            scenario_builder.db_files="$NUPLAN_MINI_DB_ROOT" \
            scenario_filter=mini-val-fixed-200-rl-v4 \
            experiment_uid="$exp_uid" \
            worker=single_machine_thread_pool \
            worker.max_workers=1 \
            worker.use_process_pool=false \
            number_of_gpus_allocated_per_simulation=1.0 \
            number_of_cpus_allocated_per_simulation=1 \
            max_callback_workers=1 \
            disable_callback_parallelization=false \
            enable_simulation_progress_bar=true \
            verbose=true \
            'hydra.searchpath=[pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]' \
            >> "$log" 2>&1
    fi

    [[ -f "$run_dir/runner_report.parquet" ]] || {
        echo "Missing Epoch${epoch} report: $run_dir/runner_report.parquet" >&2
        return 1
    }

    "$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
        --run "$label=$run_dir" \
        --output "$OUT_ROOT/epoch${epoch}_fixed200.json" \
        >> "$log" 2>&1
    echo "[$(date '+%F %T %Z')] Epoch${epoch} fixed200 completed" | tee -a "$log"
}

run_one 2 &
pid_epoch2=$!
run_one 4 &
pid_epoch4=$!

status=0
wait "$pid_epoch2" || status=1
wait "$pid_epoch4" || status=1
[[ "$status" == 0 ]] || exit "$status"

"$PYTHON_BIN" - "$BASELINE_SUMMARY" "$OUT_ROOT/epoch2_fixed200.json" \
    "$OUT_ROOT/epoch4_fixed200.json" "$OUT_ROOT/b10_vs_epoch2_epoch4_fixed200.json" <<'PY'
import json
import sys

baseline_path, epoch2_path, epoch4_path, output_path = sys.argv[1:]
baseline = json.load(open(baseline_path, encoding="utf-8"))
epoch2 = json.load(open(epoch2_path, encoding="utf-8"))
epoch4 = json.load(open(epoch4_path, encoding="utf-8"))
merged = {"hdp_b_epoch10": baseline["hdp_b_epoch10"]}
merged.update(epoch2)
merged.update(epoch4)
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(merged, file, ensure_ascii=False, indent=2)
    file.write("\n")
PY

for epoch in 2 4; do
    "$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/analyze_paired_closed_loop.py" \
        --summary "$OUT_ROOT/b10_vs_epoch2_epoch4_fixed200.json" \
        --baseline hdp_b_epoch10 \
        --candidate "hdp_rl_reference_relative_epoch${epoch}" \
        --output "$OUT_ROOT/b10_vs_epoch${epoch}_analysis.json" \
        --markdown "$OUT_ROOT/b10_vs_epoch${epoch}_analysis.md"
done

echo "[$(date '+%F %T %Z')] Epoch2/Epoch4 fixed200 comparison completed"
