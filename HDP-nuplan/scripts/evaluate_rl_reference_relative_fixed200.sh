#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1"
RUN_ROOT="$ROOT/rl_reference_relative_v1_10k_6ep_seed2026_from_b10"
OUT_ROOT="$RUN_ROOT/fixed200_eval"
BASELINE_SUMMARY="$ROOT/rl_v4_fixed200_eval_retry/b10_vs_rl_fixed200.json"
EXP_UID="rl-reference-relative-v1-10k-6ep-seed2026-fixed200"
RUN_DIR="$OUT_ROOT/exp/simulation/closed_loop_nonreactive_agents/$EXP_UID"
LOG="$OUT_ROOT/fixed200_eval.log"

find_checkpoint() {
    find "$RUN_ROOT/training_log" -type f -name 'model_epoch_6_trainloss_*.pth' \
        -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

mkdir -p "$OUT_ROOT"
CHECKPOINT="$(find_checkpoint)"
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || {
    echo "No Epoch6 checkpoint found under $RUN_ROOT/training_log" >&2
    exit 1
}
ARGS_FILE="$(dirname "$CHECKPOINT")/args.json"

export NUPLAN_DATA_ROOT="/home/yanjun/NewDisk/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/yanjun/NewDisk/nuplan/dataset/maps"
export NUPLAN_DEVKIT_ROOT="/home/yanjun/NewDisk/nuplan-devkit"
export NUPLAN_MINI_DB_ROOT="/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
export NUPLAN_EXP_ROOT="$OUT_ROOT"
export DIFFUSION_PLANNER_PYTHON="$PYTHON_BIN"
export CUDA_VISIBLE_DEVICES=0

for path in "$ARGS_FILE" "$CHECKPOINT" "$BASELINE_SUMMARY"; do
    [[ -f "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 1; }
done

if [[ ! -f "$RUN_DIR/runner_report.parquet" ]]; then
    echo "[$(date '+%F %T %Z')] Starting reference-relative fixed200 evaluation" | tee -a "$LOG"
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation=closed_loop_nonreactive_agents \
        planner=hyper_diffusion_planner \
        "planner.hyper_diffusion_planner.config.args_file=$ARGS_FILE" \
        "planner.hyper_diffusion_planner.ckpt_path=$CHECKPOINT" \
        scenario_builder=nuplan \
        scenario_builder.db_files="$NUPLAN_MINI_DB_ROOT" \
        scenario_filter=mini-val-fixed-200-rl-v4 \
        experiment_uid="$EXP_UID" \
        worker=single_machine_thread_pool \
        worker.max_workers=2 \
        worker.use_process_pool=false \
        number_of_gpus_allocated_per_simulation=0.5 \
        number_of_cpus_allocated_per_simulation=1 \
        max_callback_workers=2 \
        disable_callback_parallelization=false \
        enable_simulation_progress_bar=true \
        verbose=true \
        'hydra.searchpath=[pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]' \
        >> "$LOG" 2>&1
else
    echo "[$(date '+%F %T %Z')] Existing report found; skipping simulation" | tee -a "$LOG"
fi

[[ -f "$RUN_DIR/runner_report.parquet" ]] || {
    echo "Missing report: $RUN_DIR/runner_report.parquet" >&2
    exit 1
}

"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/summarize_closed_loop_metrics.py" \
    --run hdp_rl_reference_relative="$RUN_DIR" \
    --output "$OUT_ROOT/rl_reference_relative_fixed200.json" \
    >> "$LOG" 2>&1

"$PYTHON_BIN" - "$BASELINE_SUMMARY" "$OUT_ROOT/rl_reference_relative_fixed200.json" \
    "$OUT_ROOT/b10_vs_reference_relative_fixed200.json" <<'PY' >> "$LOG" 2>&1
import json
import sys

baseline_path, candidate_path, output_path = sys.argv[1:]
baseline = json.load(open(baseline_path, encoding="utf-8"))
candidate = json.load(open(candidate_path, encoding="utf-8"))
merged = {"hdp_b_epoch10": baseline["hdp_b_epoch10"]}
merged.update(candidate)
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(merged, file, ensure_ascii=False, indent=2)
    file.write("\n")
PY

"$PYTHON_BIN" "$PROJECT_ROOT/HDP-nuplan/scripts/analyze_paired_closed_loop.py" \
    --summary "$OUT_ROOT/b10_vs_reference_relative_fixed200.json" \
    --baseline hdp_b_epoch10 \
    --candidate hdp_rl_reference_relative \
    --output "$OUT_ROOT/b10_vs_reference_relative_fixed200_analysis.json" \
    --markdown "$OUT_ROOT/b10_vs_reference_relative_fixed200_analysis.md" \
    >> "$LOG" 2>&1

echo "[$(date '+%F %T %Z')] reference-relative fixed200 evaluation completed" | tee -a "$LOG"
