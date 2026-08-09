#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 <experiment_uid> <args.json> <checkpoint.pth> [scenario_filter] [hdp|diffusion]" >&2
  exit 2
fi

EXPERIMENT_UID=$1
ARGS_FILE=$2
CHECKPOINT_FILE=$3
SCENARIO_FILTER=${4:-mini-val-closed-loop-3}
PLANNER_KIND=${5:-hdp}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/home/yanjun/NewDisk/nuplan/dataset}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/home/yanjun/NewDisk/nuplan/dataset/maps}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$PROJECT_ROOT/tmp/closed_loop_eval}
export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/home/yanjun/NewDisk/nuplan-devkit}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

PYTHON_BIN=${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}
MINI_DB_ROOT=${NUPLAN_MINI_DB_ROOT:-/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable does not exist: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$ARGS_FILE" ]]; then
  echo "Training args file does not exist: $ARGS_FILE" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT_FILE" >&2
  exit 1
fi
if [[ ! -d "$MINI_DB_ROOT" ]]; then
  echo "NuPlan mini DB directory does not exist: $MINI_DB_ROOT" >&2
  exit 1
fi

case "$PLANNER_KIND" in
  hdp)
    PLANNER_CONFIG=hyper_diffusion_planner
    PLANNER_KEY=hyper_diffusion_planner
    ;;
  diffusion)
    PLANNER_CONFIG=diffusion_planner
    PLANNER_KEY=diffusion_planner
    ;;
  *)
    echo "Planner kind must be 'hdp' or 'diffusion': $PLANNER_KIND" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation=closed_loop_nonreactive_agents \
  planner="$PLANNER_CONFIG" \
  "planner.$PLANNER_KEY.config.args_file=$ARGS_FILE" \
  "planner.$PLANNER_KEY.ckpt_path=$CHECKPOINT_FILE" \
  scenario_builder=nuplan \
  scenario_builder.db_files="$MINI_DB_ROOT" \
  scenario_filter="$SCENARIO_FILTER" \
  experiment_uid="$EXPERIMENT_UID" \
  worker=sequential \
  number_of_gpus_allocated_per_simulation=1 \
  number_of_cpus_allocated_per_simulation=1 \
  max_callback_workers=1 \
  disable_callback_parallelization=true \
  enable_simulation_progress_bar=true \
  verbose=true \
  'hydra.searchpath=[pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'
