#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanjun/NewDisk/Diffusion-Planner"
PYTHON_BIN="/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python"
PILOT_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1"
BASELINE_ROOT="$PROJECT_ROOT/HDP-nuplan/tmp/mini_train_pilot_1000_seed3407_v1"
RUNNER="$PROJECT_ROOT/HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh"
MASTER_LOG="$PILOT_ROOT/${RL_MULTI_LOG_TAG:-rl_v3_10k_multiseed}.log"
VARIANT_PREFIX="${RL_MULTI_VARIANT_PREFIX:-safetygate03_anchor005_10k}"
ANCHOR_WEIGHT="${RL_MULTI_EXPERT_ANCHOR_WEIGHT:-0.05}"
SEEDS=(42 2026)

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$MASTER_LOG"
}

check_gate20_safety() {
    local summary_path="$1"
    "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
baseline = data["hdp_b_epoch10"]
candidate = data["hdp_rl_variant"]
if baseline["scenario_count"] != 20 or candidate["scenario_count"] != 20:
    raise SystemExit("gate20 must contain exactly 20 paired scenarios")
if baseline["failed_simulations"] or candidate["failed_simulations"]:
    raise SystemExit(
        f"gate20 simulation failure: baseline={baseline['failed_simulations']}, "
        f"candidate={candidate['failed_simulations']}"
    )

baseline_rows = {row["scenario"]: row for row in baseline["scenarios"]}
candidate_rows = {row["scenario"]: row for row in candidate["scenarios"]}
if baseline_rows.keys() != candidate_rows.keys():
    raise SystemExit("gate20 scenario IDs do not match")

focus_scenario = "3713735b94cf5a7b"
if focus_scenario not in candidate_rows:
    raise SystemExit(f"required TTC regression scenario missing: {focus_scenario}")

hard_metrics = (
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "drivable_area_compliance",
)
regressions = []
for scenario, baseline_row in baseline_rows.items():
    candidate_row = candidate_rows[scenario]
    for metric in hard_metrics:
        if candidate_row[metric] < baseline_row[metric] - 1e-12:
            regressions.append(
                (scenario, metric, baseline_row[metric], candidate_row[metric])
            )
if regressions:
    for item in regressions:
        print("hard safety regression:", item, file=sys.stderr)
    raise SystemExit(f"gate20 failed with {len(regressions)} hard safety regressions")

score_delta = candidate["means"]["score"] - baseline["means"]["score"]
print(
    "gate20 passed: no hard safety regressions; "
    f"focus_ttc={candidate_rows[focus_scenario]['time_to_collision_within_bound']}; "
    f"score_delta={score_delta:+.9f}"
)
PY
}

mkdir -p "$PILOT_ROOT"
log "Starting 10k RL v3 multi-seed experiment: seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
    variant="${VARIANT_PREFIX}_seed${seed}"
    run_root="$PILOT_ROOT/rl_${variant}_from_b10"
    gate20_summary="$run_root/closed_loop20/b10_vs_rl_${variant}_gate20.json"

    log "Seed $seed: training and gate20 started"
    RL_PILOT_ROOT="$PILOT_ROOT" \
    RL_BASELINE_ROOT="$BASELINE_ROOT" \
    RL_EXPECTED_NPZ=10000 \
    RL_VARIANT_TAG="$variant" \
    RL_SAFETY_GATE_THRESHOLD=0.3 \
    RL_EXPERT_ANCHOR_WEIGHT="$ANCHOR_WEIGHT" \
    RL_SEED="$seed" \
    RL_GATE20_ONLY=1 \
    bash "$RUNNER"

    if ! check_gate20_safety "$gate20_summary" | tee -a "$MASTER_LOG"; then
        log "Seed $seed: gate20 safety check failed; gate39 skipped"
        continue
    fi

    log "Seed $seed: gate20 passed; continuing gate39"
    RL_PILOT_ROOT="$PILOT_ROOT" \
    RL_BASELINE_ROOT="$BASELINE_ROOT" \
    RL_EXPECTED_NPZ=10000 \
    RL_VARIANT_TAG="$variant" \
    RL_SAFETY_GATE_THRESHOLD=0.3 \
    RL_EXPERT_ANCHOR_WEIGHT="$ANCHOR_WEIGHT" \
    RL_SEED="$seed" \
    RL_SKIP_TRAINING=1 \
    RL_SKIP_GATE20=1 \
    bash "$RUNNER"
    log "Seed $seed: fixed59 evaluation completed"
done

log "10k RL v3 multi-seed experiment finished"
