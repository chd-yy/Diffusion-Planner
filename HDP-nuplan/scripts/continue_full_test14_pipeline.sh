#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/yanjun/NewDisk/Diffusion-Planner}"
HDP_ROOT="$PROJECT_ROOT/HDP-nuplan"
PYTHON_BIN="${DIFFUSION_PLANNER_PYTHON:-/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python}"
WORK_ROOT="${TEST14_WORK_ROOT:-$HDP_ROOT/tmp/test14_full_remote_subset}"
INDEX="$WORK_ROOT/test14_remote_scan.sqlite"
SCAN_PID_FILE="$WORK_ROOT/full_scan.pid"
LOG="$WORK_ROOT/full_pipeline.log"
EXPECTED_DB_COUNT=1349
RESERVE_BYTES=$((15 * 1024 * 1024 * 1024))

log() {
    echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"
}

scanned_count() {
    sqlite3 "$INDEX" "SELECT COUNT(*) FROM archive_entry WHERE status = 'scanned';"
}

if [[ ! -f "$INDEX" || ! -f "$SCAN_PID_FILE" ]]; then
    log "ERROR: scan index or PID file is missing"
    exit 1
fi

scan_pid=$(<"$SCAN_PID_FILE")
restart_count=0
while true; do
    log "Waiting for remote ZIP scan PID $scan_pid"
    while kill -0 "$scan_pid" 2>/dev/null; do
        log "Scan progress: $(scanned_count)/$EXPECTED_DB_COUNT"
        sleep 300
    done

    completed=$(scanned_count)
    if [[ "$completed" -eq "$EXPECTED_DB_COUNT" ]]; then
        break
    fi
    restart_count=$((restart_count + 1))
    if [[ "$restart_count" -gt 10 ]]; then
        log "ERROR: scan stopped at $completed/$EXPECTED_DB_COUNT after 10 automatic restarts"
        exit 1
    fi
    log "Scan stopped at $completed/$EXPECTED_DB_COUNT; automatic restart $restart_count/10 in 60s"
    sleep 60
    "$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_remote_subset.py" \
        --phase scan --workers 4 --work-root "$WORK_ROOT" >> "$WORK_ROOT/full_scan.log" 2>&1 &
    scan_pid=$!
    printf '%s\n' "$scan_pid" > "$SCAN_PID_FILE"
done

log "Remote ZIP scan complete: $completed/$EXPECTED_DB_COUNT"

"$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_remote_subset.py" \
    --phase select --work-root "$WORK_ROOT" >> "$LOG" 2>&1

read -r required_bytes available_bytes < <(
    "$PYTHON_BIN" - "$WORK_ROOT/test14_selection_manifest.json" "$WORK_ROOT" <<'PY'
import json
import shutil
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["required_uncompressed_bytes"], shutil.disk_usage(sys.argv[2]).free)
PY
)
log "Selected subset size: required=$required_bytes bytes, available=$available_bytes bytes"
if (( required_bytes + RESERVE_BYTES > available_bytes )); then
    log "ERROR: selected DB subset does not fit while preserving the 15 GiB safety reserve"
    exit 1
fi

"$PYTHON_BIN" "$HDP_ROOT/scripts/prepare_test14_remote_subset.py" \
    --phase extract --work-root "$WORK_ROOT" >> "$LOG" 2>&1
log "Required Test14 DB extraction complete"

"$PYTHON_BIN" "$HDP_ROOT/scripts/validate_test14_full_coverage.py" \
    --output "$WORK_ROOT/coverage_validation.json" >> "$LOG" 2>&1
log "Coverage gate passed: test14-hard=272, test14-random=261"

bash "$HDP_ROOT/scripts/evaluate_full_test14_three_models.sh" >> "$LOG" 2>&1
log "Three-model Test14 evaluation complete"
