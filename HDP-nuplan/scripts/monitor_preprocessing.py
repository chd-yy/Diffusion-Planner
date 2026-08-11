#!/usr/bin/env python3
"""只读汇总 NuPlan 多 worker 预处理进度、进程、错误重试和磁盘余量。"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional


PROCESS_PATTERNS = {
    "scheduler": "[r]un_preprocessing_range.py",
    "downloader": "[d]ownload_nuplan_log_subset.py",
    "preprocessor": "[r]un_preprocessing_shard.py|/data_process.py",
    "supervised_trainer": "[t]rain_predictor.py",
    "rl_trainer": "[t]rain_predictor_rl.py",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _process_counts() -> Dict[str, int]:
    """用不匹配 pgrep 自身的模式统计活动进程；pgrep 不存在时返回 -1。"""
    counts = {}
    for name, pattern in PROCESS_PATTERNS.items():
        try:
            result = subprocess.run(
                ["pgrep", "-af", pattern],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            counts[name] = -1
            continue
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        counts[name] = len(lines)
    return counts


def _inspect_shard(output_root: Path, shard: Mapping) -> dict:
    shard_id = str(shard["shard_id"])
    expected = int(shard["total_scenarios"])
    shard_dir = output_root / shard_id
    manifest_path = shard_dir / "manifest.json"
    processing_path = shard_dir / "processing_report.json"
    cache_dir = shard_dir / "cache"
    npz_count = sum(1 for _ in cache_dir.glob("*.npz")) if cache_dir.is_dir() else 0
    result = {
        "shard_id": shard_id,
        "expected_npz": expected,
        "npz_count": npz_count,
        "complete": False,
        "reason": "missing_processing_report",
    }
    if not processing_path.is_file():
        return result
    try:
        processing = _read_json(processing_path)
    except (OSError, ValueError) as error:
        result["reason"] = f"invalid_processing_report:{type(error).__name__}"
        return result
    if processing.get("status") != "complete":
        result["reason"] = f"processing_status:{processing.get('status')}"
        return result
    if int(processing.get("failed", 0)) != 0:
        result["reason"] = f"processing_failed:{processing.get('failed')}"
        return result
    if not manifest_path.is_file():
        result["reason"] = "missing_manifest"
        return result
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError) as error:
        result["reason"] = f"invalid_manifest:{type(error).__name__}"
        return result
    if not isinstance(manifest, list):
        result["reason"] = "manifest_not_list"
        return result
    result["manifest_count"] = len(manifest)
    if len(manifest) != expected:
        result["reason"] = f"manifest_count:{len(manifest)}"
        return result
    if int(processing.get("manifest_count", -1)) != expected:
        result["reason"] = f"processing_manifest_count:{processing.get('manifest_count')}"
        return result
    if npz_count != expected:
        result["reason"] = f"npz_count:{npz_count}"
        return result
    result["complete"] = True
    result["reason"] = "complete"
    return result


def collect_snapshot(
    plan_path: Path,
    output_root: Path,
    raw_root: Path,
    state_dir: Path,
    worker_count: int,
    process_counts: Optional[Mapping[str, int]] = None,
    min_free_gib: float = 20.0,
) -> dict:
    if min_free_gib < 0:
        raise ValueError("min_free_gib must be non-negative")
    plan = _read_json(plan_path)
    shards = plan.get("shards", [])
    shard_results = [_inspect_shard(output_root, shard) for shard in shards]
    complete_results = [item for item in shard_results if item["complete"]]
    partial_results = [
        item
        for item in shard_results
        if not item["complete"] and item["npz_count"] > 0
    ]

    retry_count = 0
    invalid_download_reports = []
    for result in complete_results:
        report_path = output_root / result["shard_id"] / "download_report.json"
        if not report_path.is_file():
            invalid_download_reports.append(
                {"shard_id": result["shard_id"], "reason": "missing"}
            )
            continue
        try:
            report = _read_json(report_path)
            retry_count += int(report.get("retry_count", 0))
        except (OSError, ValueError, TypeError) as error:
            invalid_download_reports.append(
                {
                    "shard_id": result["shard_id"],
                    "reason": type(error).__name__,
                }
            )

    workers = []
    for worker_index in range(worker_count):
        state_path = state_dir / f"rolling_worker_{worker_index}_of_{worker_count}_state.json"
        worker = {"worker_index": worker_index, "state_path": str(state_path)}
        if not state_path.is_file():
            worker.update({"status": "missing_state", "error": None})
            workers.append(worker)
            continue
        try:
            state = _read_json(state_path)
        except (OSError, ValueError) as error:
            worker.update(
                {"status": "invalid_state", "error": f"{type(error).__name__}: {error}"}
            )
            workers.append(worker)
            continue
        running_records = [
            record
            for record in state.get("records", {}).values()
            if record.get("status") == "running"
        ]
        current = running_records[-1].get("shard_id") if running_records else None
        worker.update(
            {
                "status": state.get("status", "unknown"),
                "next_position": state.get("next_position"),
                "selected_shard_count": state.get("selected_shard_count"),
                "current_shard": current,
                "updated_at_utc": state.get("updated_at_utc"),
                "error": state.get("error"),
            }
        )
        workers.append(worker)

    process_counts = dict(process_counts or _process_counts())
    disk = shutil.disk_usage(raw_root)
    actual_npz = sum(1 for _ in output_root.glob("shard_*/cache/*.npz"))
    raw_db_count = sum(1 for _ in raw_root.glob("shard_*/trainval/*.db"))
    temporary_paths = list(raw_root.glob("shard_*/trainval/.*.tmp"))
    temporary_bytes = sum(path.stat().st_size for path in temporary_paths if path.is_file())

    issues = []
    worker_statuses = [worker["status"] for worker in workers]
    if any(worker["error"] for worker in workers):
        issues.append("worker_error")
    if any(status in {"failed", "missing_state", "invalid_state"} for status in worker_statuses):
        issues.append("worker_state_unhealthy")
    if process_counts.get("supervised_trainer", 0) > 0 or process_counts.get("rl_trainer", 0) > 0:
        issues.append("unexpected_training_process")
    if disk.free < min_free_gib * 1024 ** 3:
        issues.append("disk_below_minimum")
    if process_counts.get("scheduler", 0) > worker_count:
        issues.append("too_many_schedulers")
    if any(status == "running" for status in worker_statuses) and process_counts.get("scheduler") == 0:
        issues.append("state_running_without_scheduler")
    if invalid_download_reports:
        issues.append("complete_shard_download_report_missing_or_invalid")

    if len(complete_results) == len(shards) and actual_npz == int(plan["total_scenarios"]):
        phase = "ready_for_merge_validation"
    elif all(status == "stopped_by_user" for status in worker_statuses):
        phase = "stopped_by_user"
    elif process_counts.get("scheduler", 0) > 0:
        phase = "running"
    else:
        phase = "not_running"

    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "issues": sorted(set(issues)),
        "plan": {
            "shard_count": len(shards),
            "target_npz": int(plan["total_scenarios"]),
        },
        "progress": {
            "complete_shards": len(complete_results),
            "complete_shard_npz": sum(item["npz_count"] for item in complete_results),
            "actual_npz": actual_npz,
            "partial_shards": len(partial_results),
            "partial_shard_npz": sum(item["npz_count"] for item in partial_results),
            "download_retry_count_archived": retry_count,
        },
        "raw": {
            "db_count": raw_db_count,
            "temporary_count": len(temporary_paths),
            "temporary_bytes": temporary_bytes,
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / 1024 ** 3, 2),
            "minimum_free_gib": min_free_gib,
        },
        "processes": process_counts,
        "workers": workers,
        "incomplete_shards_with_npz": partial_results[:20],
        "invalid_download_reports": invalid_download_reports[:20],
    }


def _print_human(snapshot: Mapping) -> None:
    progress = snapshot["progress"]
    plan = snapshot["plan"]
    print(
        f"phase={snapshot['phase']} issues={snapshot['issues']} "
        f"shards={progress['complete_shards']}/{plan['shard_count']} "
        f"npz={progress['actual_npz']}/{plan['target_npz']} "
        f"complete_npz={progress['complete_shard_npz']} "
        f"retries={progress['download_retry_count_archived']} "
        f"free_gib={snapshot['disk']['free_gib']}"
    )
    processes = snapshot["processes"]
    print(
        "processes "
        + " ".join(f"{name}={count}" for name, count in sorted(processes.items()))
    )
    for worker in snapshot["workers"]:
        print(
            f"worker={worker['worker_index']} status={worker['status']} "
            f"next={worker.get('next_position')} current={worker.get('current_shard')} "
            f"error={worker.get('error')}"
        )
    print(
        f"raw_db={snapshot['raw']['db_count']} "
        f"partial_tmp={snapshot['raw']['temporary_count']} "
        f"partial_tmp_bytes={snapshot['raw']['temporary_bytes']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--raw_root", required=True, type=Path)
    parser.add_argument("--state_dir", required=True, type=Path)
    parser.add_argument("--worker_count", type=int, default=6)
    parser.add_argument("--min_free_gib", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="print full JSON snapshot")
    args = parser.parse_args()
    if args.worker_count <= 0:
        raise ValueError("worker_count must be positive")
    snapshot = collect_snapshot(
        args.plan,
        args.output_root,
        args.raw_root,
        args.state_dir,
        args.worker_count,
        min_free_gib=args.min_free_gib,
    )
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        _print_human(snapshot)


if __name__ == "__main__":
    main()
