#!/usr/bin/env python3
"""无人值守滚动执行 NuPlan shard：下载、预处理、校验、归档并回收 raw DB。"""

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from validate_processed_cache import validate_cache  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_worker_shards(
    shards: Sequence[dict],
    start_index: int,
    end_index: Optional[int],
    worker_index: int,
    worker_count: int,
    include_indices: Optional[Sequence[int]] = None,
) -> List[dict]:
    """按 shard_index 取模分配互斥任务，并保留原计划顺序。"""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must satisfy 0 <= worker_index < worker_count")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")

    include_set = None if include_indices is None else {int(item) for item in include_indices}
    if include_set is not None:
        known_indices = {int(shard["shard_index"]) for shard in shards}
        unknown = sorted(include_set - known_indices)
        if unknown:
            raise ValueError(f"unknown explicit shard indices: {unknown}")

    selected = []
    for shard in shards:
        index = int(shard["shard_index"])
        # 中文注释：显式白名单用于只运行本地归档能够完整覆盖的 shard；
        # 它在 worker 取模之前过滤，因此多 worker 分配仍保持互斥。
        if include_set is not None and index not in include_set:
            continue
        if index < start_index:
            continue
        if end_index is not None and index >= end_index:
            continue
        if index % worker_count == worker_index:
            selected.append(shard)
    return selected


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_completed_shard(output_root: Path, shard: dict) -> Optional[dict]:
    """实时强校验已有 shard；无完整报告时返回 None，交给正常流程续跑。"""
    shard_dir = output_root / shard["shard_id"]
    manifest = shard_dir / "manifest.json"
    sampling_report = shard_dir / "sampling_report.json"
    processing_report = shard_dir / "processing_report.json"
    if not all(path.is_file() for path in (manifest, sampling_report, processing_report)):
        return None
    processing = _read_json(processing_report)
    if processing.get("status") != "complete" or processing.get("failed", 0) != 0:
        return None
    sampling = _read_json(sampling_report)
    expected_log_count = len(sampling.get("selected_per_log", {}))
    try:
        result = validate_cache(
            shard_dir / "cache",
            manifest,
            sampling_report,
            expected_count=int(shard["total_scenarios"]),
            expected_log_count=expected_log_count,
        )
    except (OSError, ValueError, KeyError):
        # 已有产物不完整或损坏时走正常下载/预处理续跑；第二次强校验仍失败则终止 worker，
        # 不会归档报告或删除 raw DB。
        return None
    atomic_write_json(shard_dir / "cache_validation_report.json", result)
    return result


def archive_download_report(raw_shard_dir: Path, processed_shard_dir: Path) -> Path:
    """复制下载审计并核对哈希；没有报告时禁止后续 raw 清理。"""
    source = raw_shard_dir / "download_report.json"
    destination = processed_shard_dir / "download_report.json"
    if source.is_file():
        processed_shard_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"download report hash mismatch for {raw_shard_dir.name}")
    if not destination.is_file():
        raise FileNotFoundError(
            f"download report is not archived for {raw_shard_dir.name}: {destination}"
        )
    return destination


def cleanup_raw_trainval(raw_root: Path, shard_id: str) -> bool:
    """只删除 raw_root/<精确 shard_id>/trainval，拒绝宽泛或逃逸路径。"""
    raw_root = raw_root.resolve()
    raw_shard_dir = (raw_root / shard_id).resolve()
    if raw_shard_dir.parent != raw_root or raw_shard_dir.name != shard_id:
        raise ValueError(f"unsafe raw shard path: {raw_shard_dir}")
    trainval = (raw_shard_dir / "trainval").resolve()
    if trainval.parent != raw_shard_dir or trainval.name != "trainval":
        raise ValueError(f"unsafe raw cleanup path: {trainval}")
    if not trainval.exists():
        return False
    if not trainval.is_dir():
        raise ValueError(f"raw trainval target is not a directory: {trainval}")
    shutil.rmtree(trainval)
    return True


def _download_command(args, shard: dict, raw_shard_dir: Path) -> List[str]:
    plan_dir = args.plan.resolve().parent
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "download_nuplan_log_subset.py"),
        "--log_names_json", str(plan_dir / shard["log_names_json"]),
        "--output_dir", str(raw_shard_dir / "trainval"),
        "--archive_index", str(args.archive_index.resolve()),
        "--report", str(raw_shard_dir / "download_report.json"),
        "--sqlite_quick_check",
        "--connect_timeout_seconds", str(args.connect_timeout_seconds),
        "--read_timeout_seconds", str(args.read_timeout_seconds),
        "--max_member_retries", str(args.max_member_retries),
        "--retry_delay_seconds", str(args.retry_delay_seconds),
        "--download_backend", args.download_backend,
        "--curl_low_speed_limit_bps", str(args.curl_low_speed_limit_bps),
        "--curl_low_speed_time_seconds", str(args.curl_low_speed_time_seconds),
    ]
    for archive in args.archive:
        command.extend(["--archive", archive])
    for archive_name in args.only_archive:
        command.extend(["--only_archive", archive_name])
    return command


def _preprocess_command(args, shard: dict, raw_shard_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_preprocessing_shard.py"),
        "--plan", str(args.plan.resolve()),
        "--shard_index", str(shard["shard_index"]),
        "--data_path", str(raw_shard_dir / "trainval"),
        "--map_path", str(args.map_path.resolve()),
        "--output_root", str(args.output_root.resolve()),
        "--checksum_mode", args.checksum_mode,
        "--allow_empty_logs",
    ]
    return command


def _new_state(args, selected: Sequence[dict], plan_sha256: str) -> Dict:
    return {
        "format_version": 1,
        "status": "running",
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "shard_indices": args.shard_indices,
        "archive": args.archive,
        "only_archive": args.only_archive,
        "selected_shard_count": len(selected),
        "next_position": 0,
        "records": {},
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }


def _load_state(args, selected: Sequence[dict], plan_sha256: str) -> Dict:
    if args.state_path.is_file() and args.resume:
        state = _read_json(args.state_path)
        expected = {
            "plan_sha256": plan_sha256,
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "shard_indices": args.shard_indices,
            "archive": args.archive,
            "only_archive": args.only_archive,
            "selected_shard_count": len(selected),
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"rolling state configuration mismatch: {mismatches}")
        state["status"] = "running"
        state["resumed_at_utc"] = _utc_now()
        state.pop("error", None)
        state.pop("traceback", None)
        return state
    return _new_state(args, selected, plan_sha256)


def _save_state(path: Path, state: Dict) -> None:
    state["updated_at_utc"] = _utc_now()
    atomic_write_json(path, state)


def run(args) -> Dict:
    plan = _read_json(args.plan)
    shards = plan.get("shards", [])
    selected = select_worker_shards(
        shards,
        args.start_index,
        args.end_index,
        args.worker_index,
        args.worker_count,
        args.shard_indices,
    )
    plan_sha256 = sha256_file(args.plan)
    state = _load_state(args, selected, plan_sha256)
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    _save_state(args.state_path, state)

    start_position = int(state.get("next_position", 0))
    if start_position < 0 or start_position > len(selected):
        raise ValueError(f"invalid next_position in state: {start_position}")

    processed_this_run = 0
    try:
        for position in range(start_position, len(selected)):
            if args.max_shards is not None and processed_this_run >= args.max_shards:
                state["status"] = "paused_at_max_shards"
                _save_state(args.state_path, state)
                return state

            shard = selected[position]
            shard_id = shard["shard_id"]
            raw_shard_dir = args.raw_root.resolve() / shard_id
            processed_shard_dir = args.output_root.resolve() / shard_id
            record = {
                "shard_index": shard["shard_index"],
                "shard_id": shard_id,
                "started_at_utc": _utc_now(),
                "status": "running",
            }
            state["records"][shard_id] = record
            _save_state(args.state_path, state)

            print(
                f"[worker {args.worker_index}/{args.worker_count}] "
                f"position {position + 1}/{len(selected)}: {shard_id}",
                flush=True,
            )
            validation = _validate_completed_shard(args.output_root.resolve(), shard)
            if validation is None:
                args.raw_root.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(args.raw_root).free
                record["free_bytes_before_download"] = free_bytes
                required_free_bytes = int(args.min_free_gib * 1024 ** 3)
                if free_bytes < required_free_bytes:
                    raise RuntimeError(
                        f"free disk space below safety threshold before {shard_id}: "
                        f"{free_bytes / 1024 ** 3:.2f} GiB < {args.min_free_gib:.2f} GiB"
                    )
                raw_shard_dir.mkdir(parents=True, exist_ok=True)
                download_command = _download_command(args, shard, raw_shard_dir)
                record["download_command"] = download_command
                _save_state(args.state_path, state)
                subprocess.run(download_command, cwd=PROJECT_ROOT, check=True)

                preprocess_command = _preprocess_command(args, shard, raw_shard_dir)
                record["preprocess_command"] = preprocess_command
                _save_state(args.state_path, state)
                subprocess.run(preprocess_command, cwd=PROJECT_ROOT, check=True)
                validation = _validate_completed_shard(args.output_root.resolve(), shard)
                if validation is None:
                    raise RuntimeError(f"shard did not become complete: {shard_id}")
                record["action"] = "processed"
            else:
                record["action"] = "skipped_valid_complete"

            # 新处理 shard 必须先归档下载报告。已存在的门禁 shard 若已经归档，
            # 即使 raw trainval 已删除，也可以安全跳过。
            archived_report = archive_download_report(raw_shard_dir, processed_shard_dir)
            record["download_report"] = str(archived_report)
            record["download_report_sha256"] = sha256_file(archived_report)
            record["raw_trainval_removed"] = (
                cleanup_raw_trainval(args.raw_root, shard_id) if args.cleanup_raw else False
            )
            record["free_bytes_after_cleanup"] = shutil.disk_usage(args.raw_root).free
            record["validation"] = {
                key: validation[key]
                for key in (
                    "status",
                    "manifest_count",
                    "npz_count",
                    "log_count",
                    "total_bytes",
                    "manifest_sha256",
                    "sampling_report_sha256",
                )
            }
            record["status"] = "complete"
            record["finished_at_utc"] = _utc_now()
            state["next_position"] = position + 1
            processed_this_run += 1
            _save_state(args.state_path, state)

        state["status"] = "complete"
        state["finished_at_utc"] = _utc_now()
        _save_state(args.state_path, state)
        return state
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["traceback"] = traceback.format_exc()
        _save_state(args.state_path, state)
        raise


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--archive_index", required=True, type=Path)
    parser.add_argument("--raw_root", required=True, type=Path)
    parser.add_argument("--map_path", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--state_path", required=True, type=Path)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--shard_indices",
        nargs="+",
        type=int,
        default=None,
        help="optional explicit shard-index whitelist applied before worker assignment",
    )
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--checksum_mode", choices=["manifest", "files"], default="files")
    parser.add_argument("--cleanup_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connect_timeout_seconds", type=float, default=10.0)
    parser.add_argument("--read_timeout_seconds", type=float, default=60.0)
    parser.add_argument("--max_member_retries", type=int, default=3)
    parser.add_argument("--retry_delay_seconds", type=float, default=5.0)
    parser.add_argument(
        "--download_backend",
        choices=["remotezip", "curl", "zip"],
        default="remotezip",
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="NAME=URL_OR_PATH",
        help="forward an archive URL or local ZIP override to the subset downloader",
    )
    parser.add_argument(
        "--only_archive",
        action="append",
        default=[],
        metavar="NAME",
        help="forward an archive-name restriction to the subset downloader",
    )
    parser.add_argument("--curl_low_speed_limit_bps", type=int, default=512 * 1024)
    parser.add_argument("--curl_low_speed_time_seconds", type=int, default=5)
    parser.add_argument(
        "--min_free_gib",
        type=float,
        default=20.0,
        help="stop before a new download when data-disk free space is below this threshold",
    )
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max_shards must be positive")
    if args.shard_indices is not None:
        if any(index < 0 for index in args.shard_indices):
            raise ValueError("--shard_indices values must be non-negative")
        if len(set(args.shard_indices)) != len(args.shard_indices):
            raise ValueError("--shard_indices must not contain duplicates")
    if args.min_free_gib < 0:
        raise ValueError("min_free_gib must be non-negative")
    if args.curl_low_speed_limit_bps <= 0 or args.curl_low_speed_time_seconds <= 0:
        raise ValueError("curl low-speed thresholds must be positive")
    return args


def main() -> None:
    args = get_args()
    state = run(args)
    print(
        f"worker {args.worker_index}/{args.worker_count} finished with "
        f"status={state['status']}, next_position={state['next_position']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
