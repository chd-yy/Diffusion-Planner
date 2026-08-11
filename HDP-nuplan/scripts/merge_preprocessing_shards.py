#!/usr/bin/env python3
"""合并已完成的预处理分片，输出可直接供 Dataset 使用的总 manifest。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import atomic_write_json, sha256_file  # noqa: E402


def _safe_manifest_name(name):
    if not isinstance(name, str):
        raise ValueError(f"unsafe manifest entry: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest entry: {name!r}")
    return path.as_posix()


def _load_plan(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shards = plan.get("shards", [])
    if not isinstance(shards, list) or not shards:
        raise ValueError("preprocessing plan has no shards")
    shard_ids = [str(shard["shard_id"]) for shard in shards]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("preprocessing plan contains duplicate shard_id")
    if int(plan.get("shard_count", -1)) != len(shards):
        raise ValueError("preprocessing plan shard_count does not match shards")
    shard_indices = [int(shard["shard_index"]) for shard in shards]
    if shard_indices != list(range(len(shards))):
        raise ValueError("preprocessing plan shard_index sequence is not contiguous")
    if shard_ids != [f"shard_{index:05d}" for index in shard_indices]:
        raise ValueError("preprocessing plan shard_id does not match shard_index")
    if sum(int(shard["total_scenarios"]) for shard in shards) != int(
        plan.get("total_scenarios", -1)
    ):
        raise ValueError("preprocessing plan scenario total is inconsistent")

    plan_dir = plan_path.parent
    expected_logs = {}
    all_logs = []
    for shard in shards:
        log_path = plan_dir / str(shard["log_names_json"])
        logs = json.loads(log_path.read_text(encoding="utf-8"))
        if not isinstance(logs, list) or not all(isinstance(item, str) for item in logs):
            raise ValueError(f"invalid plan log list: {log_path}")
        if len(logs) != int(shard["log_count"]):
            raise ValueError(f"plan log_count mismatch: {shard['shard_id']}")
        expected_logs[str(shard["shard_id"])] = logs
        all_logs.extend(logs)
    if len(all_logs) != len(set(all_logs)):
        raise ValueError("preprocessing plan assigns a log to multiple shards")
    if len(all_logs) != int(plan.get("source_log_count", -1)):
        raise ValueError("preprocessing plan source_log_count is inconsistent")
    return plan_path, plan, shards, expected_logs


def _verify_checksum_payload(shard_dir, manifest, processing_path, sampling_path):
    checksum_path = shard_dir / "checksums.json"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "manifest": shard_dir / "manifest.json",
        "processing_report": processing_path,
        "sampling_report": sampling_path,
    }
    for key, path in expected_metadata.items():
        item = payload.get(key, {})
        if item.get("sha256") != sha256_file(path):
            raise ValueError(f"{key} checksum mismatch in {shard_dir.name}")
    if payload.get("npz_checksums_included") is not True:
        raise ValueError(f"NPZ checksums absent in {shard_dir.name}")
    npz_checksums = payload.get("npz_files", {})
    if set(npz_checksums) != set(manifest):
        raise ValueError(f"NPZ checksum/manifest mismatch in {shard_dir.name}")
    for name in manifest:
        if npz_checksums[name] != sha256_file(shard_dir / "cache" / name):
            raise ValueError(f"NPZ checksum mismatch: {shard_dir.name}/{name}")


def _verify_cache_validation(shard_dir, manifest_path, sampling_path, expected_count):
    validation_path = shard_dir / "cache_validation_report.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError(f"cache validation did not pass in {shard_dir.name}")
    if int(validation.get("manifest_count", -1)) != expected_count:
        raise ValueError(f"cache validation manifest count mismatch in {shard_dir.name}")
    if int(validation.get("npz_count", -1)) != expected_count:
        raise ValueError(f"cache validation NPZ count mismatch in {shard_dir.name}")
    if validation.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"cache validation manifest hash mismatch in {shard_dir.name}")
    if validation.get("sampling_report_sha256") != sha256_file(sampling_path):
        raise ValueError(f"cache validation sampling hash mismatch in {shard_dir.name}")


def _verify_download_report(shard_dir, expected_logs):
    report_path = shard_dir / "download_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"download report did not complete in {shard_dir.name}")
    if int(report.get("requested_log_count", -1)) != len(expected_logs):
        raise ValueError(f"download report log count mismatch in {shard_dir.name}")
    if report.get("log_names") != expected_logs:
        raise ValueError(f"download report/plan logs mismatch in {shard_dir.name}")
    files = report.get("files", [])
    if not isinstance(files, list) or len(files) != len(expected_logs):
        raise ValueError(f"download report file list mismatch in {shard_dir.name}")
    file_logs = [item.get("log_name") for item in files]
    if sorted(file_logs) != sorted(expected_logs):
        raise ValueError(f"download report file logs mismatch in {shard_dir.name}")
    for item in files:
        if int(item.get("size", 0)) <= 0:
            raise ValueError(f"invalid downloaded DB size in {shard_dir.name}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid downloaded DB hash in {shard_dir.name}")
    return {
        "retry_count": int(report.get("retry_count", 0)),
        "total_bytes": int(report.get("total_bytes", 0)),
    }


def merge_shards(
    shards_root,
    verify_files=True,
    plan_path=None,
    verify_checksums=False,
    require_cache_validation=False,
):
    shards_root = Path(shards_root).resolve()
    plan = None
    expected_by_id = {}
    expected_logs = {}
    if plan_path is not None:
        plan_path, plan, expected_shards, expected_logs = _load_plan(plan_path)
        expected_by_id = {
            str(shard["shard_id"]): shard for shard in expected_shards
        }
        expected_ids = list(expected_by_id)
        existing_ids = sorted(
            path.name
            for path in shards_root.glob("shard_*")
            if path.is_dir()
        )
        missing = sorted(set(expected_ids) - set(existing_ids))
        unexpected = sorted(set(existing_ids) - set(expected_ids))
        if missing or unexpected:
            raise ValueError(
                f"shard directory set does not match plan: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        report_paths = [
            shards_root / shard_id / "processing_report.json"
            for shard_id in expected_ids
        ]
        missing_reports = [str(path) for path in report_paths if not path.is_file()]
        if missing_reports:
            raise ValueError(f"missing shard processing reports: {missing_reports[:10]}")
    else:
        report_paths = sorted(shards_root.glob("shard_*/processing_report.json"))
    if not report_paths:
        raise ValueError(f"no shard processing reports found under {shards_root}")

    merged_entries = []
    merged_logs = []
    seen_output_names = set()
    seen_logs = set()
    shard_summaries = []
    selected_per_log = Counter()
    empty_logs = []
    total_raw_scenarios = 0
    total_unique_scenarios = 0
    duplicate_output_names_removed = 0
    total_download_retries = 0
    total_download_bytes = 0

    for report_path in report_paths:
        shard_dir = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"incomplete shard: {shard_dir.name}")
        if int(report.get("failed", 0)) != 0:
            raise ValueError(f"failed scenarios in complete shard: {shard_dir.name}")
        if report.get("shard_id", shard_dir.name) != shard_dir.name:
            raise ValueError(f"processing report shard_id mismatch: {shard_dir.name}")

        manifest_path = shard_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list) or not all(
            isinstance(item, str) for item in manifest
        ):
            raise ValueError(f"invalid manifest list in {shard_dir.name}")
        if len(manifest) != len(set(manifest)):
            raise ValueError(f"duplicate entries inside {manifest_path}")
        if len(manifest) != report.get("manifest_count"):
            raise ValueError(f"manifest/report count mismatch in {shard_dir.name}")

        expected_shard = expected_by_id.get(shard_dir.name)
        if expected_shard is not None:
            expected_count = int(expected_shard["total_scenarios"])
            if len(manifest) != expected_count:
                raise ValueError(f"manifest/plan count mismatch in {shard_dir.name}")
            if int(report.get("requested_scenarios", -1)) != expected_count:
                raise ValueError(f"processing report/plan target mismatch in {shard_dir.name}")

        shard_logs = report.get("log_names", [])
        if expected_shard is not None and shard_logs != expected_logs[shard_dir.name]:
            raise ValueError(f"processing report/plan logs mismatch in {shard_dir.name}")
        overlapping_logs = sorted(seen_logs.intersection(shard_logs))
        if overlapping_logs:
            raise ValueError(
                f"logs occur in multiple shards; first overlap={overlapping_logs[0]}"
            )
        seen_logs.update(shard_logs)
        merged_logs.extend(shard_logs)

        sampling_report_path = shard_dir / "sampling_report.json"
        sampling_report = json.loads(sampling_report_path.read_text(encoding="utf-8"))
        if expected_shard is not None:
            if int(sampling_report.get("requested_scenarios", -1)) != expected_count:
                raise ValueError(f"sampling report/plan target mismatch in {shard_dir.name}")
            if sampling_report.get("log_names") != expected_logs[shard_dir.name]:
                raise ValueError(f"sampling report/plan logs mismatch in {shard_dir.name}")
        shard_selected_per_log = sampling_report.get("selected_per_log", {})
        if sum(shard_selected_per_log.values()) != len(manifest):
            raise ValueError(f"sampling/manifest count mismatch in {shard_dir.name}")
        selected_per_log.update(shard_selected_per_log)
        shard_empty_logs = sampling_report.get("empty_logs", [])
        if set(shard_empty_logs).intersection(shard_selected_per_log):
            raise ValueError(f"empty log also has selected scenarios in {shard_dir.name}")
        empty_logs.extend(shard_empty_logs)
        total_raw_scenarios += int(sampling_report.get("raw_scenarios", 0))
        total_unique_scenarios += int(sampling_report.get("unique_scenarios", 0))
        duplicate_output_names_removed += int(
            sampling_report.get("duplicate_output_names_removed", 0)
        )

        if expected_shard is not None:
            download = _verify_download_report(
                shard_dir, expected_logs[shard_dir.name]
            )
            total_download_retries += download["retry_count"]
            total_download_bytes += download["total_bytes"]

        normalized_manifest = [_safe_manifest_name(name) for name in manifest]
        if normalized_manifest != sorted(normalized_manifest):
            raise ValueError(f"manifest is not deterministic/sorted in {shard_dir.name}")
        if verify_files:
            cache_names = {
                path.relative_to(shard_dir / "cache").as_posix()
                for path in (shard_dir / "cache").rglob("*.npz")
            }
            if cache_names != set(normalized_manifest):
                raise ValueError(
                    f"cache/manifest file set mismatch in {shard_dir.name}: "
                    f"missing={len(set(normalized_manifest) - cache_names)}, "
                    f"unexpected={len(cache_names - set(normalized_manifest))}"
                )
        if require_cache_validation:
            _verify_cache_validation(
                shard_dir,
                manifest_path,
                sampling_report_path,
                len(manifest),
            )
        if verify_checksums:
            _verify_checksum_payload(
                shard_dir,
                normalized_manifest,
                report_path,
                sampling_report_path,
            )

        for name in normalized_manifest:
            output_name = PurePosixPath(name).name
            if output_name in seen_output_names:
                raise ValueError(f"duplicate NPZ across shards: {output_name}")
            seen_output_names.add(output_name)

            source_path = shard_dir / "cache" / Path(name)
            if verify_files and not source_path.is_file():
                raise FileNotFoundError(source_path)
            merged_entries.append(source_path.relative_to(shards_root).as_posix())

        shard_summaries.append(
            {
                "shard_id": report.get("shard_id", shard_dir.name),
                "status": report.get("status"),
                "log_count": len(shard_logs),
                "selected_log_count": len(shard_selected_per_log),
                "empty_log_count": len(shard_empty_logs),
                "manifest_count": len(manifest),
                "failed": report.get("failed", 0),
            }
        )

    merged_report = {
        "status": "complete",
        "shards_root": str(shards_root),
        "shard_count": len(shard_summaries),
        "log_count": len(merged_logs),
        "log_names": merged_logs,
        "selected_log_count": len(selected_per_log),
        "empty_log_count": len(empty_logs),
        "empty_logs": sorted(empty_logs),
        "manifest_count": len(merged_entries),
        "requested_scenarios": len(merged_entries),
        "selected_per_log": dict(sorted(selected_per_log.items())),
        "raw_scenarios": total_raw_scenarios,
        "unique_scenarios": total_unique_scenarios,
        "duplicate_output_names_removed": duplicate_output_names_removed,
        "download_retry_count": total_download_retries,
        "download_total_bytes": total_download_bytes,
        "shards": shard_summaries,
    }
    if plan is not None:
        if len(shard_summaries) != int(plan["shard_count"]):
            raise ValueError("merged shard count does not match plan")
        if len(merged_entries) != int(plan["total_scenarios"]):
            raise ValueError("merged manifest count does not match plan")
        if len(merged_logs) != int(plan["source_log_count"]):
            raise ValueError("merged log count does not match plan")
        merged_report.update(
            {
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "planned_shard_count": int(plan["shard_count"]),
                "planned_log_count": int(plan["source_log_count"]),
                "planned_scenarios": int(plan["total_scenarios"]),
                "checksums_verified": verify_checksums,
                "cache_validation_required": require_cache_validation,
                "sampled_fraction_of_unique_candidates": (
                    len(merged_entries) / total_unique_scenarios
                    if total_unique_scenarios
                    else None
                ),
            }
        )
    return sorted(merged_entries), merged_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output_manifest", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--no_verify_files", action="store_true")
    parser.add_argument("--no_verify_checksums", action="store_true")
    args = parser.parse_args()

    manifest, report = merge_shards(
        args.shards_root,
        verify_files=not args.no_verify_files,
        plan_path=args.plan,
        verify_checksums=not args.no_verify_checksums,
        require_cache_validation=True,
    )
    atomic_write_json(args.output_manifest, manifest)
    report["manifest_path"] = str(args.output_manifest.resolve())
    report["manifest_sha256"] = sha256_file(args.output_manifest)
    atomic_write_json(args.output_report, report)
    print(
        f"Merged {report['shard_count']} shards, {report['log_count']} logs, "
        f"{report['manifest_count']} NPZ entries"
    )
    print(args.output_manifest.resolve())


if __name__ == "__main__":
    main()
