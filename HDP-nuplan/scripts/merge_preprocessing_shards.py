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
    path = PurePosixPath(name)
    if not isinstance(name, str) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest entry: {name!r}")
    return path.as_posix()


def merge_shards(shards_root, verify_files=True):
    shards_root = Path(shards_root).resolve()
    report_paths = sorted(shards_root.glob("shard_*/processing_report.json"))
    if not report_paths:
        raise ValueError(f"no shard processing reports found under {shards_root}")

    merged_entries = []
    merged_logs = []
    seen_output_names = set()
    seen_logs = set()
    shard_summaries = []
    selected_per_log = Counter()

    for report_path in report_paths:
        shard_dir = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"incomplete shard: {shard_dir.name}")

        manifest_path = shard_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest) != len(set(manifest)):
            raise ValueError(f"duplicate entries inside {manifest_path}")
        if len(manifest) != report.get("manifest_count"):
            raise ValueError(f"manifest/report count mismatch in {shard_dir.name}")

        shard_logs = report.get("log_names", [])
        overlapping_logs = sorted(seen_logs.intersection(shard_logs))
        if overlapping_logs:
            raise ValueError(
                f"logs occur in multiple shards; first overlap={overlapping_logs[0]}"
            )
        seen_logs.update(shard_logs)
        merged_logs.extend(shard_logs)

        sampling_report_path = shard_dir / "sampling_report.json"
        sampling_report = json.loads(sampling_report_path.read_text(encoding="utf-8"))
        shard_selected_per_log = sampling_report.get("selected_per_log", {})
        if sum(shard_selected_per_log.values()) != len(manifest):
            raise ValueError(f"sampling/manifest count mismatch in {shard_dir.name}")
        selected_per_log.update(shard_selected_per_log)

        for raw_name in manifest:
            name = _safe_manifest_name(raw_name)
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
                "manifest_count": len(manifest),
                "failed": report.get("failed", 0),
            }
        )

    return sorted(merged_entries), {
        "status": "complete" if all(item["status"] == "complete" for item in shard_summaries) else "partial",
        "shards_root": str(shards_root),
        "shard_count": len(shard_summaries),
        "log_count": len(merged_logs),
        "log_names": merged_logs,
        "manifest_count": len(merged_entries),
        "requested_scenarios": len(merged_entries),
        "selected_per_log": dict(sorted(selected_per_log.items())),
        "shards": shard_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_root", required=True, type=Path)
    parser.add_argument("--output_manifest", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--no_verify_files", action="store_true")
    args = parser.parse_args()

    manifest, report = merge_shards(
        args.shards_root,
        verify_files=not args.no_verify_files,
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
