#!/usr/bin/env python3
"""检查训练集与验证集是否共享 NuPlan 日志、场景文件或场景 token。"""

import argparse
import json
import sys
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import atomic_write_json  # noqa: E402


def _scenario_token(manifest_entry):
    """从 ``<map_name>_<scenario_token>.npz`` 提取场景 token。"""
    filename = PurePosixPath(manifest_entry).name
    if not filename.endswith(".npz") or "_" not in filename:
        raise ValueError(f"无法从 manifest 条目提取场景 token: {manifest_entry}")
    token = filename[:-4].rsplit("_", 1)[-1]
    if not token:
        raise ValueError(f"manifest 条目的场景 token 为空: {manifest_entry}")
    return token


def audit_splits(train_manifest, train_report, val_manifest, val_report):
    train_entries = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_entries = json.loads(Path(val_manifest).read_text(encoding="utf-8"))
    train_metadata = json.loads(Path(train_report).read_text(encoding="utf-8"))
    val_metadata = json.loads(Path(val_report).read_text(encoding="utf-8"))

    train_names = {PurePosixPath(item).name for item in train_entries}
    val_names = {PurePosixPath(item).name for item in val_entries}
    # 早期单缓存报告没有 log_names，只记录 selected_per_log；两者取并集，
    # 避免把实际含日志的验证集错误报告为 val_log_count=0。
    train_logs = set(train_metadata.get("log_names", [])) | set(
        train_metadata.get("selected_per_log", {})
    )
    val_logs = set(val_metadata.get("log_names", [])) | set(
        val_metadata.get("selected_per_log", {})
    )
    overlapping_logs = sorted(train_logs & val_logs)
    overlapping_npz = sorted(train_names & val_names)
    train_tokens = {_scenario_token(item) for item in train_entries}
    val_tokens = {_scenario_token(item) for item in val_entries}
    overlapping_tokens = sorted(train_tokens & val_tokens)

    return {
        "status": (
            "passed"
            if not overlapping_logs and not overlapping_npz and not overlapping_tokens
            else "failed"
        ),
        "train_log_count": len(train_logs),
        "val_log_count": len(val_logs),
        "train_manifest_count": len(train_entries),
        "val_manifest_count": len(val_entries),
        "overlapping_log_count": len(overlapping_logs),
        "overlapping_logs": overlapping_logs,
        "overlapping_npz_count": len(overlapping_npz),
        "overlapping_npz": overlapping_npz,
        "train_unique_token_count": len(train_tokens),
        "val_unique_token_count": len(val_tokens),
        "overlapping_token_count": len(overlapping_tokens),
        "overlapping_tokens": overlapping_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", required=True, type=Path)
    parser.add_argument("--train_report", required=True, type=Path)
    parser.add_argument("--val_manifest", required=True, type=Path)
    parser.add_argument("--val_report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = audit_splits(
        args.train_manifest,
        args.train_report,
        args.val_manifest,
        args.val_report,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
