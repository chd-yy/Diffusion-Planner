#!/usr/bin/env python3
"""Build deterministic mini train/val/test log manifests from nuPlan's official split file."""

import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-db-dir", required=True, type=Path)
    parser.add_argument("--splitter-yaml", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    local_logs = {path.stem for path in args.mini_db_dir.glob("*.db")}
    if not local_logs:
        raise RuntimeError(f"No .db files found under {args.mini_db_dir}")

    with args.splitter_yaml.open("r", encoding="utf-8") as split_file:
        official_splits = yaml.safe_load(split_file)["log_splits"]

    manifests = {
        split_name: sorted(local_logs.intersection(log_names))
        for split_name, log_names in official_splits.items()
        if split_name in {"train", "val", "test"}
    }

    assigned_logs = set().union(*(set(logs) for logs in manifests.values()))
    unassigned_logs = sorted(local_logs - assigned_logs)
    duplicate_count = sum(len(logs) for logs in manifests.values()) - len(assigned_logs)
    if duplicate_count:
        raise RuntimeError(f"Official splits overlap for {duplicate_count} local logs")
    if unassigned_logs:
        raise RuntimeError(f"Local logs missing from official splits: {unassigned_logs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        output_path = args.output_dir / f"mini_{split_name}_logs.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(manifests[split_name], output_file, indent=2)
            output_file.write("\n")
        print(f"{split_name}: {len(manifests[split_name])} -> {output_path}")


if __name__ == "__main__":
    main()
