#!/usr/bin/env python3
"""把 NuPlan 日志清单切成小分片，并精确分配场景目标。"""

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import atomic_write_json, unique_log_names  # noqa: E402


def build_plan(log_names, logs_per_shard, total_scenarios, seed, output_dir):
    """生成互不重叠的日志分片，并保证各分片场景目标之和精确。"""
    log_names = unique_log_names(log_names)
    if logs_per_shard <= 0:
        raise ValueError("logs_per_shard must be positive")
    if not log_names:
        raise ValueError("log list is empty")
    if total_scenarios < len(log_names):
        raise ValueError(
            "balanced_logs requires total_scenarios >= number of unique logs"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_per_log, remainder = divmod(total_scenarios, len(log_names))
    per_log_targets = [
        base_per_log + (1 if index < remainder else 0)
        for index in range(len(log_names))
    ]

    shards = []
    for shard_index, start in enumerate(range(0, len(log_names), logs_per_shard)):
        shard_logs = log_names[start:start + logs_per_shard]
        shard_targets = per_log_targets[start:start + len(shard_logs)]
        shard_id = f"shard_{shard_index:05d}"
        log_file = output_dir / f"{shard_id}_logs.json"
        atomic_write_json(log_file, shard_logs)
        shards.append(
            {
                "shard_index": shard_index,
                "shard_id": shard_id,
                "log_names_json": log_file.name,
                "log_count": len(shard_logs),
                "total_scenarios": sum(shard_targets),
                "seed": seed + shard_index,
            }
        )

    plan = {
        "format_version": 1,
        "strategy": "balanced_logs",
        "source_log_count": len(log_names),
        "logs_per_shard": logs_per_shard,
        "shard_count": math.ceil(len(log_names) / logs_per_shard),
        "total_scenarios": total_scenarios,
        "seed": seed,
        "shards": shards,
    }
    if sum(item["log_count"] for item in shards) != len(log_names):
        raise AssertionError("internal error: log allocation mismatch")
    if sum(item["total_scenarios"] for item in shards) != total_scenarios:
        raise AssertionError("internal error: scenario allocation mismatch")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_names_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--logs_per_shard", type=int, default=100)
    parser.add_argument("--total_scenarios", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    log_names = json.loads(args.log_names_json.read_text(encoding="utf-8"))
    plan = build_plan(
        log_names,
        args.logs_per_shard,
        args.total_scenarios,
        args.seed,
        args.output_dir,
    )
    plan_path = args.output_dir / "preprocessing_plan.json"
    atomic_write_json(plan_path, plan)
    print(
        f"Created {plan['shard_count']} shards for {plan['source_log_count']} logs; "
        f"scenario target={plan['total_scenarios']}"
    )
    print(plan_path.resolve())


if __name__ == "__main__":
    main()
