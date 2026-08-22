#!/usr/bin/env python3
"""执行 preprocessing_plan.json 中的一个 NuPlan 预处理分片。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import atomic_write_json  # noqa: E402


def build_command(args, shard):
    shard_dir = args.output_root.resolve() / shard["shard_id"]
    plan_dir = args.plan.resolve().parent
    cache_dir = (
        args.shared_cache_path.resolve()
        if args.shared_cache_path is not None
        else shard_dir / "cache"
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "data_process.py"),
        "--data_path", str(args.data_path.resolve()),
        "--map_path", str(args.map_path.resolve()),
        "--save_path", str(cache_dir),
        "--log_names_json", str(plan_dir / shard["log_names_json"]),
        "--total_scenarios", str(shard["total_scenarios"]),
        "--seed", str(shard["seed"]),
        "--sampling_strategy", "balanced_logs",
        "--shard_id", shard["shard_id"],
        "--output_list_path", str(shard_dir / "manifest.json"),
        "--sampling_report_path", str(shard_dir / "sampling_report.json"),
        "--processing_report_path", str(shard_dir / "processing_report.json"),
        "--checksum_path", str(shard_dir / "checksums.json"),
        "--checksum_mode", args.checksum_mode,
        "--skip_existing", str(args.skip_existing).lower(),
        "--fail_on_error", str(args.fail_on_error).lower(),
    ]
    if args.scenario_builder_workers is not None:
        command.extend(
            ["--scenario_builder_workers", str(args.scenario_builder_workers)]
        )
    command.extend(args.extra_args)
    return shard_dir, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--shard_index", required=True, type=int)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--map_path", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument(
        "--shared_cache_path",
        type=Path,
        help="optional common cache used by disjoint-log workers for resumable parallel processing",
    )
    parser.add_argument("--checksum_mode", choices=["manifest", "files"], default="manifest")
    parser.add_argument(
        "--scenario_builder_workers",
        type=int,
        help="maximum scenario-builder subprocesses used by this shard",
    )
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    shards = plan.get("shards", [])
    if args.shard_index < 0 or args.shard_index >= len(shards):
        raise IndexError(
            f"shard_index={args.shard_index} outside [0, {len(shards) - 1}]"
        )

    shard = shards[args.shard_index]
    shard_dir, command = build_command(args, shard)
    shard_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        shard_dir / "launch.json",
        {
            "plan": str(args.plan.resolve()),
            "shard": shard,
            "shared_cache_path": (
                str(args.shared_cache_path.resolve())
                if args.shared_cache_path is not None
                else None
            ),
            "command": command,
        },
    )
    print(f"Running {shard['shard_id']} with {shard['log_count']} logs")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
