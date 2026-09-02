#!/usr/bin/env python3
"""Validate complete Test14-hard/random scenario coverage without loading a planner."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import yaml

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


def load_filter(path: Path) -> ScenarioFilter:
    """Load only fields accepted by the nuPlan ``ScenarioFilter`` dataclass."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    accepted = {field.name for field in fields(ScenarioFilter)}
    values = {key: value for key, value in raw.items() if key in accepted}
    return ScenarioFilter(**values)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    project_root = Path(__file__).resolve().parents[2]
    hdp_root = project_root / "HDP-nuplan"
    work_root = hdp_root / "tmp" / "test14_full_remote_subset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-root", type=Path, default=work_root / "data" / "cache" / "test14"
    )
    parser.add_argument(
        "--map-root", type=Path, default=Path("/home/yanjun/NewDisk/nuplan/dataset/maps")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/yanjun/NewDisk/nuplan/dataset")
    )
    parser.add_argument(
        "--hard-config",
        type=Path,
        default=hdp_root / "hdp_nuplan" / "config" / "scenario_filter" / "test14-hard.yaml",
    )
    parser.add_argument(
        "--random-config",
        type=Path,
        default=work_root / "config" / "scenario_filter" / "test14-random-full.yaml",
    )
    parser.add_argument("--expected-hard", type=int, default=272)
    parser.add_argument("--expected-random", type=int, default=261)
    parser.add_argument("--output", type=Path, default=work_root / "coverage_validation.json")
    return parser


def evaluate_filter(builder: NuPlanScenarioBuilder, config_path: Path) -> Dict[str, object]:
    """Build scenarios and return deterministic coverage details."""

    scenarios = builder.get_scenarios(load_filter(config_path), Sequential())
    tokens = [scenario.token for scenario in scenarios]
    counts_by_type: Dict[str, int] = {}
    for scenario in scenarios:
        counts_by_type[scenario.scenario_type] = counts_by_type.get(scenario.scenario_type, 0) + 1
    return {
        "count": len(scenarios),
        "unique_token_count": len(set(tokens)),
        "duplicate_tokens": sorted(token for token in set(tokens) if tokens.count(token) > 1),
        "counts_by_type": dict(sorted(counts_by_type.items())),
        "tokens": tokens,
    }


def main() -> None:
    """Validate both full Test14 filters and fail closed on any mismatch."""

    args = build_parser().parse_args()
    if not args.db_root.is_dir():
        raise FileNotFoundError(f"Test14 DB root does not exist: {args.db_root}")
    db_count = len(list(args.db_root.glob("*.db")))
    if db_count == 0:
        raise RuntimeError(f"No DB files found in {args.db_root}")
    builder = NuPlanScenarioBuilder(
        data_root=str(args.data_root),
        map_root=str(args.map_root),
        sensor_root=str(args.data_root / "nuplan-v1.1" / "sensor_blobs"),
        db_files=str(args.db_root),
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=None,
        verbose=False,
    )
    result = {
        "db_root": str(args.db_root.resolve()),
        "db_count": db_count,
        "test14_hard": evaluate_filter(builder, args.hard_config),
        "test14_random": evaluate_filter(builder, args.random_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    hard_count = result["test14_hard"]["count"]
    random_count = result["test14_random"]["count"]
    print(f"test14-hard={hard_count}, test14-random={random_count}, db_count={db_count}")
    if hard_count != args.expected_hard:
        raise RuntimeError(f"Expected test14-hard={args.expected_hard}, got {hard_count}")
    if random_count != args.expected_random:
        raise RuntimeError(f"Expected test14-random={args.expected_random}, got {random_count}")
    for benchmark in ("test14_hard", "test14_random"):
        if result[benchmark]["unique_token_count"] != result[benchmark]["count"]:
            raise RuntimeError(f"Duplicate scenario tokens detected in {benchmark}")


if __name__ == "__main__":
    main()
