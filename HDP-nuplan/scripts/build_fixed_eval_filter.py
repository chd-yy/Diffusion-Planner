#!/usr/bin/env python3
"""Build a deterministic mini-val scenario-token filter for paired closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-db-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--source-filter", type=Path, required=True)
    parser.add_argument("--existing-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-filter", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_existing_tokens(paths: Iterable[Path]) -> List[str]:
    tokens: List[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for label in ("hdp_b_epoch10", "hdp_rl_variant"):
            for row in payload[label]["scenarios"]:
                tokens.append(row["scenario"])
    return sorted(set(tokens))


def select_balanced(scenarios: List[object], existing_tokens: List[str], target_count: int) -> List[object]:
    by_token = {scenario.token: scenario for scenario in scenarios}
    missing = sorted(set(existing_tokens) - set(by_token))
    if missing:
        raise RuntimeError(f"Existing evaluation scenarios not found in mini-val source: {missing}")

    selected = [by_token[token] for token in existing_tokens]
    if len(selected) > target_count:
        raise RuntimeError(f"Existing scenarios ({len(selected)}) exceed target count ({target_count})")

    selected_tokens = set(existing_tokens)
    candidates = [scenario for scenario in scenarios if scenario.token not in selected_tokens]
    candidates.sort(key=lambda scenario: (scenario.scenario_type, scenario.log_name, scenario.token))

    type_counts = Counter(scenario.scenario_type for scenario in selected)
    log_counts = Counter(scenario.log_name for scenario in selected)
    by_type: Dict[str, List[object]] = defaultdict(list)
    for scenario in candidates:
        by_type[scenario.scenario_type].append(scenario)

    # Greedily fill the least represented type and log. Ties are deterministic.
    while len(selected) < target_count and candidates:
        candidates.sort(
            key=lambda scenario: (
                type_counts[scenario.scenario_type],
                log_counts[scenario.log_name],
                scenario.scenario_type,
                scenario.log_name,
                scenario.token,
            )
        )
        scenario = candidates.pop(0)
        selected.append(scenario)
        type_counts[scenario.scenario_type] += 1
        log_counts[scenario.log_name] += 1

    if len(selected) != target_count:
        raise RuntimeError(f"Only {len(selected)} valid scenarios available; target was {target_count}")
    return selected


def write_filter(path: Path, scenarios: List[object]) -> None:
    payload = {
        "_target_": "nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_": "all",
        "scenario_types": None,
        "scenario_tokens": [scenario.token for scenario in scenarios],
        "log_names": None,
        "map_names": None,
        "num_scenarios_per_type": None,
        "limit_total_scenarios": None,
        "timestamp_threshold_s": None,
        "ego_displacement_minimum_m": None,
        "ego_start_speed_threshold": None,
        "ego_stop_speed_threshold": None,
        "speed_noise_tolerance": None,
        "expand_scenarios": False,
        "remove_invalid_goals": True,
        "shuffle": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = yaml.safe_load(args.source_filter.read_text(encoding="utf-8"))
    log_names = source["log_names"]
    builder = NuPlanScenarioBuilder(
        str(args.mini_db_root),
        str(args.map_root),
        sensor_root=None,
        db_files=str(args.mini_db_root),
        map_version="nuplan-maps-v1.0",
        max_workers=args.workers,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        # Existing fixed20 tokens were selected without a timestamp spacing filter;
        # keep the full mini-val pool here so all previous 59 scenarios remain valid.
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    existing_tokens = load_existing_tokens(args.existing_summary)
    selected = select_balanced(scenarios, existing_tokens, args.target_count)
    write_filter(args.output_filter, selected)

    manifest = {
        "target_count": args.target_count,
        "scenario_count": len(selected),
        "selection_policy": "preserve existing fixed59, then greedily balance scenario_type and log_name",
        "source_filter": str(args.source_filter),
        "existing_summary": [str(path) for path in args.existing_summary],
        "scenario_tokens": [scenario.token for scenario in selected],
        "scenarios": [
            {
                "scenario": scenario.token,
                "scenario_type": scenario.scenario_type,
                "log_name": scenario.log_name,
            }
            for scenario in selected
        ],
        "counts_by_log": dict(sorted(Counter(scenario.log_name for scenario in selected).items())),
        "counts_by_type": dict(sorted(Counter(scenario.scenario_type for scenario in selected).items())),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"available": len(scenarios), "existing": len(existing_tokens), "selected": len(selected)}, indent=2))
    print(f"filter: {args.output_filter}")
    print(f"manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
