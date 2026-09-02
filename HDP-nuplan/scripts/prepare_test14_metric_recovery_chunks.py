#!/usr/bin/env python3
"""Build chunks only for Test14 scenarios missing valid per-scenario metric files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import yaml


class QuotedToken(str):
    """A scenario token that must remain a YAML string under OmegaConf."""


def _represent_quoted_token(dumper: yaml.SafeDumper, value: QuotedToken) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(value), style='"')


yaml.SafeDumper.add_representer(QuotedToken, _represent_quoted_token)


FILTER_DEFAULTS = {
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


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    work_root = project_root / "HDP-nuplan" / "tmp" / "test14_full_remote_subset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=work_root / "eval_chunk_manifest.json")
    parser.add_argument("--benchmark", default="test14-hard")
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--planner-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument(
        "--config-root", type=Path, default=work_root / "config" / "scenario_filter"
    )
    parser.add_argument(
        "--manifest", type=Path, default=work_root / "aligned_original_hard_recovery_manifest.json"
    )
    parser.add_argument("--filter-prefix", default="test14-hard-aligned-original-recovery")
    return parser


def read_valid_metric_token(path: Path, expected_planner: str) -> str:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError(f"Invalid metric payload: {path}")
    scenario_name = str(payload[0]["scenario_name"])
    planner_name = str(payload[0]["planner_name"])
    if planner_name != expected_planner:
        raise RuntimeError(
            f"Planner mismatch in {path}: expected={expected_planner}, actual={planner_name}"
        )
    return scenario_name


def render_filter(tokens: List[str]) -> Dict[str, object]:
    config: Dict[str, object] = {
        "_target_": "nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_": "all",
        "scenario_types": None,
        "scenario_tokens": [QuotedToken(token) for token in tokens],
    }
    config.update(FILTER_DEFAULTS)
    return config


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    benchmark = base["benchmarks"][args.benchmark]
    all_tokens = list(benchmark["tokens"])
    expected = set(all_tokens)
    completed = [
        read_valid_metric_token(path, args.planner_name)
        for path in sorted(args.metrics_dir.glob("*.pickle.temp"))
    ]
    if len(completed) != len(set(completed)):
        raise RuntimeError("Duplicate scenario tokens in recovered metric files")
    unexpected = set(completed) - expected
    if unexpected:
        raise RuntimeError(f"Unexpected recovered metric tokens: {sorted(unexpected)}")
    completed_set = set(completed)
    missing = [token for token in all_tokens if token not in completed_set]
    args.config_root.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, len(missing), args.chunk_size)):
        tokens = missing[start : start + args.chunk_size]
        filter_name = f"{args.filter_prefix}-chunk-{index:03d}"
        config_path = args.config_root / f"{filter_name}.yaml"
        config_path.write_text(
            "# Generated metric-recovery chunk; do not hand-edit.\n"
            + yaml.safe_dump(render_filter(tokens), sort_keys=False),
            encoding="utf-8",
        )
        chunks.append(
            {
                "index": index,
                "filter": filter_name,
                "config": str(config_path.resolve()),
                "count": len(tokens),
                "tokens": tokens,
            }
        )
    recovery = {
        "base_manifest": str(args.base_manifest.resolve()),
        "chunk_size": args.chunk_size,
        "benchmarks": {
            args.benchmark: {
                "count": len(all_tokens),
                "tokens": all_tokens,
                "runner_tokens": missing,
                "recovered_metric_tokens": [
                    token for token in all_tokens if token in completed_set
                ],
                "metric_only_recovered_count": len(completed),
                "chunks": chunks,
            }
        },
    }
    args.manifest.write_text(
        json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Recovered valid metrics={len(completed)}, missing={len(missing)}, "
        f"chunks={len(chunks)}"
    )


if __name__ == "__main__":
    main()
