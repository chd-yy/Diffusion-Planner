#!/usr/bin/env python3
"""Generate and optionally validate bounded-memory Test14 evaluation chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
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
    parser.add_argument("--coverage", type=Path, default=work_root / "coverage_validation.json")
    parser.add_argument(
        "--config-root", type=Path, default=work_root / "config" / "scenario_filter"
    )
    parser.add_argument("--manifest", type=Path, default=work_root / "eval_chunk_manifest.json")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--validate-db", action="store_true")
    parser.add_argument("--db-root", type=Path, default=work_root / "data" / "cache" / "test14")
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/yanjun/NewDisk/nuplan/dataset")
    )
    parser.add_argument(
        "--map-root", type=Path, default=Path("/home/yanjun/NewDisk/nuplan/dataset/maps")
    )
    return parser


def render_filter(tokens: List[str]) -> Dict[str, object]:
    config: Dict[str, object] = {
        "_target_": "nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_": "all",
        "scenario_types": None,
        # OmegaConf can interpret unquoted hexadecimal-looking tokens containing
        # ``e`` as scientific notation. Force every token to a quoted YAML scalar.
        "scenario_tokens": [QuotedToken(token) for token in tokens],
    }
    config.update(FILTER_DEFAULTS)
    return config


def generate(args: argparse.Namespace) -> Dict[str, object]:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    args.config_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "coverage_file": str(args.coverage.resolve()),
        "chunk_size": args.chunk_size,
        "benchmarks": {},
    }
    benchmark_keys = {
        "test14-hard": "test14_hard",
        "test14-random": "test14_random",
    }
    for benchmark, coverage_key in benchmark_keys.items():
        tokens = list(coverage[coverage_key]["tokens"])
        if len(tokens) != len(set(tokens)):
            raise RuntimeError(f"Duplicate tokens in {coverage_key}")
        chunks = []
        for index, start in enumerate(range(0, len(tokens), args.chunk_size)):
            chunk_tokens = tokens[start : start + args.chunk_size]
            filter_name = f"{benchmark}-chunk-{index:03d}"
            config_path = args.config_root / f"{filter_name}.yaml"
            config_path.write_text(
                "# Generated bounded-memory Test14 chunk; do not hand-edit.\n"
                + yaml.safe_dump(render_filter(chunk_tokens), sort_keys=False),
                encoding="utf-8",
            )
            chunks.append(
                {
                    "index": index,
                    "filter": filter_name,
                    "config": str(config_path.resolve()),
                    "count": len(chunk_tokens),
                    "tokens": chunk_tokens,
                }
            )
        manifest["benchmarks"][benchmark] = {
            "coverage_key": coverage_key,
            "count": len(tokens),
            "tokens": tokens,
            "chunks": chunks,
        }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_against_db(args: argparse.Namespace, manifest: Dict[str, object]) -> None:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    accepted = {field.name for field in fields(ScenarioFilter)}
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
    for benchmark, benchmark_data in manifest["benchmarks"].items():
        observed_all: List[str] = []
        for chunk in benchmark_data["chunks"]:
            raw = yaml.safe_load(Path(chunk["config"]).read_text(encoding="utf-8"))
            scenario_filter = ScenarioFilter(
                **{key: value for key, value in raw.items() if key in accepted}
            )
            observed = [
                scenario.token
                for scenario in builder.get_scenarios(scenario_filter, Sequential())
            ]
            if set(observed) != set(chunk["tokens"]) or len(observed) != chunk["count"]:
                raise RuntimeError(
                    f"DB validation failed for {benchmark} chunk {chunk['index']}: "
                    f"expected={chunk['count']} observed={len(observed)}"
                )
            observed_all.extend(observed)
        if len(observed_all) != len(set(observed_all)):
            raise RuntimeError(f"Cross-chunk duplicate tokens in {benchmark}")
        if set(observed_all) != set(benchmark_data["tokens"]):
            raise RuntimeError(f"Cross-chunk coverage mismatch in {benchmark}")
        print(
            f"Validated {benchmark}: {len(observed_all)} scenarios in "
            f"{len(benchmark_data['chunks'])} chunks"
        )


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate(args)
    for benchmark, data in manifest["benchmarks"].items():
        print(f"Generated {benchmark}: {data['count']} scenarios, {len(data['chunks'])} chunks")
    if args.validate_db:
        validate_against_db(args, manifest)


if __name__ == "__main__":
    main()
