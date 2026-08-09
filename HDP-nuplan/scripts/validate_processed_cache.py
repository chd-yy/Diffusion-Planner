#!/usr/bin/env python3
"""Validate a NuPlan NPZ cache, its manifest, and its sampling audit."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


# 这些 shape 对应当前 HDP-nuPlan 默认数据配置，防止文件存在但张量布局错误。
EXPECTED_SHAPES = {
    "ego_agent_future": (80, 3),
    "ego_current_state": (10,),
    "lanes": (70, 20, 12),
    "lanes_has_speed_limit": (70, 1),
    "lanes_speed_limit": (70, 1),
    "neighbor_agents_future": (32, 80, 3),
    "neighbor_agents_past": (32, 21, 11),
    "route_lanes": (25, 20, 12),
    "route_lanes_has_speed_limit": (25, 1),
    "route_lanes_speed_limit": (25, 1),
    "static_objects": (5, 10),
}
METADATA_KEYS = ("log_name", "map_name", "scenario_type", "token")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(
    cache_dir: Path,
    manifest_path: Path,
    sampling_report_path: Path,
    expected_count: Optional[int] = None,
    expected_log_count: Optional[int] = None,
) -> dict:
    cache_dir = Path(cache_dir)
    manifest_path = Path(manifest_path)
    sampling_report_path = Path(sampling_report_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling_report = json.loads(sampling_report_path.read_text(encoding="utf-8"))

    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError("manifest must be a JSON list of NPZ filenames")
    if len(manifest) != len(set(manifest)):
        raise ValueError("manifest contains duplicate filenames")
    if manifest != sorted(manifest):
        raise ValueError("manifest must be sorted for deterministic replay")
    if expected_count is not None and len(manifest) != expected_count:
        raise ValueError(f"expected {expected_count} manifest items, found {len(manifest)}")

    manifest_names = set(manifest)
    # 总 manifest 可以引用 shard_00000/cache/*.npz 这样的相对路径。
    cache_names = {
        path.relative_to(cache_dir).as_posix()
        for path in cache_dir.rglob("*.npz")
    }
    missing = sorted(manifest_names - cache_names)
    unexpected = sorted(cache_names - manifest_names)
    if missing or unexpected:
        raise ValueError(
            f"cache/manifest mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    required_keys = set(EXPECTED_SHAPES) | set(METADATA_KEYS)
    log_counts: Counter[str] = Counter()
    scenario_type_counts: Counter[str] = Counter()
    total_bytes = 0

    for filename in manifest:
        path = cache_dir / filename
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            absent = sorted(required_keys - set(data.files))
            if absent:
                raise ValueError(f"{filename} is missing keys: {absent}")

            for key, expected_shape in EXPECTED_SHAPES.items():
                value = data[key]
                if value.shape != expected_shape:
                    raise ValueError(
                        f"{filename}:{key} shape={value.shape}, expected={expected_shape}"
                    )
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"{filename}:{key} contains non-finite values")

            for key in METADATA_KEYS:
                if data[key].shape != ():
                    raise ValueError(f"{filename}:{key} must be a scalar")

            log_counts[str(data["log_name"].item())] += 1
            scenario_type_counts[str(data["scenario_type"].item())] += 1

    selected_per_log = {
        str(key): int(value)
        for key, value in sampling_report.get("selected_per_log", {}).items()
    }
    if dict(sorted(log_counts.items())) != dict(sorted(selected_per_log.items())):
        raise ValueError("NPZ log counts do not match sampling_report selected_per_log")
    if expected_log_count is not None and len(log_counts) != expected_log_count:
        raise ValueError(f"expected {expected_log_count} logs, found {len(log_counts)}")
    if sampling_report.get("requested_scenarios") != len(manifest):
        raise ValueError("sampling report requested_scenarios does not match manifest count")

    return {
        "status": "passed",
        "cache_dir": str(cache_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "sampling_report_path": str(sampling_report_path.resolve()),
        "manifest_count": len(manifest),
        "unique_manifest_count": len(manifest_names),
        "npz_count": len(cache_names),
        "log_count": len(log_counts),
        "selected_per_log_min": min(log_counts.values()),
        "selected_per_log_max": max(log_counts.values()),
        "total_bytes": total_bytes,
        "log_counts": dict(sorted(log_counts.items())),
        "scenario_type_counts": dict(sorted(scenario_type_counts.items())),
        "manifest_sha256": _sha256(manifest_path),
        "sampling_report_sha256": _sha256(sampling_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sampling_report", required=True, type=Path)
    parser.add_argument("--expected_count", type=int, default=None)
    parser.add_argument("--expected_log_count", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = validate_cache(
        args.cache_dir,
        args.manifest,
        args.sampling_report,
        args.expected_count,
        args.expected_log_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
