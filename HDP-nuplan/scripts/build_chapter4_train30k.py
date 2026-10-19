"""Build a deterministic 30k Chapter-4 manifest from the existing 306,801 cache.

The script reads metadata and ego motion only.  It never copies or rewrites NPZ
files.  Selection is log-balanced, then diversity-aware within each log using
square-root allocation over (motion bin, scenario type) strata.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_bin(path_length):
    if path_length <= 1.0:
        return "stationary_0_1m"
    if path_length <= 10.0:
        return "low_1_10m"
    if path_length <= 50.0:
        return "medium_10_50m"
    return "high_over_50m"


def allocate_sqrt(counts, total):
    """Capacity-constrained Hamilton allocation with sqrt-frequency weights."""
    counts = {key: int(value) for key, value in counts.items() if value > 0}
    if total < 0 or total > sum(counts.values()):
        raise ValueError("allocation total exceeds available capacity")
    if total == 0:
        return {key: 0 for key in counts}
    weights = {key: math.sqrt(value) for key, value in counts.items()}
    weight_sum = sum(weights.values())
    ideals = {key: total * weights[key] / weight_sum for key in counts}
    allocation = {
        key: min(counts[key], int(math.floor(ideals[key]))) for key in counts
    }
    while sum(allocation.values()) < total:
        available = [key for key in counts if allocation[key] < counts[key]]
        if not available:
            raise RuntimeError("unable to complete capacity-constrained allocation")
        key = max(
            available,
            key=lambda item: (
                ideals[item] - allocation[item],
                counts[item] - allocation[item],
                str(item),
            ),
        )
        allocation[key] += 1
    return allocation


def balanced_log_quotas(counts, total, seed):
    logs = sorted(counts)
    if total < len(logs) or total > sum(counts.values()):
        raise ValueError("invalid total for log-balanced selection")
    base = total // len(logs)
    quotas = {log: min(base, counts[log]) for log in logs}
    remaining = total - sum(quotas.values())
    rng = random.Random(seed)
    order = logs[:]
    rng.shuffle(order)
    while remaining:
        progressed = False
        for log in order:
            if quotas[log] < counts[log]:
                quotas[log] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to fill log quotas")
    return quotas


def summarize(rows):
    return {
        "count": len(rows),
        "maps": dict(sorted(Counter(row["map_name"] for row in rows).items())),
        "logs": dict(sorted(Counter(row["log_name"] for row in rows).items())),
        "scenario_types": dict(
            sorted(Counter(row["scenario_type"] for row in rows).items())
        ),
        "motion_bins": dict(
            sorted(Counter(row["motion_bin"] for row in rows).items())
        ),
    }


def select_rows(rows, total, seed, required_names=()):
    by_log = defaultdict(list)
    row_by_name = {}
    for row in rows:
        by_log[row["log_name"]].append(row)
        row_by_name[row["name"]] = row
    required_names = set(required_names)
    missing_required = sorted(required_names - set(row_by_name))
    if missing_required:
        raise ValueError(f"required entries missing from source: {missing_required[:10]}")
    quotas = balanced_log_quotas(
        {log: len(items) for log, items in by_log.items()}, total, seed
    )
    rng = random.Random(seed)
    selected = [row_by_name[name] for name in sorted(required_names)]
    required_by_log = Counter(row["log_name"] for row in selected)
    strata_report = {}
    for log in sorted(by_log):
        strata = defaultdict(list)
        for row in by_log[log]:
            if row["name"] in required_names:
                continue
            strata[(row["motion_bin"], row["scenario_type"])].append(row)
        for items in strata.values():
            rng.shuffle(items)
        additional = quotas[log] - required_by_log[log]
        if additional < 0:
            raise ValueError(
                f"required manifest exceeds balanced quota for log {log}"
            )
        allocation = allocate_sqrt(
            {key: len(items) for key, items in strata.items()}, additional
        )
        for key in sorted(strata):
            selected.extend(strata[key][: allocation[key]])
        strata_report[log] = {
            f"{key[0]}::{key[1]}": allocation[key]
            for key in sorted(allocation)
            if allocation[key]
        }
    if len(selected) != total or len({row["name"] for row in selected}) != total:
        raise RuntimeError("selection did not produce the requested unique count")
    selected.sort(key=lambda row: row["name"])
    return selected, quotas, strata_report


def read_metadata(source_root, names):
    cache = source_root / "cache"
    rows = []
    for index, name in enumerate(names, start=1):
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as sample:
            future = np.asarray(sample["ego_agent_future"][:, :2], dtype=np.float64)
            if future.ndim != 2 or future.shape[1] != 2 or not np.isfinite(future).all():
                raise ValueError(f"invalid ego future: {name}")
            points = np.concatenate([np.zeros((1, 2)), future], axis=0)
            path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            rows.append({
                "name": name,
                "map_name": str(sample["map_name"].item()),
                "log_name": str(sample["log_name"].item()),
                "scenario_type": str(sample["scenario_type"].item()),
                "token": str(sample["token"].item()),
                "path_length_m": path_length,
                "motion_bin": motion_bin(path_length),
            })
        if index % 10000 == 0 or index == len(names):
            print(f"metadata={index}/{len(names)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("tmp/mini_train_full_306801_seed3407_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/mini_train_chapter4_stratified_30000_seed3407_v1"),
    )
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--required-manifest",
        type=Path,
        default=Path(
            "tmp/mini_train_balanced_10000_seed3407_v1/"
            "diffusion_planner_training.json"
        ),
        help="Entries that must all remain in the expanded manifest.",
    )
    parser.add_argument(
        "--heldout-coverage",
        type=Path,
        default=Path("tmp/test14_full_remote_subset/coverage_validation.json"),
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    source_manifest = source_root / "diffusion_planner_training.json"
    names = json.loads(source_manifest.read_text(encoding="utf-8"))
    if len(names) != 306801 or len(set(names)) != len(names):
        raise ValueError("expected the audited 306,801-entry unique source manifest")
    rows = read_metadata(source_root, names)
    required_names = json.loads(args.required_manifest.read_text(encoding="utf-8"))
    if len(required_names) != len(set(required_names)):
        raise ValueError("required manifest contains duplicate entries")
    selected, log_quotas, strata_allocation = select_rows(
        rows, args.total, args.seed, required_names=required_names
    )
    heldout = json.loads(args.heldout_coverage.read_text(encoding="utf-8"))
    heldout_tokens = set(heldout["test14_hard"]["tokens"]) | set(
        heldout["test14_random"]["tokens"]
    )
    overlap = sorted({row["token"] for row in selected} & heldout_tokens)
    if overlap:
        raise RuntimeError(f"training/Test14 token overlap: {overlap[:10]}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "diffusion_planner_training.json"
    manifest.write_text(
        json.dumps([row["name"] for row in selected], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "created_at": datetime.now().strftime("%m月%d日 %H:%M"),
        "strategy": "nested_10k_plus_balanced_log_sqrt_motion_type_strata",
        "seed": args.seed,
        "source_root": str(source_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_count": len(rows),
        "required_manifest": str(args.required_manifest.resolve()),
        "required_manifest_sha256": sha256(args.required_manifest),
        "required_count": len(required_names),
        "required_entries_retained": len(
            set(required_names) & {row["name"] for row in selected}
        ),
        "selected_manifest": str(manifest),
        "selected_manifest_sha256": sha256(manifest),
        "selected_count": len(selected),
        "npz_files_copied": 0,
        "cache_reused": str(source_root / "cache"),
        "test14_token_overlap": 0,
        "motion_thresholds_m": [1.0, 10.0, 50.0],
        "source_summary": summarize(rows),
        "selected_summary": summarize(selected),
        "log_quotas": dict(sorted(log_quotas.items())),
        "strata_allocation": strata_allocation,
        "limitations": [
            "Anchor samples are unique but are not independent driving episodes.",
            "NPZ metadata does not expose an official scene id or lidar timestamp, so this manifest does not claim temporal de-duplication.",
            "Square-root allocation improves rare-stratum coverage and intentionally does not preserve the raw empirical distribution.",
        ],
    }
    (output / "sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest),
        "selected_count": len(selected),
        "manifest_sha256": report["selected_manifest_sha256"],
        "cache_reused": report["cache_reused"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
