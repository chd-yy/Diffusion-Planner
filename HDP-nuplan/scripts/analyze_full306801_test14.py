#!/usr/bin/env python3
"""Create the auditable extended analysis for the full-306801 Test14 run."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_midterm_paired import compare, validate_run


BENCHMARKS = ("test14-hard", "test14-random")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_261_006)
    args = parser.parse_args()

    sources = {
        benchmark: args.input_dir
        / f"{benchmark}_historical_B10_vs_full306801.json"
        for benchmark in BENCHMARKS
    }
    summaries = {
        benchmark: json.loads(path.read_text(encoding="utf-8"))
        for benchmark, path in sources.items()
    }

    identities = []
    results = {}
    for benchmark in BENCHMARKS:
        summary = summaries[benchmark]
        baseline = summary["historical_B10"]
        candidate = summary["full306801_posttrain"]
        results[benchmark] = compare(
            baseline,
            candidate,
            repeats=args.resamples,
            seed=args.seed,
        )
        identities.append(set(validate_run(baseline)))

    result = {
        "created_at": datetime.now().strftime("%m月%d日 %H:%M"),
        "analysis_version": 1,
        "resamples": args.resamples,
        "seed": args.seed,
        "numpy_version": np.__version__,
        "script_sha256": sha256(Path(__file__)),
        "between_benchmark_overlap": len(identities[0] & identities[1]),
        "promotion_authorized": False,
        "benchmarks": results,
        "inputs_sha256": {
            str(path.resolve()): sha256(path) for path in sources.values()
        },
        "limitations": [
            "Intervals condition on one checkpoint and one inference seed.",
            "Hard and random are reported separately because they overlap.",
            "Historical B10 was identity-validated and reused, not rerun.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
