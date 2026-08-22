"""合并多份互不重叠的配对闭环评测汇总，并校验每份中的场景一致。"""

import argparse
import json
from pathlib import Path


METRICS = [
    "score",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
    "trajectory_runtime_mean_s",
    "simulation_duration_s",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def merge_label(parts, label):
    scenarios = []
    successful = 0
    failed = 0
    run_dirs = []
    for part in parts:
        run = part[label]
        scenarios.extend(run["scenarios"])
        successful += int(run["successful_simulations"])
        failed += int(run["failed_simulations"])
        run_dirs.append(run["run_dir"])

    ids = [row["scenario"] for row in scenarios]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RuntimeError(f"{label} 存在重复场景: {duplicates}")

    means = {
        metric: sum(float(row[metric]) for row in scenarios) / len(scenarios)
        for metric in METRICS
    }
    return {
        "run_dirs": run_dirs,
        "scenario_count": len(scenarios),
        "successful_simulations": successful,
        "failed_simulations": failed,
        "means": means,
        "scenarios": sorted(scenarios, key=lambda row: row["scenario"]),
    }


def main():
    args = parse_args()
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    for path, part in zip(args.input, parts):
        missing = {args.baseline_label, args.candidate_label}.difference(part)
        if missing:
            raise RuntimeError(f"{path} 缺少 labels: {sorted(missing)}")
        baseline_ids = {
            row["scenario"] for row in part[args.baseline_label]["scenarios"]
        }
        candidate_ids = {
            row["scenario"] for row in part[args.candidate_label]["scenarios"]
        }
        if baseline_ids != candidate_ids:
            raise RuntimeError(f"{path} 中 baseline/candidate 场景不一致")

    result = {
        "source_summaries": [str(path) for path in args.input],
        args.baseline_label: merge_label(parts, args.baseline_label),
        args.candidate_label: merge_label(parts, args.candidate_label),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
