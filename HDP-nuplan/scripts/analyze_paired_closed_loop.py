"""对两个使用同一场景集合的闭环结果做逐场景配对分析。"""

import argparse
import json
from pathlib import Path

import numpy as np


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
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def paired_metric(baseline, candidate, metric):
    base = {row["scenario"]: row.get(metric) for row in baseline["scenarios"]}
    cand = {row["scenario"]: row.get(metric) for row in candidate["scenarios"]}
    if set(base) != set(cand):
        raise RuntimeError(f"场景集合不一致，无法配对计算 {metric}")
    values = np.asarray(
        [
            float(cand[token]) - float(base[token])
            for token in sorted(base)
            if base[token] is not None and cand[token] is not None
        ],
        dtype=np.float64,
    )
    tolerance = 1e-12
    return {
        "count": int(values.size),
        "mean_delta": float(values.mean()) if values.size else None,
        "median_delta": float(np.median(values)) if values.size else None,
        "wins": int((values > tolerance).sum()),
        "losses": int((values < -tolerance).sum()),
        "ties": int((np.abs(values) <= tolerance).sum()),
    }


def render_markdown(result):
    lines = [
        "# 闭环配对评测结果",
        "",
        f"- 基线：`{result['baseline']}`",
        f"- 候选：`{result['candidate']}`",
        f"- 场景集合一致：`{result['same_scenario_set']}`",
        f"- 评测完整：`{result['evaluation_complete']}`",
        "- 差值定义：候选减基线，指标均为越大越好。",
        "",
        "| 指标 | 基线均值 | 候选均值 | 平均差值 | 胜/负/平 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = result["metrics"][metric]
        lines.append(
            f"| `{metric}` | {item['baseline_mean']:.8f} | "
            f"{item['candidate_mean']:.8f} | {item['mean_delta']:.8f} | "
            f"{item['wins']}/{item['losses']}/{item['ties']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    baseline = summary[args.baseline]
    candidate = summary[args.candidate]
    base_tokens = {row["scenario"] for row in baseline["scenarios"]}
    candidate_tokens = {row["scenario"] for row in candidate["scenarios"]}
    metrics = {}
    for metric in METRICS:
        paired = paired_metric(baseline, candidate, metric)
        paired["baseline_mean"] = float(baseline["means"][metric])
        paired["candidate_mean"] = float(candidate["means"][metric])
        metrics[metric] = paired
    result = {
        "summary_file": str(args.summary),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "same_scenario_set": base_tokens == candidate_tokens,
        "evaluation_complete": all(
            run["scenario_count"] == len(base_tokens) and run["failed_simulations"] == 0
            for run in (baseline, candidate)
        ),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
