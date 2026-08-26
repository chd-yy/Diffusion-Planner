"""对 fixed200 三组闭环结果做配对分析并生成 JSON/Markdown。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def native(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def paired_delta(baseline, candidate, metric):
    base = {row["scenario"]: row.get(metric) for row in baseline["scenarios"]}
    cand = {row["scenario"]: row.get(metric) for row in candidate["scenarios"]}
    if set(base) != set(cand):
        raise RuntimeError(f"场景集合不一致，无法配对计算 {metric}")
    values = [
        float(cand[key]) - float(base[key])
        for key in sorted(base)
        if base[key] is not None and cand[key] is not None
    ]
    array = np.asarray(values, dtype=np.float64)
    tol = 1e-12
    return {
        "count": int(array.size),
        "mean_delta": native(array.mean()) if array.size else None,
        "median_delta": native(np.median(array)) if array.size else None,
        "wins": int((array > tol).sum()),
        "losses": int((array < -tol).sum()),
        "ties": int((np.abs(array) <= tol).sum()),
    }


def analyze_pair(baseline, candidate):
    result = {}
    for metric in METRICS:
        result[metric] = paired_delta(baseline, candidate, metric)
    return result


def status(summary, pairs):
    required = {"hdp_b_epoch10", "hdp_rl_seed42", "hdp_rl_seed2026"}
    if set(summary) != required:
        return "input_incomplete"
    if any(run["failed_simulations"] or run["scenario_count"] != 200 for run in summary.values()):
        return "evaluation_incomplete"
    score42 = pairs["rl_seed42"]["score"]["mean_delta"]
    score2026 = pairs["rl_seed2026"]["score"]["mean_delta"]
    collision42 = pairs["rl_seed42"]["no_ego_at_fault_collisions"]["mean_delta"]
    collision2026 = pairs["rl_seed2026"]["no_ego_at_fault_collisions"]["mean_delta"]
    if all(value is not None and value > 0 for value in [score42, score2026]) and all(
        value is not None and value >= 0 for value in [collision42, collision2026]
    ):
        return "consistent_positive"
    if any(value is not None and value > 0 for value in [score42, score2026]):
        return "mixed"
    return "not_positive"


def render_markdown(result):
    lines = [
        "# fixed200 配对闭环自动分析",
        "",
        f"- 评测状态：`{result['status']}`",
        f"- 基线：`{result['baseline']}`",
        "- 配对规则：同一 `scenario` 逐场景相减，候选值减基线值；均值指标越大越好。",
        "",
        "## 总体均值差值",
        "",
        "| 指标 | RL seed42 - B10 | RL seed2026 - B10 |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        a = result["pairs"]["rl_seed42"][metric]["mean_delta"]
        b = result["pairs"]["rl_seed2026"][metric]["mean_delta"]
        lines.append(f"| `{metric}` | {a:.8f} | {b:.8f} |" if a is not None and b is not None else f"| `{metric}` | {a} | {b} |")
    lines += [
        "",
        "## 逐场景胜负统计",
        "",
        "| 指标 | seed42 胜/负/平 | seed2026 胜/负/平 |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        cells = []
        for label in ["rl_seed42", "rl_seed2026"]:
            item = result["pairs"][label][metric]
            cells.append(f"{item['wins']}/{item['losses']}/{item['ties']}")
        lines.append(f"| `{metric}` | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## 自动判断",
        "",
        "- `consistent_positive`：两个 seed 的 score 均值都提升，且无责任碰撞指标不下降。",
        "- `mixed`：两个 seed 的结果方向不一致，不能宣称 RL 稳定带来正收益。",
        "- `not_positive`：两个 seed 均未显示 score 正收益。",
        "- 若状态为 `evaluation_incomplete`，不能作任何收益结论。",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    baseline = summary["hdp_b_epoch10"]
    pairs = {
        "rl_seed42": analyze_pair(baseline, summary["hdp_rl_seed42"]),
        "rl_seed2026": analyze_pair(baseline, summary["hdp_rl_seed2026"]),
    }
    result = {
        "summary_file": str(args.summary),
        "baseline": "hdp_b_epoch10",
        "status": status(summary, pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
