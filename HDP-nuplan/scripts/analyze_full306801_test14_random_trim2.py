"""Post-hoc Test14-random sensitivity analysis excluding the two worst score deltas.

The original 261-scene evaluation is never changed. The exclusion set is derived
from paired score deltas in the archived random-scene summary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_midterm_paired import clustered_interval, validate_run
from analyze_paired_closed_loop import METRICS


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP")
INPUT = EXTERNAL / "test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/test14-random_historical_B10_vs_full306801.json"
EXTENDED = EXTERNAL / "test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006/extended_paired_analysis.json"
OUTPUT = PROJECT / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_full306801_posttraining"
FIGURES = OUTPUT / "figures"
REPEATS = 100_000
SEED = 20_261_006


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(keys: list[tuple[str, str]], baseline: dict, candidate: dict) -> dict:
    b = np.array([[baseline[key][metric] for metric in METRICS] for key in keys])
    c = np.array([[candidate[key][metric] for metric in METRICS] for key in keys])
    delta = c - b
    intervals = clustered_interval(delta, [key[0] for key in keys], REPEATS, SEED)
    metrics = {}
    for i, metric in enumerate(METRICS):
        d = delta[:, i]
        metrics[metric] = {
            "baseline_mean": float(b[:, i].mean()),
            "candidate_mean": float(c[:, i].mean()),
            "mean_delta": float(d.mean()),
            "wins": int((d > 1e-6).sum()),
            "losses": int((d < -1e-6).sum()),
            "ties": int((abs(d) <= 1e-6).sum()),
            "log_cluster_percentile_interval_95": intervals[i],
        }
    score_delta = delta[:, METRICS.index("score")]
    negative = np.maximum(-score_delta, 0)
    collision_regressions = [
        {"log_name": key[0], "token": key[1], "score_delta": float(score_delta[i])}
        for i, key in enumerate(keys)
        if baseline[key]["no_ego_at_fault_collisions"] == 1
        and candidate[key]["no_ego_at_fault_collisions"] == 0
    ]
    return {
        "scene_count": len(keys),
        "log_count": len({key[0] for key in keys}),
        "metrics": metrics,
        "new_collision_regressions": collision_regressions,
        "positive_score_sum": float(np.maximum(score_delta, 0).sum()),
        "negative_score_sum": float(negative.sum()),
        "top2_share_of_negative_score_sum": float(np.sort(negative)[-2:].sum() / negative.sum()),
        "top4_share_of_negative_score_sum": float(np.sort(negative)[-4:].sum() / negative.sum()),
    }


def plot_comparison(original: dict, trimmed: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "AR PL SungtiL GB",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    panels = [
        ("score", "综合得分"),
        ("ego_progress_along_expert_route", "路线进度"),
        ("no_ego_at_fault_collisions", "无责任碰撞"),
        ("speed_limit_compliance", "限速合规"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    for axis, (metric, title) in zip(axes.flat, panels):
        for y, label, summary, color, marker in [
            (0, "完整261场", original, "#245D87", "o"),
            (1, "剔除后259场", trimmed, "#C1613A", "s"),
        ]:
            data = summary["metrics"][metric]
            point = data["mean_delta"] * 100
            lo, hi = [value * 100 for value in data["log_cluster_percentile_interval_95"]]
            axis.errorbar(
                point, y, xerr=[[point - lo], [hi - point]],
                fmt=marker, color=color, capsize=4, markersize=6, elinewidth=1.5,
            )
            axis.annotate(f"{point:+.2f}", (point, y), xytext=(0, -14),
                          textcoords="offset points", ha="center", color=color, fontsize=8)
        axis.axvline(0, color="#667482", linewidth=0.9, linestyle="--")
        axis.set_yticks([0, 1], ["完整261场", "剔除后259场"])
        axis.set_ylim(1.55, -0.55)
        axis.grid(axis="x", color="#D9E1E7", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlabel("相对 B10 的均值差 / 百分点")
        axis.set_title(title, fontsize=11, pad=12)
    fig.subplots_adjust(hspace=0.43, wspace=0.38)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"Test14_random剔除最差两场_配对敏感性.{suffix}",
                    dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads(INPUT.read_text())
    archived = json.loads(EXTENDED.read_text())
    baseline = validate_run(summary["historical_B10"])
    candidate = validate_run(summary["full306801_posttrain"])
    assert baseline.keys() == candidate.keys()
    keys = sorted(baseline)
    ranking = sorted(
        keys,
        key=lambda key: (candidate[key]["score"] - baseline[key]["score"], key),
    )
    worst_four = ranking[:4]
    excluded = set(worst_four[:2])
    retained = [key for key in keys if key not in excluded]
    original = summarize(keys, baseline, candidate)
    trimmed = summarize(retained, baseline, candidate)
    # Ensure this read-only reconstruction matches the previously archived full analysis.
    for metric in METRICS:
        assert abs(
            original["metrics"][metric]["mean_delta"]
            - archived["benchmarks"]["test14-random"]["metrics"][metric]["mean_delta"]
        ) < 1e-12
    assert original["scene_count"] == 261 and trimmed["scene_count"] == 259
    excluded_details = [
        {
            "token": key[1],
            "log_name": key[0],
            "scenario_type": baseline[key]["scenario_type"],
            "score_baseline": baseline[key]["score"],
            "score_candidate": candidate[key]["score"],
            "score_delta": candidate[key]["score"] - baseline[key]["score"],
            "no_collision_baseline": baseline[key]["no_ego_at_fault_collisions"],
            "no_collision_candidate": candidate[key]["no_ego_at_fault_collisions"],
        }
        for key in worst_four
    ]
    result = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m月%d日%H时%M分"),
        "analysis_type": "post_hoc_score_tail_sensitivity",
        "resamples": REPEATS,
        "seed": SEED,
        "worst_four": excluded_details,
        "excluded_tokens": [key[1] for key in worst_four[:2]],
        "original": original,
        "trimmed": trimmed,
        "source_sha256": {"random_summary": sha256(INPUT), "extended_analysis": sha256(EXTENDED)},
    }
    (OUTPUT / "Test14_random剔除最差两场敏感性.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    plot_comparison(original, trimmed)
    for metric in ["score", "ego_progress_along_expert_route", "no_ego_at_fault_collisions", "speed_limit_compliance"]:
        a, b = original["metrics"][metric], trimmed["metrics"][metric]
        print(metric, "original", a["mean_delta"], a["log_cluster_percentile_interval_95"],
              "trimmed", b["mean_delta"], b["log_cluster_percentile_interval_95"])
    print("excluded", result["excluded_tokens"])


if __name__ == "__main__":
    main()
