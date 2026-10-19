"""Figures for hard-full and random-trimmed full-post-training analysis.

Reads archived results only. It does not run simulation or alter the original
Test14 summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
STAGE = PROJECT / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_full306801_posttraining"
FIGURES = STAGE / "figures"
EXTERNAL = Path("/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP")
TEST14 = EXTERNAL / "test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006"
BLUE = "#215D88"
TEAL = "#138779"
RED = "#B94E45"
GREY = "#667482"
LIGHT = "#D9E1E7"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "AR PL SungtiL GB",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_paired_effect(hard: dict, random_trim: dict) -> None:
    panels = [
        ("score", "综合得分"),
        ("ego_progress_along_expert_route", "路线进度"),
        ("no_ego_at_fault_collisions", "无责任碰撞"),
        ("speed_limit_compliance", "限速合规"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9))
    for axis, (metric, title) in zip(axes.flat, panels):
        for y, label, data, color, marker in [
            (0, "Hard 完整272场", hard, BLUE, "o"),
            (1, "Random 剔除后259场", random_trim, TEAL, "s"),
        ]:
            item = data["metrics"][metric]
            point = 100 * item["mean_delta"]
            lo, hi = [100 * value for value in item["log_cluster_percentile_interval_95"]]
            axis.errorbar(
                point, y, xerr=[[point - lo], [hi - point]],
                fmt=marker, color=color, capsize=4, elinewidth=1.6, markersize=6,
            )
            axis.annotate(
                f"{point:+.2f}", (point, y), xytext=(0, -14),
                textcoords="offset points", ha="center", color=color, fontsize=8,
            )
        axis.axvline(0, color=GREY, linestyle="--", linewidth=1)
        axis.set_yticks([0, 1], ["Hard 272场", "Random 259场*"])
        axis.set_ylim(1.55, -0.55)
        axis.set_xlabel("相对 B10 的均值差 / 百分点")
        axis.set_title(title, fontsize=11, pad=12)
        axis.grid(axis="x", color=LIGHT, linewidth=0.6)
        axis.set_axisbelow(True)
    fig.subplots_adjust(hspace=0.47, wspace=0.42, bottom=0.14)
    fig.text(0.5, 0.04, "* Random 事后剔除两场；区间仅描述固定剔除名单下的259场。", ha="center", fontsize=9)
    save(fig, "全量后训练_统一分析_闭环配对效应")


def plot_score_tails(excluded_tokens: set[str]) -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.7), sharey=True)
    counts = {}
    for axis, (benchmark, label) in zip(
        axes,
        [("test14-hard", "Hard 完整272场"), ("test14-random", "Random 剔除后259场")],
    ):
        source = json.loads((TEST14 / f"{benchmark}_historical_B10_vs_full306801.json").read_text())
        base = {row["scenario"]: row for row in source["historical_B10"]["scenarios"]}
        cand = {row["scenario"]: row for row in source["full306801_posttrain"]["scenarios"]}
        assert base.keys() == cand.keys()
        rows = []
        for token, b in base.items():
            if benchmark == "test14-random" and token in excluded_tokens:
                continue
            c = cand[token]
            rows.append(
                (c["score"] - b["score"],
                 b["no_ego_at_fault_collisions"] == 1 and c["no_ego_at_fault_collisions"] == 0,
                 token)
            )
        rows.sort(key=lambda item: (item[0], item[2]))
        x = np.arange(len(rows))
        delta = np.array([row[0] for row in rows])
        new_collision = np.array([row[1] for row in rows])
        axis.scatter(x, delta, s=11, color=BLUE, alpha=0.75, rasterized=True)
        axis.scatter(
            x[new_collision], delta[new_collision], s=36, color=RED,
            marker="x", linewidth=1.3, label="新增责任碰撞",
        )
        axis.axhline(0, color=GREY, linewidth=0.9, linestyle="--")
        axis.set_title(label, fontsize=11)
        axis.set_ylabel("逐场综合得分差")
        axis.set_xlabel("按得分差排序的场景")
        axis.grid(axis="y", color=LIGHT, linewidth=0.5)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        counts[benchmark] = {
            "scenes": len(rows),
            "wins": int((delta > 1e-6).sum()),
            "losses": int((delta < -1e-6).sum()),
            "ties": int((abs(delta) <= 1e-6).sum()),
            "new_collision_events": int(new_collision.sum()),
        }
    fig.subplots_adjust(hspace=0.5)
    save(fig, "全量后训练_统一分析_逐场得分尾部")
    return counts


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    original = json.loads((FIGURES / "全量后训练_图表数据.json").read_text())
    sensitivity = json.loads((STAGE / "Test14_random剔除最差两场敏感性.json").read_text())
    hard = original["closed_loop"]["test14-hard"]
    random_trim = sensitivity["trimmed"]
    assert hard["scenario_count"] == 272 and random_trim["scene_count"] == 259
    plot_paired_effect(hard, random_trim)
    counts = plot_score_tails(set(sensitivity["excluded_tokens"]))
    assert counts["test14-hard"]["new_collision_events"] == 6
    assert counts["test14-random"]["new_collision_events"] == 3
    (FIGURES / "全量后训练_统一分析_图表核对.json").write_text(
        json.dumps(
            {
                "hard_full": counts["test14-hard"],
                "random_trimmed": counts["test14-random"],
                "excluded_random_tokens": sensitivity["excluded_tokens"],
                "figure_data_sources": [
                    "全量后训练_图表数据.json",
                    "../Test14_random剔除最差两场敏感性.json",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
