#!/usr/bin/env python3
"""Generate thesis-ready figures from the frozen paired safety40 result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/"
    "chapter4_closed_loop/EXP_CH4_B10_Repair100_safety40_1004_v2/paired_results.json"
)
OUT = ROOT / "figures"
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT.read_text())
    comparison = data["comparison_Repair100_minus_B10"]
    metrics = comparison["metrics"]

    # AR PL UMing CN is the installed Chinese Song/Ming typeface used for all
    # Chinese text. Latin and numeric text deliberately use the same family.
    font_manager.fontManager.addfont(FONT)
    family = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    labels = ["综合分", "路线进度", "责任碰撞", "可行驶区", "TTC", "舒适性", "限速合规"]
    keys = [
        "score",
        "ego_progress_along_expert_route",
        "no_ego_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_is_comfortable",
        "speed_limit_compliance",
    ]
    values = np.array([metrics[key]["mean_delta"] * 100 for key in keys])
    colors = ["#2B6CB0" if value > 0 else "#C53030" if value < 0 else "#A0AEC0" for value in values]
    x = np.arange(len(labels))
    axes[0].bar(x, values, color=colors, width=0.66)
    axes[0].axhline(0, color="#2D3748", linewidth=0.8)
    axes[0].set_ylabel("Repair-100 - B10（百分点）")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    for i, value in enumerate(values):
        offset = 0.055 if value >= 0 else -0.055
        va = "bottom" if value >= 0 else "top"
        axes[0].text(i, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=8.5)
    axes[0].set_xlabel("（a）固定safety40上的总体指标变化", labelpad=8)

    b10 = json.loads((INPUT.parent / "B10.summary.json").read_text())
    repair = json.loads((INPUT.parent / "Repair100.summary.json").read_text())
    b10_by_token = {row["scenario"]: row for row in b10["scenarios"]}
    deltas = []
    tokens = []
    for row in repair["scenarios"]:
        token = row["scenario"]
        deltas.append(row["score"] - b10_by_token[token]["score"])
        tokens.append(token)
    order = np.argsort(deltas)
    sorted_deltas = np.asarray(deltas)[order]
    colors = np.where(sorted_deltas > 1e-12, "#2B6CB0", np.where(sorted_deltas < -1e-12, "#C53030", "#CBD5E0"))
    axes[1].bar(np.arange(len(sorted_deltas)), sorted_deltas, color=colors, width=0.78)
    axes[1].axhline(0, color="#2D3748", linewidth=0.8)
    axes[1].set_ylabel("逐场综合分差值")
    axes[1].set_xticks([])
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    max_index = int(np.argmax(sorted_deltas))
    axes[1].annotate(
        "单场舒适性由0变为1\n综合分 +0.125",
        xy=(max_index, sorted_deltas[max_index]),
        xytext=(max_index - 13, sorted_deltas[max_index] * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#4A5568", "linewidth": 0.8},
        ha="center",
        fontsize=8.5,
    )
    axes[1].text(
        0.02,
        0.96,
        "16场提高 / 3场下降 / 21场持平",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    axes[1].set_xlabel("（b）40个成对场景的综合分差值分布", labelpad=8)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"repair100_safety40_closed_loop.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    figure_data = {
        "source": str(INPUT),
        "delta_definition": "Repair100 minus B10",
        "overall_metric_delta_percentage_points": dict(zip(keys, values.tolist())),
        "scenario_score_deltas_sorted": sorted_deltas.tolist(),
        "scenario_tokens_sorted": np.asarray(tokens)[order].tolist(),
    }
    (OUT / "repair100_safety40_closed_loop.json").write_text(
        json.dumps(figure_data, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
