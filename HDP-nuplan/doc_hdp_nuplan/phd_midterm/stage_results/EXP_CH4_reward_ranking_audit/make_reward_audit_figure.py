#!/usr/bin/env python3
"""Plot the frozen-candidate reward audit from the immutable raw JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_reward_ranking_audit_1005/reward_ranking_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=FONT).get_name(),
            "axes.unicode_minus": False,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )

    labels = {
        "legacy": "Legacy",
        "nuplan_aligned": "NuPlan对齐",
        "nuplan_score_proxy_v2": "Score代理v2",
        "nuplan_score_proxy_v3": "Score代理v3",
    }
    colors = {
        "legacy": "#718096",
        "nuplan_aligned": "#2B6CB0",
        "nuplan_score_proxy_v2": "#38A169",
        "nuplan_score_proxy_v3": "#DD6B20",
    }
    modes = list(labels)
    thresholds = np.array([0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.15), constrained_layout=True)
    for mode in modes:
        std = np.array(
            [scene["modes"][mode]["reward_std_eligible"] for scene in data["per_scene"]]
        )
        counts = np.array([scene["eligible_count"] for scene in data["per_scene"]])
        active = [np.mean((counts >= 2) & (std >= threshold)) * 100 for threshold in thresholds]
        axes[0].plot(
            thresholds,
            active,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=labels[mode],
            color=colors[mode],
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(thresholds)
    axes[0].set_xticklabels(
        ["0.00025", "0.0005", "0.001", "0.002", "0.005", "0.01"],
        rotation=24,
        ha="right",
    )
    axes[0].set_xlabel("（a）Reward标准差门槛与有效组比例", labelpad=8)
    axes[0].set_ylabel("有效锚点比例（%）")
    axes[0].grid(alpha=0.24, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].axvline(0.01, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[0].text(0.0092, 59, "原门槛0.01", rotation=90, va="top", ha="right", fontsize=8)

    metrics = [
        ("progress_guard_reward", "进度保护"),
        ("risk_reward", "风险质量"),
        ("route_cost", "路线质量"),
        ("comfort_cost", "舒适质量"),
        ("drivable_area_compliance", "DA合规"),
    ]
    plot_modes = modes[1:]
    x = np.arange(len(metrics))
    width = 0.24
    for mode_index, mode in enumerate(plot_modes):
        paired = data["paired_top_candidate_vs_legacy"][mode]
        values = []
        for metric, _ in metrics:
            delta = paired[metric]["mean"]
            if metric in {"route_cost", "comfort_cost"}:
                delta = -delta
            values.append(delta * 1000)
        axes[1].bar(
            x + (mode_index - 1) * width,
            values,
            width=width,
            color=colors[mode],
            label=labels[mode],
        )
    axes[1].axhline(0, color="#2D3748", linewidth=0.8)
    axes[1].set_xticks(x, [label for _, label in metrics], rotation=22, ha="right")
    axes[1].set_ylabel("相对Legacy的有利方向差值（x10^-3）")
    axes[1].set_xlabel("（b）各目标选出的安全Top-1候选变化", labelpad=8)
    axes[1].grid(axis="y", alpha=0.24, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"reward_ranking_audit.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
