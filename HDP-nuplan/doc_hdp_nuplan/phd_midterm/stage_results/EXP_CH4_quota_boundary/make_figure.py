#!/usr/bin/env python3
"""Plot the noise-scale quota boundary experiment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_quota_boundary_1005/quota_boundary_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
ARMS = [
    "D_quality_fixed_quota",
    "F25_quality_interpolated_quota",
    "F50_quality_interpolated_quota",
    "F75_quality_interpolated_quota",
    "E_quality_pool_quota",
]
LABELS = ["D", "F25", "F50", "F75", "E"]
HIGH_SLOTS = [5, 8, 10, 12, 15]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = data["summary_by_arm"]
    base = summary["B_da_eligibility"]
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
        "pdf.fonttype": 42,
    })

    active = np.array([summary[a]["active_at_0p01"]["mean"] * 100 for a in ARMS])
    ade = np.array([
        (summary[a]["mean_candidate_ade_m"]["mean"] / base["mean_candidate_ade_m"]["mean"] - 1) * 100
        for a in ARMS
    ])
    route = np.array([
        (summary[a]["route_cost_top"]["mean"] - base["route_cost_top"]["mean"]) * 1e4
        for a in ARMS
    ])
    no_collision = np.array([
        (summary[a]["no_collision_top"]["mean"] - base["no_collision_top"]["mean"]) * 100
        for a in ARMS
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    axes[0].fill_between([39.9, 47], -0.4, 1, color="#C6F6D5", alpha=0.42, label="信号/质量可接受区")
    axes[0].axvline(40, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[0].axhline(1, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[0].plot(active, ade, color="#4C78A8", linewidth=1.2, alpha=0.7)
    for label, x, y, slots in zip(LABELS, active, ade, HIGH_SLOTS):
        color = "#2F855A" if label == "F25" else "#4C78A8"
        axes[0].scatter(x, y, s=75, color=color, zorder=3)
        axes[0].annotate(f"{label}（{slots}席）", (x, y), xytext=(5, 6), textcoords="offset points", fontsize=8)
    axes[0].set_xlim(32, 47)
    axes[0].set_ylim(-0.4, 1.65)
    axes[0].set_ylabel("平均候选ADE相对B变化（%）")
    axes[0].set_xlabel("（a）有效组与模仿质量的配额边界", labelpad=8)
    axes[0].grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    x = np.arange(len(ARMS))
    bars_route = axes[1].bar(x - 0.18, route, 0.36, label=r"Route cost差值（$\times10^{-4}$）", color="#F58518")
    bars_collision = axes[1].bar(x + 0.18, no_collision, 0.36, label="No-collision差值（百分点）", color="#54A24B")
    axes[1].axhline(1, color="#F58518", linestyle="--", linewidth=0.8, label="Route容差上限")
    axes[1].axhline(0, color="#4A5568", linewidth=0.8)
    axes[1].set_xticks(x, LABELS)
    axes[1].set_ylabel("相对B的变化")
    axes[1].set_ylim(-2.6, 3.4)
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].set_xlabel("（b）Route与无碰撞门禁", labelpad=8)
    for bars in (bars_route, bars_collision):
        for bar in bars:
            value = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                value + (0.12 if value >= 0 else -0.25),
                f"{value:+.2f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8,
            )

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"quota_boundary.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
