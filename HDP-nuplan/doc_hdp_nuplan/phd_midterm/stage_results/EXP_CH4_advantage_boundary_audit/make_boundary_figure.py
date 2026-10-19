#!/usr/bin/env python3
"""Plot boundary-group stability audit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_advantage_boundary_audit_1005/boundary_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main():
    data = json.loads(SOURCE.read_text())
    rows = data["boundary_scenes"]
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
        "pdf.fonttype": 42,
    })
    top_stable = np.mean([row["top1_stability"] >= 0.8 for row in rows]) * 100
    material = np.mean([row["material"] for row in rows]) * 100
    rank_stable = np.mean([row["mean_rank_spearman"] >= 0.9 for row in rows]) * 100
    all_three = np.mean([
        row["top1_stability"] >= 0.8 and row["material"] and row["mean_rank_spearman"] >= 0.9
        for row in rows
    ]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    labels = ["Top-1稳定", "物理差异", "完整排序稳定", "三项同时满足"]
    values = [top_stable, material, rank_stable, all_three]
    colors = ["#2B6CB0" if value >= 80 else "#C53030" for value in values]
    axes[0].bar(np.arange(4), values, color=colors, width=0.64)
    axes[0].axhline(80, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(np.arange(4), labels, rotation=20, ha="right")
    axes[0].set_ylabel("边界锚点比例（%）")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set_xlabel("（a）93个边界组的稳定性与物理差异", labelpad=8)
    for index, value in enumerate(values):
        axes[0].text(index, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    x = np.array([row["top1_stability"] for row in rows])
    y = np.array([row["mean_rank_spearman"] for row in rows])
    moving = np.array([row["moving_expert_gt_2m"] for row in rows])
    axes[1].scatter(x[~moving], y[~moving], s=24, alpha=0.72, label="静止/微动", color="#DD6B20")
    axes[1].scatter(x[moving], y[moving], s=24, alpha=0.72, label="运动锚点", color="#2B6CB0")
    axes[1].axvline(0.8, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.9, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_ylim(0.18, 1.02)
    axes[1].set_xlabel("（b）Top-1稳定性与完整排序相关", labelpad=8)
    axes[1].set_ylabel("平均Spearman相关")
    axes[1].grid(alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"advantage_boundary_audit.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
