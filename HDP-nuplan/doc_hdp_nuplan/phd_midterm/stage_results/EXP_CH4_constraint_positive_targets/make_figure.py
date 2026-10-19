#!/usr/bin/env python3
"""Plot constrained positive-target margin sweep."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_constraint_positive_targets_1005/constraint_positive_targets_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
KEYS = ["margin_0", "margin_0p1", "margin_0p25", "margin_0p5"]
LABELS = ["0", "0.1", "0.25", "0.5"]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    sweep = data["constraint_positive_target_sweep"]
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
        "pdf.fonttype": 42,
    })

    retention = np.array([sweep[k]["active_group_retention_rate"] * 100 for k in KEYS])
    total = np.array([sweep[k]["total_target_count"] for k in KEYS])
    mean_targets = np.array([sweep[k]["target_count_among_retained"]["mean"] for k in KEYS])
    x = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    colors = ["#4C78A8", "#2F855A", "#F58518", "#E45756"]
    bars = axes[0].bar(x, retention, color=colors, width=0.62)
    axes[0].axhline(90, color="#4A5568", linestyle="--", linewidth=0.8, label="有效组保留门槛90%")
    axes[0].set_xticks(x, LABELS)
    axes[0].set_xlabel("（a）标准化优势间隔与有效组保留", labelpad=8)
    axes[0].set_ylabel("F25有效组保留率（%）")
    axes[0].set_ylim(50, 100)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    for bar, value in zip(bars, retention):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+0.8, f"{value:.1f}%", ha="center", fontsize=9)

    bars = axes[1].bar(x, total, color=colors, width=0.62, label="合规目标总数")
    ax2 = axes[1].twinx()
    ax2.plot(x, mean_targets, color="#805AD5", marker="o", linewidth=1.4, label="保留组平均目标数")
    axes[1].set_xticks(x, LABELS)
    axes[1].set_xlabel("（b）优势间隔与更新目标规模", labelpad=8)
    axes[1].set_ylabel("任务合规目标总数")
    ax2.set_ylabel("每个保留组的平均目标数")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1+handles2, labels1+labels2, frameon=False, fontsize=9, loc="upper right")
    for bar, value in zip(bars, total):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+35, f"{value}", ha="center", fontsize=9)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"constraint_positive_targets.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
