#!/usr/bin/env python3
"""Plot task-preserving advantage-mask audit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_advantage_update_mask_1005/advantage_update_mask_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    a = data["advantage_update_mask_audit"]["aggregate"]
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    values = [
        a["active_group_count"],
        a["retained_active_group_count"],
        185,
    ]
    labels = ["F25有效组", "优势屏蔽后\n仍可更新", "硬资格G\n有效组"]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    bars = axes[0].bar(labels, values, color=colors, width=0.62)
    axes[0].axhline(0.9 * a["active_group_count"], color="#4A5568", linestyle="--", linewidth=0.8, label="90%组保留门槛")
    axes[0].set_ylabel("validation-1000锚点数")
    axes[0].set_ylim(0, 470)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlabel("（a）优势层屏蔽保留了多数有效组", labelpad=8)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+8, f"{value}", ha="center", fontsize=9)

    mass_values = [
        a["positive_mass_retention_rate"] * 100,
        a["route_violation_mass_fraction"] * 100,
        a["progress_violation_mass_fraction"] * 100,
    ]
    mass_labels = ["合规质量\n保留", "Route违规\n质量", "Progress违规\n质量"]
    mass_colors = ["#54A24B", "#F58518", "#E45756"]
    bars = axes[1].bar(mass_labels, mass_values, color=mass_colors, width=0.62)
    axes[1].axhline(50, color="#4A5568", linestyle="--", linewidth=0.8, label="正优势质量保留门槛")
    axes[1].set_ylabel("占原始正优势质量比例（%）")
    axes[1].set_ylim(0, 62)
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("（b）多数高优势质量违反任务保持条件", labelpad=8)
    for bar, value in zip(bars, mass_values):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+1.2, f"{value:.1f}%", ha="center", fontsize=9)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"advantage_update_mask.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
