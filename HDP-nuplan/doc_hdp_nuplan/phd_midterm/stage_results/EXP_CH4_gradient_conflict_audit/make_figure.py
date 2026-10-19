#!/usr/bin/env python3
"""Plot frozen Chapter-4 gradient-conflict audit results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_gradient_conflict_64_1005.json"
)


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = [row for row in data["records"] if row["has_rollout_target"]]
    song = FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc")
    plt.rcParams["axes.unicode_minus"] = False

    norms = [
        [row["rl_grad_norm"] for row in records],
        [row["weighted_expert_grad_norm"] for row in records],
        [row["weighted_reference_grad_norm"] for row in records],
    ]
    cosines = [
        [row["cos_rl_expert"] for row in records],
        [row["cos_rl_reference"] for row in records],
        [row["cos_expert_reference"] for row in records],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    boxes = axes[0].boxplot(norms, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], ["#4C78A8", "#F58518", "#72B7B2"]):
        patch.set_facecolor(color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Decoder梯度范数（对数刻度）", fontproperties=song)
    axes[0].set_xticks(
        [1, 2, 3], ["Rollout", "0.1×Expert", "0.1×Reference"],
        fontproperties=song,
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(
        0.5, -0.19, "（a）三类训练目标的有效梯度尺度",
        transform=axes[0].transAxes, ha="center", fontproperties=song,
    )

    boxes = axes[1].boxplot(cosines, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], ["#E45756", "#4C78A8", "#B279A2"]):
        patch.set_facecolor(color)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("梯度余弦相似度", fontproperties=song)
    axes[1].set_xticks(
        [1, 2, 3], ["RL-Expert", "RL-Reference", "Expert-Reference"],
        rotation=16, ha="right", fontproperties=song,
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.5, -0.19, "（b）目标之间的局部方向一致性",
        transform=axes[1].transAxes, ha="center", fontproperties=song,
    )

    figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.25, wspace=0.28)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "gradient_conflict_audit.png", dpi=300)
    figure.savefig(output / "gradient_conflict_audit.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
