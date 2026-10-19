#!/usr/bin/env python3
"""Render the constraint-aware 20-step smoke result from frozen JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
VALIDATION = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/simulation_outputs/"
    "training_outputs/EXP_CH4_constraint_m01_smoke1000_1005_v3/"
    "validation1000_3repeat.json"
)
AUDIT = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_constraint_smoke_training_manifest_audit_1005.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main() -> None:
    from matplotlib.font_manager import FontProperties

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    song = FontProperties(fname=FONT)
    plt.rcParams["axes.unicode_minus"] = False

    sweep = audit["constraint_positive_target_sweep"]["margin_0p1"]
    training_labels = ["质量有效组", "保留目标组", "验证批次有目标"]
    training_values = [
        sweep["active_group_count"] / 10.0,
        sweep["retained_group_count"] / 10.0,
        45.0,
    ]

    keys = ["reward", "progress", "collision_cost", "route_cost", "comfort_cost"]
    metric_labels = ["对齐奖励", "路线进度", "碰撞代价", "路线代价", "舒适代价"]
    deltas = np.asarray(
        [validation["paired_delta_rl_minus_supervised"][key]["mean"] for key in keys]
    ) * 1000.0

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bars = axes[0].bar(
        np.arange(3), training_values, color=["#4C78A8", "#72B7B2", "#F58518"]
    )
    axes[0].set_ylabel("场景/批次比例（%）", fontproperties=song)
    axes[0].set_xticks(np.arange(3), training_labels, fontproperties=song)
    axes[0].set_ylim(0, 50)
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, training_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )
    axes[0].text(
        0.5, -0.22, "（a）训练集目标覆盖与实际更新信号",
        transform=axes[0].transAxes, ha="center", fontproperties=song,
    )

    colors = ["#54A24B" if value < 0 and key.endswith("cost") else "#4C78A8"
              for key, value in zip(keys, deltas)]
    colors[1] = "#E45756" if deltas[1] < 0 else "#54A24B"
    colors[0] = "#54A24B" if deltas[0] > 0 else "#E45756"
    bars = axes[1].bar(np.arange(len(deltas)), deltas, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("RL-B10均值差 (x10^-3)", fontproperties=song)
    axes[1].set_xticks(
        np.arange(len(deltas)), metric_labels, rotation=25, ha="right",
        fontproperties=song,
    )
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, deltas):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.10 if value >= 0 else -0.18),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    axes[1].text(
        0.5, -0.22, "（b）validation-1000×3配对变化",
        transform=axes[1].transAxes, ha="center", fontproperties=song,
    )

    figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.28, wspace=0.28)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "constraint_smoke_result.png", dpi=300)
    figure.savefig(output / "constraint_smoke_result.pdf")
    plt.close(figure)

    summary = {
        "training_manifest_audit": {
            "f25_active_group_fraction": sweep["active_group_count"] / 1000.0,
            "margin_0p1_retained_group_fraction": sweep["retained_group_count"] / 1000.0,
            "active_group_retention_rate": sweep["active_group_retention_rate"],
            "mean_targets_among_retained": sweep["target_count_among_retained"]["mean"],
        },
        "validation_delta_rl_minus_b10": {
            key: validation["paired_delta_rl_minus_supervised"][key]
            for key in keys + ["no_collision", "ade_m", "fde_m"]
        },
    }
    (output / "constraint_smoke_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
