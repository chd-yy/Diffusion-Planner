#!/usr/bin/env python3
"""Create the thesis figure for the aligned-reward DA-gate audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_aligned_da_gate_audit_1005/aligned_da_gate_validation1000_v2.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)

    coverage = data["coverage_summary"]
    labels = ["至少1条合格候选", "至少2条合格候选", "质量空间有效组"]
    legacy = [
        coverage["legacy"]["has_eligible"]["mean"] * 100,
        coverage["legacy"]["has_two"]["mean"] * 100,
        np.nan,
    ]
    aligned_da = [
        coverage["aligned_da"]["has_eligible"]["mean"] * 100,
        coverage["aligned_da"]["has_two"]["mean"] * 100,
        data["component_active_fraction"] * 100,
    ]
    x = np.arange(3)
    axes[0].bar(x - 0.18, legacy, width=0.36, label="Legacy", color="#718096")
    axes[0].bar(x + 0.18, aligned_da, width=0.36, label="对齐Reward+DA门", color="#2B6CB0")
    axes[0].axhline(40, linestyle="--", linewidth=0.8, color="#C53030")
    axes[0].text(2.42, 40.8, "有效组门槛40%", ha="right", color="#C53030", fontsize=8.5)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("锚点比例（%）")
    axes[0].set_ylim(0, 74)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlabel("（a）安全候选覆盖与尺度无关有效组", labelpad=8)

    paired = data["paired_common_eligible"]["aligned_da_minus_legacy"]
    metric_specs = [
        ("progress_guard_reward", "进度保护", 1),
        ("risk_reward", "风险质量", 1),
        ("route_cost", "路线质量", -1),
        ("comfort_cost", "舒适质量", -1),
        ("raw_drivable_area_compliance", "DA合规", 1),
    ]
    values = [paired[key]["mean"] * direction * 1000 for key, _, direction in metric_specs]
    colors = ["#2B6CB0" if value >= 0 else "#C53030" for value in values]
    axes[1].bar(np.arange(len(values)), values, color=colors, width=0.64)
    axes[1].axhline(0, color="#2D3748", linewidth=0.8)
    axes[1].set_xticks(
        np.arange(len(values)),
        [label for _, label, _ in metric_specs],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylabel("相对Legacy的有利方向差值（x10^-3）")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].set_xlabel("（b）共同可评锚点上的Top-1质量变化", labelpad=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"aligned_da_gate_audit.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
