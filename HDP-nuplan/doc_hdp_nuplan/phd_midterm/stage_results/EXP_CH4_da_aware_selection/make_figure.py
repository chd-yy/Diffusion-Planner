#!/usr/bin/env python3
"""Plot the DA-aware candidate compression ablation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_da_aware_selection_1005/da_aware_selection_validation1000_v2.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
ARMS = [
    "pool",
    "A_current_repair",
    "B_da_eligibility",
    "C_da_eligibility_aligned_reward",
]
ARM_LABELS = ["200条搜索池", "A：当前压缩", "B：DA资格前移", "C：DA前移+对齐排序"]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    summaries = {"pool": data["pool_summary"], **data["summary_by_arm"]}
    comparisons = data["paired_vs_current_repair"]
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
    x = np.arange(4)
    width = 0.24
    metrics = [
        ("has_eligible", "至少1条合格候选", "#4C78A8"),
        ("has_two", "至少2条合格候选", "#72B7B2"),
        ("active_at_0p01", r"有效组（$D_s\geq0.01$）", "#F58518"),
    ]
    for index, (key, label, color) in enumerate(metrics):
        values = [summaries[arm][key]["mean"] * 100 for arm in ARMS]
        bars = axes[0].bar(x + (index - 1) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, value + 0.8,
                f"{value:.1f}", ha="center", va="bottom", fontsize=8,
            )
    axes[0].axhline(40, color="#4A5568", linestyle="--", linewidth=0.8, label="有效组目标40%")
    axes[0].set_xticks(x, ARM_LABELS, rotation=12, ha="right")
    axes[0].set_ylabel("锚点比例（%）")
    axes[0].set_ylim(0, 73)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_xlabel("（a）压缩后安全覆盖与有效优势信号", labelpad=8)

    keys = ["has_eligible", "has_two", "active_at_0p01", "raw_da_top", "no_collision_top"]
    labels = ["至少1条\n合格", "至少2条\n合格", "有效组", "Top-1\nDA合规", "Top-1\n无碰撞"]
    b = comparisons["B_da_eligibility_minus_A_current_repair"]
    c = comparisons["C_da_eligibility_aligned_reward_minus_A_current_repair"]
    bv = [b[key]["mean"] * 1000 for key in keys]
    cv = [c[key]["mean"] * 1000 for key in keys]
    xx = np.arange(len(keys))
    bars_b = axes[1].bar(xx - 0.18, bv, 0.36, label="B相对A", color="#4C78A8")
    bars_c = axes[1].bar(xx + 0.18, cv, 0.36, label="C相对A", color="#E45756")
    axes[1].axhline(0, color="#4A5568", linewidth=0.8)
    axes[1].set_xticks(xx, labels)
    axes[1].set_ylabel("1000个锚点中的净变化数")
    axes[1].set_ylim(-4, 12)
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("（b）相对当前压缩方案的净增减", labelpad=8)
    for bars in (bars_b, bars_c):
        for bar in bars:
            value = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.35 if value >= 0 else -0.55),
                f"{value:+.0f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8,
            )

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"da_aware_selection_ablation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
