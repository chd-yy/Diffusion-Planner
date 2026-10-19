#!/usr/bin/env python3
"""Plot quality-space compression ablation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_quality_space_compression_1005/quality_space_compression_validation1000_v2.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
ARMS = ["B_da_eligibility", "D_quality_fixed_quota", "E_quality_pool_quota"]
LABELS = ["B：几何压缩\n固定配额", "D：质量压缩\n固定配额", "E：质量压缩\n池比例配额"]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = data["summary_by_arm"]
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
    x = np.arange(3)
    has_two = [summary[arm]["has_two"]["mean"] * 100 for arm in ARMS]
    active = [summary[arm]["active_at_0p01"]["mean"] * 100 for arm in ARMS]
    bars1 = axes[0].bar(x - 0.18, has_two, 0.36, label="至少2条DA合格", color="#72B7B2")
    bars2 = axes[0].bar(x + 0.18, active, 0.36, label=r"有效组（$D_s\geq0.01$）", color="#F58518")
    axes[0].axhline(40, color="#4A5568", linestyle="--", linewidth=0.8, label="有效组目标40%")
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("锚点比例（%）")
    axes[0].set_ylim(0, 70)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_xlabel("（a）安全覆盖与有效优势信号", labelpad=8)
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)

    base_ade = summary[ARMS[0]]["mean_candidate_ade_m"]["mean"]
    ade_delta = [
        (summary[arm]["mean_candidate_ade_m"]["mean"] / base_ade - 1) * 100
        for arm in ARMS
    ]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    for label, act, ade, color in zip(LABELS, active, ade_delta, colors):
        axes[1].scatter(act, ade, s=82, color=color, label=label.replace("\n", ""), zorder=3)
        axes[1].annotate(f"({act:.1f}%, {ade:+.2f}%)", (act, ade), xytext=(5, 7), textcoords="offset points", fontsize=8)
    axes[1].axvline(40, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[1].axhline(1, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[1].axhline(0, color="#A0AEC0", linewidth=0.7)
    axes[1].fill_between([40, 47], -0.5, 1, color="#C6F6D5", alpha=0.35, label="信号/质量可接受区")
    axes[1].set_xlim(31, 47)
    axes[1].set_ylim(-0.5, 1.8)
    axes[1].set_xlabel("（b）有效信号与平均ADE权衡", labelpad=8)
    axes[1].set_ylabel("平均候选ADE相对B变化（%）")
    axes[1].grid(alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"quality_space_compression_ablation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
