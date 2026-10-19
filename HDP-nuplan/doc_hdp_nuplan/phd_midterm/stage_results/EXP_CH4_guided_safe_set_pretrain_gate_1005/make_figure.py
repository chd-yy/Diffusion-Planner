#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "minimal40_result.json").read_text(encoding="utf-8"))
FONT_PATH = "/home/yanjun/NewDisk/TinyTeX/.TinyTeX/texmf-dist/fonts/truetype/public/lxgw-fonts/LXGWNeoZhiSong.ttf"
FONT = FontProperties(fname=FONT_PATH)


def label_bars(axis, bars, suffix=""):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}{suffix}",
            ha="center",
            va="bottom",
            fontproperties=FONT,
            fontsize=9,
        )


summary = DATA["summary"]
baseline = summary["baseline"]
proposed = summary["guided_safe_set"]
readiness = summary["training_readiness"]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
colors = ["#8A9BA8", "#2F78B7"]

endpoint = [
    baseline["endpoint_pairwise_distance_m"]["p50"],
    proposed["endpoint_pairwise_distance_m"]["p50"],
]
bars = axes[0].bar([0, 1], endpoint, color=colors, width=0.58)
axes[0].axhline(0.25, color="#B23A48", linestyle="--", linewidth=1.2)
axes[0].text(0.98, 0.252, "预注册阈值", color="#B23A48", ha="right", va="bottom", fontproperties=FONT, fontsize=8)
axes[0].set_ylim(0, 0.27)
axes[0].set_xticks([0, 1], ["原始32候选", "安全集合方案"], fontproperties=FONT)
axes[0].set_ylabel("终点两两距离 P50 / m", fontproperties=FONT)
label_bars(axes[0], bars)

trajectory = [
    baseline["trajectory_pairwise_distance_m"]["p50"],
    proposed["trajectory_pairwise_distance_m"]["p50"],
]
bars = axes[1].bar([0, 1], trajectory, color=colors, width=0.58)
axes[1].axhline(0.10, color="#B23A48", linestyle="--", linewidth=1.2)
axes[1].text(0.98, 0.101, "预注册阈值", color="#B23A48", ha="right", va="bottom", fontproperties=FONT, fontsize=8)
axes[1].set_ylim(0, 0.11)
axes[1].set_xticks([0, 1], ["原始32候选", "安全集合方案"], fontproperties=FONT)
axes[1].set_ylabel("整轨迹两两距离 P50 / m", fontproperties=FONT)
label_bars(axes[1], bars)

fractions = [
    baseline["active_safe_reward_std_0_01"]["mean"],
    proposed["active_safe_reward_std_0_01"]["mean"],
    readiness["active_quality_group"]["mean"],
    readiness["has_update_target"]["mean"],
]
bars = axes[2].bar(range(4), fractions, color=[colors[0], colors[1], "#5B9E6F", "#C78732"], width=0.62)
axes[2].axhline(0.30, color="#B23A48", linestyle="--", linewidth=1.2)
axes[2].text(2.95, 0.302, "奖励有效组阈值", color="#B23A48", ha="right", va="bottom", fontproperties=FONT, fontsize=8)
axes[2].set_xticks(
    range(4),
    ["原始奖励\n有效组", "新方案奖励\n有效组", "质量门\n激活组", "具有更新\n目标组"],
    fontproperties=FONT,
)
axes[2].set_ylabel("场景比例", fontproperties=FONT)
axes[2].set_ylim(0, 0.34)
label_bars(axes[2], bars)

for index, (axis, title) in enumerate(zip(
    axes,
    ["（a）终点几何差异", "（b）整轨迹几何差异", "（c）有效训练信号覆盖率"],
)):
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.5,
        -0.27,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontproperties=FONT,
        fontsize=11,
    )
    for tick in axis.get_yticklabels():
        tick.set_fontproperties(FONT)

fig.tight_layout(w_pad=2.1)
fig.subplots_adjust(bottom=0.25)
fig.savefig(ROOT / "minimal40_pretraining_gate.png", dpi=300, bbox_inches="tight")
plt.close(fig)
