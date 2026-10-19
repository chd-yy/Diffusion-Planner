#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "minimal40_result.json").read_text(encoding="utf-8"))
FONT_PATH = "/home/yanjun/NewDisk/TinyTeX/.TinyTeX/texmf-dist/fonts/truetype/public/lxgw-fonts/LXGWNeoZhiSong.ttf"
FONT = FontProperties(fname=FONT_PATH)


def label_bars(axis, bars, digits=3):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontproperties=FONT,
            fontsize=8,
        )


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


summary = DATA["summary"]
baseline = summary["baseline"]
proposed = summary["guided_safe_set"]
scene_meta = {
    row["name"]: row
    for row in DATA["protocol"]["motion_stratification"]["selected"]
}

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
colors = ["#8A9BA8", "#2F78B7"]

# (a) Overall geometric support.
labels = ["终点距离", "整轨迹距离"]
baseline_values = [
    baseline["endpoint_pairwise_distance_m"]["p50"],
    baseline["trajectory_pairwise_distance_m"]["p50"],
]
proposed_values = [
    proposed["endpoint_pairwise_distance_m"]["p50"],
    proposed["trajectory_pairwise_distance_m"]["p50"],
]
x = [0, 1]
width = 0.34
bars_a = axes[0].bar([v - width / 2 for v in x], baseline_values, width, color=colors[0], label="原始B10")
bars_b = axes[0].bar([v + width / 2 for v in x], proposed_values, width, color=colors[1], label="两层修复")
axes[0].set_xticks(x, labels, fontproperties=FONT)
axes[0].set_ylabel("两两距离P50 / m", fontproperties=FONT)
axes[0].set_ylim(0, 0.27)
axes[0].legend(prop=FONT, frameon=False, loc="upper right")
label_bars(axes[0], bars_a)
label_bars(axes[0], bars_b)

# (b) Motion-stratified trajectory diversity.
strata = ["high_motion", "medium_motion", "low_motion"]
stratum_labels = ["高运动", "中运动", "低运动"]
per_scene = DATA["per_scene"]
baseline_by_stratum = []
proposed_by_stratum = []
for stratum in strata:
    names = {name for name, meta in scene_meta.items() if meta["motion_stratum"] == stratum}
    baseline_by_stratum.append(median([
        row["trajectory_pairwise_distance_m"]
        for row in per_scene["baseline"]
        if row["scene"] in names
    ]))
    proposed_by_stratum.append(median([
        row["trajectory_pairwise_distance_m"]
        for row in per_scene["guided_safe_set"]
        if row["scene"] in names
    ]))
x = list(range(3))
bars_a = axes[1].bar([v - width / 2 for v in x], baseline_by_stratum, width, color=colors[0])
bars_b = axes[1].bar([v + width / 2 for v in x], proposed_by_stratum, width, color=colors[1])
axes[1].axhline(0.10, color="#B23A48", linestyle="--", linewidth=1.2)
axes[1].text(2.45, 0.102, "绝对阈值", color="#B23A48", ha="right", va="bottom", fontproperties=FONT, fontsize=8)
axes[1].set_xticks(x, stratum_labels, fontproperties=FONT)
axes[1].set_ylabel("整轨迹两两距离P50 / m", fontproperties=FONT)
axes[1].set_ylim(0, 0.13)
label_bars(axes[1], bars_a)
label_bars(axes[1], bars_b)

# (c) Motion-stratified reward separability.
baseline_active = []
proposed_active = []
for stratum in strata:
    names = {name for name, meta in scene_meta.items() if meta["motion_stratum"] == stratum}
    baseline_rows = [row for row in per_scene["baseline"] if row["scene"] in names]
    proposed_rows = [row for row in per_scene["guided_safe_set"] if row["scene"] in names]
    baseline_active.append(sum(row["active_safe_reward_std_0_01"] for row in baseline_rows) / len(baseline_rows))
    proposed_active.append(sum(row["active_safe_reward_std_0_01"] for row in proposed_rows) / len(proposed_rows))
x = list(range(3))
bars_a = axes[2].bar([v - width / 2 for v in x], baseline_active, width, color=colors[0])
bars_b = axes[2].bar([v + width / 2 for v in x], proposed_active, width, color=colors[1])
axes[2].axhline(0.30, color="#B23A48", linestyle="--", linewidth=1.2)
axes[2].text(2.45, 0.305, "验收阈值", color="#B23A48", ha="right", va="bottom", fontproperties=FONT, fontsize=8)
axes[2].set_xticks(x, stratum_labels, fontproperties=FONT)
axes[2].set_ylabel("reward std≥0.01场景比例", fontproperties=FONT)
axes[2].set_ylim(0, 0.48)
label_bars(axes[2], bars_a, digits=2)
label_bars(axes[2], bars_b, digits=2)

for axis, title in zip(
    axes,
    ["（a）总体几何区分度", "（b）分运动量几何区分度", "（c）分运动量Reward可分性"],
):
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.5,
        -0.25,
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
fig.subplots_adjust(bottom=0.24)
fig.savefig(ROOT / "minimal40_pretraining_gate_v2.png", dpi=300, bbox_inches="tight")
plt.close(fig)
