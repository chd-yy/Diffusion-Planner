#!/usr/bin/env python3
"""Plot task-preserving hard-gate ablation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP/reward_audit/"
    "EXP_CH4_task_preserving_gate_1005/task_preserving_gate_validation1000.json"
)
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    s = data["summary_by_arm"]
    arms = ["B_da_eligibility", "F25_quality_interpolated_quota", "G_f25_task_preserving_gate"]
    labels = ["B：DA安全门", "F25：宽支持", "G：硬任务保持门"]
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
    metrics = [
        ("has_eligible", "至少1条合格", "#4C78A8"),
        ("has_two", "至少2条合格", "#72B7B2"),
        ("active_at_0p01", r"有效组（$D_s\geq0.01$）", "#F58518"),
    ]
    width = 0.24
    for i, (key, label, color) in enumerate(metrics):
        values = [s[a][key]["mean"] * 100 for a in arms]
        bars = axes[0].bar(x + (i-1)*width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+0.8, f"{value:.1f}", ha="center", fontsize=8)
    axes[0].axhline(40, color="#4A5568", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=10, ha="right")
    axes[0].set_ylabel("锚点比例（%）")
    axes[0].set_ylim(0, 72)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_xlabel("（a）硬任务门造成的候选支持损失", labelpad=8)

    b = s["B_da_eligibility"]
    f = s["F25_quality_interpolated_quota"]
    g = s["G_f25_task_preserving_gate"]
    names = ["Route cost", "Progress guard", "Risk reward", "Comfort cost"]
    # Scale changes only for compact visualization; signs retain their meanings.
    fv = [
        (f["route_cost_top"]["mean"]-b["route_cost_top"]["mean"])*1e4,
        (f["progress_guard_top"]["mean"]-b["progress_guard_top"]["mean"])*1e3,
        (f["risk_top"]["mean"]-b["risk_top"]["mean"])*1e3,
        (f["comfort_cost_top"]["mean"]-b["comfort_cost_top"]["mean"])*1e3,
    ]
    gv = [
        (g["route_cost_top"]["mean"]-b["route_cost_top"]["mean"])*1e4,
        (g["progress_guard_top"]["mean"]-b["progress_guard_top"]["mean"])*1e3,
        (g["risk_top"]["mean"]-b["risk_top"]["mean"])*1e3,
        (g["comfort_cost_top"]["mean"]-b["comfort_cost_top"]["mean"])*1e3,
    ]
    xx = np.arange(4)
    bars_f = axes[1].bar(xx-0.18, fv, 0.36, color="#4C78A8", label="F25相对B")
    bars_g = axes[1].bar(xx+0.18, gv, 0.36, color="#54A24B", label="G相对B")
    axes[1].axhline(0, color="#4A5568", linewidth=0.8)
    axes[1].set_xticks(xx, names, rotation=12, ha="right")
    axes[1].set_ylabel("缩放后的相对变化")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("（b）任务保持门修复Route但损失信号", labelpad=8)
    for bars in (bars_f, bars_g):
        for bar in bars:
            value = bar.get_height()
            axes[1].text(bar.get_x()+bar.get_width()/2, value+(0.18 if value>=0 else -0.28), f"{value:+.2f}", ha="center", va="bottom" if value>=0 else "top", fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"task_preserving_gate.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
