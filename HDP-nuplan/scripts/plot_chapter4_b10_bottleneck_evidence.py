#!/usr/bin/env python3
"""Create the paper figure supporting the Chapter-4 B10 bottleneck claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    scales = np.asarray(payload["protocol"]["noise_scales"], dtype=float)
    summaries = payload["summary_by_noise_scale"]
    endpoint = np.asarray([
        summaries[str(scale)]["endpoint_pairwise_distance_m"]["p50"]
        for scale in scales
    ])
    trajectory = np.asarray([
        summaries[str(scale)]["trajectory_pairwise_distance_m"]["p50"]
        for scale in scales
    ])
    active = np.asarray([
        summaries[str(scale)]["active_group_at_0_01"]["mean"] * 100
        for scale in scales
    ])
    has_eligible = np.asarray([
        summaries[str(scale)]["has_eligible_candidate"]["mean"] * 100
        for scale in scales
    ])

    # Use a Simplified-Chinese Song typeface for thesis figures.
    font_path = Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(f"Chinese Song typeface not found: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    songti_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": songti_font,
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.28, wspace=0.27)

    ax = axes[0]
    ax.plot(scales, endpoint, "o-", linewidth=2, label="终点两两距离")
    ax.plot(scales, trajectory, "s-", linewidth=2, label="整轨迹两两距离")
    ax.set(xlabel="采样噪声尺度", ylabel="两两距离中位数（m）")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.annotate(f"{endpoint[0]:.4f} m", (scales[0], endpoint[0]),
                xytext=(7, 9), textcoords="offset points")
    ax.annotate(f"{trajectory[0]:.4f} m", (scales[0], trajectory[0]),
                xytext=(7, -15), textcoords="offset points")

    ax.text(0.5, -0.26, "（a）候选轨迹多样性", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

    ax = axes[1]
    ax.plot(scales, active, "s-", linewidth=2, color="#d55e00",
            label="有效优势场景（reward std ≥ 0.01）")
    ax.plot(scales, has_eligible, "o-", linewidth=2, color="#0072b2",
            label="存在安全候选的场景")
    ax.set(xlabel="采样噪声尺度", ylabel="场景比例（%）", ylim=(0, 100))
    ax.legend()
    ax.grid(alpha=0.25)
    ax.annotate(f"{active[0]:.1f}%", (scales[0], active[0]),
                xytext=(7, -15), textcoords="offset points", color="#d55e00")
    ax.text(0.5, -0.26, "（b）有效优势信号支持", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
