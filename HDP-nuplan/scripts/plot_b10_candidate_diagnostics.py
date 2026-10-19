#!/usr/bin/env python3
"""Render the Chapter 4 B10 candidate-diagnostic summary figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    scales = payload["protocol"]["noise_scales"]
    summaries = payload["summary_by_noise_scale"]
    rows = payload["per_scene"]

    endpoint = [summaries[str(scale)]["endpoint_pairwise_distance_m"]["p50"] for scale in scales]
    conditional_reward_std = []
    for scale in scales:
        values = [
            row["reward_std_eligible"]
            for row in rows
            if row["noise_scale"] == scale and row["has_two_eligible_candidates"] > 0.5
        ]
        conditional_reward_std.append(float(np.median(values)))
    has_eligible = [summaries[str(scale)]["has_eligible_candidate"]["mean"] for scale in scales]
    active_001 = [summaries[str(scale)]["active_group_at_0_01"]["mean"] for scale in scales]
    mean_ade = [summaries[str(scale)]["mean_candidate_ade_m"]["mean"] for scale in scales]
    oracle_ade = [summaries[str(scale)]["oracle_best_ade_m"]["mean"] for scale in scales]
    reward_best_ade = [summaries[str(scale)]["best_reward_candidate_ade_m"]["mean"] for scale in scales]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))

    ax = axes[0]
    ax.plot(scales, endpoint, marker="o", linewidth=2, label="Endpoint diversity (m)")
    ax.set_xlabel("Sampling noise scale")
    ax.set_ylabel("Median endpoint pair distance (m)")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(scales, conditional_reward_std, marker="s", linewidth=2,
              color="#d55e00", label="Eligible reward std")
    twin.set_ylabel("Median eligible reward std")
    lines = ax.lines + twin.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(scales, np.asarray(has_eligible) * 100, marker="o", linewidth=2,
            label="Has eligible candidate")
    ax.plot(scales, np.asarray(active_001) * 100, marker="s", linewidth=2,
            label="Active at reward std >= 0.01")
    ax.set_xlabel("Sampling noise scale")
    ax.set_ylabel("Scene fraction (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(scales, mean_ade, marker="o", linewidth=2, label="Candidate mean ADE")
    ax.plot(scales, oracle_ade, marker="s", linewidth=2, label="Oracle best-of-32 ADE")
    ax.plot(scales, reward_best_ade, marker="^", linewidth=2,
            label="Highest-reward ADE")
    ax.set_xlabel("Sampling noise scale")
    ax.set_ylabel("ADE (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("B10 candidate diversity, usable advantage, and quality trade-off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
