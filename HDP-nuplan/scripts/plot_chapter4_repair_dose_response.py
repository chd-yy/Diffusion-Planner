#!/usr/bin/env python3
"""Plot the frozen Chapter-4 repaired-candidate dose-response results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HDP = Path(__file__).resolve().parents[1]
INPUT = HDP / "tmp/midterm_posttraining/EXP_CH4_repair_dose_response_1004/validation1000"
OUTPUT = HDP / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_repair_dose_response/figures/dose_response.png"
STEPS = np.array([20, 50, 100, 200])


def metric(step: int, name: str, field: str = "mean") -> float:
    report = json.loads((INPUT / f"step_{step:04d}.json").read_text())
    return report["paired_delta_rl_minus_supervised"][name][field]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)

    progress = np.array([metric(step, "progress") for step in STEPS])
    progress_p10 = np.array([metric(step, "progress", "p10") for step in STEPS])
    progress_p90 = np.array([metric(step, "progress", "p90") for step in STEPS])
    axes[0, 0].plot(STEPS, progress, "o-", label="Mean")
    axes[0, 0].fill_between(STEPS, progress_p10, progress_p90, alpha=0.2, label="P10-P90")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].axhline(0.005, color="tab:green", linestyle="--", label="Materiality gate")
    axes[0, 0].set(title="Progress delta vs B10", xlabel="Optimizer updates", ylabel="Delta")
    axes[0, 0].legend(fontsize=8)

    for name, color in (("collision_cost", "tab:red"), ("route_cost", "tab:orange"),
                        ("comfort_cost", "tab:purple")):
        axes[0, 1].plot(STEPS, [metric(step, name) for step in STEPS], "o-", label=name, color=color)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set(title="Cost deltas vs B10", xlabel="Optimizer updates", ylabel="Delta (lower is better)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(STEPS, [metric(step, "reward", "p50") for step in STEPS], "o-", label="Reward median")
    axes[1, 0].plot(STEPS, [metric(step, "reward") for step in STEPS], "s--", label="Reward mean")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Reward delta vs B10", xlabel="Optimizer updates", ylabel="Delta")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(STEPS, [metric(step, "ade_m", "relative_change_percent") for step in STEPS],
                    "o-", label="ADE")
    axes[1, 1].plot(STEPS, [metric(step, "fde_m", "relative_change_percent") for step in STEPS],
                    "s-", label="FDE")
    axes[1, 1].axhline(0.5, color="tab:red", linestyle="--", label="Preservation gate")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set(title="Imitation error change", xlabel="Optimizer updates", ylabel="Relative change (%)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xticks(STEPS)
    axes[0, 0].scatter([100], [progress[2]], s=120, facecolors="none", edgecolors="tab:green", linewidths=2)
    fig.suptitle("Repaired-candidate post-training dose response (validation-1000 x 3)")
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
