#!/usr/bin/env python3
"""Generate thesis-ready figures for the C0 vs historical B10 quasi-ablation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np
from fontTools.ttLib import TTCollection


REPO_ROOT = Path(__file__).resolve().parents[2]
OPEN_LOOP_JSON = (
    REPO_ROOT
    / "HDP-nuplan/tmp/chapter3_hybrid_ablation_v1/post_c0/open_loop_results.json"
)
CLOSED_LOOP_DIR = (
    REPO_ROOT
    / "HDP-nuplan/tmp/chapter3_closed_loop/EXP_CH3_C0_B10_safety40_0918"
)
PAIRED_JSON = CLOSED_LOOP_DIR / "paired_results.json"
B10_SUMMARY_JSON = CLOSED_LOOP_DIR / "B10.summary.json"
C0_SUMMARY_JSON = CLOSED_LOOP_DIR / "C0.summary.json"
OUTPUT_DIR = (
    REPO_ROOT
    / "HDP-nuplan/doc_hdp_nuplan/phd_midterm/stage_results/"
    "EXP_CH3_C0_vs_historical_B10/figures"
)
SONG_FONT_FAMILY = "Noto Serif CJK SC"
SONG_FONT_COLLECTION = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
SONG_FONT_SC_INDEX = 2


def apply_exact_font(fig: plt.Figure, font_path: Path) -> None:
    """Bind every text artist to one exact TTC-extracted font face."""

    for artist in fig.findobj(match=lambda value: isinstance(value, Text)):
        properties = font_manager.FontProperties(fname=font_path)
        properties.set_size(artist.get_fontsize())
        properties.set_weight(artist.get_fontweight())
        properties.set_style(artist.get_fontstyle())
        artist.set_fontproperties(properties)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib registers the system CJK TTC under its JP family name;
            # the collection still contains the Simplified Chinese glyphs.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
        }
    )


def plot_horizon_errors(open_loop: dict) -> None:
    horizons = np.array([1, 3, 5, 8])
    colors = {"C0": "#0072B2", "historical_B10": "#D55E00"}
    labels = {"C0": "C0（无位置辅助项）", "historical_B10": "历史B10（hybrid）"}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("ade", "fde"),
        ("分时域平均位移误差（ADE）", "分时域终点位移误差（FDE）"),
    ):
        for model in ("C0", "historical_B10"):
            values = [open_loop["models"][model][f"{metric}_{h}s_m"]["mean"] for h in horizons]
            axis.plot(
                horizons,
                values,
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=colors[model],
                label=labels[model],
            )
            for x_value, y_value in zip(horizons, values):
                axis.annotate(
                    f"{y_value:.3f}",
                    (x_value, y_value),
                    xytext=(0, 7 if model == "historical_B10" else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=colors[model],
                )
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("预测时域（s）")
        axis.set_ylabel("误差（m，越低越好）")
        axis.set_xticks(horizons)
        axis.legend(frameon=False, fontsize=9)

    fig.suptitle("C0与历史B10的分时域开环误差", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "open_loop_horizon_error.png", bbox_inches="tight")
    plt.close(fig)


def plot_horizon_errors_songti(open_loop: dict) -> None:
    """Render a publication variant while preserving the original figure."""

    if not SONG_FONT_COLLECTION.is_file():
        raise FileNotFoundError(SONG_FONT_COLLECTION)
    with tempfile.TemporaryDirectory(prefix="chapter3_c0_b10_songti_") as temporary_dir:
        extracted_font = Path(temporary_dir) / "NotoSerifCJKsc-Regular.otf"
        collection = TTCollection(SONG_FONT_COLLECTION)
        collection.fonts[SONG_FONT_SC_INDEX].save(extracted_font)
        font_manager.fontManager.addfont(extracted_font)
        family = font_manager.FontProperties(fname=extracted_font).get_name()
        if family != SONG_FONT_FAMILY:
            raise RuntimeError(f"unexpected extracted font family: {family!r}")

        horizons = np.array([1, 3, 5, 8])
        colors = {"C0": "#0072B2", "historical_B10": "#D55E00"}
        labels = {
            "C0": "Delta模型输出轨迹",
            "historical_B10": "Delta-Recon模型输出轨迹",
        }
        with plt.rc_context(
            {
                "font.family": "DejaVu Serif",
                "axes.unicode_minus": False,
                "axes.grid": False,
            }
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
            for axis, metric, title in zip(
                axes,
                ("ade", "fde"),
                ("分时域平均位移误差（ADE）", "分时域终点位移误差（FDE）"),
            ):
                plotted_values = []
                for model in ("C0", "historical_B10"):
                    values = [
                        open_loop["models"][model][f"{metric}_{h}s_m"]["mean"]
                        for h in horizons
                    ]
                    plotted_values.extend(values)
                    axis.plot(
                        horizons,
                        values,
                        marker="o",
                        linewidth=2.4,
                        markersize=6,
                        color=colors[model],
                        label=labels[model],
                    )
                axis.set_title(title, fontsize=13)
                axis.set_xlabel("预测时域（s）")
                axis.set_ylabel("误差（m）")
                axis.set_xticks(horizons)
                axis.grid(False)
                axis.legend(frameon=False, fontsize=9)
                value_min = min(plotted_values)
                value_max = max(plotted_values)
                value_span = max(value_max - value_min, 1e-6)
                axis.set_ylim(
                    max(0.0, value_min - 0.12 * value_span),
                    value_max + 0.16 * value_span,
                )
                axis.margins(x=0.06)

            apply_exact_font(fig, extracted_font)
            fig.savefig(
                OUTPUT_DIR / "open_loop_horizon_error_songti_delta_delta_recon.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


def plot_paired_delta_quantiles_songti(open_loop: dict) -> None:
    """Render the paired-delta figure with Song font and in-frame annotations."""

    if not SONG_FONT_COLLECTION.is_file():
        raise FileNotFoundError(SONG_FONT_COLLECTION)
    with tempfile.TemporaryDirectory(prefix="chapter3_c0_b10_delta_songti_") as temporary_dir:
        extracted_font = Path(temporary_dir) / "NotoSerifCJKsc-Regular.otf"
        collection = TTCollection(SONG_FONT_COLLECTION)
        collection.fonts[SONG_FONT_SC_INDEX].save(extracted_font)
        font_manager.fontManager.addfont(extracted_font)
        family = font_manager.FontProperties(fname=extracted_font).get_name()
        if family != SONG_FONT_FAMILY:
            raise RuntimeError(f"unexpected extracted font family: {family!r}")

        paired = open_loop["paired_delta_candidate_minus_reference"]["historical_B10"]
        horizons = np.array([1, 3, 5, 8])
        with plt.rc_context(
            {
                "font.family": "DejaVu Serif",
                "axes.unicode_minus": False,
                "axes.grid": False,
            }
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
            for axis, metric, title in zip(
                axes,
                ("ade", "fde"),
                ("ADE配对差", "FDE配对差"),
            ):
                stats = [paired[f"{metric}_{h}s_m"] for h in horizons]
                means = np.array([item["mean"] for item in stats])
                medians = np.array([item["p50"] for item in stats])
                p10 = np.array([item["p10"] for item in stats])
                p90 = np.array([item["p90"] for item in stats])
                axis.axhline(0.0, color="#333333", linewidth=1.2)
                axis.errorbar(
                    horizons,
                    medians,
                    yerr=np.vstack([medians - p10, p90 - medians]),
                    fmt="o",
                    capsize=5,
                    color="#009E73",
                    ecolor="#56B4E9",
                    elinewidth=2.2,
                    label="P50与P10–P90",
                )
                axis.scatter(
                    horizons,
                    means,
                    marker="D",
                    s=42,
                    color="#CC79A7",
                    label="均值",
                    zorder=4,
                )
                lower = float(min(p10.min(), means.min(), medians.min()))
                upper = float(max(p90.max(), means.max(), medians.max()))
                span = max(upper - lower, 1e-6)
                axis.set_ylim(lower - 0.13 * span, upper + 0.16 * span)
                axis.margins(x=0.08)
                for index, (x_value, mean_value) in enumerate(zip(horizons, means)):
                    is_last = index == len(horizons) - 1
                    axis.annotate(
                        f"均值 {mean_value:+.3f}",
                        (x_value, mean_value),
                        xytext=(-7 if is_last else 6, 7),
                        textcoords="offset points",
                        ha="right" if is_last else "left",
                        va="bottom",
                        fontsize=8,
                        annotation_clip=True,
                    )
                axis.set_title(title, fontsize=13)
                axis.set_xlabel("预测时域（s）")
                axis.set_ylabel("Delta-Recon − Delta（m）")
                axis.set_xticks(horizons)
                axis.grid(False)
                axis.legend(frameon=False, fontsize=9, loc="lower right")
                axis.text(
                    0.02,
                    0.98,
                    "负值：Delta-Recon误差更低\n正值：Delta误差更低",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8.5,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "#BBBBBB",
                    },
                )

            apply_exact_font(fig, extracted_font)
            fig.savefig(
                OUTPUT_DIR
                / "open_loop_paired_delta_quantiles_songti_delta_delta_recon.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


def plot_paired_delta_quantiles(open_loop: dict) -> None:
    paired = open_loop["paired_delta_candidate_minus_reference"]["historical_B10"]
    horizons = np.array([1, 3, 5, 8])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    for axis, metric, title in zip(
        axes,
        ("ade", "fde"),
        ("ADE配对差", "FDE配对差"),
    ):
        stats = [paired[f"{metric}_{h}s_m"] for h in horizons]
        means = np.array([item["mean"] for item in stats])
        medians = np.array([item["p50"] for item in stats])
        p10 = np.array([item["p10"] for item in stats])
        p90 = np.array([item["p90"] for item in stats])
        lower = medians - p10
        upper = p90 - medians

        axis.axhline(0.0, color="#333333", linewidth=1.2)
        axis.errorbar(
            horizons,
            medians,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=5,
            color="#009E73",
            ecolor="#56B4E9",
            elinewidth=2.2,
            label="P50与P10–P90",
        )
        axis.scatter(horizons, means, marker="D", s=42, color="#CC79A7", label="均值", zorder=4)
        for x_value, mean_value in zip(horizons, means):
            axis.annotate(
                f"均值 {mean_value:+.3f}",
                (x_value, mean_value),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("预测时域（s）")
        axis.set_ylabel("历史B10 − C0（m）")
        axis.set_xticks(horizons)
        axis.legend(frameon=False, fontsize=9, loc="lower right")
        axis.text(
            0.02,
            0.98,
            "负值：B10误差更低\n正值：C0误差更低",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#BBBBBB"},
        )

    fig.suptitle("开环逐观测配对差的分位数摘要（3000组配对观测）", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "open_loop_paired_delta_quantiles.png", bbox_inches="tight")
    plt.close(fig)


def paired_scenarios(b10_summary: dict, c0_summary: dict) -> list[dict]:
    b10_by_token = {item["scenario"]: item for item in b10_summary["scenarios"]}
    c0_by_token = {item["scenario"]: item for item in c0_summary["scenarios"]}
    if set(b10_by_token) != set(c0_by_token):
        raise RuntimeError("B10 and C0 scenario sets differ")

    pairs = []
    for token, baseline in b10_by_token.items():
        candidate = c0_by_token[token]
        pairs.append(
            {
                "token": token,
                "scenario_type": baseline["scenario_type"],
                "log_name": baseline["log_name"],
                "b10_score": baseline["score"],
                "c0_score": candidate["score"],
                "delta": candidate["score"] - baseline["score"],
                "b10_nc": baseline["no_ego_at_fault_collisions"],
                "c0_nc": candidate["no_ego_at_fault_collisions"],
            }
        )
    return pairs


def plot_closed_loop_waterfall(pairs: list[dict]) -> None:
    ordered = sorted(pairs, key=lambda item: item["delta"])
    collision_token = "eb1efeeaad7c54fb"
    colors = []
    for item in ordered:
        if item["token"] == collision_token:
            colors.append("#7B3294")
        elif item["delta"] > 1e-12:
            colors.append("#009E73")
        elif item["delta"] < -1e-12:
            colors.append("#D55E00")
        else:
            colors.append("#B8B8B8")

    fig, axis = plt.subplots(figsize=(14.5, 6.2), constrained_layout=True)
    x_values = np.arange(len(ordered))
    deltas = np.array([item["delta"] for item in ordered])
    axis.bar(x_values, deltas, color=colors, width=0.82)
    axis.axhline(0.0, color="#222222", linewidth=1.1)
    axis.set_xlim(-0.8, len(ordered) - 0.2)
    axis.set_xlabel("按C0−B10综合分差值排序的40个场景")
    axis.set_ylabel("闭环综合分差值（C0 − B10）")
    axis.set_title("固定safety40逐场闭环综合分差值", fontsize=15, fontweight="bold")
    axis.set_xticks([])

    collision_index = next(index for index, item in enumerate(ordered) if item["token"] == collision_token)
    collision = ordered[collision_index]
    axis.annotate(
        "新增责任碰撞\n"
        f"{collision_token}\n"
        f"ΔScore={collision['delta']:+.4f}",
        xy=(collision_index, collision["delta"]),
        xytext=(collision_index + 5.2, collision["delta"] + 0.13),
        arrowprops={"arrowstyle": "->", "color": "#7B3294", "linewidth": 1.4},
        fontsize=9.5,
        color="#5A216C",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#7B3294"},
    )

    wins = sum(item["delta"] > 1e-12 for item in ordered)
    losses = sum(item["delta"] < -1e-12 for item in ordered)
    ties = len(ordered) - wins - losses
    axis.text(
        0.99,
        0.97,
        f"C0胜/负/平：{wins}/{losses}/{ties}\n平均差：{deltas.mean():+.6f}\n中位数：{np.median(deltas):+.6f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#AAAAAA"},
    )
    fig.savefig(OUTPUT_DIR / "closed_loop_score_delta_waterfall.png", bbox_inches="tight")
    plt.close(fig)


def plot_group_score_delta(paired: dict) -> None:
    display_names = {
        "intersection": "路口",
        "vehicle_interaction": "车辆交互",
        "vru_rules": "行人及规则",
        "dynamics_environment": "动态与环境",
        "basic_preservation": "基础能力",
    }
    groups = list(display_names)
    deltas = [paired["group_results"][group]["metrics"]["score"]["mean_delta"] for group in groups]
    counts = [paired["group_results"][group]["count"] for group in groups]
    colors = ["#009E73" if value > 0 else "#D55E00" if value < 0 else "#999999" for value in deltas]

    fig, axis = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    x_values = np.arange(len(groups))
    bars = axis.bar(x_values, deltas, color=colors, width=0.65)
    axis.axhline(0.0, color="#222222", linewidth=1.1)
    axis.set_xticks(x_values, [f"{display_names[group]}\n(N={count})" for group, count in zip(groups, counts)])
    axis.set_ylabel("闭环综合分差值（C0 − B10）")
    axis.set_title("五个预注册场景组的闭环综合分差异", fontsize=15, fontweight="bold")
    for bar, value in zip(bars, deltas):
        axis.annotate(
            f"{value:+.6f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7 if value >= 0 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
        )
    axis.text(
        0.98,
        0.08,
        "车辆交互组的负差主要由\n单个新增责任碰撞驱动",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#9C2F00",
    )
    fig.savefig(OUTPUT_DIR / "closed_loop_group_score_delta.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    open_loop = load_json(OPEN_LOOP_JSON)
    paired = load_json(PAIRED_JSON)
    b10_summary = load_json(B10_SUMMARY_JSON)
    c0_summary = load_json(C0_SUMMARY_JSON)
    pairs = paired_scenarios(b10_summary, c0_summary)

    plot_horizon_errors(open_loop)
    plot_horizon_errors_songti(open_loop)
    plot_paired_delta_quantiles(open_loop)
    plot_paired_delta_quantiles_songti(open_loop)
    plot_closed_loop_waterfall(pairs)
    plot_group_score_delta(paired)

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
