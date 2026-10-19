#!/usr/bin/env python3
"""Select objective C0/B10 qualitative cases and render raw trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.text import Text
import numpy as np
from fontTools.ttLib import TTCollection


FONT_COLLECTION = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
FONT_COLLECTION_SC_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def apply_exact_font(fig: plt.Figure, font_path: Path) -> None:
    for artist in fig.findobj(match=lambda value: isinstance(value, Text)):
        properties = font_manager.FontProperties(fname=font_path)
        properties.set_size(artist.get_fontsize())
        properties.set_weight(artist.get_fontweight())
        properties.set_style(artist.get_fontstyle())
        artist.set_fontproperties(properties)


def closest_index(
    values: np.ndarray,
    candidates: np.ndarray,
    target: float,
    excluded: set[int],
) -> int:
    available = np.asarray([value for value in candidates if int(value) not in excluded])
    if available.size == 0:
        raise RuntimeError("no unique qualitative candidate remains")
    return int(available[np.argmin(np.abs(values[available] - target))])


def mean_metric(data: np.lib.npyio.NpzFile, prefix: str, metric: str, index: int) -> float:
    return float(data[f"{prefix}__{metric}"][:, index].mean())


def oriented_box_polygon(state: np.ndarray) -> np.ndarray:
    """Return the four corners of a cached traffic-participant state."""

    center = state[:2].astype(np.float64)
    direction = state[2:4].astype(np.float64)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm >= 1e-8 else np.array([1.0, 0.0])
    lateral = np.array([-direction[1], direction[0]])
    width = max(float(state[6]), 0.5)
    length = max(float(state[7]), 0.7)
    longitudinal_offset = 0.5 * length * direction
    lateral_offset = 0.5 * width * lateral
    return np.asarray(
        [
            center + longitudinal_offset + lateral_offset,
            center + longitudinal_offset - lateral_offset,
            center - longitudinal_offset - lateral_offset,
            center - longitudinal_offset + lateral_offset,
        ]
    )


def plot_scene_context(
    cache_path: Path,
    expert_xy: np.ndarray,
    c0_xy: np.ndarray,
    b10_xy: np.ndarray,
    output: Path,
    *,
    case_label: str,
    repeat_index: int,
    font_path: Path,
) -> None:
    """Render one high-resolution vector-map context and trajectory zoom."""

    with np.load(cache_path, allow_pickle=False) as cache:
        lanes = cache["lanes"].astype(np.float64)
        route_lanes = cache["route_lanes"].astype(np.float64)
        agents = cache["neighbor_agents_past"].astype(np.float64)
        static_objects = cache["static_objects"].astype(np.float64)
        map_name = str(cache["map_name"].item())
        scenario_type = str(cache["scenario_type"].item())
        token = str(cache["token"].item())

    colors = {
        "专家轨迹": "#111111",
        "Delta模型输出轨迹": "#0072B2",
        "Delta-Recon模型输出轨迹": "#D55E00",
    }
    trajectories = {
        "专家轨迹": expert_xy,
        "Delta模型输出轨迹": c0_xy,
        "Delta-Recon模型输出轨迹": b10_xy,
    }
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.1))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.11, wspace=0.12)
    scene_axis, corridor_axis = axes
    for axis in axes:
        axis.set_facecolor("#fafaf8")
        axis.grid(False)

    def draw_lanes(axis: plt.Axes, *, alpha: float) -> None:
        for lane in lanes:
            if not np.any(lane[..., :8] != 0):
                continue
            center = lane[:, :2]
            left = center + lane[:, 4:6]
            right = center + lane[:, 6:8]
            axis.plot(left[:, 0], left[:, 1], color="#a4a9ad", linewidth=0.65, alpha=alpha, zorder=1)
            axis.plot(right[:, 0], right[:, 1], color="#a4a9ad", linewidth=0.65, alpha=alpha, zorder=1)
            axis.plot(center[:, 0], center[:, 1], color="#b9bdc1", linewidth=0.45, linestyle=(0, (3, 4)), alpha=alpha, zorder=1)
        for lane in route_lanes:
            if not np.any(lane[..., :8] != 0):
                continue
            center = lane[:, :2]
            left = center + lane[:, 4:6]
            right = center + lane[:, 6:8]
            axis.add_patch(
                Polygon(
                    np.vstack([left, right[::-1]]),
                    closed=True,
                    facecolor="#b7dfb1",
                    edgecolor="none",
                    alpha=0.20,
                    zorder=1.2,
                )
            )
            axis.plot(center[:, 0], center[:, 1], color="#479447", linewidth=1.0, linestyle="--", alpha=0.65, zorder=1.5)

    draw_lanes(scene_axis, alpha=0.55)
    draw_lanes(corridor_axis, alpha=0.28)

    current_agents = agents[:, -1]
    current_valid = np.any(current_agents[:, :8] != 0, axis=-1)
    for agent_index in np.flatnonzero(current_valid):
        history = agents[agent_index]
        history_valid = np.any(history[:, :8] != 0, axis=-1)
        if history_valid.sum() > 1:
            scene_axis.plot(
                history[history_valid, 0],
                history[history_valid, 1],
                color="#7d8790",
                linewidth=0.75,
                alpha=0.35,
                zorder=2,
            )
        state = current_agents[agent_index]
        if state[8] > 0.5:
            face, edge = "#d9dee3", "#59636d"
        elif state[9] > 0.5:
            face, edge = "#f5c98a", "#a66518"
        else:
            face, edge = "#c9b5e8", "#6f4b9a"
        scene_axis.add_patch(
            Polygon(
                oriented_box_polygon(state),
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )
        )
    for state in static_objects:
        if np.any(state != 0):
            scene_axis.scatter(
                state[0], state[1], marker="s", s=12, facecolor="#555555", edgecolor="none", alpha=0.75, zorder=3
            )

    ego_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.297, 5.176])
    scene_axis.add_patch(
        Polygon(
            oriented_box_polygon(ego_state),
            closed=True,
            facecolor="#f2b134",
            edgecolor="#7c5710",
            linewidth=1.3,
            zorder=6,
        )
    )
    scene_axis.text(0.0, -2.0, "自车", ha="center", va="top", fontsize=10, color="#5c420f", zorder=7)

    for axis in axes:
        for label, xy in trajectories.items():
            axis.plot(xy[:, 0], xy[:, 1], color="white", linewidth=5.2, alpha=0.92, zorder=7)
            axis.plot(xy[:, 0], xy[:, 1], color=colors[label], linewidth=2.8, label=label, zorder=8)
            marker_indices = np.arange(9, xy.shape[0], 10)
            axis.scatter(
                xy[marker_indices, 0],
                xy[marker_indices, 1],
                color=colors[label],
                edgecolor="white",
                linewidth=0.55,
                s=29,
                zorder=9,
            )
            axis.scatter(
                xy[-1, 0],
                xy[-1, 1],
                color=colors[label],
                edgecolor="white",
                linewidth=0.75,
                marker="*",
                s=150,
                zorder=10,
            )

    all_xy = np.concatenate(list(trajectories.values()), axis=0)
    x_min = min(-15.0, float(all_xy[:, 0].min()) - 5.0)
    x_max = float(all_xy[:, 0].max()) + 8.0
    y_min = min(-14.0, float(all_xy[:, 1].min()) - 7.0)
    y_max = max(14.0, float(all_xy[:, 1].max()) + 7.0)
    scene_axis.set_xlim(x_min, x_max)
    scene_axis.set_ylim(y_min, y_max)
    scene_axis.set_aspect("equal", adjustable="box")

    x_span = max(float(np.ptp(all_xy[:, 0])), 1.0)
    y_span = max(float(np.ptp(all_xy[:, 1])), 0.25)
    corridor_axis.set_xlim(float(all_xy[:, 0].min()) - 0.025 * x_span, float(all_xy[:, 0].max()) + 0.06 * x_span)
    corridor_axis.set_ylim(float(all_xy[:, 1].min()) - 0.22 * y_span, float(all_xy[:, 1].max()) + 0.55 * y_span)

    for axis in axes:
        axis.set_xlabel("纵向位置 x（m）", fontsize=12)
        axis.set_ylabel("横向位置 y（m）", fontsize=12)
        axis.tick_params(labelsize=10.5)
    scene_axis.set_title("(a) 场景俯视图", fontsize=13, pad=10)
    corridor_axis.set_title("(b) 三轨迹走廊放大", fontsize=13, pad=10)

    legend_handles = [
        Line2D([0], [0], color=colors["专家轨迹"], linewidth=3, label="专家轨迹"),
        Line2D([0], [0], color=colors["Delta模型输出轨迹"], linewidth=3, label="Delta模型输出轨迹"),
        Line2D([0], [0], color=colors["Delta-Recon模型输出轨迹"], linewidth=3, label="Delta-Recon模型输出轨迹"),
        Patch(facecolor="#f2b134", edgecolor="#7c5710", label="自车"),
        Patch(facecolor="#d9dee3", edgecolor="#59636d", label="周边车辆"),
        Patch(facecolor="#b7dfb1", edgecolor="#479447", alpha=0.45, label="规划路线"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        frameon=False,
        fontsize=11.5,
        handlelength=2.2,
    )
    apply_exact_font(fig, font_path)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data:
        scenes = data["scene_names"].astype(str)
        logs = data["log_names"].astype(str)
        expert = data["expert_trajectories"].astype(np.float64)
        c0 = data["c0_trajectories"].astype(np.float64)
        b10 = data["historical_b10_trajectories"].astype(np.float64)
        delta_1s = (
            data["historical_b10__ade_1s_m"] - data["c0__ade_1s_m"]
        ).mean(axis=0)
        delta_8s = (
            data["historical_b10__ade_8s_m"] - data["c0__ade_8s_m"]
        ).mean(axis=0)
        endpoint_distance = np.linalg.norm(expert[:, -1, :2], axis=-1)
        dynamic = np.flatnonzero(endpoint_distance >= 10.0)
        crossover = np.flatnonzero(
            (endpoint_distance >= 10.0) & (delta_1s < 0.0) & (delta_8s > 0.0)
        )
        if crossover.size == 0:
            raise RuntimeError("no dynamic temporal-crossover case found")

        excluded: set[int] = set()
        selections: list[tuple[str, str, int]] = []
        rules = (
            (
                "typical_dynamic",
                "动态场景中位案例",
                dynamic,
                float(np.median(delta_8s[dynamic])),
            ),
            (
                "temporal_crossover",
                "短期/长期交叉案例",
                crossover,
                float(np.median(delta_8s[crossover])),
            ),
            (
                "c0_advantage_p90",
                "Delta长时优势P90案例",
                dynamic,
                float(np.percentile(delta_8s[dynamic], 90)),
            ),
            (
                "b10_advantage_p10",
                "Delta-Recon长时优势P10案例",
                dynamic,
                float(np.percentile(delta_8s[dynamic], 10)),
            ),
        )
        for key, label, candidates, target in rules:
            index = closest_index(delta_8s, candidates, target, excluded)
            excluded.add(index)
            selections.append((key, label, index))

        details = []
        for key, label, index in selections:
            repeat_delta = (
                data["historical_b10__ade_8s_m"][:, index]
                - data["c0__ade_8s_m"][:, index]
            )
            repeat_index = int(np.argmin(np.abs(repeat_delta - delta_8s[index])))
            cache_path = args.data_dir / scenes[index]
            with np.load(cache_path, allow_pickle=False) as cache:
                scenario_type = str(cache["scenario_type"].item())
                map_name = str(cache["map_name"].item())
            metrics = {}
            for horizon in (1, 3, 5, 8):
                for metric in ("ade", "fde"):
                    name = f"{metric}_{horizon}s_m"
                    metrics[name] = {
                        "delta": mean_metric(data, "c0", name, index),
                        "delta_recon": mean_metric(
                            data, "historical_b10", name, index
                        ),
                    }
            details.append(
                {
                    "key": key,
                    "label": label,
                    "scene_index": index,
                    "scene": scenes[index],
                    "log_name": logs[index],
                    "map_name": map_name,
                    "scenario_type": scenario_type,
                    "expert_endpoint_distance_m": float(endpoint_distance[index]),
                    "display_repeat_index_zero_based": repeat_index,
                    "display_repeat_index_one_based": repeat_index + 1,
                    "selection_value_b10_minus_c0_8s_ade_m": float(delta_8s[index]),
                    "b10_minus_c0_1s_ade_m": float(delta_1s[index]),
                    "metrics_are_three_repeat_means": True,
                    "metrics": metrics,
                }
            )

        if not FONT_COLLECTION.is_file():
            raise FileNotFoundError(FONT_COLLECTION)
        with tempfile.TemporaryDirectory(prefix="chapter3_c0_b10_qualitative_") as temporary_dir:
            extracted_font = Path(temporary_dir) / "NotoSerifCJKsc-Regular.otf"
            collection = TTCollection(FONT_COLLECTION)
            collection.fonts[FONT_COLLECTION_SC_INDEX].save(extracted_font)
            font_manager.fontManager.addfont(extracted_font)
            with plt.rc_context(
                {"font.family": "DejaVu Serif", "axes.unicode_minus": False}
            ):
                fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.5))
                fig.subplots_adjust(
                    left=0.07,
                    right=0.99,
                    top=0.91,
                    bottom=0.10,
                    hspace=0.42,
                    wspace=0.18,
                )
                colors = {
                    "专家轨迹": "#111111",
                    "Delta模型输出轨迹": "#0072B2",
                    "Delta-Recon模型输出轨迹": "#D55E00",
                }
                panel_letters = ("a", "b", "c", "d")
                for axis, letter, (_, label, index), detail in zip(
                    axes.flat, panel_letters, selections, details
                ):
                    repeat_index = detail["display_repeat_index_zero_based"]
                    trajectories = {
                        "专家轨迹": expert[index, :, :2],
                        "Delta模型输出轨迹": c0[
                            repeat_index, index, :, :2
                        ],
                        "Delta-Recon模型输出轨迹": b10[
                            repeat_index, index, :, :2
                        ],
                    }
                    for name, xy in trajectories.items():
                        axis.plot(
                            xy[:, 0],
                            xy[:, 1],
                            color=colors[name],
                            linewidth=2.0,
                            label=name,
                        )
                        marker_indices = np.arange(9, xy.shape[0], 10)
                        axis.scatter(
                            xy[marker_indices, 0],
                            xy[marker_indices, 1],
                            color=colors[name],
                            edgecolor="white",
                            linewidth=0.5,
                            s=25,
                            zorder=4,
                        )
                        axis.scatter(
                            xy[-1, 0],
                            xy[-1, 1],
                            color=colors[name],
                            edgecolor="white",
                            linewidth=0.7,
                            marker="*",
                            s=125,
                            zorder=5,
                        )
                    all_xy = np.concatenate(list(trajectories.values()), axis=0)
                    x_span = max(float(np.ptp(all_xy[:, 0])), 1.0)
                    y_span = max(float(np.ptp(all_xy[:, 1])), 0.5)
                    axis.set_xlim(
                        float(all_xy[:, 0].min()) - 0.04 * x_span,
                        float(all_xy[:, 0].max()) + 0.06 * x_span,
                    )
                    axis.set_ylim(
                        float(all_xy[:, 1].min()) - 0.18 * y_span,
                        float(all_xy[:, 1].max()) + 0.18 * y_span,
                    )
                    axis.set_xlabel("纵向位置 x（m）", fontsize=11)
                    axis.set_ylabel("横向位置 y（m）", fontsize=11)
                    axis.grid(False)
                    axis.tick_params(labelsize=10)
                    axis.text(
                        0.5,
                        -0.20,
                        f"({letter}) {label}",
                        transform=axis.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                    )
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.995),
                    ncol=3,
                    frameon=False,
                    fontsize=11,
                )
                apply_exact_font(fig, extracted_font)
                fig.savefig(
                    args.output_dir / "qualitative_representative_cases_songti.png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(fig)

                for (key, label, index), detail in zip(selections, details):
                    repeat_index = detail["display_repeat_index_zero_based"]
                    context_filename = f"qualitative_scene_context_{key}_songti.png"
                    plot_scene_context(
                        args.data_dir / scenes[index],
                        expert[index, :, :2],
                        c0[repeat_index, index, :, :2],
                        b10[repeat_index, index, :, :2],
                        args.output_dir / context_filename,
                        case_label=label,
                        repeat_index=repeat_index,
                        font_path=extracted_font,
                    )
                    detail["scene_context_figure"] = context_filename

    summary = {
        "selection_contract": {
            "selection_uses_three_repeat_means": True,
            "dynamic_scene_rule": "expert 8 s endpoint distance >= 10 m",
            "dynamic_scene_count": int(dynamic.size),
            "typical_dynamic": "closest to dynamic-scene median B10-C0 8 s ADE",
            "temporal_crossover": (
                "among dynamic scenes with B10-C0 1 s ADE < 0 and 8 s ADE > 0, "
                "closest to candidate median 8 s delta"
            ),
            "c0_advantage_p90": "closest to dynamic-scene P90 B10-C0 8 s ADE",
            "b10_advantage_p10": "closest to dynamic-scene P10 B10-C0 8 s ADE",
            "display_repeat": "repeat whose B10-C0 8 s ADE is closest to the scene three-repeat mean",
            "manual_outcome_selection": False,
        },
        "inputs": {
            "paired_metrics_npz": str(args.input.resolve()),
            "data_dir": str(args.data_dir.resolve()),
        },
        "cases": details,
    }
    (args.output_dir / "qualitative_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
