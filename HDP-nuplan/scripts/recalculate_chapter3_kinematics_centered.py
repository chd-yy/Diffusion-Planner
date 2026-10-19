#!/usr/bin/env python3
"""Recompute Chapter 3 R0/C0 kinematics with direct centered differences.

The primary descriptors use position-only, time-aligned finite differences:

* velocity: 3-point centered first derivative;
* acceleration: 3-point centered second derivative;
* jerk: 5-point centered third derivative.

Unavailable endpoints are excluded from summary statistics.  Separate raw
second/third waypoint differences are retained as 10 Hz roughness diagnostics
so that the centered derivative does not silently hide point-to-point jitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from scipy.signal import savgol_filter


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--scene-order", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dt", default=0.1, type=float)
    parser.add_argument("--bootstrap-seed", default=3407, type=int)
    parser.add_argument("--bootstrap-iterations", default=10_000, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_descriptors(xy: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Return per-trajectory means for ``xy[..., T, 2]``.

    Velocity and acceleration are defined at indices 1..T-2.  Jerk is
    defined at indices 2..T-3.  No endpoint extrapolation is included.
    """

    velocity = (xy[..., 2:, :] - xy[..., :-2, :]) / (2.0 * dt)
    acceleration = (
        xy[..., 2:, :] - 2.0 * xy[..., 1:-1, :] + xy[..., :-2, :]
    ) / (dt**2)
    jerk = (
        xy[..., 4:, :]
        - 2.0 * xy[..., 3:-1, :]
        + 2.0 * xy[..., 1:-3, :]
        - xy[..., :-4, :]
    ) / (2.0 * dt**3)
    return {
        "velocity_vectors": velocity,
        "acceleration_vectors": acceleration,
        "jerk_vectors": jerk,
        "mean_speed_mps": np.linalg.norm(velocity, axis=-1).mean(axis=-1),
        "mean_acceleration_mps2": np.linalg.norm(acceleration, axis=-1).mean(axis=-1),
        "mean_jerk_mps3": np.linalg.norm(jerk, axis=-1).mean(axis=-1),
    }


def waypoint_roughness(xy: np.ndarray) -> dict[str, np.ndarray]:
    displacement = np.diff(xy, axis=-2)
    second = np.diff(displacement, axis=-2)
    third = np.diff(second, axis=-2)
    return {
        "mean_second_difference_m": np.linalg.norm(second, axis=-1).mean(axis=-1),
        "mean_third_difference_m": np.linalg.norm(third, axis=-1).mean(axis=-1),
    }


def load_log_names(data_dir: Path, scenes: list[str]) -> list[str]:
    names = []
    for scene in scenes:
        with np.load(data_dir / scene, allow_pickle=False) as data:
            names.append(str(data["log_name"].item()))
    return names


def clustered_interval(
    per_scene_delta: np.ndarray,
    log_names: list[str],
    *,
    seed: int,
    iterations: int,
) -> dict:
    names = np.asarray(log_names)
    unique_logs = sorted(set(log_names))
    log_means = np.asarray(
        [per_scene_delta[names == log_name].mean() for log_name in unique_logs]
    )
    rng = np.random.default_rng(seed)
    samples = log_means[
        rng.integers(0, len(unique_logs), size=(iterations, len(unique_logs)))
    ].mean(axis=1)
    return {
        "cluster_unit": "nuPlan log_name",
        "cluster_count": len(unique_logs),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "mean_of_log_means": float(log_means.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "min_log_mean": float(log_means.min()),
        "max_log_mean": float(log_means.max()),
    }


def paired_summary(
    r0: np.ndarray,
    c0: np.ndarray,
    log_names: list[str],
    *,
    seed: int,
    iterations: int,
) -> dict:
    delta = r0 - c0
    per_scene_delta = delta.mean(axis=0)
    return {
        "r0_mean": float(r0.mean()),
        "c0_mean": float(c0.mean()),
        "r0_minus_c0_mean": float(delta.mean()),
        "per_scene_delta_p10": float(np.percentile(per_scene_delta, 10)),
        "per_scene_delta_p50": float(np.percentile(per_scene_delta, 50)),
        "per_scene_delta_p90": float(np.percentile(per_scene_delta, 90)),
        "r0_lower_fraction_scene_repeat": float((delta < 0).mean()),
        "r0_higher_fraction_scene_repeat": float((delta > 0).mean()),
        "r0_higher_fraction_per_scene": float((per_scene_delta > 0).mean()),
        "log_cluster_bootstrap95": clustered_interval(
            per_scene_delta,
            log_names,
            seed=seed,
            iterations=iterations,
        ),
    }


def plot_paired_distributions(
    ade_delta: np.ndarray,
    jerk_delta: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    panels = (
        (ade_delta, "R0−C0逐场平均8秒ADE（m）"),
        (jerk_delta, "R0−C0逐场平均中心差分jerk描述量（m/s³）"),
    )
    for axis, (delta, label) in zip(axes, panels):
        axis.hist(delta, bins=50, color="#4c78a8", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(
            delta.mean(),
            color="#e45756",
            linestyle="--",
            label=f"均值={delta.mean():.3f}",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("场景数（三次重复均值）")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_centered_profiles(
    r0_xy: np.ndarray,
    c0_xy: np.ndarray,
    expert_xy: np.ndarray,
    scene: str,
    dt: float,
    output: Path,
) -> None:
    profiles = {}
    for label, xy in (
        ("Expert", expert_xy),
        ("R0 absolute", r0_xy),
        ("C0 increment", c0_xy),
    ):
        profiles[label] = centered_descriptors(xy[None], dt)
    colors = {"Expert": "black", "R0 absolute": "#e45756", "C0 increment": "#4c78a8"}
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True, constrained_layout=True)
    specifications = (
        ("velocity_vectors", np.arange(1, r0_xy.shape[0] - 1), "Speed (m/s)"),
        ("acceleration_vectors", np.arange(1, r0_xy.shape[0] - 1), "Centered acceleration proxy (m/s^2)"),
        ("jerk_vectors", np.arange(2, r0_xy.shape[0] - 2), "Centered jerk proxy (m/s^3)"),
    )
    for axis, (key, indices, ylabel) in zip(axes, specifications):
        for label, values in profiles.items():
            magnitude = np.linalg.norm(values[key][0], axis=-1)
            time = (indices + 1) * dt
            axis.plot(time, magnitude, label=label, color=colors[label], linewidth=1.4)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Future time (s)")
    axes[0].set_title(f"Position-only centered differences: {Path(scene).stem}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representative_continuity_case(
    r0_xy: np.ndarray,
    c0_xy: np.ndarray,
    expert_xy: np.ndarray,
    scene: str,
    dt: float,
    output: Path,
    *,
    repeat_index: int,
    per_scene_r0_jerk: float,
    per_scene_c0_jerk: float,
    publication_style: bool = False,
) -> dict:
    """Plot one pre-specified high-roughness, non-extreme dynamic case."""

    # The aggregate tables remain based on the direct centered differences
    # above.  Only the two lower panels in this qualitative figure use an
    # SG(11, 3) local-polynomial derivative to avoid letting raw 10 Hz noise
    # dominate the vertical scale.  Raw positions remain unchanged above.
    sg_window = 11
    sg_polyorder = 3
    sg_half_window = sg_window // 2

    def sg_profile(xy: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "acceleration_vectors": savgol_filter(
                xy,
                window_length=sg_window,
                polyorder=sg_polyorder,
                deriv=2,
                delta=dt,
                axis=0,
                mode="interp",
            ),
            "jerk_vectors": savgol_filter(
                xy,
                window_length=sg_window,
                polyorder=sg_polyorder,
                deriv=3,
                delta=dt,
                axis=0,
                mode="interp",
            ),
        }

    r0_profile = sg_profile(r0_xy)
    c0_profile = sg_profile(c0_xy)
    expert_profile = sg_profile(expert_xy)

    r0_second = np.linalg.norm(np.diff(r0_xy, n=2, axis=0), axis=-1)
    c0_second = np.linalg.norm(np.diff(c0_xy, n=2, axis=0), axis=-1)
    roughness_delta = r0_second - c0_second
    # Nine consecutive second differences require eleven positions and span
    # exactly one second from the first to the last displayed waypoint.
    window_steps = 9
    window_scores = np.convolve(
        roughness_delta,
        np.ones(window_steps, dtype=np.float64) / window_steps,
        mode="valid",
    )
    roughness_start = int(np.argmax(window_scores))
    position_start = roughness_start
    position_stop = roughness_start + window_steps + 2
    zoom_slice = slice(position_start, position_stop)

    r0_label = "Abs模型输出轨迹" if publication_style else "R0绝对轨迹"
    c0_label = "Delta模型输出轨迹" if publication_style else "C0增量轨迹"
    colors = {"专家轨迹": "black", r0_label: "#e45756", c0_label: "#4c78a8"}
    trajectories = {
        "专家轨迹": expert_xy,
        r0_label: r0_xy,
        c0_label: c0_xy,
    }
    profiles = {
        "专家轨迹": expert_profile,
        r0_label: r0_profile,
        c0_label: c0_profile,
    }

    figure_size = (14, 8.2) if publication_style else (12, 9)
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    if publication_style:
        fig.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.985,
            bottom=0.10,
            hspace=0.38,
            wspace=0.18,
        )
    else:
        fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.10, hspace=0.52, wspace=0.25)
    full_axis, zoom_axis, acceleration_axis, jerk_axis = axes.flat
    if publication_style:
        for axis in axes.flat:
            axis.tick_params(axis="both", labelsize=11)

    for label, xy in trajectories.items():
        full_axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=colors[label],
            linewidth=2.1 if publication_style else 1.8,
            label=label,
        )
        marker_indices = np.arange(9, xy.shape[0], 10)
        full_axis.scatter(
            xy[marker_indices, 0],
            xy[marker_indices, 1],
            color=colors[label],
            s=18,
            alpha=0.8,
        )
        full_axis.scatter(
            xy[-1, 0],
            xy[-1, 1],
            color=colors[label],
            marker="*",
            s=80,
            zorder=4,
        )
        zoomed = xy[zoom_slice]
        zoom_axis.plot(
            zoomed[:, 0],
            zoomed[:, 1],
            color=colors[label],
            linewidth=1.9 if publication_style else 1.6,
            marker="o",
            markersize=3.5,
            label=label,
        )

    full_axis.scatter([0], [0], marker="x", color="black", s=35, zorder=5)
    for axis in (full_axis, zoom_axis):
        axis.set_xlabel("纵向位置 x（m）", fontsize=12 if publication_style else None)
        axis.set_ylabel("横向位置 y（m）", fontsize=12 if publication_style else None)
        if not publication_style:
            axis.grid(alpha=0.25)
        axis.legend(fontsize=9.5 if publication_style else 8)
    if publication_style:
        full_axis.set_ylim(-1.0, 1.5)
        full_axis.set_aspect("auto")
    else:
        full_axis.axis("equal")

    zoom_start_s = (position_start + 1) * dt
    zoom_end_s = position_stop * dt
    full_caption = (
        "(a) 完整8s轨迹"
        if publication_style
        else "（a）完整8秒轨迹（圆点间隔1秒，星号表示终点）"
    )
    zoom_caption = (
        "(b) 局部1s放大"
        if publication_style
        else f"（b）局部1秒放大（{zoom_start_s:.1f}–{zoom_end_s:.1f}秒，纵轴扩展）"
    )
    full_axis.text(
        0.5,
        -0.18 if publication_style else -0.22,
        full_caption,
        transform=full_axis.transAxes,
        ha="center",
        va="top",
        fontsize=12 if publication_style else 10,
    )
    zoom_axis.text(
        0.5,
        -0.18 if publication_style else -0.22,
        zoom_caption,
        transform=zoom_axis.transAxes,
        ha="center",
        va="top",
        fontsize=12 if publication_style else 10,
    )

    acceleration_caption = "(c) 加速度" if publication_style else "（c）SG平滑加速度描述量（仅用于展示）"
    jerk_caption = "(d) jerk" if publication_style else "（d）SG平滑jerk描述量（仅用于展示）"
    specifications = (
        (
            acceleration_axis,
            "acceleration_vectors",
            np.arange(sg_half_window, r0_xy.shape[0] - sg_half_window),
            acceleration_caption,
            "加速度绝对值（m/s²）" if publication_style else "加速度模长（m/s²）",
        ),
        (
            jerk_axis,
            "jerk_vectors",
            np.arange(sg_half_window, r0_xy.shape[0] - sg_half_window),
            jerk_caption,
            "jerk绝对值（m/s³）" if publication_style else "jerk模长（m/s³）",
        ),
    )
    for axis, key, indices, title, ylabel in specifications:
        for label, values in profiles.items():
            magnitude = np.linalg.norm(values[key][indices], axis=-1)
            time = (indices + 1) * dt
            axis.plot(
                time,
                magnitude,
                color=colors[label],
                linewidth=1.8 if publication_style else 1.4,
                label=label,
            )
        if not publication_style:
            axis.axvspan(zoom_start_s, zoom_end_s, color="grey", alpha=0.12)
        axis.set_xlabel("未来时刻（s）", fontsize=12 if publication_style else None)
        axis.set_ylabel(ylabel, fontsize=12 if publication_style else None)
        axis.text(
            0.5,
            -0.18 if publication_style else -0.22,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12 if publication_style else 10,
        )
        if not publication_style:
            axis.grid(alpha=0.25)
        axis.legend(fontsize=9.5 if publication_style else 8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "repeat_index_zero_based": repeat_index,
        "zoom_position_start_index_zero_based": position_start,
        "zoom_position_stop_index_exclusive": position_stop,
        "zoom_start_s": zoom_start_s,
        "zoom_end_s": zoom_end_s,
        "window_mean_r0_minus_c0_second_difference_m": float(
            window_scores[roughness_start]
        ),
    }


def oriented_box_polygon(state: np.ndarray) -> np.ndarray:
    """Return four corners for [x, y, cos, sin, ..., width, length]."""

    center = state[:2].astype(np.float64)
    direction = state[2:4].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
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


def plot_scene_context_hd(
    cache_path: Path,
    r0_xy: np.ndarray,
    c0_xy: np.ndarray,
    expert_xy: np.ndarray,
    output: Path,
    *,
    repeat_index: int,
    publication_style: bool = False,
) -> None:
    """Render a high-resolution ego-local scene with raw model trajectories."""

    with np.load(cache_path, allow_pickle=False) as data:
        lanes = data["lanes"].astype(np.float64)
        route_lanes = data["route_lanes"].astype(np.float64)
        agents = data["neighbor_agents_past"].astype(np.float64)
        static_objects = data["static_objects"].astype(np.float64)
        map_name = str(data["map_name"].item())
        scenario_type = str(data["scenario_type"].item())
        token = str(data["token"].item())

    expert_label = "专家轨迹" if publication_style else "Expert"
    r0_label = "Abs模型输出轨迹" if publication_style else "R0 absolute"
    c0_label = "Delta模型输出轨迹" if publication_style else "C0 increment"
    colors = {
        expert_label: "#111111",
        r0_label: "#d9534f",
        c0_label: "#2f6fb0",
    }
    trajectories = {
        expert_label: expert_xy,
        r0_label: r0_xy,
        c0_label: c0_xy,
    }

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    if publication_style:
        # Reserve dedicated rows for centered subfigure captions.  This keeps
        # captions below the axes without colliding with the x-axis labels.
        grid = fig.add_gridspec(4, 1, height_ratios=(2.3, 0.12, 1.0, 0.12))
        scene_axis = fig.add_subplot(grid[0])
        scene_caption_axis = fig.add_subplot(grid[1])
        corridor_axis = fig.add_subplot(grid[2])
        corridor_caption_axis = fig.add_subplot(grid[3])
        for caption_axis in (scene_caption_axis, corridor_caption_axis):
            caption_axis.set_axis_off()
    else:
        grid = fig.add_gridspec(2, 1, height_ratios=(2.3, 1.0))
        scene_axis = fig.add_subplot(grid[0])
        corridor_axis = fig.add_subplot(grid[1])
    for axis in (scene_axis, corridor_axis):
        axis.set_facecolor("#fafaf8")
        axis.grid(color="#d7d7d2", linewidth=0.5, alpha=0.35, zorder=0)

    def draw_lanes(axis: plt.Axes, *, detail_alpha: float) -> None:
        for lane in lanes:
            if not np.any(lane[..., :8] != 0):
                continue
            center = lane[:, :2]
            left = center + lane[:, 4:6]
            right = center + lane[:, 6:8]
            axis.plot(left[:, 0], left[:, 1], color="#9ea3a8", linewidth=0.75, alpha=detail_alpha, zorder=1)
            axis.plot(right[:, 0], right[:, 1], color="#9ea3a8", linewidth=0.75, alpha=detail_alpha, zorder=1)
            axis.plot(center[:, 0], center[:, 1], color="#b8bcc0", linewidth=0.55, linestyle=(0, (3, 4)), alpha=detail_alpha * 0.8, zorder=1)
        for lane in route_lanes:
            if not np.any(lane[..., :8] != 0):
                continue
            center = lane[:, :2]
            left = center + lane[:, 4:6]
            right = center + lane[:, 6:8]
            polygon = np.vstack([left, right[::-1]])
            axis.add_patch(
                Polygon(polygon, closed=True, facecolor="#b7dfb1", edgecolor="none", alpha=0.18, zorder=1.2)
            )
            axis.plot(center[:, 0], center[:, 1], color="#479447", linewidth=1.1, linestyle="--", alpha=0.65, zorder=1.5)

    draw_lanes(scene_axis, detail_alpha=0.55)
    draw_lanes(corridor_axis, detail_alpha=0.28)

    # Current traffic participants and their observed 2 s histories.
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
                linewidth=0.8,
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
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )
        )

    for state in static_objects:
        if not np.any(state != 0):
            continue
        scene_axis.scatter(
            state[0], state[1], marker="s", s=13, facecolor="#555555", edgecolor="none", alpha=0.75, zorder=3
        )

    ego_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.297, 5.176])
    scene_axis.add_patch(
        Polygon(
            oriented_box_polygon(ego_state),
            closed=True,
            facecolor="#f2b134",
            edgecolor="#7c5710",
            linewidth=1.4,
            zorder=6,
        )
    )
    ego_text = "自车" if publication_style else "Ego"
    scene_axis.text(0.0, -2.1, ego_text, ha="center", va="top", fontsize=9, color="#5c420f", zorder=7)

    for axis in (scene_axis, corridor_axis):
        for label, xy in trajectories.items():
            axis.plot(
                xy[:, 0],
                xy[:, 1],
                color="white",
                linewidth=5.4 if axis is scene_axis else 6.0,
                alpha=0.9,
                zorder=7,
            )
            axis.plot(
                xy[:, 0],
                xy[:, 1],
                color=colors[label],
                linewidth=3.0 if axis is scene_axis else 3.4,
                label=label,
                zorder=8,
            )
            marker_indices = np.arange(9, xy.shape[0], 10)
            axis.scatter(
                xy[marker_indices, 0],
                xy[marker_indices, 1],
                color=colors[label],
                edgecolor="white",
                linewidth=0.6,
                s=26 if axis is scene_axis else 34,
                zorder=9,
            )
            axis.scatter(
                xy[-1, 0],
                xy[-1, 1],
                color=colors[label],
                edgecolor="white",
                linewidth=0.9,
                marker="*",
                s=145 if axis is scene_axis else 180,
                zorder=10,
            )

    all_xy = np.concatenate([expert_xy, r0_xy, c0_xy], axis=0)
    x_min = min(-15.0, float(all_xy[:, 0].min()) - 5.0)
    x_max = float(all_xy[:, 0].max()) + 8.0
    scene_axis.set_xlim(x_min, x_max)
    scene_axis.set_ylim(-18.0, 18.0)
    scene_axis.set_aspect("equal", adjustable="box")
    if publication_style:
        scene_axis.set_xlabel("纵向位置 x（m）")
        scene_axis.set_ylabel("横向位置 y（m）")
        scene_caption_axis.text(
            0.5, 0.5, "(a) 场景与8s轨迹", ha="center", va="center", fontsize=17
        )
    else:
        scene_axis.set_xlabel("Longitudinal x in ego-local frame (m)")
        scene_axis.set_ylabel("Lateral y in ego-local frame (m)")
        scene_axis.set_title("(a) Current scene context with raw 8 s trajectories", loc="left", fontsize=13)

    corridor_lower_margin = 0.25
    corridor_upper_margin = 0.90 if publication_style else 0.12
    corridor_axis.set_xlim(min(-1.0, float(all_xy[:, 0].min()) - 1.0), x_max)
    corridor_axis.set_ylim(
        float(all_xy[:, 1].min()) - corridor_lower_margin,
        float(all_xy[:, 1].max()) + corridor_upper_margin,
    )
    if publication_style:
        corridor_axis.set_xlabel("纵向位置 x（m）")
        corridor_axis.set_ylabel("横向位置 y（m，放大）")
        corridor_caption_axis.text(
            0.5, 0.5, "(b) 轨迹走廊放大", ha="center", va="center", fontsize=17
        )
    else:
        corridor_axis.set_xlabel("Longitudinal x in ego-local frame (m)")
        corridor_axis.set_ylabel("Lateral y (m; expanded)")
        corridor_axis.set_title("(b) Trajectory-corridor zoom (raw, unsmoothed waypoints)", loc="left", fontsize=13)

    legend_handles = [
        Line2D([0], [0], color=colors[expert_label], linewidth=3, label=expert_label),
        Line2D([0], [0], color=colors[c0_label], linewidth=3, label=c0_label),
        Line2D([0], [0], color=colors[r0_label], linewidth=3, label=r0_label),
        Patch(facecolor="#f2b134", edgecolor="#7c5710", label="自车" if publication_style else "Ego vehicle"),
        Patch(facecolor="#d9dee3", edgecolor="#59636d", label="当前车辆" if publication_style else "Current vehicles"),
        Patch(facecolor="#b7dfb1", edgecolor="#479447", alpha=0.45, label="规划路线" if publication_style else "Route lanes"),
    ]
    legend_fontsize = 14 if publication_style else 9
    scene_axis.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.96,
        ncol=2,
        fontsize=legend_fontsize,
        handlelength=2.4,
        borderpad=0.8,
        labelspacing=0.6,
    )
    corridor_legend_options = {
        "framealpha": 0.96,
        "ncol": 3,
        "fontsize": legend_fontsize,
        "handlelength": 2.4,
        "borderpad": 0.8,
        "labelspacing": 0.6,
    }
    if publication_style:
        # The expanded map extent above the trajectories reserves a clear area
        # for the legend inside panel (b)'s upper-right corner.
        corridor_axis.legend(
            loc="upper right",
            borderaxespad=0.6,
            **corridor_legend_options,
        )
    else:
        corridor_axis.legend(loc="upper right", **corridor_legend_options)

    if publication_style:
        figure_title = (
            f"nuPlan场景 {token} | {map_name} | {scenario_type} | 第{repeat_index + 1}次推理\n"
            "处理缓存中的矢量地图与交通参与者；三条轨迹均为物理空间原始输出"
        )
    else:
        figure_title = (
            f"nuPlan scene {token} | {map_name} | {scenario_type} | displayed inference repeat {repeat_index + 1}\n"
            "Vector-map and actor context from the processed scene cache; all three trajectories are raw physical-space outputs"
        )
    if not publication_style:
        fig.suptitle(figure_title, fontsize=14)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("dt must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(args.scene_order.read_text(encoding="utf-8"))
    with np.load(args.input_npz, allow_pickle=False) as data:
        r0_xy = data["r0_trajectories"][..., :2].astype(np.float64)
        c0_xy = data["c0_trajectories"][..., :2].astype(np.float64)
        expert_xy = data["expert_trajectories"][..., :2].astype(np.float64)
        r0_ade = data["r0__ade_8s_m"].astype(np.float64)
        c0_ade = data["c0__ade_8s_m"].astype(np.float64)
    if r0_xy.shape != c0_xy.shape:
        raise ValueError("R0 and C0 trajectory shapes differ")
    if r0_xy.shape[1] != len(scenes):
        raise ValueError("scene order length does not match trajectory array")

    log_names = load_log_names(args.data_dir, scenes)
    r0_centered = centered_descriptors(r0_xy, args.dt)
    c0_centered = centered_descriptors(c0_xy, args.dt)
    expert_centered = centered_descriptors(expert_xy, args.dt)
    r0_roughness = waypoint_roughness(r0_xy)
    c0_roughness = waypoint_roughness(c0_xy)
    expert_roughness = waypoint_roughness(expert_xy)

    centered_summary = {}
    for metric in ("mean_speed_mps", "mean_acceleration_mps2", "mean_jerk_mps3"):
        centered_summary[metric] = paired_summary(
            r0_centered[metric],
            c0_centered[metric],
            log_names,
            seed=args.bootstrap_seed,
            iterations=args.bootstrap_iterations,
        )
        centered_summary[metric]["expert_mean"] = float(expert_centered[metric].mean())

    roughness_summary = {}
    for metric in ("mean_second_difference_m", "mean_third_difference_m"):
        roughness_summary[metric] = paired_summary(
            r0_roughness[metric],
            c0_roughness[metric],
            log_names,
            seed=args.bootstrap_seed,
            iterations=args.bootstrap_iterations,
        )
        roughness_summary[metric]["expert_mean"] = float(expert_roughness[metric].mean())

    ade_delta_per_scene = (r0_ade - c0_ade).mean(axis=0)
    jerk_delta_per_scene = (
        r0_centered["mean_jerk_mps3"] - c0_centered["mean_jerk_mps3"]
    ).mean(axis=0)
    endpoint_distance = np.linalg.norm(expert_xy[:, -1], axis=-1)
    plot_paired_distributions(
        ade_delta_per_scene,
        jerk_delta_per_scene,
        args.output_dir / "paired_delta_distributions_centered.png",
    )

    largest_jerk_index = int(np.argmax(jerk_delta_per_scene))
    plot_centered_profiles(
        r0_xy[0, largest_jerk_index],
        c0_xy[0, largest_jerk_index],
        expert_xy[largest_jerk_index],
        scenes[largest_jerk_index],
        args.dt,
        args.output_dir / "largest_centered_jerk_kinematic_profiles.png",
    )

    dynamic_mask = endpoint_distance >= 10.0
    dynamic_indices = np.flatnonzero(dynamic_mask)
    dynamic_p90 = float(np.percentile(jerk_delta_per_scene[dynamic_mask], 90))
    representative_index = int(
        dynamic_indices[
            np.argmin(np.abs(jerk_delta_per_scene[dynamic_mask] - dynamic_p90))
        ]
    )
    representative_repeat_deltas = (
        r0_centered["mean_jerk_mps3"][:, representative_index]
        - c0_centered["mean_jerk_mps3"][:, representative_index]
    )
    representative_repeat = int(
        np.argmin(
            np.abs(
                representative_repeat_deltas
                - representative_repeat_deltas.mean()
            )
        )
    )
    representative_plot = plot_representative_continuity_case(
        r0_xy[representative_repeat, representative_index],
        c0_xy[representative_repeat, representative_index],
        expert_xy[representative_index],
        scenes[representative_index],
        args.dt,
        args.output_dir / "representative_p90_continuity_case.png",
        repeat_index=representative_repeat,
        per_scene_r0_jerk=float(
            r0_centered["mean_jerk_mps3"][:, representative_index].mean()
        ),
        per_scene_c0_jerk=float(
            c0_centered["mean_jerk_mps3"][:, representative_index].mean()
        ),
    )
    plot_scene_context_hd(
        args.data_dir / scenes[representative_index],
        r0_xy[representative_repeat, representative_index],
        c0_xy[representative_repeat, representative_index],
        expert_xy[representative_index],
        args.output_dir / "representative_p90_scene_context_hd.png",
        repeat_index=representative_repeat,
    )

    bins = (
        ("<1 m", endpoint_distance < 1.0),
        ("1-10 m", (endpoint_distance >= 1.0) & (endpoint_distance < 10.0)),
        ("10-30 m", (endpoint_distance >= 10.0) & (endpoint_distance < 30.0)),
        (">=30 m", endpoint_distance >= 30.0),
    )
    stratified = []
    for label, mask in bins:
        stratified.append(
            {
                "expert_endpoint_distance_bin": label,
                "scene_count": int(mask.sum()),
                "r0_centered_jerk_mps3": float(
                    r0_centered["mean_jerk_mps3"][:, mask].mean()
                ),
                "c0_centered_jerk_mps3": float(
                    c0_centered["mean_jerk_mps3"][:, mask].mean()
                ),
                "r0_minus_c0_centered_jerk_mps3": float(
                    (
                        r0_centered["mean_jerk_mps3"][:, mask]
                        - c0_centered["mean_jerk_mps3"][:, mask]
                    ).mean()
                ),
            }
        )

    summary = {
        "method": {
            "dt_s": args.dt,
            "position_channels_only": True,
            "yaw_channels_used": False,
            "velocity": "(p[i+1] - p[i-1]) / (2*dt), indices 1..T-2",
            "acceleration": "(p[i+1] - 2*p[i] + p[i-1]) / dt^2, indices 1..T-2",
            "jerk": "(p[i+2] - 2*p[i+1] + 2*p[i-1] - p[i-2]) / (2*dt^3), indices 2..T-3",
            "endpoint_policy": "exclude unavailable endpoints from means",
            "scene_distribution_unit": "1000 scenes after averaging 3 inference repeats",
        },
        "inputs": {
            "trajectory_npz": str(args.input_npz.resolve()),
            "trajectory_npz_sha256": sha256(args.input_npz),
            "scene_order": str(args.scene_order.resolve()),
            "scene_order_sha256": sha256(args.scene_order),
            "data_dir": str(args.data_dir.resolve()),
            "trajectory_shape": list(r0_xy.shape),
        },
        "centered_kinematics": centered_summary,
        "waypoint_roughness": roughness_summary,
        "stratified_centered_jerk": stratified,
        "largest_centered_jerk_degradation": {
            "scene_index": largest_jerk_index,
            "scene": scenes[largest_jerk_index],
            "log_name": log_names[largest_jerk_index],
            "per_scene_repeat_mean_delta_mps3": float(
                jerk_delta_per_scene[largest_jerk_index]
            ),
        },
        "representative_p90_dynamic_continuity_case": {
            "selection_rule": "closest per-scene centered-jerk delta to P90 among expert endpoint distance >= 10 m",
            "dynamic_scene_count": int(dynamic_mask.sum()),
            "target_p90_delta_mps3": dynamic_p90,
            "scene_index": representative_index,
            "scene": scenes[representative_index],
            "log_name": log_names[representative_index],
            "expert_endpoint_distance_m": float(endpoint_distance[representative_index]),
            "r0_centered_jerk_mps3": float(
                r0_centered["mean_jerk_mps3"][:, representative_index].mean()
            ),
            "c0_centered_jerk_mps3": float(
                c0_centered["mean_jerk_mps3"][:, representative_index].mean()
            ),
            "r0_minus_c0_centered_jerk_mps3": float(
                jerk_delta_per_scene[representative_index]
            ),
            "display_rule": "actual repeat whose paired jerk delta is closest to the scene's 3-repeat mean",
            **representative_plot,
        },
        "paired_distribution": {
            "ade_8s_per_scene_delta_mean_m": float(ade_delta_per_scene.mean()),
            "centered_jerk_per_scene_delta_mean_mps3": float(
                jerk_delta_per_scene.mean()
            ),
        },
    }
    (args.output_dir / "centered_kinematics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
