#!/usr/bin/env python3
"""Regenerate two Chapter-3 representation figures without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from fontTools.ttLib import TTCollection

from recalculate_chapter3_kinematics_centered import (
    plot_representative_continuity_case,
    plot_scene_context_hd,
)


FONT_FAMILY = "Noto Serif CJK SC"
FONT_COLLECTION = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
FONT_COLLECTION_SC_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--centered-output-dir", type=Path)
    return parser.parse_args()


def plot_horizon_error(data: np.lib.npyio.NpzFile, output: Path) -> None:
    horizons = (1, 3, 5, 8)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "ade", "平均位移误差（ADE）"),
        (axes[1], "fde", "终点位移误差（FDE）"),
    ):
        for key_prefix, label, marker in (
            ("r0", "Abs模型输出轨迹", "o"),
            ("c0", "Delta模型输出轨迹", "s"),
        ):
            values = [data[f"{key_prefix}__{prefix}_{h}s_m"].mean() for h in horizons]
            axis.plot(horizons, values, marker=marker, linewidth=2, label=label)
        axis.set_title(title)
        axis.set_xlabel("预测时域（s）")
        axis.set_ylabel("误差（m）")
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_paired_distributions(data: np.lib.npyio.NpzFile, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    pairs = (
        ("ade_8s_m", "Abs − Delta 8 s ADE（m）"),
        ("mean_jerk_mps3", "Abs − Delta 平均jerk代理（m/s³）"),
    )
    for axis, (metric, label) in zip(axes, pairs):
        delta = (data[f"r0__{metric}"] - data[f"c0__{metric}"]).reshape(-1)
        axis.hist(delta, bins=50, color="#4c78a8", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(
            delta.mean(),
            color="#e45756",
            linestyle="--",
            label=f"均值={delta.mean():.3f}",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("场景数")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def centered_mean_jerk(xy: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Match the position-only centered third difference used in the audit."""

    jerk = (
        xy[..., 4:, :]
        - 2.0 * xy[..., 3:-1, :]
        + 2.0 * xy[..., 1:-3, :]
        - xy[..., :-4, :]
    ) / (2.0 * dt**3)
    return np.linalg.norm(jerk, axis=-1).mean(axis=-1)


def plot_centered_paired_distributions(
    data: np.lib.npyio.NpzFile,
    output: Path,
) -> None:
    ade_delta = (data["r0__ade_8s_m"] - data["c0__ade_8s_m"]).mean(axis=0)
    r0_xy = data["r0_trajectories"][..., :2].astype(np.float64)
    c0_xy = data["c0_trajectories"][..., :2].astype(np.float64)
    jerk_delta = (centered_mean_jerk(r0_xy) - centered_mean_jerk(c0_xy)).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    panels = (
        (ade_delta, "Abs−Delta逐场平均8秒ADE（m）"),
        (jerk_delta, "Abs−Delta逐场平均中心差分jerk（m/s³）"),
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
        axis.set_ylabel("场景数")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.centered_output_dir is not None:
        args.centered_output_dir.mkdir(parents=True, exist_ok=True)
    if not FONT_COLLECTION.is_file():
        raise FileNotFoundError(f"missing Song-style CJK font: {FONT_COLLECTION}")
    # Matplotlib does not honor the fontconfig face index of a TTC collection.
    # Extract its Simplified-Chinese face temporarily, register that exact face,
    # and keep it alive until both figures have been rendered.
    with tempfile.TemporaryDirectory(prefix="chapter3_songti_") as temporary_dir:
        extracted_font = Path(temporary_dir) / "NotoSerifCJKsc-Regular.otf"
        collection = TTCollection(FONT_COLLECTION)
        collection.fonts[FONT_COLLECTION_SC_INDEX].save(extracted_font)
        resolved_family = font_manager.FontProperties(fname=extracted_font).get_name()
        if resolved_family != FONT_FAMILY:
            raise RuntimeError(
                f"unexpected extracted font family: {resolved_family!r}"
            )
        font_manager.fontManager.addfont(extracted_font)
        plt.rcParams.update(
            {
                "font.family": FONT_FAMILY,
                "axes.unicode_minus": False,
            }
        )
        with np.load(args.input, allow_pickle=False) as data:
            plot_horizon_error(
                data,
                args.output_dir / "horizon_error_comparison_songti_abs_delta.png",
            )
            plot_paired_distributions(
                data,
                args.output_dir / "paired_delta_distributions_songti_abs_delta.png",
            )
            if args.centered_output_dir is not None:
                plot_centered_paired_distributions(
                    data,
                    args.centered_output_dir
                    / "paired_delta_distributions_centered_songti_abs_delta.png",
                )
                summary = json.loads(
                    (args.centered_output_dir / "centered_kinematics_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                case = summary["representative_p90_dynamic_continuity_case"]
                scene_order = json.loads(
                    Path(summary["inputs"]["scene_order"]).read_text(encoding="utf-8")
                )
                scene_index = int(case["scene_index"])
                repeat_index = int(case["repeat_index_zero_based"])
                r0_xy = data["r0_trajectories"][repeat_index, scene_index, :, :2].astype(
                    np.float64
                )
                c0_xy = data["c0_trajectories"][repeat_index, scene_index, :, :2].astype(
                    np.float64
                )
                expert_xy = data["expert_trajectories"][scene_index, :, :2].astype(
                    np.float64
                )
                scene = scene_order[scene_index]
                plot_representative_continuity_case(
                    r0_xy,
                    c0_xy,
                    expert_xy,
                    scene,
                    float(summary["method"]["dt_s"]),
                    args.centered_output_dir
                    / "representative_p90_continuity_case_songti_abs_delta.png",
                    repeat_index=repeat_index,
                    per_scene_r0_jerk=float(case["r0_centered_jerk_mps3"]),
                    per_scene_c0_jerk=float(case["c0_centered_jerk_mps3"]),
                    publication_style=True,
                )
                plot_scene_context_hd(
                    Path(summary["inputs"]["data_dir"]) / scene,
                    r0_xy,
                    c0_xy,
                    expert_xy,
                    args.centered_output_dir
                    / "representative_p90_scene_context_hd_songti_abs_delta.png",
                    repeat_index=repeat_index,
                    publication_style=True,
                )


if __name__ == "__main__":
    main()
