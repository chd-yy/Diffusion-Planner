"""Rebuild Chapter 4 evidence figures from the completed 306,801-scene run.

Read only. This script does not perform rollout, training, or simulation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/media/yanjun/拯救者PSSD/Diffusion-Planner-HDP")
RUN = EXTERNAL / "simulation_outputs/training_outputs/EXP_CH4_guided_safeset_full306801_seed2026_v3_1006"
REPLAY = RUN / "replay"
VALIDATION = EXTERNAL / "validation_outputs/EXP_CH4_full306801_vs_B10_validation1000x3_1006/B10_vs_full306801.json"
TEST14 = EXTERNAL / "test14_outputs/EXP_CH4_full306801_vs_historical_B10_1006"
EXTENDED = TEST14 / "extended_paired_analysis.json"
OUTPUT = PROJECT / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH4_full306801_posttraining/figures"

BLUE = "#215D88"
TEAL = "#138779"
RED = "#B94E45"
GREY = "#667482"
LIGHT = "#D9E1E7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_style() -> None:
    # The installed AR PL SungtiL GB is a Chinese Song-type serif font.
    plt.rcParams.update(
        {
            "font.family": "AR PL SungtiL GB",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def read_replay() -> dict[str, int]:
    index = json.loads((REPLAY / "index.json").read_text())
    args = json.loads((RUN / "run.json").read_text())["effective_args"]
    counts = {
        "scene_groups": 0,
        "stored_candidates": 0,
        "safe_candidates": 0,
        "task_preserving_candidates": 0,
        "active_groups": 0,
        "target_groups": 0,
        "regression_targets": 0,
        "shards": len(index["shard_lengths"]),
    }
    for shard_id in range(counts["shards"]):
        payload = torch.load(
            REPLAY / f"shard_{shard_id:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        rewards = payload["rewards"]
        safe = payload["candidate_masks"]
        task = payload["update_masks"]
        candidate_count = safe.sum(dim=1)
        denom = candidate_count.clamp_min(1).to(rewards.dtype)
        mean = torch.where(safe, rewards, torch.zeros_like(rewards)).sum(dim=1) / denom
        centered = torch.where(safe, rewards - mean[:, None], torch.zeros_like(rewards))
        std = torch.sqrt(centered.square().sum(dim=1) / denom)
        active = (
            payload["active_groups"].bool()
            & (candidate_count >= 2)
            & (std >= float(args["rl_min_reward_std"]))
        )
        advantage = (rewards - payload["reference_rewards"][:, None]) / (std[:, None] + 1e-6)
        targets = (
            safe
            & task
            & active[:, None]
            & (advantage >= float(args["rl_positive_advantage_margin"]))
        )
        counts["scene_groups"] += len(payload["scene_names"])
        counts["stored_candidates"] += rewards.numel()
        counts["safe_candidates"] += int(safe.sum())
        counts["task_preserving_candidates"] += int(task.sum())
        counts["active_groups"] += int(active.sum())
        counts["target_groups"] += int(targets.any(dim=1).sum())
        counts["regression_targets"] += int(targets.sum())
    counts["proposals_by_protocol"] = counts["scene_groups"] * (
        int(args["rl_guided_reference_count"])
        + int(args["rl_guided_policy_count"])
        + int(args["rl_guided_medium_count"])
        + int(args["rl_guided_count"])
    )
    assert counts["scene_groups"] == index["count"] == 306801
    assert counts["stored_candidates"] == 306801 * 32
    assert counts["active_groups"] == 37145
    assert counts["target_groups"] == 30159
    assert counts["regression_targets"] == 167730
    return counts


def save_figure(fig: plt.Figure, basename: str) -> None:
    fig.savefig(OUTPUT / f"{basename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_support(counts: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6), gridspec_kw={"wspace": 0.46})
    candidate_rows = [
        ("88条提案", counts["proposals_by_protocol"] / 1e6),
        ("32条入库", counts["stored_candidates"] / 1e6),
        ("硬安全资格", counts["safe_candidates"] / 1e6),
        ("任务保持掩码", counts["task_preserving_candidates"] / 1e6),
        ("最终RL目标", counts["regression_targets"] / 1e6),
    ]
    group_rows = [
        ("全部场景", counts["scene_groups"] / 1e3),
        ("有效激活组", counts["active_groups"] / 1e3),
        ("目标组", counts["target_groups"] / 1e3),
        ("实际覆盖目标组", counts["target_groups"] / 1e3),
    ]
    for axis, rows, xmax, xlabel, color in [
        (axes[0], candidate_rows, 30, "候选轨迹数 / 百万条", BLUE),
        (axes[1], group_rows, 340, "场景组数 / 千组", TEAL),
    ]:
        labels, values = zip(*rows)
        y = np.arange(len(rows))
        axis.barh(y, values, color=[GREY] + [color] * (len(rows) - 2) + [RED], height=0.58)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, xmax)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color=LIGHT, linewidth=0.6)
        axis.set_axisbelow(True)
        for yi, value in zip(y, values):
            axis.text(value + xmax * 0.015, yi, f"{value:.3f}", va="center", fontsize=9)
    fig.subplots_adjust(bottom=0.19)
    fig.text(0.28, 0.06, "（a）候选轨迹筛选", ha="center", fontsize=11)
    fig.text(0.75, 0.06, "（b）有效场景组覆盖", ha="center", fontsize=11)
    save_figure(fig, "全量后训练_候选与目标组筛选")


def plot_closed_loop(extended: dict) -> None:
    metrics = [
        ("score", "综合得分"),
        ("ego_progress_along_expert_route", "路线进度"),
        ("no_ego_at_fault_collisions", "无责任碰撞"),
        ("speed_limit_compliance", "限速合规"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 6.8))
    for axis, (key, label) in zip(axes.flat, metrics):
        for y, (benchmark, marker, color) in enumerate(
            [("test14-hard", "o", BLUE), ("test14-random", "s", TEAL)]
        ):
            value = extended["benchmarks"][benchmark]["metrics"][key]
            point = 100 * value["mean_delta"]
            lo, hi = (100 * v for v in value["log_cluster_percentile_interval_95"])
            axis.errorbar(
                point,
                y,
                xerr=[[point - lo], [hi - point]],
                fmt=marker,
                markersize=6,
                capsize=4,
                color=color,
                elinewidth=1.6,
            )
            axis.text(point, y - 0.18, f"{point:+.2f}", color=color, ha="center", fontsize=8)
        axis.axvline(0, color=GREY, linewidth=1, linestyle="--")
        axis.set_yticks([0, 1], ["Hard", "Random"])
        axis.set_ylim(1.5, -0.55)
        axis.set_xlabel("相对 B10 的均值差 / 百分点")
        axis.grid(axis="x", color=LIGHT, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_title(label, fontsize=11, pad=12)
    fig.subplots_adjust(hspace=0.46, wspace=0.36)
    save_figure(fig, "全量后训练_Test14闭环配对效应")


def plot_score_distribution() -> dict[str, dict[str, int]]:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharey=True)
    result = {}
    for axis, benchmark in zip(axes, ["test14-hard", "test14-random"]):
        source = json.loads((TEST14 / f"{benchmark}_historical_B10_vs_full306801.json").read_text())
        baseline = {row["scenario"]: row for row in source["historical_B10"]["scenarios"]}
        candidate = {row["scenario"]: row for row in source["full306801_posttrain"]["scenarios"]}
        assert baseline.keys() == candidate.keys()
        pairs = []
        for token in baseline:
            b, c = baseline[token], candidate[token]
            new_collision = b["no_ego_at_fault_collisions"] == 1 and c["no_ego_at_fault_collisions"] == 0
            pairs.append((c["score"] - b["score"], new_collision, token))
        pairs.sort(key=lambda item: item[0])
        deltas = np.array([item[0] for item in pairs])
        collisions = np.array([item[1] for item in pairs])
        x = np.arange(len(pairs))
        axis.scatter(x, deltas, s=10, color=BLUE, alpha=0.8, rasterized=True)
        axis.scatter(x[collisions], deltas[collisions], s=34, color=RED, marker="x", linewidth=1.2, label="新增责任碰撞")
        axis.axhline(0, color=GREY, linewidth=0.9, linestyle="--")
        axis.set_ylabel("逐场综合得分差")
        axis.set_xlabel("按得分差排序的场景")
        axis.set_title(f"{benchmark}（{len(pairs)}场）", fontsize=11)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        axis.grid(axis="y", color=LIGHT, linewidth=0.5)
        result[benchmark] = {
            "wins": int((deltas > 1e-12).sum()),
            "losses": int((deltas < -1e-12).sum()),
            "ties": int((abs(deltas) <= 1e-12).sum()),
            "new_collision_events": int(collisions.sum()),
        }
    fig.subplots_adjust(hspace=0.48)
    save_figure(fig, "全量后训练_Test14逐场得分分布")
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    counts = read_replay()
    validation = json.loads(VALIDATION.read_text())
    extended = json.loads(EXTENDED.read_text())
    assert not extended["promotion_authorized"]
    plot_support(counts)
    plot_closed_loop(extended)
    score_distribution = plot_score_distribution()
    output = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m月%d日%H时%M分"),
        "replay_counts": counts,
        "validation_protocol": validation["comparison_protocol"],
        "validation_key_deltas": {
            key: validation["paired_delta_rl_minus_supervised"][key]
            for key in ["reward", "progress", "no_collision", "drivable_area_compliance", "route_cost", "comfort_cost", "ade_m", "fde_m"]
        },
        "closed_loop": {
            benchmark: {
                "scenario_count": extended["benchmarks"][benchmark]["scenario_count"],
                "log_count": extended["benchmarks"][benchmark]["log_count"],
                "metrics": {
                    key: extended["benchmarks"][benchmark]["metrics"][key]
                    for key in ["score", "ego_progress_along_expert_route", "no_ego_at_fault_collisions", "speed_limit_compliance", "drivable_area_compliance"]
                },
                "score_distribution": score_distribution[benchmark],
            }
            for benchmark in ["test14-hard", "test14-random"]
        },
        "test14_overlap_tokens": extended["between_benchmark_overlap"],
        "source_sha256": {
            "replay_index": sha256(REPLAY / "index.json"),
            "run_metadata": sha256(RUN / "run.json"),
            "validation": sha256(VALIDATION),
            "closed_loop_extended": sha256(EXTENDED),
            "test14_hard_scenarios": sha256(TEST14 / "test14-hard_historical_B10_vs_full306801.json"),
            "test14_random_scenarios": sha256(TEST14 / "test14-random_historical_B10_vs_full306801.json"),
        },
    }
    (OUTPUT / "全量后训练_图表数据.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"replay_counts": counts, "score_distribution": score_distribution}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
