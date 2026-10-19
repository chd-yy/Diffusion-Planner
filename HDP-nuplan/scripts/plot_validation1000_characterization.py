#!/usr/bin/env python3
"""Characterize the frozen balanced validation-1000 cache and render a paper figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_length(xy: np.ndarray) -> float:
    origin = np.zeros((1, 2), dtype=xy.dtype)
    increments = np.diff(np.concatenate([origin, xy], axis=0), axis=0)
    return float(np.linalg.norm(increments, axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--sampling-report", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument(
        "--db-root",
        required=True,
        type=Path,
        help="Directory containing the source mini-val nuPlan SQLite logs",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    names = json.loads(args.manifest.read_text(encoding="utf-8"))
    sampling = json.loads(args.sampling_report.read_text(encoding="utf-8"))
    split_audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    records = []
    for name in names:
        data = np.load(args.cache_dir / name)
        neighbors = data["neighbor_agents_past"]
        current_valid = np.abs(neighbors[:, -1, :6]).sum(axis=-1) > 0
        records.append({
            "file": name,
            "map": str(data["map_name"]),
            "log": str(data["log_name"]),
            "scenario_type": str(data["scenario_type"]),
            "token": str(data["token"]),
            "expert_path_length_m": path_length(data["ego_agent_future"][:, :2]),
            "current_tracked_neighbors": int(current_valid.sum()),
        })

    maps = Counter(row["map"] for row in records)
    logs = Counter(row["log"] for row in records)
    types = Counter(row["scenario_type"] for row in records)
    paths = np.asarray([row["expert_path_length_m"] for row in records])
    neighbors = np.asarray([row["current_tracked_neighbors"] for row in records])

    # nuPlan distinguishes the official ~20 s `scene` rows from planning samples
    # anchored at individual 20 Hz lidar_pc tokens.  Reconstruct that hierarchy so
    # the report does not incorrectly call every expanded anchor an official scene.
    source_log_rows = []
    selected_parent_counts = Counter()
    selected_by_log = {
        log_name: {row["token"] for row in records if row["log"] == log_name}
        for log_name in sampling["log_names"]
    }
    for log_name in sampling["log_names"]:
        db_path = args.db_root / f"{log_name}.db"
        with sqlite3.connect(db_path) as connection:
            official_scene_count = connection.execute("SELECT COUNT(*) FROM scene").fetchone()[0]
            lidar_pc_count = connection.execute("SELECT COUNT(*) FROM lidar_pc").fetchone()[0]
            expanded_rows = connection.execute(
                """
                WITH ordered_scenes AS
                (
                    SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
                    FROM scene
                ),
                num_scenes AS
                (
                    SELECT COUNT(*) AS cnt FROM scene
                ),
                valid_scenes AS
                (
                    SELECT o.token
                    FROM ordered_scenes AS o
                    CROSS JOIN num_scenes AS n
                    WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
                )
                SELECT lp.token, lp.scene_token
                FROM lidar_pc AS lp
                INNER JOIN valid_scenes AS vs ON lp.scene_token = vs.token
                """
            ).fetchall()

        token_to_parent = {bytes(token).hex(): bytes(parent).hex() for token, parent in expanded_rows}
        selected_tokens = selected_by_log[log_name]
        missing_tokens = selected_tokens - token_to_parent.keys()
        if missing_tokens:
            raise RuntimeError(f"{log_name}: {len(missing_tokens)} selected tokens missing from source DB")
        parent_counts = Counter(token_to_parent[token] for token in selected_tokens)
        for parent, count in parent_counts.items():
            selected_parent_counts[(log_name, parent)] = count
        source_log_rows.append({
            "log_name": log_name,
            "official_scene_count": official_scene_count,
            "sdk_valid_scene_count": len({bytes(parent).hex() for _, parent in expanded_rows}),
            "all_lidar_pc_count": lidar_pc_count,
            "expanded_anchor_sample_count": len(expanded_rows),
            "selected_anchor_sample_count": len(selected_tokens),
            "represented_official_scene_count": len(parent_counts),
        })

    anchors_per_parent = np.asarray(list(selected_parent_counts.values()))

    semantic_groups = {
        "Stationary / queued": [
            "stationary", "stationary_in_traffic",
            "stationary_at_traffic_light_without_lead",
        ],
        "Traversal": [
            "traversing_traffic_light_intersection", "traversing_pickup_dropoff",
            "traversing_intersection", "traversing_crosswalk",
        ],
        "Speed magnitude": [
            "low_magnitude_speed", "medium_magnitude_speed", "high_magnitude_speed",
        ],
        "Following / proximity": [
            "following_lane_with_slow_lead", "following_lane_without_lead",
            "near_long_vehicle", "near_high_speed_vehicle", "near_multiple_vehicles",
            "near_pedestrian_on_crosswalk", "near_construction_zone_sign",
        ],
        "Control / stopping": [
            "on_stopline_stop_sign", "on_stopline_traffic_light", "on_pickup_dropoff",
            "on_intersection", "stopping_with_lead",
            "stopping_at_traffic_light_without_lead", "stopping_at_crosswalk",
        ],
        "Unknown": ["unknown"],
    }
    grouped_types = {
        group: sum(types[name] for name in members)
        for group, members in semantic_groups.items()
    }
    if sum(grouped_types.values()) != len(records):
        raise RuntimeError("scenario grouping does not cover validation-1000 exactly")

    motion_labels = ["<=2", "2-10", "10-20", "20-40", ">40"]
    motion_counts = [
        int(np.sum(paths <= 2)),
        int(np.sum((paths > 2) & (paths <= 10))),
        int(np.sum((paths > 10) & (paths <= 20))),
        int(np.sum((paths > 20) & (paths <= 40))),
        int(np.sum(paths > 40)),
    ]
    neighbor_labels = ["0", "1-5", "6-10", "11-20", "21-32"]
    neighbor_counts = [
        int(np.sum(neighbors == 0)),
        int(np.sum((neighbors >= 1) & (neighbors <= 5))),
        int(np.sum((neighbors >= 6) & (neighbors <= 10))),
        int(np.sum((neighbors >= 11) & (neighbors <= 20))),
        int(np.sum(neighbors >= 21)),
    ]

    log_rows = []
    for index, log_name in enumerate(sampling["log_names"], start=1):
        maps_for_log = {row["map"] for row in records if row["log"] == log_name}
        if len(maps_for_log) != 1:
            raise RuntimeError(f"log {log_name} maps are ambiguous: {maps_for_log}")
        log_rows.append({
            "id": f"L{index}",
            "log_name": log_name,
            "map": next(iter(maps_for_log)),
            "count": logs[log_name],
        })

    report = {
        "dataset": "mini_val_balanced_1000_seed3407_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "selection": {
            "source_expanded_anchor_samples": sampling["raw_scenarios"],
            "strategy": sampling["strategy"],
            "seed": sampling["seed"],
            "selected_anchor_samples": len(records),
            "logs": len(logs),
            "per_log": sorted(set(logs.values())),
        },
        "official_scene_hierarchy": {
            "downloaded_official_scenes": sum(row["official_scene_count"] for row in source_log_rows),
            "sdk_valid_official_scenes": sum(row["sdk_valid_scene_count"] for row in source_log_rows),
            "all_lidar_pc_rows": sum(row["all_lidar_pc_count"] for row in source_log_rows),
            "expanded_anchor_samples": sum(row["expanded_anchor_sample_count"] for row in source_log_rows),
            "selected_anchor_samples": len(records),
            "represented_official_scenes": len(selected_parent_counts),
            "anchors_per_represented_scene": {
                "min": int(anchors_per_parent.min()),
                "p10": float(np.quantile(anchors_per_parent, 0.1)),
                "p50": float(np.quantile(anchors_per_parent, 0.5)),
                "p90": float(np.quantile(anchors_per_parent, 0.9)),
                "max": int(anchors_per_parent.max()),
                "mean": float(anchors_per_parent.mean()),
            },
            "per_log": source_log_rows,
        },
        "split_audit": split_audit,
        "map_counts": dict(sorted(maps.items())),
        "log_rows": log_rows,
        "scenario_type_counts": dict(types.most_common()),
        "scenario_group_counts": grouped_types,
        "expert_path_length_m": {
            "p10": float(np.quantile(paths, 0.1)),
            "p50": float(np.quantile(paths, 0.5)),
            "p90": float(np.quantile(paths, 0.9)),
            "bins": dict(zip(motion_labels, motion_counts)),
        },
        "current_tracked_neighbors": {
            "p10": float(np.quantile(neighbors, 0.1)),
            "p50": float(np.quantile(neighbors, 0.5)),
            "p90": float(np.quantile(neighbors, 0.9)),
            "capped_at": 32,
            "bins": dict(zip(neighbor_labels, neighbor_counts)),
        },
    }

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)

    map_labels = {
        "us-nv-las-vegas-strip": "Las Vegas",
        "sg-one-north": "Singapore",
        "us-ma-boston": "Boston",
        "us-pa-pittsburgh-hazelwood": "Pittsburgh",
    }
    ordered_maps = sorted(maps, key=lambda name: (-maps[name], name))
    values = [maps[name] for name in ordered_maps]
    bars = axes[0, 0].bar([map_labels.get(name, name) for name in ordered_maps], values,
                          color="#0072b2")
    axes[0, 0].set(title="(a) Map distribution", ylabel="Anchor samples", ylim=(0, 800))
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].bar_label(bars)

    log_ids = [row["id"] for row in log_rows]
    log_values = [row["count"] for row in log_rows]
    bars = axes[0, 1].bar(log_ids, log_values, color="#009e73")
    axes[0, 1].set(title="(b) Log-balanced sampling", xlabel="Validation log",
                   ylabel="Anchor samples", ylim=(0, 125))
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].bar_label(bars, fontsize=8)

    group_names = list(grouped_types)
    group_values = [grouped_types[name] for name in group_names]
    short_groups = ["Stationary / queued", "Traversal", "Speed", "Following / near",
                    "Control / stop", "Unknown"]
    bars = axes[1, 0].barh(short_groups[::-1], group_values[::-1], color="#d55e00")
    axes[1, 0].set(title="(c) Anchor-label composition", xlabel="Anchor samples", xlim=(0, 520))
    axes[1, 0].grid(axis="x", alpha=0.25)
    axes[1, 0].bar_label(bars, padding=3)

    bars = axes[1, 1].bar(motion_labels, motion_counts, color="#cc79a7")
    axes[1, 1].set(title="(d) Expert 8-s path length", xlabel="Path length (m)",
                   ylabel="Anchor samples", ylim=(0, 500))
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].bar_label(bars)

    fig.suptitle("Composition of the frozen validation-1000 evaluation set", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plt.close(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
