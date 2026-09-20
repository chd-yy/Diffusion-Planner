"""Freeze a model-result-independent, temporally separated safety development set.

Use the same eight held-out mini logs as Fixed200, including additional tagged
anchors when Fixed200's adjacent-frame sampling cannot satisfy time separation.
Official canonical type and the selection tag are recorded separately.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
HDP = ROOT / "HDP-nuplan"
DEVKIT = ROOT.parent / "nuplan-devkit"
MINI = ROOT.parent / "nuplan/dataset/nuplan-v1.1_mini/data/cache/mini"
SOURCE = HDP / "tmp/mini_train_balanced_10000_seed3407_v1/mini_val_fixed_200_rl_v4_manifest.json"
REPLAY = HDP / "doc_hdp_nuplan/phd_midterm/stage_results/EXP003_replay_coverage/replay_coverage.json"
DEST = HDP / "benchmarks/midterm_safety40_v1"
MINIMUM_ANCHOR_SEPARATION_S = 15
GROUPS = {
    "intersection": {"count": 12, "types": {
        "starting_unprotected_noncross_turn": 4, "traversing_intersection": 2,
        "traversing_traffic_light_intersection": 2, "on_traffic_light_intersection": 1,
        "on_intersection": 1, "starting_straight_stop_sign_intersection_traversal": 2}},
    "vehicle_interaction": {"count": 10, "types": {
        "following_lane_with_slow_lead": 3, "near_multiple_vehicles": 3,
        "near_high_speed_vehicle": 2, "near_long_vehicle": 2}},
    "vru_rules": {"count": 8, "types": {
        "near_pedestrian_on_crosswalk": 3, "traversing_crosswalk": 2,
        "on_stopline_crosswalk": 1, "stopping_with_lead": 1,
        "stopping_at_stop_sign_without_lead": 1}},
    "dynamics_environment": {"count": 6, "types": {
        "high_lateral_acceleration": 3, "high_magnitude_speed": 1,
        "near_construction_zone_sign": 2}},
    "basic_preservation": {"count": 4, "types": {
        "following_lane_without_lead": 1, "medium_magnitude_speed": 1,
        "stationary": 1, "stationary_in_traffic": 1}},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tie_key(token):
    return hashlib.sha256(("midterm-safety40-v1:" + token).encode()).hexdigest()


def load_candidates():
    sys.path.insert(0, str(DEVKIT))
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
    source = json.loads(SOURCE.read_text())
    replay = json.loads(REPLAY.read_text())
    logs = sorted(source["counts_by_log"])
    if len(logs) != 8 or set(logs) & set(replay["full"]["logs"]):
        raise ValueError("Require the same eight non-training logs")
    tags_allowed = {t: g for g, d in GROUPS.items() for t in d["types"]}
    connections, candidates, identities, counts = {}, [], {}, {}
    for log in logs:
        path = MINI / (log + ".db")
        connections[log] = conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        valid = {r["token"].hex(): dict(scenario=r["token"].hex(), timestamp_us=r["timestamp"],
                 log_name=log, map_name=r["map_name"], scenario_type=r["scenario_type"] or "unknown")
                 for r in get_scenarios_from_db(str(path), None, None, None, False)}
        tags = defaultdict(set)
        for r in conn.execute("SELECT lidar_pc_token,type FROM scenario_tag"):
            tags[r[0].hex()].add(r[1])
        buckets = defaultdict(list)
        for token, row in valid.items():
            for tag in tags[token] & tags_allowed.keys():
                buckets[(tag, row["timestamp_us"] // 5_000_000)].append(token)
        for (tag, bucket), tokens in sorted(buckets.items()):
            token = min(tokens, key=tie_key)
            candidates.append(dict(valid[token], selection_tag=tag, group=tags_allowed[tag],
                                   all_tags=sorted(tags[token])))
        counts[log] = dict(valid_anchors=len(valid), tag_time_buckets=len(buckets))
        identities[str(path)] = sha(path)
        print(f"[{datetime.now():%H:%M}] {log}: {len(buckets)} candidate tag/buckets", flush=True)
    return candidates, connections, identities, counts


def select_candidates(rows):
    """MILP: exact major quotas, 15s separation, 1--8 per log; soft fine quotas.

    Fine quota deviations dominate log balance, which dominates hash tie-breaks.
    No planner output, metric, or score is an input.
    """
    types = [t for d in GROUPS.values() for t in d["types"]]
    targets = {t: n for d in GROUPS.values() for t, n in d["types"].items()}
    logs = sorted({r["log_name"] for r in rows})
    n = len(rows)
    width = n + len(types) + len(logs)
    coefficients, lower, upper = [], [], []

    def add(values, lo, hi):
        coefficients.append(values); lower.append(lo); upper.append(hi)

    for group, spec in GROUPS.items():
        add({i: 1 for i, r in enumerate(rows) if r["group"] == group}, spec["count"], spec["count"])
    # The intersection group is explicitly intended to probe traversal,
    # turning and progress.  A tag such as on_intersection can also be attached
    # to a fully stationary anchor; those anchors belong in the separately
    # reserved basic-preservation group and do not satisfy this group's role.
    add(
        {
            i: 1
            for i, r in enumerate(rows)
            if r["group"] == "intersection"
            and r["scenario_type"] in {"stationary", "stationary_in_traffic"}
        },
        0,
        0,
    )
    for k, typ in enumerate(types):
        entries = {i: 1 for i, r in enumerate(rows) if r["selection_tag"] == typ}
        # At least one example per selected tag; at most four, except unavoidable deviations rejected.
        add(entries, 1, 4)
        add(dict(entries, **{}) | {n + k: -1}, -np.inf, targets[typ])
        add({i: -v for i, v in entries.items()} | {n + k: -1}, -np.inf, -targets[typ])
    for k, log in enumerate(logs):
        entries = {i: 1 for i, r in enumerate(rows) if r["log_name"] == log}
        add(entries, 1, 8)
        add(entries | {n + len(types) + k: -1}, -np.inf, 5)
        add({i: -v for i, v in entries.items()} | {n + len(types) + k: -1}, -np.inf, -5)
        ordered = sorted(entries, key=lambda i: (rows[i]["timestamp_us"], rows[i]["scenario"], rows[i]["selection_tag"]))
        # A sliding interval constraint also prevents selecting the same token under two tags.
        for offset, i in enumerate(ordered):
            nearby = {j: 1 for j in ordered[offset:]
                      if rows[j]["timestamp_us"] - rows[i]["timestamp_us"]
                      < MINIMUM_ANCHOR_SEPARATION_S * 1_000_000}
            if len(nearby) > 1: add(nearby, 0, 1)
    matrix = lil_matrix((len(coefficients), width))
    for i, values in enumerate(coefficients):
        for j, v in values.items(): matrix[i, j] = v
    # Candidate-level costs only break ties between solutions with the same
    # major quotas, fine-tag deviation and log balance.  Avoid filling an
    # interaction group with nominally stationary anchors when active anchors
    # are available; stationary behavior remains explicitly represented by
    # the basic-preservation quota.  This uses recorded scenario metadata, not
    # planner scores or outcomes.
    candidate_cost = [
        int(tie_key(r["scenario"] + r["selection_tag"])[:12], 16) / 16**12 * 0.001
        + (
            1.0
            if r["group"] != "basic_preservation"
            and r["scenario_type"] in {"stationary", "stationary_in_traffic"}
            else 0.0
        )
        for r in rows
    ]
    cost = np.array(candidate_cost + [1000.] * len(types) + [10.] * len(logs))
    result = milp(cost, integrality=np.r_[np.ones(n), np.zeros(width - n)],
                  bounds=Bounds(np.zeros(width), np.r_[np.ones(n), np.full(width - n, np.inf)]),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 60, "mip_rel_gap": 0.00001})
    if result.x is None:
        raise ValueError(f"No valid 40-set: {result.message}; do not silently weaken constraints")
    selected = [r for r, x in zip(rows, result.x[:n]) if x > .5]
    validate_selection(selected)
    actual = Counter(r["selection_tag"] for r in selected)
    return selected, dict(solver_status=int(result.status), message=result.message,
                         optimal=bool(result.success), objective=float(result.fun),
                         requested_tag_counts=targets, actual_tag_counts=dict(actual),
                         tag_quota_adjustments={t: dict(requested=targets[t], actual=actual[t])
                                                for t in types if targets[t] != actual[t]})


def validate_selection(rows):
    if len(rows) != 40 or len({r["scenario"] for r in rows}) != 40:
        raise ValueError("Need 40 unique tokens")
    if Counter(r["group"] for r in rows) != Counter({g: s["count"] for g, s in GROUPS.items()}):
        raise ValueError("Major group quota mismatch")
    logs = Counter(r["log_name"] for r in rows)
    if len(logs) != 8 or min(logs.values()) < 1 or max(logs.values()) > 8:
        raise ValueError("Log coverage/concentration constraint violated")
    for i, r in enumerate(rows):
        if r["selection_tag"] not in r["all_tags"]:
            raise ValueError("Selection tag missing from original annotations")
        for s in rows[:i]:
            if (
                r["log_name"] == s["log_name"]
                and abs(r["timestamp_us"] - s["timestamp_us"])
                < MINIMUM_ANCHOR_SEPARATION_S * 1_000_000
            ):
                raise ValueError("Adjacent/overlapping event anchors")


def context_features(row, conn):
    """Audit recorded context over actual simulation window [-3,+12]s at ~1Hz.

    Distances are center/rear-axle geometric descriptors, NOT collision metrics.
    Ego yaw/speed/lateral acceleration are descriptors, not feasibility proofs.
    """
    start = row["timestamp_us"] - 3_000_000
    frames = conn.execute("""SELECT p.token,p.timestamp,e.x,e.y,e.qw,e.qx,e.qy,e.qz,e.vx,e.vy
        FROM lidar_pc p JOIN ego_pose e ON e.token=p.ego_pose_token
        WHERE p.timestamp BETWEEN ? AND ? ORDER BY p.timestamp""", (start, start + 15_000_000)).fetchall()
    sampled, last = [], -float("inf")
    for frame in frames:
        if frame["timestamp"] - last >= 950_000:
            sampled.append(frame); last = frame["timestamp"]
    if len(sampled) < 14:
        raise ValueError(f"Insufficient recorded context {row['scenario']}")
    speeds = np.array([math.hypot(f["vx"], f["vy"]) for f in sampled])
    yaw = np.unwrap([math.atan2(2*(f["qw"]*f["qz"]+f["qx"]*f["qy"]),
                               1-2*(f["qy"]**2+f["qz"]**2)) for f in sampled])
    times = np.array([f["timestamp"] for f in sampled]) / 1e6
    distances = defaultdict(list); max_near_vehicles = 0; max_vehicle_speed = 0.; max_vehicle_length = 0.
    ped_front_min = float("inf"); closing_max = 0.; aligned_lead_min = float("inf")
    world_vx = np.gradient([f["x"] for f in sampled], times)
    world_vy = np.gradient([f["y"] for f in sampled], times)
    for k, f in enumerate(sampled):
        objects = conn.execute("""SELECT b.x,b.y,b.vx,b.vy,b.length,c.name FROM lidar_box b
            JOIN track t ON t.token=b.track_token JOIN category c ON c.token=t.category_token
            WHERE b.lidar_pc_token=?""", (f["token"],)).fetchall()
        near = 0
        for obj in objects:
            dx, dy = obj["x"] - f["x"], obj["y"] - f["y"]
            distance = math.hypot(dx, dy)
            distances[obj["name"]].append(distance)
            longitudinal = dx*math.cos(yaw[k]) + dy*math.sin(yaw[k])
            lateral = -dx*math.sin(yaw[k]) + dy*math.cos(yaw[k])
            if obj["name"] == "vehicle" and distance <= 25:
                near += 1
                max_vehicle_speed = max(max_vehicle_speed, math.hypot(obj["vx"], obj["vy"]))
                max_vehicle_length = max(max_vehicle_length, obj["length"])
                closing = -(dx*(obj["vx"]-world_vx[k]) + dy*(obj["vy"]-world_vy[k])) / max(distance, .001)
                closing_max = max(closing_max, closing)
                if 0 < longitudinal <= 25 and abs(lateral) < 2.5:
                    aligned_lead_min = min(aligned_lead_min, longitudinal)
            if obj["name"] == "pedestrian" and -3 <= longitudinal <= 25 and abs(lateral) <= 5:
                ped_front_min = min(ped_front_min, distance)
        max_near_vehicles = max(max_near_vehicles, near)
    def finite_min(name):
        return min(distances[name]) if distances[name] else None
    return dict(window="anchor-3s to anchor+12s", sampled_frames=len(sampled),
        timestamp_first_us=sampled[0]["timestamp"], timestamp_last_us=sampled[-1]["timestamp"],
        expert_speed_min_mps=float(speeds.min()), expert_speed_max_mps=float(speeds.max()),
        expert_path_length_m=float(sum(math.hypot(b["x"]-a["x"],b["y"]-a["y"]) for a,b in zip(sampled,sampled[1:]))),
        expert_heading_range_deg=float(np.ptp(yaw)*180/math.pi),
        expert_lateral_acc_proxy_max_mps2=float(np.max(np.abs(np.gradient(yaw,times)*speeds))),
        max_vehicles_within_25m=max_near_vehicles, min_vehicle_distance_m=finite_min("vehicle"),
        min_pedestrian_distance_m=finite_min("pedestrian"),
        min_front_corridor_pedestrian_distance_m=ped_front_min if math.isfinite(ped_front_min) else None,
        min_front_corridor_vehicle_longitudinal_m=aligned_lead_min if math.isfinite(aligned_lead_min) else None,
        max_near_vehicle_speed_mps=max_vehicle_speed, max_near_vehicle_length_m=max_vehicle_length,
        max_near_vehicle_radial_closing_mps=closing_max, min_construction_sign_distance_m=finite_min("czone_sign"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    cli = parser.parse_args()
    output = cli.output.resolve()
    if output.exists(): raise ValueError("Frozen output already exists; never regenerate in place")
    candidates, connections, db_ids, counts = load_candidates()
    selected, solver = select_candidates(candidates)
    order = {g: i for i, g in enumerate(GROUPS)}
    selected.sort(key=lambda r: (order[r["group"]], r["selection_tag"], r["log_name"], r["timestamp_us"]))
    for row in selected:
        row["context"] = context_features(row, connections[row["log_name"]])
        print(f"Context checked: {row['selection_tag']} {row['scenario']}", flush=True)
    train = json.loads(REPLAY.read_text())
    names = json.loads(Path(train["manifest"]).read_text())
    overlap = {Path(n).stem.rsplit("_",1)[-1] for n in names} & {r["scenario"] for r in selected}
    if overlap: raise ValueError("Training token overlap")
    output.mkdir(parents=True)
    manifest = dict(benchmark_id="midterm_safety40_v1", created_at=datetime.now().strftime("%m月%d日 %H:%M"),
        scenario_count=40, scenario_tokens=[r["scenario"] for r in selected], scenarios=selected,
        counts_by_group=dict(Counter(r["group"] for r in selected)),
        counts_by_type=dict(Counter(r["scenario_type"] for r in selected)),
        counts_by_selection_tag=dict(Counter(r["selection_tag"] for r in selected)),
        counts_by_log=dict(Counter(r["log_name"] for r in selected)),
        minimum_anchor_separation_s=MINIMUM_ANCHOR_SEPARATION_S,
        selection="Same eight validation logs; original tags; 5s tag/time bucket hash-thinning; MILP with 15s separation; no planner metrics",
        stationary_tie_break=(
            "A unit penalty is applied to canonical stationary/stationary_in_traffic "
            "anchors outside basic_preservation. This is below log-balance (10) and "
            "fine-tag-deviation (1000) costs, so it only prefers active anchors among "
            "otherwise equivalent feasible selections."
        ),
        intersection_motion_gate=(
            "The intersection group excludes canonical stationary and "
            "stationary_in_traffic anchors because its declared role is "
            "turning/traversal/progress; stationary capability remains in the "
            "basic_preservation quota."
        ),
        log_coverage_rationale=(
            "All eight logs remain represented, with 1--8 anchors per log. "
            "The Boston log has only stationary anchors under the predeclared "
            "22 selection tags; requiring two anchors there conflicts with the "
            "intersection motion gate, so it contributes one explicit basic-"
            "preservation anchor rather than duplicated stationary intersection "
            "anchors."
        ),
        separation_rationale=(
            "The recorded audit window is anchor-3s to anchor+12s (15s). "
            "The original 20s constraint was infeasible because the sole "
            "on_stopline_crosswalk anchor is 18.751s from one of only two "
            "starting_unprotected_noncross_turn anchors, while the other turn "
            "anchor is 3.399s from the sole stopping_at_stop_sign_without_lead "
            "anchor. 15s preserves non-overlapping audit windows without "
            "dropping a predeclared scenario tag."
        ),
        source_expansion_reason="Fixed200 cannot satisfy fine quotas with temporal separation; expand within same held-out logs",
        solver=solver, input_sha256={str(SOURCE):sha(SOURCE), str(REPLAY):sha(REPLAY),
        str(Path(train["manifest"])):sha(train["manifest"]), str(Path(__file__).resolve()):sha(__file__), **db_ids},
        candidate_counts=counts, training_token_overlap=0, training_log_overlap=0,
        limitations=["Challenge-stratified development set, not a blind or population-representative test",
                      "Raw recorded context audit is not a formal safety or difficulty label",
                      "Single training/inference seed; correlated logs; not a safety guarantee",
                      "Selection tag may differ from official canonical scenario type"],
        future_default=True, changing_tokens_requires_user_confirmation=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "candidate_pool.json", candidates)
    source_filter = json.loads(json.dumps(__import__('yaml').safe_load((HDP / 'hdp_nuplan/config/scenario_filter/mini-val-fixed-200-rl-v4.yaml').read_text())))
    source_filter["scenario_tokens"] = manifest["scenario_tokens"]
    write_json(output / "midterm-safety40-v1.yaml", source_filter)
    write_json(output / "SHA256.json", {p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(output=str(output), groups=manifest["counts_by_group"], logs=manifest["counts_by_log"],
                         adjustments=solver["tag_quota_adjustments"]), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
