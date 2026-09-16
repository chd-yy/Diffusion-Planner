"""Extract official collision/road-departure timing from trusted local traces."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    Collisions, EgoAtFaultCollisionStatistics, classify_at_fault_collisions,
)
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_corners


def analyze(path):
    log = SimulationLog.load_data(Path(path))
    history, scenario = log.simulation_history, log.scenario
    states = history.extract_ego_state
    t0 = states[0].time_us
    lane = EgoLaneChangeStatistics("ego_lane_change", "Planning", 0.3)
    lane.compute(history, scenario)
    collision = EgoAtFaultCollisionStatistics("no_ego_at_fault_collisions", "Dynamics", lane)
    collision.compute(history, scenario)
    road = DrivableAreaComplianceStatistics("drivable_area_compliance", "Planning", lane, 0.3)
    road.compute(history, scenario)
    first_departure = None
    for state, corners, routes, center_route in zip(
            states, extract_ego_corners(states), lane.corners_route, lane.ego_driven_route):
        _, violation = road.compute_violation_for_iteration(history.map_api, corners, routes, center_route, False)
        if violation:
            first_departure = (state.time_us - t0) / 1e6
            break
    speeds = [float(s.dynamic_car_state.speed) for s in states]
    positions = np.array([[s.rear_axle.x, s.rear_axle.y] for s in states])
    events = []
    for event in collision.all_collisions:
        index = next(i for i, s in enumerate(states) if s.time_us == event.timestamp)
        for token, data in event.collisions_id_data.items():
            fault_times, _ = classify_at_fault_collisions(
                [Collisions(event.timestamp, {token: data})],
                lane.timestamps_in_common_or_connected_route_objs,
            )
            events.append(dict(time_s=(event.timestamp - t0) / 1e6, track=token,
                               collision_type=data.collision_type.name,
                               object_type=data.tracked_object_type.name,
                               ego_speed_mps=speeds[index],
                               at_fault=bool(fault_times)))
    return dict(token=scenario.token, trace=path, frames=len(states),
                duration_s=(states[-1].time_us - t0) / 1e6,
                collision_score=float(collision.results[0].metric_score),
                drivable_score=float(road.results[0].metric_score),
                first_drivable_violation_s=first_departure, collisions=events,
                distance_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                max_speed_mps=max(speeds), final_speed_mps=speeds[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.diagnosis.read_text())
    results = {"created_at": datetime.now().strftime("%m月%d日 %H:%M"), "runs": {}}
    for name, run in manifest["runs"].items():
        rows = []
        for trace in run["traces"]:
            row = analyze(trace)
            expected = next(r for r in run["summary"]["scenarios"] if r["scenario"] == row["token"])
            for source, key in [("collision_score", "no_ego_at_fault_collisions"),
                                ("drivable_score", "drivable_area_compliance")]:
                if not np.isclose(row[source], expected[key]):
                    raise RuntimeError(f"Recomputed official metric mismatch: {name} {row['token']} {key}")
            rows.append(row)
            print(name, json.dumps(row, ensure_ascii=False), flush=True)
        results["runs"][name] = rows
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
