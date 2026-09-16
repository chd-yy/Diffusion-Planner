"""Compare historical and corrected reward boxes at actual collision frames."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from hdp_nuplan.rl.reward import NuPlanRewardConfig, NuPlanTensorRewardScorer


def separation(scorer, ego, obj):
    pose = ego.rear_axle
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    return scorer._rectangle_signed_separation(
        tensor([[[[pose.x, pose.y]]]]), tensor([[[[math.cos(pose.heading), math.sin(pose.heading)]]]]),
        tensor([[[[obj.center.x, obj.center.y]]]]),
        tensor([[[[math.cos(obj.center.heading), math.sin(obj.center.heading)]]]]),
        tensor([[obj.box.width]]), tensor([[obj.box.length]]),
    ).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_analysis", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v = get_pacifica_parameters()
    legacy = NuPlanTensorRewardScorer(NuPlanRewardConfig())
    corrected = NuPlanTensorRewardScorer(NuPlanRewardConfig(
        ego_width=v.width, ego_length=v.length, ego_rear_axle_to_center=v.rear_axle_to_center))
    result = dict(created_at=datetime.now().strftime("%m月%d日 %H:%M"), events=[])
    for name, rows in json.loads(args.trace_analysis.read_text())["runs"].items():
        for row in rows:
            if not row["collisions"]:
                continue
            log = SimulationLog.load_data(Path(row["trace"]))
            samples = log.simulation_history.data
            t0 = samples[0].ego_state.time_us
            for event in row["collisions"]:
                frame = min(samples, key=lambda s: abs((s.ego_state.time_us - t0) / 1e6 - event["time_s"]))
                if abs((frame.ego_state.time_us - t0) / 1e6 - event["time_s"]) > 1e-5:
                    raise RuntimeError("Collision frame timestamp mismatch")
                obj = next(o for o in frame.observation.tracked_objects.tracked_objects
                           if o.track_token == event["track"])
                official = frame.ego_state.car_footprint.geometry.intersects(obj.box.geometry)
                old_gap, new_gap = separation(legacy, frame.ego_state, obj), separation(corrected, frame.ego_state, obj)
                if not official or new_gap > 1e-6:
                    raise RuntimeError("Corrected geometry disagrees with official collision")
                item = dict(model=name, token=row["token"], **event, official_overlap=bool(official),
                            legacy_signed_separation_m=old_gap, corrected_signed_separation_m=new_gap,
                            legacy_missed_overlap=old_gap > 0)
                result["events"].append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
