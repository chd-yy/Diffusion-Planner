#!/usr/bin/env python3
"""Export one paired R0/C0 open-loop prediction as native nuBoard logs.

The resulting nuBoard experiment contains two side-by-side planner tiles.  In
each tile, nuBoard renders the exact saved 8 s prediction and obtains the
expert trajectory, map, traffic lights, and tracked objects from the original
nuPlan database.  This is an open-loop snapshot, not a closed-loop rollout.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.history.simulation_history import (
    SimulationHistory,
    SimulationHistorySample,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-data", required=True, type=Path)
    parser.add_argument("--scene-order", required=True, type=Path)
    parser.add_argument("--scene-cache", required=True, type=Path)
    parser.add_argument("--db-file", required=True, type=Path)
    parser.add_argument("--map-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat-index", default=0, type=int)
    return parser.parse_args()


def build_scenario(scene_metadata: dict[str, str], db_file: Path, map_root: Path):
    builder = NuPlanScenarioBuilder(
        data_root=str(db_file.parent),
        map_root=str(map_root),
        sensor_root=None,
        db_files=[str(db_file)],
        map_version="nuplan-maps-v1.0",
        max_workers=1,
        verbose=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=[scene_metadata["token"]],
        log_names=[scene_metadata["log_name"]],
        map_names=[scene_metadata["map_name"]],
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )
    scenarios = builder.get_scenarios(
        scenario_filter,
        SingleMachineParallelExecutor(use_process_pool=False, max_workers=1),
    )
    if len(scenarios) != 1:
        raise RuntimeError(f"Expected one scenario, found {len(scenarios)}")
    scenario = scenarios[0]
    if scenario.token != scene_metadata["token"]:
        raise RuntimeError("Loaded scenario token does not match cached metadata")
    return scenario


def prediction_to_trajectory(prediction: np.ndarray, scenario) -> InterpolatedTrajectory:
    poses = np.column_stack(
        [prediction[:, 0], prediction[:, 1], np.arctan2(prediction[:, 3], prediction[:, 2])]
    ).astype(np.float32)
    past = list(scenario.get_ego_past_trajectory(0, time_horizon=2.0, num_samples=20))
    history = deque([*past, scenario.initial_ego_state])
    states = transform_predictions_to_states(
        poses,
        history,
        future_horizon=8.0,
        step_interval=0.1,
        include_ego_state=True,
    )
    return InterpolatedTrajectory(states)


def save_log(output_dir: Path, planner_name: str, scenario, trajectory) -> Path:
    log_path = (
        output_dir
        / "simulation"
        / planner_name
        / scenario.scenario_type
        / scenario.log_name
        / scenario.token
        / "0.msgpack.xz"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    history.add_sample(
        SimulationHistorySample(
            iteration=SimulationIteration(scenario.get_time_point(0), 0),
            ego_state=scenario.initial_ego_state,
            trajectory=trajectory,
            observation=scenario.get_tracked_objects_at_iteration(0),
            traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0)),
        )
    )
    planner = SimplePlanner(
        horizon_seconds=8.0,
        sampling_time=0.1,
        acceleration=np.zeros(2, dtype=np.float32),
    )
    SimulationLog(log_path, scenario, planner, history).save_to_file()
    return log_path


def main() -> None:
    args = parse_args()
    scene_names = json.loads(args.scene_order.read_text(encoding="utf-8"))
    scene_name = args.scene_cache.name
    if scene_name not in scene_names:
        raise ValueError(f"Scene {scene_name} not present in paired scene order")
    scene_index = scene_names.index(scene_name)

    with np.load(args.scene_cache, allow_pickle=False) as cache:
        scene_metadata = {
            key: str(cache[key].item())
            for key in ("map_name", "log_name", "scenario_type", "token")
        }
        cached_expert = cache["ego_agent_future"].copy()
    if args.db_file.stem != scene_metadata["log_name"]:
        raise ValueError("DB filename does not match cached log_name")

    paired = np.load(args.paired_data, allow_pickle=False)
    repeats = paired["r0_trajectories"].shape[0]
    if not 0 <= args.repeat_index < repeats:
        raise ValueError(f"repeat-index must be in [0, {repeats - 1}]")
    r0 = paired["r0_trajectories"][args.repeat_index, scene_index]
    c0 = paired["c0_trajectories"][args.repeat_index, scene_index]
    expert = paired["expert_trajectories"][scene_index]

    scenario = build_scenario(scene_metadata, args.db_file, args.map_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs = {
        "R0_absolute": save_log(
            args.output_dir,
            "R0_absolute",
            scenario,
            prediction_to_trajectory(r0, scenario),
        ),
        "C0_increment": save_log(
            args.output_dir,
            "C0_increment",
            scenario,
            prediction_to_trajectory(c0, scenario),
        ),
    }

    # nuBoard itself loads the expert from the scenario DB.  Fail closed if
    # the database, preprocessing cache, and paired-evaluation target do not
    # describe exactly the same 8 s expert trajectory.
    database_future = list(scenario.get_ego_future_trajectory(0, 8.0, 80))
    database_relative = np.asarray(
        [
            pose.serialize()
            for pose in absolute_to_relative_poses(
                [scenario.initial_ego_state.rear_axle]
                + [state.rear_axle for state in database_future]
            )[1:]
        ],
        dtype=np.float32,
    )
    if cached_expert.shape != (80, 3) or not np.array_equal(database_relative, cached_expert):
        raise RuntimeError("nuPlan DB expert does not exactly match the preprocessing cache")
    paired_from_cache = np.column_stack(
        [
            cached_expert[:, :2],
            np.cos(cached_expert[:, 2]),
            np.sin(cached_expert[:, 2]),
        ]
    ).astype(np.float32)
    if expert.shape != (80, 4) or not np.array_equal(expert, paired_from_cache):
        raise RuntimeError("Paired-evaluation expert does not exactly match the preprocessing cache")

    nuboard_path = args.output_dir / f"R0_C0_Expert_{scene_metadata['token']}.nuboard"
    NuBoardFile(
        simulation_main_path=str(args.output_dir.resolve()),
        simulation_folder="simulation",
        metric_main_path=str(args.output_dir.resolve()),
        metric_folder="metrics",
        aggregator_metric_folder="aggregator_metric",
    ).save_nuboard_file(nuboard_path)

    manifest = {
        "visualization_semantics": "paired_open_loop_snapshot_not_closed_loop_execution",
        "scene_index": scene_index,
        "repeat_index_zero_based": args.repeat_index,
        **scene_metadata,
        "paired_data": str(args.paired_data.resolve()),
        "scene_cache": str(args.scene_cache.resolve()),
        "db_file": str(args.db_file.resolve()),
        "nuboard_file": str(nuboard_path.resolve()),
        "simulation_logs": {name: str(path.resolve()) for name, path in logs.items()},
        "expert_source": "nuPlan scenario database",
        "expert_audit": "DB relative poses == cached [x,y,yaw] == paired [x,y,cos(yaw),sin(yaw)]",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
