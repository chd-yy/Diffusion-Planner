import argparse
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hdp_nuplan.data_process.data_processor import DataProcessor
from hdp_nuplan.data_process.run_utils import (
    atomic_write_json,
    build_checksum_payload,
    unique_log_names,
)
from hdp_nuplan.data_process.sampling import (
    deduplicate_scenarios,
    scenario_output_name,
    select_scenarios_balanced_by_log,
)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def str_to_bool(value):
    """Parse an explicit command-line boolean without Python's bool("False") pitfall."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def _default_metadata_path(output_list_path, filename):
    return Path(output_list_path).resolve().parent / filename


def _random_sampling_report(scenarios, requested_scenarios, seed):
    selected_per_log = Counter(str(scenario.log_name) for scenario in scenarios)
    return {
        "strategy": "random",
        "seed": seed,
        "target_scenarios": requested_scenarios,
        "requested_scenarios": len(scenarios),
        "selected_scenarios": len(scenarios),
        "selected_per_log": dict(sorted(selected_per_log.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable and sharded NuPlan preprocessing")
    parser.add_argument("--data_path", default="/data/nuplan-v1.1/trainval", type=str, help="path to raw data")
    parser.add_argument("--map_path", default="/data/nuplan-v1.1/maps", type=str, help="path to map data")
    parser.add_argument("--save_path", default="./cache", type=str, help="path to save processed data")
    parser.add_argument("--scenarios_per_type", type=int, default=None, help="number of scenarios per type")
    parser.add_argument("--total_scenarios", type=int, default=10, help="scenario target for this shard")
    parser.add_argument("--shuffle_scenarios", type=str_to_bool, default=True, help="shuffle scenarios")
    parser.add_argument("--seed", type=int, default=3407, help="random seed used for scenario sampling")
    parser.add_argument(
        "--sampling_strategy",
        choices=["random", "balanced_logs"],
        default="random",
        help="global random sampling or per-log quota followed by random fill",
    )
    parser.add_argument(
        "--log_names_json",
        default="./nuplan_train.json",
        type=str,
        help="JSON list for this shard; split the full log list before invoking this program",
    )
    parser.add_argument(
        "--output_list_path",
        default="./diffusion_planner_training.json",
        type=str,
        help="path used to save this shard's successful NPZ manifest",
    )
    parser.add_argument("--sampling_report_path", default=None, type=str)
    parser.add_argument("--processing_report_path", default=None, type=str)
    parser.add_argument("--checksum_path", default=None, type=str)
    parser.add_argument("--shard_id", default="shard_00000", type=str)
    parser.add_argument(
        "--scenario_builder_workers",
        type=int,
        default=None,
        help="limit NuPlan scenario-builder subprocesses; useful when multiple shards run concurrently",
    )
    parser.add_argument(
        "--skip_existing",
        type=str_to_bool,
        default=True,
        help="resume by skipping readable NPZ files already in save_path",
    )
    parser.add_argument(
        "--fail_on_error",
        type=str_to_bool,
        default=True,
        help="finish the shard and write reports, then return an error if any scenario failed",
    )
    parser.add_argument(
        "--checksum_mode",
        choices=["manifest", "files"],
        default="manifest",
        help="checksum metadata only, or metadata plus every NPZ file",
    )

    parser.add_argument("--agent_num", type=int, help="number of agents", default=32)
    parser.add_argument("--static_objects_num", type=int, help="number of static objects", default=5)
    parser.add_argument("--lane_len", type=int, help="number of lane points", default=20)
    parser.add_argument("--lane_num", type=int, help="number of lanes", default=70)
    parser.add_argument("--route_len", type=int, help="number of route lane points", default=20)
    parser.add_argument("--route_num", type=int, help="number of route lanes", default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_scenarios <= 0:
        raise ValueError("--total_scenarios must be positive")
    if args.scenario_builder_workers is not None and args.scenario_builder_workers <= 0:
        raise ValueError("--scenario_builder_workers must be positive")

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    random.seed(args.seed)

    save_path = Path(args.save_path).resolve()
    manifest_path = Path(args.output_list_path).resolve()
    sampling_report_path = Path(args.sampling_report_path).resolve() if args.sampling_report_path else _default_metadata_path(manifest_path, "sampling_report.json")
    processing_report_path = Path(args.processing_report_path).resolve() if args.processing_report_path else _default_metadata_path(manifest_path, "processing_report.json")
    checksum_path = Path(args.checksum_path).resolve() if args.checksum_path else _default_metadata_path(manifest_path, "checksums.json")
    save_path.mkdir(parents=True, exist_ok=True)

    with Path(args.log_names_json).open("r", encoding="utf-8") as file:
        log_names = unique_log_names(json.load(file))
    if not log_names:
        raise ValueError("log_names_json contains no logs")

    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )
    builder_total = None if args.sampling_strategy == "balanced_logs" else args.total_scenarios
    builder_shuffle = False if args.sampling_strategy == "balanced_logs" else args.shuffle_scenarios
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            builder_total,
            builder_shuffle,
            log_names=log_names,
        )
    )

    worker = SingleMachineParallelExecutor(
        use_process_pool=True,
        max_workers=args.scenario_builder_workers,
    )
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, builder, scenario_filter

    if args.sampling_strategy == "balanced_logs":
        scenarios, sampling_report = select_scenarios_balanced_by_log(
            scenarios,
            log_names,
            args.total_scenarios,
            args.seed,
        )
    else:
        raw_scenario_count = len(scenarios)
        scenarios, duplicate_count = deduplicate_scenarios(scenarios)
        sampling_report = _random_sampling_report(scenarios, args.total_scenarios, args.seed)
        sampling_report["raw_scenarios"] = raw_scenario_count
        sampling_report["duplicate_output_names_removed"] = duplicate_count

    sampling_report.update(
        {
            "shard_id": args.shard_id,
            "requested_log_count": len(log_names),
            "log_names": log_names,
        }
    )
    atomic_write_json(sampling_report_path, sampling_report)
    print(f"Shard {args.shard_id}: {len(log_names)} logs, {len(scenarios)} scenarios")

    processor = DataProcessor(args)
    process_result = processor.work(scenarios, skip_existing=args.skip_existing)
    completed_files = process_result.pop("completed_files")

    expected_files = sorted({scenario_output_name(scenario) for scenario in scenarios})
    completed_set = set(completed_files)
    missing_files = sorted(set(expected_files) - completed_set)
    unexpected_files = sorted(completed_set - set(expected_files))
    if unexpected_files:
        raise RuntimeError(f"processor returned unexpected files: {unexpected_files[:3]}")

    atomic_write_json(manifest_path, completed_files)
    elapsed_seconds = time.monotonic() - start_time
    processing_report = {
        "status": "complete" if not missing_files and process_result["failed"] == 0 else "partial_failed",
        "shard_id": args.shard_id,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "data_path": str(Path(args.data_path).resolve()),
        "map_path": str(Path(args.map_path).resolve()),
        "cache_dir": str(save_path),
        "manifest_path": str(manifest_path),
        "sampling_report_path": str(sampling_report_path),
        "log_names_json": str(Path(args.log_names_json).resolve()),
        "log_names": log_names,
        "log_count": len(log_names),
        "sampling_strategy": args.sampling_strategy,
        "seed": args.seed,
        "requested_scenarios": args.total_scenarios,
        "selected_scenarios": len(scenarios),
        "manifest_count": len(completed_files),
        "missing_files": missing_files,
        **process_result,
    }
    atomic_write_json(processing_report_path, processing_report)

    checksum_payload = build_checksum_payload(
        manifest_path,
        processing_report_path,
        sampling_report_path,
        save_path,
        completed_files,
        include_npz_files=args.checksum_mode == "files",
    )
    atomic_write_json(checksum_path, checksum_payload)

    print(
        f"Shard {args.shard_id} finished: processed={process_result['processed']}, "
        f"skipped={process_result['skipped_existing']}, failed={process_result['failed']}, "
        f"manifest={len(completed_files)}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Processing report: {processing_report_path}")
    print(f"Checksums: {checksum_path}")

    if args.fail_on_error and processing_report["status"] != "complete":
        raise RuntimeError(
            f"Shard {args.shard_id} completed with {process_result['failed']} failed scenarios; "
            f"inspect {processing_report_path} and rerun with --skip_existing true"
        )


if __name__ == "__main__":
    main()
