#!/usr/bin/env python3
"""Finish R0 Test14 with three isolated single-worker simulation lanes.

Each chunk receives its own NUPLAN_EXP_ROOT and therefore its own stateful
MetricsEngine instances.  Only validated runner and aggregator parquets are
atomically copied into the canonical Chapter-3 run directory before merging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from datetime import datetime


PROJECT_ROOT = Path("/home/yanjun/NewDisk/Diffusion-Planner")
HDP_ROOT = PROJECT_ROOT / "HDP-nuplan"
PYTHON = Path("/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python")
DEVKIT_ROOT = Path("/home/yanjun/NewDisk/nuplan-devkit")
DATA_ROOT = Path("/home/yanjun/NewDisk/nuplan/dataset")
MAPS_ROOT = DATA_ROOT / "maps"
TEST_ROOT = HDP_ROOT / "tmp/test14_full_remote_subset"
TEST_DB_ROOT = TEST_ROOT / "data/cache/test14"
CONFIG_ROOT = TEST_ROOT / "config"
MANIFEST = TEST_ROOT / "eval_chunk_manifest.json"
MANAGER = HDP_ROOT / "scripts/manage_test14_chunk_results.py"

CANONICAL_ROOT = HDP_ROOT / "tmp/chapter3_test14_c0_r0_v1"
CANONICAL_SIM_ROOT = (
    CANONICAL_ROOT / "exp/simulation/closed_loop_nonreactive_agents"
)
RESULT_ROOT = (
    HDP_ROOT
    / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH3_Test14_C0_R0"
)
SCRATCH_ROOT = Path(
    "/media/yanjun/拯救者PSSD/nuplan_eval_scratch/R0_test14_parallel3_20261007"
)
SUPERVISOR_STATE = SCRATCH_ROOT / "parallel_state.json"
LOCK_PATH = CANONICAL_ROOT / "pipeline.lock"

R0_RUN = (
    HDP_ROOT
    / "tmp/chapter3_representation_ablation_v1/R0_absolute/training_log/"
    "chapter3-R0-absolute-hybrid0/2026-10-02-16:52:32"
)
R0_ARGS = R0_RUN / "args.json"
R0_CKPT = R0_RUN / "model_epoch_10_trainloss_0.0022.pth"

OLD_EVAL_ROOT = TEST_ROOT / "three_model_eval"
COMPARATOR_RUNS = {
    "test14-hard": {
        "aligned_original_dp": OLD_EVAL_ROOT
        / "exp/simulation/closed_loop_nonreactive_agents/"
        "aligned-original-dp-test14-hard",
        "hdp_b_epoch10": OLD_EVAL_ROOT
        / "exp/simulation/closed_loop_nonreactive_agents/"
        "hdp-b-epoch10-test14-hard",
        "C0": CANONICAL_SIM_ROOT / "C0-test14-hard",
    },
    "test14-random": {
        "aligned_original_dp": OLD_EVAL_ROOT
        / "exp/simulation/closed_loop_nonreactive_agents/"
        "aligned-original-dp-test14-random",
        "hdp_b_epoch10": OLD_EVAL_ROOT
        / "exp/simulation/closed_loop_nonreactive_agents/"
        "hdp-b-epoch10-test14-random",
        "C0": CANONICAL_SIM_ROOT / "C0-test14-random",
    },
}

EXPECTED_SHA256 = {
    MANIFEST: "a6b495893945d719055d9841a91e1e637063da7f3d6b439c5e82077d5964230d",
    MANAGER: "3ed4d57db3693a542d78b7f36b58fd2d85f87f9b93419b620b7a3006bdaea2cc",
    R0_ARGS: "de641ae7e3a647b3b48bd829cfe0a5521e3a075659b22c216997dbdb5c294882",
    R0_CKPT: "0f4343025e0e2031b3c120bac36d46addee60eb86d6bb0976c64b6c513ae1e33",
}

# Balanced from prior C0 per-chunk wall times.  Every remaining chunk appears
# exactly once; hard chunks 0 and 1 already passed canonical validation.
LANES = {
    0: [
        ("test14-hard", 2),
        ("test14-hard", 5),
        ("test14-random", 1),
        ("test14-random", 6),
    ],
    1: [
        ("test14-hard", 3),
        ("test14-hard", 6),
        ("test14-random", 2),
        ("test14-random", 4),
    ],
    2: [
        ("test14-hard", 4),
        ("test14-random", 0),
        ("test14-random", 3),
        ("test14-random", 5),
    ],
}

STATE_LOCK = threading.Lock()
STATE = {
    "started_at": None,
    "status": "initializing",
    "completed": [],
    "running": {},
    "failed": [],
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def write_state() -> None:
    with STATE_LOCK:
        payload = json.dumps(STATE, ensure_ascii=False, indent=2) + "\n"
        temporary = SUPERVISOR_STATE.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(SUPERVISOR_STATE)


def update_running(lane: int, value: dict | None) -> None:
    with STATE_LOCK:
        if value is None:
            STATE["running"].pop(str(lane), None)
        else:
            STATE["running"][str(lane)] = value
    write_state()


def validate_inputs(manifest: dict) -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if len(list(TEST_DB_ROOT.glob("*.db"))) != 217:
        raise RuntimeError("Test14 DB gate failed: expected 217 DB files")
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")
    all_tasks = [task for tasks in LANES.values() for task in tasks]
    expected_tasks = {
        *(('test14-hard', index) for index in range(2, 7)),
        *(('test14-random', index) for index in range(7)),
    }
    if len(all_tasks) != len(set(all_tasks)) or set(all_tasks) != expected_tasks:
        raise RuntimeError("parallel lane assignment is incomplete or duplicated")
    if manifest["benchmarks"]["test14-hard"]["count"] != 272:
        raise RuntimeError("hard manifest count drift")
    if manifest["benchmarks"]["test14-random"]["count"] != 261:
        raise RuntimeError("random manifest count drift")
    if shutil.disk_usage(SCRATCH_ROOT.parent).free < 20 * 1024**3:
        raise RuntimeError("less than 20 GiB free on PSSD scratch filesystem")
    # The two already completed R0 hard chunks must remain valid.
    canonical_hard = CANONICAL_SIM_ROOT / "R0-test14-hard"
    for index in (0, 1):
        run_checked(
            [
                str(PYTHON),
                str(MANAGER),
                "validate-chunk-artifacts",
                "--manifest",
                str(MANIFEST),
                "--benchmark",
                "test14-hard",
                "--chunk-index",
                str(index),
                "--run-dir",
                str(canonical_hard),
            ],
            stdout=subprocess.DEVNULL,
        )
    # Existing comparator runs are immutable inputs to the final four-model summary.
    for benchmark, models in COMPARATOR_RUNS.items():
        for run_dir in models.values():
            run_checked(
                [
                    str(PYTHON),
                    str(MANAGER),
                    "validate-run",
                    "--manifest",
                    str(MANIFEST),
                    "--benchmark",
                    benchmark,
                    "--run-dir",
                    str(run_dir),
                ],
                stdout=subprocess.DEVNULL,
            )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def run_chunk(manifest: dict, lane: int, benchmark: str, index: int) -> None:
    spec = manifest["benchmarks"][benchmark]["chunks"][index]
    expected_count = int(spec["count"])
    job_name = f"{benchmark}-chunk-{index:03d}"
    job_root = SCRATCH_ROOT / f"lane-{lane}" / job_name
    uid = f"R0-{benchmark}-parallel-{index:03d}"
    run_dir = job_root / "exp/simulation/closed_loop_nonreactive_agents" / uid
    job_log = job_root / "simulation.log"
    if job_root.exists():
        raise RuntimeError(f"refusing to reuse non-empty scratch job: {job_root}")
    job_root.mkdir(parents=True)
    update_running(
        lane,
        {
            "benchmark": benchmark,
            "chunk_index": index,
            "expected_count": expected_count,
            "started_at": timestamp(),
            "log": str(job_log),
        },
    )
    command = [
        str(PYTHON),
        str(DEVKIT_ROOT / "nuplan/planning/script/run_simulation.py"),
        "+simulation=closed_loop_nonreactive_agents",
        "planner=hyper_diffusion_planner",
        f"planner.hyper_diffusion_planner.config.args_file={R0_ARGS}",
        f"planner.hyper_diffusion_planner.ckpt_path={R0_CKPT}",
        "scenario_builder=nuplan",
        f"scenario_builder.db_files={TEST_DB_ROOT}",
        f"scenario_filter={spec['filter']}",
        f"experiment_uid={uid}",
        "seed=0",
        "worker=single_machine_thread_pool",
        "worker.max_workers=1",
        "worker.use_process_pool=false",
        "number_of_gpus_allocated_per_simulation=1",
        "number_of_cpus_allocated_per_simulation=1",
        "max_callback_workers=1",
        "disable_callback_parallelization=true",
        "enable_simulation_progress_bar=true",
        "verbose=true",
        "~callback.simulation_log_callback",
        (
            "hydra.searchpath=["
            f"file://{CONFIG_ROOT},"
            "pkg://hdp_nuplan.config.scenario_filter,"
            "pkg://hdp_nuplan.config,"
            "pkg://diffusion_planner.config,"
            "pkg://nuplan.planning.script.config.common,"
            "pkg://nuplan.planning.script.experiments]"
        ),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "NUPLAN_DATA_ROOT": str(DATA_ROOT),
            "NUPLAN_MAPS_ROOT": str(MAPS_ROOT),
            "NUPLAN_DEVKIT_ROOT": str(DEVKIT_ROOT),
            "NUPLAN_EXP_ROOT": str(job_root),
            "CUDA_VISIBLE_DEVICES": "0",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log(f"lane={lane} starting {job_name}, scenarios={expected_count}")
    with job_log.open("w", encoding="utf-8") as stream:
        run_checked(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    run_checked(
        [
            str(PYTHON),
            str(MANAGER),
            "archive-chunk",
            "--manifest",
            str(MANIFEST),
            "--benchmark",
            benchmark,
            "--chunk-index",
            str(index),
            "--run-dir",
            str(run_dir),
            "--report",
            str(run_dir / "runner_report.parquet"),
        ],
        stdout=subprocess.DEVNULL,
    )
    canonical_run = CANONICAL_SIM_ROOT / f"R0-{benchmark}"
    report_name = f"chunk-{index:03d}.parquet"
    atomic_copy(run_dir / "chunk_runner_reports" / report_name, canonical_run / "chunk_runner_reports" / report_name)
    atomic_copy(run_dir / "chunk_aggregator_metrics" / report_name, canonical_run / "chunk_aggregator_metrics" / report_name)
    run_checked(
        [
            str(PYTHON),
            str(MANAGER),
            "validate-chunk-artifacts",
            "--manifest",
            str(MANIFEST),
            "--benchmark",
            benchmark,
            "--chunk-index",
            str(index),
            "--run-dir",
            str(canonical_run),
        ],
        stdout=subprocess.DEVNULL,
    )
    with STATE_LOCK:
        STATE["completed"].append(
            {
                "lane": lane,
                "benchmark": benchmark,
                "chunk_index": index,
                "count": expected_count,
                "completed_at": timestamp(),
            }
        )
    update_running(lane, None)
    log(f"lane={lane} completed {job_name}, scenarios={expected_count}")


def run_lane(manifest: dict, lane: int) -> None:
    try:
        for benchmark, index in LANES[lane]:
            run_chunk(manifest, lane, benchmark, index)
    except Exception as error:
        with STATE_LOCK:
            STATE["failed"].append(
                {"lane": lane, "time": timestamp(), "error": repr(error)}
            )
        update_running(lane, None)
        log(f"lane={lane} FAILED: {error!r}")
        raise


def merge_and_summarize() -> None:
    for benchmark in ("test14-hard", "test14-random"):
        r0_run = CANONICAL_SIM_ROOT / f"R0-{benchmark}"
        run_checked(
            [
                str(PYTHON),
                str(MANAGER),
                "merge",
                "--manifest",
                str(MANIFEST),
                "--benchmark",
                benchmark,
                "--run-dir",
                str(r0_run),
            ]
        )
        run_checked(
            [
                str(PYTHON),
                str(MANAGER),
                "validate-run",
                "--manifest",
                str(MANIFEST),
                "--benchmark",
                benchmark,
                "--run-dir",
                str(r0_run),
            ]
        )
        suffix = benchmark.removeprefix("test14-")
        summary = RESULT_ROOT / f"test14_{suffix}_four_models.json"
        summary_command = [
            str(PYTHON),
            str(HDP_ROOT / "scripts/summarize_closed_loop_metrics.py"),
        ]
        for name, run_dir in COMPARATOR_RUNS[benchmark].items():
            summary_command.extend(["--run", f"{name}={run_dir}"])
        summary_command.extend(["--run", f"R0={r0_run}", "--output", str(summary)])
        run_checked(summary_command)
        run_checked(
            [
                str(PYTHON),
                str(HDP_ROOT / "scripts/analyze_paired_closed_loop.py"),
                "--summary",
                str(summary),
                "--baseline",
                "C0",
                "--candidate",
                "R0",
                "--output",
                str(RESULT_ROOT / f"test14_{suffix}_C0_vs_R0.json"),
                "--markdown",
                str(RESULT_ROOT / f"test14_{suffix}_C0_vs_R0.md"),
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another Test14 pipeline holds {LOCK_PATH}") from error
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=False)
    validate_inputs(manifest)
    log("preflight passed: frozen inputs, comparators, R0 hard chunks 0/1, PSSD")
    if not args.execute:
        log(f"dry-run only; remove empty scratch root before execution: {SCRATCH_ROOT}")
        return
    STATE.update({"started_at": timestamp(), "status": "running"})
    write_state()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_lane, manifest, lane) for lane in sorted(LANES)]
            for future in futures:
                future.result()
        STATE["status"] = "merging"
        write_state()
        merge_and_summarize()
        STATE.update({"status": "complete", "completed_at": timestamp()})
        write_state()
        log("R0 Test14 hard/random parallel evaluation complete and validated")
    except Exception:
        STATE.update({"status": "failed", "failed_at": timestamp()})
        write_state()
        raise
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
