"""Resume-safe, full Test14 paired inference-seed verification of fixed models.

This evaluates existing B10 and safety-geometry checkpoints; it never trains or
promotes a model. Each subprocess has one worker and a private metric engine.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from diagnose_rl_safety_regressions import compare_safety
from manage_test14_chunk_results import (
    archive_chunk, archived_paths, merge, validate_chunk_aggregator,
    validate_report, validate_run,
)
from summarize_closed_loop_metrics import summarize_run


ROOT = Path(__file__).resolve().parents[2]
HDP = ROOT / "HDP-nuplan"
WORK = HDP / "tmp/test14_full_remote_subset"
DEVKIT = ROOT.parent / "nuplan-devkit"
DATA = ROOT.parent / "nuplan/dataset"
BENCHMARKS = ("test14-hard", "test14-random")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_models():
    b10 = HDP / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
    repaired = HDP / "tmp/rl_safety_geometry_training_0915_2112/training_log/hdp-rl-safety-repair-controlled/2026-09-15-21:12:50"
    return {
        "b10": {"args": str(b10 / "args.json"),
                "checkpoint": str(b10 / "model_epoch_10_trainloss_0.0091.pth"),
                "checkpoint_sha256": "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"},
        "safety_geometry": {"args": str(repaired / "args.json"),
                            "checkpoint": str(repaired / "model_epoch_2_trainloss_0.0009.pth"),
                            "checkpoint_sha256": "3a2348a12eb88f32018921514ffd4bce46720392370af9928aa0392ac4e56948"},
    }


def validate_manifest(manifest):
    if manifest.get("chunk_size") != 40:
        raise ValueError("Pairing requires the original 40-scenario chunk protocol")
    for name, count in zip(BENCHMARKS, (272, 261)):
        benchmark = manifest["benchmarks"][name]
        flattened = [t for chunk in benchmark["chunks"] for t in chunk["tokens"]]
        if len(flattened) != count or len(set(flattened)) != count:
            raise ValueError(f"Incomplete or duplicated {name} tokens")
        if flattened != benchmark["tokens"] or benchmark["count"] != count:
            raise ValueError(f"Changed chunk order or count for {name}")
        for index, chunk in enumerate(benchmark["chunks"]):
            expected = min(40, count - index * 40)
            if chunk["index"] != index or chunk["count"] != expected or len(chunk["tokens"]) != expected:
                raise ValueError(f"Invalid chunk boundaries for {name}")


def artifact_state(run_dir, benchmark, chunk):
    """Never silently overwrite any archived result, including damaged pairs."""
    report, aggregate = archived_paths(run_dir, chunk["index"])
    if not report.exists() and not aggregate.exists():
        return "missing"
    validate_report(report, set(chunk["tokens"]))
    validate_chunk_aggregator(aggregate, benchmark, chunk)
    return "complete"


def command_for(model, benchmark, seed, chunk, output):
    uid = f"{model}-{benchmark}-seed{seed}"
    spec = checkpoint_models()[model]
    return [sys.executable, str(DEVKIT / "nuplan/planning/script/run_simulation.py"),
            "+simulation=closed_loop_nonreactive_agents", "planner=hyper_diffusion_planner",
            f"planner.hyper_diffusion_planner.config.args_file={spec['args']}",
            f"planner.hyper_diffusion_planner.ckpt_path={spec['checkpoint']}",
            "scenario_builder=nuplan", f"scenario_builder.db_files={WORK / 'data/cache/test14'}",
            f"scenario_filter={chunk['filter']}", f"experiment_uid={uid}", f"seed={seed}",
            "worker=single_machine_thread_pool", "worker.max_workers=1", "worker.use_process_pool=false",
            "number_of_gpus_allocated_per_simulation=1", "number_of_cpus_allocated_per_simulation=1",
            "max_callback_workers=1", "disable_callback_parallelization=true",
            "enable_simulation_progress_bar=true", "verbose=true", "~callback.simulation_log_callback",
            f"hydra.searchpath=[file://{output / 'config'},pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"]


def checked_files(manifest):
    """Record code/config identity; refuse changed dependencies during resume."""
    paths = set((HDP / "hdp_nuplan").rglob("*.py"))
    paths.update((ROOT / "diffusion_planner").rglob("*.py"))
    paths.update((DEVKIT / "nuplan").rglob("*.py"))
    paths.update([Path(__file__).resolve(), WORK / "eval_chunk_manifest.json",
                  HDP / "scripts/manage_test14_chunk_results.py",
                  HDP / "scripts/summarize_closed_loop_metrics.py",
                  HDP / "scripts/diagnose_rl_safety_regressions.py"])
    paths.update(Path(chunk["config"]) for b in manifest["benchmarks"].values() for chunk in b["chunks"])
    for spec in checkpoint_models().values():
        paths.update([Path(spec["args"]), Path(spec["checkpoint"])])
    return {str(path): sha256(path) for path in sorted(paths)}


def check_identity(identity):
    for path, expected in identity.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Input changed during paired evaluation: {path}")


def run_job(model, benchmark, seed, output, manifest_path, manifest, identity):
    uid = f"{model}-{benchmark}-seed{seed}"
    run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
    state_path = output / f"{uid}.state.json"
    state = {"model": model, "benchmark": benchmark, "seed": seed,
             "status": "running", "started_at": datetime.now().strftime("%m月%d日 %H:%M"), "chunks": []}
    write_json(state_path, state)
    env = dict(os.environ, NUPLAN_DATA_ROOT=str(DATA), NUPLAN_MAPS_ROOT=str(DATA / "maps"),
               NUPLAN_DEVKIT_ROOT=str(DEVKIT), NUPLAN_EXP_ROOT=str(output), CUDA_VISIBLE_DEVICES="0",
               HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(HDP), str(ROOT), str(DEVKIT), env.get("PYTHONPATH", "")])
    benchmark_data = manifest["benchmarks"][benchmark]
    args = SimpleNamespace(manifest=manifest_path, benchmark=benchmark, run_dir=run_dir)
    try:
        with (output / f"{uid}.log").open("a") as log:
            for chunk in benchmark_data["chunks"]:
                check_identity(identity)
                args.chunk_index = chunk["index"]
                if artifact_state(run_dir, benchmark_data, chunk) == "complete":
                    state["chunks"].append({"index": chunk["index"], "status": "kept"})
                    write_json(state_path, state)
                    continue
                command = command_for(model, benchmark, seed, chunk, output)
                state["active_command"] = command
                state["active_chunk"] = chunk["index"]
                write_json(state_path, state)
                print(f"[{datetime.now():%m月%d日 %H:%M}] {uid} chunk={chunk['index']} start", flush=True)
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{uid} chunk {chunk['index']} exit {result.returncode}")
                args.report = run_dir / "runner_report.parquet"
                archive_chunk(args)
                state["chunks"].append({"index": chunk["index"], "status": "complete"})
                write_json(state_path, state)
            merge(args)
            validate_run(args)
        summary = summarize_run(run_dir)
        state.update(status="complete", finished_at=datetime.now().strftime("%m月%d日 %H:%M"))
        state.pop("active_command", None)
        state.pop("active_chunk", None)
        write_json(output / f"{uid}.summary.json", summary)
        write_json(state_path, state)
        return summary
    except Exception as error:
        state.update(status="failed", error=str(error), finished_at=datetime.now().strftime("%m月%d日 %H:%M"))
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 1 for seed in args.seeds):
        parser.error("Use distinct positive inference seeds; seed0 remains untouched")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((WORK / "eval_chunk_manifest.json").read_text())
        validate_manifest(manifest)
        models = checkpoint_models()
        for spec in models.values():
            if sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
                raise RuntimeError("Fixed model identity mismatch")
        identity = checked_files(manifest)
        protocol = {"seeds": args.seeds, "models": models, "file_sha256": identity,
                    "chunk_size": 40, "max_processes": 2, "worker_per_process": 1,
                    "purpose": "full-distribution paired inference-seed verification; not a blind test",
                    "promotion_authorized": False}
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Protocol/input drift: use a new output directory, do not mix results")
        else:
            if list(output.glob("*.state.json")) or (output / "exp").exists():
                raise RuntimeError("Untracked evaluation artifacts in output directory")
            write_json(protocol_path, protocol)
        config_root = output / "config/scenario_filter"
        config_root.mkdir(parents=True, exist_ok=True)
        for benchmark in manifest["benchmarks"].values():
            for chunk in benchmark["chunks"]:
                source = Path(chunk["config"])
                destination = config_root / source.name
                if destination.exists() and destination.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"Snapshot config changed: {destination}")
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
        manifest_path = output / "eval_chunk_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Snapshot manifest changed")
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        runtime_identity = dict(identity)
        runtime_identity.update({str(path): sha256(path) for path in config_root.glob("*.yaml")})
        runtime_identity[str(manifest_path)] = sha256(manifest_path)
        all_seeds = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for seed in args.seeds:
                comparisons = {}
                for benchmark in BENCHMARKS:
                    # Pair both fixed policies on the same benchmark and seed.
                    futures = {name: pool.submit(run_job, name, benchmark, seed, output,
                                                manifest_path, manifest, runtime_identity) for name in models}
                    summaries = {name: future.result() for name, future in futures.items()}
                    comparisons[benchmark] = {"baseline": summaries["b10"],
                                              "candidate": summaries["safety_geometry"],
                                              "gate": compare_safety(summaries["b10"], summaries["safety_geometry"])}
                    write_json(output / f"comparison_seed{seed}.json", comparisons)
                all_seeds[str(seed)] = comparisons
                print(f"[{datetime.now():%m月%d日 %H:%M}] paired seed={seed} complete", flush=True)
        write_json(output / "multiseed_comparison.json",
                   {"created_at": datetime.now().strftime("%m月%d日 %H:%M"), "seeds": all_seeds,
                    "promotion_authorized": False})
        print("Full paired multiseed evaluation complete", flush=True)


if __name__ == "__main__":
    main()
