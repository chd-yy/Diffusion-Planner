"""Reproduce known Test14 regressions with paired seeds and simulation traces.

This is a diagnostic set, never a training manifest or a model selection score.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import yaml

from prepare_test14_eval_chunks import render_filter
from summarize_closed_loop_metrics import summarize_run


TOKENS = [
    "2e56997063c057a0", "6eb38f317f2251f2", "33187fb09d0e52f8",
    "75d56d4c7b6f5013", "9e507708119c5596", "a874cefaca8e5b69",
]


def compare_safety(baseline, candidate):
    """A fail-closed diagnostic gate, not independent model certification."""
    required = ["score", "no_ego_at_fault_collisions", "drivable_area_compliance",
                "ego_is_making_progress", "driving_direction_compliance",
                "time_to_collision_within_bound", "ego_progress_along_expert_route",
                "speed_limit_compliance"]
    indexed = []
    for summary in (baseline, candidate):
        rows = summary["scenarios"]
        if not summary["runner_report_complete"] or summary["failed_simulations"]:
            raise ValueError("Incomplete simulation cannot pass the safety gate")
        mapping = {(r["log_name"], r["scenario"]): r for r in rows}
        if not rows or len(mapping) != len(rows):
            raise ValueError("Empty or duplicated scenarios cannot pass the safety gate")
        for row in rows:
            if any(row.get(k) is None or not math.isfinite(row[k]) for k in required):
                raise ValueError("Missing or non-finite safety metrics")
        indexed.append(mapping)
    base, cand = indexed
    if base.keys() != cand.keys():
        raise ValueError("Paired scenario coverage mismatch")
    regressions = {metric: [token for (_, token), row in base.items()
                           if cand[(row["log_name"], token)][metric] < row[metric] - 1e-6]
                   for metric in required[1:6]}
    deltas = {metric: sum(cand[k][metric] - base[k][metric] for k in base) / len(base)
              for metric in required}
    passed = not any(regressions.values()) and all(
        deltas[k] >= -1e-6 for k in ["score", "ego_progress_along_expert_route", "speed_limit_compliance"])
    return dict(diagnostic_gate_passed=passed, new_regressions=regressions, mean_deltas=deltas,
                promotion_authorized=False,
                limitation="Diagnosis only. Independent held-out paired evaluation is still required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--models", nargs="+", choices=["b10", "rl_epoch2", "safety_repair"],
                        default=["b10", "rl_epoch2"])
    parser.add_argument("--candidate-args", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--baseline-diagnosis", type=Path,
                        help="Reuse completed paired B10 runs at the same seed for a diagnostic gate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    hdp = root / "HDP-nuplan"
    devkit = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", str(root.parent / "nuplan-devkit")))
    data = root.parent / "nuplan/dataset"
    work = hdp / "tmp/test14_full_remote_subset"
    b10 = hdp / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
    rl = hdp / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45"
    models = {
        "b10": (b10 / "args.json", b10 / "model_epoch_10_trainloss_0.0091.pth", "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"),
        "rl_epoch2": (rl / "args.json", rl / "model_epoch_2_trainloss_0.0003.pth", "8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c"),
    }
    if "safety_repair" in args.models:
        if args.candidate_args is None or args.candidate_checkpoint is None:
            parser.error("safety_repair requires --candidate-args and --candidate-checkpoint")
        candidate = args.candidate_checkpoint.resolve()
        models["safety_repair"] = (args.candidate_args.resolve(), candidate,
                                    hashlib.sha256(candidate.read_bytes()).hexdigest())
    models = {name: models[name] for name in dict.fromkeys(args.models)}
    for _, checkpoint, expected in models.values():
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = output / "config/scenario_filter"
    config.mkdir(parents=True)
    (config / "rl-safety-regressions.yaml").write_text(
        yaml.safe_dump(render_filter(TOKENS), sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now().strftime("%m月%d日 %H:%M"),
        "purpose": "diagnosis_only", "tokens": TOKENS, "seeds": args.seeds,
        "models": {k: {"args": str(a), "checkpoint": str(c), "checkpoint_sha256": s,
                       "args_sha256": hashlib.sha256(a.read_bytes()).hexdigest()}
                   for k, (a, c, s) in models.items()},
        "runs": {},
    }
    def save():
        (output / "diagnosis.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save()
    baseline_manifest = None
    if args.baseline_diagnosis:
        baseline_manifest = json.loads(args.baseline_diagnosis.read_text())
        if set(baseline_manifest["tokens"]) != set(TOKENS):
            raise ValueError("Baseline diagnosis token mismatch")
        expected_sha = "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"
        if baseline_manifest["models"]["b10"]["checkpoint_sha256"] != expected_sha:
            raise ValueError("Baseline diagnosis is not the historical B10")
        manifest["baseline_diagnosis"] = str(args.baseline_diagnosis.resolve())
        for seed in args.seeds:
            if baseline_manifest["runs"][f"b10-safety-seed{seed}"]["status"] != "complete":
                raise ValueError("Baseline run incomplete")
        save()
    env = dict(os.environ, NUPLAN_DATA_ROOT=str(data), NUPLAN_MAPS_ROOT=str(data / "maps"),
               NUPLAN_EXP_ROOT=str(output), HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(hdp), str(root), str(devkit), env.get("PYTHONPATH", "")])
    for seed in args.seeds:
        for name, (model_args, checkpoint, _) in models.items():
            uid = f"{name}-safety-seed{seed}"
            command = [sys.executable, str(devkit / "nuplan/planning/script/run_simulation.py"),
                "+simulation=closed_loop_nonreactive_agents", "planner=hyper_diffusion_planner",
                f"planner.hyper_diffusion_planner.config.args_file={model_args}",
                f"planner.hyper_diffusion_planner.ckpt_path={checkpoint}",
                "scenario_builder=nuplan", f"scenario_builder.db_files={work / 'data/cache/test14'}",
                "scenario_filter=rl-safety-regressions", f"experiment_uid={uid}", f"seed={seed}",
                "worker=single_machine_thread_pool", "worker.max_workers=1", "worker.use_process_pool=false",
                "number_of_gpus_allocated_per_simulation=1", "number_of_cpus_allocated_per_simulation=1",
                "max_callback_workers=1", "disable_callback_parallelization=true",
                "enable_simulation_progress_bar=true",
                f"hydra.searchpath=[file://{output / 'config'},pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
            ]
            manifest["runs"][uid] = {"command": command, "status": "running"}
            save()
            print(f"{datetime.now():%m月%d日 %H:%M} starting {uid}", flush=True)
            with (output / f"{uid}.log").open("w") as log:
                result = subprocess.run(command, cwd=hdp, env=env, stdout=log, stderr=subprocess.STDOUT)
            run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
            try:
                summary = summarize_run(run_dir)
                if summary["failed_simulations"] or not summary["runner_report_complete"]:
                    raise RuntimeError(f"Incomplete simulation: {uid}")
                if {r["scenario"] for r in summary["scenarios"]} != set(TOKENS):
                    raise RuntimeError(f"Scenario coverage mismatch: {uid}")
                traces = list((run_dir / "simulation_log").rglob("*.msgpack.xz"))
                if len(traces) != len(TOKENS):
                    raise RuntimeError(f"Expected {len(TOKENS)} traces, got {len(traces)}: {uid}")
            except Exception as error:
                manifest["runs"][uid].update(status="failed", returncode=result.returncode, error=str(error))
                save()
                raise
            manifest["runs"][uid].update(status="complete", returncode=result.returncode,
                                         summary=summary, traces=[str(p) for p in traces])
            if baseline_manifest is not None and name != "b10":
                baseline = baseline_manifest["runs"][f"b10-safety-seed{seed}"]["summary"]
                manifest["runs"][uid]["safety_gate"] = compare_safety(baseline, summary)
            save()
            print(f"completed {uid}: score={summary['means']['score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
