#!/usr/bin/env python3
"""Run one repaired-candidate rollout and a 20/50/100/200-step dose response."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HDP = Path(__file__).resolve().parents[1]
ROOT = HDP / "tmp/midterm_posttraining/EXP_CH4_repair_dose_response_1004"
PYTHON = Path(sys.executable)
B10 = HDP / "reproducibility/B_Epoch10_RL_Epoch2/checkpoints/B_Epoch10.pth"
DATA_DIR = HDP / "tmp/mini_val_balanced_1000_seed3407_v1/cache"
DATA_LIST = HDP / "tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json"
POOL_PROFILE = "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
SELECTION_PROFILE = "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"
MILESTONES = (20, 50, 100, 200)


def now() -> str:
    return datetime.now().strftime("%m月%d日 %H:%M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_state(state: dict) -> None:
    (ROOT / "matrix_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_logged(command: list[str], log_path: Path) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=HDP,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with code {completed.returncode}: {' '.join(command)}"
        )


def unique_artifact(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def training_command(output: Path) -> list[str]:
    return [
        str(PYTHON),
        str(HDP / "scripts/run_midterm_posttraining.py"),
        "--variant", "safety_reference",
        "--seed", "2026",
        "--output", str(output),
        "--execute",
        "--rl-max-update-steps", str(MILESTONES[-1]),
        "--rl-update-checkpoint-steps", ",".join(map(str, MILESTONES)),
        "--rl-sampling-noise-scale", "0.1",
        "--rl-min-reward-std", "0.01",
        "--rl-candidate-pool-profile", POOL_PROFILE,
        "--rl-candidate-selection-profile", SELECTION_PROFILE,
    ]


def comparison_command(args_file: Path, checkpoint: Path, output: Path) -> list[str]:
    return [
        str(PYTHON),
        str(HDP / "scripts/compare_checkpoint_behavior.py"),
        "--args-file", str(args_file),
        "--supervised-checkpoint", str(B10),
        "--rl-checkpoint", str(checkpoint),
        "--supervised-weight-source", "ema_state_dict",
        "--rl-weight-source", "ema_state_dict",
        "--data-dir", str(DATA_DIR),
        "--data-list", str(DATA_LIST),
        "--batch-size", "16",
        "--num-workers", "2",
        "--repeats", "3",
        "--seed", "3407",
        "--num-samples", "1",
        "--diffusion-steps", "6",
        "--sampling-noise-scale", "0.1",
        "--trajectory-augmentation-std", "0",
        "--device", "cuda",
        "--output", str(output),
    ]


def gate_result(report: dict) -> dict:
    delta = report["paired_delta_rl_minus_supervised"]
    checks = {
        "reward_median_positive": delta["reward"]["p50"] > 0,
        "progress_material": delta["progress"]["mean"] >= 0.005,
        "progress_p10_nonnegative": delta["progress"]["p10"] >= 0,
        "collision_cost_nonincreasing": delta["collision_cost"]["mean"] <= 0,
        "no_collision_non_decreasing": delta["no_collision"]["mean"] >= 0,
        "route_cost_within_tolerance": delta["route_cost"]["mean"] <= 1e-4,
        "comfort_cost_within_tolerance": delta["comfort_cost"]["mean"] <= 1e-4,
        "ade_preserved": delta["ade_m"]["relative_change_percent"] <= 0.5,
        "fde_preserved": delta["fde_m"]["relative_change_percent"] <= 0.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            metric: delta[metric]
            for metric in (
                "reward", "progress", "collision_cost", "no_collision",
                "route_cost", "comfort_cost", "ade_m", "fde_m",
            )
        },
    }


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to reuse experiment root: {ROOT}")
    ROOT.mkdir(parents=True)
    training_output = ROOT / "repair_training"
    state = {
        "experiment": "chapter4-repaired-candidate-dose-response",
        "created_at": now(),
        "status": "training",
        "source_checkpoint": str(B10),
        "source_checkpoint_sha256": sha256(B10),
        "controlled": {
            "seed": 2026,
            "single_fixed_rollout": True,
            "group_size": 32,
            "candidate_pool_size": 200,
            "diffusion_steps": 6,
            "min_reward_std": 0.01,
            "optimizer_step_milestones": list(MILESTONES),
            "validation_scenes": 1000,
            "validation_repeats": 3,
        },
        "automatic_safety40": False,
        "milestones": {},
    }
    save_state(state)
    try:
        run_logged(training_command(training_output), ROOT / "training.launcher.log")
        run_record = json.loads(
            (training_output / "run.json").read_text(encoding="utf-8")
        )
        if run_record.get("status") != "complete":
            raise RuntimeError(f"training did not complete: {run_record.get('status')}")
        args_file = unique_artifact(training_output, "args.json")
        state.update(
            status="validating",
            training_finished_at=now(),
            args_file=str(args_file),
        )
        save_state(state)

        validation_dir = ROOT / "validation1000"
        validation_dir.mkdir()
        for step in MILESTONES:
            checkpoint = unique_artifact(
                training_output, f"model_update_step_{step:04d}.pth"
            )
            output = validation_dir / f"step_{step:04d}.json"
            state["milestones"][str(step)] = {
                "status": "validating",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            }
            save_state(state)
            run_logged(
                comparison_command(args_file, checkpoint, output),
                ROOT / f"step_{step:04d}.validation.log",
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            state["milestones"][str(step)].update(
                status="complete",
                validation_output=str(output),
                validation_sha256=sha256(output),
                gate=gate_result(report),
                finished_at=now(),
            )
            save_state(state)

        passed = [
            int(step)
            for step, item in state["milestones"].items()
            if item["gate"]["passed"]
        ]
        state.update(
            status="complete",
            finished_at=now(),
            passed_milestones=passed,
            safety40_candidate=(min(passed) if passed else None),
            safety40_authorized=False,
        )
        save_state(state)
    except BaseException as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_state(state)
        raise


if __name__ == "__main__":
    main()
