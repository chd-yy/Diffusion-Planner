#!/usr/bin/env python3
"""Run the frozen Chapter-4 original-vs-repaired candidate ablation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HDP = Path(__file__).resolve().parents[1]
ROOT = HDP / "tmp/midterm_posttraining/EXP_CH4_candidate_repair_ablation_1004"
PYTHON = Path(sys.executable)
B10 = HDP / "reproducibility/B_Epoch10_RL_Epoch2/checkpoints/B_Epoch10.pth"
DATA_DIR = HDP / "tmp/mini_val_balanced_1000_seed3407_v1/cache"
DATA_LIST = HDP / "tmp/mini_val_balanced_1000_seed3407_v1/diffusion_planner_validation.json"
POOL_PROFILE = "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
SELECTION_PROFILE = "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"


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


def run_logged(command, log_path: Path) -> None:
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
            f"command failed with code {completed.returncode}: {' '.join(map(str, command))}"
        )


def training_command(output: Path, repaired: bool):
    command = [
        str(PYTHON),
        str(HDP / "scripts/run_midterm_posttraining.py"),
        "--variant", "safety_reference",
        "--seed", "2026",
        "--output", str(output),
        "--execute",
        "--rl-max-update-steps", "20",
        "--rl-sampling-noise-scale", "0.1",
        "--rl-min-reward-std", "0.01",
    ]
    if repaired:
        command.extend([
            "--rl-candidate-pool-profile", POOL_PROFILE,
            "--rl-candidate-selection-profile", SELECTION_PROFILE,
        ])
    return command


def unique_artifact(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def comparison_command(args_file: Path, checkpoint: Path, output: Path):
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


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to reuse experiment root: {ROOT}")
    ROOT.mkdir(parents=True)
    state = {
        "experiment": "chapter4-candidate-generation-repair-ablation",
        "created_at": now(),
        "status": "running",
        "source_checkpoint": str(B10),
        "source_checkpoint_sha256": sha256(B10),
        "controlled": {
            "seed": 2026,
            "group_size": 32,
            "diffusion_steps": 6,
            "min_reward_std": 0.01,
            "optimizer_steps": 20,
            "validation_scenes": 1000,
            "validation_repeats": 3,
        },
        "single_factor": "candidate generation",
        "arms": {},
    }
    save_state(state)
    arms = [
        ("control_original_01x32", False),
        ("repair_pool200_to32", True),
    ]
    try:
        for name, repaired in arms:
            output = ROOT / name
            state["arms"][name] = {
                "status": "training",
                "started_at": now(),
                "repaired_candidate_generation": repaired,
            }
            save_state(state)
            run_logged(
                training_command(output, repaired),
                ROOT / f"{name}.launcher.log",
            )
            run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
            if run_record.get("status") != "complete":
                raise RuntimeError(f"arm {name} did not complete: {run_record.get('status')}")
            checkpoint = unique_artifact(output, "model_epoch_*.pth")
            args_file = unique_artifact(output, "args.json")
            state["arms"][name].update(
                status="trained",
                training_finished_at=now(),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                args_file=str(args_file),
            )
            save_state(state)

        validation_dir = ROOT / "validation1000"
        validation_dir.mkdir()
        for name, _ in arms:
            arm = state["arms"][name]
            arm["status"] = "validating"
            save_state(state)
            validation_output = validation_dir / f"{name}.json"
            run_logged(
                comparison_command(
                    Path(arm["args_file"]),
                    Path(arm["checkpoint"]),
                    validation_output,
                ),
                ROOT / f"{name}.validation.log",
            )
            arm.update(
                status="complete",
                validation_output=str(validation_output),
                validation_sha256=sha256(validation_output),
                finished_at=now(),
            )
            save_state(state)
        state.update(status="complete", finished_at=now())
        save_state(state)
    except BaseException as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_state(state)
        raise


if __name__ == "__main__":
    main()
