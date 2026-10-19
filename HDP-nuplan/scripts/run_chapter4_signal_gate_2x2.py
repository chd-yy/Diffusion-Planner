#!/usr/bin/env python3
"""Run the Chapter 4 noise-scale x reward-std 20-step gate sequentially."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ARMS = (
    ("A_noise01_std1e6", 0.1, 1e-6),
    ("B_noise01_std001", 0.1, 0.01),
    ("C_noise05_std1e6", 0.5, 1e-6),
    ("D_noise05_std001", 0.5, 0.01),
)


def save(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    allowed = (Path(__file__).resolve().parents[1] / "tmp/midterm_posttraining").resolve()
    if allowed not in root.parents or root.exists():
        raise ValueError(f"Use a new child directory of {allowed}")
    root.mkdir(parents=True)
    state_path = root / "matrix_state.json"
    state = {
        "experiment": "chapter4-candidate-signal-gate-2x2",
        "created_at": datetime.now().strftime("%m月%d日 %H:%M"),
        "status": "running",
        "common": {
            "source": "B10 Epoch10 EMA",
            "variant": "safety_reference",
            "seed": args.seed,
            "group_size": 32,
            "diffusion_steps": 6,
            "update_steps": 20,
            "training_epochs": 2,
        },
        "arms": {},
    }
    save(state_path, state)

    runner = Path(__file__).with_name("run_midterm_posttraining.py")
    for arm, noise, min_std in ARMS:
        output = root / arm
        record = {
            "noise_scale": noise,
            "min_reward_std": min_std,
            "status": "running",
            "started_at": datetime.now().strftime("%m月%d日 %H:%M"),
            "output": str(output),
        }
        state["arms"][arm] = record
        state["current_arm"] = arm
        save(state_path, state)
        command = [
            sys.executable,
            str(runner),
            "--variant", "safety_reference",
            "--seed", str(args.seed),
            "--output", str(output),
            "--execute",
            "--timeout-seconds", "0",
            "--rl-max-update-steps", "20",
            "--rl-sampling-noise-scale", str(noise),
            "--rl-min-reward-std", str(min_std),
        ]
        with (root / f"{arm}.launcher.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=runner.parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = completed.returncode
        record["finished_at"] = datetime.now().strftime("%m月%d日 %H:%M")
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        save(state_path, state)
        if completed.returncode != 0:
            state["status"] = "failed"
            state["failed_arm"] = arm
            save(state_path, state)
            raise SystemExit(completed.returncode)

    state.pop("current_arm", None)
    state["status"] = "complete"
    state["finished_at"] = datetime.now().strftime("%m月%d日 %H:%M")
    save(state_path, state)


if __name__ == "__main__":
    main()
