"""Reproduce the historical B10 post-training protocol with explicit safe updates.

Outputs are isolated; Test14 diagnosis scenarios are not training samples.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-geometry", action="store_true",
                        help="ablation only: preserve the historical rear-axle/center mismatch")
    cli = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    hdp = root / "HDP-nuplan"
    historical = hdp / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json"
    args = json.loads(historical.read_text())
    output = cli.output.resolve()
    args.update(
        name="hdp-rl-safety-repair-controlled", save_dir=str(output),
        rl_center_reward_weights=False, rl_weighting_mode="positive_advantage",
        rl_filter_safety_eligible_candidates=True,
        rl_filter_collision_candidates=True,
        rl_filter_progress_guard_candidates=False,
        reward_safety_gate_require_drivable_area=False,
        rl_reference_anchor_weight=0.0, reward_objective_mode="legacy",
        reward_use_nuplan_vehicle_geometry=not cli.legacy_geometry,
    )
    checkpoint = Path(args["pretrained_model_path"])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce":
        raise RuntimeError("B10 source checkpoint identity mismatch")
    manifest = Path(args["train_set_list"])
    samples = json.loads(manifest.read_text())
    if len(samples) != 10000:
        raise RuntimeError("Expected the historical 10k training manifest")
    from diagnose_rl_safety_regressions import TOKENS
    if any(token in str(sample) for sample in samples for token in TOKENS):
        raise RuntimeError("Diagnosis token leaked into training manifest")
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
               "--nproc-per-node", "1", "--standalone", str(hdp / "train_predictor_rl.py")]
    for key, value in args.items():
        if key in ("state_normalizer", "observation_normalizer"):
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Unsupported argument {key}: {type(value)}")
        command.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
    output.mkdir(parents=True, exist_ok=False)
    record = dict(created_at=datetime.now().strftime("%m月%d日 %H:%M"),
                  source_args=str(historical), source_checkpoint_sha256=sha,
                  source_args_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),
                  training_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  command=command, status="running")
    def save():
        (output / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    with (output / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode,
                  finished_at=datetime.now().strftime("%m月%d日 %H:%M"),
                  checkpoints=[str(p) for p in output.rglob("model_epoch_*.pth")])
    save()
    if result.returncode:
        raise RuntimeError(f"Training failed; inspect {output / 'train.log'}")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
