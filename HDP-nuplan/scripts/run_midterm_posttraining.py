"""Identity-checked B10 post-training controls for the midterm study.

Default is prepare-only. --execute runs exactly one named variant, never an
automatic sweep or evaluation. Existing outputs and historical models are kept.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HDP = ROOT / "HDP-nuplan"
B10_SHA = "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"
MANIFEST_SHA = "1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514"
VARIANTS = ("expert_only", "unfiltered", "safety", "safety_reference")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_args(source, variant, seed, output):
    if variant not in VARIANTS or seed < 0:
        raise ValueError("Unknown variant or negative seed")
    args = {k: v for k, v in source.items()
            if k not in ("state_normalizer", "observation_normalizer")}
    args.update(
        name=f"midterm-{variant}-seed{seed}", save_dir=str(output), seed=seed,
        train_epochs=2, batch_size=2, learning_rate=4e-7, warm_up_epoch=1,
        rl_group_size=32, rl_rollout_steps=6, rl_sampling_noise_scale=0.1,
        rl_buffer_size=1024, rl_buffer_update_epoch=2,
        rl_max_update_steps_per_epoch=500,
        rl_weighting_mode="positive_advantage", rl_center_reward_weights=False,
        rl_normalize_weights=True, rl_min_reward_std=1e-6,
        rl_expert_anchor_weight=0.1, rl_reference_anchor_weight=0.0,
        rl_relative_to_reference=False, rl_reference_noise_scale=0.0,
        rl_rollout_loss_weight=1.0, rl_detach_window_size=0,
        rl_filter_safety_eligible_candidates=True, rl_filter_collision_candidates=True,
        rl_filter_progress_guard_candidates=False,
        reward_objective_mode="legacy", reward_use_nuplan_vehicle_geometry=True,
        reward_safety_gate_require_drivable_area=False,
        reward_safety_gate_threshold=0.3, reward_safety_gate_min_ttc_seconds=1.0,
        reward_progress_guard_weight=5.0,
        rl_trajectory_augmentation_std=0.0, rl_trajectory_augmentation_epochs=0,
        rl_freeze_encoder=True, rl_deterministic_update=True,
    )
    if variant == "expert_only":
        args["rl_rollout_loss_weight"] = 0.0
    elif variant == "unfiltered":
        args["rl_filter_safety_eligible_candidates"] = False
        args["rl_filter_collision_candidates"] = False
    elif variant == "safety_reference":
        args["rl_reference_anchor_weight"] = 0.1
    return args


def validate_manifest(samples, held_out_tokens):
    if len(samples) != 10000 or len(set(samples)) != len(samples):
        raise ValueError("Expected exactly 10000 distinct training entries")
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz")
           for name in samples):
        raise ValueError("Expected flat NPZ cache entries")
    tokens = {Path(name).stem.rsplit("_", 1)[-1] for name in samples}
    overlap = tokens & set(held_out_tokens)
    if overlap:
        raise ValueError(f"Training/Test14 overlap: {sorted(overlap)}")


def build_command(args):
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
               "--nproc-per-node", "1", "--standalone", str(HDP / "train_predictor_rl.py")]
    for key, value in args.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Unsupported CLI value {key}")
        command.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
    return command


def runtime_timeout(seconds):
    """Zero permits an unlimited run; a positive timeout remains opt-in."""
    if seconds < 0:
        raise ValueError("Timeout must be nonnegative")
    return seconds or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="0 (default) means no per-run time limit")
    cli = parser.parse_args()
    timeout = runtime_timeout(cli.timeout_seconds)
    output = cli.output.resolve()
    # Keep outputs on the experiment disk and outside any historical run.
    allowed = (HDP / "tmp/midterm_posttraining").resolve()
    if allowed not in output.parents or output.exists():
        raise ValueError(f"Use a new child directory of {allowed}")
    historical = HDP / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json"
    source = json.loads(historical.read_text())
    args = variant_args(source, cli.variant, cli.seed, output)
    checkpoint, manifest = Path(args["pretrained_model_path"]), Path(args["train_set_list"])
    if sha256(checkpoint) != B10_SHA or sha256(manifest) != MANIFEST_SHA:
        raise ValueError("B10/10k manifest identity changed")
    coverage = HDP / "tmp/test14_full_remote_subset/coverage_validation.json"
    cov = json.loads(coverage.read_text())
    samples = json.loads(manifest.read_text())
    validate_manifest(samples, cov["test14_hard"]["tokens"] + cov["test14_random"]["tokens"])
    if any(not (Path(args["train_set"]) / name).is_file() for name in samples):
        raise ValueError("Training cache is incomplete")
    free = shutil.disk_usage(HDP).free
    if free < 15 * 1024**3:
        raise RuntimeError("Require at least 15 GiB free on the experiment disk")
    if cli.execute:
        gpu = subprocess.check_output([
            "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(gpu.strip()) > 1024:
            raise RuntimeError("GPU0 already uses >1 GiB; refusing competing training")
    command = build_command(args)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    sources = sorted(set((HDP / "hdp_nuplan").rglob("*.py")) |
                     {HDP / "train_predictor_rl.py", Path(__file__).resolve(),
                      HDP / "normalization.json"})
    identities = {}
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        identities[str(path)] = sha256(path)
    record = dict(
        protocol="midterm-b10-controlled-v1", created_at=datetime.now().strftime("%m月%d日 %H:%M"),
        variant=cli.variant, seed=cli.seed, status="prepared", timeout_seconds=cli.timeout_seconds,
        command=command, effective_args=args, source_args=str(historical),
        source_args_sha256=sha256(historical), source_checkpoint_sha256=B10_SHA,
        training_manifest_sha256=MANIFEST_SHA, test14_coverage_sha256=sha256(coverage),
        training_test14_token_overlap=0, free_bytes_before=free, source_sha256=identities,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        limitations=["10k rollout but FIFO retains 1024 groups", "500 batches, 2 scenes per batch",
                     "No closed-loop evaluation in this launcher", "Reference MSE is not policy KL"],
        promotion_authorized=False,
    )
    def save():
        (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    if not cli.execute:
        print(f"Prepared only: {output}; no training started")
        return
    record["status"] = "running"
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    with (output / "train.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record["child_pid"] = process.pid
        save()
        def stop_child():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            code = process.wait(timeout=timeout)
            record.update(returncode=code, status="complete" if code == 0 else "failed")
        except subprocess.TimeoutExpired:
            stop_child()
            record.update(status="timed_out", returncode=process.returncode)
        except KeyboardInterrupt:
            stop_child()
            record.update(status="interrupted", returncode=process.returncode)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            record["finished_at"] = datetime.now().strftime("%m月%d日 %H:%M")
            record["checkpoints"] = {str(p): sha256(p) for p in output.rglob("model_epoch_*.pth")}
            changed = [p for p, digest in identities.items() if sha256(p) != digest]
            record["source_changed_during_run"] = changed
            if record["status"] == "complete" and (changed or len(record["checkpoints"]) != 1):
                record["status"] = "invalid_artifacts"
            save()
    print(f"{record['status']}: {output}")
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
