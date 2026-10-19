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
import random
import shutil
import signal
import subprocess
import sys

from scripts.storage_paths import require_allowed_output


ROOT = Path(__file__).resolve().parents[2]
HDP = ROOT / "HDP-nuplan"
B10_SHA = "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce"
MANIFEST_SHA = "1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514"
FULL_MINITRAIN_COUNT = 306801
FULL_MINITRAIN_MANIFEST_SHA = (
    "a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0"
)
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
        rl_candidate_pool_profile="", rl_candidate_selection_profile="",
        rl_candidate_selection_mode="geometry",
        rl_task_reference_selection_profile="",
        rl_component_dispersion_gate=False, rl_min_component_dispersion=0.01,
        rl_task_preserving_update_mask=False,
        rl_route_cost_tolerance=1e-4, rl_progress_guard_tolerance=0.005,
        rl_positive_advantage_margin=0.0,
        rl_min_active_group_fraction=0.0,
        rl_min_target_group_retention=0.0,
        rl_min_targets_per_retained_group=0.0,
        rl_min_target_group_count=0,
        rl_update_target_fraction=0.0,
        rl_guided_safe_set_protocol=False,
        rl_buffer_size=1024, rl_buffer_update_epoch=2,
        rl_replay_backend="memory", rl_replay_directory="",
        rl_replay_shard_size=1000, rl_replay_cache_shards=2,
        rl_update_cover_all_targets=False,
        rl_max_update_steps_per_epoch=500,
        rl_update_checkpoint_steps="",
        rl_weighting_mode="positive_advantage", rl_center_reward_weights=False,
        rl_normalize_weights=True, rl_min_reward_std=1e-6,
        rl_expert_anchor_weight=0.1, rl_reference_anchor_weight=0.0,
        rl_relative_to_reference=False, rl_reference_noise_scale=0.0,
        rl_rollout_loss_weight=1.0, rl_detach_window_size=0,
        rl_filter_safety_eligible_candidates=True, rl_filter_collision_candidates=True,
        rl_filter_progress_guard_candidates=False,
        reward_objective_mode="legacy", reward_use_nuplan_vehicle_geometry=True,
        reward_follow_assignment_mode="expert_reference",
        reward_follow_reference_full_weight_steps=8,
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


def validate_manifest(samples, held_out_tokens, expected_count=10000):
    if len(samples) != expected_count or len(set(samples)) != len(samples):
        raise ValueError(
            f"Expected exactly {expected_count} distinct training entries"
        )
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
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        # --key=value is unambiguous even when a string value starts with '-'
        # (the guided deviation ladder is the concrete case).
        command.append(f"--{key}={rendered}")
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
    parser.add_argument("--rl-max-update-steps", type=int, default=500,
                        help="Maximum optimizer steps; 0 lets full target coverage determine it")
    parser.add_argument("--rl-update-checkpoint-steps", type=str, default="",
                        help="Comma-separated actual optimizer-step milestones")
    parser.add_argument("--rl-sampling-noise-scale", type=float, default=0.1)
    parser.add_argument("--rl-min-reward-std", type=float, default=1e-6)
    parser.add_argument("--rl-candidate-pool-profile", type=str, default="")
    parser.add_argument("--rl-candidate-selection-profile", type=str, default="")
    parser.add_argument(
        "--chapter4-constraint-protocol",
        action="store_true",
        help="Apply the pre-registered F25/quality/constraint/margin-0.1 protocol.",
    )
    parser.add_argument(
        "--chapter4-guided-safe-set-protocol",
        action="store_true",
        help="Apply the 88-to-32 guided safe-set protocol without high-noise proposals.",
    )
    parser.add_argument(
        "--smoke-scenes",
        type=int,
        default=0,
        help="Deterministic training-manifest subset; 0 keeps the complete manifest.",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=None,
        help="Optional audited manifest backed by the existing full cache.",
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=None,
        help="Cache backing --training-manifest; no NPZ files are copied.",
    )
    parser.add_argument(
        "--rl-update-target-fraction",
        type=float,
        default=None,
        help="Target-bearing replay fraction per update batch; guided protocol defaults to 0.5.",
    )
    cli = parser.parse_args()
    timeout = runtime_timeout(cli.timeout_seconds)
    output, storage_backend = require_allowed_output(
        cli.output,
        HDP / "tmp/midterm_posttraining",
        "training_outputs",
    )
    if output.exists():
        raise ValueError(f"Refusing to reuse existing output: {output}")
    historical = HDP / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json"
    source = json.loads(historical.read_text())
    args = variant_args(source, cli.variant, cli.seed, output)
    if cli.chapter4_constraint_protocol and cli.chapter4_guided_safe_set_protocol:
        raise ValueError("chapter4 protocols are mutually exclusive")
    if cli.rl_max_update_steps < 0:
        raise ValueError("--rl-max-update-steps must be nonnegative")
    if cli.smoke_scenes < 0:
        raise ValueError("--smoke-scenes must be nonnegative")
    if cli.smoke_scenes and cli.smoke_scenes < 64:
        raise ValueError("smoke protocol requires at least 64 scenes")
    if bool(cli.training_manifest) != bool(cli.training_cache):
        raise ValueError("--training-manifest and --training-cache must be set together")
    if cli.rl_update_target_fraction is not None and not (
        0 <= cli.rl_update_target_fraction <= 1
    ):
        raise ValueError("--rl-update-target-fraction must be in [0,1]")
    if cli.rl_sampling_noise_scale <= 0 or cli.rl_min_reward_std < 0:
        raise ValueError("noise scale must be positive and reward std nonnegative")
    if bool(cli.rl_candidate_pool_profile) != bool(
        cli.rl_candidate_selection_profile
    ):
        raise ValueError(
            "candidate pool and selection profiles must be set together"
        )
    args.update(
        rl_max_update_steps_per_epoch=cli.rl_max_update_steps,
        rl_update_checkpoint_steps=cli.rl_update_checkpoint_steps,
        rl_sampling_noise_scale=cli.rl_sampling_noise_scale,
        rl_min_reward_std=cli.rl_min_reward_std,
        rl_candidate_pool_profile=cli.rl_candidate_pool_profile,
        rl_candidate_selection_profile=cli.rl_candidate_selection_profile,
    )
    if cli.chapter4_constraint_protocol:
        if cli.rl_candidate_pool_profile or cli.rl_candidate_selection_profile:
            raise ValueError(
                "chapter4 constraint protocol fixes candidate profiles; do not override them"
            )
        args.update(
            rl_candidate_pool_profile=(
                "0.1x40+0.5x64+1.0x32+1.5x32+2.0x32"
            ),
            rl_candidate_selection_profile=(
                "0.1x10+0.5x14+1.0x3+1.5x3+2.0x2"
            ),
            rl_candidate_selection_mode="quality",
            rl_task_reference_selection_profile=(
                "0.1x11+0.5x16+1.0x2+1.5x2+2.0x1"
            ),
            rl_component_dispersion_gate=True,
            rl_min_component_dispersion=0.01,
            rl_task_preserving_update_mask=True,
            rl_route_cost_tolerance=1e-4,
            rl_progress_guard_tolerance=0.005,
            rl_positive_advantage_margin=0.1,
            rl_min_active_group_fraction=0.40,
            rl_min_target_group_retention=0.90,
            rl_min_targets_per_retained_group=2.0,
            rl_weighting_mode="constraint_uniform_positive",
            rl_normalize_weights=False,
            rl_min_reward_std=1e-6,
            reward_objective_mode="nuplan_aligned",
            reward_safety_gate_require_drivable_area=True,
        )
    if cli.chapter4_guided_safe_set_protocol:
        if cli.rl_candidate_pool_profile or cli.rl_candidate_selection_profile:
            raise ValueError("guided safe-set protocol fixes its 88-proposal pool")
        args.update(
            rl_guided_safe_set_protocol=True,
            rl_guided_reference_count=16,
            rl_guided_policy_count=16,
            rl_guided_medium_count=24,
            rl_guided_count=32,
            rl_guided_low_noise_scale=0.1,
            rl_guided_medium_noise_scale=0.5,
            rl_guidance_scale=0.01,
            rl_guidance_deviations="-0.5,-0.25,-0.125,0,0.125,0.25,0.5",
            rl_guidance_progress_threshold=5.358,
            rl_guidance_corridor_half_width=3.5,
            rl_guidance_corridor_range=15.0,
            rl_safe_set_natural_slots=8,
            rl_safe_set_medium_elite_slots=8,
            rl_safe_set_medium_diverse_slots=8,
            rl_safe_set_global_elite_slots=8,
            rl_reference_reward_margin=0.0,
            rl_component_dispersion_gate=True,
            rl_min_component_dispersion=0.01,
            rl_task_preserving_update_mask=False,
            rl_route_cost_tolerance=1e-4,
            rl_progress_guard_tolerance=0.005,
            rl_positive_advantage_margin=0.1,
            rl_min_active_group_fraction=0.40,
            rl_min_target_group_retention=0.90,
            rl_min_targets_per_retained_group=2.0,
            rl_weighting_mode="constraint_uniform_positive",
            rl_normalize_weights=False,
            rl_min_reward_std=1e-6,
            rl_relative_to_reference=True,
            rl_reference_anchor_weight=0.1,
            rl_filter_safety_eligible_candidates=False,
            rl_filter_collision_candidates=False,
            reward_objective_mode="nuplan_aligned",
            reward_safety_gate_require_drivable_area=True,
            reward_safety_gate_threshold=0.0,
            reward_safety_gate_min_ttc_seconds=0.0,
        )
    args["rl_update_target_fraction"] = (
        cli.rl_update_target_fraction
        if cli.rl_update_target_fraction is not None
        else (0.5 if cli.chapter4_guided_safe_set_protocol else 0.0)
    )
    if (
        cli.rl_max_update_steps != 500
        or cli.rl_sampling_noise_scale != 0.1
        or cli.rl_min_reward_std != 1e-6
        or cli.rl_candidate_pool_profile
        or cli.chapter4_constraint_protocol
        or cli.chapter4_guided_safe_set_protocol
    ):
        std_tag = f"{cli.rl_min_reward_std:g}".replace(".", "p").replace("-", "m")
        noise_tag = f"{cli.rl_sampling_noise_scale:g}".replace(".", "p")
        args["name"] = (
            f"chapter4-{cli.variant}-noise{noise_tag}-std{std_tag}-"
            f"steps{cli.rl_max_update_steps}-seed{cli.seed}"
        )
        if cli.rl_candidate_pool_profile:
            args["name"] += "-candidate-repair"
        if cli.chapter4_constraint_protocol:
            args["name"] += "-constraint-m01"
        if cli.chapter4_guided_safe_set_protocol:
            args["name"] += "-guided-safeset88"
    checkpoint = Path(args["pretrained_model_path"])
    if sha256(checkpoint) != B10_SHA:
        raise ValueError("B10 checkpoint identity changed")
    if cli.training_manifest:
        manifest = cli.training_manifest.resolve()
        training_cache = cli.training_cache.resolve()
        manifest_digest = sha256(manifest)
        samples = json.loads(manifest.read_text())
        expected_count = len(samples)
        if expected_count == FULL_MINITRAIN_COUNT:
            if manifest_digest != FULL_MINITRAIN_MANIFEST_SHA:
                raise ValueError("Full mini-train manifest identity changed")
            dataset_tag = "full306801"
        elif expected_count == 30000:
            dataset_tag = "pilot30k"
        else:
            raise ValueError(
                "Custom training manifest must be the audited 30k pilot or "
                f"the complete {FULL_MINITRAIN_COUNT}-scene mini-train manifest"
            )
        args["train_set_list"] = str(manifest)
        args["train_set"] = str(training_cache)
        args["name"] += f"-{dataset_tag}"
    else:
        manifest = Path(args["train_set_list"])
        training_cache = Path(args["train_set"])
        expected_count = 10000
        if sha256(manifest) != MANIFEST_SHA:
            raise ValueError("10k manifest identity changed")
        samples = json.loads(manifest.read_text())
    if cli.smoke_scenes > expected_count:
        raise ValueError("--smoke-scenes exceeds the selected manifest")
    coverage = HDP / "tmp/test14_full_remote_subset/coverage_validation.json"
    cov = json.loads(coverage.read_text())
    validate_manifest(
        samples,
        cov["test14_hard"]["tokens"] + cov["test14_random"]["tokens"],
        expected_count=expected_count,
    )
    if any(not (training_cache / name).is_file() for name in samples):
        raise ValueError("Training cache is incomplete")
    # The replay must retain the complete rollout.  Increasing the manifest
    # without this assignment would silently keep only the final FIFO window.
    args["rl_buffer_size"] = len(samples)
    use_disk_replay = expected_count == FULL_MINITRAIN_COUNT
    if use_disk_replay:
        args.update(
            rl_replay_backend="disk_sharded",
            rl_replay_directory=str(output / "replay"),
            rl_replay_shard_size=1000,
            rl_replay_cache_shards=2,
            rl_update_cover_all_targets=True,
        )
        if args["rl_update_target_fraction"] <= 0:
            raise ValueError("Full target coverage requires target-aware sampling")
        # Full-data target-aware updates consume only exact target-bearing groups.
        # Population *fraction* is therefore diagnostic; the fail-closed support
        # contract is an absolute 30k effective groups plus per-group quality.
        args.update(
            rl_min_active_group_fraction=0.0,
            rl_min_target_group_retention=0.0,
            rl_min_target_group_count=(1 if cli.smoke_scenes else 30000),
            rl_min_targets_per_retained_group=2.0,
        )
    free = shutil.disk_usage(output.parent).free
    if free < 15 * 1024**3:
        raise RuntimeError("Require at least 15 GiB free on the experiment disk")
    if cli.execute:
        gpu = subprocess.check_output([
            "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(gpu.strip()) > 1024:
            raise RuntimeError("GPU0 already uses >1 GiB; refusing competing training")
    output.mkdir(parents=True, exist_ok=False)
    smoke_manifest_sha = None
    if cli.smoke_scenes:
        selected_samples = random.Random(cli.seed).sample(samples, cli.smoke_scenes)
        smoke_manifest = output / "smoke_train_manifest.json"
        smoke_manifest.write_text(
            json.dumps(selected_samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args["train_set_list"] = str(smoke_manifest)
        args["rl_buffer_size"] = cli.smoke_scenes
        smoke_manifest_sha = sha256(smoke_manifest)
    command = build_command(args)
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
        storage_backend=storage_backend,
        chapter4_constraint_protocol=cli.chapter4_constraint_protocol,
        chapter4_guided_safe_set_protocol=cli.chapter4_guided_safe_set_protocol,
        smoke_scenes=cli.smoke_scenes,
        smoke_manifest_sha256=smoke_manifest_sha,
        controlled_overrides={
            "rl_sampling_noise_scale": cli.rl_sampling_noise_scale,
            "rl_min_reward_std": cli.rl_min_reward_std,
            "rl_max_update_steps_per_epoch": cli.rl_max_update_steps,
            "rl_update_checkpoint_steps": cli.rl_update_checkpoint_steps,
            "rl_candidate_pool_profile": cli.rl_candidate_pool_profile,
            "rl_candidate_selection_profile": cli.rl_candidate_selection_profile,
            "effective_rl_candidate_pool_profile": args["rl_candidate_pool_profile"],
            "effective_rl_candidate_selection_profile": args["rl_candidate_selection_profile"],
        },
        command=command, effective_args=args, source_args=str(historical),
        source_args_sha256=sha256(historical), source_checkpoint_sha256=B10_SHA,
        training_manifest_sha256=sha256(manifest),
        training_manifest_count=len(samples),
        training_cache=str(training_cache),
        replay_buffer_capacity=args["rl_buffer_size"],
        replay_backend=args["rl_replay_backend"],
        replay_directory=args["rl_replay_directory"],
        replay_shard_size=args["rl_replay_shard_size"],
        replay_update_cover_all_targets=args["rl_update_cover_all_targets"],
        replay_update_target_fraction=args["rl_update_target_fraction"],
        test14_coverage_sha256=sha256(coverage),
        training_test14_token_overlap=0, free_bytes_before=free, source_sha256=identities,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        limitations=(
            ([f"Functional smoke uses {cli.smoke_scenes} deterministic training scenes"]
             if cli.smoke_scenes else
             [f"Full rollout retains all {len(samples)} groups"])
            + [
                f"At most {cli.rl_max_update_steps} optimizer steps",
                "No closed-loop evaluation in this launcher",
                "Reference MSE is not policy KL",
                "Training reward has no explicit speeding event label",
                "Collision/DA are trajectory-level gates without survival credit",
            ]
        ),
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
