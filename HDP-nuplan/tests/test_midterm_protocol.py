from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_midterm_posttraining import build_command, validate_manifest, variant_args, runtime_timeout


def test_unlimited_and_explicit_timeout():
    assert runtime_timeout(0) is None
    assert runtime_timeout(86400) == 86400
    with pytest.raises(ValueError): runtime_timeout(-1)


def test_variants_only_change_prescribed_algorithmic_factors():
    base = variant_args({}, "safety", 2026, Path("run"))
    expected = {"expert_only": {"rl_rollout_loss_weight"},
                "unfiltered": {"rl_filter_safety_eligible_candidates", "rl_filter_collision_candidates"},
                "safety_reference": {"rl_reference_anchor_weight"}}
    for variant, keys in expected.items():
        args = variant_args({}, variant, 2026, Path("run"))
        changed = {k for k in base if base[k] != args[k]} - {"name"}
        assert changed == keys


def test_no_centering_and_fixed_geometry_in_all_arms():
    for variant in ("expert_only", "unfiltered", "safety", "safety_reference"):
        args = variant_args({}, variant, 42, Path("run"))
        assert args["reward_use_nuplan_vehicle_geometry"] is True
        assert args["rl_center_reward_weights"] is False
        assert args["rl_relative_to_reference"] is False
        assert args["rl_max_update_steps_per_epoch"] == 500


def test_manifest_rejects_all_heldout_overlap_not_only_diagnosis_cases():
    samples = [f"map_{i:016x}.npz" for i in range(10000)]
    validate_manifest(samples, ["ffffffffffffffff"])
    with pytest.raises(ValueError): validate_manifest(samples, ["0000000000000007"])
    with pytest.raises(ValueError): validate_manifest(samples[:-1], [])
    with pytest.raises(ValueError): validate_manifest(samples[:-1] + [samples[0]], [])


def test_command_booleans_and_invalid_arguments():
    command = build_command(dict(rl_center_reward_weights=False, seed=42))
    assert command[-4:] == ["--rl_center_reward_weights", "false", "--seed", "42"]
    with pytest.raises(ValueError): build_command({"invalid": None})
    with pytest.raises(ValueError): variant_args({}, "not_a_variant", 1, Path("run"))
