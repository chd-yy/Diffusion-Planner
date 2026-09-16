"""Cheap tests of full paired evaluation identity and resume contracts."""

import copy
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_rl_safety_geometry_multiseed import (
    artifact_state, check_identity, command_for, sha256, validate_manifest,
)
from manage_test14_chunk_results import archived_paths


def manifest():
    result = {"chunk_size": 40, "benchmarks": {}}
    for name, count in [("test14-hard", 272), ("test14-random", 261)]:
        tokens = [f"{index:016x}" for index in range(count)]
        chunks = [{"index": index, "tokens": tokens[start:start + 40],
                   "count": len(tokens[start:start + 40])}
                  for index, start in enumerate(range(0, count, 40))]
        result["benchmarks"][name] = {"count": count, "tokens": tokens, "chunks": chunks}
    return result


def test_expected_full_coverage():
    validate_manifest(manifest())


@pytest.mark.parametrize("fault", ["duplicate", "order", "boundary", "size"])
def test_reject_changed_pairing_protocol(fault):
    value = copy.deepcopy(manifest())
    benchmark = value["benchmarks"]["test14-hard"]
    if fault == "duplicate":
        benchmark["chunks"][0]["tokens"][1] = benchmark["chunks"][0]["tokens"][0]
    elif fault == "order":
        benchmark["tokens"].reverse()
    elif fault == "boundary":
        benchmark["chunks"][0]["count"] -= 1
    else:
        value["chunk_size"] = 20
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_pair_commands_keep_seed_and_single_worker(tmp_path):
    chunk = {"filter": "test14-hard-chunk-002"}
    commands = [command_for(model, "test14-hard", 2, chunk, tmp_path)
                for model in ["b10", "safety_geometry"]]
    for command in commands:
        assert "seed=2" in command
        assert "worker.max_workers=1" in command
        assert "worker.use_process_pool=false" in command
        assert "scenario_filter=test14-hard-chunk-002" in command
        assert "~callback.simulation_log_callback" in command
    assert next(x for x in commands[0] if x.startswith("experiment_uid=")) != next(
        x for x in commands[1] if x.startswith("experiment_uid="))


def test_missing_artifacts_can_run_but_partial_archive_is_rejected(tmp_path):
    chunk = {"index": 0, "tokens": ["a"]}
    assert artifact_state(tmp_path, {}, chunk) == "missing"
    report, _ = archived_paths(tmp_path, 0)
    report.parent.mkdir(parents=True)
    pd.DataFrame({"scenario_name": ["a"], "succeeded": [True]}).to_parquet(report)
    with pytest.raises(FileNotFoundError):
        artifact_state(tmp_path, {}, chunk)


def test_valid_archives_are_read_only_and_failed_runner_is_rejected(tmp_path):
    chunk = {"index": 0, "tokens": ["a"]}
    report, aggregate = archived_paths(tmp_path, 0)
    report.parent.mkdir(parents=True)
    aggregate.parent.mkdir(parents=True)
    pd.DataFrame({"scenario_name": ["a"], "succeeded": [True]}).to_parquet(report)
    pd.DataFrame({"scenario": ["a"], "log_name": ["log"]}).to_parquet(aggregate)
    original = [sha256(path) for path in (report, aggregate)]
    assert artifact_state(tmp_path, {}, chunk) == "complete"
    assert original == [sha256(path) for path in (report, aggregate)]
    pd.DataFrame({"scenario_name": ["a"], "succeeded": [False]}).to_parquet(report)
    with pytest.raises(RuntimeError, match="Failed simulations"):
        artifact_state(tmp_path, {}, chunk)


def test_source_drift_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n")
    identity = {str(path): sha256(path)}
    check_identity(identity)
    path.write_text("seed: 2\n")
    with pytest.raises(RuntimeError, match="Input changed"):
        check_identity(identity)
