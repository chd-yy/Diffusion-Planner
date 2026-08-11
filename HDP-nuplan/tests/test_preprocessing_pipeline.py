import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_dataset_splits import audit_splits  # noqa: E402
from build_preprocessing_plan import build_plan  # noqa: E402
from merge_preprocessing_shards import merge_shards  # noqa: E402
from hdp_nuplan.data_process.data_processor import DataProcessor  # noqa: E402
from hdp_nuplan.data_process.run_utils import sha256_file  # noqa: E402


@dataclass
class FakeScenario:
    token: str
    log_name: str = "log-a"
    scenario_type: str = "test"
    _map_name: str = "test-map"


class FakeProcessor(DataProcessor):
    def process_scenario(self, scenario):
        if scenario.token == "bad":
            raise RuntimeError("synthetic failure")
        return {
            "map_name": scenario._map_name,
            "token": scenario.token,
            "value": np.array([1.0], dtype=np.float32),
        }


def _config(save_path):
    return SimpleNamespace(
        save_path=str(save_path),
        agent_num=32,
        static_objects_num=5,
        lane_len=20,
        lane_num=70,
        route_len=20,
        route_num=25,
    )


def test_data_processor_isolates_failures_and_resumes(tmp_path):
    processor = FakeProcessor(_config(tmp_path))
    scenarios = [FakeScenario("good-a"), FakeScenario("bad"), FakeScenario("good-b")]

    first = processor.work(scenarios, skip_existing=True)

    assert first["processed"] == 2
    assert first["failed"] == 1
    assert first["failures"][0]["token"] == "bad"
    assert sorted(first["completed_files"]) == [
        "test-map_good-a.npz",
        "test-map_good-b.npz",
    ]
    assert not list(tmp_path.glob("*.tmp"))

    # 可读文件会跳过；损坏文件不会被误当作已完成，而会重新原子生成。
    (tmp_path / "test-map_good-b.npz").write_bytes(b"broken")
    second = processor.work(scenarios, skip_existing=True)
    assert second["skipped_existing"] == 1
    assert second["reprocessed_invalid"] == 1
    assert second["processed"] == 1
    assert second["failed"] == 1
    with np.load(tmp_path / "test-map_good-b.npz", allow_pickle=False) as data:
        assert data["value"].item() == 1.0


def test_build_plan_partitions_logs_and_preserves_exact_target(tmp_path):
    logs = ["log-a", "log-b", "log-c", "log-d", "log-e"]
    plan = build_plan(logs, logs_per_shard=2, total_scenarios=12, seed=3407, output_dir=tmp_path)

    assert plan["shard_count"] == 3
    assert sum(item["log_count"] for item in plan["shards"]) == 5
    assert sum(item["total_scenarios"] for item in plan["shards"]) == 12
    recovered_logs = []
    for shard in plan["shards"]:
        recovered_logs.extend(
            json.loads((tmp_path / shard["log_names_json"]).read_text(encoding="utf-8"))
        )
    assert recovered_logs == logs


def _write_shard(root, shard_id, log_name, npz_name, empty_logs=None):
    shard_dir = root / shard_id
    cache_dir = shard_dir / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / npz_name).write_bytes(b"npz-placeholder")
    (shard_dir / "manifest.json").write_text(json.dumps([npz_name]), encoding="utf-8")
    (shard_dir / "sampling_report.json").write_text(
        json.dumps(
            {
                "requested_scenarios": 1,
                "selected_per_log": {log_name: 1},
                "empty_logs": empty_logs or [],
                "log_names": [log_name] + (empty_logs or []),
                "raw_scenarios": 10,
                "unique_scenarios": 9,
                "duplicate_output_names_removed": 1,
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "processing_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shard_id": shard_id,
                "manifest_count": 1,
                "failed": 0,
                "log_names": [log_name] + (empty_logs or []),
                "requested_scenarios": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_formal_metadata(root, shard_id):
    shard_dir = root / shard_id
    manifest_path = shard_dir / "manifest.json"
    sampling_path = shard_dir / "sampling_report.json"
    processing_path = shard_dir / "processing_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    logs = processing["log_names"]
    (shard_dir / "download_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "requested_log_count": len(logs),
                "log_names": logs,
                "retry_count": 1,
                "total_bytes": 123,
                "files": [
                    {
                        "log_name": log_name,
                        "size": 123,
                        "sha256": "a" * 64,
                    }
                    for log_name in logs
                ],
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "manifest": {"sha256": sha256_file(manifest_path)},
                "processing_report": {"sha256": sha256_file(processing_path)},
                "sampling_report": {"sha256": sha256_file(sampling_path)},
                "npz_checksums_included": True,
                "npz_files": {
                    name: sha256_file(shard_dir / "cache" / name)
                    for name in manifest
                },
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "cache_validation_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "manifest_count": len(manifest),
                "npz_count": len(manifest),
                "manifest_sha256": sha256_file(manifest_path),
                "sampling_report_sha256": sha256_file(sampling_path),
            }
        ),
        encoding="utf-8",
    )


def _write_two_shard_plan(root):
    plan_dir = root / "plan"
    plan_dir.mkdir()
    (plan_dir / "shard_00000_logs.json").write_text(
        json.dumps(["log-a"]), encoding="utf-8"
    )
    (plan_dir / "shard_00001_logs.json").write_text(
        json.dumps(["log-b"]), encoding="utf-8"
    )
    plan_path = plan_dir / "preprocessing_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "shard_count": 2,
                "source_log_count": 2,
                "total_scenarios": 2,
                "shards": [
                    {
                        "shard_index": 0,
                        "shard_id": "shard_00000",
                        "log_names_json": "shard_00000_logs.json",
                        "log_count": 1,
                        "total_scenarios": 1,
                    },
                    {
                        "shard_index": 1,
                        "shard_id": "shard_00001",
                        "log_names_json": "shard_00001_logs.json",
                        "log_count": 1,
                        "total_scenarios": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def test_merge_shards_creates_dataset_relative_paths(tmp_path):
    _write_shard(tmp_path, "shard_00000", "log-a", "map_a.npz")
    _write_shard(tmp_path, "shard_00001", "log-b", "map_b.npz")

    manifest, report = merge_shards(tmp_path)

    assert manifest == [
        "shard_00000/cache/map_a.npz",
        "shard_00001/cache/map_b.npz",
    ]
    assert report["status"] == "complete"
    assert report["shard_count"] == 2
    assert report["log_count"] == 2
    assert report["selected_per_log"] == {"log-a": 1, "log-b": 1}


def test_merge_shards_keeps_empty_logs_for_split_audit(tmp_path):
    _write_shard(
        tmp_path,
        "shard_00000",
        "log-a",
        "map_a.npz",
        empty_logs=["short-empty-log"],
    )

    _, report = merge_shards(tmp_path)

    assert report["log_count"] == 2
    assert report["selected_log_count"] == 1
    assert report["empty_log_count"] == 1
    assert report["empty_logs"] == ["short-empty-log"]


def test_formal_merge_requires_exact_plan_and_verifies_hashes(tmp_path):
    shards_root = tmp_path / "processed"
    _write_shard(shards_root, "shard_00000", "log-a", "map_a.npz")
    _write_shard(shards_root, "shard_00001", "log-b", "map_b.npz")
    _write_formal_metadata(shards_root, "shard_00000")
    _write_formal_metadata(shards_root, "shard_00001")
    plan_path = _write_two_shard_plan(tmp_path)

    manifest, report = merge_shards(
        shards_root,
        plan_path=plan_path,
        verify_checksums=True,
        require_cache_validation=True,
    )

    assert len(manifest) == 2
    assert report["status"] == "complete"
    assert report["planned_shard_count"] == 2
    assert report["planned_scenarios"] == 2
    assert report["checksums_verified"] is True
    assert report["download_retry_count"] == 2
    assert report["unique_scenarios"] == 18
    assert report["sampled_fraction_of_unique_candidates"] == pytest.approx(2 / 18)

    (shards_root / "shard_00001" / "cache" / "map_b.npz").write_bytes(
        b"silently-corrupted"
    )
    with pytest.raises(ValueError, match="NPZ checksum mismatch"):
        merge_shards(
            shards_root,
            plan_path=plan_path,
            verify_checksums=True,
            require_cache_validation=True,
        )


def test_formal_merge_rejects_missing_or_unexpected_shard_directories(tmp_path):
    shards_root = tmp_path / "processed"
    _write_shard(shards_root, "shard_00000", "log-a", "map_a.npz")
    plan_path = _write_two_shard_plan(tmp_path)

    with pytest.raises(ValueError, match="shard directory set does not match plan"):
        merge_shards(shards_root, plan_path=plan_path)

    _write_shard(shards_root, "shard_00001", "log-b", "map_b.npz")
    _write_shard(shards_root, "shard_99999", "log-x", "map_x.npz")
    with pytest.raises(ValueError, match="shard directory set does not match plan"):
        merge_shards(shards_root, plan_path=plan_path)


def test_formal_merge_rejects_unlisted_npz(tmp_path):
    shards_root = tmp_path / "processed"
    _write_shard(shards_root, "shard_00000", "log-a", "map_a.npz")
    _write_shard(shards_root, "shard_00001", "log-b", "map_b.npz")
    _write_formal_metadata(shards_root, "shard_00000")
    _write_formal_metadata(shards_root, "shard_00001")
    (shards_root / "shard_00000" / "cache" / "extra.npz").write_bytes(b"extra")
    plan_path = _write_two_shard_plan(tmp_path)

    with pytest.raises(ValueError, match="cache/manifest file set mismatch"):
        merge_shards(shards_root, plan_path=plan_path)


def test_audit_splits_detects_log_and_scenario_leakage(tmp_path):
    train_manifest = tmp_path / "train_manifest.json"
    val_manifest = tmp_path / "val_manifest.json"
    train_report = tmp_path / "train_report.json"
    val_report = tmp_path / "val_report.json"
    train_manifest.write_text(json.dumps(["shard/cache/map_same.npz"]), encoding="utf-8")
    val_manifest.write_text(json.dumps(["other/cache/map_same.npz"]), encoding="utf-8")
    train_report.write_text(json.dumps({"log_names": ["log-shared"]}), encoding="utf-8")
    val_report.write_text(json.dumps({"log_names": ["log-shared"]}), encoding="utf-8")

    result = audit_splits(train_manifest, train_report, val_manifest, val_report)

    assert result["status"] == "failed"
    assert result["overlapping_log_count"] == 1
    assert result["overlapping_npz_count"] == 1


def test_audit_splits_reads_logs_from_selected_per_log(tmp_path):
    train_manifest = tmp_path / "train_manifest.json"
    val_manifest = tmp_path / "val_manifest.json"
    train_report = tmp_path / "train_report.json"
    val_report = tmp_path / "val_report.json"
    train_manifest.write_text(json.dumps(["train.npz"]), encoding="utf-8")
    val_manifest.write_text(json.dumps(["val.npz"]), encoding="utf-8")
    train_report.write_text(
        json.dumps({"selected_per_log": {"train-log": 1}}), encoding="utf-8"
    )
    val_report.write_text(
        json.dumps({"selected_per_log": {"val-log": 1}}), encoding="utf-8"
    )

    result = audit_splits(train_manifest, train_report, val_manifest, val_report)

    assert result["status"] == "passed"
    assert result["train_log_count"] == 1
    assert result["val_log_count"] == 1
