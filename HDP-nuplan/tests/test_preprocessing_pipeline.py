import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_dataset_splits import audit_splits  # noqa: E402
from build_preprocessing_plan import build_plan  # noqa: E402
from merge_preprocessing_shards import merge_shards  # noqa: E402
from hdp_nuplan.data_process.data_processor import DataProcessor  # noqa: E402


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


def _write_shard(root, shard_id, log_name, npz_name):
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
                "log_names": [log_name],
            }
        ),
        encoding="utf-8",
    )


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
