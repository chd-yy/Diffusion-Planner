import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from monitor_preprocessing import collect_snapshot  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_monitor_separates_complete_and_partial_shards(tmp_path):
    plan_path = tmp_path / "plan.json"
    output_root = tmp_path / "processed"
    raw_root = tmp_path / "raw"
    state_dir = tmp_path / "logs"
    raw_root.mkdir()
    _write_json(
        plan_path,
        {
            "total_scenarios": 3,
            "shards": [
                {"shard_id": "shard_00000", "total_scenarios": 2},
                {"shard_id": "shard_00001", "total_scenarios": 1},
            ],
        },
    )

    complete = output_root / "shard_00000"
    _write_json(complete / "manifest.json", ["a.npz", "b.npz"])
    _write_json(
        complete / "processing_report.json",
        {"status": "complete", "failed": 0, "manifest_count": 2},
    )
    _write_json(complete / "download_report.json", {"retry_count": 2})
    (complete / "cache").mkdir()
    (complete / "cache" / "a.npz").write_bytes(b"a")
    (complete / "cache" / "b.npz").write_bytes(b"b")

    partial = output_root / "shard_00001" / "cache"
    partial.mkdir(parents=True)
    (partial / "partial.npz").write_bytes(b"partial")
    raw_trainval = raw_root / "shard_00001" / "trainval"
    raw_trainval.mkdir(parents=True)
    (raw_trainval / "log.db").write_bytes(b"db")
    (raw_trainval / ".next.db.123.tmp").write_bytes(b"temporary")

    for worker_index, status in enumerate(("running", "stopped_by_user")):
        _write_json(
            state_dir / f"rolling_worker_{worker_index}_of_2_state.json",
            {
                "status": status,
                "next_position": worker_index,
                "selected_shard_count": 1,
                "records": {},
            },
        )

    snapshot = collect_snapshot(
        plan_path,
        output_root,
        raw_root,
        state_dir,
        worker_count=2,
        process_counts={
            "scheduler": 1,
            "downloader": 1,
            "preprocessor": 0,
            "supervised_trainer": 0,
            "rl_trainer": 0,
        },
        min_free_gib=0,
    )

    assert snapshot["phase"] == "running"
    assert snapshot["issues"] == []
    assert snapshot["progress"] == {
        "complete_shards": 1,
        "complete_shard_npz": 2,
        "actual_npz": 3,
        "partial_shards": 1,
        "partial_shard_npz": 1,
        "download_retry_count_archived": 2,
    }
    assert snapshot["raw"]["db_count"] == 1
    assert snapshot["raw"]["temporary_count"] == 1


def test_monitor_flags_false_complete_and_unexpected_training(tmp_path):
    plan_path = tmp_path / "plan.json"
    output_root = tmp_path / "processed"
    raw_root = tmp_path / "raw"
    state_dir = tmp_path / "logs"
    raw_root.mkdir()
    _write_json(
        plan_path,
        {
            "total_scenarios": 1,
            "shards": [{"shard_id": "shard_00000", "total_scenarios": 1}],
        },
    )
    shard = output_root / "shard_00000"
    _write_json(shard / "manifest.json", ["missing.npz"])
    _write_json(
        shard / "processing_report.json",
        {"status": "complete", "failed": 0, "manifest_count": 1},
    )
    (shard / "cache").mkdir()
    _write_json(
        state_dir / "rolling_worker_0_of_1_state.json",
        {"status": "failed", "error": "network", "records": {}},
    )

    snapshot = collect_snapshot(
        plan_path,
        output_root,
        raw_root,
        state_dir,
        worker_count=1,
        process_counts={
            "scheduler": 0,
            "downloader": 0,
            "preprocessor": 0,
            "supervised_trainer": 1,
            "rl_trainer": 0,
        },
        min_free_gib=0,
    )

    assert snapshot["progress"]["complete_shards"] == 0
    assert snapshot["incomplete_shards_with_npz"] == []
    assert snapshot["phase"] == "not_running"
    assert "worker_error" in snapshot["issues"]
    assert "worker_state_unhealthy" in snapshot["issues"]
    assert "unexpected_training_process" in snapshot["issues"]
