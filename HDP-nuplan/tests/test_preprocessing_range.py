import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_preprocessing_range import (  # noqa: E402
    archive_download_report,
    cleanup_raw_trainval,
    select_worker_shards,
)


def test_worker_shard_assignment_is_disjoint_and_complete():
    shards = [
        {"shard_index": index, "shard_id": f"shard_{index:05d}"}
        for index in range(11)
    ]
    assignments = [
        select_worker_shards(shards, 0, None, worker, 3)
        for worker in range(3)
    ]
    recovered = sorted(
        shard["shard_index"]
        for assignment in assignments
        for shard in assignment
    )
    assert recovered == list(range(11))
    assert [item["shard_index"] for item in assignments[0]] == [0, 3, 6, 9]
    assert [item["shard_index"] for item in assignments[1]] == [1, 4, 7, 10]
    assert [item["shard_index"] for item in assignments[2]] == [2, 5, 8]


def test_worker_shard_assignment_respects_half_open_range():
    shards = [
        {"shard_index": index, "shard_id": f"shard_{index:05d}"}
        for index in range(10)
    ]
    selected = select_worker_shards(shards, 3, 9, worker_index=1, worker_count=2)
    assert [item["shard_index"] for item in selected] == [3, 5, 7]


def test_worker_shard_assignment_respects_explicit_indices():
    shards = [
        {"shard_index": index, "shard_id": f"shard_{index:05d}"}
        for index in range(12)
    ]
    selected = select_worker_shards(
        shards,
        0,
        None,
        worker_index=0,
        worker_count=1,
        include_indices=[2, 5, 9],
    )
    assert [item["shard_index"] for item in selected] == [2, 5, 9]


def test_archive_report_before_exact_raw_cleanup(tmp_path):
    raw_root = tmp_path / "raw"
    raw_shard = raw_root / "shard_00007"
    trainval = raw_shard / "trainval"
    trainval.mkdir(parents=True)
    (trainval / "log.db").write_bytes(b"temporary raw DB")
    report_payload = {"status": "complete", "files": [{"sha256": "abc"}]}
    (raw_shard / "download_report.json").write_text(
        json.dumps(report_payload), encoding="utf-8"
    )
    processed_shard = tmp_path / "processed" / "shard_00007"

    archived = archive_download_report(raw_shard, processed_shard)
    removed = cleanup_raw_trainval(raw_root, "shard_00007")

    assert json.loads(archived.read_text(encoding="utf-8")) == report_payload
    assert removed is True
    assert not trainval.exists()
    assert (raw_shard / "download_report.json").is_file()


def test_cleanup_missing_trainval_is_idempotent(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "shard_00003").mkdir(parents=True)
    assert cleanup_raw_trainval(raw_root, "shard_00003") is False
