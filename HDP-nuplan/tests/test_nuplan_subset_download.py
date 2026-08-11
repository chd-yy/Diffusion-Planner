import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download_nuplan_log_subset as downloader  # noqa: E402
from download_nuplan_log_subset import (  # noqa: E402
    build_archive_index,
    cleanup_stale_member_temporaries,
    download_log_subset,
    load_or_build_archive_index,
)


def _sqlite_bytes(path: Path) -> bytes:
    import sqlite3

    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sample(value INTEGER)")
    connection.execute("INSERT INTO sample VALUES (1)")
    connection.commit()
    connection.close()
    return path.read_bytes()


def _make_archive(path: Path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def test_subset_download_builds_index_downloads_and_resumes(tmp_path):
    sqlite_a = _sqlite_bytes(tmp_path / "a.sqlite")
    sqlite_b = _sqlite_bytes(tmp_path / "b.sqlite")
    archive_path = tmp_path / "train.zip"
    _make_archive(
        archive_path,
        {
            "nuplan-v1.1/splits/trainval/log-a.db": sqlite_a,
            "nuplan-v1.1/splits/trainval/log-b.db": sqlite_b,
            "README.txt": b"ignored",
        },
    )

    def opener(url):
        return zipfile.ZipFile(url)

    archives = {"test": str(archive_path)}
    index_path = tmp_path / "archive_index.json"
    index = load_or_build_archive_index(index_path, archives, opener)
    assert sorted(index) == ["log-a", "log-b"]
    assert json.loads(index_path.read_text())["log_count"] == 2

    output_dir = tmp_path / "db"
    first = download_log_subset(
        ["log-a", "nested/log-b.db"],
        output_dir,
        index,
        archives,
        archive_opener=opener,
        sqlite_quick_check=True,
    )
    assert first["downloaded"] == 2
    assert first["skipped_existing"] == 0
    assert (output_dir / "log-a.db").read_bytes() == sqlite_a

    second = download_log_subset(
        ["log-a", "log-b"],
        output_dir,
        index,
        archives,
        archive_opener=opener,
        sqlite_quick_check=True,
    )
    assert second["downloaded"] == 0
    assert second["skipped_existing"] == 2


def test_default_archive_opener_reads_complete_local_zip(tmp_path):
    archive_path = tmp_path / "train.zip"
    _make_archive(archive_path, {"split/log.db": b"payload"})

    with downloader._default_archive_opener(str(archive_path)) as archive:
        assert archive.read("split/log.db") == b"payload"

    with pytest.raises(FileNotFoundError, match="local ZIP archive does not exist"):
        downloader._default_archive_opener(str(tmp_path / "missing.zip"))


def test_subset_download_preserves_dots_in_bare_nuplan_log_name(tmp_path):
    log_name = "2021.05.12.19.36.12_veh-35_00005_00204"
    sqlite_payload = _sqlite_bytes(tmp_path / "source.sqlite")
    archive_path = tmp_path / "train.zip"
    _make_archive(
        archive_path,
        {f"nuplan-v1.1/splits/trainval/{log_name}.db": sqlite_payload},
    )

    def opener(url):
        return zipfile.ZipFile(url)

    archives = {"test": str(archive_path)}
    index = build_archive_index(archives, opener)
    report = download_log_subset(
        [log_name],
        tmp_path / "db",
        index,
        archives,
        archive_opener=opener,
    )

    assert report["log_names"] == [log_name]
    assert (tmp_path / "db" / f"{log_name}.db").is_file()


def test_stale_member_cleanup_only_removes_exact_db_dead_pid_files(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "db"
    output_dir.mkdir()
    destination = output_dir / "log.db"
    stale = output_dir / ".log.db.111.tmp"
    active = output_dir / ".log.db.222.deflate.tmp"
    unrelated = output_dir / ".other.db.111.tmp"
    for path in (stale, active, unrelated):
        path.write_bytes(b"partial")

    monkeypatch.setattr(downloader, "_pid_is_alive", lambda pid: pid == 222)

    assert cleanup_stale_member_temporaries(destination) == 1
    assert not stale.exists()
    assert active.is_file()
    assert unrelated.is_file()


def test_archive_index_rejects_duplicate_logs(tmp_path):
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    _make_archive(archive_a, {"one/log.db": b"first"})
    _make_archive(archive_b, {"two/log.db": b"second"})

    def opener(url):
        return zipfile.ZipFile(url)

    try:
        build_archive_index({"a": str(archive_a), "b": str(archive_b)}, opener)
    except ValueError as error:
        assert "duplicate log DBs" in str(error)
    else:
        raise AssertionError("duplicate log names must be rejected")


def test_subset_download_reopens_archive_and_retries_failed_member(tmp_path, monkeypatch):
    sqlite_payload = _sqlite_bytes(tmp_path / "source.sqlite")
    archive_path = tmp_path / "train.zip"
    _make_archive(archive_path, {"split/log.db": sqlite_payload})

    open_count = 0

    def opener(url):
        nonlocal open_count
        open_count += 1
        return zipfile.ZipFile(url)

    index = build_archive_index({"test": str(archive_path)}, opener)
    # build_archive_index 自己打开过一次；下面只统计 download 阶段的重新打开。
    open_count = 0
    original_copy = downloader._copy_member_atomic
    copy_attempts = 0

    def flaky_copy(archive, member, destination):
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            raise TimeoutError("simulated stalled Range read")
        return original_copy(archive, member, destination)

    monkeypatch.setattr(downloader, "_copy_member_atomic", flaky_copy)
    report = download_log_subset(
        ["log"],
        tmp_path / "db",
        index,
        {"test": str(archive_path)},
        archive_opener=opener,
        max_member_retries=1,
        retry_delay_seconds=0,
    )

    assert report["downloaded"] == 1
    assert report["retry_count"] == 1
    assert report["files"][0]["download_attempts"] == 2
    assert open_count == 2
    assert (tmp_path / "db" / "log.db").read_bytes() == sqlite_payload


def test_curl_member_copier_extracts_exact_range_and_validates_crc(tmp_path):
    sqlite_payload = _sqlite_bytes(tmp_path / "source.sqlite")
    archive_path = tmp_path / "train.zip"
    member = "split/log.db"
    _make_archive(archive_path, {member: sqlite_payload})
    destination = tmp_path / "db" / "log.db"
    observed = {}

    def local_range_runner(command, check):
        assert check is True
        data_range = command[command.index("--range") + 1]
        output = Path(command[command.index("--output") + 1])
        start, end = (int(value) for value in data_range.split("-"))
        archive_bytes = archive_path.read_bytes()
        output.write_bytes(archive_bytes[start : end + 1])
        observed["range"] = (start, end)

    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        downloader._copy_member_curl_atomic(
            str(archive_path),
            archive,
            member,
            destination,
            connect_timeout_seconds=10,
            low_speed_limit_bps=1024,
            low_speed_time_seconds=5,
            command_runner=local_range_runner,
        )

    assert destination.read_bytes() == sqlite_payload
    assert observed["range"][1] - observed["range"][0] + 1 == info.compress_size
    assert not list(destination.parent.glob(".*.tmp"))


def test_curl_member_copier_removes_partial_files_after_slow_transfer(tmp_path):
    sqlite_payload = _sqlite_bytes(tmp_path / "source.sqlite")
    archive_path = tmp_path / "train.zip"
    member = "split/log.db"
    _make_archive(archive_path, {member: sqlite_payload})
    destination = tmp_path / "db" / "log.db"

    def slow_runner(command, check):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"partial compressed bytes")
        raise subprocess.CalledProcessError(28, command)

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(subprocess.CalledProcessError):
            downloader._copy_member_curl_atomic(
                str(archive_path),
                archive,
                member,
                destination,
                connect_timeout_seconds=10,
                low_speed_limit_bps=1024,
                low_speed_time_seconds=5,
                command_runner=slow_runner,
            )

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))
