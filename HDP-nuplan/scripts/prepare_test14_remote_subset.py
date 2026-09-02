#!/usr/bin/env python3
"""Build a disk-efficient NuPlan Test14 DB subset from the official remote ZIP.

The official test archive is too large to keep compressed and extracted on the
local disk at the same time.  This utility therefore has three resumable phases:

1. ``scan``: range-download one ZIP member at a time, query its scenario rows,
   persist a compact SQLite index, and remove the temporary DB;
2. ``select``: reproduce nuPlan's deterministic ``test14-random`` filtering and
   locate every DB required by ``test14-hard`` and ``test14-random``;
3. ``extract``: range-download only those required DB members permanently.

Every extracted member is checked against the ZIP central-directory size and
CRC32.  Parallel scan workers do not affect sample order: selection always uses
the same sorted DB-name order used by ``NuPlanScenarioBuilder``.
"""

from __future__ import annotations

import argparse
import binascii
import json
import logging
import os
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db


DEFAULT_URL = (
    "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/"
    "public/nuplan-v1.1/nuplan-v1.1_test.zip"
)
DEFAULT_ARCHIVE_SIZE = 95_919_476_643
RANDOM_TARGET_PER_TYPE = 20
TIMESTAMP_THRESHOLD_US = 15_000_000
ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
LOGGER = logging.getLogger("prepare_test14_remote_subset")
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ZipEntry:
    """Metadata needed to extract one remote ZIP member."""

    name: str
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    crc32: int
    method: int


@dataclass(frozen=True)
class ScenarioRow:
    """Compact scenario metadata required to reproduce Test14 filtering."""

    row_index: int
    token: str
    timestamp: int
    map_name: str
    scenario_type: str


def _request_range(url: str, start: int, end: int, timeout_s: int = 120):
    """Open one inclusive HTTP byte range and require a partial response."""

    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "HDP-nuplan-test14-subset/1.0",
        },
    )
    response = urllib.request.urlopen(request, timeout=timeout_s)
    if response.status != 206:
        response.close()
        raise RuntimeError(f"Server ignored byte range {start}-{end}: HTTP {response.status}")
    expected = end - start + 1
    content_range = response.headers.get("Content-Range", "")
    if not content_range.startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
    if response.headers.get("Content-Length") not in (None, str(expected)):
        response.close()
        raise RuntimeError(
            f"Unexpected Content-Length for range {start}-{end}: "
            f"{response.headers.get('Content-Length')}"
        )
    return response


def read_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Read a small remote byte range with bounded retries."""

    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            with _request_range(url, start, end) as response:
                payload = response.read()
            if len(payload) != expected:
                raise RuntimeError(f"Short range read: expected {expected}, got {len(payload)}")
            return payload
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if attempt == retries:
                raise
            delay = min(2**attempt, 30)
            LOGGER.warning("Range read failed (%s); retrying in %ss", error, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _zip64_values(
    extra: bytes,
    uncompressed_size: int,
    compressed_size: int,
    local_header_offset: int,
) -> Tuple[int, int, int]:
    """Resolve ZIP64 placeholders in one central-directory record."""

    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, payload_size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + payload_size]
        cursor += 4 + payload_size
        if header_id != 0x0001:
            continue
        value_cursor = 0
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = struct.unpack_from("<Q", payload, value_cursor)[0]
            value_cursor += 8
        if compressed_size == 0xFFFFFFFF:
            compressed_size = struct.unpack_from("<Q", payload, value_cursor)[0]
            value_cursor += 8
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = struct.unpack_from("<Q", payload, value_cursor)[0]
        break
    return uncompressed_size, compressed_size, local_header_offset


def load_remote_zip_entries(url: str, archive_size: int) -> List[ZipEntry]:
    """Read ZIP64 end records and return all central-directory entries."""

    tail_size = min(65_536, archive_size)
    tail_start = archive_size - tail_size
    tail = read_range(url, tail_start, archive_size - 1)
    eocd_position = tail.rfind(ZIP_EOCD_SIGNATURE)
    locator_position = tail.rfind(ZIP64_LOCATOR_SIGNATURE)
    if eocd_position < 0 or locator_position < 0:
        raise RuntimeError("ZIP64 end records were not found")
    _, zip64_eocd_offset, _ = struct.unpack_from("<IQI", tail, locator_position + 4)
    zip64_header = read_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if zip64_header[:4] != ZIP64_EOCD_SIGNATURE:
        raise RuntimeError("Invalid ZIP64 EOCD signature")
    fields = struct.unpack_from("<QHHIIQQQQ", zip64_header, 4)
    entries_total = fields[6]
    central_size = fields[7]
    central_offset = fields[8]
    central = read_range(url, central_offset, central_offset + central_size - 1)

    entries: List[ZipEntry] = []
    cursor = 0
    while cursor < len(central):
        if central[cursor : cursor + 4] != CENTRAL_SIGNATURE:
            raise RuntimeError(f"Invalid central-directory signature at byte {cursor}")
        values = struct.unpack_from("<6H3I5H2I", central, cursor + 4)
        (
            _made_by,
            _needed,
            _flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            _disk,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = values
        name_start = cursor + 46
        name = central[name_start : name_start + name_length].decode("utf-8")
        extra_start = name_start + name_length
        extra = central[extra_start : extra_start + extra_length]
        uncompressed_size, compressed_size, local_header_offset = _zip64_values(
            extra, uncompressed_size, compressed_size, local_header_offset
        )
        entries.append(
            ZipEntry(
                name=name,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
                crc32=crc32,
                method=method,
            )
        )
        cursor += 46 + name_length + extra_length + comment_length

    if len(entries) != entries_total:
        raise RuntimeError(f"Expected {entries_total} ZIP entries, parsed {len(entries)}")
    return entries


def extract_remote_entry(url: str, entry: ZipEntry, output_path: Path, retries: int = 5) -> None:
    """Range-download, decompress, and validate one ZIP entry."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")
    header = read_range(url, entry.local_header_offset, entry.local_header_offset + 29)
    if header[:4] != LOCAL_SIGNATURE:
        raise RuntimeError(f"Invalid local ZIP header for {entry.name}")
    local_values = struct.unpack_from("<5H3I2H", header, 4)
    name_length = local_values[-2]
    extra_length = local_values[-1]
    data_start = entry.local_header_offset + 30 + name_length + extra_length
    data_end = data_start + entry.compressed_size - 1

    for attempt in range(1, retries + 1):
        crc32 = 0
        written = 0
        try:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if entry.method == 8 else None
            if entry.method not in (0, 8):
                raise RuntimeError(f"Unsupported ZIP method {entry.method} for {entry.name}")
            with _request_range(url, data_start, data_end, timeout_s=180) as response:
                with temporary_path.open("wb") as output_file:
                    while True:
                        compressed_chunk = response.read(1024 * 1024)
                        if not compressed_chunk:
                            break
                        chunk = (
                            decompressor.decompress(compressed_chunk)
                            if decompressor is not None
                            else compressed_chunk
                        )
                        if chunk:
                            output_file.write(chunk)
                            written += len(chunk)
                            crc32 = binascii.crc32(chunk, crc32)
                    if decompressor is not None:
                        final_chunk = decompressor.flush()
                        if final_chunk:
                            output_file.write(final_chunk)
                            written += len(final_chunk)
                            crc32 = binascii.crc32(final_chunk, crc32)
                    output_file.flush()
                    os.fsync(output_file.fileno())
            crc32 &= 0xFFFFFFFF
            if written != entry.uncompressed_size:
                raise RuntimeError(
                    f"Size mismatch for {entry.name}: {written} != {entry.uncompressed_size}"
                )
            if crc32 != entry.crc32:
                raise RuntimeError(f"CRC mismatch for {entry.name}: {crc32:08x} != {entry.crc32:08x}")
            os.replace(temporary_path, output_path)
            return
        except (OSError, RuntimeError, urllib.error.URLError, zlib.error) as error:
            temporary_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            delay = min(2**attempt, 30)
            LOGGER.warning("Extraction failed for %s (%s); retrying in %ss", entry.name, error, delay)
            time.sleep(delay)


def open_index(path: Path) -> sqlite3.Connection:
    """Open and initialize the resumable scan index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS archive_entry (
            name TEXT PRIMARY KEY,
            compressed_size INTEGER NOT NULL,
            uncompressed_size INTEGER NOT NULL,
            local_header_offset INTEGER NOT NULL,
            crc32 INTEGER NOT NULL,
            method INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            scenario_count INTEGER,
            scanned_at TEXT
        );
        CREATE TABLE IF NOT EXISTS scenario_candidate (
            db_name TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            token TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            map_name TEXT NOT NULL,
            scenario_type TEXT NOT NULL,
            PRIMARY KEY (db_name, row_index),
            UNIQUE (db_name, token)
        );
        CREATE INDEX IF NOT EXISTS scenario_candidate_type_idx
            ON scenario_candidate (scenario_type, db_name, row_index);
        CREATE INDEX IF NOT EXISTS scenario_candidate_token_idx
            ON scenario_candidate (token);
        """
    )
    return connection


def register_entries(connection: sqlite3.Connection, entries: Sequence[ZipEntry]) -> None:
    """Persist central-directory metadata without resetting completed scan state."""

    connection.executemany(
        """
        INSERT INTO archive_entry (
            name, compressed_size, uncompressed_size, local_header_offset, crc32, method
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            compressed_size=excluded.compressed_size,
            uncompressed_size=excluded.uncompressed_size,
            local_header_offset=excluded.local_header_offset,
            crc32=excluded.crc32,
            method=excluded.method
        """,
        [
            (
                entry.name,
                entry.compressed_size,
                entry.uncompressed_size,
                entry.local_header_offset,
                entry.crc32,
                entry.method,
            )
            for entry in entries
            if entry.name.endswith(".db")
        ],
    )
    connection.commit()


def load_filter_config(path: Path) -> Dict[str, object]:
    """Load one Hydra scenario-filter YAML as a plain mapping."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping in {path}")
    return config


def scan_one_entry(
    url: str,
    entry: ZipEntry,
    temporary_root: Path,
    scenario_types: Sequence[str],
) -> Tuple[str, List[ScenarioRow]]:
    """Extract and query one DB, always removing its temporary copy."""

    thread_name = threading.current_thread().name.replace("/", "_")
    thread_root = temporary_root / thread_name
    thread_root.mkdir(parents=True, exist_ok=True)
    db_path = thread_root / Path(entry.name).name
    try:
        extract_remote_entry(url, entry, db_path)
        rows: List[ScenarioRow] = []
        for row_index, row in enumerate(
            get_scenarios_from_db(
                str(db_path),
                filter_tokens=None,
                filter_types=list(scenario_types),
                filter_map_names=None,
                include_invalid_mission_goals=False,
                include_cameras=False,
            )
        ):
            scenario_type = row["scenario_type"]
            if scenario_type is None:
                continue
            rows.append(
                ScenarioRow(
                    row_index=row_index,
                    token=row["token"].hex(),
                    timestamp=int(row["timestamp"]),
                    map_name=str(row["map_name"]),
                    scenario_type=str(scenario_type),
                )
            )
        return entry.name, rows
    finally:
        db_path.unlink(missing_ok=True)


def run_scan(args: argparse.Namespace, connection: sqlite3.Connection, entries: Sequence[ZipEntry]) -> None:
    """Run the resumable parallel scan phase."""

    random_config = load_filter_config(args.random_config)
    hard_config = load_filter_config(args.hard_config)
    scenario_types = sorted(
        set(random_config["scenario_types"]) | set(hard_config["scenario_types"])  # type: ignore[arg-type]
    )
    completed = {
        row[0]
        for row in connection.execute("SELECT name FROM archive_entry WHERE status = 'scanned'")
    }
    pending = sorted(
        (entry for entry in entries if entry.name.endswith(".db") and entry.name not in completed),
        key=lambda entry: entry.name,
    )
    if args.max_db is not None:
        pending = pending[: args.max_db]
    total_entries = connection.execute("SELECT COUNT(*) FROM archive_entry").fetchone()[0]
    LOGGER.info(
        "Scan status: %d/%d complete, scheduling %d DBs with %d workers",
        len(completed),
        total_entries,
        len(pending),
        args.workers,
    )
    args.temporary_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    completed_this_run = 0
    compressed_this_run = 0
    entry_by_name = {entry.name: entry for entry in pending}
    compressed_total = sum(entry.compressed_size for entry in pending)
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="test14-scan") as executor:
        pending_iterator = iter(pending)
        futures: Dict[Future[Tuple[str, List[ScenarioRow]]], str] = {}

        def submit_next() -> bool:
            """Submit one DB while keeping the in-flight queue bounded."""

            try:
                entry = next(pending_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                scan_one_entry,
                args.url,
                entry,
                args.temporary_root,
                scenario_types,
            )
            futures[future] = entry.name
            return True

        for _ in range(args.workers):
            submit_next()

        while futures:
            completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                name = futures.pop(future)
                try:
                    db_name, rows = future.result()
                except Exception:
                    for unfinished in futures:
                        unfinished.cancel()
                    LOGGER.exception("DB scan failed: %s", name)
                    raise
                with connection:
                    connection.execute("DELETE FROM scenario_candidate WHERE db_name = ?", (db_name,))
                    connection.executemany(
                        """
                        INSERT INTO scenario_candidate (
                            db_name, row_index, token, timestamp, map_name, scenario_type
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                db_name,
                                row.row_index,
                                row.token,
                                row.timestamp,
                                row.map_name,
                                row.scenario_type,
                            )
                            for row in rows
                        ],
                    )
                    connection.execute(
                        """
                        UPDATE archive_entry
                        SET status = 'scanned', scenario_count = ?, scanned_at = datetime('now')
                        WHERE name = ?
                        """,
                        (len(rows), db_name),
                    )
                completed_this_run += 1
                compressed_this_run += entry_by_name[db_name].compressed_size
                elapsed = max(time.monotonic() - started, 1e-6)
                speed_mib = compressed_this_run / elapsed / 2**20
                remaining_bytes = max(compressed_total - compressed_this_run, 0)
                eta_hours = remaining_bytes / max(compressed_this_run / elapsed, 1e-6) / 3600
                overall = len(completed) + completed_this_run
                LOGGER.info(
                    "Scanned %d/%d: %s (%d candidate rows, %.2f MiB/s, ETA %.2fh)",
                    overall,
                    total_entries,
                    Path(db_name).name,
                    len(rows),
                    speed_mib,
                    eta_hours,
                )
                submit_next()


def _select_random_rows(rows_by_type: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    """Reproduce ``filter_num_scenarios_per_type`` then timestamp filtering."""

    selected: List[Dict[str, object]] = []
    for scenario_type, rows in rows_by_type.items():
        step = max(len(rows) // RANDOM_TARGET_PER_TYPE, 1)
        equisampled = rows[::step][:RANDOM_TARGET_PER_TYPE]
        equisampled.sort(key=lambda row: int(row["timestamp"]))
        min_next_timestamp: Optional[int] = None
        for row in equisampled:
            timestamp = int(row["timestamp"])
            if min_next_timestamp is None or timestamp >= min_next_timestamp:
                selected.append(row)
                min_next_timestamp = timestamp + TIMESTAMP_THRESHOLD_US
    return selected


def run_select(args: argparse.Namespace, connection: sqlite3.Connection) -> Dict[str, object]:
    """Select full Test14 scenarios and persist the required DB manifest."""

    total, scanned = connection.execute(
        "SELECT COUNT(*), SUM(CASE WHEN status = 'scanned' THEN 1 ELSE 0 END) FROM archive_entry"
    ).fetchone()
    if scanned != total:
        raise RuntimeError(f"Selection requires a complete scan: {scanned}/{total} DBs scanned")

    random_config = load_filter_config(args.random_config)
    hard_config = load_filter_config(args.hard_config)
    scenario_types = list(random_config["scenario_types"])  # type: ignore[arg-type]
    rows_by_type: Dict[str, List[Dict[str, object]]] = {name: [] for name in scenario_types}
    query = """
        SELECT db_name, row_index, token, timestamp, map_name, scenario_type
        FROM scenario_candidate
        ORDER BY db_name ASC, row_index ASC
    """
    all_rows: List[Dict[str, object]] = []
    for db_name, row_index, token, timestamp, map_name, scenario_type in connection.execute(query):
        row = {
            "db_name": db_name,
            "row_index": row_index,
            "token": token,
            "timestamp": timestamp,
            "map_name": map_name,
            "scenario_type": scenario_type,
        }
        all_rows.append(row)
        if scenario_type in rows_by_type:
            rows_by_type[scenario_type].append(row)

    random_rows = _select_random_rows(rows_by_type)
    hard_tokens = {str(token) for token in hard_config["scenario_tokens"]}  # type: ignore[arg-type]
    hard_rows = [row for row in all_rows if str(row["token"]) in hard_tokens]
    found_hard_tokens = {str(row["token"]) for row in hard_rows}
    missing_hard_tokens = sorted(hard_tokens - found_hard_tokens)
    required_dbs = sorted(
        {str(row["db_name"]) for row in random_rows} | {str(row["db_name"]) for row in hard_rows}
    )
    placeholders = ",".join("?" for _ in required_dbs)
    size_row = connection.execute(
        f"""
        SELECT SUM(compressed_size), SUM(uncompressed_size)
        FROM archive_entry WHERE name IN ({placeholders})
        """,
        required_dbs,
    ).fetchone()
    manifest: Dict[str, object] = {
        "source_url": args.url,
        "archive_size": args.archive_size,
        "scan": {"db_count": total, "candidate_count": len(all_rows)},
        "test14_hard": {
            "configured_token_count": len(hard_tokens),
            "found_token_count": len(found_hard_tokens),
            "missing_tokens": missing_hard_tokens,
            "scenarios": hard_rows,
        },
        "test14_random": {
            "scenario_count": len(random_rows),
            "scenarios": random_rows,
            "counts_by_type": {
                scenario_type: sum(
                    1 for row in random_rows if row["scenario_type"] == scenario_type
                )
                for scenario_type in scenario_types
            },
        },
        "required_db_count": len(required_dbs),
        "required_dbs": required_dbs,
        "required_compressed_bytes": int(size_row[0] or 0),
        "required_uncompressed_bytes": int(size_row[1] or 0),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    generated_random_config = dict(random_config)
    generated_random_config["scenario_tokens"] = [row["token"] for row in random_rows]
    generated_random_config["num_scenarios_per_type"] = None
    generated_random_config["timestamp_threshold_s"] = None
    generated_random_config["shuffle"] = False
    args.generated_random_config.parent.mkdir(parents=True, exist_ok=True)
    with args.generated_random_config.open("w", encoding="utf-8") as file:
        file.write("# Generated from the complete official test split; do not hand-edit.\n")
        yaml.safe_dump(generated_random_config, file, sort_keys=False, allow_unicode=True)
    LOGGER.info(
        "Selection result: hard=%d/%d, random=%d, required DBs=%d, extracted size=%.2f GiB",
        len(found_hard_tokens),
        len(hard_tokens),
        len(random_rows),
        len(required_dbs),
        int(size_row[1] or 0) / 2**30,
    )
    if missing_hard_tokens:
        raise RuntimeError(f"Missing {len(missing_hard_tokens)} test14-hard tokens")
    if len(random_rows) != args.expected_random_count:
        raise RuntimeError(
            f"Expected {args.expected_random_count} test14-random scenarios, selected {len(random_rows)}"
        )
    return manifest


def _quick_check_db(path: Path) -> None:
    """Run a lightweight SQLite integrity check on one permanent DB."""

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        result = connection.execute("PRAGMA quick_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"SQLite quick_check failed for {path}: {result}")


def run_extract(args: argparse.Namespace, connection: sqlite3.Connection) -> None:
    """Permanently extract only DBs referenced by the selection manifest."""

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Selection manifest does not exist: {args.manifest}")
    with args.manifest.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    required_names = list(manifest["required_dbs"])
    rows = connection.execute(
        """
        SELECT name, compressed_size, uncompressed_size, local_header_offset, crc32, method
        FROM archive_entry
        """
    ).fetchall()
    entries = {
        row[0]: ZipEntry(
            name=row[0],
            compressed_size=row[1],
            uncompressed_size=row[2],
            local_header_offset=row[3],
            crc32=row[4],
            method=row[5],
        )
        for row in rows
    }
    args.output_db_root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(required_names, start=1):
        entry = entries[name]
        output_path = args.output_db_root / Path(name).name
        if output_path.is_file() and output_path.stat().st_size == entry.uncompressed_size:
            LOGGER.info("Keeping existing DB %d/%d: %s", index, len(required_names), output_path.name)
            continue
        LOGGER.info("Extracting required DB %d/%d: %s", index, len(required_names), output_path.name)
        extract_remote_entry(args.url, entry, output_path)
        _quick_check_db(output_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    script_root = Path(__file__).resolve().parents[1]
    default_work_root = script_root / "tmp" / "test14_full_remote_subset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("scan", "select", "extract", "all"), default="all")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive-size", type=int, default=DEFAULT_ARCHIVE_SIZE)
    parser.add_argument("--work-root", type=Path, default=default_work_root)
    parser.add_argument(
        "--random-config",
        type=Path,
        default=script_root / "hdp_nuplan" / "config" / "scenario_filter" / "test14-random.yaml",
    )
    parser.add_argument(
        "--hard-config",
        type=Path,
        default=script_root / "hdp_nuplan" / "config" / "scenario_filter" / "test14-hard.yaml",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-db", type=int, default=None, help="Smoke-test limit for this invocation")
    parser.add_argument("--expected-random-count", type=int, default=261)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """Run the requested preparation phase."""

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.work_root = args.work_root.resolve()
    args.index = args.work_root / "test14_remote_scan.sqlite"
    args.manifest = args.work_root / "test14_selection_manifest.json"
    args.generated_random_config = (
        args.work_root / "config" / "scenario_filter" / "test14-random-full.yaml"
    )
    args.temporary_root = args.work_root / "scan_tmp"
    args.output_db_root = args.work_root / "data" / "cache" / "test14"

    entries = load_remote_zip_entries(args.url, args.archive_size)
    db_entries = [entry for entry in entries if entry.name.endswith(".db")]
    LOGGER.info(
        "Official archive: %d DBs, %.2f GiB compressed, %.2f GiB extracted",
        len(db_entries),
        sum(entry.compressed_size for entry in db_entries) / 2**30,
        sum(entry.uncompressed_size for entry in db_entries) / 2**30,
    )
    connection = open_index(args.index)
    try:
        register_entries(connection, db_entries)
        if args.phase in ("scan", "all"):
            run_scan(args, connection, db_entries)
        if args.phase in ("select", "all"):
            run_select(args, connection)
        if args.phase in ("extract", "all"):
            run_extract(args, connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
