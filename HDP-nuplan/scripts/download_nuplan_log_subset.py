#!/usr/bin/env python3
"""按日志清单从 NuPlan 官方公开 ZIP 中下载所需 DB，不落盘完整城市归档。"""

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hdp_nuplan.data_process.run_utils import atomic_write_json, unique_log_names  # noqa: E402


OFFICIAL_BASE_URL = (
    "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/"
    "public/nuplan-v1.1"
)
OFFICIAL_TRAIN_ARCHIVES = {
    "boston": f"{OFFICIAL_BASE_URL}/nuplan-v1.1_train_boston.zip",
    "pittsburgh": f"{OFFICIAL_BASE_URL}/nuplan-v1.1_train_pittsburgh.zip",
    "singapore": f"{OFFICIAL_BASE_URL}/nuplan-v1.1_train_singapore.zip",
    **{
        f"vegas_{index}": f"{OFFICIAL_BASE_URL}/nuplan-v1.1_train_vegas_{index}.zip"
        for index in range(1, 7)
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_log_name(value: str) -> str:
    """把 manifest 中可能带路径或 .db 后缀的值统一成 NuPlan log name。"""
    # NuPlan log 本身含多个点，例如 2021.05.12.19.36.12_veh-35_...。
    # Path.stem 会把不带 .db 的合法日志名末段误当成扩展名，因此这里只显式移除 .db。
    filename = Path(str(value)).name
    return filename[:-3] if filename.endswith(".db") else filename


def _validate_sqlite(path: Path, quick_check: bool = False) -> None:
    """检查文件头和 SQLite 可读性；ZIP 解压读到 EOF 时还会自动验证 CRC。"""
    with path.open("rb") as file:
        if file.read(16) != b"SQLite format 3\x00":
            raise ValueError(f"not a SQLite database: {path}")
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        if quick_check:
            result = connection.execute("PRAGMA quick_check(1)").fetchone()
            if result != ("ok",):
                raise ValueError(f"SQLite quick_check failed for {path}: {result}")


def _default_archive_opener(
    location: str,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 60.0,
):
    # 中文注释：正式全量流程默认读取官方 HTTP ZIP；本地已经下载完整归档时，
    # 直接使用 ZipFile 读取 member，保留相同的 CRC、SQLite 和审计报告检查。
    if location.startswith("file://"):
        local_path = Path(unquote(urlparse(location).path)).expanduser()
    elif "://" not in location:
        local_path = Path(location).expanduser()
    else:
        local_path = None
    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(f"local ZIP archive does not exist: {local_path}")
        return zipfile.ZipFile(local_path)

    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError(
            "remotezip is required; install HDP-nuplan/requirements_data_download.txt"
        ) from error
    # requests 的 timeout=(连接超时, 单次 socket 读取超时)。没有显式读取超时的话，
    # 某个 ZIP member 的 HTTP Range 连接停滞后可能永久占住整个 shard 下载任务。
    return RemoteZip(
        location,
        timeout=(connect_timeout_seconds, read_timeout_seconds),
    )


def _close_archive(archive) -> None:
    """关闭当前远程 ZIP；重试时不让 close 异常掩盖原始网络错误。"""
    try:
        archive.close()
    except Exception:
        pass


def _pid_is_alive(pid: int) -> bool:
    """保守判断 PID 是否仍存在；无权限探测时按存活处理，避免误删活动文件。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_member_temporaries(destination: Path) -> int:
    """删除该 DB 由已退出进程遗留的原子写临时文件，保留活动进程文件。"""
    if not destination.parent.is_dir():
        return 0
    prefix = f".{destination.name}."
    removed = 0
    for temporary in destination.parent.glob(f"{prefix}*.tmp"):
        # 当前两种临时文件名分别为：
        # .<log>.db.<pid>.tmp 与 .<log>.db.<pid>.deflate.tmp。
        remainder = temporary.name[len(prefix):]
        pid_text = remainder.split(".", 1)[0]
        if not pid_text.isdigit() or _pid_is_alive(int(pid_text)):
            continue
        try:
            temporary.unlink()
            removed += 1
        except FileNotFoundError:
            # 另一个恢复进程已完成清理时保持幂等。
            pass
    return removed


def build_archive_index(
    archives: Mapping[str, str],
    archive_opener: Callable[[str], object] = _default_archive_opener,
) -> Dict[str, dict]:
    """只读取 ZIP 中央目录，建立唯一的 log -> archive/member 映射。"""
    index: Dict[str, dict] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    for archive_name, archive_url in archives.items():
        print(f"Indexing {archive_name}: {archive_url}", flush=True)
        with archive_opener(archive_url) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.endswith(".db"):
                    continue
                log_name = Path(info.filename).stem
                if log_name in index:
                    duplicates[log_name].extend(
                        [index[log_name]["archive"], archive_name]
                    )
                    continue
                index[log_name] = {
                    "archive": archive_name,
                    "member": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc": int(info.CRC),
                }
    if duplicates:
        sample = dict(list(sorted(duplicates.items()))[:10])
        raise ValueError(f"duplicate log DBs across train archives: {sample}")
    return dict(sorted(index.items()))


def load_or_build_archive_index(
    index_path: Path,
    archives: Mapping[str, str],
    archive_opener: Callable[[str], object] = _default_archive_opener,
) -> Dict[str, dict]:
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("archives") != dict(archives):
            raise ValueError(
                f"archive definitions changed; remove or replace stale index: {index_path}"
            )
        return payload["logs"]

    logs = build_archive_index(archives, archive_opener)
    atomic_write_json(
        index_path,
        {
            "format_version": 1,
            "archives": dict(archives),
            "log_count": len(logs),
            "logs": logs,
        },
    )
    return logs


def _copy_member_atomic(archive, member: str, destination: Path) -> None:
    """将一个 ZIP member 写到同目录临时文件，成功后原子替换目标 DB。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with archive.open(member) as source, temporary.open("wb") as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _zip_member_data_offset(archive, info: zipfile.ZipInfo) -> int:
    """读取本地文件头，精确计算 ZIP member 压缩数据的起始偏移。"""
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is unsupported: {info.filename}")
    archive.fp.seek(info.header_offset)
    header = archive.fp.read(zipfile.sizeFileHeader)
    if len(header) != zipfile.sizeFileHeader:
        raise ValueError(f"truncated local ZIP header: {info.filename}")
    fields = struct.unpack(zipfile.structFileHeader, header)
    if fields[zipfile._FH_SIGNATURE] != zipfile.stringFileHeader:
        raise ValueError(f"invalid local ZIP header signature: {info.filename}")
    filename_size = int(fields[zipfile._FH_FILENAME_LENGTH])
    extra_size = int(fields[zipfile._FH_EXTRA_FIELD_LENGTH])
    return int(info.header_offset + zipfile.sizeFileHeader + filename_size + extra_size)


def _inflate_compressed_member(
    compressed_path: Path,
    destination_temporary: Path,
    info: zipfile.ZipInfo,
) -> None:
    """流式解压 curl 下载的单个 member，并验证大小、CRC 和 DEFLATE 结束标志。"""
    if info.compress_type == zipfile.ZIP_STORED:
        decompressor = None
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"unsupported ZIP compression type {info.compress_type}: {info.filename}"
        )

    crc = 0
    output_size = 0
    with compressed_path.open("rb") as source, destination_temporary.open("wb") as target:
        for compressed_chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            chunk = (
                compressed_chunk
                if decompressor is None
                else decompressor.decompress(compressed_chunk)
            )
            if chunk:
                target.write(chunk)
                crc = zlib.crc32(chunk, crc)
                output_size += len(chunk)
        if decompressor is not None:
            tail = decompressor.flush()
            if tail:
                target.write(tail)
                crc = zlib.crc32(tail, crc)
                output_size += len(tail)
            if not decompressor.eof:
                raise ValueError(f"incomplete DEFLATE stream: {info.filename}")
        target.flush()
        os.fsync(target.fileno())

    if output_size != int(info.file_size):
        raise ValueError(
            f"uncompressed size mismatch for {info.filename}: "
            f"{output_size} != {info.file_size}"
        )
    if crc & 0xFFFFFFFF != int(info.CRC):
        raise ValueError(
            f"CRC mismatch for {info.filename}: "
            f"{crc & 0xFFFFFFFF} != {info.CRC}"
        )


def _copy_member_curl_atomic(
    archive_url: str,
    archive,
    member: str,
    destination: Path,
    connect_timeout_seconds: float,
    low_speed_limit_bps: int,
    low_speed_time_seconds: int,
    command_runner: Callable[..., object] = subprocess.run,
) -> None:
    """用 curl 下载精确压缩区间；低速连接失败后由外层 member 重试逻辑重开。"""
    if low_speed_limit_bps <= 0 or low_speed_time_seconds <= 0:
        raise ValueError("curl low-speed thresholds must be positive")
    info = archive.getinfo(member)
    data_start = _zip_member_data_offset(archive, info)
    data_end = data_start + int(info.compress_size) - 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed_temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.deflate.tmp"
    )
    destination_temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        command = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout", str(connect_timeout_seconds),
            "--speed-limit", str(low_speed_limit_bps),
            "--speed-time", str(low_speed_time_seconds),
            "--header", "Accept-Encoding: identity",
            "--range", f"{data_start}-{data_end}",
            "--output", str(compressed_temporary),
            archive_url,
        ]
        command_runner(command, check=True)
        compressed_size = compressed_temporary.stat().st_size
        if compressed_size != int(info.compress_size):
            raise ValueError(
                f"compressed size mismatch for {member}: "
                f"{compressed_size} != {info.compress_size}"
            )
        _inflate_compressed_member(compressed_temporary, destination_temporary, info)
        os.replace(destination_temporary, destination)
    finally:
        if compressed_temporary.exists():
            compressed_temporary.unlink()
        if destination_temporary.exists():
            destination_temporary.unlink()


def download_log_subset(
    log_names: Sequence[str],
    output_dir: Path,
    archive_index: Mapping[str, dict],
    archives: Mapping[str, str],
    archive_opener: Callable[[str], object] = _default_archive_opener,
    skip_existing: bool = True,
    sqlite_quick_check: bool = False,
    max_member_retries: int = 3,
    retry_delay_seconds: float = 5.0,
    download_backend: str = "remotezip",
    connect_timeout_seconds: float = 10.0,
    curl_low_speed_limit_bps: int = 512 * 1024,
    curl_low_speed_time_seconds: int = 5,
) -> dict:
    """按归档分组下载 DB，并返回可持久化的审计报告。"""
    if max_member_retries < 0:
        raise ValueError("max_member_retries must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    if download_backend not in {"remotezip", "curl", "zip"}:
        raise ValueError(f"unsupported download_backend: {download_backend}")
    if connect_timeout_seconds <= 0:
        raise ValueError("connect_timeout_seconds must be positive")
    if curl_low_speed_limit_bps <= 0 or curl_low_speed_time_seconds <= 0:
        raise ValueError("curl low-speed thresholds must be positive")
    requested = unique_log_names([_normalize_log_name(item) for item in log_names])
    missing = sorted(set(requested) - set(archive_index))
    if missing:
        raise KeyError(f"{len(missing)} requested logs are absent from train archives: {missing[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for log_name in requested:
        grouped[archive_index[log_name]["archive"]].append(log_name)

    started_at = time.time()
    files = []
    processed = 0
    skipped_existing = 0
    reprocessed_invalid = 0
    retry_count = 0
    stale_temporaries_removed = 0
    for archive_name in sorted(grouped):
        if archive_name not in archives:
            raise KeyError(f"archive URL missing for {archive_name}")
        print(
            f"Opening {archive_name} for {len(grouped[archive_name])} requested logs",
            flush=True,
        )
        archive = archive_opener(archives[archive_name])
        try:
            for item_index, log_name in enumerate(grouped[archive_name], start=1):
                metadata = archive_index[log_name]
                destination = output_dir / f"{log_name}.db"
                action = "processed"
                if destination.exists() and skip_existing:
                    try:
                        if destination.stat().st_size != metadata["file_size"]:
                            raise ValueError("existing DB size differs from ZIP metadata")
                        _validate_sqlite(destination, quick_check=sqlite_quick_check)
                        action = "skipped_existing"
                        skipped_existing += 1
                    except Exception:
                        reprocessed_invalid += 1
                if action == "processed":
                    stale_temporaries_removed += cleanup_stale_member_temporaries(
                        destination
                    )
                    print(
                        f"[{item_index}/{len(grouped[archive_name])}] downloading {log_name}.db",
                        flush=True,
                    )
                    attempts = 0
                    while True:
                        attempts += 1
                        try:
                            if download_backend == "curl":
                                _copy_member_curl_atomic(
                                    archives[archive_name],
                                    archive,
                                    metadata["member"],
                                    destination,
                                    connect_timeout_seconds=connect_timeout_seconds,
                                    low_speed_limit_bps=curl_low_speed_limit_bps,
                                    low_speed_time_seconds=curl_low_speed_time_seconds,
                                )
                            else:
                                _copy_member_atomic(
                                    archive, metadata["member"], destination
                                )
                            break
                        except Exception as error:
                            if attempts > max_member_retries:
                                raise
                            retry_count += 1
                            print(
                                f"Retrying {log_name}.db after {type(error).__name__}: "
                                f"attempt {attempts}/{max_member_retries + 1}",
                                flush=True,
                            )
                            # 已发生读取错误的 RemoteZip/HTTP 连接不能继续复用；关闭后
                            # 重新读取中央目录并建立新连接，只重试当前 member。
                            _close_archive(archive)
                            if retry_delay_seconds:
                                time.sleep(retry_delay_seconds)
                            archive = archive_opener(archives[archive_name])
                    if destination.stat().st_size != metadata["file_size"]:
                        raise ValueError(f"downloaded size mismatch: {destination}")
                    _validate_sqlite(destination, quick_check=sqlite_quick_check)
                    processed += 1
                else:
                    attempts = 0

                files.append(
                    {
                        "log_name": log_name,
                        "filename": destination.name,
                        "archive": archive_name,
                        "member": metadata["member"],
                        "size": destination.stat().st_size,
                        "sha256": _sha256(destination),
                        "action": action,
                        "download_attempts": attempts,
                    }
                )
        finally:
            _close_archive(archive)

    return {
        "status": "complete",
        "requested_log_count": len(requested),
        "downloaded": processed,
        "skipped_existing": skipped_existing,
        "reprocessed_invalid": reprocessed_invalid,
        "retry_count": retry_count,
        "stale_temporaries_removed": stale_temporaries_removed,
        "download_backend": download_backend,
        "curl_low_speed_limit_bps": (
            curl_low_speed_limit_bps if download_backend == "curl" else None
        ),
        "curl_low_speed_time_seconds": (
            curl_low_speed_time_seconds if download_backend == "curl" else None
        ),
        "elapsed_seconds": time.time() - started_at,
        "total_bytes": sum(item["size"] for item in files),
        "log_names": requested,
        "archives_used": sorted(grouped),
        "files": sorted(files, key=lambda item: item["log_name"]),
    }


def _parse_archive_overrides(values: Iterable[str]) -> Dict[str, str]:
    archives = dict(OFFICIAL_TRAIN_ARCHIVES)
    for value in values:
        if "=" not in value:
            raise ValueError(f"archive override must be NAME=URL_OR_PATH: {value}")
        name, location = value.split("=", 1)
        if not name or not location:
            raise ValueError(f"archive override must be NAME=URL_OR_PATH: {value}")
        archives[name] = location
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_names_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--archive_index", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="NAME=URL_OR_PATH",
        help="override or add an archive using an HTTP URL or local ZIP path",
    )
    parser.add_argument(
        "--only_archive",
        action="append",
        default=[],
        metavar="NAME",
        help="limit index construction and lookup to the named archive(s)",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sqlite_quick_check",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--connect_timeout_seconds",
        type=float,
        default=10.0,
        help="HTTP connection timeout used by RemoteZip",
    )
    parser.add_argument(
        "--read_timeout_seconds",
        type=float,
        default=60.0,
        help="maximum idle time for one HTTP Range socket read",
    )
    parser.add_argument(
        "--max_member_retries",
        type=int,
        default=3,
        help="number of retries after the first failed ZIP member download",
    )
    parser.add_argument(
        "--retry_delay_seconds",
        type=float,
        default=5.0,
        help="delay between ZIP member download attempts",
    )
    parser.add_argument(
        "--download_backend",
        choices=["remotezip", "curl", "zip"],
        default="remotezip",
        help="ZIP member transfer backend; use zip for a complete local archive",
    )
    parser.add_argument(
        "--curl_low_speed_limit_bps",
        type=int,
        default=512 * 1024,
        help="curl abort threshold in bytes/s when sustained for --curl_low_speed_time_seconds",
    )
    parser.add_argument(
        "--curl_low_speed_time_seconds",
        type=int,
        default=5,
        help="seconds below the curl low-speed threshold before aborting the member attempt",
    )
    args = parser.parse_args()

    if args.connect_timeout_seconds <= 0 or args.read_timeout_seconds <= 0:
        raise ValueError("HTTP timeout values must be positive")

    def archive_opener(url: str):
        return _default_archive_opener(
            url,
            connect_timeout_seconds=args.connect_timeout_seconds,
            read_timeout_seconds=args.read_timeout_seconds,
        )

    archives = _parse_archive_overrides(args.archive)
    if args.only_archive:
        requested_archives = list(dict.fromkeys(args.only_archive))
        unknown = sorted(set(requested_archives) - set(archives))
        if unknown:
            raise KeyError(f"unknown --only_archive values: {unknown}")
        archives = {name: archives[name] for name in requested_archives}
    log_names = json.loads(args.log_names_json.read_text(encoding="utf-8"))
    if not isinstance(log_names, list) or not all(isinstance(item, str) for item in log_names):
        raise ValueError("log_names_json must contain a JSON list of strings")

    archive_index = load_or_build_archive_index(
        args.archive_index,
        archives,
        archive_opener=archive_opener,
    )
    report = download_log_subset(
        log_names,
        args.output_dir,
        archive_index,
        archives,
        archive_opener=archive_opener,
        skip_existing=args.skip_existing,
        sqlite_quick_check=args.sqlite_quick_check,
        max_member_retries=args.max_member_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        download_backend=args.download_backend,
        connect_timeout_seconds=args.connect_timeout_seconds,
        curl_low_speed_limit_bps=args.curl_low_speed_limit_bps,
        curl_low_speed_time_seconds=args.curl_low_speed_time_seconds,
    )
    atomic_write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
