"""可恢复 NuPlan 预处理任务使用的通用文件工具。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256，避免把大 NPZ 一次性读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """先写同目录临时文件，再原子替换目标 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def unique_log_names(log_names: Iterable[str]) -> List[str]:
    """按输入顺序去重，并拒绝空日志名或非字符串。"""
    unique: List[str] = []
    seen = set()
    for log_name in log_names:
        if not isinstance(log_name, str) or not log_name.strip():
            raise ValueError(f"invalid NuPlan log name: {log_name!r}")
        if log_name not in seen:
            unique.append(log_name)
            seen.add(log_name)
    return unique


def build_checksum_payload(
    manifest_path: Path,
    processing_report_path: Path,
    sampling_report_path: Optional[Path],
    cache_dir: Path,
    filenames: Iterable[str],
    include_npz_files: bool,
) -> Dict[str, Any]:
    """生成分片元数据校验和；按需附加每个 NPZ 的 SHA256。"""
    manifest_path = Path(manifest_path)
    processing_report_path = Path(processing_report_path)
    payload: Dict[str, Any] = {
        "algorithm": "sha256",
        "manifest": {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "processing_report": {
            "path": processing_report_path.name,
            "sha256": sha256_file(processing_report_path),
        },
        "npz_checksums_included": include_npz_files,
    }
    if sampling_report_path is not None and Path(sampling_report_path).is_file():
        sampling_report_path = Path(sampling_report_path)
        payload["sampling_report"] = {
            "path": sampling_report_path.name,
            "sha256": sha256_file(sampling_report_path),
        }
    if include_npz_files:
        cache_dir = Path(cache_dir)
        payload["npz_files"] = {
            filename: sha256_file(cache_dir / filename)
            for filename in sorted(filenames)
        }
    return payload
