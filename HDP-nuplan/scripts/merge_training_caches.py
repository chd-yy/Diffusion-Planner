#!/usr/bin/env python3
"""合并多个 NuPlan 训练 NPZ cache，去重并排除 validation 泄漏。"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


METADATA_KEYS = ("token", "log_name", "map_name", "scenario_type")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--source 必须使用 LABEL=DIR 格式: {value}")
    label, raw_path = value.split("=", 1)
    if not label:
        raise ValueError("source label 不能为空")
    path = Path(raw_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return label, path


def read_metadata(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(METADATA_KEYS) - set(data.files))
        if missing:
            raise ValueError(f"{path} 缺少正式合并所需元数据: {missing}")
        result = {}
        for key in METADATA_KEYS:
            if data[key].shape != ():
                raise ValueError(f"{path}:{key} 必须是标量")
            result[key] = str(data[key].item())
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="按优先级重复传入 LABEL=CACHE_DIR；先出现的重复场景优先保留。",
    )
    parser.add_argument("--validation-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    validation_cache = args.validation_cache.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    if not validation_cache.is_dir():
        raise FileNotFoundError(validation_cache)

    validation_names = set()
    validation_tokens = set()
    validation_logs = set()
    for path in sorted(validation_cache.glob("*.npz")):
        metadata = read_metadata(path)
        validation_names.add(path.name)
        validation_tokens.add(metadata["token"])
        validation_logs.add(metadata["log_name"])

    selected_by_name = {}
    selected_by_token = {}
    source_stats = {}
    duplicate_records = []
    excluded_records = []

    for label, source_dir in sources:
        stats = Counter()
        # source 既可以是单个 cache，也可以是包含多个 shard_*/cache 的根目录。
        # 递归扫描使滚动预处理每完成一个 shard 后，无需先复制到平铺目录即可增量合并。
        files = sorted(source_dir.rglob("*.npz"))
        stats["input"] = len(files)
        for path in files:
            metadata = read_metadata(path)
            token = metadata["token"]
            log_name = metadata["log_name"]

            existing = selected_by_name.get(path.name)
            if existing is not None:
                if existing["metadata"]["token"] != token:
                    raise ValueError(f"同名文件 token 冲突: {path.name}")
                if sha256(existing["path"]) != sha256(path):
                    raise ValueError(f"同名文件内容冲突: {path.name}")
                stats["duplicate_filename"] += 1
                duplicate_records.append(
                    {
                        "filename": path.name,
                        "kept_source": existing["source"],
                        "skipped_source": label,
                    }
                )
                continue

            existing_token = selected_by_token.get(token)
            if existing_token is not None:
                stats["duplicate_token"] += 1
                duplicate_records.append(
                    {
                        "token": token,
                        "kept_filename": existing_token["path"].name,
                        "skipped_filename": path.name,
                        "kept_source": existing_token["source"],
                        "skipped_source": label,
                    }
                )
                continue

            leakage_reasons = []
            if path.name in validation_names:
                leakage_reasons.append("filename")
            if token in validation_tokens:
                leakage_reasons.append("token")
            if log_name in validation_logs:
                leakage_reasons.append("log_name")
            if leakage_reasons:
                stats["excluded_validation"] += 1
                excluded_records.append(
                    {
                        "filename": path.name,
                        "source": label,
                        "reasons": leakage_reasons,
                    }
                )
                continue

            record = {
                "path": path,
                "source": label,
                "metadata": metadata,
            }
            selected_by_name[path.name] = record
            selected_by_token[token] = record
            stats["selected"] += 1

        source_stats[label] = dict(stats)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        cache_dir = staging_dir / "cache"
        cache_dir.mkdir()
        manifest = sorted(selected_by_name)
        log_counts = Counter()
        map_counts = Counter()
        scenario_type_counts = Counter()
        source_counts = Counter()

        for filename in manifest:
            record = selected_by_name[filename]
            os.link(record["path"], cache_dir / filename)
            metadata = record["metadata"]
            log_counts[metadata["log_name"]] += 1
            map_counts[metadata["map_name"]] += 1
            scenario_type_counts[metadata["scenario_type"]] += 1
            source_counts[record["source"]] += 1

        manifest_path = staging_dir / "diffusion_planner_training.json"
        sampling_report_path = staging_dir / "sampling_report.json"
        merge_report_path = staging_dir / "merge_report.json"
        write_json(manifest_path, manifest)
        write_json(
            sampling_report_path,
            {
                "status": "complete",
                "strategy": "deduplicated_training_cache_union",
                "requested_scenarios": len(manifest),
                "selected_scenarios": len(manifest),
                "log_names": sorted(log_counts),
                "selected_per_log": dict(sorted(log_counts.items())),
                "selected_per_map": dict(sorted(map_counts.items())),
                "selected_per_scenario_type": dict(
                    sorted(scenario_type_counts.items())
                ),
            },
        )
        write_json(
            merge_report_path,
            {
                "status": "passed",
                "link_mode": "hardlink",
                "sources": [
                    {"label": label, "cache_dir": str(path)}
                    for label, path in sources
                ],
                "source_stats": source_stats,
                "source_contribution": dict(sorted(source_counts.items())),
                "validation_cache": str(validation_cache),
                "validation_filename_overlap": 0,
                "validation_token_overlap": 0,
                "validation_log_overlap": 0,
                "selected_count": len(manifest),
                "unique_filename_count": len(selected_by_name),
                "unique_token_count": len(selected_by_token),
                "log_count": len(log_counts),
                "map_counts": dict(sorted(map_counts.items())),
                "duplicate_count": len(duplicate_records),
                "duplicates": duplicate_records,
                "excluded_validation_count": len(excluded_records),
                "excluded_validation": excluded_records,
            },
        )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"Merged {len(manifest)} unique training NPZ files into {output_dir}")
    print(f"Manifest: {output_dir / 'diffusion_planner_training.json'}")


if __name__ == "__main__":
    main()
