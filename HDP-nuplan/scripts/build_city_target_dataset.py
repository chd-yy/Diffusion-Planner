#!/usr/bin/env python3
"""在一个已验证缓存上按城市补充样本，构建可复现且无验证集泄漏的训练集。"""

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

import numpy as np


METADATA_KEYS = ("map_name", "log_name", "scenario_type", "token")


@dataclass(frozen=True)
class Sample:
    path: Path
    filename: str
    map_name: str
    log_name: str
    scenario_type: str
    token: str
    source: str


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path, source: str) -> Sample:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(METADATA_KEYS) - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing metadata keys: {missing}")
        values = {}
        for key in METADATA_KEYS:
            if data[key].shape != ():
                raise ValueError(f"{path}:{key} must be a scalar")
            values[key] = str(data[key].item())
    return Sample(
        path=path.resolve(),
        filename=path.name,
        map_name=values["map_name"],
        log_name=values["log_name"],
        scenario_type=values["scenario_type"],
        token=values["token"],
        source=source,
    )


def load_manifest_samples(cache_dir: Path, manifest_path: Path, source: str) -> list[Sample]:
    cache_dir = Path(cache_dir)
    entries = _read_json(manifest_path)
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError(f"{manifest_path} must contain a JSON list of filenames")
    if len(entries) != len(set(entries)):
        raise ValueError(f"{manifest_path} contains duplicate entries")
    samples = []
    for entry in entries:
        relative = PurePosixPath(entry)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest entry: {entry!r}")
        path = cache_dir.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append(_metadata(path, source))
    return samples


def scan_source_samples(root: Path, source: str) -> list[Sample]:
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise ValueError(f"no NPZ files found below {root}")
    return [_metadata(path, source) for path in paths]


def _assert_unique(samples: Iterable[Sample], label: str) -> None:
    samples = list(samples)
    filenames = [sample.filename for sample in samples]
    tokens = [sample.token for sample in samples]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"{label} contains duplicate filenames")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{label} contains duplicate tokens")


def select_balanced_new_logs(
    candidates: Iterable[Sample],
    count: int,
    seed: int,
    excluded_filenames: set[str],
    excluded_tokens: set[str],
    excluded_logs: set[str],
) -> tuple[list[Sample], dict]:
    """优先从训练集尚未覆盖的新日志中轮询取样，再用旧日志补足。"""
    if count < 0:
        raise ValueError("count must be non-negative")

    unique_candidates = {}
    rejected = Counter()
    for sample in sorted(candidates, key=lambda item: (item.filename, str(item.path))):
        if sample.filename in excluded_filenames:
            rejected["filename_overlap"] += 1
            continue
        if sample.token in excluded_tokens:
            rejected["token_overlap"] += 1
            continue
        if sample.filename in unique_candidates:
            rejected["duplicate_source_filename"] += 1
            continue
        unique_candidates[sample.filename] = sample

    new_log_groups = defaultdict(list)
    existing_log_groups = defaultdict(list)
    for sample in unique_candidates.values():
        target = existing_log_groups if sample.log_name in excluded_logs else new_log_groups
        target[sample.log_name].append(sample)

    rng = random.Random(seed)
    for groups in (new_log_groups, existing_log_groups):
        for log_name in groups:
            groups[log_name].sort(key=lambda item: item.filename)
            rng.shuffle(groups[log_name])

    selected = []
    selected_names = set()
    selected_tokens = set()

    def take_round_robin(groups) -> None:
        log_names = sorted(groups)
        rng.shuffle(log_names)
        depth = 0
        while len(selected) < count:
            added = False
            for log_name in log_names:
                items = groups[log_name]
                if depth >= len(items):
                    continue
                sample = items[depth]
                if sample.filename in selected_names or sample.token in selected_tokens:
                    continue
                selected.append(sample)
                selected_names.add(sample.filename)
                selected_tokens.add(sample.token)
                added = True
                if len(selected) == count:
                    return
            if not added:
                return
            depth += 1

    take_round_robin(new_log_groups)
    selected_from_new_logs = len(selected)
    take_round_robin(existing_log_groups)
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} eligible unique samples are available; {count} requested"
        )

    selected.sort(key=lambda item: item.filename)
    return selected, {
        "requested": count,
        "selected": len(selected),
        "selected_from_new_logs": selected_from_new_logs,
        "selected_from_existing_logs": len(selected) - selected_from_new_logs,
        "new_candidate_log_count": len(new_log_groups),
        "existing_candidate_log_count": len(existing_log_groups),
        "selected_log_count": len({sample.log_name for sample in selected}),
        "rejected": dict(sorted(rejected.items())),
    }


def build_dataset(
    base_cache: Path,
    base_manifest: Path,
    sources: list[tuple[str, Path, int]],
    output_dir: Path,
    seed: int,
    val_cache: Optional[Path] = None,
    val_manifest: Optional[Path] = None,
    val_report: Optional[Path] = None,
) -> dict:
    base_samples = load_manifest_samples(base_cache, base_manifest, "base")
    _assert_unique(base_samples, "base dataset")

    validation_samples = []
    if val_cache is not None or val_manifest is not None:
        if val_cache is None or val_manifest is None:
            raise ValueError("val_cache and val_manifest must be provided together")
        validation_samples = load_manifest_samples(val_cache, val_manifest, "validation")
        _assert_unique(validation_samples, "validation dataset")

    val_logs_from_report = set()
    if val_report is not None:
        report = _read_json(val_report)
        val_logs_from_report.update(str(item) for item in report.get("log_names", []))
        val_logs_from_report.update(str(item) for item in report.get("selected_per_log", {}))

    base_filenames = {sample.filename for sample in base_samples}
    base_tokens = {sample.token for sample in base_samples}
    base_logs = {sample.log_name for sample in base_samples}
    val_filenames = {sample.filename for sample in validation_samples}
    val_tokens = {sample.token for sample in validation_samples}
    val_logs = {sample.log_name for sample in validation_samples} | val_logs_from_report

    if base_filenames & val_filenames or base_tokens & val_tokens or base_logs & val_logs:
        raise ValueError("base dataset overlaps validation by filename, token, or log")

    additions = []
    source_reports = []
    occupied_filenames = set(base_filenames) | val_filenames
    occupied_tokens = set(base_tokens) | val_tokens
    train_logs = set(base_logs)

    for source_index, (map_name, root, requested_count) in enumerate(sources):
        candidates = scan_source_samples(root, str(root.resolve()))
        wrong_map = Counter(sample.map_name for sample in candidates if sample.map_name != map_name)
        if wrong_map:
            raise ValueError(
                f"source {root} contains maps other than {map_name}: {dict(wrong_map)}"
            )
        # 验证日志永不回退使用；已有训练日志仅在新日志样本不足时才会补位。
        candidates = [sample for sample in candidates if sample.log_name not in val_logs]
        selected, selection_report = select_balanced_new_logs(
            candidates,
            requested_count,
            seed + source_index,
            occupied_filenames,
            occupied_tokens,
            train_logs,
        )
        additions.extend(selected)
        occupied_filenames.update(sample.filename for sample in selected)
        occupied_tokens.update(sample.token for sample in selected)
        train_logs.update(sample.log_name for sample in selected)
        source_reports.append(
            {
                "map_name": map_name,
                "root": str(root.resolve()),
                "available_npz": len(candidates),
                **selection_report,
            }
        )

    final_samples = base_samples + additions
    _assert_unique(final_samples, "final dataset")
    if {sample.filename for sample in final_samples} & val_filenames:
        raise AssertionError("final dataset overlaps validation filenames")
    if {sample.token for sample in final_samples} & val_tokens:
        raise AssertionError("final dataset overlaps validation tokens")
    if {sample.log_name for sample in final_samples} & val_logs:
        raise AssertionError("final dataset overlaps validation logs")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    manifest_names = sorted(sample.filename for sample in final_samples)
    selected_per_log = Counter(sample.log_name for sample in final_samples)
    map_counts = Counter(sample.map_name for sample in final_samples)
    scenario_type_counts = Counter(sample.scenario_type for sample in final_samples)
    source_counts = Counter(sample.source for sample in final_samples)

    try:
        cache_dir = temporary_dir / "cache"
        cache_dir.mkdir(parents=True)
        by_filename = {sample.filename: sample for sample in final_samples}
        for filename in manifest_names:
            os.link(by_filename[filename].path, cache_dir / filename)

        manifest_path = temporary_dir / "diffusion_planner_training.json"
        sampling_report_path = temporary_dir / "sampling_report.json"
        selection_report_path = temporary_dir / "selection_report.json"
        _write_json(manifest_path, manifest_names)
        _write_json(
            sampling_report_path,
            {
                "strategy": "base_plus_city_target_balanced_new_logs",
                "seed": seed,
                "requested_scenarios": len(final_samples),
                "base_scenarios": len(base_samples),
                "added_scenarios": len(additions),
                "log_names": sorted(selected_per_log),
                "selected_per_log": dict(sorted(selected_per_log.items())),
                "selected_per_map": dict(sorted(map_counts.items())),
                "selected_per_scenario_type": dict(sorted(scenario_type_counts.items())),
            },
        )
        _write_json(
            selection_report_path,
            {
                "status": "passed",
                "seed": seed,
                "link_mode": "hardlink",
                "base_cache": str(Path(base_cache).resolve()),
                "base_manifest": str(Path(base_manifest).resolve()),
                "validation_cache": str(Path(val_cache).resolve()) if val_cache else None,
                "validation_manifest": str(Path(val_manifest).resolve()) if val_manifest else None,
                "validation_report": str(Path(val_report).resolve()) if val_report else None,
                "base_count": len(base_samples),
                "addition_count": len(additions),
                "final_count": len(final_samples),
                "map_counts": dict(sorted(map_counts.items())),
                "scenario_type_counts": dict(sorted(scenario_type_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "source_selection": source_reports,
                "unique_filename_count": len({sample.filename for sample in final_samples}),
                "unique_token_count": len({sample.token for sample in final_samples}),
                "train_log_count": len(selected_per_log),
                "validation_log_count": len(val_logs),
                "validation_overlap": {"filenames": 0, "tokens": 0, "logs": 0},
            },
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    result = {
        "status": "passed",
        "output_dir": str(output_dir.resolve()),
        "manifest_count": len(final_samples),
        "map_counts": dict(sorted(map_counts.items())),
        "train_log_count": len(selected_per_log),
        "manifest_sha256": _sha256(output_dir / "diffusion_planner_training.json"),
        "sampling_report_sha256": _sha256(output_dir / "sampling_report.json"),
    }
    _write_json(output_dir / "build_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_cache", required=True, type=Path)
    parser.add_argument("--base_manifest", required=True, type=Path)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("MAP_NAME", "ROOT", "COUNT"),
        required=True,
        help="可重复传入：地图名、递归 NPZ 根目录、需要新增的样本数",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val_cache", type=Path)
    parser.add_argument("--val_manifest", type=Path)
    parser.add_argument("--val_report", type=Path)
    args = parser.parse_args()

    sources = [(map_name, Path(root), int(count)) for map_name, root, count in args.source]
    result = build_dataset(
        args.base_cache,
        args.base_manifest,
        sources,
        args.output_dir,
        args.seed,
        args.val_cache,
        args.val_manifest,
        args.val_report,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
