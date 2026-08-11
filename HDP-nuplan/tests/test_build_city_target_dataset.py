import json
from pathlib import Path
import sys

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_city_target_dataset import build_dataset  # noqa: E402


def _write_sample(root: Path, map_name: str, log_name: str, token: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{map_name}_{token}.npz"
    np.savez(
        path,
        map_name=map_name,
        log_name=log_name,
        scenario_type="test-type",
        token=token,
    )
    return path


def _write_manifest(path: Path, names: list[str]) -> None:
    path.write_text(json.dumps(sorted(names)), encoding="utf-8")


def test_build_dataset_is_exact_deterministic_and_validation_safe(tmp_path):
    base_cache = tmp_path / "base" / "cache"
    base = _write_sample(base_cache, "las-vegas-strip", "base-log", "base")
    base_manifest = tmp_path / "base" / "manifest.json"
    _write_manifest(base_manifest, [base.name])

    val_cache = tmp_path / "val" / "cache"
    val = _write_sample(val_cache, "sg-one-north", "val-log", "val")
    val_manifest = tmp_path / "val" / "manifest.json"
    _write_manifest(val_manifest, [val.name])
    val_report = tmp_path / "val" / "sampling_report.json"
    val_report.write_text(json.dumps({"log_names": ["val-log"]}), encoding="utf-8")

    source = tmp_path / "source"
    # base token、validation token 和 validation log 都必须被排除。
    _write_sample(source / "shard-a", "sg-one-north", "other-log", "base")
    _write_sample(source / "shard-a", "sg-one-north", "other-log", "val")
    _write_sample(source / "shard-a", "sg-one-north", "val-log", "blocked-by-log")
    first = _write_sample(source / "shard-a", "sg-one-north", "new-log-a", "new-a")
    second = _write_sample(source / "shard-b", "sg-one-north", "new-log-b", "new-b")
    _write_sample(source / "shard-b", "sg-one-north", "new-log-a", "new-c")

    output = tmp_path / "output"
    result = build_dataset(
        base_cache,
        base_manifest,
        [("sg-one-north", source, 2)],
        output,
        3407,
        val_cache,
        val_manifest,
        val_report,
    )

    manifest = json.loads((output / "diffusion_planner_training.json").read_text())
    report = json.loads((output / "selection_report.json").read_text())
    assert result["manifest_count"] == 3
    assert result["map_counts"] == {"las-vegas-strip": 1, "sg-one-north": 2}
    assert len(manifest) == len(set(manifest)) == 3
    assert "sg-one-north_val.npz" not in manifest
    assert "sg-one-north_blocked-by-log.npz" not in manifest
    assert report["validation_overlap"] == {"filenames": 0, "tokens": 0, "logs": 0}
    assert report["source_selection"][0]["selected_from_new_logs"] == 2
    assert report["source_selection"][0]["selected_log_count"] == 2

    source_by_name = {first.name: first, second.name: second}
    for name in set(manifest) & set(source_by_name):
        assert (output / "cache" / name).stat().st_ino == source_by_name[name].stat().st_ino
