import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from validate_processed_cache import (  # noqa: E402
    EXPECTED_SHAPES,
    validate_cache,
)


def _write_sample(path: Path, log_name: str, token: str) -> None:
    values = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in EXPECTED_SHAPES.items()
    }
    values.update(
        {
            "log_name": log_name,
            "map_name": "test-map",
            "scenario_type": "test-type",
            "token": token,
        }
    )
    np.savez(path, **values)


def _write_metadata(tmp_path: Path, filenames: list[str]) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(sorted(filenames)), encoding="utf-8")
    report = tmp_path / "sampling_report.json"
    report.write_text(
        json.dumps(
            {
                "requested_scenarios": len(filenames),
                "selected_per_log": {"log-a": 1, "log-b": 1},
            }
        ),
        encoding="utf-8",
    )
    return manifest, report


def test_validate_cache_accepts_exact_finite_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    filenames = ["map-a.npz", "map-b.npz"]
    _write_sample(cache / filenames[0], "log-a", "a")
    _write_sample(cache / filenames[1], "log-b", "b")
    manifest, report = _write_metadata(tmp_path, filenames)

    result = validate_cache(cache, manifest, report, expected_count=2, expected_log_count=2)

    assert result["status"] == "passed"
    assert result["manifest_count"] == 2
    assert result["selected_per_log_min"] == 1
    assert result["selected_per_log_max"] == 1


def test_validate_cache_rejects_unexpected_npz(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    filenames = ["map-a.npz", "map-b.npz"]
    _write_sample(cache / filenames[0], "log-a", "a")
    _write_sample(cache / filenames[1], "log-b", "b")
    _write_sample(cache / "stale.npz", "log-a", "stale")
    manifest, report = _write_metadata(tmp_path, filenames)

    with pytest.raises(ValueError, match="unexpected=1"):
        validate_cache(cache, manifest, report)


def test_validate_cache_accepts_nested_shard_paths(tmp_path):
    cache = tmp_path / "cache-root"
    nested = cache / "shard_00000" / "cache"
    nested.mkdir(parents=True)
    filename = "shard_00000/cache/map-a.npz"
    _write_sample(cache / filename, "log-a", "a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([filename]), encoding="utf-8")
    report = tmp_path / "sampling_report.json"
    report.write_text(
        json.dumps(
            {
                "requested_scenarios": 1,
                "selected_per_log": {"log-a": 1},
            }
        ),
        encoding="utf-8",
    )

    result = validate_cache(cache, manifest, report, expected_count=1)

    assert result["status"] == "passed"
    assert result["manifest_count"] == 1
