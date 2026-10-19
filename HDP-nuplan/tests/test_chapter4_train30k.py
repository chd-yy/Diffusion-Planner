from collections import Counter
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_chapter4_train30k import (  # noqa: E402
    allocate_sqrt,
    motion_bin,
    select_rows,
)


def test_motion_bins_have_explicit_boundaries():
    assert motion_bin(1.0) == "stationary_0_1m"
    assert motion_bin(1.01) == "low_1_10m"
    assert motion_bin(10.0) == "low_1_10m"
    assert motion_bin(10.01) == "medium_10_50m"
    assert motion_bin(50.0) == "medium_10_50m"
    assert motion_bin(50.01) == "high_over_50m"


def test_sqrt_allocation_is_exact_and_capacity_safe():
    allocation = allocate_sqrt({"common": 100, "rare": 4}, 20)
    assert sum(allocation.values()) == 20
    assert 0 < allocation["rare"] <= 4
    assert allocation["common"] <= 100


def test_log_quotas_and_stratified_selection_are_deterministic():
    rows = []
    for log in ("a", "b", "c"):
        for index in range(20):
            rows.append({
                "name": f"{log}_{index:02d}.npz",
                "map_name": "map",
                "log_name": log,
                "scenario_type": "rare" if index < 2 else "common",
                "token": f"{log}{index:02d}",
                "path_length_m": float(index),
                "motion_bin": motion_bin(float(index)),
            })
    first, quotas, _ = select_rows(rows, 30, 3407)
    second, _, _ = select_rows(rows, 30, 3407)
    assert [row["name"] for row in first] == [row["name"] for row in second]
    assert len(first) == len({row["name"] for row in first}) == 30
    assert Counter(row["log_name"] for row in first) == Counter(quotas)
    assert set(quotas.values()) == {10}

    required = {"a_00.npz", "b_01.npz", "c_02.npz"}
    nested, _, _ = select_rows(rows, 30, 3407, required_names=required)
    assert required <= {row["name"] for row in nested}
