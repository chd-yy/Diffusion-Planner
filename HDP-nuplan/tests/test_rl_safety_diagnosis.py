import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_rl_safety_regressions import compare_safety
from summarize_closed_loop_metrics import METRIC_COLUMNS


def summary():
    rows = []
    for token in ("a", "b"):
        row = dict(log_name="log", scenario=token)
        row.update({metric: 1.0 for metric in METRIC_COLUMNS})
        rows.append(row)
    return dict(scenarios=rows, runner_report_complete=True, failed_simulations=0)


def test_identical_diagnosis_passes_but_never_authorizes_promotion():
    result = compare_safety(summary(), summary())
    assert result["diagnostic_gate_passed"]
    assert not result["promotion_authorized"]


def test_mean_gain_cannot_hide_new_collision():
    base = summary()
    base["scenarios"][1]["score"] = 0.0
    candidate = copy.deepcopy(base)
    candidate["scenarios"][0]["no_ego_at_fault_collisions"] = 0.0
    candidate["scenarios"][1]["score"] = 1.0
    result = compare_safety(base, candidate)
    assert result["mean_deltas"]["score"] > 0
    assert not result["diagnostic_gate_passed"]
    assert result["new_regressions"]["no_ego_at_fault_collisions"] == ["a"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "failed", "incomplete"])
def test_bad_evaluation_cannot_pass(defect):
    candidate = summary()
    if defect == "missing":
        candidate["scenarios"].pop()
    elif defect == "duplicate":
        candidate["scenarios"].append(copy.deepcopy(candidate["scenarios"][0]))
    elif defect == "nan":
        candidate["scenarios"][0]["drivable_area_compliance"] = float("nan")
    elif defect == "failed":
        candidate["failed_simulations"] = 1
    else:
        candidate["runner_report_complete"] = False
    with pytest.raises(ValueError):
        compare_safety(summary(), candidate)
