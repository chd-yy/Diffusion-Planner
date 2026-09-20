import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_midterm_paired import METRICS, clustered_interval, compare, validate_run


def run():
    rows = [dict(log_name=f"log{i // 2}", scenario=str(i), scenario_type="following",
                 **{m: 0.5 for m in METRICS}) for i in range(6)]
    return dict(scenarios=rows, scenario_count=6, successful_simulations=6,
                failed_simulations=0, runner_report_count=6, runner_report_complete=True,
                means={m: 0.5 for m in METRICS})


def test_identity_and_seed_reproducibility():
    r = compare(run(), run(), repeats=100)
    assert r == compare(run(), run(), repeats=100)
    assert all(d["log_cluster_percentile_interval_95"] == [0., 0.] for d in r["metrics"].values())
    assert r["top4_share_of_total_score_decrease"] is None


def test_pair_order_invariant_and_constant_improvement():
    b, c = run(), run()
    for row in c["scenarios"]:
        row["score"] = 0.75
    c["means"]["score"] = 0.75
    c["scenarios"].reverse()
    r = compare(b, c, repeats=100)
    assert r["metrics"]["score"]["mean_delta"] == .25
    assert r["metrics"]["score"]["log_cluster_percentile_interval_95"] == [.25, .25]


@pytest.mark.parametrize("fault", ["duplicate", "nan", "missing", "mean", "count", "runner", "type"])
def test_invalid_summaries_fail_closed(fault):
    b, c = run(), run()
    if fault == "duplicate": c["scenarios"][1] = copy.deepcopy(c["scenarios"][0])
    if fault == "nan": c["scenarios"][0]["score"] = float("nan")
    if fault == "missing": c["scenarios"][0].pop("score")
    if fault == "mean": c["means"]["score"] = .6
    if fault == "count": c["scenario_count"] = 5
    if fault == "runner": c["runner_report_complete"] = False
    if fault == "type": c["scenarios"][0]["scenario_type"] = "changed"
    with pytest.raises(ValueError): compare(b, c, repeats=100)


def test_one_cluster_and_cluster_not_row_bootstrap():
    assert clustered_interval(np.array([[0.], [1.]]), ["a", "a"], repeats=100) is None
    # Identical rows within each block must never be separated when resampling.
    interval = clustered_interval(np.array([[0.], [0.], [1.], [1.]]),
                                  ["a", "a", "b", "b"], repeats=1000)
    assert interval == [[0., 1.]]


def test_changed_coverage_is_rejected():
    b, c = run(), run()
    c["scenarios"][0]["scenario"] = "different"
    with pytest.raises(ValueError): compare(b, c, repeats=100)
