from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from summarize_closed_loop_metrics import METRIC_COLUMNS, summarize_run  # noqa: E402


def test_closed_loop_summary_ignores_aggregated_rows(tmp_path):
    aggregate_dir = tmp_path / "aggregator_metric"
    aggregate_dir.mkdir()

    scenario = {
        "scenario": "token-a",
        "log_name": "log-a",
        "scenario_type": "stationary",
    }
    scenario.update({metric: 0.5 for metric in METRIC_COLUMNS})
    final_row = dict(scenario)
    final_row.update({"scenario": "final_score", "log_name": None})
    pd.DataFrame([scenario, final_row]).to_parquet(aggregate_dir / "metrics.parquet")

    pd.DataFrame(
        [
            {
                "succeeded": True,
                "scenario_name": "token-a",
                "compute_trajectory_runtimes_mean": 0.2,
                "duration": 12.0,
            }
        ]
    ).to_parquet(tmp_path / "runner_report.parquet")

    result = summarize_run(tmp_path)

    assert result["scenario_count"] == 1
    assert result["successful_simulations"] == 1
    assert result["failed_simulations"] == 0
    assert result["means"]["score"] == pytest.approx(0.5)
    assert result["means"]["trajectory_runtime_mean_s"] == pytest.approx(0.2)
    assert result["scenarios"][0]["scenario"] == "token-a"
