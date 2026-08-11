from dataclasses import dataclass

import pytest

from hdp_nuplan.data_process.sampling import (
    scenario_output_name,
    select_scenarios_balanced_by_log,
)


@dataclass
class FakeScenario:
    log_name: str
    token: str
    scenario_type: str = "test"
    _map_name: str = "map"


def make_scenarios(log_name, count):
    return [FakeScenario(log_name, f"{log_name}-{index}") for index in range(count)]


def test_balanced_sampling_covers_every_log_and_is_reproducible():
    scenarios = (
        make_scenarios("log-a", 5)
        + make_scenarios("log-b", 5)
        + make_scenarios("log-c", 5)
    )

    first, report = select_scenarios_balanced_by_log(
        scenarios, ["log-a", "log-b", "log-c"], 10, seed=3407
    )
    second, _ = select_scenarios_balanced_by_log(
        scenarios, ["log-a", "log-b", "log-c"], 10, seed=3407
    )

    assert [scenario_output_name(item) for item in first] == [
        scenario_output_name(item) for item in second
    ]
    assert len(first) == 10
    assert report["base_quota_per_log"] == 3
    assert all(count >= 3 for count in report["selected_per_log"].values())


def test_balanced_sampling_removes_duplicate_output_names():
    scenarios = make_scenarios("log-a", 3) + make_scenarios("log-b", 3)
    scenarios.append(FakeScenario("log-a", "log-a-0", scenario_type="another-tag"))

    selected, report = select_scenarios_balanced_by_log(
        scenarios, ["log-a", "log-b"], 6, seed=1
    )

    assert len(selected) == 6
    assert report["duplicate_output_names_removed"] == 1
    assert len({scenario_output_name(item) for item in selected}) == 6


def test_balanced_sampling_rejects_too_few_scenarios_for_log_coverage():
    with pytest.raises(ValueError, match="smaller than"):
        select_scenarios_balanced_by_log(
            make_scenarios("log-a", 3) + make_scenarios("log-b", 3),
            ["log-a", "log-b"],
            total_scenarios=1,
            seed=1,
        )


def test_balanced_sampling_requires_explicit_permission_for_empty_logs():
    with pytest.raises(RuntimeError, match="No scenarios found"):
        select_scenarios_balanced_by_log(
            make_scenarios("log-a", 10),
            ["log-a", "short-empty-log"],
            total_scenarios=8,
            seed=1,
        )


def test_balanced_sampling_reports_empty_logs_and_preserves_total_target():
    selected, report = select_scenarios_balanced_by_log(
        make_scenarios("log-a", 10) + make_scenarios("log-b", 10),
        ["log-a", "short-empty-log", "log-b"],
        total_scenarios=12,
        seed=1,
        allow_empty_logs=True,
    )

    assert len(selected) == 12
    assert report["requested_log_count"] == 3
    assert report["eligible_log_count"] == 2
    assert report["empty_log_count"] == 1
    assert report["empty_logs"] == ["short-empty-log"]
    assert report["base_quota_per_log"] == 6
    assert report["selected_per_log"] == {"log-a": 6, "log-b": 6}
