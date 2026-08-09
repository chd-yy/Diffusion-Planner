import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_closed_loop_20_selection_matches_filter_and_audit_rules():
    audit = json.loads(
        (PROJECT_DIR / "config/mini_splits/mini_val_closed_loop_20_selection.json")
        .read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (PROJECT_DIR / "hdp_nuplan/config/scenario_filter/mini-val-closed-loop-20.yaml")
        .read_text(encoding="utf-8")
    )

    selected = audit["selected"]
    tokens = [item["token"] for item in selected]
    assert len(tokens) == len(set(tokens)) == audit["target_scenarios"] == 20
    assert config["scenario_tokens"] == tokens
    assert config["remove_invalid_goals"] is True
    assert config["expand_scenarios"] is False
    assert config["shuffle"] is False

    fixed_three = {
        "08764235932a5530",
        "134f95eed3775e22",
        "263da852dba35cac",
    }
    assert fixed_three.issubset(tokens)

    minimum_separation = audit["selection_rules"][
        "minimum_same_log_start_separation_s"
    ]
    by_log = {}
    for item in selected:
        by_log.setdefault(item["log_name"], []).append(item["start_time_s"])
        assert audit["valid_goal_audit"]["valid_candidates_per_log"][item["log_name"]] > 0

    for start_times in by_log.values():
        start_times.sort()
        assert all(
            later - earlier >= minimum_separation
            for earlier, later in zip(start_times, start_times[1:])
        )
