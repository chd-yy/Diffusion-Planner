from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_midterm_closed_loop import command_for, frozen_write, validate_fixed, check_deadline


def fixture_fixed():
    tokens = [f"{i:016x}" for i in range(200)]
    return dict(scenario_count=200, scenario_tokens=tokens,
                scenarios=[dict(scenario=t, log_name="validation", scenario_type="turn") for t in tokens])


def test_complete_disjoint_fixed200():
    fixed = fixture_fixed()
    validate_fixed(fixed, ["ffffffffffffffff"], ["training"])
    with pytest.raises(ValueError): validate_fixed(fixed, [fixed["scenario_tokens"][0]], [])
    with pytest.raises(ValueError): validate_fixed(fixed, [], ["validation"])


@pytest.mark.parametrize("change", ["duplicate", "missing", "numeric"])
def test_bad_fixed200_rejected(change):
    fixed = deepcopy(fixture_fixed())
    if change == "duplicate": fixed["scenario_tokens"][-1] = fixed["scenario_tokens"][0]
    if change == "missing": fixed["scenarios"].pop()
    if change == "numeric": fixed["scenario_tokens"][0] = 0
    with pytest.raises(ValueError): validate_fixed(fixed, [], [])


def test_frozen_inputs_cannot_be_silently_overwritten(tmp_path):
    path = tmp_path / "protocol.json"
    frozen_write(path, {"seed": 0})
    frozen_write(path, {"seed": 0})
    with pytest.raises(RuntimeError): frozen_write(path, {"seed": 1})
    assert json.loads(path.read_text()) == {"seed": 0}


def test_single_worker_and_frozen_seed_command(tmp_path):
    cmd = command_for(dict(args="/a.json", checkpoint="/b.pth"), 7,
                      dict(filter="fixed"), "test", tmp_path)
    assert "worker.max_workers=1" in cmd
    assert "worker.use_process_pool=false" in cmd
    assert "disable_callback_parallelization=true" in cmd
    assert "seed=7" in cmd
    assert "~callback.simulation_log_callback" in cmd


def test_freeze_blocks_new_work():
    with pytest.raises(RuntimeError):
        check_deadline({"stop_new_work_at": "2020-09-21T12:00:00+08:00"})
    check_deadline({"stop_new_work_at": "2099-09-21T12:00:00+08:00"})
