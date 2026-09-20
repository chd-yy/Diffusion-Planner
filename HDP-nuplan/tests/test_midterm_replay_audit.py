from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_midterm_replay import retained_indices, summarize


def test_sampler_seed_is_not_global_training_seed():
    random.seed(42)
    torch.manual_seed(42)
    a = retained_indices(100, 10)
    random.seed(2026)
    torch.manual_seed(2026)
    assert a == retained_indices(100, 10)
    assert a != retained_indices(100, 10, epoch=1)
    assert len(set(a)) == 10


def test_full_capacity_and_bad_capacity():
    assert sorted(retained_indices(10, 10)) == list(range(10))
    with pytest.raises(ValueError): retained_indices(10, 0)
    with pytest.raises(ValueError): retained_indices(10, 11)


def test_metadata_subset_counts():
    metadata = [dict(log_name="a", scenario_type="x"), dict(log_name="b", scenario_type="y")]
    assert summarize(metadata, [1]) == dict(count=1, logs={"b": 1}, types={"y": 1})
