import pytest

from scripts.validate_candidate_generation_profiles import parse_profiles


def test_parse_profiles_expands_all_candidate_scales():
    profiles = parse_profiles(
        "baseline=0.1x2;anchor_explore=0.1x1+0.5x1",
        group_size=2,
    )
    assert profiles["baseline"] == (0.1, 0.1)
    assert profiles["anchor_explore"] == (0.1, 0.5)


def test_parse_profiles_rejects_wrong_group_size():
    with pytest.raises(ValueError, match="at least group-size"):
        parse_profiles("invalid=0.1x1", group_size=2)


def test_parse_profiles_accepts_oversampled_pool():
    profiles = parse_profiles("pool=0.1x2+0.5x2", group_size=2)
    assert profiles["pool"] == (0.1, 0.1, 0.5, 0.5)


def test_parse_profiles_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        parse_profiles("same=0.1x2;same=0.5x2", group_size=2)
