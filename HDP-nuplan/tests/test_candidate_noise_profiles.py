import math

import pytest

from hdp_nuplan.model.diffusion_utils.candidate_noise import (
    expand_scale_counts,
    parse_scale_count_profile,
    validate_candidate_noise_scales,
)


def test_uniform_scale_preserves_historical_api():
    assert validate_candidate_noise_scales(4, 0.1, None) == (0.1,) * 4


def test_structured_scale_schedule_is_preserved():
    scales = (0.1, 0.1, 0.5, 1.0)
    assert validate_candidate_noise_scales(4, 0.1, scales) == scales


def test_structured_scale_schedule_requires_one_scale_per_candidate():
    with pytest.raises(ValueError, match="exactly num_samples"):
        validate_candidate_noise_scales(4, 0.1, (0.1, 0.5))


@pytest.mark.parametrize("scale", [0.0, -0.1, math.inf, math.nan])
def test_candidate_scales_must_be_positive_and_finite(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        validate_candidate_noise_scales(1, 0.1, (scale,))


def test_expand_scale_counts_builds_anchor_explore_profile():
    assert expand_scale_counts(((0.1, 2), (0.5, 1), (1.0, 1))) == (
        0.1,
        0.1,
        0.5,
        1.0,
    )


def test_parse_scale_count_profile():
    assert parse_scale_count_profile("0.1x2+0.5x1") == (0.1, 0.1, 0.5)


def test_parse_scale_count_profile_requires_count_separator():
    with pytest.raises(ValueError, match="lacks 'x'"):
        parse_scale_count_profile("0.1+0.5x1")


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_expand_scale_counts_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="positive integers"):
        expand_scale_counts(((0.1, count),))
