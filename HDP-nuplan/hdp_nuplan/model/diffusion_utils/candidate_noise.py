"""Utilities for deterministic, auditable candidate-noise profiles."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


def validate_candidate_noise_scales(
    num_samples: int,
    noise_scale: float,
    candidate_noise_scales: Optional[Sequence[float]],
) -> tuple[float, ...]:
    """Return one positive finite initial-noise scale per candidate.

    ``noise_scale`` preserves the historical uniform-scale API.  Supplying
    ``candidate_noise_scales`` enables a structured group in which quality
    anchors and exploratory candidates share the same scene conditioning but
    start from different prior temperatures.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if candidate_noise_scales is None:
        scales = (float(noise_scale),) * num_samples
    else:
        scales = tuple(float(value) for value in candidate_noise_scales)
        if len(scales) != num_samples:
            raise ValueError(
                "candidate_noise_scales must contain exactly num_samples "
                f"values, got {len(scales)} for {num_samples} candidates"
            )
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("candidate noise scales must be positive and finite")
    return scales


def expand_scale_counts(scale_counts: Iterable[tuple[float, int]]) -> tuple[float, ...]:
    """Expand ``[(scale, count), ...]`` into a candidate-scale schedule."""
    schedule = []
    for scale, count in scale_counts:
        if int(count) != count or count < 1:
            raise ValueError("candidate scale counts must be positive integers")
        schedule.extend([float(scale)] * int(count))
    if not schedule:
        raise ValueError("candidate scale schedule must not be empty")
    return validate_candidate_noise_scales(
        len(schedule), schedule[0], schedule
    )


def parse_scale_count_profile(specification: str) -> tuple[float, ...]:
    """Parse ``0.1x11+0.5x16`` into an explicit candidate schedule."""
    if not specification or not specification.strip():
        raise ValueError("candidate scale profile must not be empty")
    scale_counts = []
    for raw_segment in specification.split("+"):
        segment = raw_segment.strip()
        if "x" not in segment:
            raise ValueError(f"candidate scale segment lacks 'x': {segment}")
        raw_scale, raw_count = segment.split("x", 1)
        scale_counts.append((float(raw_scale), int(raw_count)))
    return expand_scale_counts(scale_counts)
