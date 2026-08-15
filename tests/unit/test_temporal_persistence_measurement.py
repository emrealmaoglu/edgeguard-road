"""Pin the two judgement calls in the temporal-persistence measurement.

Both are the kind that silently produce a flattering number rather than an error.

The first is the late-birth exclusion: a track that first appears in the last few frames
cannot accumulate lifetime, so counting it as transient measures where the sequence was
cut rather than how much the system flickers. The second is that `detector_overlap` must
be zero-weighted in *both* scorings -- weighted at a zero value it drags every score down
equally, which is exactly the error zero-weighting exists to prevent, and here it would
also sit on top of the one difference being measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgeguard.context import RiskWeights, contextual_risk  # noqa: E402
from scripts.measure_temporal_persistence import DECLARED_WEIGHTS  # noqa: E402


def _judged(first_seen: dict[int, int], frames: int, saturation: int) -> list[int]:
    """The exclusion rule as the script applies it."""
    deadline = frames - saturation
    return [track for track, start in first_seen.items() if start < deadline]


def test_tracks_born_too_late_are_not_counted_as_flicker() -> None:
    first_seen = {1: 0, 2: 40, 3: 96, 4: 99}

    judged = _judged(first_seen, frames=100, saturation=5)

    assert judged == [1, 2]


def test_the_exclusion_does_not_swallow_a_track_that_had_its_chance() -> None:
    """The boundary matters: a track starting exactly at the deadline is excluded, one
    frame earlier is judged. Off by one here quietly changes the headline number.
    """
    first_seen = {1: 94, 2: 95}

    assert _judged(first_seen, frames=100, saturation=5) == [1]


def test_an_unmeasured_feature_at_zero_weight_does_not_move_the_score() -> None:
    """The contract the whole design rests on, stated as a test: excluding a feature by
    weight leaves the remaining features' relative scores untouched, while excluding it by
    value would drag the total down.
    """
    features = {
        "anomaly_score": 0.8,
        "component_area": 0.5,
        "image_position": 0.9,
        "road_overlap": 0.4,
        "relative_proximity": 0.7,
        "detector_overlap": 0.0,
        "temporal_persistence": 0.0,
    }
    without_detector = RiskWeights(**{**DECLARED_WEIGHTS, "detector_overlap": 0.0})
    detector_weighted_at_zero_value = RiskWeights(**DECLARED_WEIGHTS)

    excluded = contextual_risk(features, without_detector)["total_risk_score"]
    dragged = contextual_risk(features, detector_weighted_at_zero_value)["total_risk_score"]

    assert excluded > dragged


def test_activating_persistence_demotes_a_flicker_and_promotes_a_survivor() -> None:
    """The comparison the measurement actually makes, and why it has teeth.

    The two scorings use different weightings, so they normalise by different totals: 0.90
    without persistence, 0.95 with it. A region seen once gains almost nothing in the
    numerator while the denominator grows, so it genuinely falls -- this is a demotion, not
    merely other regions rising. A region that has survived gains enough to rise. Were both
    scored under one weighting the feature could only ever add, and the measurement would
    be reporting nothing.
    """
    base = {
        "anomaly_score": 0.6,
        "component_area": 0.3,
        "image_position": 0.5,
        "road_overlap": 0.2,
        "relative_proximity": 0.4,
        "detector_overlap": 0.0,
        "temporal_persistence": 0.0,
    }
    with_persistence = RiskWeights(**{**DECLARED_WEIGHTS, "detector_overlap": 0.0})
    without = RiskWeights(
        **{**DECLARED_WEIGHTS, "detector_overlap": 0.0, "temporal_persistence": 0.0}
    )

    baseline = contextual_risk(base, without)["total_risk_score"]
    flicker = contextual_risk(base, with_persistence)["total_risk_score"]
    survivor = contextual_risk({**base, "temporal_persistence": 1.0}, with_persistence)[
        "total_risk_score"
    ]

    assert flicker < baseline < survivor
