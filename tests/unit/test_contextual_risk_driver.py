"""Tests for the contextual-risk frame driver's honesty and feature contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgeguard.context import RiskWeights, contextual_risk  # noqa: E402
from scripts.analyze_contextual_risk import (  # noqa: E402
    DECLARED_WEIGHTS,
    _normalise_anomaly,
    _road_adjacency,
)


def test_absent_features_are_zero_weighted_not_zero_valued() -> None:
    """A feature nobody measured must not look like a measured absence of risk.

    `contextual_risk` divides by the summed weight, so feeding 0.0 for an unavailable
    signal silently scales every region down. Dropping its weight instead renormalises
    over the features actually present, leaving the measured ones' relative pull intact.
    """
    measured = {
        "anomaly_score": 0.9,
        "component_area": 0.5,
        "image_position": 0.8,
        "road_overlap": 1.0,
        "relative_proximity": 0.7,
    }
    absent = {"detector_overlap": 0.0, "temporal_persistence": 0.0}

    zero_valued = contextual_risk({**measured, **absent}, RiskWeights(**DECLARED_WEIGHTS))
    zero_weighted = contextual_risk(
        {**measured, **absent},
        RiskWeights(**{**DECLARED_WEIGHTS, "detector_overlap": 0.0, "temporal_persistence": 0.0}),
    )

    assert zero_weighted["total_risk_score"] > zero_valued["total_risk_score"]
    # With the unmeasured pair removed the remaining weights carry the whole score.
    assert zero_weighted["total_risk_score"] == pytest.approx(
        sum(measured[name] * DECLARED_WEIGHTS[name] for name in measured)
        / sum(DECLARED_WEIGHTS[name] for name in measured)
    )


def test_zero_weighting_every_feature_is_refused() -> None:
    """`RiskWeights.validated` requires a positive sum, so an all-absent frame cannot
    silently produce a score of zero that reads like 'no risk'.
    """
    with pytest.raises(ValueError, match="non-negative"):
        contextual_risk(
            dict.fromkeys(DECLARED_WEIGHTS, 0.5),
            RiskWeights(**dict.fromkeys(DECLARED_WEIGHTS, 0.0)),
        )


def test_road_adjacency_measures_contact_rather_than_containment() -> None:
    """An obstacle on the carriageway matters more than one against a wall, and what
    separates them is whether the region *touches* road, not whether road surrounds it.
    """
    region = np.zeros((7, 7), dtype=np.bool_)
    region[3, 3] = True

    all_road = np.ones((7, 7), dtype=np.bool_)
    all_road[region] = False
    assert _road_adjacency(region, all_road) == pytest.approx(1.0)

    no_road = np.zeros((7, 7), dtype=np.bool_)
    assert _road_adjacency(region, no_road) == 0.0

    half_road = np.zeros((7, 7), dtype=np.bool_)
    half_road[4, 3] = True
    half_road[2, 3] = True
    assert _road_adjacency(region, half_road) == pytest.approx(0.5)


def test_a_region_with_no_border_scores_no_road_contact() -> None:
    """A region covering the whole frame has no four-neighbourhood outside itself."""
    full = np.ones((4, 4), dtype=np.bool_)
    assert _road_adjacency(full, np.ones((4, 4), dtype=np.bool_)) == 0.0


def test_anomaly_normalisation_is_bounded_and_reports_its_reference() -> None:
    """Energy is unbounded and its scale moves per architecture and scene, so the record
    has to carry the bounds a score was formed against or it cannot be traced back.
    """
    generator = np.random.default_rng(20260728)
    energy = generator.normal(loc=-7.0, scale=2.0, size=(32, 64)).astype(np.float32)

    scaled, bounds = _normalise_anomaly(energy)

    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    assert bounds["low"] < bounds["high"]
    assert bounds["low"] == pytest.approx(float(np.percentile(energy, 5)))


def test_a_flat_energy_map_normalises_to_zero_instead_of_dividing_by_zero() -> None:
    scaled, bounds = _normalise_anomaly(np.full((8, 8), -4.0, dtype=np.float32))
    assert not scaled.any()
    assert bounds["low"] == bounds["high"]


def test_declared_weights_cover_every_feature_the_fusion_requires() -> None:
    """`contextual_risk` rejects any features dict that is not exactly its contract, so a
    drifting weight table would fail at the frame rather than at import.
    """
    assert contextual_risk(dict.fromkeys(DECLARED_WEIGHTS, 0.5), RiskWeights(**DECLARED_WEIGHTS))[
        "risk_category"
    ] in {"low", "medium", "high"}
