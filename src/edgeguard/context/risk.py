"""Explainable, config-weighted operational-risk fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FEATURE_NAMES = (
    "anomaly_score",
    "component_area",
    "image_position",
    "road_overlap",
    "relative_proximity",
    "detector_overlap",
    "temporal_persistence",
)


@dataclass(frozen=True)
class RiskWeights:
    """Non-negative contribution weights; values are not scientifically tuned here."""

    anomaly_score: float
    component_area: float
    image_position: float
    road_overlap: float
    relative_proximity: float
    detector_overlap: float
    temporal_persistence: float

    def validated(self) -> RiskWeights:
        values = self.as_dict()
        if any(value < 0.0 for value in values.values()) or sum(values.values()) <= 0.0:
            raise ValueError("risk weights must be non-negative with a positive sum")
        return self

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


def contextual_risk(
    features: dict[str, float],
    weights: RiskWeights,
    *,
    low_max: float = 0.33,
    medium_max: float = 0.66,
) -> dict[str, Any]:
    """Return normalized contributions and an operational label, not physical risk."""
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("contextual risk requires the exact feature contract")
    if not 0.0 < low_max < medium_max < 1.0:
        raise ValueError("risk category boundaries must be ordered inside (0,1)")
    if any(not 0.0 <= float(value) <= 1.0 for value in features.values()):
        raise ValueError("contextual risk features must lie in [0,1]")
    configured = weights.validated().as_dict()
    total_weight = sum(configured.values())
    contributions = {
        name: float(features[name]) * configured[name] / total_weight for name in FEATURE_NAMES
    }
    total = sum(contributions.values())
    category = "low" if total <= low_max else "medium" if total <= medium_max else "high"
    return {
        "schema_version": "1.0",
        "record_type": "contextual_operational_risk",
        "features": {name: float(features[name]) for name in FEATURE_NAMES},
        "configured_weights": configured,
        "normalized_feature_contributions": contributions,
        "total_risk_score": total,
        "risk_category": category,
        "metric_distance": False,
        "calibrated_physical_risk_probability": False,
        "explanation": sorted(contributions, key=lambda name: contributions[name], reverse=True),
    }
