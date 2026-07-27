"""Scalar semantic temperature scaling and calibration metrics."""

from edgeguard.calibration.semantic import (
    TemperatureFitResult,
    apply_temperature,
    calibration_metrics,
    fit_temperature,
)

__all__ = (
    "TemperatureFitResult",
    "apply_temperature",
    "calibration_metrics",
    "fit_temperature",
)
