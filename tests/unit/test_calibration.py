"""Tests for deterministic scalar semantic calibration."""

import numpy as np
import pytest

from edgeguard.calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
)

BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _calibration_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 3, size=(2, 4, 5), dtype=np.int64)
    targets[0, 0, 0] = 255
    logits = rng.normal(0.0, 1.0, size=(2, 3, 4, 5)).astype(np.float32)
    for batch_index, target in enumerate(targets):
        rows, columns = np.where(target != 255)
        logits[batch_index, target[rows, columns], rows, columns] += np.float32(0.5)
    return np.asarray(logits * np.float32(3.0), dtype=np.float32), targets


def test_temperature_fit_is_positive_deterministic_and_does_not_worsen_nll() -> None:
    logits, targets = _calibration_fixture()
    original = logits.copy()

    first = fit_temperature(logits, targets, max_iterations=48)
    second = fit_temperature(logits, targets, max_iterations=48)

    assert first == second
    assert first.final_temperature > 0.0
    assert first.final_nll <= first.initial_nll + 1.0e-12
    assert first.valid_pixel_count == targets.size - 1
    np.testing.assert_array_equal(logits, original)


def test_apply_temperature_preserves_shape_dtype_and_input() -> None:
    logits, _targets = _calibration_fixture()
    original = logits.copy()

    scaled = apply_temperature(logits, 2.0)

    assert scaled.shape == logits.shape
    assert scaled.dtype == logits.dtype
    assert scaled is not logits
    np.testing.assert_array_equal(logits, original)
    np.testing.assert_allclose(scaled, logits / np.float32(2.0))


def test_calibration_metrics_ignore_255_and_emit_explicit_reliability_bins() -> None:
    logits, targets = _calibration_fixture()

    metrics = calibration_metrics(
        logits,
        targets,
        temperature=1.0,
        bin_edges=BIN_EDGES,
    )

    assert metrics["valid_pixel_count"] == targets.size - 1
    assert metrics["bin_edges"] == list(BIN_EDGES)
    assert sum(item["support"] for item in metrics["reliability_diagram"]) == targets.size - 1
    assert metrics["nll"] >= 0.0
    assert metrics["ece"] >= 0.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["probability_interpretation"] == "semantic_class_confidence_only"
    assert metrics["calibrates_raw_ood_scores"] is False


def test_calibration_rejects_empty_valid_region_and_invalid_temperature() -> None:
    logits = np.zeros((1, 3, 2, 2), dtype=np.float32)
    ignored = np.full((1, 2, 2), 255, dtype=np.int64)

    with pytest.raises(ValueError, match="no valid pixels"):
        fit_temperature(logits, ignored)
    with pytest.raises(ValueError, match="positive finite"):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError, match="positive finite"):
        calibration_metrics(logits, np.zeros_like(ignored), temperature=0.0, bin_edges=BIN_EDGES)
