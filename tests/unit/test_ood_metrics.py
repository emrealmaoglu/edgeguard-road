"""Tests for threshold-free pixel-level OOD development metrics."""

import numpy as np
import pytest

from edgeguard.evaluation.ood import (
    bootstrap_ood_metrics,
    per_source_ood_metrics,
    pixel_ood_metrics,
    score_distribution,
    threshold_policies,
)


def test_ood_completion_records_are_deterministic_and_development_only() -> None:
    scores = np.asarray([0.05, 0.2, 0.7, 0.95, 0.8, 0.1], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 1, 255], dtype=np.uint8)
    policies = threshold_policies(scores, labels, fixed_threshold=0.5, risk_budget_fpr=0.0)
    assert policies["scope"] == "development_only"
    assert policies["holdout_or_sealed_tuning_permitted"] is False
    assert policies["risk_budget_operating_point"]["fpr"] == 0.0
    distribution = score_distribution(scores, labels)
    assert distribution["scientific_evidence"] is False
    first = bootstrap_ood_metrics(scores, labels, resamples=100, seed=8)
    second = bootstrap_ood_metrics(scores, labels, resamples=100, seed=8)
    assert first == second
    assert first["auroc"]["lower_95"] <= first["auroc"]["upper_95"]


def test_ood_per_source_metrics_preserve_source_identity() -> None:
    scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    sources = np.asarray(["static", "static", "synthetic", "synthetic"])
    result = per_source_ood_metrics(scores, labels, sources)
    assert list(result) == ["static", "synthetic"]
    assert all(row["score_direction"] == "higher_means_more_anomalous" for row in result.values())


def test_pixel_ood_metrics_perfect_ranking() -> None:
    result = pixel_ood_metrics(
        np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    )

    assert result.average_precision == pytest.approx(1.0)
    assert result.fpr_at_95_tpr == pytest.approx(0.0)


def test_pixel_ood_metrics_reversed_ranking() -> None:
    result = pixel_ood_metrics(
        np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    )

    assert result.average_precision == pytest.approx(5.0 / 12.0)
    assert result.fpr_at_95_tpr == pytest.approx(1.0)


def test_pixel_ood_metrics_ties_are_grouped() -> None:
    result = pixel_ood_metrics(
        np.ones(4, dtype=np.float32),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )

    assert result.average_precision == pytest.approx(0.5)
    assert result.fpr_at_95_tpr == pytest.approx(1.0)


def test_pixel_ood_metrics_extreme_class_imbalance() -> None:
    scores = np.zeros(10_001, dtype=np.float32)
    scores[0] = 1.0
    labels = np.zeros(10_001, dtype=np.uint8)
    labels[0] = 1

    result = pixel_ood_metrics(scores, labels)

    assert result.average_precision == pytest.approx(1.0)
    assert result.fpr_at_95_tpr == pytest.approx(0.0)
    assert result.anomaly_pixel_count == 1
    assert result.id_pixel_count == 10_000


def test_pixel_ood_metrics_ignore_void_pixels() -> None:
    result = pixel_ood_metrics(
        np.array([0.9, 0.1, 100.0], dtype=np.float32),
        np.array([1, 0, 255], dtype=np.uint8),
    )

    assert result.average_precision == pytest.approx(1.0)
    assert result.fpr_at_95_tpr == pytest.approx(0.0)
    assert result.ignored_pixel_count == 1


def test_pixel_ood_metrics_without_positive_pixels_is_undefined() -> None:
    result = pixel_ood_metrics(
        np.array([0.2, 0.1], dtype=np.float32),
        np.array([0, 0], dtype=np.uint8),
    )

    assert result.average_precision is None
    assert result.fpr_at_95_tpr is None


def test_pixel_ood_metrics_without_negative_pixels_has_undefined_fpr() -> None:
    result = pixel_ood_metrics(
        np.array([0.2, 0.1], dtype=np.float32),
        np.array([1, 1], dtype=np.uint8),
    )

    assert result.average_precision == pytest.approx(1.0)
    assert result.fpr_at_95_tpr is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_pixel_ood_metrics_rejects_non_finite_scores(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        pixel_ood_metrics(
            np.array([0.0, invalid], dtype=np.float32),
            np.array([0, 1], dtype=np.uint8),
        )


def test_pixel_ood_metrics_rejects_invalid_labels_and_shapes() -> None:
    with pytest.raises(ValueError, match="only ID=0"):
        pixel_ood_metrics(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0, 2], dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="same non-empty shape"):
        pixel_ood_metrics(
            np.array([0.0], dtype=np.float32),
            np.array([0, 1], dtype=np.uint8),
        )
