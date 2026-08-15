"""Tests for threshold-free pixel-level OOD development metrics."""

import pathlib

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


def test_auroc_does_not_depend_on_a_version_specific_numpy_integrator() -> None:
    """NumPy 2.0 renamed `trapz` to `trapezoid` and later dropped `trapz` entirely, while
    `pyproject.toml` allows `numpy>=1.24,<3` and the pinned Colab training runtime resolves
    to 1.26.4, where only `trapz` exists. Naming either one made AUROC raise
    `AttributeError` on one half of the supported range; it stayed invisible because the
    only callers fed synthetic logits under a NumPy 2.x venv, and first surfaced on a real
    anomaly dataset. The module must therefore reference neither name.
    """
    source = pathlib.Path("src/edgeguard/evaluation/ood.py").read_text(encoding="utf-8")
    assert "np.trapezoid" not in source
    assert "np.trapz" not in source

    # A perfectly separable ranking integrates to exactly 1.0 under the trapezoid rule.
    perfect = pixel_ood_metrics(
        np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.int64),
    )
    assert perfect.auroc == 1.0
    # An interleaved ranking lands strictly between the extremes rather than erroring.
    partial = pixel_ood_metrics(
        np.array([0.9, 0.4, 0.6, 0.1], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.int64),
    )
    assert partial.auroc is not None and 0.0 < partial.auroc < 1.0


def _brute_force_threshold_policies(
    scores: np.ndarray, labels: np.ndarray, *, fixed_threshold: float, risk_budget_fpr: float
) -> tuple[dict[str, float], dict[str, float]]:
    """The original per-candidate sweep, kept only as a correctness oracle.

    It scans the whole array once per distinct score, which is O(n^2) and unusable on real
    data -- that is exactly why the shipped implementation was vectorised. Keeping the
    naive version here pins the fast path to the semantics it replaced.
    """
    valid = labels != 255
    valid_scores = scores[valid].astype(np.float64)
    positives = labels[valid] == 1
    candidates = np.concatenate(
        (np.unique(valid_scores), [np.nextafter(valid_scores.max(), np.inf)])
    )

    def rates(threshold: float) -> dict[str, float]:
        prediction = valid_scores >= threshold
        true_positive = int(np.count_nonzero(prediction & positives))
        false_positive = int(np.count_nonzero(prediction & ~positives))
        false_negative = int(np.count_nonzero(~prediction & positives))
        denominator = 2 * true_positive + false_positive + false_negative
        return {
            "threshold": float(threshold),
            "tpr": float(true_positive / np.count_nonzero(positives)),
            "fpr": float(false_positive / np.count_nonzero(~positives)),
            "f1": float(2 * true_positive / denominator) if denominator else 0.0,
        }

    evaluated = [rates(float(value)) for value in candidates]
    development = max(evaluated, key=lambda row: (row["f1"], -row["fpr"], row["threshold"]))
    budgeted = [row for row in evaluated if row["fpr"] <= risk_budget_fpr]
    risk_budget = max(budgeted, key=lambda row: (row["tpr"], -row["fpr"], row["threshold"]))
    return development, risk_budget


@pytest.mark.parametrize("seed", [1, 7, 20260728])
def test_vectorised_threshold_policies_match_the_naive_sweep(seed: int) -> None:
    """Real anomaly data has ~1.2M distinct scores, where the naive sweep took over an
    hour; the vectorised sweep must return the identical operating points, ties included.
    Duplicated scores and both label classes are forced in so the tie-break ordering
    (f1, then lowest fpr, then highest threshold) is actually exercised.
    """
    generator = np.random.default_rng(seed)
    scores = np.round(generator.normal(size=400), 2).astype(np.float32)
    labels = generator.integers(0, 2, size=400).astype(np.int64)
    labels[generator.choice(400, size=40, replace=False)] = 255

    expected_development, expected_budget = _brute_force_threshold_policies(
        scores, labels, fixed_threshold=0.0, risk_budget_fpr=0.1
    )
    actual = threshold_policies(scores, labels, fixed_threshold=0.0, risk_budget_fpr=0.1)

    assert actual["development_optimal_f1"] == expected_development
    assert {
        key: value
        for key, value in actual["risk_budget_operating_point"].items()
        if key != "maximum_fpr"
    } == expected_budget
