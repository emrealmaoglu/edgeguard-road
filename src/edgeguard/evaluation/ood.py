"""Pixel-level OOD development metrics with no threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ID_LABEL = 0
ANOMALY_LABEL = 1
IGNORE_LABEL = 255


@dataclass(frozen=True)
class OODPixelMetricResult:
    """Threshold-free pixel OOD metrics and evaluated class counts."""

    auroc: float | None
    average_precision: float | None
    fpr_at_95_tpr: float | None
    anomaly_pixel_count: int
    id_pixel_count: int
    ignored_pixel_count: int
    score_direction: str = "higher_means_more_anomalous"


def _validated_pixels(
    scores: npt.NDArray[np.floating], labels: npt.NDArray[np.integer], *, ignore_index: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], int]:
    if not isinstance(scores, np.ndarray) or not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("OOD scores must be a floating-point numpy array")
    if not isinstance(labels, np.ndarray) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("OOD labels must be an integer numpy array")
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("OOD scores and labels must have the same non-empty shape")
    if not np.isfinite(scores).all():
        raise ValueError("OOD scores must be finite")
    allowed = np.isin(labels, np.array([ID_LABEL, ANOMALY_LABEL, ignore_index]))
    if not bool(np.all(allowed)):
        raise ValueError("OOD labels must contain only ID=0, anomaly=1, or ignore")

    valid = labels != ignore_index
    valid_scores = scores[valid].astype(np.float64, copy=False)
    positives = (labels[valid] == ANOMALY_LABEL).astype(np.bool_, copy=False)
    return valid_scores, positives, int(np.count_nonzero(~valid))


def _ranking_curve(
    scores: npt.NDArray[np.float64], positives: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positives[order]
    group_ends = np.concatenate(
        (np.flatnonzero(np.diff(sorted_scores)) + 1, np.array([scores.size], dtype=np.int64))
    )
    true_positives = np.cumsum(sorted_positive, dtype=np.int64)[group_ends - 1]
    false_positives = group_ends - true_positives
    return true_positives.astype(np.float64), false_positives.astype(np.float64)


def pixel_ood_metrics(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    ignore_index: int = IGNORE_LABEL,
) -> OODPixelMetricResult:
    """Compute AP and FPR at at least 95% TPR; higher scores mean anomaly."""
    valid_scores, positives, ignored_count = _validated_pixels(
        scores, labels, ignore_index=ignore_index
    )
    positive_count = int(np.count_nonzero(positives))
    negative_count = int(positives.size - positive_count)
    if positive_count == 0:
        return OODPixelMetricResult(None, None, None, 0, negative_count, ignored_count)

    true_positives, false_positives = _ranking_curve(valid_scores, positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    recall_steps = np.diff(np.concatenate((np.array([0.0]), recall)))
    average_precision = float(np.sum(recall_steps * precision))

    fpr_at_95_tpr: float | None = None
    auroc: float | None = None
    if negative_count > 0:
        first_at_target = int(np.flatnonzero(recall >= 0.95)[0])
        fpr_at_95_tpr = float(false_positives[first_at_target] / negative_count)
        tpr_curve = np.concatenate((np.array([0.0]), true_positives / positive_count))
        fpr_curve = np.concatenate((np.array([0.0]), false_positives / negative_count))
        auroc = float(np.trapezoid(tpr_curve, fpr_curve))
    return OODPixelMetricResult(
        auroc=auroc,
        average_precision=average_precision,
        fpr_at_95_tpr=fpr_at_95_tpr,
        anomaly_pixel_count=positive_count,
        id_pixel_count=negative_count,
        ignored_pixel_count=ignored_count,
    )


def select_anomaly_threshold(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    target_tpr: float,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, float | int | str]:
    """Select the highest-score anomaly threshold meeting a declared development TPR."""
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must lie in (0, 1]")
    valid_scores, positives, ignored_count = _validated_pixels(
        scores, labels, ignore_index=ignore_index
    )
    positive_count = int(np.count_nonzero(positives))
    if positive_count == 0:
        raise ValueError("threshold selection requires anomaly pixels")
    order = np.argsort(-valid_scores, kind="stable")
    sorted_scores = valid_scores[order]
    sorted_positives = positives[order]
    cumulative = np.cumsum(sorted_positives, dtype=np.int64) / positive_count
    index = int(np.flatnonzero(cumulative >= target_tpr)[0])
    threshold = float(sorted_scores[index])
    predicted = valid_scores >= threshold
    true_positive_rate = float(np.count_nonzero(predicted & positives) / positive_count)
    negatives = ~positives
    false_positive_rate = (
        float(np.count_nonzero(predicted & negatives) / np.count_nonzero(negatives))
        if bool(negatives.any())
        else 0.0
    )
    return {
        "schema_version": "1.0",
        "record_type": "ood_development_threshold",
        "score_direction": "higher_means_more_anomalous",
        "target_tpr": target_tpr,
        "threshold": threshold,
        "observed_tpr": true_positive_rate,
        "observed_fpr": false_positive_rate,
        "valid_pixel_count": int(valid_scores.size),
        "ignored_pixel_count": ignored_count,
    }


def score_distribution(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, object]:
    """Return compact class-conditional development distributions without raw scores."""
    valid_scores, positives, ignored_count = _validated_pixels(
        scores, labels, ignore_index=ignore_index
    )

    def summary(values: npt.NDArray[np.float64]) -> dict[str, float | int | None]:
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else None,
            "minimum": float(np.min(values)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
        }

    return {
        "schema_version": "1.0",
        "record_type": "ood_score_distribution",
        "score_direction": "higher_means_more_anomalous",
        "id": summary(valid_scores[~positives]),
        "anomaly": summary(valid_scores[positives]),
        "ignored_pixel_count": ignored_count,
        "scientific_evidence": False,
    }


def threshold_policies(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    fixed_threshold: float,
    risk_budget_fpr: float,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, object]:
    """Evaluate three explicit development-only threshold policies."""
    if not np.isfinite(fixed_threshold) or not 0 <= risk_budget_fpr <= 1:
        raise ValueError("fixed threshold and risk budget must be finite/valid")
    valid_scores, positives, ignored = _validated_pixels(scores, labels, ignore_index=ignore_index)
    if not positives.any() or bool(positives.all()):
        raise ValueError("threshold policies require both ID and anomaly pixels")
    candidates = np.concatenate(
        (
            np.unique(valid_scores),
            np.asarray([np.nextafter(valid_scores.max(), np.inf)], dtype=np.float64),
        )
    )

    def rates(threshold: float) -> dict[str, float]:
        prediction = valid_scores >= threshold
        true_positive = np.count_nonzero(prediction & positives)
        false_positive = np.count_nonzero(prediction & ~positives)
        false_negative = np.count_nonzero(~prediction & positives)
        tpr = true_positive / np.count_nonzero(positives)
        fpr = false_positive / np.count_nonzero(~positives)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        return {
            "threshold": float(threshold),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "f1": float(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
        }

    evaluated = [rates(float(value)) for value in candidates]
    development = max(evaluated, key=lambda row: (row["f1"], -row["fpr"], row["threshold"]))
    within_budget = [row for row in evaluated if row["fpr"] <= risk_budget_fpr]
    risk_budget = max(
        within_budget,
        key=lambda row: (row["tpr"], -row["fpr"], row["threshold"]),
    )
    return {
        "development_optimal_f1": development,
        "fixed_operating_point": rates(fixed_threshold),
        "risk_budget_operating_point": {**risk_budget, "maximum_fpr": risk_budget_fpr},
        "ignored_pixel_count": ignored,
        "scope": "development_only",
        "holdout_or_sealed_tuning_permitted": False,
    }


def per_source_ood_metrics(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    sources: npt.NDArray[np.str_],
    *,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, dict[str, object]]:
    """Compute OOD metrics independently for each explicit source identity."""
    if sources.shape != labels.shape:
        raise ValueError("OOD source identities must match label geometry")
    result: dict[str, dict[str, object]] = {}
    for source in sorted(str(value) for value in np.unique(sources)):
        selected = sources == source
        source_scores = np.asarray(scores[selected], dtype=np.float32)
        source_labels = np.asarray(labels[selected], dtype=np.int64)
        metrics = pixel_ood_metrics(source_scores, source_labels, ignore_index=ignore_index)
        result[source] = metrics.__dict__
    return result


def bootstrap_ood_metrics(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    resamples: int = 200,
    seed: int = 20260727,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, object]:
    """Bootstrap pixel OOD metrics deterministically without a significance claim."""
    valid_scores, positives, ignored = _validated_pixels(scores, labels, ignore_index=ignore_index)
    if valid_scores.size < 2 or not positives.any() or bool(positives.all()) or resamples < 100:
        raise ValueError("OOD bootstrap requires two classes and at least 100 resamples")
    generator = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {"auroc": [], "average_precision": [], "fpr_at_95_tpr": []}
    attempts = 0
    while len(rows["auroc"]) < resamples and attempts < resamples * 10:
        attempts += 1
        indices = generator.integers(0, valid_scores.size, size=valid_scores.size)
        sampled_positive = positives[indices]
        if not sampled_positive.any() or bool(sampled_positive.all()):
            continue
        sampled = pixel_ood_metrics(
            valid_scores[indices], sampled_positive.astype(np.uint8), ignore_index=ignore_index
        )
        for name in rows:
            value = getattr(sampled, name)
            if value is None:
                raise RuntimeError("valid OOD bootstrap unexpectedly produced a null metric")
            rows[name].append(value)
    if len(rows["auroc"]) != resamples:
        raise RuntimeError("OOD bootstrap could not draw enough two-class samples")
    return {
        name: {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in rows.items()
    } | {"resamples": resamples, "seed": seed, "ignored_pixel_count": ignored}
