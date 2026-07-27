"""Deterministic scalar temperature fitting for semantic class confidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from edgeguard.scoring.uncertainty import stable_softmax


@dataclass(frozen=True)
class TemperatureFitResult:
    """Path-free result of bounded one-dimensional log-temperature fitting."""

    initial_temperature: float
    final_temperature: float
    initial_log_temperature: float
    final_log_temperature: float
    initial_nll: float
    final_nll: float
    iterations: int
    valid_pixel_count: int
    ignore_index: int
    optimizer: str = "bounded_golden_section_log_temperature"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record without scientific-result claims."""
        return {
            "schema_version": "1.0",
            "record_type": "semantic_temperature_fit",
            **asdict(self),
            "objective": "multiclass_negative_log_likelihood",
            "mutates_input_logits": False,
            "scientific_evidence": False,
        }


def _prepare_valid_pixels(
    logits: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    *,
    ignore_index: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    if not isinstance(logits, np.ndarray) or logits.ndim != 4:
        raise ValueError("calibration logits must be a NumPy NCHW array")
    if any(int(dimension) <= 0 for dimension in logits.shape):
        raise ValueError("calibration logits dimensions must be positive")
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("calibration logits must use a floating dtype")
    if not bool(np.isfinite(logits).all()):
        raise ValueError("calibration logits must contain only finite values")
    if not isinstance(targets, np.ndarray) or targets.shape != (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
    ):
        raise ValueError("calibration targets must have the matching NHW shape")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("calibration targets must use an integer dtype")
    flattened_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    valid = flattened_targets != ignore_index
    if not bool(valid.any()):
        raise ValueError("calibration targets contain no valid pixels")
    valid_targets = flattened_targets[valid]
    class_count = int(logits.shape[1])
    if bool(((valid_targets < 0) | (valid_targets >= class_count)).any()):
        raise ValueError("calibration targets contain an out-of-range class ID")
    flattened_logits = np.transpose(logits, (0, 2, 3, 1)).reshape(-1, class_count)
    return np.asarray(flattened_logits[valid], dtype=np.float64), valid_targets


def _nll_at_log_temperature(
    valid_logits: npt.NDArray[np.float64],
    valid_targets: npt.NDArray[np.int64],
    log_temperature: float,
) -> float:
    temperature = math.exp(log_temperature)
    scaled = valid_logits / temperature
    maximum = np.max(scaled, axis=1, keepdims=True)
    log_partition = maximum[:, 0] + np.log(
        np.sum(np.exp(scaled - maximum), axis=1, dtype=np.float64)
    )
    correct = scaled[np.arange(valid_targets.size), valid_targets]
    result = float(np.mean(log_partition - correct, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("temperature NLL became non-finite")
    return result


def apply_temperature(logits: npt.NDArray[Any], temperature: float) -> npt.NDArray[Any]:
    """Return new logits divided by one positive scalar without mutating input."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    if not isinstance(logits, np.ndarray) or not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("temperature scaling requires a floating NumPy array")
    if logits.ndim != 4 or any(int(dimension) <= 0 for dimension in logits.shape):
        raise ValueError("temperature scaling requires positive NCHW logits")
    if not bool(np.isfinite(logits).all()):
        raise ValueError("temperature scaling requires finite logits")
    return np.asarray(logits / temperature, dtype=logits.dtype).copy()


def fit_temperature(
    logits: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    *,
    ignore_index: int = 255,
    initial_log_temperature: float = 0.0,
    log_temperature_bounds: tuple[float, float] = (-5.0, 5.0),
    max_iterations: int = 64,
    tolerance: float = 1.0e-7,
) -> TemperatureFitResult:
    """Fit one positive scalar through bounded deterministic log-temperature search."""
    if not math.isfinite(initial_log_temperature):
        raise ValueError("initial log-temperature must be finite")
    lower, upper = log_temperature_bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("log-temperature bounds must be finite and increasing")
    if not lower <= initial_log_temperature <= upper:
        raise ValueError("initial log-temperature must lie within the search bounds")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    valid_logits, valid_targets = _prepare_valid_pixels(logits, targets, ignore_index=ignore_index)
    initial_nll = _nll_at_log_temperature(valid_logits, valid_targets, initial_log_temperature)
    best_log_temperature = initial_log_temperature
    best_nll = initial_nll
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower, upper
    candidate_left = right - ratio * (right - left)
    candidate_right = left + ratio * (right - left)
    nll_left = _nll_at_log_temperature(valid_logits, valid_targets, candidate_left)
    nll_right = _nll_at_log_temperature(valid_logits, valid_targets, candidate_right)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        for candidate, nll in (
            (candidate_left, nll_left),
            (candidate_right, nll_right),
        ):
            if nll < best_nll:
                best_log_temperature, best_nll = candidate, nll
        if right - left <= tolerance:
            break
        if nll_left <= nll_right:
            right = candidate_right
            candidate_right, nll_right = candidate_left, nll_left
            candidate_left = right - ratio * (right - left)
            nll_left = _nll_at_log_temperature(valid_logits, valid_targets, candidate_left)
        else:
            left = candidate_left
            candidate_left, nll_left = candidate_right, nll_right
            candidate_right = left + ratio * (right - left)
            nll_right = _nll_at_log_temperature(valid_logits, valid_targets, candidate_right)
    return TemperatureFitResult(
        initial_temperature=math.exp(initial_log_temperature),
        final_temperature=math.exp(best_log_temperature),
        initial_log_temperature=initial_log_temperature,
        final_log_temperature=best_log_temperature,
        initial_nll=initial_nll,
        final_nll=best_nll,
        iterations=iterations,
        valid_pixel_count=int(valid_targets.size),
        ignore_index=ignore_index,
    )


def _validate_bin_edges(bin_edges: Sequence[float]) -> npt.NDArray[np.float64]:
    edges = np.asarray(tuple(bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not bool(np.isfinite(edges).all()):
        raise ValueError("ECE bin edges must be a finite one-dimensional sequence")
    if edges[0] != 0.0 or edges[-1] != 1.0 or not bool(np.all(np.diff(edges) > 0.0)):
        raise ValueError("ECE bin edges must increase strictly from 0.0 to 1.0")
    return edges


def calibration_metrics(
    logits: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    *,
    temperature: float,
    bin_edges: Sequence[float],
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Compute semantic NLL, ECE, Brier score, and reliability-bin data."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    valid_logits, valid_targets = _prepare_valid_pixels(logits, targets, ignore_index=ignore_index)
    edges = _validate_bin_edges(bin_edges)
    scaled = np.asarray(valid_logits / temperature, dtype=np.float64)
    probabilities = np.asarray(
        stable_softmax(scaled[:, :, None, None])[:, :, 0, 0], dtype=np.float64
    )
    correct_probabilities = probabilities[np.arange(valid_targets.size), valid_targets]
    nll = float(-np.mean(np.log(np.clip(correct_probabilities, np.finfo(np.float64).tiny, 1.0))))
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = predictions == valid_targets
    brier_per_pixel = (
        np.sum(probabilities * probabilities, axis=1) - 2.0 * correct_probabilities + 1.0
    )
    brier = float(np.mean(brier_per_pixel, dtype=np.float64))
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == edges.size - 2:
            members = (confidence >= left) & (confidence <= right)
        else:
            members = (confidence >= left) & (confidence < right)
        support = int(np.count_nonzero(members))
        mean_confidence = float(np.mean(confidence[members])) if support else None
        accuracy = float(np.mean(correctness[members])) if support else None
        if support:
            assert mean_confidence is not None and accuracy is not None
            ece += support / valid_targets.size * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "bin_index": index,
                "left": float(left),
                "right": float(right),
                "right_inclusive": index == edges.size - 2,
                "support": support,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "schema_version": "1.0",
        "record_type": "semantic_calibration_metrics",
        "temperature": float(temperature),
        "ignore_index": ignore_index,
        "valid_pixel_count": int(valid_targets.size),
        "nll": nll,
        "ece": float(ece),
        "brier_score": brier,
        "bin_edges": [float(edge) for edge in edges],
        "reliability_diagram": reliability,
        "probability_interpretation": "semantic_class_confidence_only",
        "calibrates_raw_ood_scores": False,
        "scientific_evidence": False,
    }
