"""Stable semantic predictions and uncertainty scores from raw logits."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from edgeguard.contracts import validate_anomaly_map, validate_semantic_logits


def stable_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute class probabilities with max-shifted float32 softmax."""
    validate_semantic_logits(logits)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted).astype(np.float32, copy=False)
    denominator = np.sum(exponentials, axis=1, keepdims=True, dtype=np.float32)
    probabilities = (exponentials / denominator).astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("softmax produced non-finite probabilities")
    return probabilities


def semantic_mask(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
    """Return the class argmax on the provided logits grid."""
    validate_semantic_logits(logits)
    return np.argmax(logits, axis=1).astype(np.int64, copy=False)


def msp_anomaly_score(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return 1 - maximum softmax probability; higher means more anomalous."""
    probabilities = stable_softmax(logits)
    scores = (np.float32(1.0) - np.max(probabilities, axis=1)).astype(np.float32, copy=False)
    return validate_anomaly_map(scores)


def predictive_entropy(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return unnormalized natural-log predictive entropy per pixel."""
    probabilities = stable_softmax(logits)
    clipped = np.clip(probabilities, np.finfo(np.float32).tiny, np.float32(1.0))
    scores = -np.sum(probabilities * np.log(clipped), axis=1, dtype=np.float32)
    return validate_anomaly_map(scores.astype(np.float32, copy=False))


def max_logit_anomaly_score(
    logits: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Return negative maximum raw logit; higher means more anomalous."""
    validate_semantic_logits(logits)
    scores = -np.max(logits, axis=1)
    return validate_anomaly_map(scores.astype(np.float32, copy=False))


def energy_anomaly_score(
    logits: npt.NDArray[np.float32], *, temperature: float = 1.0
) -> npt.NDArray[np.float32]:
    """Return negative temperature-scaled log-sum-exp energy anomaly score."""
    validate_semantic_logits(logits)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("energy temperature must be a positive finite number")
    temperature32 = np.float32(temperature)
    scaled = logits / temperature32
    maximum = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - maximum
    log_sum_exp = maximum[:, 0] + np.log(np.sum(np.exp(shifted), axis=1, dtype=np.float32))
    scores = -temperature32 * log_sum_exp
    return validate_anomaly_map(scores.astype(np.float32, copy=False))
