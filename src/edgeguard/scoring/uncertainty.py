"""Stable semantic predictions and uncertainty scores from raw logits."""

from __future__ import annotations

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
