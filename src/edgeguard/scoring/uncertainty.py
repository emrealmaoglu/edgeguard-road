"""Backend-neutral semantic uncertainty scores from unnormalized NCHW logits."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np

Backend = Literal["numpy", "torch"]


def _backend(logits: Any) -> Backend:
    if isinstance(logits, np.ndarray):
        return "numpy"
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise TypeError("logits must be a NumPy array or Torch tensor") from error
    if bool(torch.is_tensor(logits)):
        return "torch"
    raise TypeError("logits must be a NumPy array or Torch tensor")


def _validate_logits(logits: Any) -> Backend:
    backend = _backend(logits)
    if int(logits.ndim) != 4:
        raise ValueError(f"semantic logits must have NCHW rank 4, got rank {logits.ndim}")
    shape = tuple(int(dimension) for dimension in logits.shape)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"semantic logits dimensions must be positive, got {shape}")
    if backend == "numpy":
        if not np.issubdtype(logits.dtype, np.floating):
            raise TypeError(f"semantic logits must use a floating dtype, got {logits.dtype}")
        if not bool(np.isfinite(logits).all()):
            raise ValueError("semantic logits must contain only finite values")
    else:
        if not bool(logits.is_floating_point()):
            raise TypeError(f"semantic logits must use a floating dtype, got {logits.dtype}")
        torch = importlib.import_module("torch")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("semantic logits must contain only finite values")
    return backend


def _working_logits(logits: Any, backend: Backend) -> Any:
    if backend == "numpy":
        dtype = np.float64 if logits.dtype == np.float64 else np.float32
        return np.asarray(logits, dtype=dtype)
    torch = importlib.import_module("torch")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    return logits.to(dtype=dtype)


def _validate_score(score: Any, backend: Backend) -> Any:
    shape = tuple(int(dimension) for dimension in score.shape)
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"uncertainty score must have positive NHW shape, got {shape}")
    if backend == "numpy":
        if not bool(np.isfinite(score).all()):
            raise ValueError("uncertainty score contains a non-finite value")
    else:
        torch = importlib.import_module("torch")
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError("uncertainty score contains a non-finite value")
    return score


def uncertainty_score_contract() -> dict[str, Any]:
    """Return explicit semantics without treating raw OOD scores as probabilities."""
    return {
        "schema_version": "1.0",
        "record_type": "semantic_uncertainty_scoring_contract",
        "input": "unnormalized_nchw_floating_logits",
        "output_layout": "NHW",
        "common_direction": "higher_means_more_anomalous",
        "calibrated_anomaly_probability": False,
        "methods": {
            "msp": {
                "definition": "1-max(softmax(logits))",
                "score_kind": "bounded_uncertainty_score",
                "range": [0.0, 1.0],
                "shift_invariant": True,
                "anomaly_probability": False,
            },
            "predictive_entropy": {
                "definition": "-sum(p*log(p))",
                "score_kind": "bounded_by_log_class_count",
                "shift_invariant": True,
                "anomaly_probability": False,
            },
            "max_logit": {
                "definition": "-max(logits)",
                "score_kind": "unbounded_raw_ood_score",
                "shift_invariant": False,
                "anomaly_probability": False,
            },
            "energy": {
                "definition": "-T*logsumexp(logits/T)",
                "score_kind": "unbounded_temperature_explicit_raw_ood_score",
                "shift_invariant": False,
                "anomaly_probability": False,
            },
        },
        "normalization_across_methods": "none",
        "scientific_evidence": False,
    }


def stable_softmax(logits: Any) -> Any:
    """Compute max-shifted class probabilities while preserving N/H/W."""
    backend = _validate_logits(logits)
    work = _working_logits(logits, backend)
    if backend == "numpy":
        shifted = work - np.max(work, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        denominator = np.sum(exponentials, axis=1, keepdims=True)
        probabilities = exponentials / denominator
        if not bool(np.isfinite(probabilities).all()):
            raise ValueError("softmax produced non-finite probabilities")
        return probabilities
    torch = importlib.import_module("torch")
    probabilities = torch.softmax(work, dim=1)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("softmax produced non-finite probabilities")
    return probabilities


def semantic_mask(logits: Any) -> Any:
    """Return the class argmax on the provided logits grid."""
    backend = _validate_logits(logits)
    if backend == "numpy":
        return np.argmax(logits, axis=1).astype(np.int64, copy=False)
    return logits.argmax(dim=1)


def msp_anomaly_score(logits: Any) -> Any:
    """Return 1 - maximum softmax probability; higher means more anomalous."""
    backend = _validate_logits(logits)
    probabilities = stable_softmax(logits)
    if backend == "numpy":
        scores = 1.0 - np.max(probabilities, axis=1)
    else:
        scores = 1.0 - probabilities.max(dim=1).values
    return _validate_score(scores, backend)


def predictive_entropy(logits: Any) -> Any:
    """Return natural-log predictive entropy; it is not anomaly probability."""
    backend = _validate_logits(logits)
    probabilities = stable_softmax(logits)
    if backend == "numpy":
        clipped = np.clip(probabilities, np.finfo(probabilities.dtype).tiny, 1.0)
        scores = -np.sum(probabilities * np.log(clipped), axis=1)
    else:
        torch = importlib.import_module("torch")
        clipped = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        scores = -(probabilities * clipped.log()).sum(dim=1)
    return _validate_score(scores, backend)


def max_logit_anomaly_score(logits: Any) -> Any:
    """Return negative maximum raw logit; higher means more anomalous."""
    backend = _validate_logits(logits)
    work = _working_logits(logits, backend)
    if backend == "numpy":
        scores = -np.max(work, axis=1)
    else:
        scores = -work.max(dim=1).values
    return _validate_score(scores, backend)


def energy_anomaly_score(logits: Any, *, temperature: float = 1.0) -> Any:
    """Return stable ``-T*logsumexp(logits/T)``; higher means more anomalous."""
    backend = _validate_logits(logits)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("energy temperature must be a positive finite number")
    work = _working_logits(logits, backend)
    if backend == "numpy":
        scaled = work / temperature
        maximum = np.max(scaled, axis=1, keepdims=True)
        scores = -temperature * (
            maximum[:, 0] + np.log(np.sum(np.exp(scaled - maximum), axis=1, dtype=work.dtype))
        )
    else:
        torch = importlib.import_module("torch")
        scores = -temperature * torch.logsumexp(work / temperature, dim=1)
    return _validate_score(scores, backend)
