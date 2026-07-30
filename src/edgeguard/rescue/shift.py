"""Source-calibrated frame uncertainty summaries and frozen shift alerts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.evaluation.ood import pixel_ood_metrics
from edgeguard.serialization import canonical_json, sha256_payload


def uncertainty_maps(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Return MSP, entropy, maximum-logit and energy maps from finite CHW logits."""
    if logits.ndim != 3 or logits.shape[0] != 19 or not bool(np.isfinite(logits).all()):
        raise ValueError("logits must be a finite 19-channel CHW array")
    shifted = logits.astype(np.float64) - np.max(logits, axis=0, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=0, keepdims=True)
    confidence = np.max(probabilities, axis=0)
    safe = np.clip(probabilities, 1.0e-12, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=0) / np.log(probabilities.shape[0])
    maximum_logit = np.max(logits, axis=0)
    maximum = np.max(logits, axis=0, keepdims=True)
    logsumexp = maximum[0] + np.log(np.sum(np.exp(logits - maximum), axis=0))
    energy = -logsumexp
    return {
        "maximum_softmax_probability": confidence.astype(np.float32),
        "normalized_entropy": entropy.astype(np.float32),
        "maximum_logit": maximum_logit.astype(np.float32),
        "energy": energy.astype(np.float32),
    }


def frame_uncertainty_summary(
    confidence: np.ndarray,
    entropy: np.ndarray,
    *,
    energy: np.ndarray | None = None,
    confidence_threshold: float = 0.5,
) -> dict[str, float | None]:
    """Summarize one frame without interpreting it as pixel-level anomaly truth."""
    if confidence.shape != entropy.shape or confidence.ndim != 2:
        raise ValueError("confidence and entropy maps must share two-dimensional geometry")
    if not bool(np.isfinite(confidence).all()) or not bool(np.isfinite(entropy).all()):
        raise ValueError("frame uncertainty maps must be finite")
    if energy is not None and (
        energy.shape != confidence.shape or not bool(np.isfinite(energy).all())
    ):
        raise ValueError("energy map must be finite and match confidence geometry")
    return {
        "mean_normalized_entropy": float(np.mean(entropy)),
        "low_confidence_pixel_ratio": float(np.mean(confidence < confidence_threshold)),
        "mean_energy": float(np.mean(energy)) if energy is not None else None,
    }


def fit_shift_reference(
    source_summaries: Sequence[dict[str, float | None]],
    *,
    checkpoint_sha256: str,
    dataset_manifest_sha256s: dict[str, str],
    quantile: float = 0.95,
) -> dict[str, Any]:
    """Freeze source-only alert thresholds before any external-domain observation."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("shift reference quantile must be between 0.5 and 1")
    if len(checkpoint_sha256) != 64 or set(dataset_manifest_sha256s) != {
        "cityscapes",
        "bdd100k",
        "idd20k",
    }:
        raise ValueError("shift reference requires one checkpoint and all source manifests")
    if not source_summaries:
        raise ValueError("shift reference needs source calibration frames")
    fields = (
        "mean_normalized_entropy",
        "low_confidence_pixel_ratio",
        "mean_energy",
    )
    thresholds: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for field in fields:
        values = []
        for row in source_summaries:
            value = row.get(field)
            if value is not None:
                values.append(float(value))
        if field != "mean_energy" and len(values) != len(source_summaries):
            raise ValueError(f"source shift summaries are missing {field}")
        thresholds[field] = float(np.quantile(values, quantile)) if values else None
        counts[field] = len(values)
    artifact: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_source_shift_reference",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256s": dataset_manifest_sha256s,
        "quantile": quantile,
        "thresholds": thresholds,
        "sample_counts": counts,
        "minimum_alert_votes": 2,
        "external_data_used": False,
        "pixel_anomaly_segmentation": False,
    }
    artifact["artifact_sha256"] = sha256_payload(artifact)
    return artifact


def apply_shift_reference(
    summary: dict[str, float | None], reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply immutable source quantiles with a two-of-three vote."""
    if reference.get("record_type") != "edgeguard_source_shift_reference":
        raise ValueError("invalid shift reference artifact")
    recorded_hash = reference.get("artifact_sha256")
    unhashed = {key: value for key, value in reference.items() if key != "artifact_sha256"}
    if recorded_hash != sha256_payload(unhashed):
        raise ValueError("shift reference artifact hash mismatch")
    votes: dict[str, bool | None] = {}
    thresholds = reference["thresholds"]
    for field, threshold in thresholds.items():
        value = summary.get(field)
        votes[field] = (
            None if threshold is None or value is None else float(value) > float(threshold)
        )
    available = [value for value in votes.values() if value is not None]
    minimum = int(reference.get("minimum_alert_votes", 2))
    return {
        "domain_shift_alert": sum(bool(value) for value in available) >= minimum,
        "alert_votes": sum(bool(value) for value in available),
        "available_votes": len(available),
        "minimum_alert_votes": minimum,
        "votes": votes,
        "thresholds": thresholds,
        "external_threshold_tuning": False,
    }


def frame_shift_metrics(
    source_summaries: Sequence[dict[str, float | None]],
    external_summaries: Sequence[dict[str, float | None]],
    *,
    source_alerts: Sequence[bool] | None = None,
    external_alerts: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Measure source-vs-external separation without tuning on external frames."""
    if not source_summaries or not external_summaries:
        raise ValueError("frame-shift metrics require source and external frames")
    if source_alerts is not None and len(source_alerts) != len(source_summaries):
        raise ValueError("source alert count must match source summaries")
    if external_alerts is not None and len(external_alerts) != len(external_summaries):
        raise ValueError("external alert count must match external summaries")
    scores: dict[str, dict[str, Any]] = {}
    for field in (
        "mean_normalized_entropy",
        "low_confidence_pixel_ratio",
        "negative_mean_maximum_logit",
        "mean_energy",
    ):
        source = [row.get(field) for row in source_summaries]
        external = [row.get(field) for row in external_summaries]
        if any(value is None for value in (*source, *external)):
            scores[field] = {
                "auroc": None,
                "average_precision": None,
                "reason": "score_missing",
            }
            continue
        values = np.asarray([*source, *external], dtype=np.float64)
        labels = np.asarray(
            [0] * len(source_summaries) + [1] * len(external_summaries),
            dtype=np.uint8,
        )
        metrics = pixel_ood_metrics(values, labels)
        scores[field] = {
            "auroc": metrics.auroc,
            "average_precision": metrics.average_precision,
            "source_frames": len(source_summaries),
            "external_frames": len(external_summaries),
        }
    return {
        "record_type": "edgeguard_frame_shift_metrics",
        "positive_class": "external",
        "score_direction": "higher_means_more_shifted",
        "scores": scores,
        "source_alert_rate": (float(np.mean(source_alerts)) if source_alerts is not None else None),
        "external_alert_rate": (
            float(np.mean(external_alerts)) if external_alerts is not None else None
        ),
        "external_threshold_tuning": False,
        "pixel_anomaly_segmentation": False,
    }


def save_shift_reference(path: Path, artifact: dict[str, Any]) -> None:
    """Persist a new source-only shift artifact without overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shift reference: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")


def load_shift_reference(path: Path) -> dict[str, Any]:
    """Load one hash-validated source shift artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    apply_shift_reference(
        {
            "mean_normalized_entropy": 0.0,
            "low_confidence_pixel_ratio": 0.0,
            "mean_energy": 0.0,
        },
        payload,
    )
    return payload
