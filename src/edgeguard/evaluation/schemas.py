"""Task-specific analysis records that remain valid before real measurements exist."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def detection_evaluation_record(
    *,
    map_value: float | None,
    ap50: float | None,
    ap75: float | None,
    per_class_ap: Mapping[str, float | None],
    size_buckets: Mapping[str, float | None],
    box_error_counts: Mapping[str, int],
    scientific_evidence: bool,
) -> dict[str, Any]:
    """Build a detection result schema without inventing unavailable metrics."""
    values = [map_value, ap50, ap75, *per_class_ap.values(), *size_buckets.values()]
    invalid = any(
        value is not None and (not np.isfinite(value) or not 0 <= value <= 1) for value in values
    )
    if invalid:
        raise ValueError("detection metrics must be null or finite values in [0, 1]")
    if any(value < 0 for value in box_error_counts.values()):
        raise ValueError("box error counts must be non-negative")
    return {
        "schema_version": "1.0",
        "record_type": "detection_evaluation",
        "mAP": map_value,
        "AP50": ap50,
        "AP75": ap75,
        "per_class_AP": dict(sorted(per_class_ap.items())),
        "size_buckets": dict(sorted(size_buckets.items())),
        "box_error_counts": dict(sorted(box_error_counts.items())),
        "scientific_evidence": scientific_evidence,
    }


def aggregate_seed_metrics(records: Sequence[Mapping[str, float]], metric: str) -> dict[str, Any]:
    """Aggregate explicit seed records with no significance claim."""
    if len(records) < 2 or any(metric not in record or "seed" not in record for record in records):
        raise ValueError("seed aggregation requires two or more explicit seed/metric records")
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed aggregation cannot contain duplicate seed identities")
    values = np.asarray([record[metric] for record in records], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("seed metrics must be finite")
    return {
        "metric": metric,
        "seeds": seeds,
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "significance_claim": False,
    }


def temporal_evaluation_record(
    *,
    persistence_gain: float,
    transient_suppression: float,
    track_continuity: float,
    missed_frame_recovery: float,
    scientific_evidence: bool,
) -> dict[str, Any]:
    """Build bounded temporal metrics with their interpretation status explicit."""
    values = (persistence_gain, transient_suppression, track_continuity, missed_frame_recovery)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("temporal metrics must be finite")
    return {
        "persistence_gain": persistence_gain,
        "transient_suppression": transient_suppression,
        "track_continuity": track_continuity,
        "missed_frame_recovery": missed_frame_recovery,
        "scientific_evidence": scientific_evidence,
    }


def efficiency_record(
    *,
    throughput: float,
    batch_latency_seconds: float,
    per_frame_latency_seconds: float,
    peak_memory_bytes: int,
    model_size_bytes: int,
    preprocessing_seconds: float,
    postprocessing_seconds: float,
    includes_ui: bool = False,
) -> dict[str, Any]:
    """Validate a complete efficiency schema and exclude dashboard time."""
    numeric = (
        throughput,
        batch_latency_seconds,
        per_frame_latency_seconds,
        preprocessing_seconds,
        postprocessing_seconds,
    )
    if any(not np.isfinite(value) or value < 0 for value in numeric):
        raise ValueError("efficiency values must be finite and non-negative")
    if peak_memory_bytes < 0 or model_size_bytes < 0 or includes_ui:
        raise ValueError("efficiency byte counts must be non-negative and UI time excluded")
    return {
        "throughput": throughput,
        "batch_latency_seconds": batch_latency_seconds,
        "per_frame_latency_seconds": per_frame_latency_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "model_size_bytes": model_size_bytes,
        "preprocessing_seconds": preprocessing_seconds,
        "postprocessing_seconds": postprocessing_seconds,
        "includes_ui": False,
    }
