"""Equal-domain global calibration and confidence/entropy summaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.calibration import apply_temperature, calibration_metrics, fit_temperature
from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.scoring.uncertainty import stable_softmax
from edgeguard.serialization import canonical_json, sha256_file


def confidence_entropy_summary(logits: np.ndarray, *, temperature: float = 1.0) -> dict[str, float]:
    """Summarize maximum-softmax confidence and normalized predictive entropy."""
    scaled = apply_temperature(logits, temperature)
    probabilities = stable_softmax(scaled)
    confidence = np.max(probabilities, axis=1)
    safe = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1) / np.log(probabilities.shape[1])
    return {
        "mean_max_softmax_confidence": float(np.mean(confidence)),
        "mean_normalized_entropy": float(np.mean(entropy)),
    }


def save_calibration_evidence(
    path: Path,
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    dataset_id: str,
    dataset_manifest_sha256: str | None,
    checkpoint_sha256: str,
) -> None:
    """Persist sampled source-calibration pixels with immutable provenance."""
    flat_logits = logits.reshape(logits.shape[1], -1)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != 255
    if not bool(valid.any()):
        raise ValueError("calibration evidence contains no valid semantic pixels")
    logits = flat_logits[:, valid][None, :, None, :]
    targets = flat_targets[valid][None, None, :]
    metadata = {
        "schema_version": "2.0",
        "record_type": "source_calibration_evidence",
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "sampled_pixel_count": int(targets.size),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=logits.astype(np.float32),
        targets=targets.astype(np.int64),
        metadata_json=np.asarray(canonical_json(metadata)),
    )


def fit_global_temperature_from_evidence(
    evidence_paths: Sequence[Path], output_path: Path, *, seed: int
) -> dict[str, Any]:
    """Fit one scalar from equal pixel counts across three source domains."""
    if len(evidence_paths) != 3:
        raise ValueError("global calibration requires exactly three source-domain evidence files")
    loaded: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    checkpoint_hashes: set[str] = set()
    dataset_ids: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    for path in evidence_paths:
        with np.load(path, allow_pickle=False) as archive:
            logits = np.asarray(archive["logits"], dtype=np.float32)
            targets = np.asarray(archive["targets"], dtype=np.int64)
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("record_type") != "source_calibration_evidence":
            raise ValueError("invalid calibration evidence record")
        checkpoint_hashes.add(str(metadata["checkpoint_sha256"]))
        dataset_id = str(metadata["dataset_id"])
        manifest_sha256 = metadata.get("dataset_manifest_sha256")
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ValueError("calibration evidence is not bound to a dataset manifest")
        if dataset_id in manifest_hashes:
            raise ValueError("global calibration repeats a source domain")
        manifest_hashes[dataset_id] = manifest_sha256
        dataset_ids.add(dataset_id)
        loaded.append((logits, targets, metadata))
    if dataset_ids != {"cityscapes", "bdd100k", "idd20k"}:
        raise ValueError("global calibration evidence must cover the three source domains")
    if len(checkpoint_hashes) != 1:
        raise ValueError("global calibration evidence must come from one frozen checkpoint")
    equal_count = min(int(targets.size) for _, targets, _ in loaded)
    if equal_count <= 0:
        raise ValueError("calibration evidence contains no pixels")
    rng = np.random.default_rng(seed)
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for logits, targets, _ in sorted(loaded, key=lambda item: item[2]["dataset_id"]):
        flat_logits = logits.reshape(logits.shape[1], -1)
        flat_targets = targets.reshape(-1)
        indices = rng.choice(flat_targets.size, size=equal_count, replace=False)
        logits_parts.append(flat_logits[:, indices])
        target_parts.append(flat_targets[indices])
    combined_logits = np.concatenate(logits_parts, axis=1)[None, :, None, :]
    combined_targets = np.concatenate(target_parts, axis=0)[None, None, :]
    fit = fit_temperature(combined_logits, combined_targets)
    bins = [float(value) for value in np.linspace(0.0, 1.0, 16)]
    result = {
        **fit.to_dict(),
        "schema_version": "2.0",
        "record_type": "multi_domain_global_temperature",
        "source_domains": sorted(dataset_ids),
        "equal_pixels_per_domain": equal_count,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "dataset_manifest_sha256s": manifest_hashes,
        "evidence_sha256s": sorted(sha256_file(path) for path in evidence_paths),
        "before": calibration_metrics(
            combined_logits, combined_targets, temperature=1.0, bin_edges=bins
        ),
        "after": calibration_metrics(
            combined_logits,
            combined_targets,
            temperature=fit.final_temperature,
            bin_edges=bins,
        ),
        "confidence_entropy_before": confidence_entropy_summary(combined_logits),
        "confidence_entropy_after": confidence_entropy_summary(
            combined_logits, temperature=fit.final_temperature
        ),
        "scientific_evidence": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    append_run_ledger(
        output_path.parent / "run_ledger.jsonl",
        operation="global_temperature_calibration",
        result=result,
    )
    return result
