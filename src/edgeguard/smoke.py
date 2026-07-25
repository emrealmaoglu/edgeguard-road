"""Deterministic synthetic vertical slice for repository verification."""

from __future__ import annotations

import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from edgeguard.config import SmokeConfig, config_sha256
from edgeguard.contracts import (
    RunMetadata,
    SmokeResult,
    SmokeScientificPayload,
    validate_anomaly_map,
    validate_model_input,
    validate_pipeline_shapes,
    validate_raw_rgb,
    validate_semantic_logits,
)
from edgeguard.provenance import detect_git_provenance, experiment_fingerprint
from edgeguard.serialization import canonical_json

DETERMINISTIC_RUN_ID = "00000000-0000-0000-0000-000000000000"
DETERMINISTIC_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


class DummyModelBackend:
    """CPU-only deterministic backend that emits synthetic raw logits."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def infer(self, model_input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Create class-specific affine views of the input mean."""
        validate_model_input(model_input)
        base = np.mean(model_input, axis=1, dtype=np.float32)
        channels = []
        for class_index in range(self.num_classes):
            scale = np.float32((class_index + 1) / self.num_classes)
            offset = np.float32((class_index - self.num_classes / 2) * 0.125)
            channels.append(base * scale + offset)
        logits = np.stack(channels, axis=1).astype(np.float32, copy=False)
        return validate_semantic_logits(logits)


class DummyAnomalyScorer:
    """Non-scientific normalized magnitude scorer for smoke testing only."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = np.float32(epsilon)

    def score(self, logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return a finite NHW map normalized independently per sample."""
        validate_semantic_logits(logits)
        magnitude = np.mean(np.abs(logits), axis=1, dtype=np.float32)
        minimum = np.min(magnitude, axis=(1, 2), keepdims=True)
        maximum = np.max(magnitude, axis=(1, 2), keepdims=True)
        denominator = maximum - minimum
        scores = np.divide(
            magnitude - minimum,
            denominator,
            out=np.zeros_like(magnitude, dtype=np.float32),
            where=denominator > self.epsilon,
        ).astype(np.float32, copy=False)
        return validate_anomaly_map(scores)


def _raw_image(config: SmokeConfig) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(config.seed)
    shape = (config.input.height, config.input.width, config.input.channels)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return validate_raw_rgb(image)


def _preprocess(raw_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    model_input = np.transpose(raw_image, (2, 0, 1))[None, ...].astype(np.float32)
    model_input /= np.float32(255.0)
    return validate_model_input(model_input)


def _stable_float(value: float | np.floating[Any]) -> float:
    return float(f"{float(value):.8f}")


def build_smoke_result(
    config: SmokeConfig,
    *,
    config_path: Path,
    deterministic: bool,
) -> SmokeResult:
    """Execute the in-memory dummy pipeline and build a validated record."""
    raw_image = _raw_image(config)
    model_input = _preprocess(raw_image)
    logits = DummyModelBackend(config.model.num_classes).infer(model_input)
    anomaly_map = DummyAnomalyScorer(config.scorer.epsilon).score(logits)
    validate_pipeline_shapes(model_input, logits, anomaly_map)

    config_digest = config_sha256(config)
    git = detect_git_provenance(config_path.resolve().parent)
    fingerprint = experiment_fingerprint(
        config_sha256=config_digest,
        contract_version=config.contract_version,
        pipeline_name=config.pipeline_name,
        backend=config.model.backend,
        scorer=config.scorer.name,
        git=git,
    )
    if deterministic:
        run_id = DETERMINISTIC_RUN_ID
        created_at = DETERMINISTIC_TIME
        hostname = "deterministic-smoke"
        command = ["edgeguard", "smoke", "--deterministic"]
        execution_mode: Literal["normal", "deterministic_smoke"] = "deterministic_smoke"
    else:
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        hostname = socket.gethostname() or "unknown-host"
        command = ["edgeguard", "smoke"]
        execution_mode = "normal"

    metadata = RunMetadata(
        run_id=run_id,
        created_at=created_at,
        hostname=hostname,
        command=command,
        git_commit=git.commit,
        git_state=git.state,
        git_dirty=git.dirty,
        config_sha256=config_digest,
        experiment_fingerprint=fingerprint,
        execution_mode=execution_mode,
    )
    payload = SmokeScientificPayload(
        seed=config.seed,
        backend=config.model.backend,
        scorer=config.scorer.name,
        raw_shape=(
            int(raw_image.shape[0]),
            int(raw_image.shape[1]),
            int(raw_image.shape[2]),
        ),
        model_input_shape=(
            int(model_input.shape[0]),
            int(model_input.shape[1]),
            int(model_input.shape[2]),
            int(model_input.shape[3]),
        ),
        logits_shape=(
            int(logits.shape[0]),
            int(logits.shape[1]),
            int(logits.shape[2]),
            int(logits.shape[3]),
        ),
        anomaly_shape=(
            int(anomaly_map.shape[0]),
            int(anomaly_map.shape[1]),
            int(anomaly_map.shape[2]),
        ),
        raw_dtype="uint8",
        model_input_dtype="float32",
        logits_dtype="float32",
        anomaly_dtype="float32",
        anomaly_min=_stable_float(np.min(anomaly_map)),
        anomaly_max=_stable_float(np.max(anomaly_map)),
        anomaly_mean=_stable_float(np.mean(anomaly_map, dtype=np.float32)),
    )
    return SmokeResult(metadata=metadata, scientific_payload=payload)


def write_smoke_result(result: SmokeResult, output_path: Path) -> None:
    """Write one canonical JSON record followed by a newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
