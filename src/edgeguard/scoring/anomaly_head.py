"""Small exportable linear anomaly-head baseline and synthetic outlier exposure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from edgeguard.serialization import sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json


def synthetic_outlier_exposure(
    *, seed: int, sample_count: int, feature_count: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate deterministic project-owned ID/anomaly features for plumbing tests."""
    if sample_count < 4 or feature_count <= 0:
        raise ValueError("synthetic OE requires at least four samples and one feature")
    generator = np.random.default_rng(seed)
    id_count = sample_count // 2
    anomaly_count = sample_count - id_count
    features = np.concatenate(
        (
            generator.normal(-0.75, 0.3, (id_count, feature_count)),
            generator.normal(0.75, 0.3, (anomaly_count, feature_count)),
        ),
        axis=0,
    ).astype(np.float32)
    targets = np.concatenate(
        (np.zeros(id_count, dtype=np.float32), np.ones(anomaly_count, dtype=np.float32))
    )
    return features, targets


@dataclass
class LinearAnomalyHead:
    """One-logit linear candidate baseline trained with BCE."""

    weights: npt.NDArray[np.float32]
    bias: float = 0.0
    optimizer_step: int = 0

    @classmethod
    def initialized(cls, feature_count: int) -> LinearAnomalyHead:
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        return cls(np.zeros(feature_count, dtype=np.float32))

    def logits(self, features: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        if features.ndim != 2 or features.shape[1] != self.weights.size:
            raise ValueError("anomaly-head features must have matching NF shape")
        result = np.asarray(features, dtype=np.float32) @ self.weights + self.bias
        if not bool(np.isfinite(result).all()):
            raise ValueError("anomaly-head logits became non-finite")
        return np.asarray(result, dtype=np.float32)

    def train_steps(
        self,
        features: npt.NDArray[np.floating],
        targets: npt.NDArray[np.floating],
        *,
        steps: int,
        learning_rate: float,
    ) -> list[float]:
        """Run bounded full-batch BCE optimization and return finite losses."""
        if steps <= 0 or learning_rate <= 0.0:
            raise ValueError("steps and learning_rate must be positive")
        values = np.asarray(features, dtype=np.float32)
        expected = np.asarray(targets, dtype=np.float32)
        if expected.shape != (values.shape[0],) or not bool(np.isin(expected, [0.0, 1.0]).all()):
            raise ValueError("anomaly-head targets must be binary and match samples")
        losses: list[float] = []
        for _ in range(steps):
            logits = self.logits(values)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            loss = float(
                -np.mean(
                    expected * np.log(np.clip(probabilities, 1e-7, 1.0))
                    + (1.0 - expected) * np.log(np.clip(1.0 - probabilities, 1e-7, 1.0))
                )
            )
            if not np.isfinite(loss):
                raise FloatingPointError("anomaly-head loss became non-finite")
            gradient = probabilities - expected
            weight_gradient = np.asarray(values.T @ gradient / values.shape[0], dtype=np.float32)
            self.weights -= learning_rate * weight_gradient
            self.bias -= learning_rate * float(np.mean(gradient))
            self.optimizer_step += 1
            losses.append(loss)
        return losses

    def checkpoint(self, path: Path, *, identity: dict[str, str]) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "record_type": "linear_anomaly_head_checkpoint",
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "optimizer_step": self.optimizer_step,
            "identity": identity,
            "scientific_evidence": False,
        }
        payload["checkpoint_sha256"] = sha256_payload(payload)
        atomic_write_json(path, payload)
        return payload

    @classmethod
    def resume(cls, path: Path, *, identity: dict[str, str]) -> LinearAnomalyHead:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("anomaly-head checkpoint is missing or corrupt") from error
        digest = payload.pop("checkpoint_sha256", None)
        if digest != sha256_payload(payload) or payload.get("identity") != identity:
            raise ValueError("anomaly-head checkpoint hash or identity mismatch")
        weights = np.asarray(payload["weights"], dtype=np.float32)
        if weights.ndim != 1 or not bool(np.isfinite(weights).all()):
            raise ValueError("anomaly-head checkpoint weights are invalid")
        return cls(weights, float(payload["bias"]), int(payload["optimizer_step"]))
