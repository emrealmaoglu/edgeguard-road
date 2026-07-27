"""Minimal PIDNet-S Cityscapes-val evaluation runner."""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

from edgeguard.config import config_sha256, load_pidnet_eval_config
from edgeguard.contracts import ArtifactManifest
from edgeguard.data.cityscapes import (
    CityscapesValSample,
    build_cityscapes_val_manifest,
    discover_cityscapes_val,
    load_cityscapes_val_sample,
    resize_train_ids,
    select_city_round_robin,
)
from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.models.pidnet_spike import (
    PIDNetSpikeError,
    infer_pidnet,
    load_pidnet_session,
    preprocess_pidnet_rgb,
)
from edgeguard.provenance import detect_git_provenance, experiment_fingerprint
from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
    msp_anomaly_score,
    predictive_entropy,
    semantic_mask,
)
from edgeguard.serialization import (
    canonical_json,
    sha256_array,
    sha256_file,
    sha256_payload,
)

DeviceName = Literal["auto", "cpu", "mps", "cuda"]


@dataclass
class _NumericSummary:
    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total: float = 0.0

    def update(self, values: npt.NDArray[np.float32]) -> None:
        if not np.isfinite(values).all():
            raise ValueError("score summary received non-finite values")
        self.count += int(values.size)
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.total += float(np.sum(values, dtype=np.float64))

    def result(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
            "mean": self.total / self.count if self.count else None,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ValueError("output directory is non-empty; collision refused")
    else:
        output_dir.mkdir(parents=True)


def _load_verified_dataset_manifest(root: Path) -> dict[str, Any]:
    computed = build_cityscapes_val_manifest(root)
    recorded_path = root / "dataset_manifest.json"
    if not recorded_path.is_file():
        return computed
    try:
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cityscapes dataset manifest is invalid") from error
    if not isinstance(recorded, dict):
        raise ValueError("Cityscapes dataset manifest must be a JSON object")
    for key, value in computed.items():
        if key != "manifest_sha256" and recorded.get(key) != value:
            raise ValueError(f"Cityscapes dataset manifest mismatch for {key}")
    recorded_hash = recorded.get("manifest_sha256")
    payload = {key: value for key, value in recorded.items() if key != "manifest_sha256"}
    if recorded_hash != sha256_payload(payload):
        raise ValueError("Cityscapes dataset manifest SHA-256 is invalid")
    return recorded


def _selection_payload(
    samples: list[CityscapesValSample],
    *,
    strategy: str,
    config_sha256_value: str,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_selection_manifest",
        "strategy": strategy,
        "config_sha256": config_sha256_value,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "sample_count": len(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "city": sample.city,
                "sequence": sample.sequence,
                "frame": sample.frame,
                "image_relative_path": sample.image_relative_path,
                "label_relative_path": sample.label_relative_path,
            }
            for sample in samples
        ],
    }
    payload["selection_sha256"] = sha256_payload(payload)
    return payload


def _select_samples(
    all_samples: list[CityscapesValSample],
    *,
    subset_size: int | None,
    subset_manifest_path: Path | None,
    select_all: bool,
    strategy: str,
    config_sha256_value: str,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[list[CityscapesValSample], dict[str, Any]]:
    choices = sum((subset_size is not None, subset_manifest_path is not None, select_all))
    if choices != 1:
        raise ValueError("choose exactly one of subset size, subset manifest, or all")
    if subset_size is not None:
        selected = select_city_round_robin(all_samples, subset_size)
        recorded_strategy = strategy
    elif select_all:
        selected = sorted(all_samples, key=lambda sample: sample.sample_id)
        recorded_strategy = "all_sorted_v1"
    else:
        assert subset_manifest_path is not None
        try:
            requested = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("subset manifest is missing or invalid") from error
        expected_identity = {
            "config_sha256": config_sha256_value,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
        if any(requested.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("subset manifest identity does not match this evaluation")
        requested_ids = [item.get("sample_id") for item in requested.get("samples", [])]
        by_id = {sample.sample_id: sample for sample in all_samples}
        if not requested_ids or len(set(requested_ids)) != len(requested_ids):
            raise ValueError("subset manifest sample IDs must be non-empty and unique")
        try:
            selected = [by_id[sample_id] for sample_id in requested_ids]
        except KeyError as error:
            raise ValueError(f"subset manifest contains unknown sample: {error.args[0]}") from error
        recorded_strategy = "subset_manifest_preserved_v1"
    return selected, _selection_payload(
        selected,
        strategy=recorded_strategy,
        config_sha256_value=config_sha256_value,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def _select_visual_sample_ids(selected: list[CityscapesValSample], visual_count: int) -> set[str]:
    """Choose deterministic cross-city visuals without changing evaluation order."""
    if visual_count < 0:
        raise ValueError("visual count must be non-negative")
    if visual_count == 0 or not selected:
        return set()
    count = min(visual_count, len(selected))
    return {sample.sample_id for sample in select_city_round_robin(selected, count)}


def _visualize_map(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - np.float32(minimum)) / np.float32(maximum - minimum)
    return np.round(normalized * np.float32(255.0)).astype(np.uint8)


def _write_visuals(
    visual_dir: Path,
    sample: CityscapesValSample,
    image: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8],
    prediction: npt.NDArray[np.int64],
    scores: dict[str, npt.NDArray[np.float32]],
    *,
    height: int,
    width: int,
) -> None:
    sample_dir = visual_dir / sample.sample_id
    sample_dir.mkdir(parents=True)
    resized_image = Image.fromarray(image, mode="RGB").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    resized_image.save(sample_dir / "image.png")
    Image.fromarray(target, mode="L").save(sample_dir / "target_train_ids.png")
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(
        sample_dir / "semantic_prediction.png"
    )
    for name, score in scores.items():
        Image.fromarray(_visualize_map(score), mode="L").save(sample_dir / f"{name}.png")


def _environment_payload(session: Any) -> dict[str, Any]:
    mps_backend = getattr(session.torch.backends, "mps", None)
    return {
        "schema_version": "1.0",
        "record_type": "evaluation_environment",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "torch_version": session.torch_version,
        "selected_device": session.device,
        "cuda_available": bool(session.torch.cuda.is_available()),
        "mps_available": bool(mps_backend is not None and mps_backend.is_available()),
    }


def run_cityscapes_evaluation(
    *,
    config_path: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    upstream_checkout: Path,
    subset_size: int | None,
    subset_manifest_path: Path | None,
    select_all: bool,
    device: DeviceName,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the minimal path-free Cityscapes val evaluation pipeline."""
    _prepare_output_dir(output_dir)
    if not dataset_root.is_dir():
        raise ValueError("Cityscapes dataset root does not exist")
    config = load_pidnet_eval_config(config_path)
    config_digest = config_sha256(config)
    dataset_manifest = _load_verified_dataset_manifest(dataset_root)
    all_samples = discover_cityscapes_val(dataset_root)
    selected, selection_manifest = _select_samples(
        all_samples,
        subset_size=subset_size,
        subset_manifest_path=subset_manifest_path,
        select_all=select_all,
        strategy=config.selection.strategy,
        config_sha256_value=config_digest,
        checkpoint_sha256=config.checkpoint.sha256,
        dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
    )

    git = detect_git_provenance(config_path.parent)
    fingerprint = experiment_fingerprint(
        config_sha256=config_digest,
        contract_version=config.contract_version,
        pipeline_name=config.pipeline_name,
        backend=config.model.backend,
        scorer="+".join(config.scorers),
        git=git,
        dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
        model_artifact_sha256=config.checkpoint.sha256,
    )
    run_id = str(uuid.uuid4())
    load_started = time.perf_counter()
    session = load_pidnet_session(
        checkout=upstream_checkout,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=config.checkpoint.sha256,
        config=config,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started
    if session.device == "cuda":
        session.torch.cuda.reset_peak_memory_stats()

    metrics = SemanticConfusionMatrix(config.model.num_classes, ignore_index=255)
    summaries: dict[str, _NumericSummary] = {name: _NumericSummary() for name in config.scorers}
    sample_timings = _NumericSummary()
    failures: list[dict[str, str]] = []
    first_output: dict[str, Any] | None = None
    evaluation_started = time.perf_counter()
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir()
    visual_sample_ids = _select_visual_sample_ids(selected, config.visual_count)

    for sample in selected:
        sample_started = time.perf_counter()
        try:
            image, target = load_cityscapes_val_sample(dataset_root, sample)
            model_input = preprocess_pidnet_rgb(
                image,
                height=config.input.height,
                width=config.input.width,
                pixel_scale=config.preprocess.pixel_scale,
                mean=config.preprocess.mean,
                std=config.preprocess.std,
            )
            inference = infer_pidnet(
                session,
                model_input,
                alignment_height=config.input.height,
                alignment_width=config.input.width,
                alignment_mode=config.alignment.mode,
                align_corners=config.alignment.align_corners,
            )
            prediction = semantic_mask(inference.aligned_logits)[0]
            metric_target = (
                target
                if target.shape == (config.input.height, config.input.width)
                else resize_train_ids(target, height=config.input.height, width=config.input.width)
            )
            if config.metric_grid == "source_label" and metric_target.shape != target.shape:
                raise ValueError("source-label metric grid requires full source resolution")
            metrics.update(prediction, metric_target)
            scores = {
                "msp": msp_anomaly_score(inference.aligned_logits)[0],
                "predictive_entropy": predictive_entropy(inference.aligned_logits)[0],
                "max_logit": max_logit_anomaly_score(inference.aligned_logits)[0],
                "energy": energy_anomaly_score(
                    inference.aligned_logits, temperature=config.energy_temperature
                )[0],
            }
            for name, score in scores.items():
                summaries[name].update(score)
            if first_output is None:
                first_output = {
                    "sample_id": sample.sample_id,
                    "native_logits_shape": list(inference.native_logits.shape),
                    "native_logits_sha256": sha256_array(inference.native_logits),
                    "aligned_logits_shape": list(inference.aligned_logits.shape),
                    "aligned_logits_sha256": sha256_array(inference.aligned_logits),
                }
            if sample.sample_id in visual_sample_ids:
                _write_visuals(
                    visual_dir,
                    sample,
                    image,
                    metric_target,
                    prediction,
                    scores,
                    height=config.input.height,
                    width=config.input.width,
                )
        except (OSError, ValueError, PIDNetSpikeError) as error:
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "error_type": type(error).__name__,
                    "error": " ".join(str(error).split())[:300],
                }
            )
        finally:
            sample_timings.update(
                np.array([time.perf_counter() - sample_started], dtype=np.float32)
            )

    evaluation_seconds = time.perf_counter() - evaluation_started
    success_count = len(selected) - len(failures)
    environment = _environment_payload(session)
    created_at = datetime.now(timezone.utc)
    run_metadata = {
        "schema_version": "1.0",
        "record_type": "cityscapes_eval_run_metadata",
        "evaluation_name": "EdgeGuard-Road single-scale PIDNet-S Cityscapes-val evaluation",
        "claim_scope": "id_common_evaluation_and_score_summary_only",
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "command": ["python", "scripts/run_cityscapes_eval.py"],
        "config_sha256": config_digest,
        "experiment_fingerprint": fingerprint,
        "git_commit": git.commit,
        "git_state": git.state.value,
        "git_dirty": git.dirty,
        "checkpoint_sha256": config.checkpoint.sha256,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "selection_manifest_sha256": selection_manifest["selection_sha256"],
        "selected_device": session.device,
        "torch_version": session.torch_version,
        "model_input_shape": [
            config.input.batch_size,
            config.input.channels,
            config.input.height,
            config.input.width,
        ],
        "metric_grid": config.metric_grid,
        "native_logits_kind": "direct_model_output_raw_logits",
        "aligned_logits_kind": "bilinear_derivative_raw_logits",
        "semantic_mask_logits_kind": "aligned_logits",
        "msp_logits_kind": "aligned_logits",
        "entropy_logits_kind": "aligned_logits",
        "max_logit_logits_kind": "aligned_logits",
        "energy_logits_kind": "aligned_logits",
        "energy_temperature": config.energy_temperature,
        "score_direction": "higher_means_more_anomalous",
        "score_claim": "not_anomaly_probability",
        "selected_sample_count": len(selected),
        "successful_sample_count": success_count,
        "failed_sample_count": len(failures),
        "first_successful_output": first_output,
        "timing": {
            "model_load_seconds": load_seconds,
            "evaluation_seconds": evaluation_seconds,
            "per_sample_seconds": sample_timings.result(),
        },
        "peak_cuda_memory_bytes": (
            int(session.torch.cuda.max_memory_allocated()) if session.device == "cuda" else None
        ),
    }
    uncertainty_summary = {
        "schema_version": "1.0",
        "record_type": "uncertainty_summary",
        "logits_kind": "aligned_logits",
        "direction": "higher_means_more_anomalous",
        "claim": "not_anomaly_probability",
        "scores": {name: summaries[name].result() for name in config.scorers},
    }

    _write_json(output_dir / "run_metadata.json", run_metadata)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "selection_manifest.json", selection_manifest)
    _write_json(output_dir / "semantic_metrics.json", metrics.result())
    _write_json(output_dir / "uncertainty_summary.json", uncertainty_summary)
    failures_path = output_dir / "failures.jsonl"
    failures_path.write_text(
        "".join(canonical_json(failure) + "\n" for failure in failures),
        encoding="utf-8",
    )

    artifact_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    file_hashes = {
        path.relative_to(output_dir).as_posix(): sha256_file(path) for path in artifact_files
    }
    artifact_manifest = ArtifactManifest(
        artifact_type="cityscapes_eval_bundle",
        artifact_name="edgeguard-cityscapes-eval",
        sha256=sha256_payload(file_hashes),
        model_artifact_sha256=config.checkpoint.sha256,
        git_commit=git.commit,
        git_state=git.state,
        git_dirty=git.dirty,
        config_sha256=config_digest,
        experiment_fingerprint=fingerprint,
        dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
        source_run_id=run_id,
        model_source=(f"{config.upstream.repository_url}@{config.upstream.commit}"),
        input_shape=[
            config.input.batch_size,
            config.input.channels,
            config.input.height,
            config.input.width,
        ],
        precision="float32",
        environment=environment,
        files=file_hashes,
        created_at=created_at,
        created_by="edgeguard-road",
        notes=" ".join(
            (
                "Single-scale Cityscapes-val common evaluation;",
                "no sealed, OOD, or probability claim.",
            )
        ),
    )
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {
        "status": "ok" if not failures else "completed_with_failures",
        "run_id": run_id,
        "selected_sample_count": len(selected),
        "successful_sample_count": success_count,
        "failed_sample_count": len(failures),
        "semantic_metrics_file": "semantic_metrics.json",
        "uncertainty_summary_file": "uncertainty_summary.json",
        "artifact_manifest_file": "artifact_manifest.json",
    }
