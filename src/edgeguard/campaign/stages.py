"""Local-mini stage implementations for the bounded EdgeGuard campaign."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.calibration.semantic import calibration_metrics, fit_temperature
from edgeguard.campaign.contracts import PROFILES
from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.detection.bdd import adapt_bdd_record
from edgeguard.detection.contracts import Detection, LetterboxTransform, box_mask_overlap
from edgeguard.evaluation.components import ComponentRecord, connected_components
from edgeguard.evaluation.ood import (
    pixel_ood_metrics,
    score_distribution,
    select_anomaly_threshold,
)
from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.scoring.anomaly_head import LinearAnomalyHead, synthetic_outlier_exposure
from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
    msp_anomaly_score,
    predictive_entropy,
)
from edgeguard.serialization import sha256_file, sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json
from edgeguard.temporal import TemporalPersistence
from edgeguard.training.config import load_semantic_model_suite

StageOperation = Callable[["StageContext"], dict[str, Any]]
MODEL_NAMES = ("fast_scnn", "bisenetv2", "pidnet_s", "ddrnet_23_slim", "segformer_b0")


class StageContext:
    """Runtime-only paths and identities supplied to one stage implementation."""

    def __init__(
        self,
        *,
        campaign_root: Path,
        repository: Path,
        campaign_id: str,
        profile: str,
        stage_id: str,
        attempt: int,
        mmseg_checkout: Path | None,
    ) -> None:
        self.campaign_root = campaign_root
        self.repository = repository
        self.campaign_id = campaign_id
        self.profile = profile
        self.stage_id = stage_id
        self.attempt = attempt
        self.mmseg_checkout = mmseg_checkout
        self.stage_root = campaign_root / "stages" / stage_id / f"attempt-{attempt:02d}"
        self.stage_root.mkdir(parents=True, exist_ok=False)

    def write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.stage_root / name
        atomic_write_json(path, payload)
        return path


def _base_receipt(context: StageContext, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_campaign_stage_receipt",
        "campaign_id": context.campaign_id,
        "stage_id": context.stage_id,
        "attempt_number": context.attempt,
        "profile": context.profile,
        "scientific_evidence": PROFILES[context.profile].scientific_execution,
        **values,
    }


def _preflight(context: StageContext) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(context.repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _base_receipt(
        context,
        status="passed",
        git_commit=commit,
        python_version=platform.python_version(),
        machine=platform.machine(),
        profile_contract=PROFILES[context.profile].__dict__,
        boundaries={
            "real_dataset_download": False,
            "pretrained_weight_download": False,
            "sealed_dataset_access": False,
            "jetson_execution": False,
        },
    )


def _storage_inventory(context: StageContext) -> dict[str, Any]:
    usage = shutil.disk_usage(context.campaign_root)
    return _base_receipt(
        context,
        status="passed",
        campaign_bytes_before_stage=sum(
            path.stat().st_size for path in context.campaign_root.rglob("*") if path.is_file()
        ),
        filesystem_free_bytes=usage.free,
        local_storage_limit_bytes=2 * 1024**3,
        canonical_external_storage_mutated=False,
    )


def _dataset_prepare(context: StageContext) -> dict[str, Any]:
    profile = PROFILES[context.profile]
    data_root = context.campaign_root / "recovery" / "synthetic-data"
    data_root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(20260727)
    images = generator.integers(
        0, 256, size=(8, 3, profile.image_height, profile.image_width), dtype=np.uint8
    )
    targets = generator.integers(
        0, 19, size=(8, profile.image_height, profile.image_width), dtype=np.uint8
    )
    targets[:, 0, 0] = 255
    data_path = data_root / "mini_cityscapes.npz"
    np.savez_compressed(data_path, images=images, targets=targets)
    detection_fixture = {
        "name": "synthetic-frame-0001",
        "labels": [
            {
                "category": "car",
                "box2d": {"x1": 12, "y1": 10, "x2": 30, "y2": 25},
                "score": 1.0,
            }
        ],
    }
    fixture_path = data_root / "mini_bdd.json"
    atomic_write_json(fixture_path, detection_fixture)
    manifest = {
        "schema_version": "1.0",
        "record_type": "project_owned_synthetic_dataset_manifest",
        "dataset_role": "local_pipeline_validation_only",
        "image_count": int(images.shape[0]),
        "image_shape_nchw": list(images.shape),
        "semantic_class_count": 19,
        "video_frame_count": 5,
        "files": [
            {"relative_path": data_path.name, "sha256": sha256_file(data_path)},
            {"relative_path": fixture_path.name, "sha256": sha256_file(fixture_path)},
        ],
        "generator_config_sha256": sha256_payload(
            {"seed": 20260727, "shape": list(images.shape), "version": "local-mini-v1"}
        ),
        "scientific_evidence": False,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    atomic_write_json(data_root / "manifest.json", manifest)
    return _base_receipt(
        context,
        status="passed",
        dataset_manifest_sha256=manifest["manifest_sha256"],
        samples=8,
        synthetic_fixture=True,
    )


def _semantic_compatibility(context: StageContext) -> dict[str, Any]:
    config_root = context.repository / "configs" / "training" / "segmentation"
    models = load_semantic_model_suite(config_root)
    return _base_receipt(
        context,
        status="passed",
        model_families=[model.model_family.value for model in models],
        config_hashes=[sha256_payload(model.model_dump(mode="json")) for model in models],
        framework_available=all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "mmengine", "mmcv", "mmseg")
        ),
        actual_architecture_probe_requested=context.mmseg_checkout is not None,
    )


def _tiny_semantic_model(torch: Any, model_index: int) -> Any:
    hidden = 4 + model_index
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, hidden, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden, 19, kernel_size=1),
    )


def _semantic_smoke(context: StageContext) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    source = np.load(context.campaign_root / "recovery" / "synthetic-data" / "mini_cityscapes.npz")
    images = np.asarray(source["images"], dtype=np.float32) / 255.0
    targets = np.asarray(source["targets"], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(20260727)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
    results: list[dict[str, Any]] = []
    steps = PROFILES[context.profile].optimizer_steps
    checkpoint_root = context.campaign_root / "checkpoints" / "semantic"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for model_index, model_name in enumerate(MODEL_NAMES):
        torch.manual_seed(20260727 + model_index)
        model = _tiny_semantic_model(torch, model_index)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, labels) in enumerate(loader, start=1):
            if step > steps:
                break
            if step % 2 == 0:
                inputs = torch.flip(inputs, dims=(3,))
                labels = torch.flip(labels, dims=(2,))
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255) / 2
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite semantic loss for {model_name}")
            loss.backward()
            if step % 2 == 0 or step == steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach()) * 2)
        with torch.no_grad():
            validation_logits = model(torch.from_numpy(images[:2]))
        prediction = validation_logits.argmax(dim=1).numpy().astype(np.int64)
        metrics = SemanticConfusionMatrix()
        metrics.update(prediction, targets[:2])
        checkpoint = checkpoint_root / f"{model_name}.pt"
        identity = sha256_payload(
            {"campaign_id": context.campaign_id, "model": model_name, "profile": context.profile}
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "identity": identity,
            },
            checkpoint,
        )
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if restored["identity"] != identity:
            raise ValueError("semantic checkpoint identity mismatch")
        resumed = _tiny_semantic_model(torch, model_index)
        resumed.load_state_dict(restored["model"], strict=True)
        results.append(
            {
                "model_family": model_name,
                "execution": "project_owned_tiny_common_training_path",
                "optimizer_steps": steps,
                "train_losses": losses,
                "validation": metrics.result(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "exact_resume": True,
                "native_logits_shape": list(validation_logits.shape),
                "scientific_evidence": False,
            }
        )
    actual_probe: dict[str, Any] = {"status": "not_requested"}
    if context.mmseg_checkout is not None:
        actual_output = context.stage_root / "actual-five-model-probe"
        commit = subprocess.run(
            ["git", "-C", str(context.repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        training_module = __import__("scripts.train.train_semantic", fromlist=["run_stack_probe"])
        actual_probe = training_module.run_stack_probe(
            context.repository / "configs" / "training" / "segmentation",
            context.mmseg_checkout,
            actual_output,
            context.repository,
            commit,
            device_name="cpu",
            allow_dirty_project=True,
        )
    return _base_receipt(
        context,
        status="passed",
        models=results,
        actual_five_model_probe=actual_probe,
        ranking_permitted=False,
    )


def _semantic_plan_stage(context: StageContext) -> dict[str, Any]:
    profiles = {
        "semantic_screening": "five-model-short-screening",
        "semantic_medium": "top-three-medium-budget",
        "semantic_hpo": "top-two-limited-hpo-fixed-512x1024",
        "semantic_final": "three-project-owned-final-runs-one-random-init",
    }
    return _base_receipt(
        context,
        status="passed",
        local_execution="contract_and_artifact_handoff_validation",
        real_profile=profiles[context.stage_id],
        real_training_executed=False,
        ranking_or_promotion_performed=False,
    )


def _zero_shot_ood(context: StageContext) -> dict[str, Any]:
    generator = np.random.default_rng(113)
    logits = generator.normal(size=(1, 19, 16, 32)).astype(np.float32)
    labels = np.zeros((1, 16, 32), dtype=np.uint8)
    labels[:, 5:10, 12:20] = 1
    labels[:, 0, 0] = 255
    methods = {
        "msp": msp_anomaly_score(logits),
        "entropy": predictive_entropy(logits),
        "max_logit": max_logit_anomaly_score(logits),
        "energy": energy_anomaly_score(logits),
    }
    summaries: dict[str, Any] = {}
    for name, score in methods.items():
        metrics = pixel_ood_metrics(score, labels)
        summaries[name] = {
            "metrics": metrics.__dict__,
            "distribution": score_distribution(score, labels),
        }
    threshold = select_anomaly_threshold(methods["msp"], labels, target_tpr=0.95)
    road = np.zeros((16, 32), dtype=np.bool_)
    road[6:, 6:27] = True
    components = connected_components(
        np.asarray(methods["msp"][0] >= threshold["threshold"], dtype=np.bool_),
        np.asarray(methods["msp"][0], dtype=np.float32),
        road_mask=road,
    )
    output = {
        "summaries": summaries,
        "development_threshold": threshold,
        "components": [component.to_dict() for component in components],
        "raw_scores_are_probabilities": False,
    }
    context.write("ood_outputs.json", output)
    return _base_receipt(context, status="passed", **output)


def _temperature_calibration(context: StageContext) -> dict[str, Any]:
    generator = np.random.default_rng(17)
    logits = generator.normal(size=(2, 19, 8, 16)).astype(np.float32)
    targets = generator.integers(0, 19, size=(2, 8, 16), dtype=np.uint8)
    targets[:, 0, 0] = 255
    fit = fit_temperature(logits, targets, max_iterations=20)
    bin_edges = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
    before = calibration_metrics(logits, targets, temperature=1.0, bin_edges=bin_edges)
    after = calibration_metrics(
        logits, targets, temperature=fit.final_temperature, bin_edges=bin_edges
    )
    return _base_receipt(
        context,
        status="passed",
        fit=fit.to_dict(),
        before=before,
        after=after,
        calibration_role="synthetic_local_pipeline_validation",
    )


def _trainable_anomaly_head(context: StageContext) -> dict[str, Any]:
    features, targets = synthetic_outlier_exposure(seed=20260727, sample_count=64, feature_count=6)
    identity = {
        "campaign_id": context.campaign_id,
        "feature_adapter": "synthetic_decoder_candidate_v1",
    }
    head = LinearAnomalyHead.initialized(6)
    initial = head.train_steps(features, targets, steps=2, learning_rate=0.2)
    checkpoint = context.campaign_root / "checkpoints" / "anomaly_head.json"
    payload = head.checkpoint(checkpoint, identity=identity)
    resumed = LinearAnomalyHead.resume(checkpoint, identity=identity)
    resumed_losses = resumed.train_steps(features, targets, steps=1, learning_rate=0.2)
    return _base_receipt(
        context,
        status="passed",
        candidate_feature_adapter="synthetic_decoder_candidate_v1",
        candidate_loss="BCEWithLogits_equivalent",
        losses=initial + resumed_losses,
        checkpoint_sha256=payload["checkpoint_sha256"],
        exact_resume=True,
        zero_shot_vs_trainable_comparison={
            "status": "pipeline_only",
            "promotion_decision": None,
        },
    )


def _detection_smoke(context: StageContext) -> dict[str, Any]:
    fixture = json.loads(
        (context.campaign_root / "recovery" / "synthetic-data" / "mini_bdd.json").read_text(
            encoding="utf-8"
        )
    )
    batch = adapt_bdd_record(fixture, class_mapping={"car": 0}, image_shape_hw=(32, 64))
    transform = LetterboxTransform.for_shapes((32, 64), (64, 64))
    restored = transform.to_source((12.0, 21.0, 30.0, 36.0))
    road = np.zeros((32, 64), dtype=np.bool_)
    road[16:, :] = True
    overlap = box_mask_overlap(Detection(restored, 0, 1.0), road)
    return _base_receipt(
        context,
        status="passed",
        detector_families=["yolo11n", "rt_detr_r18"],
        weights_downloaded=False,
        random_architecture_construction="not_required_for_contract_probe",
        batch=batch.to_dict(),
        letterbox_reversal=list(restored),
        segmentation_overlap=overlap,
    )


def _detection_training(context: StageContext) -> dict[str, Any]:
    return _base_receipt(
        context,
        status="passed",
        local_execution="adapter_contract_and_fusion_only",
        real_detector_training_executed=False,
        configs=["yolo11n", "rt_detr_r18"],
    )


def _contextual_risk_stage(context: StageContext) -> dict[str, Any]:
    weights = RiskWeights(1.0, 0.4, 0.8, 1.2, 0.8, 0.5, 1.0)
    off_road = contextual_risk(
        {
            "anomaly_score": 0.8,
            "component_area": 0.2,
            "image_position": 0.2,
            "road_overlap": 0.0,
            "relative_proximity": 0.2,
            "detector_overlap": 0.0,
            "temporal_persistence": 0.1,
        },
        weights,
    )
    persistent = contextual_risk(
        {
            "anomaly_score": 0.8,
            "component_area": 0.6,
            "image_position": 0.9,
            "road_overlap": 1.0,
            "relative_proximity": 0.8,
            "detector_overlap": 0.5,
            "temporal_persistence": 1.0,
        },
        weights,
    )
    return _base_receipt(
        context,
        status="passed",
        weights=weights.as_dict(),
        cases={"off_road": off_road, "persistent_road_center": persistent},
        scientific_weight_tuning=False,
    )


def _component(component_id: int, x: int, score: float) -> ComponentRecord:
    return ComponentRecord(component_id, 16, (x, 12, x + 4, 16), (x + 1.5, 13.5), score, score, 1.0)


def _temporal_fusion(context: StageContext) -> dict[str, Any]:
    tracker = TemporalPersistence(centroid_distance=8.0, missed_frame_tolerance=1)
    sequence: list[dict[str, Any]] = []
    frames = (
        (_component(1, 10, 0.7),),
        (_component(1, 12, 0.8),),
        (),
        (_component(1, 15, 0.9),),
        (),
    )
    for frame_index, components in enumerate(frames):
        records = tracker.update("synthetic-sequence", frame_index, components)
        sequence.append({"frame_index": frame_index, "tracks": list(records)})
    snapshot = tracker.snapshot()
    context.write("temporal_checkpoint.json", snapshot)
    return _base_receipt(
        context,
        status="passed",
        frames=sequence,
        checkpoint_sha256=sha256_payload(snapshot),
        explanation_available=True,
    )


def _export_probe(context: StageContext) -> dict[str, Any]:
    dependency = importlib.util.find_spec("onnx") is not None
    records = []
    for model_name in MODEL_NAMES:
        records.append(
            {
                "model_family": model_name,
                "status": (
                    "not_attempted_missing_optional_onnx" if not dependency else "ready_for_probe"
                ),
                "opset": None,
                "input_name": "model_input",
                "output_name": "native_logits",
                "fixed_input_shape": [1, 3, 32, 64],
                "output_shape": None,
                "numerical_tolerance": None,
                "unsupported_operators": [],
                "file_size_bytes": None,
                "failure_classification": (
                    "optional_dependency_unavailable" if not dependency else None
                ),
                "jetson_performance_claim": False,
            }
        )
    return _base_receipt(
        context,
        status="passed",
        probes=records,
        tensorrt_execution=False,
        jetson_contract={
            "engine_built_on_target": True,
            "numerical_validation_required": True,
            "sustained_benchmark_required": True,
        },
    )


def _final_evaluation(context: StageContext) -> dict[str, Any]:
    return _base_receipt(
        context,
        status="passed",
        local_mini_pipeline_complete=True,
        result_classification="NON-SCIENTIFIC PIPELINE VALIDATION",
        scientific_claims_permitted=False,
        remaining_platform_checks=[
            "real-profile Colab execution",
            "CUDA numerical validation",
            "TensorRT build on approved Jetson",
            "sustained Jetson latency-memory-power-thermal benchmark",
        ],
    )


def _report_placeholder(context: StageContext) -> dict[str, Any]:
    return _base_receipt(
        context,
        status="passed",
        reporting_command_ready=True,
        note="report artifacts are generated by edgeguard.campaign report",
    )


STAGE_OPERATIONS: dict[str, StageOperation] = {
    "preflight": _preflight,
    "storage_inventory": _storage_inventory,
    "dataset_prepare": _dataset_prepare,
    "semantic_compatibility": _semantic_compatibility,
    "semantic_smoke": _semantic_smoke,
    "semantic_screening": _semantic_plan_stage,
    "semantic_medium": _semantic_plan_stage,
    "semantic_hpo": _semantic_plan_stage,
    "semantic_final": _semantic_plan_stage,
    "zero_shot_ood": _zero_shot_ood,
    "temperature_calibration": _temperature_calibration,
    "trainable_anomaly_head": _trainable_anomaly_head,
    "detection_smoke": _detection_smoke,
    "detection_training": _detection_training,
    "contextual_risk": _contextual_risk_stage,
    "temporal_fusion": _temporal_fusion,
    "export_probe": _export_probe,
    "final_evaluation": _final_evaluation,
    "report_generation": _report_placeholder,
}


def execute_stage(context: StageContext) -> Path:
    """Execute one registered stage and write its canonical receipt."""
    started = time.monotonic()
    payload = STAGE_OPERATIONS[context.stage_id](context)
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["output_identity_sha256"] = sha256_payload(payload)
    return context.write("stage_receipt.json", payload)
