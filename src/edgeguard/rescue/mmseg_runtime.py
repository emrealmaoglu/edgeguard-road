"""Minimal standard-MMSeg training and evaluation runtime for real data."""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import json
import os
import platform
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.calibration import apply_temperature, calibration_metrics, fit_temperature
from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.rescue.colab_recovery import latest_checkpoint, restore_recovery_file
from edgeguard.rescue.config import RescueConfig, model_by_name
from edgeguard.rescue.dataset import (
    CITYSCAPES_CLASSES,
    discover_cityscapes,
    role_records,
    validate_split_manifest,
)
from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.rescue.multidomain import validate_dataset_manifest, verify_sealed_release
from edgeguard.rescue.reliability import confidence_entropy_summary, save_calibration_evidence
from edgeguard.rescue.shift import frame_uncertainty_summary, uncertainty_maps
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

CITYSCAPES_PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def install_mmcv_lite_guard() -> bool:
    """Permit pure-Python model imports and fail closed if a compiled op is called."""
    try:
        importlib.metadata.version("mmcv-lite")
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        __import__("mmcv._ext")
    except ModuleNotFoundError:
        extension = types.ModuleType("mmcv._ext")
        extension.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)

        def unavailable(name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)

            def fail(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    f"selected model attempted unavailable compiled MMCV operation: {name}"
                )

            return fail

        extension.__getattr__ = unavailable  # type: ignore[method-assign]
        sys.modules["mmcv._ext"] = extension
        return True
    return False


def _imports() -> tuple[Any, Any, Any]:
    install_mmcv_lite_guard()
    try:
        torch = __import__("torch")
        mmengine = __import__("mmengine")
        __import__("mmengine.visualization")
        mmseg = __import__("mmseg")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "semantic runtime is missing; run scripts/train/install_semantic_stack.py "
            "or use the canonical Colab notebook"
        ) from error
    from edgeguard.rescue.mmseg_components import register_mmseg_components

    register_mmseg_components()
    return torch, mmengine, mmseg


def _config_import() -> Any:
    try:
        return __import__("mmengine")
    except ModuleNotFoundError as error:
        raise RuntimeError("install MMEngine before resolving an MMSeg config") from error


def _strip_pretrained(value: Any) -> None:
    if isinstance(value, dict):
        if "pretrained" in value:
            value["pretrained"] = None
        init_cfg = value.get("init_cfg")
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "Pretrained":
            value["init_cfg"] = None
        for child in value.values():
            _strip_pretrained(child)
    elif isinstance(value, list):
        for child in value:
            _strip_pretrained(child)


def _apply_verified_pretraining(value: Any, checkpoint: Path) -> int:
    """Replace declared classification pretraining with one verified local file."""
    replacements = 0
    if isinstance(value, dict):
        init_cfg = value.get("init_cfg")
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "Pretrained":
            init_cfg["checkpoint"] = str(checkpoint)
            replacements += 1
        for child in value.values():
            replacements += _apply_verified_pretraining(child, checkpoint)
    elif isinstance(value, list):
        for child in value:
            replacements += _apply_verified_pretraining(child, checkpoint)
    return replacements


def _verified_pretrained_checkpoint(manifest_path: Path, model_name: str) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "record_type": "edgeguard_pretrained_initialization",
        "model": model_name,
        "source_task": "image_classification",
        "human_approved": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("pretrained manifest is not approved for this model/classification task")
    checkpoint = Path(str(payload.get("checkpoint_path", "")))
    if not checkpoint.is_file() or sha256_file(checkpoint) != payload.get("checkpoint_sha256"):
        raise ValueError("pretrained checkpoint path/hash mismatch")
    if (
        not payload.get("source_url")
        or not payload.get("license_id")
        or not payload.get("access_date")
    ):
        raise ValueError("pretrained manifest lacks source/license/access provenance")
    return checkpoint


def _set_num_classes_and_loss(value: Any, *, class_weights: list[float] | None) -> None:
    if isinstance(value, dict):
        if "num_classes" in value:
            value["num_classes"] = 19
        if value.get("type") == "CrossEntropyLoss":
            value["class_weight"] = class_weights
        for child in value.values():
            _set_num_classes_and_loss(child, class_weights=class_weights)
    elif isinstance(value, list):
        for child in value:
            _set_num_classes_and_loss(child, class_weights=class_weights)


def materialize_role_file(manifest_path: Path, role: str, destination: Path) -> int:
    """Write a standard MMSeg Cityscapes ann_file from the frozen root-free manifest."""
    records = role_records(manifest_path, role)
    prefix = "leftImg8bit/train/"
    suffix = "_leftImg8bit.png"
    lines: list[str] = []
    for record in records:
        image = str(record["image"])
        if not image.startswith(prefix) or not image.endswith(suffix):
            raise ValueError(f"role {role} contains a non-train Cityscapes path")
        lines.append(image[len(prefix) : -len(suffix)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def validate_scientific_split(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Bind a frozen split to the complete real Cityscapes train inventory."""
    audit_summary_path = manifest_path.with_name("summary.json")
    if not audit_summary_path.is_file():
        raise FileNotFoundError(
            "scientific training requires summary.json beside the frozen split manifest"
        )
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    if audit_summary.get("record_type") != "cityscapes_dataset_audit":
        raise ValueError("split-adjacent summary.json is not a Cityscapes audit")
    if audit_summary.get("audit_passed") is not True or audit_summary.get("valid_pairs") != 2_975:
        raise ValueError("Cityscapes audit gate is not a passing 2,975-pair inventory")
    samples, missing_masks = discover_cityscapes(dataset_root, split="train")
    if missing_masks:
        raise ValueError(
            f"Cityscapes train inventory has {len(missing_masks)} images without train-id masks"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_split_manifest(payload, samples)
    return payload


def _train_pipeline(config: RescueConfig) -> list[dict[str, Any]]:
    return [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "reduce_zero_label": False},
        {
            "type": "RandomResize",
            "scale": config.base_scale,
            "ratio_range": (0.5, 2.0),
            "keep_ratio": True,
        },
        {"type": "RandomCrop", "crop_size": config.crop_size, "cat_max_ratio": 0.75},
        {"type": "RandomFlip", "prob": 0.5},
        {"type": "PhotoMetricDistortion"},
        {"type": "PackSegInputs"},
    ]


def _evaluation_pipeline(config: RescueConfig) -> list[dict[str, Any]]:
    return [
        {"type": "LoadImageFromFile"},
        {
            "type": "Resize",
            "scale": (config.crop_size[1], config.crop_size[0]),
            "keep_ratio": True,
        },
        {"type": "LoadAnnotations", "reduce_zero_label": False},
        {
            "type": "Pad",
            "size": config.crop_size,
            "pad_val": 0,
            "seg_pad_val": config.ignore_index,
        },
        {"type": "PackSegInputs"},
    ]


def _inference_pipeline(config: RescueConfig) -> list[dict[str, Any]]:
    """Match evaluation geometry without requiring a ground-truth mask."""
    return [
        {"type": "LoadImageFromFile"},
        {
            "type": "Resize",
            "scale": (config.crop_size[1], config.crop_size[0]),
            "keep_ratio": True,
        },
        {"type": "Pad", "size": config.crop_size, "pad_val": 0},
        {"type": "PackSegInputs"},
    ]


def _cityscapes_dataset(
    dataset_root: Path,
    config: RescueConfig,
    *,
    ann_file: Path | None,
    split: str,
    training: bool,
) -> dict[str, Any]:
    return {
        "type": "CityscapesDataset",
        "data_root": str(dataset_root),
        "data_prefix": {
            "img_path": f"leftImg8bit/{split}",
            "seg_map_path": f"gtFine/{split}",
        },
        "ann_file": str(ann_file) if ann_file is not None else "",
        "pipeline": _train_pipeline(config) if training else _evaluation_pipeline(config),
    }


def _acdc_dataset(dataset_root: Path, config: RescueConfig, *, condition: str) -> dict[str, Any]:
    image_prefix = f"rgb_anon/{condition}/val"
    mask_prefix = f"gt/{condition}/val"
    for relative in (image_prefix, mask_prefix):
        if not (dataset_root / relative).is_dir():
            raise FileNotFoundError(
                f"missing original-layout ACDC validation directory: {dataset_root / relative}"
            )
    return {
        "type": "BaseSegDataset",
        "data_root": str(dataset_root),
        "data_prefix": {
            "img_path": image_prefix,
            "seg_map_path": mask_prefix,
        },
        "img_suffix": "_rgb_anon.png",
        "seg_map_suffix": "_gt_labelTrainIds.png",
        "metainfo": {"classes": CITYSCAPES_CLASSES, "palette": CITYSCAPES_PALETTE},
        "pipeline": _evaluation_pipeline(config),
    }


def _dataloader(
    dataset: dict[str, Any],
    config: RescueConfig,
    *,
    training: bool,
    iter_based: bool = False,
) -> dict[str, Any]:
    loader = {
        "batch_size": config.device_batch if training else 1,
        "num_workers": config.workers,
        "persistent_workers": config.workers > 0,
        "pin_memory": True,
        "sampler": {
            "type": "InfiniteSampler" if training and iter_based else "DefaultSampler",
            "shuffle": training,
        },
        "dataset": dataset,
    }
    if config.workers > 0:
        loader["prefetch_factor"] = 2
    return loader


def _manifest_dataset(
    manifest_path: Path,
    config: RescueConfig,
    *,
    role: str,
    training: bool,
) -> dict[str, Any]:
    """Build an explicit-pair dataset without relying on vendor filename inference."""
    validate_dataset_manifest(manifest_path)
    return {
        "type": "EdgeGuardManifestDataset",
        "manifest_path": str(manifest_path),
        "role": role,
        "pipeline": _train_pipeline(config) if training else _evaluation_pipeline(config),
    }


def _manifest_dataloader(
    manifests: Sequence[Path],
    config: RescueConfig,
    *,
    role: str,
    training: bool,
) -> dict[str, Any]:
    """Build a single- or multi-domain loader with uniform domain probability."""
    datasets = [_manifest_dataset(path, config, role=role, training=training) for path in manifests]
    sampler: dict[str, Any]
    if len(datasets) == 1:
        dataset: dict[str, Any] = datasets[0]
        sampler = {
            "type": "InfiniteSampler" if training else "DefaultSampler",
            "shuffle": training,
        }
    else:
        dataset = {"type": "ConcatDataset", "datasets": datasets}
        sampler = (
            {
                "type": "EdgeGuardDomainBalancedSampler",
                "shuffle": True,
                "seed": config.seed,
                "round_up": True,
            }
            if training
            else {"type": "DefaultSampler", "shuffle": False}
        )
    loader = {
        "batch_size": config.device_batch if training else 1,
        "num_workers": config.workers,
        "persistent_workers": config.workers > 0,
        "pin_memory": True,
        "sampler": sampler,
        "dataset": dataset,
    }
    if config.workers > 0:
        loader["prefetch_factor"] = 2
    return loader


def _load_class_weights(
    audit_report: Path | None, loss: str, *, split_manifest: Path
) -> list[float] | None:
    if loss == "ce":
        return None
    if loss != "median_frequency":
        raise ValueError("loss must be ce or median_frequency")
    if audit_report is None:
        raise ValueError("median_frequency loss requires --audit-report")
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    if payload.get("source_role") != "train_fit":
        raise ValueError("class weights must be derived only from the frozen train_fit role")
    if payload.get("split_manifest_sha256") != sha256_file(split_manifest):
        raise ValueError("class weights do not match the selected split manifest")
    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != 19:
        raise ValueError("audit class_weights.json does not contain 19 frozen weights")
    result = [float(value) for value in weights]
    if any(not np.isfinite(value) or value <= 0 for value in result):
        raise ValueError("class weights must be positive and finite")
    return result


def build_training_config(
    protocol: RescueConfig,
    *,
    model_name: str,
    stage_name: str,
    mmseg_root: Path,
    dataset_root: Path | None,
    split_manifest: Path | None,
    work_dir: Path,
    loss: str,
    audit_report: Path | None,
    resume: bool,
    data_manifests: Sequence[Path] | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    scheduler: str = "poly",
    warmup_ratio: float = 0.03,
    max_steps_override: int | None = None,
    scheduler_steps_override: int | None = None,
    initialization: str = "random",
    pretrained_manifest: Path | None = None,
    precision: str = "fp32",
    recovery_root: Path | None = None,
    recovery_artifact_id: str | None = None,
    campaign_id: str | None = None,
    project_commit: str | None = None,
    identity_sha256: str | None = None,
    intentional_interrupt_optimizer_step: int | None = None,
) -> Any:
    """Resolve one standard MMSeg Runner config with no custom loop or hook."""
    mmengine = _config_import()
    model = model_by_name(protocol, model_name)
    stage = protocol.stages[stage_name]
    upstream = mmseg_root / model.upstream_config
    if not upstream.is_file():
        raise FileNotFoundError(f"missing upstream MMSeg config: {upstream}")
    cfg = mmengine.Config.fromfile(str(upstream))
    manifests = tuple(data_manifests or ())
    if manifests and loss == "median_frequency":
        if audit_report is None:
            raise ValueError("multi-domain median_frequency loss requires --audit-report")
        weight_payload = json.loads(audit_report.read_text(encoding="utf-8"))
        expected_hashes = sorted(sha256_file(path) for path in manifests)
        if weight_payload.get("source_role") != "train_fit":
            raise ValueError("multi-domain class weights must come only from train_fit")
        if sorted(weight_payload.get("dataset_manifest_sha256s", [])) != expected_hashes:
            raise ValueError("multi-domain class weights do not match every dataset manifest")
        values = weight_payload.get("weights")
        if not isinstance(values, list) or len(values) != 19:
            raise ValueError("multi-domain class-weight report must contain 19 values")
        class_weights = [float(value) for value in values]
    else:
        if split_manifest is None:
            if loss != "ce":
                raise ValueError("single-domain weighted loss requires a split manifest")
            class_weights = None
        else:
            class_weights = _load_class_weights(audit_report, loss, split_manifest=split_manifest)
    if initialization == "random":
        if pretrained_manifest is not None:
            raise ValueError("random initialization cannot receive a pretrained manifest")
        _strip_pretrained(cfg.model)
    elif initialization == "pretrained":
        if pretrained_manifest is None:
            raise ValueError("pretrained initialization requires --pretrained-manifest")
        checkpoint = _verified_pretrained_checkpoint(pretrained_manifest, model_name)
        if _apply_verified_pretraining(cfg.model, checkpoint) == 0:
            raise ValueError(f"model {model_name} declares no supported pretrained init_cfg")
    else:
        raise ValueError("initialization must be random or pretrained")
    _set_num_classes_and_loss(cfg.model, class_weights=class_weights)
    data_preprocessor = cfg.model.get("data_preprocessor")
    if not isinstance(data_preprocessor, dict) or "size" not in data_preprocessor:
        raise ValueError(f"model {model_name} has no fixed-size SegDataPreProcessor")
    data_preprocessor["size"] = protocol.crop_size
    cfg.crop_size = protocol.crop_size
    cfg.test_pipeline = _inference_pipeline(protocol)
    cfg.work_dir = str(work_dir)
    cfg.randomness = {"seed": protocol.seed, "deterministic": False}
    cfg.resume = resume
    cfg.load_from = None
    if manifests:
        cfg.train_dataloader = _manifest_dataloader(
            manifests, protocol, role="train_fit", training=True
        )
        cfg.val_dataloader = _manifest_dataloader(
            manifests, protocol, role="train_select", training=False
        )
    else:
        if dataset_root is None or split_manifest is None:
            raise ValueError("legacy Cityscapes training requires dataset root and split manifest")
        train_file = work_dir / "manifests" / "train_fit.txt"
        select_file = work_dir / "manifests" / "train_select.txt"
        materialize_role_file(split_manifest, "train_fit", train_file)
        materialize_role_file(split_manifest, "train_select", select_file)
        cfg.train_dataloader = _dataloader(
            _cityscapes_dataset(
                dataset_root, protocol, ann_file=train_file, split="train", training=True
            ),
            protocol,
            training=True,
            iter_based=True,
        )
        cfg.val_dataloader = _dataloader(
            _cityscapes_dataset(
                dataset_root, protocol, ann_file=select_file, split="train", training=False
            ),
            protocol,
            training=False,
        )
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}
    cfg.test_evaluator = cfg.val_evaluator
    cfg.val_cfg = {"type": "ValLoop"}
    cfg.test_cfg = {"type": "TestLoop"}
    assert stage.max_steps is not None
    max_steps = stage.max_steps if max_steps_override is None else max_steps_override
    scheduler_steps = max_steps if scheduler_steps_override is None else scheduler_steps_override
    if max_steps <= 0 or scheduler_steps < max_steps:
        raise ValueError("step overrides must be positive and scheduler horizon cannot be shorter")
    by_epoch = False
    max_iterations = max_steps * protocol.gradient_accumulation
    scheduler_iterations = scheduler_steps * protocol.gradient_accumulation
    validation_steps = (
        max_steps if stage_name in {"smoke", "pilot", "hpo"} else min(2_000, max_steps)
    )
    validation_interval = validation_steps * protocol.gradient_accumulation
    cfg.train_cfg = {
        "type": "IterBasedTrainLoop",
        "max_iters": max_iterations,
        "val_interval": validation_interval,
    }
    scheduler_end = scheduler_iterations
    resolved_lr = protocol.learning_rate if learning_rate is None else learning_rate
    resolved_weight_decay = protocol.weight_decay if weight_decay is None else weight_decay
    if resolved_lr <= 0 or resolved_weight_decay < 0:
        raise ValueError("optimizer overrides must be positive")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    cfg.optim_wrapper = {
        "type": "OptimWrapper" if precision == "fp32" else "AmpOptimWrapper",
        "optimizer": {
            "type": "AdamW",
            "lr": resolved_lr,
            "weight_decay": resolved_weight_decay,
        },
        "accumulative_counts": protocol.gradient_accumulation,
        "clip_grad": {
            "max_norm": float("inf"),
            "norm_type": 2.0,
            "error_if_nonfinite": True,
        },
    }
    if precision != "fp32":
        cfg.optim_wrapper["dtype"] = "bfloat16" if precision == "bf16" else "float16"
        if precision == "fp16":
            cfg.optim_wrapper["loss_scale"] = "dynamic"
    if scheduler not in {"poly", "cosine"}:
        raise ValueError("scheduler must be poly or cosine")
    if warmup_ratio not in {0.01, 0.03, 0.05}:
        raise ValueError("warmup_ratio must be one of the frozen HPO values")
    warmup_end = max(1, round(scheduler_end * warmup_ratio))
    main_scheduler = {
        "type": "PolyLR" if scheduler == "poly" else "CosineAnnealingLR",
        "eta_min": 1.0e-6,
        "begin": warmup_end,
        "end": scheduler_end,
        "by_epoch": False,
    }
    if scheduler == "poly":
        main_scheduler["power"] = 0.9
    cfg.param_scheduler = [
        {
            "type": "LinearLR",
            "start_factor": 1.0e-3,
            "begin": 0,
            "end": warmup_end,
            "by_epoch": False,
        },
        main_scheduler,
    ]
    cfg.default_hooks["checkpoint"] = {
        "type": "CheckpointHook",
        "by_epoch": by_epoch,
        "interval": 1 if by_epoch else min(500, max_steps) * protocol.gradient_accumulation,
        "save_best": "mIoU",
        "rule": "greater",
        "max_keep_ckpts": 2,
    }
    if recovery_root is not None:
        if not all((recovery_artifact_id, campaign_id, project_commit, identity_sha256)):
            raise ValueError("Drive recovery requires artifact, campaign, commit, and identity")
        recovery_hook = {
            "type": "EdgeGuardRecoveryHook",
            "store_root": str(recovery_root),
            "artifact_id": recovery_artifact_id,
            "campaign_id": campaign_id,
            "project_commit": project_commit,
            "identity_sha256": identity_sha256,
            "accumulation": protocol.gradient_accumulation,
            "status_path": str(recovery_root.parents[1] / "state/status.json"),
            "optimizer_interval": 500,
            "maximum_seconds": 600,
            "intentional_interrupt_optimizer_step": intentional_interrupt_optimizer_step,
        }
        cfg.custom_hooks = [*list(cfg.get("custom_hooks", [])), recovery_hook]
    cfg.custom_hooks = [
        *list(cfg.get("custom_hooks", [])),
        {
            "type": "EdgeGuardMetricsHistoryHook",
            "accumulation": protocol.gradient_accumulation,
            "optimizer_interval": 50,
        },
    ]
    cfg.default_hooks.pop("visualization", None)
    cfg.visualizer = {
        "_scope_": "mmengine",
        "type": "Visualizer",
        "name": "edgeguard_visualizer",
        "vis_backends": [],
    }
    cfg.log_processor = {"by_epoch": by_epoch}
    cfg.edgeguard_metadata = {
        "schema_version": "1.0",
        "path": "semantic_first",
        "model": model_name,
        "stage": stage_name,
        "loss": loss,
        "initialization": initialization,
        "pretrained_manifest_sha256": (
            sha256_file(pretrained_manifest) if pretrained_manifest else None
        ),
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest else None,
        "dataset_manifest_sha256s": [sha256_file(path) for path in manifests],
        "datasets": [validate_dataset_manifest(path)["dataset_id"] for path in manifests],
        "domain_sampling": "uniform" if len(manifests) > 1 else "single_domain",
        "effective_batch": protocol.effective_batch,
        "device_batch": protocol.device_batch,
        "gradient_accumulation": protocol.gradient_accumulation,
        "precision": precision,
        "workers": protocol.workers,
        "learning_rate": resolved_lr,
        "weight_decay": resolved_weight_decay,
        "scheduler": scheduler,
        "warmup_ratio": warmup_ratio,
        "max_steps": max_steps,
        "scheduler_steps": scheduler_steps,
        "scientific_evidence_requires_real_data": True,
    }
    return cfg


def train_model(
    protocol: RescueConfig,
    *,
    model_name: str,
    stage_name: str,
    mmseg_root: Path,
    dataset_root: Path | None,
    split_manifest: Path | None,
    output_root: Path,
    loss: str,
    audit_report: Path | None,
    resume: bool,
    data_manifests: Sequence[Path] | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    scheduler: str = "poly",
    warmup_ratio: float = 0.03,
    run_name: str | None = None,
    max_steps_override: int | None = None,
    scheduler_steps_override: int | None = None,
    initialization: str = "random",
    pretrained_manifest: Path | None = None,
    device_batch: int | None = None,
    workers: int | None = None,
    precision: str = "auto",
    recovery_root: Path | None = None,
    campaign_id: str | None = None,
    project_commit: str | None = None,
    crop_size_override: tuple[int, int] | None = None,
    intentional_interrupt_optimizer_step: int | None = None,
) -> dict[str, Any]:
    """Train through the stock MMEngine Runner and record only measured evidence."""
    torch, mmengine, mmseg = _imports()
    if crop_size_override is not None:
        if len(crop_size_override) != 2 or min(crop_size_override) <= 0:
            raise ValueError("crop override must contain two positive integers")
        protocol = replace(protocol, crop_size=crop_size_override)
    scientific_protocol = asdict(protocol)
    scientific_protocol["device_batch"] = None
    scientific_protocol["gradient_accumulation"] = None
    resolved_device_batch = protocol.device_batch if device_batch is None else device_batch
    if resolved_device_batch <= 0 or protocol.effective_batch % resolved_device_batch:
        raise ValueError("device batch must be a positive divisor of frozen effective batch")
    resolved_workers = protocol.workers if workers is None else workers
    if resolved_workers < 0:
        raise ValueError("workers cannot be negative")
    protocol = replace(
        protocol,
        device_batch=resolved_device_batch,
        gradient_accumulation=protocol.effective_batch // resolved_device_batch,
        workers=resolved_workers,
    )
    stage = protocol.stages[stage_name]
    assert stage.max_steps is not None
    resolved_max_steps = stage.max_steps if max_steps_override is None else max_steps_override
    resolved_scheduler_steps = (
        resolved_max_steps if scheduler_steps_override is None else scheduler_steps_override
    )
    if resolved_max_steps <= 0 or resolved_scheduler_steps < resolved_max_steps:
        raise ValueError("step overrides must be positive and scheduler horizon cannot be shorter")
    if intentional_interrupt_optimizer_step is not None and not (
        0 < intentional_interrupt_optimizer_step < resolved_max_steps
    ):
        raise ValueError("intentional interruption must be inside the optimizer-step budget")
    if precision == "auto":
        if not torch.cuda.is_available():
            precision = "fp32"
        elif bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            precision = "bf16"
        else:
            precision = "fp16"
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if precision == "bf16" and not bool(
        torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ):
        raise ValueError("bf16 was requested on an unsupported runtime")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    manifests = tuple(data_manifests or ())
    if manifests:
        datasets: list[str] = []
        for manifest in manifests:
            payload = validate_dataset_manifest(manifest)
            if payload.get("audit_passed") is not True:
                raise ValueError(f"dataset manifest did not pass audit: {manifest}")
            if payload.get("scientific_eligible") is not True:
                raise ValueError(
                    f"dataset manifest is fixture/incomplete and cannot train: {manifest}"
                )
            if "train_fit" not in payload["roles"] or "train_select" not in payload["roles"]:
                raise ValueError("training manifests require train_fit and train_select roles")
            datasets.append(str(payload["dataset_id"]))
        if len(datasets) != len(set(datasets)):
            raise ValueError("training cannot repeat a source domain")
        approved_compositions = (
            {"cityscapes"},
            set(protocol.datasets.training),
        )
        if set(datasets) not in approved_compositions:
            raise ValueError("training manifests do not match an approved dataset ablation")
    else:
        if dataset_root is None or split_manifest is None:
            raise ValueError("provide Cityscapes root/split or one or more --data-manifest values")
        validate_scientific_split(dataset_root, split_manifest)
        datasets = ["cityscapes"]
    suffix = run_name or loss
    work_dir = output_root / stage_name / model_name / suffix
    if work_dir.exists() and any(work_dir.iterdir()) and not resume:
        raise FileExistsError(f"refusing non-empty run directory without --resume: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    model = model_by_name(protocol, model_name)
    upstream = mmseg_root / model.upstream_config
    identity = {
        "schema_version": "1.0",
        "model": model_name,
        "stage": stage_name,
        "loss": loss,
        "protocol_sha256": sha256_payload(scientific_protocol),
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest else None,
        "dataset_manifest_sha256s": [sha256_file(path) for path in manifests],
        "datasets": datasets,
        "learning_rate": protocol.learning_rate if learning_rate is None else learning_rate,
        "weight_decay": protocol.weight_decay if weight_decay is None else weight_decay,
        "scheduler": scheduler,
        "warmup_ratio": warmup_ratio,
        "initialization": initialization,
        "pretrained_manifest_sha256": (
            sha256_file(pretrained_manifest) if pretrained_manifest else None
        ),
        "upstream_config_sha256": sha256_file(upstream),
        "effective_batch": protocol.effective_batch,
        "workers": protocol.workers,
        "precision": precision,
        "max_steps": resolved_max_steps,
        "scheduler_steps": resolved_scheduler_steps,
        "intentional_interrupt_optimizer_step": intentional_interrupt_optimizer_step,
        "project_commit": project_commit,
        "class_weights_sha256": (
            sha256_file(audit_report) if loss == "median_frequency" and audit_report else None
        ),
    }
    identity_path = work_dir / "run_identity.json"
    identity_sha256 = sha256_payload(identity)
    recovery_artifact_id = f"{stage_name}-{model_name}-{suffix}".replace("_", "-")
    restored_from_drive = False
    needs_recovery_checkpoint = False
    if resume:
        try:
            latest_checkpoint(work_dir)
        except FileNotFoundError:
            needs_recovery_checkpoint = True
    if needs_recovery_checkpoint and recovery_root is not None:
        if not identity_path.is_file():
            identity_path.write_text(canonical_json(identity) + "\n", encoding="utf-8")
        restored = restore_recovery_file(
            recovery_root,
            artifact_id=recovery_artifact_id,
            destination=work_dir / "recovered.pth",
        )
        metadata = restored.get("metadata", {})
        if metadata.get("identity_sha256") != identity_sha256:
            raise ValueError("Drive recovery checkpoint belongs to a different immutable run")
        (work_dir / "last_checkpoint").write_text("recovered.pth\n", encoding="utf-8")
        restored_from_drive = True
    if resume:
        if not identity_path.is_file():
            raise FileNotFoundError("resume requires an existing run_identity.json")
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("resume identity mismatch; start a new immutable run directory")
    else:
        identity_path.write_text(canonical_json(identity) + "\n", encoding="utf-8")
    cfg = build_training_config(
        protocol,
        model_name=model_name,
        stage_name=stage_name,
        mmseg_root=mmseg_root,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
        work_dir=work_dir,
        loss=loss,
        audit_report=audit_report,
        resume=resume,
        data_manifests=manifests,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        max_steps_override=max_steps_override,
        scheduler_steps_override=scheduler_steps_override,
        initialization=initialization,
        pretrained_manifest=pretrained_manifest,
        precision=precision,
        recovery_root=recovery_root,
        recovery_artifact_id=recovery_artifact_id,
        campaign_id=campaign_id,
        project_commit=project_commit,
        identity_sha256=identity_sha256,
        intentional_interrupt_optimizer_step=intentional_interrupt_optimizer_step,
    )
    resolved_path = work_dir / "resolved.py"
    cfg.dump(str(resolved_path))
    started = time.perf_counter()
    runner = mmengine.runner.Runner.from_cfg(cfg)
    runner.train()
    elapsed = time.perf_counter() - started
    final_checkpoint = latest_checkpoint(work_dir)
    checkpoints = sorted(work_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime_ns)
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_training_summary",
        "model": model_name,
        "stage": stage_name,
        "loss": loss,
        "initialization": initialization,
        "datasets": datasets,
        "domain_sampling": "uniform" if len(datasets) > 1 else "single_domain",
        "seed": protocol.seed,
        "effective_batch": protocol.effective_batch,
        "device_batch": protocol.device_batch,
        "gradient_accumulation": protocol.gradient_accumulation,
        "precision": precision,
        "workers": protocol.workers,
        "restored_from_drive": restored_from_drive,
        "dataset_manifest_sha256s": [sha256_file(path) for path in manifests],
        "ontology_sha256s": sorted(
            {str(validate_dataset_manifest(path).get("ontology_sha256")) for path in manifests}
        ),
        "elapsed_seconds": elapsed,
        "checkpoints": [
            {"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in checkpoints
        ],
        "last_checkpoint": final_checkpoint.name,
        "resolved_config_sha256": sha256_file(resolved_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mmengine": mmengine.__version__,
            "mmseg": mmseg.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32)
            if torch.cuda.is_available()
            else False,
        },
        "scientific_evidence": True,
    }
    (work_dir / "summary.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    append_run_ledger(
        output_root / "run_ledger.jsonl", operation="semantic_training", result=result
    )
    return result


def _evaluation_dataset(
    protocol: RescueConfig,
    *,
    dataset: str,
    dataset_root: Path | None,
    split_manifest: Path | None,
    role: str,
    condition: str | None,
    output_dir: Path,
    dataset_manifest: Path | None = None,
) -> dict[str, Any]:
    if dataset_manifest is not None:
        payload = validate_dataset_manifest(dataset_manifest)
        if payload["dataset_id"] != dataset:
            raise ValueError("--dataset does not match --dataset-manifest")
        if role not in payload["roles"]:
            raise ValueError(f"dataset manifest has no role {role!r}")
        if any(record.get("canonical_mask") is None for record in payload["roles"][role]):
            raise ValueError("label-free sealed data must use external prediction packaging")
        return _manifest_dataset(dataset_manifest, protocol, role=role, training=False)
    if dataset_root is None:
        raise ValueError("legacy evaluation requires --dataset-root")
    if dataset == "acdc":
        if role != "domain_shift_val":
            raise ValueError("ACDC evaluation role must be domain_shift_val")
        if condition not in {"fog", "night", "rain", "snow"}:
            raise ValueError("ACDC requires one condition: fog, night, rain, or snow")
        return _acdc_dataset(dataset_root, protocol, condition=condition)
    if dataset == "cityscapes_stress":
        if role != "synthetic_stress_test":
            raise ValueError("Cityscapes stress role must be synthetic_stress_test")
        return _cityscapes_dataset(
            dataset_root, protocol, ann_file=None, split="val", training=False
        )
    if dataset != "cityscapes":
        raise ValueError("dataset must be cityscapes, cityscapes_stress, or acdc")
    if role == "official_val_common_eval":
        return _cityscapes_dataset(
            dataset_root, protocol, ann_file=None, split="val", training=False
        )
    if role not in {"train_select", "train_calibration"} or split_manifest is None:
        raise ValueError("Cityscapes trial evaluation requires a split manifest and valid role")
    ann_file = output_dir / "manifests" / f"{role}.txt"
    materialize_role_file(split_manifest, role, ann_file)
    return _cityscapes_dataset(
        dataset_root, protocol, ann_file=ann_file, split="train", training=False
    )


def _collect_reporting_evidence(
    runner: Any, *, max_pixels: int, temperature: float = 1.0
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    torch = __import__("torch")
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    collected = 0
    confusion = SemanticConfusionMatrix()
    frame_summaries: list[dict[str, Any]] = []
    runner.model.eval()
    for data_batch in runner.test_dataloader:
        with torch.no_grad():
            outputs = runner.model.test_step(data_batch)
        for output in outputs:
            if not hasattr(output, "seg_logits") or not hasattr(output, "gt_sem_seg"):
                raise RuntimeError("MMSeg output must preserve seg_logits and gt_sem_seg")
            logits = output.seg_logits.data.detach().cpu().numpy()
            maps = uncertainty_maps(logits / temperature)
            frame_summary: dict[str, Any] = dict(
                frame_uncertainty_summary(
                    maps["maximum_softmax_probability"],
                    maps["normalized_entropy"],
                    energy=maps["energy"],
                )
            )
            frame_summary["mean_maximum_logit"] = float(np.mean(maps["maximum_logit"]))
            frame_summary["negative_mean_maximum_logit"] = -float(
                frame_summary["mean_maximum_logit"]
            )
            metadata = getattr(output, "metainfo", {})
            image_path = metadata.get("img_path") if isinstance(metadata, dict) else None
            frame_summary["sample_id"] = Path(str(image_path)).stem if image_path else None
            frame_summaries.append(frame_summary)
            target = output.gt_sem_seg.data.detach().cpu().numpy().squeeze(0)
            if logits.shape[1:] != target.shape:
                resized = torch.nn.functional.interpolate(
                    torch.from_numpy(target[None, None].astype(np.float32)),
                    size=logits.shape[1:],
                    mode="nearest",
                )
                target = resized[0, 0].numpy().astype(np.int64)
            prediction = output.pred_sem_seg.data.detach().cpu().numpy().squeeze(0)
            if prediction.shape != target.shape:
                resized_prediction = torch.nn.functional.interpolate(
                    torch.from_numpy(prediction[None, None].astype(np.float32)),
                    size=target.shape,
                    mode="nearest",
                )
                prediction = resized_prediction[0, 0].numpy().astype(np.int64)
            confusion.update(prediction.astype(np.int64), target.astype(np.int64))
            flat_logits = logits.reshape(logits.shape[0], -1)
            flat_target = target.reshape(-1)
            remaining = max_pixels - collected
            if remaining <= 0:
                continue
            stride = max(1, int(np.ceil(flat_target.size / remaining)))
            indices = np.arange(0, flat_target.size, stride, dtype=np.int64)[:remaining]
            logits_parts.append(flat_logits[:, indices])
            target_parts.append(flat_target[indices])
            collected += indices.size
    if not logits_parts:
        return None, None, confusion.result(), frame_summaries
    logits_array = np.concatenate(logits_parts, axis=1)[None, :, None, :]
    targets_array = np.concatenate(target_parts, axis=0)[None, None, :]
    return (
        logits_array.astype(np.float32),
        targets_array.astype(np.int64),
        confusion.result(),
        frame_summaries,
    )


def evaluate_model(
    protocol: RescueConfig,
    *,
    resolved_config: Path,
    checkpoint: Path,
    dataset: str,
    dataset_root: Path | None,
    split_manifest: Path | None,
    role: str,
    condition: str | None,
    output_dir: Path,
    fit_calibrator: bool,
    temperature_file: Path | None,
    max_reliability_pixels: int,
    rare_classes_file: Path | None,
    dataset_manifest: Path | None = None,
    sealed_release: Path | None = None,
    calibration_evidence_output: Path | None = None,
    collect_classwise: bool = True,
) -> dict[str, Any]:
    """Collect accuracy, classwise IoU, uncertainty and calibration in one inference pass."""
    _, mmengine, _ = _imports()
    if dataset_manifest is not None:
        verify_sealed_release(dataset_manifest, checkpoint, sealed_release)
    if dataset == "cityscapes" and role != "official_val_common_eval" and dataset_manifest is None:
        if split_manifest is None:
            raise ValueError("Cityscapes trial evaluation requires a split manifest")
        if dataset_root is None:
            raise ValueError("Cityscapes trial evaluation requires a dataset root")
        validate_scientific_split(dataset_root, split_manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = mmengine.Config.fromfile(str(resolved_config))
    cfg.work_dir = str(output_dir)
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    dataset_cfg = _evaluation_dataset(
        protocol,
        dataset=dataset,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
        role=role,
        condition=condition,
        output_dir=output_dir,
        dataset_manifest=dataset_manifest,
    )
    cfg.test_dataloader = _dataloader(dataset_cfg, protocol, training=False)
    cfg.test_evaluator = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}
    cfg.test_cfg = {"type": "TestLoop"}
    runner = mmengine.runner.Runner.from_cfg(cfg)
    needs_logits = (
        fit_calibrator or temperature_file is not None or calibration_evidence_output is not None
    )
    reporting_temperature = 1.0
    if temperature_file is not None:
        reporting_temperature = float(
            json.loads(temperature_file.read_text(encoding="utf-8"))["final_temperature"]
        )
    if not np.isfinite(reporting_temperature) or reporting_temperature <= 0.0:
        raise ValueError("reporting temperature must be finite and positive")
    logits, targets, classwise, frame_summaries = _collect_reporting_evidence(
        runner,
        max_pixels=max_reliability_pixels if needs_logits else 0,
        temperature=reporting_temperature,
    )
    metrics = {
        "mIoU": classwise["mean_iou"],
        "mAcc": classwise["mean_class_accuracy"],
        "aAcc": classwise["pixel_accuracy"],
    }
    if not collect_classwise:
        classwise = {
            **classwise,
            "confusion_matrix": None,
            "per_class_accuracy": None,
            "collection_compacted": True,
        }
    rare_class_miou: float | None = None
    if rare_classes_file is not None:
        rare_payload = json.loads(rare_classes_file.read_text(encoding="utf-8"))
        rare_ids = rare_payload.get("groups", {}).get("rare")
        if not isinstance(rare_ids, list) or not rare_ids:
            raise ValueError("rare classes file does not define a non-empty rare group")
        per_class = classwise["per_class_iou"]
        defined = [
            float(per_class[int(class_id)])
            for class_id in rare_ids
            if per_class[int(class_id)] is not None
        ]
        rare_class_miou = float(np.mean(defined)) if defined else None
    reliability: dict[str, Any] | None = None
    if calibration_evidence_output is not None:
        if role != "train_calibration" or dataset not in {"cityscapes", "bdd100k", "idd20k"}:
            raise ValueError("calibration evidence is allowed only for source train_calibration")
        if logits is None or targets is None:
            raise RuntimeError("evaluation produced no logits for calibration evidence")
        save_calibration_evidence(
            calibration_evidence_output,
            logits=logits,
            targets=targets,
            dataset_id=dataset,
            dataset_manifest_sha256=(
                sha256_file(dataset_manifest) if dataset_manifest is not None else None
            ),
            checkpoint_sha256=sha256_file(checkpoint),
        )
    if fit_calibrator or temperature_file is not None:
        if logits is None or targets is None:
            raise RuntimeError("evaluation produced no logits for reliability analysis")
        bins = [float(value) for value in np.linspace(0.0, 1.0, 16)]
        if fit_calibrator:
            if role != "train_calibration" or dataset not in {"cityscapes", "bdd100k", "idd20k"}:
                raise ValueError(
                    "temperature fitting is allowed only on source-domain train_calibration"
                )
            fit = fit_temperature(logits, targets)
            temperature = fit.final_temperature
            fit_payload = fit.to_dict()
            fit_payload["scientific_evidence"] = True
            (output_dir / "temperature.json").write_text(
                canonical_json(fit_payload) + "\n", encoding="utf-8"
            )
        else:
            assert temperature_file is not None
            fit_payload = json.loads(temperature_file.read_text(encoding="utf-8"))
            temperature = float(fit_payload["final_temperature"])
        reliability = {
            "before": calibration_metrics(logits, targets, temperature=1.0, bin_edges=bins),
            "after": calibration_metrics(
                apply_temperature(logits, temperature),
                targets,
                temperature=1.0,
                bin_edges=bins,
            ),
            "temperature": temperature,
            "sampled_pixel_count": int(targets.size),
            "confidence_entropy_before": confidence_entropy_summary(logits),
            "confidence_entropy_after": confidence_entropy_summary(logits, temperature=temperature),
        }
        reliability["before"]["scientific_evidence"] = True
        reliability["after"]["scientific_evidence"] = True
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_evaluation_summary",
        "dataset": dataset,
        "role": role,
        "condition": condition,
        "model": cfg.get("edgeguard_metadata", {}).get("model"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": metrics,
        "classwise_metrics": classwise,
        "rare_class_mIoU": rare_class_miou,
        "reliability": reliability,
        "scientific_evidence": dataset != "cityscapes_stress",
        "external_domain_shift_evidence": dataset in {"acdc", "wilddash2", "muses", "kitti"},
        "dataset_manifest_sha256": (
            sha256_file(dataset_manifest) if dataset_manifest is not None else None
        ),
        "official_val_used_for_selection": False,
    }
    shift_summary = {
        "schema_version": "1.0",
        "record_type": "edgeguard_frame_uncertainty_summaries",
        "dataset": dataset,
        "role": role,
        "condition": condition,
        "checkpoint_sha256": sha256_file(checkpoint),
        "temperature": reporting_temperature,
        "frames": frame_summaries,
        "pixel_anomaly_segmentation": False,
    }
    (output_dir / "frame_uncertainty.json").write_text(
        canonical_json(shift_summary) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    append_run_ledger(
        output_dir.parent / "run_ledger.jsonl", operation="semantic_evaluation", result=result
    )
    return result


def default_mmseg_root(value: Path | None) -> Path:
    """Resolve an explicit path or the documented MMSEG_ROOT runtime variable."""
    if value is not None:
        return value.resolve()
    configured = os.environ.get("MMSEG_ROOT")
    if configured:
        return Path(configured).resolve()
    raise ValueError("provide --mmseg-root or set MMSEG_ROOT")
