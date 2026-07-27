"""Validate the semantic laboratory or execute its synthetic CUDA stack probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.training.config import (
    load_semantic_common_config,
    load_semantic_framework_config,
    load_semantic_model_config,
    load_semantic_model_suite,
)
from edgeguard.training.contracts import (
    CheckpointMetadata,
    DatasetIdentity,
    ExperimentRegistryRecord,
    ProjectStatus,
    SemanticExperimentContract,
    SemanticModelConfig,
    ValidationIntervalRecord,
)
from edgeguard.training.data import load_policy_selected_cityscapes_split
from edgeguard.training.identity import build_experiment_contract, validate_resume_identity
from edgeguard.training.logits import validate_native_logits_tensor
from edgeguard.training.registry import append_registry

_RUNTIME_DATASET_ROOT: Path | None = None
_RUNTIME_TRAINING_SAMPLES: tuple[Any, ...] = ()
_RUNTIME_RECOVERY_SYNC_ROOT: Path | None = None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_clean_checkout(repo: Path, expected_commit: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("project checkout does not match the reviewed commit")
    if _git(repo, "status", "--porcelain=v1"):
        raise ValueError("stack probe requires a clean project checkout")


def _strip_pretrained(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "Pretrained":
            value.clear()
        if "pretrained" in value:
            value["pretrained"] = None
        for nested in value.values():
            _strip_pretrained(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_pretrained(nested)


def _replace_pretrained(value: Any, checkpoint: Path) -> int:
    replacements = 0
    if isinstance(value, dict):
        if value.get("type") == "Pretrained":
            value["checkpoint"] = str(checkpoint)
            replacements += 1
        if value.get("pretrained") is not None:
            value["pretrained"] = str(checkpoint)
            replacements += 1
        for nested in value.values():
            replacements += _replace_pretrained(nested, checkpoint)
    elif isinstance(value, list):
        for nested in value:
            replacements += _replace_pretrained(nested, checkpoint)
    return replacements


def _environment(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("semantic stack probe requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "schema_version": "1.0",
        "record_type": "semantic_stack_environment",
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_version": importlib.metadata.version("torch"),
        "mmsegmentation_version": importlib.metadata.version("mmsegmentation"),
        "mmengine_version": importlib.metadata.version("mmengine"),
        "mmcv_version": importlib.metadata.version("mmcv"),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_bytes": int(properties.total_memory),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "precision_mode": "fp32",
        "device_batch": 1,
        "effective_global_batch": 1,
        "gradient_accumulation": 1,
        "dataloader": {"workers": 0, "prefetch": None, "pinned_memory": False},
        "dataset_source": "synthetic-semantic-stack-v1",
        "scientific_evidence": False,
    }


def _model_from_official_config(
    model_spec: SemanticModelConfig, mmseg_checkout: Path, *, torch: Any
) -> Any:
    from mmengine.config import Config
    from mmseg.registry import MODELS

    config_path = mmseg_checkout / model_spec.mmseg_config_relative_path
    if not config_path.is_file():
        raise ValueError(f"official MMSeg config is missing: {config_path.name}")
    config = Config.fromfile(config_path)
    model_config = config.model
    _strip_pretrained(model_config)
    model = MODELS.build(model_config)
    model.init_weights()
    return model.to(torch.device("cuda"))


def _probe_model(model: Any, model_spec: SemanticModelConfig, *, torch: Any) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    inputs = torch.randn((1, 3, 128, 256), device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(inputs, mode="tensor")
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("framework returned an empty native-logit sequence")
        output = output[0]
    native_shape = validate_native_logits_tensor(output, is_tensor=torch.is_tensor)
    if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
        raise ValueError("native logits must be finite float32 during the FP32 probe")
    synthetic_loss = output.square().mean()
    if not bool(torch.isfinite(synthetic_loss)):
        raise ValueError("synthetic stack-probe scalar is non-finite")
    synthetic_loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    aligned = torch.nn.functional.interpolate(
        output,
        size=(128, 256),
        mode="bilinear",
        align_corners=model_spec.logits.align_corners,
    )
    aligned_shape = validate_native_logits_tensor(aligned, is_tensor=torch.is_tensor)
    return {
        "experiment_id": model_spec.experiment_id,
        "model_family": model_spec.model_family.value,
        "initialization": "random_stack_probe_no_download",
        "native_logits_shape": list(native_shape),
        "native_logits_kind": "direct_pre_softmax_model_output",
        "aligned_logits_shape": list(aligned_shape),
        "alignment_mode": "bilinear",
        "align_corners": model_spec.logits.align_corners,
        "forward_backward_seconds": elapsed,
        "images_per_second": 1.0 / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "synthetic_scalar_probe_loss": float(synthetic_loss.detach().cpu()),
        "scientific_accuracy_evidence": False,
    }


def _checkpoint_round_trip(
    model: Any,
    contract: SemanticExperimentContract,
    output_dir: Path,
    *,
    torch: Any,
) -> CheckpointMetadata:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    optimizer.step()
    scheduler.step()
    metadata = CheckpointMetadata(
        experiment_id=contract.experiment_id,
        config_sha256=contract.config_sha256,
        experiment_fingerprint=contract.experiment_fingerprint,
        dataset_manifest_sha256=contract.dataset.dataset_manifest_sha256,
        split_manifest_sha256=contract.dataset.split_manifest_sha256,
        initialization_checkpoint_sha256=contract.initialization_checkpoint_sha256,
        model_family=contract.model_family,
        framework_identity_sha256=contract.framework_identity_sha256,
        git_commit=contract.git_commit,
        precision_mode=contract.precision_mode,
        seed=contract.training_seed,
        epoch=0,
        optimizer_step=1,
        best_metric=None,
        last_metric=None,
        contains_optimizer_state=True,
        contains_scheduler_state=True,
        contains_amp_scaler_state=False,
    )
    checkpoint_path = output_dir / "last.checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": None,
            "metadata": metadata.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded_metadata = CheckpointMetadata.model_validate(loaded["metadata"])
    validate_resume_identity(contract, loaded_metadata)
    model.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    scheduler.load_state_dict(loaded["scheduler"])
    return loaded_metadata


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _package(output_dir: Path, names: list[str]) -> Path:
    package = output_dir / "semantic-stack-probe-evidence.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (output_dir / name).read_bytes())
    return package


def validate_configs(config_root: Path) -> dict[str, Any]:
    """Validate all path-free documents without importing the training stack."""
    framework = load_semantic_framework_config(config_root / "framework_mmseg.yaml")
    common = load_semantic_common_config(config_root / "common_cityscapes.yaml")
    models = load_semantic_model_suite(config_root)
    return {
        "schema_version": "1.0",
        "record_type": "semantic_training_config_validation",
        "framework_commit": framework.commit,
        "framework_identity_sha256": sha256_payload(framework.model_dump(mode="json")),
        "common_config_sha256": sha256_payload(common.model_dump(mode="json")),
        "models": [
            {
                "experiment_id": model.experiment_id,
                "model_family": model.model_family.value,
                "config_sha256": sha256_payload(model.model_dump(mode="json")),
                "stack_probe_initialization": model.initialization.stack_probe,
                "project_training_initialization": model.initialization.project_training,
            }
            for model in models
        ],
        "scientific_evidence": False,
    }


def _register_manifest_dataset(torch: Any) -> str:
    """Register the one project-specific Cityscapes manifest dataset at runtime."""
    from mmseg.datasets import CityscapesDataset
    from mmseg.registry import DATASETS

    @DATASETS.register_module(force=True)
    class EdgeGuardCityscapesSelectedDataset(CityscapesDataset):
        def __init__(self, *, role: str, **kwargs: Any) -> None:
            self.edgeguard_role = role
            super().__init__(**kwargs)

        def load_data_list(self) -> list[dict[str, Any]]:
            if _RUNTIME_DATASET_ROOT is None:
                raise RuntimeError("runtime dataset root was not supplied")
            records = [
                sample for sample in _RUNTIME_TRAINING_SAMPLES if sample.role == self.edgeguard_role
            ]
            if not records:
                raise ValueError(f"policy-selected split has no samples for {self.edgeguard_role}")
            return [
                {
                    "img_path": str(_RUNTIME_DATASET_ROOT / sample.image_relative_path),
                    "seg_map_path": str(_RUNTIME_DATASET_ROOT / sample.train_id_relative_path),
                    "label_map": None,
                    "reduce_zero_label": False,
                    "seg_fields": [],
                }
                for sample in records
            ]

    del torch
    return EdgeGuardCityscapesSelectedDataset.__name__


def _validation_interval_record(
    *,
    epoch: int,
    optimizer_step: int,
    train_losses: list[float],
    metrics: dict[str, float | None],
    learning_rate: float,
) -> ValidationIntervalRecord:
    """Validate the complete metric row required at every selection interval."""
    if not train_losses:
        raise ValueError("validation interval has no recorded training loss")
    select_loss = metrics.get("train_select_loss")
    select_miou = metrics.get("mIoU")
    if select_loss is None or select_miou is None:
        raise ValueError("validation interval is missing train_select loss or mIoU")
    per_class = tuple(metrics.get(f"edgeguard_iou_{class_id:02d}") for class_id in range(19))
    train_loss = sum(train_losses) / len(train_losses)
    return ValidationIntervalRecord(
        epoch=epoch,
        optimizer_step=optimizer_step,
        train_loss=train_loss,
        train_select_loss=select_loss,
        train_select_miou=select_miou,
        per_class_iou=per_class,
        learning_rate=learning_rate,
        generalization_gap_inputs={
            "train_loss": train_loss,
            "train_select_loss": select_loss,
            "train_select_miou": select_miou,
        },
    )


def _register_selection_metric() -> str:
    """Register IoU reporting that retains all 19 per-class values."""
    from mmseg.evaluation.metrics import IoUMetric
    from mmseg.registry import METRICS

    @METRICS.register_module(force=True)
    class EdgeGuardIoUMetric(IoUMetric):
        def compute_metrics(self, results: list[tuple[Any, Any, Any, Any]]) -> dict[str, float]:
            metrics = dict(super().compute_metrics(results))
            if not results:
                raise ValueError("selection evaluation produced no area statistics")
            total_intersection = sum(result[0] for result in results)
            total_union = sum(result[1] for result in results)
            for class_id in range(19):
                union = int(total_union[class_id].item())
                metrics[f"edgeguard_iou_{class_id:02d}"] = (
                    100.0 * float(total_intersection[class_id].item()) / union if union else None
                )
            return metrics

    return EdgeGuardIoUMetric.__name__


def _register_loss_val_loop(torch: Any) -> str:
    """Register a validation loop that measures selection loss and IoU together."""
    from mmengine.registry import LOOPS
    from mmengine.runner.amp import autocast
    from mmengine.runner.loops import ValLoop

    @LOOPS.register_module(force=True)
    class EdgeGuardLossValLoop(ValLoop):
        @torch.no_grad()
        def run(self) -> dict[str, float]:
            self.runner.call_hook("before_val")
            self.runner.call_hook("before_val_epoch")
            self.runner.model.eval()
            self.edgeguard_loss_sum = 0.0
            self.edgeguard_sample_count = 0
            for index, data_batch in enumerate(self.dataloader):
                self.run_iter(index, data_batch)
            metrics = dict(self.evaluator.evaluate(len(self.dataloader.dataset)))
            if self.edgeguard_sample_count == 0:
                raise ValueError("selection evaluation produced no loss samples")
            metrics["train_select_loss"] = self.edgeguard_loss_sum / self.edgeguard_sample_count
            self.runner.call_hook("after_val_epoch", metrics=metrics)
            self.runner.call_hook("after_val")
            return metrics

        @torch.no_grad()
        def run_iter(self, idx: int, data_batch: Any) -> None:
            self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
            with autocast(enabled=self.fp16):
                outputs = self.runner.model.val_step(data_batch)
                model = (
                    self.runner.model.module
                    if hasattr(self.runner.model, "module")
                    else self.runner.model
                )
                processed = model.data_preprocessor(data_batch, False)
                losses = model._run_forward(processed, mode="loss")
                parsed_loss, _log_vars = model.parse_losses(losses)
            batch_size = len(outputs)
            self.edgeguard_loss_sum += float(parsed_loss.detach().cpu()) * batch_size
            self.edgeguard_sample_count += batch_size
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
            self.runner.call_hook(
                "after_val_iter",
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs,
            )

    return EdgeGuardLossValLoop.__name__


def _register_checkpoint_metadata_hook() -> str:
    """Persist identity-protected progress beside MMEngine recovery checkpoints."""
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    @HOOKS.register_module(force=True)
    class EdgeGuardCheckpointMetadataHook(Hook):
        def __init__(self, metadata: dict[str, Any]) -> None:
            self.metadata = CheckpointMetadata.model_validate(metadata)
            self.train_losses: list[float] = []

        def after_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
            outputs: dict[str, Any] | None = None,
        ) -> None:
            del runner, batch_idx, data_batch
            loss = (outputs or {}).get("loss")
            if loss is None:
                raise ValueError("training iteration did not expose a loss")
            if hasattr(loss, "detach"):
                loss = loss.detach().cpu()
            self.train_losses.append(float(loss))

        def _persist(
            self,
            runner: Any,
            *,
            last_metric: float | None = None,
            best_metric: float | None = None,
        ) -> None:
            self.metadata = self.metadata.model_copy(
                update={
                    "epoch": int(runner.epoch + 1),
                    "optimizer_step": int(runner.iter + 1),
                    "last_metric": last_metric,
                    "best_metric": best_metric,
                }
            )
            _write_json(
                Path(runner.work_dir) / "checkpoint_metadata.json",
                self.metadata.model_dump(mode="json"),
            )

        def after_train_epoch(self, runner: Any) -> None:
            self._persist(runner)
            if _RUNTIME_RECOVERY_SYNC_ROOT is not None:
                _atomic_sync_checkpoint(Path(runner.work_dir), _RUNTIME_RECOVERY_SYNC_ROOT)

        def after_val_epoch(
            self,
            runner: Any,
            metrics: dict[str, float] | None = None,
        ) -> None:
            metric_values = metrics or {}
            last_metric = metric_values.get("mIoU")
            previous = self.metadata.best_metric
            best_metric = (
                last_metric
                if previous is None
                else max(previous, last_metric)
                if last_metric is not None
                else previous
            )
            self._persist(runner, last_metric=last_metric, best_metric=best_metric)
            learning_rates = runner.optim_wrapper.get_lr()
            flat_rates = [float(value) for values in learning_rates.values() for value in values]
            if not flat_rates:
                raise ValueError("optimizer did not expose a learning rate")
            record = _validation_interval_record(
                epoch=int(runner.epoch + 1),
                optimizer_step=int(runner.iter + 1),
                train_losses=self.train_losses,
                metrics=metric_values,
                learning_rate=flat_rates[0],
            )
            with (Path(runner.work_dir) / "validation_intervals.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(canonical_json(record.model_dump(mode="json")) + "\n")
            self.train_losses.clear()

    return EdgeGuardCheckpointMetadataHook.__name__


def _atomic_sync_checkpoint(work_dir: Path, sync_dir: Path) -> dict[str, Any]:
    """Atomically sync the current recovery checkpoint and verify its SHA-256."""
    marker = work_dir / "last_checkpoint"
    if not marker.is_file():
        raise ValueError("MMEngine last_checkpoint marker is missing")
    recorded = Path(marker.read_text(encoding="utf-8").strip())
    checkpoint = recorded if recorded.is_absolute() else work_dir / recorded
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != work_dir.resolve() or not checkpoint.is_file():
        raise ValueError("last checkpoint must be a regular file inside the active work dir")
    metadata = work_dir / "checkpoint_metadata.json"
    if not metadata.is_file():
        raise ValueError("checkpoint metadata is missing")
    sync_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for source in (checkpoint, metadata):
        incoming = sync_dir / f".{source.name}.incoming"
        destination = sync_dir / source.name
        if incoming.exists():
            raise ValueError("recovery sync has a stale incoming file")
        shutil.copy2(source, incoming)
        expected_sha = sha256_file(source)
        if sha256_file(incoming) != expected_sha:
            raise RuntimeError("recovery checkpoint sync verification failed")
        os.replace(incoming, destination)
        files.append(
            {
                "filename": source.name,
                "byte_size": destination.stat().st_size,
                "sha256": expected_sha,
            }
        )
    receipt = {
        "schema_version": "1.0",
        "record_type": "semantic_recovery_sync_receipt",
        "files": files,
    }
    _write_json(sync_dir / "recovery_sync_receipt.json", receipt)
    return receipt


def _training_pipelines(
    model_spec: SemanticModelConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_pipeline: list[dict[str, Any]] = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations"},
        {
            "type": "RandomResize",
            "scale": (2048, 1024),
            "ratio_range": (0.5, 2.0),
            "keep_ratio": True,
        },
        {"type": "RandomCrop", "crop_size": (512, 1024), "cat_max_ratio": 0.75},
        {"type": "RandomFlip", "prob": 0.5},
        {
            "type": "PhotoMetricDistortion",
            "brightness_delta": 20,
            "contrast_range": (0.8, 1.2),
            "saturation_range": (0.8, 1.2),
            "hue_delta": 10,
        },
    ]
    if model_spec.model_family.value == "pidnet_s":
        train_pipeline.append({"type": "GenerateEdge", "edge_width": 4})
    train_pipeline.append({"type": "PackSegInputs"})
    validation_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations"},
        {"type": "Resize", "scale": (1024, 512), "keep_ratio": False},
    ]
    if model_spec.model_family.value == "pidnet_s":
        validation_pipeline.append({"type": "GenerateEdge", "edge_width": 4})
    validation_pipeline.append({"type": "PackSegInputs"})
    return train_pipeline, validation_pipeline


def _resolved_runner_config(
    model_spec: SemanticModelConfig,
    common: Any,
    mmseg_checkout: Path,
    output_dir: Path,
    *,
    torch: Any,
    precision: str,
    initialization_checkpoint: Path | None,
    checkpoint_metadata: CheckpointMetadata,
) -> Any:
    from mmengine.config import Config

    config = Config.fromfile(mmseg_checkout / model_spec.mmseg_config_relative_path)
    if model_spec.initialization.project_training == "random":
        _strip_pretrained(config.model)
    else:
        source = model_spec.initialization.source
        if source.status != "resolved" or initialization_checkpoint is None:
            raise ValueError(
                "pretrained project training is blocked on exact human-approved source"
            )
        if source.filename_or_model_id != initialization_checkpoint.name:
            raise ValueError("initialization checkpoint filename mismatch")
        if sha256_file(initialization_checkpoint) != source.sha256:
            raise ValueError("initialization checkpoint SHA-256 mismatch")
        if _replace_pretrained(config.model, initialization_checkpoint) == 0:
            raise ValueError(
                "official model config has no explicit pretrained initialization point"
            )
    config.model["decode_head"]["num_classes"] = 19
    if "data_preprocessor" in config.model:
        config.model["data_preprocessor"]["size"] = (512, 1024)
        config.model["data_preprocessor"]["seg_pad_val"] = 255
    train_pipeline, validation_pipeline = _training_pipelines(model_spec)
    dataset_type = _register_manifest_dataset(torch)
    config.train_dataloader = {
        "batch_size": common.device_batch,
        "num_workers": 0,
        "persistent_workers": False,
        "sampler": {"type": "DefaultSampler", "shuffle": True},
        "dataset": {"type": dataset_type, "role": "train_fit", "pipeline": train_pipeline},
    }
    config.val_dataloader = {
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "sampler": {"type": "DefaultSampler", "shuffle": False},
        "dataset": {
            "type": dataset_type,
            "role": "train_select",
            "pipeline": validation_pipeline,
        },
    }
    config.test_dataloader = config.val_dataloader
    optimizer: dict[str, Any] = {
        "type": common.optimizer.name,
        "lr": common.optimizer.learning_rate,
        "weight_decay": common.optimizer.weight_decay,
    }
    if common.optimizer.momentum is not None:
        optimizer["momentum"] = common.optimizer.momentum
    wrapper_type = "AmpOptimWrapper" if precision != "fp32" else "OptimWrapper"
    config.optim_wrapper = {
        "type": wrapper_type,
        "optimizer": optimizer,
        "accumulative_counts": common.gradient_accumulation,
    }
    if precision == "bf16":
        config.optim_wrapper["dtype"] = "bfloat16"
    elif precision == "fp16":
        config.optim_wrapper["loss_scale"] = "dynamic"
    config.param_scheduler = [
        {
            "type": "PolyLR",
            "eta_min": 0.0,
            "power": common.scheduler.power,
            "begin": 0,
            "end": common.training_epochs,
            "by_epoch": True,
        }
    ]
    config.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": common.training_epochs,
        "val_interval": common.checkpoint.validation_interval_epochs,
    }
    val_loop_type = _register_loss_val_loop(torch)
    selection_metric_type = _register_selection_metric()
    config.val_cfg = {"type": val_loop_type}
    config.test_cfg = {"type": "TestLoop"}
    config.val_evaluator = {"type": selection_metric_type, "iou_metrics": ["mIoU"]}
    config.test_evaluator = config.val_evaluator
    config.default_hooks["checkpoint"] = {
        "type": "CheckpointHook",
        "by_epoch": True,
        "interval": 1,
        "save_last": True,
        "save_best": "mIoU",
        "rule": "greater",
        "max_keep_ckpts": 3,
    }
    config.randomness = {"seed": common.seed, "deterministic": False}
    config.work_dir = str(output_dir)
    config.launcher = "none"
    hook_type = _register_checkpoint_metadata_hook()
    config.custom_hooks = [
        {
            "type": hook_type,
            "metadata": checkpoint_metadata.model_dump(mode="json"),
            "priority": "LOW",
        }
    ]
    return config


def run_real_training(
    *,
    config_root: Path,
    model_config_path: Path,
    mmseg_checkout: Path,
    project_root: Path,
    project_commit: str,
    dataset_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    precision: str,
    initialization_checkpoint: Path | None,
    resume: bool,
    recovery_sync_dir: Path | None,
) -> dict[str, Any]:
    """Run one policy-selected split through MMEngine's interruption-safe runner."""
    global _RUNTIME_DATASET_ROOT, _RUNTIME_RECOVERY_SYNC_ROOT, _RUNTIME_TRAINING_SAMPLES

    _verify_clean_checkout(project_root, project_commit)
    if not dataset_root.is_dir():
        raise ValueError("runtime dataset root is missing")
    identity, samples = load_policy_selected_cityscapes_split(
        dataset_manifest_path, split_manifest_path
    )
    framework = load_semantic_framework_config(config_root / "framework_mmseg.yaml")
    common = load_semantic_common_config(config_root / "common_cityscapes.yaml")
    model_spec = load_semantic_model_config(model_config_path)
    if _git(mmseg_checkout, "rev-parse", "HEAD") != framework.commit:
        raise ValueError("MMSegmentation checkout commit mismatch")
    import torch
    from mmengine.runner import Runner
    from mmseg.utils import register_all_modules

    if not torch.cuda.is_available():
        raise RuntimeError("semantic training requires a CUDA runtime")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("selected CUDA runtime does not support BF16")
    environment = _environment(torch)
    environment["precision_mode"] = precision
    environment["device_batch"] = common.device_batch
    environment["effective_global_batch"] = common.effective_global_batch
    environment["gradient_accumulation"] = common.gradient_accumulation
    contract = build_experiment_contract(
        framework,
        common,
        model_spec,
        dataset=identity,
        git_commit=project_commit,
        environment=environment,
        precision_mode=precision,
        status=ProjectStatus.COLAB_MEASURED,
    )
    metadata_path = output_dir / "checkpoint_metadata.json"
    resume_metadata: CheckpointMetadata | None = None
    if resume:
        if not metadata_path.is_file():
            raise ValueError("resume requires existing checkpoint metadata")
        resume_metadata = CheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        validate_resume_identity(contract, resume_metadata)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("training output directory is non-empty; use exact resume")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    _RUNTIME_DATASET_ROOT = dataset_root
    _RUNTIME_TRAINING_SAMPLES = samples
    _RUNTIME_RECOVERY_SYNC_ROOT = recovery_sync_dir
    register_all_modules(init_default_scope=True)
    initial_metadata = resume_metadata or CheckpointMetadata(
        experiment_id=contract.experiment_id,
        config_sha256=contract.config_sha256,
        experiment_fingerprint=contract.experiment_fingerprint,
        dataset_manifest_sha256=identity.dataset_manifest_sha256,
        split_manifest_sha256=identity.split_manifest_sha256,
        initialization_checkpoint_sha256=contract.initialization_checkpoint_sha256,
        model_family=contract.model_family,
        framework_identity_sha256=contract.framework_identity_sha256,
        git_commit=contract.git_commit,
        precision_mode=contract.precision_mode,
        seed=contract.training_seed,
        epoch=0,
        optimizer_step=0,
        best_metric=None,
        last_metric=None,
        contains_optimizer_state=True,
        contains_scheduler_state=True,
        contains_amp_scaler_state=precision == "fp16",
    )
    runner_config = _resolved_runner_config(
        model_spec,
        common,
        mmseg_checkout,
        output_dir,
        torch=torch,
        precision=precision,
        initialization_checkpoint=initialization_checkpoint,
        checkpoint_metadata=initial_metadata,
    )
    runner_config.resume = resume
    _write_json(output_dir / "experiment_contract.json", contract.model_dump(mode="json"))
    if not resume:
        _write_json(metadata_path, initial_metadata.model_dump(mode="json"))
    runner = Runner.from_cfg(runner_config)
    runner.train()
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_training_completion",
        "experiment_id": contract.experiment_id,
        "status": "colab_measured",
        "config_sha256": contract.config_sha256,
        "experiment_fingerprint": contract.experiment_fingerprint,
        "dataset_manifest_sha256": identity.dataset_manifest_sha256,
        "split_manifest_sha256": identity.split_manifest_sha256,
        "model_family": model_spec.model_family.value,
        "precision_mode": precision,
        "artifact_root": f"experiments/segmentation/{contract.experiment_id}/",
        "scientific_interpretation": "pending_human_review",
    }
    _write_json(output_dir / "completion.json", result)
    return result


def run_stack_probe(
    config_root: Path,
    mmseg_checkout: Path,
    output_dir: Path,
    project_root: Path,
    project_commit: str,
) -> dict[str, Any]:
    """Run five synthetic CUDA probes and one exact checkpoint resume."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("stack-probe output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_clean_checkout(project_root, project_commit)
    framework = load_semantic_framework_config(config_root / "framework_mmseg.yaml")
    common = load_semantic_common_config(config_root / "common_cityscapes.yaml")
    models = load_semantic_model_suite(config_root)
    if _git(mmseg_checkout, "rev-parse", "HEAD") != framework.commit:
        raise ValueError("MMSegmentation checkout commit mismatch")

    import torch
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    environment = _environment(torch)
    dataset = DatasetIdentity(
        kind="synthetic_stack_fixture",
        synthetic_fixture_identity=common.synthetic_fixture_identity,
    )
    probes: list[dict[str, Any]] = []
    checkpoint_metadata: CheckpointMetadata | None = None
    registry_path = output_dir / "registry.jsonl"
    for index, model_spec in enumerate(models):
        contract = build_experiment_contract(
            framework,
            common,
            model_spec,
            dataset=dataset,
            git_commit=project_commit,
            environment=environment,
            status=ProjectStatus.COLAB_MEASURED,
        )
        model = _model_from_official_config(model_spec, mmseg_checkout, torch=torch)
        probe = _probe_model(model, model_spec, torch=torch)
        probes.append(probe)
        if index == 0:
            checkpoint_metadata = _checkpoint_round_trip(model, contract, output_dir, torch=torch)
        append_registry(
            registry_path,
            ExperimentRegistryRecord(
                experiment_id=contract.experiment_id,
                status=ProjectStatus.COLAB_MEASURED,
                config_sha256=contract.config_sha256,
                git_commit=project_commit,
                git_dirty=False,
                framework_identity_sha256=contract.framework_identity_sha256,
                dataset_manifest_sha256=None,
                split_manifest_sha256=None,
                initialization_checkpoint_sha256=None,
                seed=common.seed,
                runtime={"device": "cuda", "synthetic_stack_probe": True},
                final_metrics={},
                last_metrics={},
                artifact_paths=("experiments/segmentation/stack-probe/",),
                failure_summary=None,
            ),
        )
        del model
        torch.cuda.empty_cache()
    assert checkpoint_metadata is not None
    summary = {
        "schema_version": "1.0",
        "record_type": "semantic_stack_probe_summary",
        "run_family": "EGX-SEG-STACK-*",
        "project_commit": project_commit,
        "framework_commit": framework.commit,
        "model_count": len(probes),
        "models": probes,
        "checkpoint_resume_verified": True,
        "scientific_accuracy_evidence": False,
        "status": "colab_measured",
    }
    config_receipt = validate_configs(config_root)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "stack_probe_summary.json", summary)
    _write_json(output_dir / "config_receipt.json", config_receipt)
    _write_json(
        output_dir / "checkpoint_metadata.json", checkpoint_metadata.model_dump(mode="json")
    )
    package_names = [
        "environment.json",
        "stack_probe_summary.json",
        "config_receipt.json",
        "checkpoint_metadata.json",
        "registry.jsonl",
    ]
    package = _package(output_dir, package_names)
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_stack_probe_completion",
        "status": "colab_measured",
        "scientific_accuracy_evidence": False,
        "project_commit": project_commit,
        "framework_commit": framework.commit,
        "model_count": len(probes),
        "checkpoint_resume_verified": True,
        "evidence_package": package.name,
        "evidence_package_sha256": sha256_file(package),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "completion.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-configs")
    validate.add_argument("--config-root", type=Path, required=True)
    probe = subparsers.add_parser("stack-probe")
    probe.add_argument("--config-root", type=Path, required=True)
    probe.add_argument("--mmseg-checkout", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--project-root", type=Path, required=True)
    probe.add_argument("--project-commit", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config-root", type=Path, required=True)
    train.add_argument("--model-config", type=Path, required=True)
    train.add_argument("--mmseg-checkout", type=Path, required=True)
    train.add_argument("--project-root", type=Path, required=True)
    train.add_argument("--project-commit", required=True)
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--dataset-manifest", type=Path, required=True)
    train.add_argument("--split-policy-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    train.add_argument("--initialization-checkpoint", type=Path)
    train.add_argument("--recovery-sync-dir", type=Path)
    train.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-configs":
        result = validate_configs(args.config_root)
    elif args.command == "stack-probe":
        result = run_stack_probe(
            args.config_root,
            args.mmseg_checkout,
            args.output_dir,
            args.project_root,
            args.project_commit,
        )
    else:
        result = run_real_training(
            config_root=args.config_root,
            model_config_path=args.model_config,
            mmseg_checkout=args.mmseg_checkout,
            project_root=args.project_root,
            project_commit=args.project_commit,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            split_manifest_path=args.split_policy_manifest,
            output_dir=args.output_dir,
            precision=args.precision,
            initialization_checkpoint=args.initialization_checkpoint,
            resume=args.resume,
            recovery_sync_dir=args.recovery_sync_dir,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
