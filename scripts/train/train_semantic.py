"""Validate the semantic laboratory or execute its synthetic CUDA stack probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import (
    LongRunStatus,
    append_jsonl,
    ensure_disk_space,
    require_finite,
    utc_now,
)
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
from edgeguard.training.fractions import build_train_fit_fraction
from edgeguard.training.identity import build_experiment_contract, validate_resume_identity
from edgeguard.training.logits import validate_native_logits_tensor
from edgeguard.training.registry import append_registry

_RUNTIME_DATASET_ROOT: Path | None = None
_RUNTIME_TRAINING_SAMPLES: tuple[Any, ...] = ()
_RUNTIME_RECOVERY_SYNC_ROOT: Path | None = None
_RUNTIME_STATUS: LongRunStatus | None = None


def _distribution_version(*names: str) -> tuple[str, str]:
    for name in names:
        try:
            return name, importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError(f"required distribution is unavailable: {', '.join(names)}")


def _install_mmcv_lite_ops_guard() -> bool:
    """Permit pure-model imports while failing closed if a compiled MMCV op is used."""
    try:
        importlib.metadata.version("mmcv-lite")
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        import mmcv._ext  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        extension = types.ModuleType("mmcv._ext")

        def unavailable(name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)

            def fail(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    f"selected model attempted unavailable compiled MMCV operation: {name}"
                )

            return fail

        extension.__getattr__ = unavailable  # type: ignore[attr-defined]
        sys.modules["mmcv._ext"] = extension
        return True
    return False


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_clean_checkout(repo: Path, expected_commit: str, *, allow_dirty: bool = False) -> None:
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("project checkout does not match the reviewed commit")
    if not allow_dirty and _git(repo, "status", "--porcelain=v1"):
        raise ValueError("stack probe requires a clean project checkout")


def _strip_pretrained(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in tuple(value.items()):
            if key == "init_cfg" and isinstance(nested, dict):
                if nested.get("type") == "Pretrained":
                    value[key] = None
                    continue
            if key == "pretrained":
                value[key] = None
                continue
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


def _environment(torch: Any, *, device_name: str = "cuda") -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("semantic stack probe requires an available CUDA device")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("semantic stack probe device must be cpu or cuda")
    properties = torch.cuda.get_device_properties(0) if device_name == "cuda" else None
    capability = torch.cuda.get_device_capability(0) if device_name == "cuda" else None
    mmcv_distribution, mmcv_version = _distribution_version("mmcv", "mmcv-lite")
    return {
        "schema_version": "1.0",
        "record_type": "semantic_stack_environment",
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_version": importlib.metadata.version("torch"),
        "mmsegmentation_version": importlib.metadata.version("mmsegmentation"),
        "mmengine_version": importlib.metadata.version("mmengine"),
        "mmcv_distribution": mmcv_distribution,
        "mmcv_version": mmcv_version,
        "device": device_name,
        "cuda_version": torch.version.cuda if device_name == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(0) if device_name == "cuda" else None,
        "vram_bytes": int(properties.total_memory) if properties is not None else None,
        "compute_capability": (
            [int(capability[0]), int(capability[1])] if capability is not None else None
        ),
        "precision_mode": "fp32",
        "device_batch": 1,
        "effective_global_batch": 1,
        "gradient_accumulation": 1,
        "dataloader": {"workers": 0, "prefetch": None, "pinned_memory": False},
        "dataset_source": "synthetic-semantic-stack-v1",
        "scientific_evidence": False,
    }


def _model_from_official_config(
    model_spec: SemanticModelConfig,
    mmseg_checkout: Path,
    *,
    torch: Any,
    device_name: str,
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
    return model.to(torch.device(device_name))


def _probe_model(
    model: Any,
    model_spec: SemanticModelConfig,
    *,
    torch: Any,
    device_name: str,
) -> dict[str, Any]:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    model.zero_grad(set_to_none=True)
    inputs = torch.randn((1, 3, 128, 256), device=device_name, dtype=torch.float32)
    if device_name == "cuda":
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
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    aligned = torch.nn.functional.interpolate(
        output,
        size=(128, 256),
        mode="bilinear",
        align_corners=model_spec.logits.align_corners,
    )
    aligned_shape = validate_native_logits_tensor(aligned, is_tensor=torch.is_tensor)
    fp16_finite_verified = False
    if device_name == "cuda":
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            fp16_output = model(inputs, mode="tensor")
            if isinstance(fp16_output, (list, tuple)):
                if not fp16_output:
                    raise ValueError("framework returned an empty FP16 native-logit sequence")
                fp16_output = fp16_output[0]
            validate_native_logits_tensor(fp16_output, is_tensor=torch.is_tensor)
            fp16_loss = fp16_output.float().square().mean()
        if not bool(torch.isfinite(fp16_output).all()) or not bool(torch.isfinite(fp16_loss)):
            raise ValueError("AMP/FP16 stack-probe output or loss is non-finite")
        scaler.scale(fp16_loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise ValueError("AMP/FP16 stack-probe gradient is missing or non-finite")
        scaler.step(optimizer)
        scaler.update()
        fp16_finite_verified = True
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
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device_name == "cuda" else None
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device_name == "cuda" else None
        ),
        "synthetic_scalar_probe_loss": float(synthetic_loss.detach().cpu()),
        "fp16_finite_verified": fp16_finite_verified,
        "batch_norm_mode": "frozen_eval_for_batch_size_one_probe",
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


def _register_checkpoint_metadata_hook(
    *, model_name: str, total_steps: int, device_batch: int
) -> str:
    """Persist identity-protected progress beside MMEngine recovery checkpoints."""
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    @HOOKS.register_module(force=True)
    class EdgeGuardCheckpointMetadataHook(Hook):
        def __init__(self, metadata: dict[str, Any]) -> None:
            self.metadata = CheckpointMetadata.model_validate(metadata)
            self.train_losses: list[float] = []
            self.iteration_started = time.perf_counter()
            self.run_started = time.perf_counter()
            self.last_sync = time.monotonic()
            self.last_sync_utc: str | None = None

        def before_train_iter(self, runner: Any, batch_idx: int, data_batch: Any = None) -> None:
            del runner, batch_idx, data_batch
            self.iteration_started = time.perf_counter()

        def after_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
            outputs: dict[str, Any] | None = None,
        ) -> None:
            del batch_idx, data_batch
            loss = (outputs or {}).get("loss")
            if loss is None:
                raise ValueError("training iteration did not expose a loss")
            if hasattr(loss, "detach"):
                loss = loss.detach().cpu()
            loss_value = require_finite(float(loss), "training loss")
            self.train_losses.append(loss_value)
            step = int(runner.iter + 1)
            step_time = time.perf_counter() - self.iteration_started
            if step_time > 300:
                raise TimeoutError("training iteration exceeded the dataloader-stall limit")
            gradient_norm: float | None = None
            try:
                gradient_scalar = runner.message_hub.get_scalar("train/grad_norm")
                gradient_norm = require_finite(float(gradient_scalar.current()), "gradient norm")
            except (KeyError, RuntimeError, ValueError):
                gradient_norm = None
            squared_norm = 0.0
            gradient_seen = False
            model = runner.model.module if hasattr(runner.model, "module") else runner.model
            for parameter in model.parameters():
                if parameter.grad is not None:
                    value = float(parameter.grad.detach().float().norm().cpu())
                    require_finite(value, "gradient norm component")
                    squared_norm += value * value
                    gradient_seen = True
            if gradient_seen:
                gradient_norm = require_finite(squared_norm**0.5, "gradient norm")
            if step % 25 == 0 or step == 1 or step == total_steps:
                ensure_disk_space(Path(runner.work_dir), 0, reserve_bytes=512 * 1024**2)
                learning_rates = runner.optim_wrapper.get_lr()
                flat_rates = [
                    float(value) for values in learning_rates.values() for value in values
                ]
                elapsed = max(time.perf_counter() - self.run_started, 1e-9)
                speed = step * device_batch / elapsed
                torch_module = __import__("torch")
                cuda_available = bool(torch_module.cuda.is_available())
                data_time: float | None = None
                try:
                    data_time = require_finite(
                        float(runner.message_hub.get_scalar("train/data_time").current()),
                        "data time",
                    )
                except (KeyError, RuntimeError, ValueError):
                    data_time = None
                record = {
                    "schema_version": "1.0",
                    "record_type": "semantic_training_progress",
                    "model": model_name,
                    "epoch": int(runner.epoch + 1),
                    "optimizer_step": step,
                    "completed": step,
                    "total": total_steps,
                    "train_loss": loss_value,
                    "learning_rate": flat_rates[0] if flat_rates else None,
                    "data_time_seconds": data_time,
                    "step_time_seconds": step_time,
                    "images_per_second": device_batch / max(step_time, 1e-9),
                    "eta_seconds": max(0, total_steps - step) / max(speed / device_batch, 1e-9),
                    "gradient_norm": gradient_norm,
                    "gpu_allocated_bytes": int(
                        torch_module.cuda.memory_allocated() if cuda_available else 0
                    ),
                    "gpu_reserved_bytes": int(
                        torch_module.cuda.memory_reserved() if cuda_available else 0
                    ),
                    "last_recovery_sync_utc": self.last_sync_utc,
                }
                append_jsonl(Path(runner.work_dir) / "metrics.jsonl", record)
                print(canonical_json(record), flush=True)
                if _RUNTIME_STATUS is not None:
                    _RUNTIME_STATUS.update(
                        phase=f"train-{model_name}",
                        completed=step,
                        total=total_steps,
                        speed_per_second=speed / device_batch,
                    )
            sync_due = step % 500 == 0 or time.monotonic() - self.last_sync >= 600
            if sync_due and _RUNTIME_RECOVERY_SYNC_ROOT is not None:
                self._persist(runner)
                filename = f"recovery_step_{step:07d}.pth"
                runner.save_checkpoint(
                    str(runner.work_dir),
                    filename,
                    save_optimizer=True,
                    save_param_scheduler=True,
                    meta={"edgeguard_optimizer_step": step},
                )
                (Path(runner.work_dir) / "last_checkpoint").write_text(
                    filename + "\n", encoding="utf-8"
                )
                _atomic_sync_checkpoint(Path(runner.work_dir), _RUNTIME_RECOVERY_SYNC_ROOT)
                self.last_sync = time.monotonic()
                self.last_sync_utc = utc_now()
                if _RUNTIME_STATUS is not None:
                    _RUNTIME_STATUS.update(last_checkpoint=filename, force=True)

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
            append_jsonl(
                Path(runner.work_dir) / "validation_intervals.jsonl",
                record.model_dump(mode="json"),
            )
            print(canonical_json(record.model_dump(mode="json")), flush=True)
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


def _restore_recovery_checkpoint(
    recovery_dir: Path, output_dir: Path, contract: SemanticExperimentContract
) -> CheckpointMetadata:
    """Restore one SHA-verified compatible Drive recovery into ephemeral storage."""
    receipt_path = recovery_dir / "recovery_sync_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("recovery directory has no sync receipt")
    import json

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("recovery sync receipt is malformed")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name: str | None = None
    for record in files:
        name = record.get("filename") if isinstance(record, dict) else None
        expected_sha = record.get("sha256") if isinstance(record, dict) else None
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected_sha, str)
        ):
            raise ValueError("recovery sync receipt contains an unsafe file identity")
        source = recovery_dir / name
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError("recovery file is missing or corrupt")
        destination = output_dir / name
        shutil.copy2(source, destination)
        if sha256_file(destination) != expected_sha:
            raise RuntimeError("restored recovery file failed SHA-256 verification")
        if name.endswith(".pth"):
            checkpoint_name = name
    if checkpoint_name is None:
        raise ValueError("recovery receipt does not identify a checkpoint")
    metadata_path = output_dir / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("recovery receipt does not include checkpoint metadata")
    metadata = CheckpointMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    validate_resume_identity(contract, metadata)
    (output_dir / "last_checkpoint").write_text(checkpoint_name + "\n", encoding="utf-8")
    return metadata


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
    max_optimizer_steps: int | None = None,
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
        "clip_grad": {
            "max_norm": float("inf"),
            "norm_type": 2.0,
            "error_if_nonfinite": True,
        },
    }
    if precision == "bf16":
        config.optim_wrapper["dtype"] = "bfloat16"
    elif precision == "fp16":
        config.optim_wrapper["loss_scale"] = "dynamic"
    scheduler_end = max_optimizer_steps or common.training_epochs
    config.param_scheduler = [
        {
            "type": "PolyLR",
            "eta_min": 0.0,
            "power": common.scheduler.power,
            "begin": 0,
            "end": scheduler_end,
            "by_epoch": max_optimizer_steps is None,
        }
    ]
    config.train_cfg = (
        {
            "type": "IterBasedTrainLoop",
            "max_iters": max_optimizer_steps,
            "val_interval": max_optimizer_steps,
        }
        if max_optimizer_steps is not None
        else {
            "type": "EpochBasedTrainLoop",
            "max_epochs": common.training_epochs,
            "val_interval": common.checkpoint.validation_interval_epochs,
        }
    )
    val_loop_type = _register_loss_val_loop(torch)
    selection_metric_type = _register_selection_metric()
    config.val_cfg = {"type": val_loop_type}
    config.test_cfg = {"type": "TestLoop"}
    config.val_evaluator = {"type": selection_metric_type, "iou_metrics": ["mIoU"]}
    config.test_evaluator = config.val_evaluator
    config.default_hooks["checkpoint"] = {
        "type": "CheckpointHook",
        "by_epoch": max_optimizer_steps is None,
        "interval": min(500, max_optimizer_steps) if max_optimizer_steps else 1,
        "save_last": True,
        "save_best": "mIoU",
        "rule": "greater",
        "max_keep_ckpts": 3,
    }
    config.randomness = {"seed": common.seed, "deterministic": False}
    config.work_dir = str(output_dir)
    config.launcher = "none"
    config.default_hooks["logger"] = {"type": "LoggerHook", "interval": 25}
    config.visualizer = {
        "type": "SegLocalVisualizer",
        "vis_backends": [{"type": "LocalVisBackend"}, {"type": "TensorboardVisBackend"}],
        "name": "visualizer",
    }
    hook_type = _register_checkpoint_metadata_hook(
        model_name=model_spec.model_family.value,
        total_steps=max_optimizer_steps or common.planned_optimizer_steps,
        device_batch=common.device_batch,
    )
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
    max_optimizer_steps: int | None = None,
    validation_subset_size: int | None = None,
    device_batch: int | None = None,
    gradient_accumulation: int | None = None,
    train_fit_fraction: float = 1.0,
    smoke_random_initialization: bool = False,
) -> dict[str, Any]:
    """Run one policy-selected split through MMEngine's interruption-safe runner."""
    global _RUNTIME_DATASET_ROOT, _RUNTIME_RECOVERY_SYNC_ROOT, _RUNTIME_STATUS
    global _RUNTIME_TRAINING_SAMPLES

    _verify_clean_checkout(project_root, project_commit)
    if not dataset_root.is_dir():
        raise ValueError("runtime dataset root is missing")
    identity, samples = load_policy_selected_cityscapes_split(
        dataset_manifest_path, split_manifest_path
    )
    framework = load_semantic_framework_config(config_root / "framework_mmseg.yaml")
    common = load_semantic_common_config(config_root / "common_cityscapes.yaml")
    model_spec = load_semantic_model_config(model_config_path)
    selected_batch = device_batch or common.device_batch
    selected_accumulation = gradient_accumulation or common.gradient_accumulation
    common = common.model_copy(
        update={
            "device_batch": selected_batch,
            "gradient_accumulation": selected_accumulation,
            "effective_global_batch": selected_batch * selected_accumulation,
            "planned_optimizer_steps": max_optimizer_steps or common.planned_optimizer_steps,
        }
    )
    if smoke_random_initialization:
        model_payload = model_spec.model_dump(mode="json")
        model_payload["initialization"] = {
            "stack_probe": "random",
            "project_training": "random",
            "source": {"status": "not_applicable"},
            "notes": "Random initialization for the bounded EG-SEG-002 training-path smoke.",
        }
        model_spec = SemanticModelConfig.model_validate(model_payload)
        initialization_checkpoint = None
    fraction_manifest = build_train_fit_fraction(
        samples,
        train_fit_fraction,
        seed=common.seed,
        split_manifest_sha256=identity.split_manifest_sha256 or "",
    )
    selected_ids = set(fraction_manifest["selected_sample_ids"])
    train_fit = tuple(
        sample
        for sample in samples
        if sample.role != "train_fit" or sample.sample_id in selected_ids
    )
    if validation_subset_size is not None:
        if validation_subset_size <= 0:
            raise ValueError("validation subset size must be positive")
        select_ids = {
            sample.sample_id
            for sample in sorted(
                (item for item in train_fit if item.role == "train_select"),
                key=lambda item: item.sample_id,
            )[:validation_subset_size]
        }
        train_fit = tuple(
            sample
            for sample in train_fit
            if sample.role != "train_select" or sample.sample_id in select_ids
        )
    if _git(mmseg_checkout, "rev-parse", "HEAD") != framework.commit:
        raise ValueError("MMSegmentation checkout commit mismatch")
    import torch
    from mmengine.runner import Runner

    _install_mmcv_lite_ops_guard()
    from mmseg.utils import register_all_modules

    if not torch.cuda.is_available():
        raise RuntimeError("semantic training requires a CUDA runtime")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("selected CUDA runtime does not support BF16")
    environment = _environment(torch, device_name="cuda")
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
    if not output_dir.exists() and recovery_sync_dir is not None and recovery_sync_dir.exists():
        resume_metadata = _restore_recovery_checkpoint(recovery_sync_dir, output_dir, contract)
        resume = True
    elif resume:
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
    _RUNTIME_TRAINING_SAMPLES = train_fit
    _RUNTIME_RECOVERY_SYNC_ROOT = recovery_sync_dir
    _RUNTIME_STATUS = LongRunStatus(output_dir / "run_status.json")
    _RUNTIME_STATUS.update(
        status="running",
        phase=f"train-{model_spec.model_family.value}",
        completed=resume_metadata.optimizer_step if resume_metadata else 0,
        total=max_optimizer_steps or common.planned_optimizer_steps,
        force=True,
    )
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
        max_optimizer_steps=max_optimizer_steps,
    )
    runner_config.resume = resume
    _write_json(output_dir / "experiment_contract.json", contract.model_dump(mode="json"))
    _write_json(output_dir / "train_fit_fraction.json", fraction_manifest)
    if not resume:
        _write_json(metadata_path, initial_metadata.model_dump(mode="json"))
    runner = Runner.from_cfg(runner_config)
    try:
        runner.train()
    except BaseException as error:
        _RUNTIME_STATUS.fail(error)
        raise
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_training_completion",
        "experiment_id": contract.experiment_id,
        "status": "passed_smoke" if max_optimizer_steps is not None else "colab_measured",
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
    if recovery_sync_dir is not None:
        _atomic_sync_checkpoint(output_dir, recovery_sync_dir)
    _RUNTIME_STATUS.complete(
        last_checkpoint=(output_dir / "last_checkpoint").read_text(encoding="utf-8").strip()
    )
    return result


def run_stack_probe(
    config_root: Path,
    mmseg_checkout: Path,
    output_dir: Path,
    project_root: Path,
    project_commit: str,
    *,
    device_name: str = "cuda",
    allow_dirty_project: bool = False,
    model_families: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run selected synthetic CPU/CUDA probes and exact checkpoint resumes."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("stack-probe output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_clean_checkout(
        project_root,
        project_commit,
        allow_dirty=allow_dirty_project,
    )
    project_dirty = bool(_git(project_root, "status", "--porcelain=v1"))
    framework = load_semantic_framework_config(config_root / "framework_mmseg.yaml")
    common = load_semantic_common_config(config_root / "common_cityscapes.yaml")
    models = load_semantic_model_suite(config_root)
    if model_families is not None:
        requested = tuple(dict.fromkeys(model_families))
        available = {model.model_family.value: model for model in models}
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"unknown stack-probe model families: {unknown}")
        models = tuple(available[name] for name in requested)
        if not models:
            raise ValueError("stack-probe model selection cannot be empty")
    if _git(mmseg_checkout, "rev-parse", "HEAD") != framework.commit:
        raise ValueError("MMSegmentation checkout commit mismatch")

    import torch

    mmcv_lite_ops_guard = _install_mmcv_lite_ops_guard()
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    environment = _environment(torch, device_name=device_name)
    environment["mmcv_lite_ops_guard"] = mmcv_lite_ops_guard
    project_status = (
        ProjectStatus.COLAB_MEASURED if device_name == "cuda" else ProjectStatus.LOCALLY_TESTED
    )
    dataset = DatasetIdentity(
        kind="synthetic_stack_fixture",
        synthetic_fixture_identity=common.synthetic_fixture_identity,
    )
    probes: list[dict[str, Any]] = []
    checkpoint_metadata: CheckpointMetadata | None = None
    checkpoint_resume_experiment_ids: list[str] = []
    registry_path = output_dir / "registry.jsonl"
    for model_spec in models:
        contract = build_experiment_contract(
            framework,
            common,
            model_spec,
            dataset=dataset,
            git_commit=project_commit,
            environment=environment,
            status=project_status,
        )
        model = _model_from_official_config(
            model_spec,
            mmseg_checkout,
            torch=torch,
            device_name=device_name,
        )
        probe = _probe_model(
            model,
            model_spec,
            torch=torch,
            device_name=device_name,
        )
        probes.append(probe)
        checkpoint_metadata = _checkpoint_round_trip(model, contract, output_dir, torch=torch)
        checkpoint_resume_experiment_ids.append(contract.experiment_id)
        if project_dirty:
            append_jsonl(
                registry_path,
                {
                    "record_type": "dirty_stack_probe_receipt",
                    "experiment_id": contract.experiment_id,
                    "git_commit": project_commit,
                    "git_dirty": True,
                    "scientific_accuracy_evidence": False,
                },
            )
        else:
            append_registry(
                registry_path,
                ExperimentRegistryRecord(
                    experiment_id=contract.experiment_id,
                    status=project_status,
                    config_sha256=contract.config_sha256,
                    git_commit=project_commit,
                    git_dirty=False,
                    framework_identity_sha256=contract.framework_identity_sha256,
                    dataset_manifest_sha256=None,
                    split_manifest_sha256=None,
                    initialization_checkpoint_sha256=None,
                    seed=common.seed,
                    runtime={"device": device_name, "synthetic_stack_probe": True},
                    final_metrics={},
                    last_metrics={},
                    artifact_paths=("experiments/segmentation/stack-probe/",),
                    failure_summary=None,
                ),
            )
        del model
        if device_name == "cuda":
            torch.cuda.empty_cache()
    assert checkpoint_metadata is not None
    summary = {
        "schema_version": "1.0",
        "record_type": "semantic_stack_probe_summary",
        "run_family": "EGX-SEG-STACK-*",
        "project_commit": project_commit,
        "git_dirty": project_dirty,
        "framework_commit": framework.commit,
        "model_count": len(probes),
        "models": probes,
        "checkpoint_resume_verified": True,
        "checkpoint_resume_model_count": len(checkpoint_resume_experiment_ids),
        "checkpoint_resume_experiment_ids": checkpoint_resume_experiment_ids,
        "scientific_accuracy_evidence": False,
        "status": project_status.value,
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
        "status": project_status.value,
        "scientific_accuracy_evidence": False,
        "project_commit": project_commit,
        "git_dirty": project_dirty,
        "framework_commit": framework.commit,
        "model_count": len(probes),
        "checkpoint_resume_verified": True,
        "checkpoint_resume_model_count": len(checkpoint_resume_experiment_ids),
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
    probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    probe.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model family to probe; repeat to select a staged subset.",
    )
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
    train.add_argument("--max-optimizer-steps", type=int)
    train.add_argument("--validation-subset-size", type=int)
    train.add_argument("--device-batch", type=int)
    train.add_argument("--gradient-accumulation", type=int)
    train.add_argument("--train-fit-fraction", type=float, default=1.0)
    train.add_argument("--smoke-random-initialization", action="store_true")
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
            device_name=args.device,
            model_families=tuple(args.models) if args.models else None,
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
            max_optimizer_steps=args.max_optimizer_steps,
            validation_subset_size=args.validation_subset_size,
            device_batch=args.device_batch,
            gradient_accumulation=args.gradient_accumulation,
            train_fit_fraction=args.train_fit_fraction,
            smoke_random_initialization=args.smoke_random_initialization,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
