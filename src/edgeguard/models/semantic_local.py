"""Bounded real-architecture MMSeg validation without pretrained weights."""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.scoring.feature_adapter import train_feature_anomaly_head
from edgeguard.serialization import sha256_file, sha256_payload
from edgeguard.training.config import load_semantic_framework_config, load_semantic_model_suite


def _install_mmcv_lite_guard() -> bool:
    try:
        importlib.metadata.version("mmcv-lite")
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        __import__("mmcv._ext")
    except ModuleNotFoundError:
        extension = types.ModuleType("mmcv._ext")

        def unavailable(name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)

            def fail(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(f"selected model attempted unavailable compiled MMCV op: {name}")

            return fail

        extension.__getattr__ = unavailable  # type: ignore[method-assign]
        sys.modules["mmcv._ext"] = extension
        return True
    return False


def _strip_pretrained(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in tuple(value.items()):
            if (
                key == "init_cfg"
                and isinstance(nested, dict)
                and nested.get("type") == "Pretrained"
            ):
                value[key] = None
            elif key == "pretrained":
                value[key] = None
            else:
                _strip_pretrained(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_pretrained(nested)


def _build_model(spec: Any, mmseg_checkout: Path, torch: Any) -> Any:
    _install_mmcv_lite_guard()
    from mmengine.config import Config  # type: ignore[import-untyped]
    from mmseg.registry import MODELS  # type: ignore[import-not-found]
    from mmseg.utils import register_all_modules  # type: ignore[import-not-found]

    register_all_modules(init_default_scope=True)
    config_path = mmseg_checkout / spec.mmseg_config_relative_path
    if not config_path.is_file():
        raise ValueError(f"pinned MMSeg config is missing: {config_path.name}")
    config = Config.fromfile(config_path)
    _strip_pretrained(config.model)
    config.model["decode_head"]["num_classes"] = 19
    model = MODELS.build(config.model)
    model.init_weights()
    return model.to(torch.device("cpu"))


def _native_logits_and_features(
    model: Any, inputs: Any, torch: Any
) -> tuple[Any, Any, list[list[int]]]:
    features = model.extract_feat(inputs)
    decoded = model.decode_head.forward(features)
    if isinstance(decoded, (list, tuple)):
        reviewed_semantic_output = {"PIDHead": 1, "DDRHead": 0}
        head_name = type(model.decode_head).__name__
        if head_name not in reviewed_semantic_output:
            raise ValueError(f"unreviewed multi-output semantic decode head: {head_name}")
        native_logits = decoded[reviewed_semantic_output[head_name]]
    else:
        native_logits = decoded
    if not torch.is_tensor(native_logits) or native_logits.ndim != 4:
        raise ValueError("semantic decode head did not produce one NCHW tensor")
    transformed = model.decode_head._transform_inputs(features)
    candidates = list(transformed) if isinstance(transformed, (list, tuple)) else [transformed]
    tensors = [candidate for candidate in candidates if torch.is_tensor(candidate)]
    if not tensors:
        raise ValueError("semantic decode head exposed no tensor feature candidate")
    target_hw = tensors[0].shape[-2:]
    aligned = [
        tensor
        if tensor.shape[-2:] == target_hw
        else torch.nn.functional.interpolate(
            tensor, size=target_hw, mode="bilinear", align_corners=False
        )
        for tensor in tensors
    ]
    feature_input = torch.cat(aligned, dim=1)
    return native_logits, feature_input, [list(tensor.shape) for tensor in tensors]


def _fixture_loader(torch: Any, *, seed: int) -> Any:
    torch_data = __import__("torch.utils.data", fromlist=["DataLoader", "TensorDataset"])
    generator = np.random.default_rng(seed)
    images = generator.normal(size=(4, 3, 128, 256)).astype(np.float32)
    labels = generator.integers(0, 19, size=(4, 128, 256), dtype=np.int64)
    labels[:, 0, 0] = 255
    dataset = torch_data.TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    return torch_data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def run_five_model_mini_training(
    config_root: Path,
    mmseg_checkout: Path,
    output_root: Path,
    *,
    optimizer_steps: int = 2,
    fail_model: str | None = None,
    model_families: tuple[str, ...] | None = None,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
) -> dict[str, Any]:
    """Train all five real random-weight MMSeg architectures for bounded CPU steps."""
    if optimizer_steps < 2 or optimizer_steps > 5:
        raise ValueError("local semantic mini training requires 2..5 optimizer steps")
    if learning_rate <= 0 or weight_decay < 0 or optimizer_name not in {"sgd", "adamw"}:
        raise ValueError("semantic mini optimizer parameters are invalid")
    torch = __import__("torch")
    output_root.mkdir(parents=True, exist_ok=False)
    framework_commit = load_semantic_framework_config(config_root / "framework_mmseg.yaml").commit
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    suite = load_semantic_model_suite(config_root)
    selected = (
        tuple(spec for spec in suite if spec.model_family.value in model_families)
        if model_families is not None
        else suite
    )
    if not selected or (model_families is not None and len(selected) != len(set(model_families))):
        raise ValueError("semantic mini model selection is empty or contains unknown families")
    for model_index, spec in enumerate(selected):
        name = spec.model_family.value
        model_root = output_root / name
        model_root.mkdir()
        try:
            if fail_model == name:
                raise RuntimeError("bounded semantic model failure injection")
            torch.manual_seed(20260727 + model_index)
            model = _build_model(spec, mmseg_checkout, torch)
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            optimizer = (
                torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
                if optimizer_name == "sgd"
                else torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            )
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=optimizer_steps, power=0.9
            )
            loader = _fixture_loader(torch, seed=20260727 + model_index)
            losses: list[float] = []
            step_times: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            last_native: Any = None
            last_feature: Any = None
            candidate_shapes: list[list[int]] = []
            for step, (inputs, labels) in enumerate(loader, start=1):
                if step > optimizer_steps * 2:
                    break
                if step % 2 == 0:
                    inputs = torch.flip(inputs, dims=(3,))
                    labels = torch.flip(labels, dims=(2,))
                started = time.perf_counter()
                native, feature, candidate_shapes = _native_logits_and_features(
                    model, inputs, torch
                )
                aligned = torch.nn.functional.interpolate(
                    native, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = torch.nn.functional.cross_entropy(aligned, labels, ignore_index=255) / 2
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("semantic mini-training loss is non-finite")
                loss.backward()
                if step % 2 == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    losses.append(float(loss.detach()) * 2)
                    step_times.append(time.perf_counter() - started)
                    last_native = native.detach()
                    last_feature = feature.detach()
                    if len(losses) == optimizer_steps:
                        break
            if len(losses) != optimizer_steps or last_native is None or last_feature is None:
                raise RuntimeError("semantic mini training did not reach the requested steps")
            anomaly_target = torch.zeros((1, 128, 256), dtype=torch.float32)
            anomaly_target[:, 48:80, 96:160] = 1.0
            valid_mask = torch.ones_like(anomaly_target, dtype=torch.bool)
            feature_result = train_feature_anomaly_head(
                last_feature,
                anomaly_target,
                valid_mask,
                model_root / "anomaly-feature-head",
                model_family=name,
                feature_identity=sha256_payload(candidate_shapes),
                optimizer_steps=optimizer_steps,
            )
            checkpoint = model_root / "last.pt"
            identity = sha256_payload(
                {
                    "model": name,
                    "framework_commit": framework_commit,
                    "steps": optimizer_steps,
                    "seed": 20260727 + model_index,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "identity": identity,
                    "optimizer_steps": optimizer_steps,
                },
                checkpoint,
            )
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            resumed = _build_model(spec, mmseg_checkout, torch)
            resumed.load_state_dict(restored["model"], strict=True)
            resumed_optimizer = (
                torch.optim.SGD(
                    resumed.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
                if optimizer_name == "sgd"
                else torch.optim.AdamW(
                    resumed.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            )
            resumed_optimizer.load_state_dict(restored["optimizer"])
            resumed_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                resumed_optimizer, total_iters=optimizer_steps, power=0.9
            )
            resumed_scheduler.load_state_dict(restored["scheduler"])
            with torch.no_grad():
                validation_inputs, validation_labels = next(iter(_fixture_loader(torch, seed=99)))
                validation_native, resumed_feature, _ = _native_logits_and_features(
                    resumed.eval(), validation_inputs, torch
                )
                validation_aligned = torch.nn.functional.interpolate(
                    validation_native,
                    size=validation_labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                validation_loss = torch.nn.functional.cross_entropy(
                    validation_aligned, validation_labels, ignore_index=255
                )
                if not bool(torch.isfinite(validation_loss)):
                    raise FloatingPointError("semantic validation loss is non-finite")
                validation_prediction = validation_aligned.argmax(dim=1).numpy()
                semantic_metrics = SemanticConfusionMatrix()
                semantic_metrics.update(validation_prediction, validation_labels.numpy())
            results.append(
                {
                    "model_family": name,
                    "architecture": spec.mmseg_config_relative_path,
                    "framework_commit": framework_commit,
                    "initialization": "random_no_download",
                    "trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "input_shape": [1, 3, 128, 256],
                    "native_output_shape": list(last_native.shape),
                    "feature_candidate_shapes": candidate_shapes,
                    "common_feature_shape": list(last_feature.shape),
                    "resumed_feature_shape": list(resumed_feature.shape),
                    "feature_anomaly_head": feature_result,
                    "loss_components": ["cross_entropy_ignore_255"],
                    "train_losses": losses,
                    "validation_loss": float(validation_loss),
                    "validation_metrics": semantic_metrics.result(),
                    "optimizer_steps": optimizer_steps,
                    "gradient_accumulation": 2,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "step_seconds": step_times,
                    "peak_process_rss_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "resume_identity": identity,
                    "exact_resume": restored["identity"] == identity,
                    "scientific_evidence": False,
                }
            )
            (model_root / "result.json").write_text(
                json.dumps(results[-1], sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            failures.append(
                {
                    "model_family": name,
                    "classification": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
    report = {
        "schema_version": "1.0",
        "record_type": "five_real_semantic_mini_training",
        "models": results,
        "failures": failures,
        "all_models_succeeded": len(results) == len(selected),
        "ranking_permitted": False,
        "scientific_evidence": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report


def semantic_environment_ready(mmseg_checkout: Path) -> bool:
    """Return whether the optional local stack and pinned checkout are usable."""
    required = ("torch", "mmengine", "mmcv", "mmsegmentation")
    try:
        for distribution in required:
            importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return mmseg_checkout.is_dir() and (mmseg_checkout / ".git").exists()


def local_mmseg_checkout_from_environment() -> Path | None:
    """Resolve an optional runtime checkout without embedding a private path."""
    value = os.environ.get("EDGEGUARD_MMSEG_CHECKOUT")
    return Path(value).resolve() if value else None
