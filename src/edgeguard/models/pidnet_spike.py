"""Isolated PIDNet-S inference helpers for the first real vertical slice."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

from edgeguard.config import PIDNetEvalConfig, PIDNetSpikeConfig
from edgeguard.contracts import validate_model_input, validate_raw_rgb, validate_semantic_logits
from edgeguard.serialization import sha256_file, sha256_payload


class PIDNetSpikeError(ValueError):
    """Raised when fixed-source, checkpoint, or inference evidence is invalid."""


@dataclass(frozen=True)
class PIDNetForwardResult:
    """Direct and derived logits plus repeatability evidence from two forwards."""

    native_logits: npt.NDArray[np.float32]
    aligned_logits: npt.NDArray[np.float32]
    repeated_native_logits: npt.NDArray[np.float32]
    repeated_aligned_logits: npt.NDArray[np.float32]
    device: str
    torch_version: str
    checkpoint_load_report: dict[str, Any]


@dataclass(frozen=True)
class PIDNetInferenceResult:
    """One direct native-logit output and its aligned derivative."""

    native_logits: npt.NDArray[np.float32]
    aligned_logits: npt.NDArray[np.float32]


@dataclass(frozen=True)
class PIDNetInferenceSession:
    """One strictly loaded PIDNet-S model reused across evaluation samples."""

    model: Any
    torch: Any
    functional: Any
    device: str
    torch_version: str
    num_classes: int
    checkpoint_load_report: dict[str, Any]


@dataclass(frozen=True)
class NormalizedStateDict:
    """Strictly reviewed checkpoint transformation and its audit evidence."""

    state_dict: dict[str, Any]
    transformation_policy: str
    raw_checkpoint_key_count: int
    loaded_inference_key_count: int
    excluded_auxiliary_key_count: int
    excluded_auxiliary_group_counts: dict[str, int]
    excluded_auxiliary_prefixes: tuple[str, ...]
    ignored_training_root_keys: tuple[str, ...]


@dataclass(frozen=True)
class _VerifiedPIDNetCheckpoint:
    """Runtime objects and path-free evidence from one strict checkpoint load."""

    model: Any
    torch: Any
    report: dict[str, Any]


_OFFICIAL_TRAINING_ROOT_SHAPES: dict[str, tuple[int, ...]] = {
    "sem_loss.criterion.weight": (19,),
    "sb_loss.criterion.weight": (19,),
}
_OFFICIAL_AUXILIARY_PREFIX_COUNTS: dict[str, int] = {
    "seghead_p.": 13,
    "seghead_d.": 13,
}


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise PIDNetSpikeError(f"upstream checkout verification failed: {detail}")
    return completed.stdout.strip()


def verify_upstream_checkout(
    checkout: Path,
    *,
    expected_repository_url: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Require the official remote, exact detached commit, and a clean checkout."""
    actual_commit = _git(checkout, "rev-parse", "HEAD")
    actual_repository = _git(checkout, "remote", "get-url", "origin")
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=normal")
    if actual_commit != expected_commit:
        raise PIDNetSpikeError(
            f"upstream commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if actual_repository != expected_repository_url:
        raise PIDNetSpikeError(
            "upstream origin mismatch: "
            f"expected {expected_repository_url!r}, got {actual_repository!r}"
        )
    if status:
        raise PIDNetSpikeError("upstream checkout is dirty; refuse unreviewed source changes")
    return {
        "repository_url": actual_repository,
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "git_clean": True,
    }


def verify_checkpoint_file(
    checkpoint_path: Path,
    *,
    expected_filename: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Require the approved filename and an explicit 64-character SHA-256."""
    if checkpoint_path.name != expected_filename:
        raise PIDNetSpikeError(
            f"checkpoint filename mismatch: expected {expected_filename!r}, "
            f"got {checkpoint_path.name!r}"
        )
    normalized_hash = expected_sha256.lower()
    if expected_sha256 != normalized_hash:
        raise PIDNetSpikeError("expected checkpoint SHA-256 must use lowercase hex")
    if len(normalized_hash) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_hash
    ):
        raise PIDNetSpikeError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != normalized_hash:
        raise PIDNetSpikeError(
            f"checkpoint SHA-256 mismatch: expected {normalized_hash}, got {actual_hash}"
        )
    return {
        "filename": checkpoint_path.name,
        "sha256": actual_hash,
        "size_bytes": checkpoint_path.stat().st_size,
    }


def preprocess_pidnet_rgb(
    raw_image: npt.NDArray[np.uint8],
    *,
    height: int,
    width: int,
    pixel_scale: float,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> npt.NDArray[np.float32]:
    """Resize RGB bytes and apply the official channel-wise normalization."""
    validate_raw_rgb(raw_image)
    if height <= 0 or width <= 0:
        raise PIDNetSpikeError("preprocessing dimensions must be positive")
    if any(value <= 0.0 for value in std):
        raise PIDNetSpikeError("preprocessing standard deviations must be positive")

    resized = Image.fromarray(raw_image, mode="RGB").resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    hwc = np.asarray(resized, dtype=np.float32)
    scaled = hwc * np.float32(pixel_scale)
    normalized = (scaled - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    model_input = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    return validate_model_input(np.ascontiguousarray(model_input))


def normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
    model_keys: set[str],
    *,
    is_tensor: Callable[[Any], bool],
) -> NormalizedStateDict:
    """Accept exact inference mappings or the one reviewed official train layout."""
    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise PIDNetSpikeError("checkpoint state dict must have non-empty string keys")

    keys = set(state_dict)
    if keys == model_keys:
        _require_tensor_entries(state_dict, is_tensor=is_tensor)
        return _normalized_state_dict(
            state_dict,
            transformation_policy="exact_inference_state_dict",
            raw_checkpoint_key_count=len(state_dict),
        )

    for prefix in ("model.", "module."):
        if all(key.startswith(prefix) for key in keys):
            stripped = {key[len(prefix) :]: value for key, value in state_dict.items()}
            if set(stripped) == model_keys:
                _require_tensor_entries(stripped, is_tensor=is_tensor)
                return _normalized_state_dict(
                    stripped,
                    transformation_policy=f"removed_uniform_prefix:{prefix}",
                    raw_checkpoint_key_count=len(state_dict),
                )

    return _normalize_official_training_state_dict(state_dict, model_keys, is_tensor=is_tensor)


def _normalized_state_dict(
    state_dict: Mapping[str, Any],
    *,
    transformation_policy: str,
    raw_checkpoint_key_count: int,
    excluded_auxiliary_group_counts: Mapping[str, int] | None = None,
    ignored_training_root_keys: tuple[str, ...] = (),
) -> NormalizedStateDict:
    group_counts = dict(excluded_auxiliary_group_counts or {})
    return NormalizedStateDict(
        state_dict=dict(state_dict),
        transformation_policy=transformation_policy,
        raw_checkpoint_key_count=raw_checkpoint_key_count,
        loaded_inference_key_count=len(state_dict),
        excluded_auxiliary_key_count=sum(group_counts.values()),
        excluded_auxiliary_group_counts=group_counts,
        excluded_auxiliary_prefixes=tuple(sorted(group_counts)),
        ignored_training_root_keys=ignored_training_root_keys,
    )


def _normalize_official_training_state_dict(
    state_dict: Mapping[str, Any],
    model_keys: set[str],
    *,
    is_tensor: Callable[[Any], bool],
) -> NormalizedStateDict:
    root_keys = {key for key in state_dict if not key.startswith("model.")}
    expected_root_keys = set(_OFFICIAL_TRAINING_ROOT_SHAPES)
    if root_keys != expected_root_keys:
        missing = sorted(expected_root_keys - root_keys)
        unexpected = sorted(root_keys - expected_root_keys)
        raise PIDNetSpikeError(
            "checkpoint key mismatch; refusing partial or unreviewed layout: "
            f"official_training_roots_missing={missing}, "
            f"official_training_roots_unexpected={unexpected}"
        )

    for key, expected_shape in _OFFICIAL_TRAINING_ROOT_SHAPES.items():
        actual_shape = _tensor_shape(state_dict[key], key, is_tensor=is_tensor)
        if actual_shape != expected_shape:
            raise PIDNetSpikeError(
                f"training-only root shape mismatch for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    stripped_entries = {
        key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")
    }
    group_counts: dict[str, int] = {}
    auxiliary_keys: set[str] = set()
    for prefix, expected_count in _OFFICIAL_AUXILIARY_PREFIX_COUNTS.items():
        grouped_keys = {key for key in stripped_entries if key.startswith(prefix)}
        if len(grouped_keys) != expected_count:
            raise PIDNetSpikeError(
                f"official auxiliary group {prefix} has {len(grouped_keys)} keys; "
                f"expected {expected_count}"
            )
        for key in grouped_keys:
            _tensor_shape(stripped_entries[key], f"model.{key}", is_tensor=is_tensor)
        group_counts[prefix] = len(grouped_keys)
        auxiliary_keys.update(grouped_keys)

    inference_entries = {
        key: value for key, value in stripped_entries.items() if key not in auxiliary_keys
    }
    inference_keys = set(inference_entries)
    if inference_keys != model_keys:
        missing = sorted(model_keys - inference_keys)
        unexpected = sorted(inference_keys - model_keys)
        raise PIDNetSpikeError(
            "official training checkpoint inference keys do not match model; "
            f"missing_sample={missing[:10]}, unexpected_sample={unexpected[:10]}"
        )
    _require_tensor_entries(inference_entries, is_tensor=is_tensor)

    return _normalized_state_dict(
        inference_entries,
        transformation_policy="reviewed_official_training_checkpoint",
        raw_checkpoint_key_count=len(state_dict),
        excluded_auxiliary_group_counts=group_counts,
        ignored_training_root_keys=tuple(sorted(root_keys)),
    )


def validate_state_dict_shapes(
    state_dict: Mapping[str, Any],
    model_state_dict: Mapping[str, Any],
    *,
    is_tensor: Callable[[Any], bool],
) -> dict[str, list[int]]:
    """Require every checkpoint parameter shape to match its model parameter."""
    shape_manifest: dict[str, list[int]] = {}
    shape_mismatches: list[str] = []
    for key, expected_value in model_state_dict.items():
        checkpoint_value = state_dict[key]
        expected_shape = _tensor_shape(expected_value, f"model:{key}", is_tensor=is_tensor)
        actual_shape = _tensor_shape(checkpoint_value, key, is_tensor=is_tensor)
        if actual_shape != expected_shape:
            shape_mismatches.append(f"{key}: expected {expected_shape}, got {actual_shape}")
        shape_manifest[key] = [int(dimension) for dimension in actual_shape]
    if shape_mismatches:
        raise PIDNetSpikeError(
            "checkpoint parameter shape mismatch; refusing load: "
            + "; ".join(shape_mismatches[:10])
        )
    return shape_manifest


def _require_tensor(value: Any, key: str, *, is_tensor: Callable[[Any], bool]) -> None:
    try:
        accepted = is_tensor(value)
    except Exception as error:
        raise PIDNetSpikeError(f"tensor predicate failed for checkpoint entry {key}") from error
    if not accepted:
        raise PIDNetSpikeError(f"checkpoint entry {key} is not a tensor")


def _require_tensor_entries(
    entries: Mapping[str, Any], *, is_tensor: Callable[[Any], bool]
) -> None:
    for key, value in entries.items():
        _require_tensor(value, key, is_tensor=is_tensor)


def _tensor_shape(value: Any, key: str, *, is_tensor: Callable[[Any], bool]) -> tuple[int, ...]:
    _require_tensor(value, key, is_tensor=is_tensor)
    shape = getattr(value, "shape", None)
    if shape is None:
        raise PIDNetSpikeError(f"checkpoint tensor entry {key} has no shape")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise PIDNetSpikeError(f"checkpoint entry {key} has an invalid tensor shape") from error


def _load_state_dict_strict(model: Any, state_dict: Mapping[str, Any]) -> None:
    try:
        incompatible = model.load_state_dict(dict(state_dict), strict=True)
    except Exception as error:
        raise PIDNetSpikeError(
            f"strict checkpoint load failed: {type(error).__name__}: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PIDNetSpikeError(
            "strict checkpoint load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _extract_state_dict(payload: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise PIDNetSpikeError("checkpoint payload must be a mapping")
    for container_key in ("state_dict", "model"):
        candidate = payload.get(container_key)
        if isinstance(candidate, Mapping):
            return candidate, container_key
    if payload and all(isinstance(key, str) for key in payload):
        return payload, "root"
    raise PIDNetSpikeError("checkpoint has no supported state_dict/model mapping")


def _pidnet_module(checkout: Path) -> Any:
    checkout_string = str(checkout.resolve())
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, checkout_string)
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("models.pidnet")
    finally:
        sys.path.remove(checkout_string)
        sys.dont_write_bytecode = previous_dont_write_bytecode
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise PIDNetSpikeError("PIDNet module has no verifiable source file")
    module_path = Path(module_file).resolve()
    if not module_path.is_relative_to(checkout.resolve()):
        raise PIDNetSpikeError(f"PIDNet module resolved outside fixed checkout: {module_path}")
    return module


def _load_verified_pidnet_checkpoint(
    *,
    checkout: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    config: PIDNetSpikeConfig | PIDNetEvalConfig,
) -> _VerifiedPIDNetCheckpoint:
    upstream_report = verify_upstream_checkout(
        checkout,
        expected_repository_url=config.upstream.repository_url,
        expected_commit=config.upstream.commit,
    )
    checkpoint_report = verify_checkpoint_file(
        checkpoint_path,
        expected_filename=config.checkpoint.filename,
        expected_sha256=expected_checkpoint_sha256,
    )

    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise PIDNetSpikeError(
            f"PyTorch import failed in execution environment: {error}"
        ) from error

    pidnet = _pidnet_module(checkout)
    model = pidnet.get_pred_model(
        name=config.model.backend,
        num_classes=config.model.num_classes,
    )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PIDNetSpikeError(
            "safe checkpoint load with weights_only=True failed; do not retry with "
            f"weights_only=False without human approval: {type(error).__name__}: {error}"
        ) from error

    raw_state_dict, container = _extract_state_dict(payload)
    model_state_dict = model.state_dict()
    normalized = normalize_state_dict_keys(
        raw_state_dict,
        set(model_state_dict.keys()),
        is_tensor=torch.is_tensor,
    )
    shape_manifest = validate_state_dict_shapes(
        normalized.state_dict,
        model_state_dict,
        is_tensor=torch.is_tensor,
    )
    _load_state_dict_strict(model, normalized.state_dict)

    report = {
        **checkpoint_report,
        "payload_container": container,
        "transformation_policy": normalized.transformation_policy,
        "key_transform": normalized.transformation_policy,
        "raw_checkpoint_key_count": normalized.raw_checkpoint_key_count,
        "loaded_inference_key_count": normalized.loaded_inference_key_count,
        "loaded_key_count": normalized.loaded_inference_key_count,
        "excluded_auxiliary_key_count": normalized.excluded_auxiliary_key_count,
        "excluded_auxiliary_group_counts": normalized.excluded_auxiliary_group_counts,
        "excluded_auxiliary_prefixes": list(normalized.excluded_auxiliary_prefixes),
        "ignored_training_root_keys": list(normalized.ignored_training_root_keys),
        "strict": True,
        "shape_check": "exact",
        "parameter_shapes_sha256": sha256_payload(shape_manifest),
        "missing_keys": [],
        "unexpected_keys": [],
        "weights_only": True,
        "torch_version": str(torch.__version__),
        "upstream": upstream_report,
    }
    return _VerifiedPIDNetCheckpoint(model=model, torch=torch, report=report)


def verify_pidnet_checkpoint_layout(
    *,
    checkout: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    config: PIDNetSpikeConfig | PIDNetEvalConfig,
) -> dict[str, Any]:
    """Strictly load the approved PIDNet-S checkpoint and return path-free evidence."""
    return _load_verified_pidnet_checkpoint(
        checkout=checkout,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        config=config,
    ).report


def _select_torch_device(torch: Any, requested: Literal["auto", "cpu", "mps", "cuda"]) -> str:
    """Resolve an explicit available device without hiding backend failures."""
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise PIDNetSpikeError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise PIDNetSpikeError("MPS was requested but is not available")
    return requested


def load_pidnet_session(
    *,
    checkout: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    config: PIDNetSpikeConfig | PIDNetEvalConfig,
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
) -> PIDNetInferenceSession:
    """Strictly load one PIDNet-S model for repeated inference."""
    verified = _load_verified_pidnet_checkpoint(
        checkout=checkout,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        config=config,
    )
    torch = verified.torch
    try:
        functional = importlib.import_module("torch.nn.functional")
    except (ImportError, OSError) as error:
        raise PIDNetSpikeError(f"PyTorch functional import failed: {error}") from error

    torch.manual_seed(config.seed)
    selected_device = _select_torch_device(torch, device)
    if selected_device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = verified.model.eval().to(selected_device)
    return PIDNetInferenceSession(
        model=model,
        torch=torch,
        functional=functional,
        device=selected_device,
        torch_version=str(torch.__version__),
        num_classes=config.model.num_classes,
        checkpoint_load_report=verified.report,
    )


def infer_pidnet(
    session: PIDNetInferenceSession,
    model_input: npt.NDArray[np.float32],
    *,
    alignment_height: int,
    alignment_width: int,
    alignment_mode: str = "bilinear",
    align_corners: bool = True,
) -> PIDNetInferenceResult:
    """Run one forward while preserving native and aligned raw logits separately."""
    validate_model_input(model_input)
    if alignment_height <= 0 or alignment_width <= 0:
        raise PIDNetSpikeError("alignment dimensions must be positive")
    tensor = session.torch.from_numpy(model_input).to(session.device)
    with session.torch.inference_mode():
        native = session.model(tensor)
        if not isinstance(native, session.torch.Tensor):
            raise PIDNetSpikeError(
                "augment=False forward did not return one semantic tensor directly"
            )
        if native.dtype != session.torch.float32:
            raise PIDNetSpikeError(f"native logits must be torch.float32, got {native.dtype}")
        aligned = session.functional.interpolate(
            native,
            size=(alignment_height, alignment_width),
            mode=alignment_mode,
            align_corners=align_corners,
        )
        if aligned.dtype != session.torch.float32:
            raise PIDNetSpikeError(f"aligned logits must be torch.float32, got {aligned.dtype}")
        native_array = validate_semantic_logits(native.detach().cpu().numpy())
        aligned_array = validate_semantic_logits(aligned.detach().cpu().numpy())

    for name, array in (("native_logits", native_array), ("aligned_logits", aligned_array)):
        if array.shape[0] != model_input.shape[0] or array.shape[1] != session.num_classes:
            raise PIDNetSpikeError(f"{name} has unexpected batch/classes shape: {array.shape}")
    if aligned_array.shape[2:] != (alignment_height, alignment_width):
        raise PIDNetSpikeError(f"aligned logits have unexpected grid: {aligned_array.shape}")
    return PIDNetInferenceResult(native_logits=native_array, aligned_logits=aligned_array)


def run_pidnet_forward(
    model_input: npt.NDArray[np.float32],
    *,
    checkout: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    config: PIDNetSpikeConfig,
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
) -> PIDNetForwardResult:
    """Load the fixed model strictly and run two native/aligned forward passes."""
    session = load_pidnet_session(
        checkout=checkout,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        config=config,
        device=device,
    )
    native_results: list[npt.NDArray[np.float32]] = []
    aligned_results: list[npt.NDArray[np.float32]] = []
    for _ in range(2):
        inference = infer_pidnet(
            session,
            model_input,
            alignment_height=config.input.height,
            alignment_width=config.input.width,
            alignment_mode=config.alignment.mode,
            align_corners=config.alignment.align_corners,
        )
        native_results.append(inference.native_logits)
        aligned_results.append(inference.aligned_logits)

    return PIDNetForwardResult(
        native_logits=native_results[0],
        aligned_logits=aligned_results[0],
        repeated_native_logits=native_results[1],
        repeated_aligned_logits=aligned_results[1],
        device=session.device,
        torch_version=session.torch_version,
        checkpoint_load_report=session.checkpoint_load_report,
    )


def difference_summary(
    first: npt.NDArray[Any], second: npt.NDArray[Any]
) -> dict[str, float | bool]:
    """Report repeat differences without imposing a numerical pass threshold."""
    if first.shape != second.shape:
        raise PIDNetSpikeError(f"repeat shape mismatch: {first.shape} != {second.shape}")
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "byte_equal": first.tobytes(order="C") == second.tobytes(order="C"),
        "max_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "p95_absolute_difference": float(np.percentile(difference, 95.0)),
    }
