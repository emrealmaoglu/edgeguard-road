"""Legal single-image loading and deterministic manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps, UnidentifiedImageError

from edgeguard.contracts import validate_raw_rgb
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload


def load_rgb_image(path: Path) -> npt.NDArray[np.uint8]:
    """Decode one image, apply EXIF orientation, and return owned HWC RGB bytes."""
    try:
        with Image.open(path) as encoded:
            encoded.load()
            oriented = ImageOps.exif_transpose(encoded)
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not decode legal image {path.name!r}: {error}") from error
    return validate_raw_rgb(array)


def _relative_file(path: Path, root: Path) -> tuple[Path, Path]:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"image root is not a directory: {root}")
    if not resolved_path.is_file():
        raise ValueError(f"image path is not a file: {path}")
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("legal image must resolve inside the approved image root") from error
    return resolved_path, relative


def build_single_image_manifest(
    image_path: Path,
    *,
    image_root: Path,
    sample_id: str,
    source_reference: str,
    license_reference: str,
) -> dict[str, Any]:
    """Build a timestamp-free, root-independent manifest for one approved image."""
    if not sample_id.strip():
        raise ValueError("sample_id must not be empty")
    if not source_reference.strip():
        raise ValueError("source_reference must not be empty")
    if not license_reference.strip():
        raise ValueError("license_reference must not be empty")

    resolved_path, relative_path = _relative_file(image_path, image_root)
    image = load_rgb_image(resolved_path)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "single_image_manifest",
        "sample_id": sample_id,
        "relative_path": relative_path.as_posix(),
        "sha256": sha256_file(resolved_path),
        "original_shape": [int(dimension) for dimension in image.shape],
        "dtype": "uint8",
        "color_space": "RGB",
        "decode_policy": "pillow_exif_transpose_rgb",
        "source_reference": source_reference,
        "license_reference": license_reference,
        "role": "plumbing_only",
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    return manifest


def build_upstream_sample_manifest(
    image_path: Path,
    *,
    checkout_root: Path,
    sample_id: str,
    upstream_repository: str,
    upstream_commit: str,
    source_access_date: str,
    expected_relative_path: str,
    expected_filename: str,
    expected_sha256: str,
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Verify and describe one approved image from the fixed upstream checkout."""
    resolved_path, relative_path = _relative_file(image_path, checkout_root)
    image = load_rgb_image(resolved_path)
    actual_relative_path = relative_path.as_posix()
    actual_hash = sha256_file(resolved_path)
    actual_shape = tuple(int(dimension) for dimension in image.shape)
    if actual_relative_path != expected_relative_path:
        raise ValueError(
            f"upstream sample path mismatch: expected {expected_relative_path!r}, "
            f"got {actual_relative_path!r}"
        )
    if resolved_path.name != expected_filename:
        raise ValueError(
            f"upstream sample filename mismatch: expected {expected_filename!r}, "
            f"got {resolved_path.name!r}"
        )
    if actual_hash != expected_sha256:
        raise ValueError(
            f"upstream sample SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    if actual_shape != expected_shape:
        raise ValueError(
            f"upstream sample shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "upstream_sample_manifest",
        "sample_id": sample_id,
        "upstream_repository": upstream_repository,
        "upstream_commit": upstream_commit,
        "relative_path": actual_relative_path,
        "filename": resolved_path.name,
        "sha256": actual_hash,
        "byte_size": resolved_path.stat().st_size,
        "original_shape": list(actual_shape),
        "dtype": "uint8",
        "color_space": "RGB",
        "decode_policy": "pillow_exif_transpose_rgb",
        "access_date": source_access_date,
        "usage_scope": "noncommercial_internal_plumbing",
        "dataset_role": "plumbing_only",
        "license_status": "OPEN QUESTION",
        "redistribution_permitted": False,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    return manifest


def write_single_image_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write one canonical manifest without volatile metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
