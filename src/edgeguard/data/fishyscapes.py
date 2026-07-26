"""Narrow adapter foundation for manually prepared Fishyscapes development data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from edgeguard.data.single_image import load_rgb_image
from edgeguard.serialization import sha256_file, sha256_payload

FISHYSCAPES_LOST_AND_FOUND_ROLE = "ood_development"
FISHYSCAPES_STATIC_ROLE = "ood_development"
FISHYSCAPES_ID_LABEL = 0
FISHYSCAPES_ANOMALY_LABEL = 1
FISHYSCAPES_IGNORE_LABEL = 255

_ANNOTATION_PATTERN = re.compile(r"^(?P<index>[0-9]{4})_(?P<body>.+)_labels\.png$")


@dataclass(frozen=True)
class FishyscapesLostAndFoundSample:
    """One manually paired public validation image and Fishyscapes mask."""

    sample_id: str
    image_relative_path: str
    mask_relative_path: str


def _image_relative_path(annotation_name: str, split: str) -> Path:
    parts = annotation_name.split("_")
    if len(parts) < 7:
        raise ValueError(f"malformed Fishyscapes annotation filename: {annotation_name}")
    scene = "_".join(parts[1:-3])
    image_name = "_".join(parts[1:]).replace("_labels.png", "_leftImg8bit.png")
    return Path("lostandfound") / "leftImg8bit" / split / scene / image_name


def discover_fishyscapes_lost_and_found(
    root: Path,
) -> list[FishyscapesLostAndFoundSample]:
    """Pair manually supplied public Lost & Found validation masks and images."""
    annotation_root = root / "fishyscapes_lostandfound"
    image_root = root / "lostandfound" / "leftImg8bit"
    if not annotation_root.is_dir() or not image_root.is_dir():
        raise ValueError(
            "Fishyscapes root must contain fishyscapes_lostandfound and lostandfound/leftImg8bit"
        )

    samples: list[FishyscapesLostAndFoundSample] = []
    seen: set[str] = set()
    for mask_path in sorted(annotation_root.glob("*_labels.png")):
        match = _ANNOTATION_PATTERN.fullmatch(mask_path.name)
        if match is None:
            raise ValueError(f"malformed Fishyscapes annotation filename: {mask_path.name}")
        sample_id = f"{match.group('index')}_{match.group('body')}"
        if sample_id in seen:
            raise ValueError(f"duplicate Fishyscapes sample: {sample_id}")
        candidates: list[Path] = []
        for split in ("train", "test"):
            relative = _image_relative_path(mask_path.name, split)
            if (root / relative).is_file():
                candidates.append(relative)
        if len(candidates) != 1:
            raise ValueError(
                f"Fishyscapes image pairing requires exactly one train/test image for {sample_id}"
            )
        seen.add(sample_id)
        samples.append(
            FishyscapesLostAndFoundSample(
                sample_id=sample_id,
                image_relative_path=candidates[0].as_posix(),
                mask_relative_path=mask_path.relative_to(root).as_posix(),
            )
        )
    if not samples:
        raise ValueError("Fishyscapes Lost & Found validation contains no annotation masks")
    return samples


def normalize_fishyscapes_mask(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Validate the native 0/1/255 ID/anomaly/void convention."""
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
        raise ValueError("Fishyscapes mask must be a uint8 numpy array")
    if mask.ndim != 2 or any(dimension <= 0 for dimension in mask.shape):
        raise ValueError("Fishyscapes mask must have positive HW dimensions")
    allowed = np.isin(
        mask,
        np.array(
            [FISHYSCAPES_ID_LABEL, FISHYSCAPES_ANOMALY_LABEL, FISHYSCAPES_IGNORE_LABEL],
            dtype=np.uint8,
        ),
    )
    if not bool(np.all(allowed)):
        raise ValueError("Fishyscapes mask values must be ID=0, anomaly=1, or void=255")
    return mask.copy()


def load_fishyscapes_lost_and_found_sample(
    root: Path, sample: FishyscapesLostAndFoundSample
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load one RGB image and native public-validation OOD mask."""
    image = load_rgb_image(root / sample.image_relative_path)
    mask_path = root / sample.mask_relative_path
    try:
        with Image.open(mask_path) as encoded:
            encoded.load()
            mask = np.asarray(encoded, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not decode Fishyscapes mask {mask_path.name!r}") from error
    mask = normalize_fishyscapes_mask(mask)
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Fishyscapes image-mask geometry mismatch for {sample.sample_id}: "
            f"image={image.shape[:2]}, mask={mask.shape}"
        )
    return image, mask


def build_fishyscapes_lost_and_found_manifest(root: Path) -> dict[str, Any]:
    """Build a deterministic root-free manifest for manually prepared validation data."""
    samples = discover_fishyscapes_lost_and_found(root)
    records: list[dict[str, Any]] = []
    for sample in samples:
        image, mask = load_fishyscapes_lost_and_found_sample(root, sample)
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_relative_path": sample.image_relative_path,
                "mask_relative_path": sample.mask_relative_path,
                "image_sha256": sha256_file(root / sample.image_relative_path),
                "mask_sha256": sha256_file(root / sample.mask_relative_path),
                "shape": [int(image.shape[0]), int(image.shape[1])],
                "anomaly_pixel_count": int(np.count_nonzero(mask == FISHYSCAPES_ANOMALY_LABEL)),
                "ignore_pixel_count": int(np.count_nonzero(mask == FISHYSCAPES_IGNORE_LABEL)),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "fishyscapes_lost_and_found_validation_manifest",
        "dataset": "Fishyscapes Lost & Found",
        "dataset_role": FISHYSCAPES_LOST_AND_FOUND_ROLE,
        "split": "validation",
        "source_mode": "manual_only",
        "score_direction": "higher_means_more_anomalous",
        "sample_count": len(records),
        "samples": records,
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload
