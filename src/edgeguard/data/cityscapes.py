"""Minimal deterministic Cityscapes validation adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from edgeguard.data.single_image import load_rgb_image
from edgeguard.serialization import sha256_payload

_IMAGE_PATTERN = re.compile(
    r"^(?P<city>[a-z][a-z0-9]*)_(?P<sequence>[0-9]{6})_"
    r"(?P<frame>[0-9]{6})_leftImg8bit\.png$"
)
_LABEL_PATTERN = re.compile(
    r"^(?P<city>[a-z][a-z0-9]*)_(?P<sequence>[0-9]{6})_"
    r"(?P<frame>[0-9]{6})_gtFine_labelIds\.png$"
)

CITYSCAPES_LABEL_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    CITYSCAPES_LABEL_ID_TO_TRAIN_ID[_label_id] = _train_id


@dataclass(frozen=True)
class CityscapesValSample:
    """One paired Cityscapes validation image and label record."""

    sample_id: str
    city: str
    sequence: str
    frame: str
    image_relative_path: str
    label_relative_path: str


def _parse_name(path: Path, pattern: re.Pattern[str], kind: str) -> tuple[str, str, str]:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"malformed Cityscapes {kind} filename: {path.name}")
    city = match.group("city")
    if path.parent.name != city:
        raise ValueError(
            f"Cityscapes {kind} city directory mismatch for {path.name}: {path.parent.name}"
        )
    return city, match.group("sequence"), match.group("frame")


def discover_cityscapes_val(root: Path) -> list[CityscapesValSample]:
    """Discover and strictly pair the Cityscapes validation split."""
    image_root = root / "leftImg8bit" / "val"
    label_root = root / "gtFine" / "val"
    if not image_root.is_dir() or not label_root.is_dir():
        raise ValueError("Cityscapes val root must contain leftImg8bit/val and gtFine/val")

    images: dict[str, tuple[Path, str, str, str]] = {}
    for path in sorted(image_root.rglob("*.png")):
        city, sequence, frame = _parse_name(path, _IMAGE_PATTERN, "image")
        sample_id = f"{city}_{sequence}_{frame}"
        if sample_id in images:
            raise ValueError(f"duplicate Cityscapes val image sample: {sample_id}")
        images[sample_id] = (path, city, sequence, frame)

    labels: dict[str, Path] = {}
    for path in sorted(label_root.rglob("*_gtFine_labelIds.png")):
        city, sequence, frame = _parse_name(path, _LABEL_PATTERN, "label")
        sample_id = f"{city}_{sequence}_{frame}"
        if sample_id in labels:
            raise ValueError(f"duplicate Cityscapes val label sample: {sample_id}")
        labels[sample_id] = path

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise ValueError(
            "Cityscapes val image-label pairing mismatch: "
            f"missing_labels={missing_labels[:10]}, missing_images={missing_images[:10]}"
        )

    samples: list[CityscapesValSample] = []
    for sample_id in sorted(images):
        image_path, city, sequence, frame = images[sample_id]
        label_path = labels[sample_id]
        samples.append(
            CityscapesValSample(
                sample_id=sample_id,
                city=city,
                sequence=sequence,
                frame=frame,
                image_relative_path=image_path.relative_to(root).as_posix(),
                label_relative_path=label_path.relative_to(root).as_posix(),
            )
        )
    return samples


def label_ids_to_train_ids(
    label_ids: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Map native Cityscapes label IDs to 19 train IDs plus ignore 255."""
    if not isinstance(label_ids, np.ndarray) or label_ids.dtype != np.uint8:
        raise ValueError("Cityscapes label IDs must be a uint8 numpy array")
    if label_ids.ndim != 2 or any(dimension <= 0 for dimension in label_ids.shape):
        raise ValueError("Cityscapes label IDs must have positive HW dimensions")
    return CITYSCAPES_LABEL_ID_TO_TRAIN_ID[label_ids]


def load_cityscapes_val_sample(
    root: Path, sample: CityscapesValSample
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Load one RGB image and its mapped train-ID mask with exact geometry."""
    image = load_rgb_image(root / sample.image_relative_path)
    label_path = root / sample.label_relative_path
    try:
        with Image.open(label_path) as encoded:
            encoded.load()
            label_ids = np.asarray(encoded, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not decode Cityscapes label {label_path.name!r}") from error
    train_ids = label_ids_to_train_ids(label_ids)
    if image.shape[:2] != train_ids.shape:
        raise ValueError(
            f"Cityscapes image-label geometry mismatch for {sample.sample_id}: "
            f"image={image.shape[:2]}, label={train_ids.shape}"
        )
    return image, train_ids


def resize_train_ids(
    train_ids: npt.NDArray[np.uint8], *, height: int, width: int
) -> npt.NDArray[np.uint8]:
    """Resize a categorical train-ID mask with nearest-neighbor only."""
    if height <= 0 or width <= 0:
        raise ValueError("mask resize dimensions must be positive")
    label_ids_to_train_ids(train_ids)
    resized = Image.fromarray(train_ids, mode="L").resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8).copy()


def build_cityscapes_val_manifest(root: Path) -> dict[str, Any]:
    """Build a root-free deterministic manifest for paired validation samples."""
    samples = discover_cityscapes_val(root)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_val_manifest",
        "dataset": "Cityscapes",
        "dataset_role": "official_val_common_eval",
        "split": "val",
        "image_count": len(samples),
        "label_count": len(samples),
        "city_count": len({sample.city for sample in samples}),
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
        "duplicate_count": 0,
        "missing_pair_count": 0,
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


def select_city_round_robin(
    samples: list[CityscapesValSample], count: int
) -> list[CityscapesValSample]:
    """Select deterministically by cycling through alphabetically sorted cities."""
    if count <= 0 or count > len(samples):
        raise ValueError(f"subset count must be between 1 and {len(samples)}")
    grouped: dict[str, list[CityscapesValSample]] = {}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        grouped.setdefault(sample.city, []).append(sample)

    selected: list[CityscapesValSample] = []
    offset = 0
    cities = sorted(grouped)
    while len(selected) < count:
        added = False
        for city in cities:
            city_samples = grouped[city]
            if offset < len(city_samples):
                selected.append(city_samples[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("could not complete deterministic Cityscapes selection")
        offset += 1
    return selected
