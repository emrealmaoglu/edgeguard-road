"""Deterministic Cityscapes synthetic robustness fallback with explicit non-OOD status."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from edgeguard.rescue.dataset import discover_cityscapes
from edgeguard.serialization import canonical_json


def _corrupt(image: Image.Image, *, condition: str, severity: float, seed: int) -> Image.Image:
    rgb = image.convert("RGB")
    if condition == "night":
        return ImageEnhance.Brightness(rgb).enhance(max(0.05, 1.0 - 0.75 * severity))
    if condition == "fog":
        fog = Image.new("RGB", rgb.size, color=(225, 225, 225))
        return Image.blend(rgb, fog, min(0.8, 0.55 * severity))
    rng = np.random.default_rng(seed)
    if condition == "snow":
        array = np.asarray(rgb).copy()
        count = int(array.shape[0] * array.shape[1] * 0.02 * severity)
        rows = rng.integers(0, array.shape[0], size=count)
        columns = rng.integers(0, array.shape[1], size=count)
        array[rows, columns] = 255
        return Image.fromarray(array, mode="RGB")
    if condition == "rain":
        result = rgb.copy()
        draw = ImageDraw.Draw(result)
        count = max(1, int(rgb.width * severity / 8))
        for _ in range(count):
            x = int(rng.integers(0, rgb.width))
            y = int(rng.integers(0, rgb.height))
            length = int(max(4, rgb.height * 0.03 * severity))
            draw.line((x, y, x + length // 3, y + length), fill=(190, 200, 215), width=1)
        return result
    raise ValueError("condition must be fog, night, rain, or snow")


def build_stress_dataset(
    source_root: Path,
    output_root: Path,
    *,
    condition: str,
    severity: float,
    limit: int | None = None,
) -> dict[str, Any]:
    """Materialize a fixed synthetic-val root while preserving original labels."""
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite stress dataset: {output_root}")
    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must be in (0, 1]")
    samples, missing = discover_cityscapes(source_root, split="val")
    if missing:
        raise ValueError("source Cityscapes val has missing masks")
    selected = samples if limit is None else samples[:limit]
    if not selected:
        raise ValueError("stress dataset selection is empty")
    records: list[dict[str, str]] = []
    for sample in selected:
        destination_image = output_root / sample.image
        destination_mask = output_root / sample.mask
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        seed = int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16], 16)
        with Image.open(source_root / sample.image) as image:
            transformed = _corrupt(image, condition=condition, severity=severity, seed=seed)
            transformed.save(destination_image)
        shutil.copy2(source_root / sample.mask, destination_mask)
        records.append({"sample_id": sample.sample_id, "image": sample.image, "mask": sample.mask})
    manifest = {
        "schema_version": "1.0",
        "record_type": "synthetic_cityscapes_stress_dataset",
        "condition": condition,
        "severity": severity,
        "sample_count": len(records),
        "records": records,
        "external_ood_evidence": False,
        "scientific_label": "synthetic robustness stress test",
    }
    (output_root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest
