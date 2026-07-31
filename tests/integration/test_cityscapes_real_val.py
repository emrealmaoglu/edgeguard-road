"""Opt-in verification for the human-provided Cityscapes validation root."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from edgeguard.data.cityscapes import (
    build_cityscapes_val_manifest,
    discover_cityscapes_val,
    load_cityscapes_val_sample,
)

CITYSCAPES_ROOT = os.environ.get("EDGEGUARD_CITYSCAPES_ROOT")


@pytest.mark.skipif(not CITYSCAPES_ROOT, reason="real Cityscapes val root was not supplied")
def test_real_cityscapes_val_has_approved_counts_and_geometry() -> None:
    root = Path(CITYSCAPES_ROOT or "")

    manifest = build_cityscapes_val_manifest(root)
    samples = discover_cityscapes_val(root)
    image, target = load_cityscapes_val_sample(root, samples[0])

    assert manifest["image_count"] == 500
    assert manifest["label_count"] == 500
    assert manifest["city_count"] == 3
    assert image.shape[:2] == target.shape
    assert set(target.ravel()).issubset(set(range(19)) | {255})
