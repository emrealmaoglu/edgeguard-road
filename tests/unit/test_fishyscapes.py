"""Tests for the narrow Fishyscapes Lost & Found development adapter."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeguard.data.fishyscapes import (
    build_fishyscapes_lost_and_found_manifest,
    discover_fishyscapes_lost_and_found,
    load_fishyscapes_lost_and_found_sample,
    normalize_fishyscapes_mask,
)


def _write_pair(root: Path, *, split: str = "train", height: int = 3) -> None:
    mask_name = "0000_04_Maurener_Weg_8_000000_000030_labels.png"
    image_name = "04_Maurener_Weg_8_000000_000030_leftImg8bit.png"
    image_dir = root / "lostandfound/leftImg8bit" / split / "04_Maurener_Weg_8"
    mask_dir = root / "fishyscapes_lostandfound"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, 4, 3), dtype=np.uint8), mode="RGB").save(
        image_dir / image_name
    )
    mask = np.array([[0, 1, 255, 0]] * 3, dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(mask_dir / mask_name)


def test_fishyscapes_lost_and_found_manifest_is_root_free_and_deterministic(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_pair(first_root)
    _write_pair(second_root)

    first = build_fishyscapes_lost_and_found_manifest(first_root)
    second = build_fishyscapes_lost_and_found_manifest(second_root)

    assert first == second
    assert first["dataset_role"] == "ood_development"
    assert first["source_mode"] == "manual_only"
    assert first["score_direction"] == "higher_means_more_anomalous"
    assert str(first_root) not in str(first)


def test_fishyscapes_lost_and_found_loads_native_mask_semantics(tmp_path: Path) -> None:
    _write_pair(tmp_path, split="test")
    sample = discover_fishyscapes_lost_and_found(tmp_path)[0]

    image, mask = load_fishyscapes_lost_and_found_sample(tmp_path, sample)

    assert image.shape == (3, 4, 3)
    assert set(np.unique(mask)) == {0, 1, 255}


def test_fishyscapes_lost_and_found_rejects_missing_or_ambiguous_pair(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    sample = discover_fishyscapes_lost_and_found(tmp_path)[0]
    (tmp_path / sample.image_relative_path).unlink()
    with pytest.raises(ValueError, match="exactly one"):
        discover_fishyscapes_lost_and_found(tmp_path)

    _write_pair(tmp_path, split="train")
    _write_pair(tmp_path, split="test")
    with pytest.raises(ValueError, match="exactly one"):
        discover_fishyscapes_lost_and_found(tmp_path)


def test_fishyscapes_lost_and_found_rejects_geometry_mismatch(tmp_path: Path) -> None:
    _write_pair(tmp_path, height=2)
    sample = discover_fishyscapes_lost_and_found(tmp_path)[0]

    with pytest.raises(ValueError, match="geometry mismatch"):
        load_fishyscapes_lost_and_found_sample(tmp_path, sample)


def test_fishyscapes_mask_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="ID=0"):
        normalize_fishyscapes_mask(np.array([[0, 2]], dtype=np.uint8))
