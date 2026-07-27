"""Tests for the narrow Cityscapes validation adapter."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeguard.data.cityscapes import (
    build_cityscapes_val_manifest,
    discover_cityscapes_val,
    label_ids_to_train_ids,
    load_cityscapes_val_sample,
    resize_train_ids,
    select_city_round_robin,
)


def _write_pair(
    root: Path,
    sample_id: str,
    *,
    image_shape: tuple[int, int] = (2, 3),
    label_shape: tuple[int, int] = (2, 3),
) -> None:
    city = sample_id.split("_", maxsplit=1)[0]
    image_dir = root / "leftImg8bit" / "val" / city
    label_dir = root / "gtFine" / "val" / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((*image_shape, 3), dtype=np.uint8), mode="RGB").save(
        image_dir / f"{sample_id}_leftImg8bit.png"
    )
    Image.fromarray(np.full(label_shape, 7, dtype=np.uint8), mode="L").save(
        label_dir / f"{sample_id}_gtFine_labelIds.png"
    )


def test_cityscapes_val_discovery_is_sorted_and_root_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        _write_pair(root, "frankfurt_000001_000002")
        _write_pair(root, "frankfurt_000000_000003")

    first = build_cityscapes_val_manifest(first_root)
    second = build_cityscapes_val_manifest(second_root)

    assert first == second
    assert first["dataset_role"] == "official_val_common_eval"
    assert [sample["sample_id"] for sample in first["samples"]] == [
        "frankfurt_000000_000003",
        "frankfurt_000001_000002",
    ]
    assert str(first_root) not in str(first)


def test_cityscapes_val_rejects_missing_pair(tmp_path: Path) -> None:
    _write_pair(tmp_path, "frankfurt_000000_000001")
    (tmp_path / "gtFine/val/frankfurt/frankfurt_000000_000001_gtFine_labelIds.png").unlink()

    with pytest.raises(ValueError, match="pairing mismatch"):
        discover_cityscapes_val(tmp_path)


def test_cityscapes_val_rejects_malformed_image_filename(tmp_path: Path) -> None:
    image_dir = tmp_path / "leftImg8bit/val/frankfurt"
    label_dir = tmp_path / "gtFine/val/frankfurt"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(
        image_dir / "malformed_leftImg8bit.png"
    )

    with pytest.raises(ValueError, match="malformed Cityscapes image filename"):
        discover_cityscapes_val(tmp_path)


def test_cityscapes_val_rejects_image_label_geometry_mismatch(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "frankfurt_000000_000001",
        image_shape=(2, 3),
        label_shape=(3, 2),
    )
    sample = discover_cityscapes_val(tmp_path)[0]

    with pytest.raises(ValueError, match="geometry mismatch"):
        load_cityscapes_val_sample(tmp_path, sample)


def test_cityscapes_label_id_mapping_preserves_train_ids_and_ignore() -> None:
    label_ids = np.array(
        [[7, 8, 11, 12, 13, 17, 19, 20, 21, 22], [23, 24, 25, 26, 27, 28, 31, 32, 33, 0]],
        dtype=np.uint8,
    )

    train_ids = label_ids_to_train_ids(label_ids)

    np.testing.assert_array_equal(
        train_ids,
        np.array([list(range(10)), list(range(10, 19)) + [255]], dtype=np.uint8),
    )


def test_cityscapes_mask_resize_uses_nearest_neighbor() -> None:
    mask = np.array([[0, 18], [255, 1]], dtype=np.uint8)

    resized = resize_train_ids(mask, height=4, width=4)

    assert set(np.unique(resized)) == {0, 1, 18, 255}
    np.testing.assert_array_equal(resized[:2, :2], np.zeros((2, 2), dtype=np.uint8))


def test_city_round_robin_selection_is_deterministic_across_cities(tmp_path: Path) -> None:
    for sample_id in (
        "munster_000000_000002",
        "frankfurt_000000_000002",
        "lindau_000000_000002",
        "frankfurt_000000_000001",
        "lindau_000000_000001",
        "munster_000000_000001",
    ):
        _write_pair(tmp_path, sample_id)

    selected = select_city_round_robin(discover_cityscapes_val(tmp_path), 5)

    assert [sample.sample_id for sample in selected] == [
        "frankfurt_000000_000001",
        "lindau_000000_000001",
        "munster_000000_000001",
        "frankfurt_000000_000002",
        "lindau_000000_000002",
    ]
