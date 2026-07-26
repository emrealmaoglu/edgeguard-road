"""Tests for the legal single-image loader and deterministic manifest."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeguard.data.single_image import (
    build_single_image_manifest,
    build_upstream_sample_manifest,
    load_rgb_image,
)
from edgeguard.serialization import canonical_json, sha256_file


def _write_rgb_fixture(path: Path) -> None:
    pixels = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_load_rgb_image_preserves_red_blue_channel_order(tmp_path: Path) -> None:
    image_path = tmp_path / "road.png"
    _write_rgb_fixture(image_path)

    image = load_rgb_image(image_path)

    assert image.shape == (1, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [255, 0, 0]
    assert image[0, 1].tolist() == [0, 0, 255]


def test_single_image_manifest_is_deterministic_and_root_independent(tmp_path: Path) -> None:
    image_root = tmp_path / "approved"
    image_root.mkdir()
    image_path = image_root / "road.png"
    _write_rgb_fixture(image_path)

    first = build_single_image_manifest(
        image_path,
        image_root=image_root,
        sample_id="legal-road-001",
        source_reference="human-owned fixture",
        license_reference="owner approved research use",
    )
    second = build_single_image_manifest(
        image_path,
        image_root=image_root,
        sample_id="legal-road-001",
        source_reference="human-owned fixture",
        license_reference="owner approved research use",
    )

    assert canonical_json(first) == canonical_json(second)
    assert first["relative_path"] == "road.png"
    assert first["original_shape"] == [1, 2, 3]
    assert str(image_root) not in canonical_json(first)
    assert "created_at" not in first


def test_single_image_manifest_rejects_path_outside_approved_root(tmp_path: Path) -> None:
    image_root = tmp_path / "approved"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    _write_rgb_fixture(outside)

    with pytest.raises(ValueError, match="inside the approved image root"):
        build_single_image_manifest(
            outside,
            image_root=image_root,
            sample_id="outside",
            source_reference="test",
            license_reference="test",
        )


def test_load_rgb_image_rejects_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="could not decode"):
        load_rgb_image(corrupt)


def test_upstream_sample_manifest_records_required_provenance(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    sample_dir = checkout / "samples"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "road.png"
    _write_rgb_fixture(image_path)

    manifest = build_upstream_sample_manifest(
        image_path,
        checkout_root=checkout,
        sample_id="pidnet-upstream-sample-primary",
        upstream_repository="https://github.com/XuJiacong/PIDNet.git",
        upstream_commit="a" * 40,
        source_access_date="2026-07-26",
        expected_relative_path="samples/road.png",
        expected_filename="road.png",
        expected_sha256=sha256_file(image_path),
        expected_shape=(1, 2, 3),
    )

    assert manifest["upstream_repository"] == "https://github.com/XuJiacong/PIDNet.git"
    assert manifest["upstream_commit"] == "a" * 40
    assert manifest["relative_path"] == "samples/road.png"
    assert manifest["filename"] == "road.png"
    assert manifest["original_shape"] == [1, 2, 3]
    assert manifest["access_date"] == "2026-07-26"
    assert manifest["usage_scope"] == "noncommercial_internal_plumbing"
    assert manifest["dataset_role"] == "plumbing_only"
    assert manifest["redistribution_permitted"] is False


def test_upstream_sample_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    sample_dir = checkout / "samples"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "road.png"
    _write_rgb_fixture(image_path)

    with pytest.raises(ValueError, match="sample SHA-256 mismatch"):
        build_upstream_sample_manifest(
            image_path,
            checkout_root=checkout,
            sample_id="pidnet-upstream-sample-primary",
            upstream_repository="https://github.com/XuJiacong/PIDNet.git",
            upstream_commit="a" * 40,
            source_access_date="2026-07-26",
            expected_relative_path="samples/road.png",
            expected_filename="road.png",
            expected_sha256="0" * 64,
            expected_shape=(1, 2, 3),
        )
