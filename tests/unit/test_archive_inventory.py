from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from edgeguard.rescue.archive_inventory import (
    build_inventory_report,
    classify_entry,
    inspect_path,
    inventory_identity,
    known_role_for_filename,
)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _rgb_png() -> bytes:
    return _png_bytes((np.arange(10 * 12 * 3) % 255).reshape(10, 12, 3).astype("uint8"))


def _label_png() -> bytes:
    mask = np.zeros((8, 8), dtype="uint8")
    mask[0:2, :] = 1
    mask[2:4, :] = 2
    return _png_bytes(mask)


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("images/foo.png", "image"),
        ("readme.txt", "document"),
        ("notes.md", "document"),
        ("run.sh", "script"),
        ("model.pt", "model_weights"),
        ("index.json", "structured_data"),
        ("archive.zip", "other"),
    ],
)
def test_classify_entry_is_extension_only(suffix: str, expected: str) -> None:
    assert classify_entry(suffix) == expected


def test_classify_entry_does_not_special_case_by_dataset_name() -> None:
    assert classify_entry("cityscapes/anything.png") == classify_entry("wilddash2/anything.png")
    assert classify_entry("cityscapes/anything.png") == classify_entry(
        "totally_unknown/anything.png"
    )


def _build_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images/img1.png", _rgb_png())
        archive.writestr("labels/mask1.png", _label_png())
        archive.writestr("labels/corrupt.png", b"not a real png")
        archive.writestr("README.txt", "hello")


def _build_tar(path: Path, tmp_path: Path) -> None:
    image_path = tmp_path / "img2.png"
    image_path.write_bytes(_rgb_png())
    with tarfile.open(path, "w:gz") as archive:
        archive.add(image_path, arcname="img2.png")


def test_inspect_path_is_exhaustive_not_sampled_for_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "toy.zip"
    _build_zip(zip_path)
    result = inspect_path(zip_path)
    assert result.entry_count == 4
    assert result.image_count == 3  # rgb, mask, and the corrupt attempted-image
    assert result.corrupt_count == 1
    assert result.classification_counts == {"image": 3, "document": 1}


def test_inspect_path_measures_real_per_class_pixel_histogram(tmp_path: Path) -> None:
    zip_path = tmp_path / "toy.zip"
    _build_zip(zip_path)
    result = inspect_path(zip_path)
    # 8x8 mask: rows 0-1 -> value 1 (16 px), rows 2-3 -> value 2 (16 px), rest -> value 0 (32 px)
    assert result.class_pixel_histogram == {"0": 32, "1": 16, "2": 16}
    assert result.class_image_histogram == {"0": 1, "1": 1, "2": 1}


def test_inspect_path_handles_tar_gz(tmp_path: Path) -> None:
    tar_path = tmp_path / "toy.tar.gz"
    _build_tar(tar_path, tmp_path)
    result = inspect_path(tar_path)
    assert result.file_type == "tar"
    assert result.entry_count == 1
    assert result.image_count == 1
    assert result.corrupt_count == 0


def test_inspect_path_handles_standalone_non_archive_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"\x00" * 128)
    result = inspect_path(checkpoint)
    assert result.file_type == "standalone"
    assert result.entry_count == 1
    assert result.classification_counts == {"model_weights": 1}
    assert result.byte_size == 128


def test_known_role_for_filename_matches_declared_package_exactly() -> None:
    access_plan = {
        "datasets": {
            "cityscapes": {
                "campaign_role": "train_fit",
                "packages": [
                    {"filename": "leftImg8bit_trainvaltest.zip", "purpose": "images"},
                ],
                "engineering_packages": [],
            }
        },
        "excluded_sources": [],
    }
    role = known_role_for_filename("leftImg8bit_trainvaltest.zip", access_plan)
    assert role is not None
    assert role["dataset_id"] == "cityscapes"
    assert role["campaign_role"] == "train_fit"


def test_known_role_for_filename_is_none_for_undeclared_file_not_guessed() -> None:
    access_plan = {
        "datasets": {
            "cityscapes": {
                "campaign_role": "train_fit",
                "packages": [{"filename": "leftImg8bit_trainvaltest.zip", "purpose": "images"}],
            }
        },
        "excluded_sources": [],
    }
    # A plausibly-related but undeclared file must NOT be guessed into a role.
    assert known_role_for_filename("leftImg8bit_trainvaltest_extra.zip", access_plan) is None
    assert known_role_for_filename("some_never_seen_file.zip", access_plan) is None


def test_inventory_identity_changes_when_folder_contents_change(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    (private_inputs / "a.zip").write_bytes(b"x")
    identity_before = inventory_identity(private_inputs)
    (private_inputs / "b.zip").write_bytes(b"y")
    identity_after = inventory_identity(private_inputs)
    assert identity_before != identity_after


def test_inventory_identity_stable_for_unchanged_folder(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    (private_inputs / "a.zip").write_bytes(b"x")
    assert inventory_identity(private_inputs) == inventory_identity(private_inputs)


def test_build_inventory_report_covers_every_file_and_writes_outputs(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    _build_zip(private_inputs / "toy.zip")
    _build_tar(private_inputs / "toy.tar.gz", tmp_path)
    (private_inputs / "checkpoint.pt").write_bytes(b"\x00" * 64)

    access_plan_path = tmp_path / "access_plan.yaml"
    access_plan_path.write_text(
        yaml.safe_dump({"datasets": {}, "excluded_sources": []}), encoding="utf-8"
    )

    output_root = tmp_path / "out"
    report = build_inventory_report(private_inputs, output_root, access_plan_path=access_plan_path)

    assert report["record_type"] == "raw_archive_inventory"
    assert "scientific_status" not in report
    assert report["archive_count"] == 3
    assert {a["filename"] for a in report["archives"]} == {
        "toy.zip",
        "toy.tar.gz",
        "checkpoint.pt",
    }
    assert report["sampling"].startswith("exhaustive")
    assert (output_root / "dataset_inventory.json").is_file()
    assert (output_root / "dataset_inventory.md").is_file()


def test_build_inventory_report_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    (private_inputs / "toy.zip").write_bytes(b"")
    output_root = tmp_path / "out"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        build_inventory_report(private_inputs, output_root)


def _run_inventory_cli(private_inputs: Path, output_parent: Path, **extra: str) -> dict:
    script = Path("scripts/inventory_private_inputs.py").resolve()
    args = [
        sys.executable,
        str(script),
        "--private-inputs-root",
        str(private_inputs),
        "--output-parent",
        str(output_parent),
    ]
    for key, value in extra.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return dict(json.loads(completed.stdout))


def test_inventory_cli_scans_then_reuses_cache_on_second_run(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    _build_zip(private_inputs / "toy.zip")
    output_parent = tmp_path / "out"

    first = _run_inventory_cli(private_inputs, output_parent)
    assert first["cache_hit"] is False
    assert first["archive_count"] == 1

    second = _run_inventory_cli(private_inputs, output_parent)
    assert second["cache_hit"] is True
    assert second["inventory_identity_sha256"] == first["inventory_identity_sha256"]

    (private_inputs / "extra.zip").write_bytes(b"")
    third = _run_inventory_cli(private_inputs, output_parent)
    assert third["cache_hit"] is False
    assert third["inventory_identity_sha256"] != first["inventory_identity_sha256"]


def test_inventory_cli_zip_out_is_hash_verified(tmp_path: Path) -> None:
    private_inputs = tmp_path / "private_inputs"
    private_inputs.mkdir()
    _build_zip(private_inputs / "toy.zip")
    output_parent = tmp_path / "out"
    zip_out = tmp_path / "EdgeGuard_Data_Inventory.zip"

    _run_inventory_cli(private_inputs, output_parent, zip_out=str(zip_out))
    assert zip_out.is_file()
    with zipfile.ZipFile(zip_out) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"dataset_inventory.json", "dataset_inventory.md"}
