"""Tests for the project-specific Cityscapes val preparation script."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import prepare_cityscapes


def test_cityscapes_archive_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / prepare_cityscapes.LEFT_ARCHIVE_NAME
    archive.write_bytes(b"not the approved archive")

    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        prepare_cityscapes._verify_archive(
            archive,
            prepare_cityscapes.LEFT_ARCHIVE_NAME,
            prepare_cityscapes.LEFT_ARCHIVE_SHA256,
        )


def test_cityscapes_zip_rejects_unsafe_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="unsafe ZIP member"):
        prepare_cityscapes._validated_infos(archive)


def test_cityscapes_zip_rejects_duplicate_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("leftImg8bit/val/city/sample.png", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("leftImg8bit/val/city/sample.png", "second")

    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="duplicate ZIP member"):
        prepare_cityscapes._validated_infos(archive)


def test_cityscapes_preparation_refuses_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cityscapes-val"
    destination.mkdir()
    monkeypatch.setattr(prepare_cityscapes, "_verify_archive", lambda *_args: None)

    with pytest.raises(ValueError, match="destination already exists"):
        prepare_cityscapes.prepare_cityscapes_val(
            tmp_path / prepare_cityscapes.LEFT_ARCHIVE_NAME,
            tmp_path / prepare_cityscapes.LABEL_ARCHIVE_NAME,
            destination,
        )
