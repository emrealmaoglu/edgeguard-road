"""Tests for minimal evaluation artifact packaging."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.package_eval_artifacts import REQUIRED_FILES, package_eval_artifacts


def _complete_output(root: Path) -> None:
    root.mkdir()
    for name in REQUIRED_FILES:
        (root / name).write_text("{}\n", encoding="utf-8")


def test_package_eval_artifacts_uses_relative_sorted_members(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    _complete_output(output)
    visual = output / "visuals/sample"
    visual.mkdir(parents=True)
    (visual / "prediction.png").write_bytes(b"png fixture")
    archive_path = tmp_path / "bundle.zip"

    result = package_eval_artifacts(output, archive_path)

    with ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
    assert result["filename"] == "bundle.zip"
    assert result["file_count"] == len(REQUIRED_FILES) + 1


def test_package_eval_artifacts_rejects_incomplete_output(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()

    with pytest.raises(ValueError, match="incomplete"):
        package_eval_artifacts(output, tmp_path / "bundle.zip")


def test_package_eval_artifacts_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    _complete_output(output)
    archive_path = tmp_path / "bundle.zip"
    archive_path.write_bytes(b"existing")

    with pytest.raises(ValueError, match="overwrite"):
        package_eval_artifacts(output, archive_path)
