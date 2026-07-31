from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from edgeguard.rescue.colab_artifacts import (
    create_campaign_snapshot,
    create_review_package,
    restore_campaign_snapshot,
)

COMMIT = "a" * 40


def test_review_package_is_bounded_and_excludes_models_and_data(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "reports/figures").mkdir(parents=True)
    (work / "reports/figures/classes.png").write_bytes(b"png")
    (work / "manifests").mkdir()
    (work / "manifests/source.json").write_text("{}", encoding="utf-8")
    (work / "checkpoints").mkdir()
    (work / "checkpoints/model.pth").write_bytes(b"model")
    output = tmp_path / "review.zip"
    receipt = create_review_package(
        work, output, campaign_id="semantic-first-v1", project_commit=COMMIT
    )
    assert receipt["file_count"] == 2
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "reports/figures/classes.png" in names
    assert "manifests/source.json" in names
    assert "checkpoints/model.pth" not in names
    assert "REVIEW_MANIFEST.json" in names


def test_snapshot_round_trip_uses_only_explicit_roots(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "manifests").mkdir(parents=True)
    (work / "manifests/frozen.json").write_text("{}", encoding="utf-8")
    (work / "staged-data").mkdir()
    (work / "staged-data/image.png").write_bytes(b"data")
    snapshot = tmp_path / "campaign.tar.gz"
    create_campaign_snapshot(
        work,
        snapshot,
        relative_paths=["manifests"],
        campaign_id="semantic-first-v1",
        project_commit=COMMIT,
    )
    restored = tmp_path / "restored"
    result = restore_campaign_snapshot(snapshot, restored)
    assert result["status"] == "restored_verified"
    assert (restored / "manifests/frozen.json").is_file()
    assert not (restored / "staged-data").exists()
    (work / "manifests/second.json").write_text("{}", encoding="utf-8")
    create_campaign_snapshot(
        work,
        snapshot,
        relative_paths=["manifests"],
        campaign_id="semantic-first-v1",
        project_commit=COMMIT,
    )
    previous = snapshot.with_name(f"{snapshot.stem}.previous{snapshot.suffix}")
    assert previous.is_file()
    assert previous.with_suffix(previous.suffix + ".receipt.json").is_file()


def test_restore_rejects_traversal_even_with_valid_receipt(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    snapshot = tmp_path / "campaign.tar.gz"
    create_campaign_snapshot(
        work,
        snapshot,
        relative_paths=["missing"],
        campaign_id="semantic-first-v1",
        project_commit=COMMIT,
    )
    receipt_path = snapshot.with_suffix(snapshot.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with tarfile.open(snapshot, "w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 0
        archive.addfile(info)
    receipt["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt hash mismatch|snapshot hash mismatch"):
        restore_campaign_snapshot(snapshot, tmp_path / "restored")
