"""Tests for read-only zero-redownload storage inventory."""

import json
from pathlib import Path

from scripts.data.inventory_edgeguard_storage import inventory_edgeguard_storage


def test_inventory_reports_reusable_cityscapes_without_hashing_or_writing(tmp_path: Path) -> None:
    for relative in (
        "datasets/cityscapes/fine/v1",
        "manifests/cityscapes/fine/v1",
        "manifests/cityscapes/fine/v1/split-policy-v1",
        "datasets/cityscapes/fine/bundles",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "manifests/cityscapes/fine/v1/dataset_manifest.json",
        "manifests/cityscapes/fine/v1/group_summary.json",
        "manifests/cityscapes/fine/v1/split-policy-v1/policy_selected_split.json",
    ):
        (tmp_path / relative).write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "datasets/cityscapes/fine/bundles/fixture.tar.gz"
    bundle.write_bytes(b"bundle")
    Path(f"{bundle}.receipt.json").write_text(json.dumps({"verified": True}), encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = inventory_edgeguard_storage(tmp_path)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["cityscapes_fine"]["reusable"] is True
    assert result["cityscapes_fine"]["expected_download_bytes"] == 0
    assert result["cityscapes_fine"]["expected_drive_write_bytes"] == 0
    assert result["cityscapes_fine"]["expected_local_staging_bytes"] == len(b"bundle")
    assert result["hash_policy"] == "metadata_and_sizes_only"
    assert before == after


def test_inventory_blocks_missing_assets_but_still_predicts_zero_download(tmp_path: Path) -> None:
    result = inventory_edgeguard_storage(tmp_path)

    assert result["status"] == "blocked_missing_verified_assets"
    assert result["cityscapes_fine"]["reusable"] is False
    assert result["cityscapes_fine"]["expected_download_bytes"] == 0
    assert result["missing_required_files"]
