from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from edgeguard.rescue.colab_data import (
    create_dataset_bundle,
    inventory_colab_data,
    load_colab_data_access,
    stage_dataset_bundles,
)

PLAN = Path("configs/dataset/colab_data_access_v1.yaml")


def _fixture_plan(tmp_path: Path) -> tuple[dict[str, object], Path]:
    plan = load_colab_data_access(PLAN)
    mutated = deepcopy(plan)
    mutated["storage"]["colab_ephemeral_limit_gib"] = 1
    mutated["storage"]["reserve_gib"] = 0
    mutated["storage"]["maximum_staged_gib"] = 1
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    return load_colab_data_access(path), tmp_path / "drive"


def test_colab_access_plan_has_only_official_routes_and_bounded_storage() -> None:
    plan = load_colab_data_access(PLAN)
    assert plan["storage"]["colab_ephemeral_limit_gib"] == 200
    assert plan["storage"]["maximum_staged_gib"] == 175
    assert set(("cityscapes", "bdd100k", "idd20k")).issubset(plan["datasets"])
    assert all(
        str(record["official_url"]).startswith("https://") for record in plan["datasets"].values()
    )
    excluded = {row["source_id"] for row in plan["excluded_sources"]}
    assert "huggingface-segments-sidewalk-semantic" in excluded
    assert "third_party_dataset_mirrors" in excluded


def test_inventory_bundle_and_stage_round_trip(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    prepared = drive / "EdgeGuard/datasets/cityscapes"
    for relative in plan["datasets"]["cityscapes"]["required_paths"]:
        directory = prepared / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fixture.bin").write_bytes(str(relative).encode())
    inventory = inventory_colab_data(plan, drive)
    cityscapes = next(row for row in inventory["datasets"] if row["dataset_id"] == "cityscapes")
    assert cityscapes["state"] == "prepared"
    receipt = create_dataset_bundle(plan, drive, "cityscapes")
    assert receipt["status"] == "created"
    assert create_dataset_bundle(plan, drive, "cityscapes")["status"] == "reused"
    staged = stage_dataset_bundles(plan, drive, tmp_path / "content", ("cityscapes",))
    assert staged["datasets"] == [{"dataset_id": "cityscapes", "status": "staged_verified"}]
    assert (tmp_path / "content/cityscapes/leftImg8bit/train/fixture.bin").is_file()


def test_inventory_hashes_present_archives_and_checks_published_md5(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    package = plan["datasets"]["bdd100k"]["packages"][0]
    archive = drive / "EdgeGuard/archives/bdd100k" / package["filename"]
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"official-fixture")
    package["published_md5"] = "a" * 32
    inventory = inventory_colab_data(plan, drive, hash_archives=True)
    bdd = next(row for row in inventory["datasets"] if row["dataset_id"] == "bdd100k")
    row = bdd["packages"][0]
    assert len(row["sha256"]) == 64
    assert len(row["md5"]) == 32
    assert row["published_md5_matches"] is False


def test_bundle_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    prepared = drive / "EdgeGuard/datasets/cityscapes"
    for relative in plan["datasets"]["cityscapes"]["required_paths"]:
        (prepared / relative).mkdir(parents=True, exist_ok=True)
    create_dataset_bundle(plan, drive, "cityscapes")
    receipt_path = drive / "EdgeGuard/bundles/cityscapes.prepared.tar.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_bytes"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        stage_dataset_bundles(plan, drive, tmp_path / "content", ("cityscapes",))


def test_stage_rejects_duplicate_dataset_ids(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    with pytest.raises(ValueError, match="non-empty and unique"):
        stage_dataset_bundles(plan, drive, tmp_path / "content", ("cityscapes", "cityscapes"))
