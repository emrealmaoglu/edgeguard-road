from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from edgeguard.rescue.colab_data import (
    copy_archive_to_local,
    create_dataset_bundle,
    initialize_drive_layout,
    inventory_colab_data,
    load_colab_data_access,
    preparation_disk_budget,
    stage_dataset_bundles,
)
from edgeguard.serialization import sha256_file

PLAN = Path("configs/dataset/colab_data_access_v1.yaml")


def test_colab_inventory_cli_default_plan_is_cwd_independent(tmp_path: Path) -> None:
    script = Path("scripts/prepare_colab_data.py").resolve()
    output = tmp_path / "inventory.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--drive-root",
            str(tmp_path / "drive"),
            "--output",
            str(output),
            "inventory",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["record_type"] == (
        "edgeguard_colab_data_inventory"
    )


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


def test_initialize_drive_layout_creates_scientific_and_review_roots(
    tmp_path: Path,
) -> None:
    plan, drive = _fixture_plan(tmp_path)
    paths = initialize_drive_layout(plan, drive)
    for name in (
        "archives",
        "bundles",
        "manifests",
        "campaigns",
        "downloads",
        "quarantine",
        "source",
        "private_inputs",
        "failures",
        "prepared",
        "runtime_cache",
        "review_packages",
    ):
        assert Path(paths[name]).is_dir()
    assert (Path(paths["prepared"]) / "v2").is_dir()
    assert (Path(paths["manifests"]) / "v2").is_dir()
    assert (Path(paths["quarantine"]) / "kaggle/bdd100k").is_dir()


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
    assert staged["datasets"] == [
        {
            "dataset_id": "cityscapes",
            "status": "staged_verified",
            "bundle_profile": "canonical_v1:official",
        }
    ]
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


def test_inventory_reuses_archive_digest_receipt_without_rereading_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, drive = _fixture_plan(tmp_path)
    package = plan["datasets"]["cityscapes"]["packages"][0]
    archive = drive / "EdgeGuard/private_inputs" / package["filename"]
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"stable-drive-archive")
    first = inventory_colab_data(plan, drive, hash_archives=True)
    first_row = next(row for row in first["datasets"] if row["dataset_id"] == "cityscapes")[
        "packages"
    ][0]
    assert first_row["hash_status"] == "computed"

    def unexpected_hash(_path: Path) -> tuple[str, str]:
        raise AssertionError("a stat-stable Drive archive must use its pinned digest receipt")

    monkeypatch.setattr("edgeguard.rescue.colab_data._file_digests", unexpected_hash)
    second = inventory_colab_data(plan, drive, hash_archives=True)
    second_row = next(row for row in second["datasets"] if row["dataset_id"] == "cityscapes")[
        "packages"
    ][0]
    assert second_row["hash_status"] == "cached"
    assert second_row["sha256"] == first_row["sha256"]


def test_inventory_accepts_hash_pinned_legacy_private_input(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    package = plan["datasets"]["cityscapes"]["packages"][0]
    archive = drive / "EdgeGuard/private_inputs" / package["filename"]
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"legacy-official-fixture")
    package["published_sha256"] = sha256_file(archive)
    inventory = inventory_colab_data(plan, drive, hash_archives=True)
    cityscapes = next(row for row in inventory["datasets"] if row["dataset_id"] == "cityscapes")
    row = cityscapes["packages"][0]
    assert row["location_profile"] == "legacy_private_inputs"
    assert row["published_sha256_matches"] is True


def test_inventory_finds_all_current_private_inputs_without_moving_them(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    private = drive / "EdgeGuard/private_inputs"
    private.mkdir(parents=True)
    idd_package = plan["datasets"]["idd20k"]["packages"][0]
    (private / idd_package["filename"]).write_bytes(b"idd-official-fixture")
    (private / "bdd100k.zip").write_bytes(b"bdd-kaggle-fixture")

    inventory = inventory_colab_data(plan, drive)
    idd = next(row for row in inventory["datasets"] if row["dataset_id"] == "idd20k")
    bdd = next(row for row in inventory["datasets"] if row["dataset_id"] == "bdd100k")

    assert idd["packages"][0]["location_profile"] == "private_inputs"
    assert bdd["engineering_packages"] == [
        {
            "filename": "bdd100k.zip",
            "purpose": "Kaggle mirror already uploaded for preparation/audit/smoke only",
            "source_profile": "kaggle_mirror",
            "scientific_eligible": False,
            "path": str(private / "bdd100k.zip"),
            "present": True,
            "byte_size": len(b"bdd-kaggle-fixture"),
            "sha256": None,
            "md5": None,
            "hash_status": "not_requested",
            "hash_error": None,
            "location_profile": "private_inputs",
        }
    ]


def test_inventory_records_drive_hash_read_error_without_losing_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, drive = _fixture_plan(tmp_path)
    package = plan["datasets"]["cityscapes"]["packages"][0]
    archive = drive / "EdgeGuard/private_inputs" / package["filename"]
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"mounted-drive-fixture")

    def fail_digest(_path: Path) -> tuple[str, str]:
        raise OSError("transient mounted Drive read failure")

    monkeypatch.setattr("edgeguard.rescue.colab_data._file_digests", fail_digest)
    inventory = inventory_colab_data(plan, drive, hash_archives=True)
    cityscapes = next(row for row in inventory["datasets"] if row["dataset_id"] == "cityscapes")
    row = cityscapes["packages"][0]
    assert row["present"] is True
    assert row["hash_status"] == "read_error"
    assert "transient mounted Drive" in row["hash_error"]
    assert row["sha256"] is None


def test_archive_copy_retries_and_publishes_only_complete_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "drive/archive.zip"
    source.parent.mkdir()
    source.write_bytes(b"archive-payload")
    destination = tmp_path / "content/archive.zip"
    real_copy = shutil.copyfileobj
    calls = 0

    def flaky_copy(input_stream: object, output_stream: object, *, length: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fixture Drive interruption")
        real_copy(input_stream, output_stream, length=length)  # type: ignore[arg-type]

    monkeypatch.setattr("edgeguard.rescue.colab_data.shutil.copyfileobj", flaky_copy)
    receipt = copy_archive_to_local(source, destination, attempts=3)
    assert receipt["attempts"] == 2
    assert destination.read_bytes() == source.read_bytes()
    assert not (destination.parent / ".archive.zip.partial").exists()


def test_preparation_budget_uses_archive_multiplier_and_reserved_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _drive = _fixture_plan(tmp_path)
    plan["storage"]["preparation_archive_multiplier"] = 3.0
    plan["storage"]["preparation_reserve_gib"] = 1
    plan["storage"]["colab_ephemeral_limit_gib"] = 10
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"x" * 1024)
    usage = SimpleNamespace(total=10 * 1024**3, used=2 * 1024**3, free=8 * 1024**3)
    monkeypatch.setattr("edgeguard.rescue.colab_data.shutil.disk_usage", lambda _root: usage)

    budget = preparation_disk_budget(plan, (archive,), tmp_path / "content")

    assert budget["passes"] is True
    assert budget["estimated_peak_bytes"] == 3 * 1024 + 1024**3


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


def test_bundle_can_stream_from_ephemeral_prepared_root(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    source = tmp_path / "ephemeral-cityscapes"
    for relative in plan["datasets"]["cityscapes"]["required_paths"]:
        directory = source / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fixture.bin").write_bytes(str(relative).encode())
    (source / "preparation_receipt.json").write_text("{}", encoding="utf-8")
    receipt = create_dataset_bundle(plan, drive, "cityscapes", source_root=source)
    assert receipt["status"] == "created"
    assert len(receipt["source_preparation_receipt_sha256"]) == 64


def test_stage_reuses_pinned_legacy_cityscapes_bundle_without_moving_drive_data(
    tmp_path: Path,
) -> None:
    plan, drive = _fixture_plan(tmp_path)
    legacy = plan["storage"]["legacy_compatibility"]["cityscapes"]
    bundle_root = drive / "EdgeGuard" / legacy["bundle_directory"]
    bundle_root.mkdir(parents=True)
    source = tmp_path / "legacy-source"
    for relative in legacy["required_paths"]:
        directory = source / relative
        directory.mkdir(parents=True)
        (directory / "fixture.bin").write_bytes(str(relative).encode())
    bundle = bundle_root / legacy["bundle_filename"]
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted(source.rglob("*.bin")):
            archive.add(path, arcname=path.relative_to(source))
    source_bytes = sum(path.stat().st_size for path in source.rglob("*.bin"))
    legacy.update(
        {
            "bundle_sha256": sha256_file(bundle),
            "byte_size": bundle.stat().st_size,
            "source_bytes": source_bytes,
            "file_count": 2,
        }
    )
    receipt = {
        "record_type": "cityscapes_training_bundle_receipt",
        "filename": legacy["bundle_filename"],
        "byte_size": legacy["byte_size"],
        "sha256": legacy["bundle_sha256"],
        "dataset_manifest_sha256": legacy["dataset_manifest_sha256"],
        "split_manifest_sha256": legacy["split_manifest_sha256"],
        "file_count": legacy["file_count"],
    }
    (bundle_root / legacy["receipt_filename"]).write_text(json.dumps(receipt), encoding="utf-8")
    inventory = inventory_colab_data(plan, drive)
    cityscapes = next(row for row in inventory["datasets"] if row["dataset_id"] == "cityscapes")
    assert cityscapes["state"] == "legacy_bundle_ready"
    assert cityscapes["legacy_compatibility"]["usable_for_training_staging"] is True
    staged = stage_dataset_bundles(plan, drive, tmp_path / "content", ("cityscapes",))
    assert staged["datasets"][0]["bundle_profile"] == "legacy_cityscapes_v1"
    assert (tmp_path / "content/cityscapes/leftImg8bit/train/fixture.bin").is_file()


def test_kaggle_bdd_bundle_is_physically_separate_and_science_blocked(tmp_path: Path) -> None:
    plan, drive = _fixture_plan(tmp_path)
    source = tmp_path / "bdd-kaggle"
    for relative in plan["datasets"]["bdd100k"]["required_paths"]:
        directory = source / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fixture.bin").write_bytes(str(relative).encode())
    (source / "preparation_receipt.json").write_text(
        json.dumps(
            {
                "dataset_id": "bdd100k",
                "source_profile": "kaggle_mirror",
                "scientific_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    receipt = create_dataset_bundle(plan, drive, "bdd100k", source_root=source)
    assert receipt["filename"] == "bdd100k.kaggle_mirror.prepared.tar"
    assert receipt["scientific_eligible"] is False
    with pytest.raises(ValueError, match="blocked from scientific staging"):
        stage_dataset_bundles(plan, drive, tmp_path / "scientific", ("bdd100k",))
    staged = stage_dataset_bundles(
        plan,
        drive,
        tmp_path / "smoke",
        ("bdd100k",),
        allow_ineligible=True,
    )
    assert staged["datasets"][0]["bundle_profile"] == "canonical_v1:kaggle_mirror"
