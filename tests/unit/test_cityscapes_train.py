"""Tests for Cityscapes Fine train preparation and split analysis."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from edgeguard.data.cityscapes_train import (
    analyze_prepared_train,
    build_split_candidates,
    build_split_comparison,
    discover_cityscapes_train,
    encode_train_ids_png,
    generate_train_id_masks,
    load_source_label_ids,
)
from scripts import prepare_cityscapes

REPO_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY = REPO_ROOT / "configs/dataset/ontology_v1.yaml"
COMMIT = "1" * 40


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def _sample_arrays(index: int, *, shape: tuple[int, int] = (4, 6)) -> tuple[bytes, bytes]:
    image = np.full((*shape, 3), index, dtype=np.uint8)
    labels = np.full(shape, 7 + index % 2, dtype=np.uint8)
    labels[0, 0] = 0
    labels[-1, -1] = 33 - index % 3
    return _png_bytes(image, "RGB"), _png_bytes(labels, "L")


def _write_archives(
    root: Path,
    *,
    sample_ids: tuple[str, ...] = (
        "alpha_000000_000001",
        "alpha_000001_000001",
        "beta_000000_000001",
        "beta_000001_000001",
        "gamma_000000_000001",
        "gamma_000001_000001",
    ),
    missing_label: str | None = None,
    geometry_mismatch: str | None = None,
    unknown_label: str | None = None,
) -> tuple[Path, Path]:
    left = root / prepare_cityscapes.LEFT_ARCHIVE_NAME
    labels = root / prepare_cityscapes.LABEL_ARCHIVE_NAME
    with ZipFile(left, "w") as archive:
        archive.writestr("leftImg8bit/val/ignored/ignored_000000_000001_leftImg8bit.png", b"x")
        for index, sample_id in enumerate(sample_ids):
            city = sample_id.split("_", maxsplit=1)[0]
            image, _label = _sample_arrays(index)
            archive.writestr(f"leftImg8bit/train/{city}/{sample_id}_leftImg8bit.png", image)
    with ZipFile(labels, "w") as archive:
        archive.writestr("gtFine/val/ignored/ignored_000000_000001_gtFine_labelIds.png", b"x")
        for index, sample_id in enumerate(sample_ids):
            if sample_id == missing_label:
                continue
            city = sample_id.split("_", maxsplit=1)[0]
            _image, label = _sample_arrays(
                index, shape=(3, 6) if sample_id == geometry_mismatch else (4, 6)
            )
            if sample_id == unknown_label:
                array = np.full((4, 6), 34, dtype=np.uint8)
                label = _png_bytes(array, "L")
            archive.writestr(f"gtFine/train/{city}/{sample_id}_gtFine_labelIds.png", label)
            archive.writestr(f"gtFine/train/{city}/{sample_id}_gtFine_color.png", b"ignored")
    return left, labels


def _patch_archive_hashes(monkeypatch: pytest.MonkeyPatch, left: Path, labels: Path) -> None:
    monkeypatch.setattr(
        prepare_cityscapes,
        "LEFT_ARCHIVE_SHA256",
        hashlib.sha256(left.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_cityscapes,
        "LABEL_ARCHIVE_SHA256",
        hashlib.sha256(labels.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(prepare_cityscapes, "MIN_FREE_MARGIN_BYTES", 0)


def _extract_sources(root: Path, left: Path, labels: Path) -> None:
    with ZipFile(left) as left_archive, ZipFile(labels) as label_archive:
        left_infos = prepare_cityscapes._validated_infos(left_archive)
        label_infos = prepare_cityscapes._validated_infos(label_archive)
        root.mkdir(parents=True)
        prepare_cityscapes._extract_selected(
            left_archive, prepare_cityscapes._selected_train_images(left_infos), root
        )
        prepare_cityscapes._extract_selected(
            label_archive, prepare_cityscapes._selected_train_labels(label_infos), root
        )


def test_train_archive_selection_is_train_only_and_preserves_label_ids(tmp_path: Path) -> None:
    left, labels = _write_archives(tmp_path)
    prepared = tmp_path / "prepared"

    _extract_sources(prepared, left, labels)

    assert len(list(prepared.glob("leftImg8bit/train/*/*.png"))) == 6
    assert len(list(prepared.glob("gtFine/train/labelIds/*/*.png"))) == 6
    assert not list(prepared.rglob("val"))
    assert not list(prepared.rglob("*_gtFine_color.png"))


def test_train_archive_rejects_unexpected_train_image_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "unexpected.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("leftImg8bit/train/alpha/readme.txt", "unexpected")

    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="unexpected"):
        prepare_cityscapes._selected_train_images(prepare_cityscapes._validated_infos(archive))


def test_train_discovery_rejects_missing_pair(tmp_path: Path) -> None:
    left, labels = _write_archives(tmp_path, missing_label="alpha_000000_000001")
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)

    with pytest.raises(ValueError, match="pairing mismatch"):
        discover_cityscapes_train(prepared, require_train_ids=False)


def test_train_discovery_parses_atomic_city_sequence_groups(tmp_path: Path) -> None:
    left, labels = _write_archives(tmp_path)
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)

    samples = discover_cityscapes_train(prepared, require_train_ids=False)

    assert samples[0].group_id == "alpha_000000"
    assert len({sample.group_id for sample in samples}) == 6


def test_train_id_encoding_is_byte_deterministic() -> None:
    mask = np.array([[0, 1, 18], [255, 7, 9]], dtype=np.uint8)

    assert encode_train_ids_png(mask) == encode_train_ids_png(mask.copy())


def test_train_generation_rejects_unknown_source_label(tmp_path: Path) -> None:
    left, labels = _write_archives(tmp_path, unknown_label="alpha_000000_000001")
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)

    with pytest.raises(ValueError, match="unknown Cityscapes source label IDs"):
        generate_train_id_masks(prepared)


def test_source_label_loader_rejects_corrupt_png(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a PNG")

    with pytest.raises(ValueError, match="could not decode"):
        load_source_label_ids(path)


def test_train_analysis_rejects_geometry_mismatch(tmp_path: Path) -> None:
    sample_ids = (
        "alpha_000000_000001",
        "beta_000000_000001",
        "gamma_000000_000001",
    )
    left, labels = _write_archives(
        tmp_path,
        sample_ids=sample_ids,
        geometry_mismatch="alpha_000000_000001",
    )
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)

    with pytest.raises(ValueError, match="geometry mismatch"):
        generate_train_id_masks(prepared)
        analyze_prepared_train(prepared)


def test_split_candidates_are_deterministic_nonempty_and_leakage_free(
    tmp_path: Path,
) -> None:
    left, labels = _write_archives(tmp_path)
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)
    generate_train_id_masks(prepared)
    analysis = analyze_prepared_train(prepared)

    first = build_split_candidates(analysis)
    second = build_split_candidates(analysis)

    assert first == second
    assert first["selection_status"] == "recommended_pending_human_approval"
    assert len(first["candidates"]) == 3
    assert analysis["heuristic_rare_class_ids"]
    for candidate in first["candidates"]:
        assert candidate["leakage_validated"] is True
        assert all(
            candidate["roles"][role]["sample_count"] > 0
            for role in ("train_fit", "train_select", "train_calibration")
        )
        group_roles = {row["group_id"]: row["role"] for row in candidate["group_manifest"]}
        assert len(group_roles) == len(candidate["group_manifest"])


def test_split_comparison_omits_full_sample_manifests(tmp_path: Path) -> None:
    left, labels = _write_archives(tmp_path)
    prepared = tmp_path / "prepared"
    _extract_sources(prepared, left, labels)
    generate_train_id_masks(prepared)

    comparison = build_split_comparison(build_split_candidates(analyze_prepared_train(prepared)))

    assert comparison["selection_status"] == "recommended_pending_human_approval"
    assert "sample_manifest" not in json.dumps(comparison)


def test_full_train_preparation_is_root_free_and_idempotently_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, labels = _write_archives(tmp_path)
    _patch_archive_hashes(monkeypatch, left, labels)
    destination = tmp_path / "external" / "datasets" / "cityscapes" / "fine" / "v1"
    manifests = tmp_path / "external" / "manifests" / "cityscapes" / "fine" / "v1"
    work = tmp_path / "work"

    result = prepare_cityscapes.prepare_cityscapes_train(
        left,
        labels,
        destination,
        manifests,
        work,
        preparation_git_commit=COMMIT,
        ontology_config=ONTOLOGY,
        now=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    verified = prepare_cityscapes.prepare_cityscapes_train(
        left,
        labels,
        destination,
        manifests,
        work,
        preparation_git_commit=COMMIT,
        ontology_config=ONTOLOGY,
        verify_only=True,
    )

    assert result == verified
    assert result["image_count"] == 6
    assert result["selection_status"] == "recommended_pending_human_approval"
    manifest_text = (manifests / "dataset_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    class_frequency = json.loads((manifests / "class_frequency.json").read_text(encoding="utf-8"))
    assert "sample_ids" not in class_frequency["global"]
    assert all("sample_ids" not in city for city in class_frequency["cities"].values())
    with ZipFile(manifests / result["evidence_package_filename"]) as archive:
        names = archive.namelist()
    assert not any("leftImg8bit" in name or "trainIds" in name for name in names)


def test_dataset_and_split_identities_exclude_time_and_absolute_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, labels = _write_archives(tmp_path)
    _patch_archive_hashes(monkeypatch, left, labels)
    outputs: list[tuple[Path, dict[str, object]]] = []

    for suffix, prepared_at in (
        ("first", datetime(2026, 7, 27, tzinfo=timezone.utc)),
        ("second", datetime(2026, 7, 28, tzinfo=timezone.utc)),
    ):
        manifests = tmp_path / suffix / "manifests"
        prepare_cityscapes.prepare_cityscapes_train(
            left,
            labels,
            tmp_path / suffix / "dataset",
            manifests,
            tmp_path / suffix / "work",
            preparation_git_commit=COMMIT,
            ontology_config=ONTOLOGY,
            now=prepared_at,
        )
        receipt = json.loads((manifests / "preparation_receipt.json").read_text(encoding="utf-8"))
        outputs.append((manifests, receipt))

    first, second = outputs
    deterministic_files = (
        "dataset_manifest.json",
        "split_candidate_comparison.json",
        "split_candidates/CSF-SPLIT-A.samples.json",
        "split_candidates/CSF-SPLIT-A.groups.json",
        "split_candidates/CSF-SPLIT-B.samples.json",
        "split_candidates/CSF-SPLIT-B.groups.json",
        "split_candidates/CSF-SPLIT-C.samples.json",
        "split_candidates/CSF-SPLIT-C.groups.json",
    )
    assert all(
        (first[0] / name).read_bytes() == (second[0] / name).read_bytes()
        for name in deterministic_files
    )
    assert first[1]["prepared_at"] != second[1]["prepared_at"]


def test_train_preparation_refuses_nonempty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, labels = _write_archives(tmp_path)
    _patch_archive_hashes(monkeypatch, left, labels)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "partial.txt").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="destination already exists"):
        prepare_cityscapes.prepare_cityscapes_train(
            left,
            labels,
            destination,
            tmp_path / "manifests",
            tmp_path / "work",
            preparation_git_commit=COMMIT,
            ontology_config=ONTOLOGY,
        )


def test_verify_only_rejects_partial_prepared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, labels = _write_archives(tmp_path)
    _patch_archive_hashes(monkeypatch, left, labels)
    destination = tmp_path / "destination"
    manifests = tmp_path / "manifests"
    destination.mkdir()
    manifests.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        prepare_cityscapes.prepare_cityscapes_train(
            left,
            labels,
            destination,
            manifests,
            tmp_path / "work",
            preparation_git_commit=COMMIT,
            ontology_config=ONTOLOGY,
            verify_only=True,
        )
