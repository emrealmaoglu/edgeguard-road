from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

import edgeguard.data.preparation as preparation
from edgeguard.data.preparation import prepare_dataset, render_idd_source_mask
from edgeguard.rescue.multidomain import discover_domain_samples


def _png_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format="PNG")
    return stream.getvalue()


def _jpg_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format="JPEG")
    return stream.getvalue()


def _write_cityscapes_archives(root: Path) -> tuple[Path, Path]:
    images = root / "leftImg8bit_trainvaltest.zip"
    labels = root / "gtFine_trainvaltest.zip"
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    source = np.full((4, 8), 7, dtype=np.uint8)
    with ZipFile(images, "w") as archive:
        for split, city in (("train", "alpha"), ("val", "beta")):
            identifier = f"{city}_000001_000002"
            archive.writestr(
                f"leftImg8bit/{split}/{city}/{identifier}_leftImg8bit.png", _png_bytes(rgb)
            )
    with ZipFile(labels, "w") as archive:
        for split, city in (("train", "alpha"), ("val", "beta")):
            identifier = f"{city}_000001_000002"
            archive.writestr(
                f"gtFine/{split}/{city}/{identifier}_gtFine_labelIds.png", _png_bytes(source)
            )
    return images, labels


def _write_bdd_kaggle(root: Path) -> Path:
    archive_path = root / "bdd100k.zip"
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    mask = np.arange(32, dtype=np.uint8).reshape(4, 8) % 19
    with ZipFile(archive_path, "w") as archive:
        for split in ("train", "val"):
            name = f"sequence-{split}"
            prefix = "bdd100k_seg/bdd100k/seg"
            archive.writestr(f"{prefix}/images/{split}/{name}.jpg", _jpg_bytes(rgb))
            archive.writestr(f"{prefix}/labels/{split}/{name}_train_id.png", _png_bytes(mask))
    return archive_path


def _write_acdc_archives(root: Path) -> tuple[Path, Path]:
    images = root / "rgb_anon_trainvaltest.zip"
    labels = root / "gt_trainval.zip"
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    mask = np.arange(32, dtype=np.uint8).reshape(4, 8) % 19
    with ZipFile(images, "w") as archive:
        archive.writestr("ACDC/rgb_anon/fog/val/GOPR/scene_rgb_anon.png", _png_bytes(rgb))
        archive.writestr("ACDC/rgb_anon/fog/train/GOPR/train_rgb_anon.png", _png_bytes(rgb))
    with ZipFile(labels, "w") as archive:
        archive.writestr("ACDC/gt/fog/val/GOPR/scene_gt_labelTrainIds.png", _png_bytes(mask))
        archive.writestr("ACDC/gt/fog/val/GOPR/scene_gt_invIds.png", _png_bytes(mask))
    return images, labels


def _add_tar_file(archive: tarfile.TarFile, name: str, contents: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    archive.addfile(info, io.BytesIO(contents))


def _write_idd_archives(root: Path) -> tuple[Path, Path]:
    paths = (root / "idd-20k-I.tar.gz", root / "idd-20k-II.tar.gz")
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    annotation = json.dumps(
        {
            "imgWidth": 8,
            "imgHeight": 4,
            "objects": [
                {"label": "road", "polygon": [[0, 0], [7, 0], [7, 3], [0, 3]]},
                {"label": "cargroup", "polygon": [[2, 1], [4, 1], [4, 3], [2, 3]]},
            ],
        }
    ).encode()
    for index, (archive_path, split, extension, encoder) in enumerate(
        (
            (paths[0], "train", "png", _png_bytes),
            (paths[1], "val", "jpg", _jpg_bytes),
        )
    ):
        root_name = "IDD_Segmentation" if index == 0 else "idd20kII"
        identifier = f"{index + 1:06d}"
        with tarfile.open(archive_path, "w:gz") as archive:
            _add_tar_file(
                archive,
                f"{root_name}/leftImg8bit/{split}/sequence-{index}/"
                f"{identifier}_leftImg8bit.{extension}",
                encoder(rgb),
            )
            _add_tar_file(
                archive,
                f"{root_name}/gtFine/{split}/sequence-{index}/{identifier}_gtFine_polygons.json",
                annotation,
            )
    return paths


def test_cityscapes_preparation_generates_separate_train_ids(tmp_path: Path) -> None:
    archives = _write_cityscapes_archives(tmp_path)
    destination = tmp_path / "prepared"
    result = prepare_dataset(
        "cityscapes",
        archives,
        destination,
        allow_fixture_count=True,
        verify_archive_hashes=False,
    )
    assert result["scientific_eligible"] is False
    mask_path = next(destination.glob("gtFine/train/trainIds/*/*.png"))
    assert np.unique(np.asarray(Image.open(mask_path))).tolist() == [0]
    assert next(destination.glob("gtFine/train/labelIds/*/*.png")).is_file()
    assert (
        prepare_dataset(
            "cityscapes",
            archives,
            destination,
            allow_fixture_count=True,
            verify_archive_hashes=False,
            verify_only=True,
        )
        == result
    )


def test_bdd_kaggle_preparation_is_normalized_but_ineligible(tmp_path: Path) -> None:
    archive = _write_bdd_kaggle(tmp_path)
    destination = tmp_path / "bdd"
    result = prepare_dataset(
        "bdd100k",
        (archive,),
        destination,
        source_profile="kaggle_mirror",
        allow_fixture_count=True,
    )
    assert result["source_profile"] == "kaggle_mirror"
    assert result["scientific_eligible"] is False
    assert (destination / "images/10k/train/sequence-train.jpg").is_file()
    assert (destination / "labels/sem_seg/masks/train/sequence-train.png").is_file()


def test_acdc_preparation_extracts_only_paired_adverse_validation(tmp_path: Path) -> None:
    archives = _write_acdc_archives(tmp_path)
    destination = tmp_path / "acdc"
    result = prepare_dataset(
        "acdc",
        archives,
        destination,
        allow_fixture_count=True,
    )
    assert result["counts"] == {"val": 1}
    assert result["scientific_eligible"] is False
    assert (destination / "rgb_anon/fog/val/GOPR/scene_rgb_anon.png").is_file()
    assert (destination / "gt/fog/val/GOPR/scene_gt_labelTrainIds.png").is_file()
    assert not (destination / "rgb_anon/fog/train").exists()
    assert not list(destination.rglob("*_gt_invIds.png"))


def test_idd_parts_render_source_ids_and_support_part_two_jpg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archives = _write_idd_archives(tmp_path)
    destination = tmp_path / "idd"
    archive_modes: list[str | None] = []
    original_tar_open = cast(Callable[..., tarfile.TarFile], tarfile.open)

    def tracked_tar_open(*args: object, **kwargs: object) -> tarfile.TarFile:
        mode = kwargs.get("mode")
        if mode is None and len(args) > 1:
            mode = args[1]
        archive_modes.append(str(mode) if mode is not None else None)
        return original_tar_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", tracked_tar_open)
    result = prepare_dataset(
        "idd20k",
        archives,
        destination,
        allow_fixture_count=True,
        verify_archive_hashes=False,
    )
    assert result["mapping_version"] == "autonue-polygon-source-id-and-cityscapes19-v2"
    assert len(result["ontology_sha256"]) == 64
    assert next(destination.glob("gtFine/train/**/*_gtFine_polygons.json")).is_file()
    mask_path = next(destination.glob("gtFine/train/**/*_gtFine_labelids.png"))
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened)
    assert {0, 12}.issubset(set(np.unique(mask)))
    canonical_path = next(destination.glob("gtFine/train/**/*_gtFine_labelTrainIds.png"))
    with Image.open(canonical_path) as opened:
        canonical = np.asarray(opened)
    assert {0, 13}.issubset(set(np.unique(canonical)))
    train = discover_domain_samples(destination, "idd20k", split="train")
    val = discover_domain_samples(destination, "idd20k", split="val")
    assert train[0].group_id == "idd20k:sequence-0"
    assert train[0].canonical_mask is not None
    assert val[0].image.endswith(".jpg")
    assert archive_modes == ["r|gz", "r|gz"]
    progress = capsys.readouterr().out
    assert '"phase":"archive_hash"' in progress
    assert '"phase":"idd_extract"' in progress
    assert '"phase":"idd_mask_render"' in progress


def test_archive_hashes_are_computed_without_second_archive_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archives = _write_idd_archives(tmp_path)
    hashed_paths: list[Path] = []
    original_sha256_file = preparation.sha256_file

    def tracked_sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        hashed_paths.append(path.resolve())
        return original_sha256_file(path, chunk_size=chunk_size)

    monkeypatch.setattr(preparation, "sha256_file", tracked_sha256_file)
    prepare_dataset(
        "idd20k",
        archives,
        tmp_path / "idd",
        allow_fixture_count=True,
        verify_archive_hashes=False,
    )
    assert not ({path.resolve() for path in archives} & set(hashed_paths))


def test_idd_shards_are_atomic_and_reused_after_local_runtime_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archives = _write_idd_archives(tmp_path)
    shard_root = tmp_path / "drive/prepared/v2/idd20k/shards"
    monkeypatch.setitem(preparation.EXPECTED_COUNTS, "idd20k", {"train": 1, "val": 1})
    first = prepare_dataset(
        "idd20k",
        archives,
        tmp_path / "content-first",
        verify_archive_hashes=False,
        idd_shard_root=shard_root,
        idd_shard_size=1,
    )
    assert Path(first["shard_index"]).is_file()
    shard_mtimes = {path.name: path.stat().st_mtime_ns for path in shard_root.glob("*.tar")}
    assert len(shard_mtimes) == 2
    assert not list(shard_root.glob("*.incoming"))

    def unexpected_render(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a completed IDD shard must not be rendered again")

    monkeypatch.setattr(preparation, "_render_idd_mask_pair", unexpected_render)
    second = prepare_dataset(
        "idd20k",
        archives,
        tmp_path / "content-after-reset",
        verify_archive_hashes=False,
        idd_shard_root=shard_root,
        idd_shard_size=1,
    )
    assert Path(second["shard_index"]).is_file()
    assert shard_mtimes == {path.name: path.stat().st_mtime_ns for path in shard_root.glob("*.tar")}


def test_idd_streaming_preparation_rejects_unsafe_member_and_cleans_staging(
    tmp_path: Path,
) -> None:
    archives = _write_idd_archives(tmp_path)
    with tarfile.open(archives[0], "w:gz") as archive:
        _add_tar_file(archive, "../escape", b"unsafe")
    destination = tmp_path / "idd"
    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_dataset(
            "idd20k",
            archives,
            destination,
            allow_fixture_count=True,
            verify_archive_hashes=False,
        )
    assert not destination.exists()
    assert not destination.with_name(".idd.incoming").exists()


def test_idd_renderer_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        render_idd_source_mask(
            {
                "imgWidth": 2,
                "imgHeight": 2,
                "objects": [{"label": "invented", "polygon": [[0, 0], [1, 0], [1, 1]]}],
            }
        )


def test_preparation_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    archive = tmp_path / "bdd100k.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../escape.jpg", b"unsafe")
    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_dataset(
            "bdd100k",
            (archive,),
            tmp_path / "prepared",
            source_profile="kaggle_mirror",
            allow_fixture_count=True,
        )
