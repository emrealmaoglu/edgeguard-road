"""Safe, dataset-specific preparation from immutable road-scene archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import IO, Any
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from edgeguard.data.cityscapes import label_ids_to_train_ids
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import ensure_disk_space

CITYSCAPES_ARCHIVES = {
    "leftImg8bit_trainvaltest.zip": (
        "3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525"
    ),
    "gtFine_trainvaltest.zip": "40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884",
}
IDD_ARCHIVES = {
    "idd-20k-I.tar.gz": "49c0085eadd86d53d5b7ae0bfa98e3584d49a725aa58a2895e7c594a07b40802",
    "idd-20k-II.tar.gz": "86e5b31ded2d096adfc40a2645abe640b7073b56131a7b95bbb52253d567516e",
}
BDD_OFFICIAL_MD5 = {
    "bdd100k_images_10k.zip": "08f26aecceda982568063d3d5873378e",
    "bdd100k_sem_seg_labels_trainval.zip": "9a2968dde3345eeb689cffb1e26f9c78",
}

EXPECTED_COUNTS = {
    "cityscapes": {"train": 2_975, "val": 500},
    "bdd100k": {"train": 7_000, "val": 1_000},
    "idd20k": {"train": 14_000, "val": 2_000},
}

# Exact source IDs from AutoNUE public-code commit
# 312a6ba2219a39fe8b105e44ad0612079249bad8/helpers/anue_labels.py.
IDD_LABEL_IDS = {
    "road": 0,
    "parking": 1,
    "drivable fallback": 2,
    "sidewalk": 3,
    "rail track": 4,
    "non-drivable fallback": 5,
    "person": 6,
    "animal": 7,
    "rider": 8,
    "motorcycle": 9,
    "bicycle": 10,
    "autorickshaw": 11,
    "car": 12,
    "truck": 13,
    "bus": 14,
    "caravan": 15,
    "trailer": 16,
    "train": 17,
    "vehicle fallback": 18,
    "curb": 19,
    "wall": 20,
    "fence": 21,
    "guard rail": 22,
    "billboard": 23,
    "traffic sign": 24,
    "traffic light": 25,
    "pole": 26,
    "polegroup": 27,
    "obs-str-bar-fallback": 28,
    "building": 29,
    "bridge": 30,
    "tunnel": 31,
    "vegetation": 32,
    "sky": 33,
    "fallback background": 34,
    "unlabeled": 35,
    "ego vehicle": 36,
    "rectification border": 37,
    "out of roi": 38,
    "license plate": 39,
}


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith(("/", "\\"))
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def _validated_zip_members(archive: ZipFile) -> dict[str, ZipInfo]:
    members: dict[str, ZipInfo] = {}
    for info in archive.infolist():
        path = _safe_member_name(info.filename)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"unsafe archive member: {info.filename!r}")
        name = path.as_posix()
        if name in members:
            raise ValueError(f"duplicate archive member: {name}")
        members[name] = info
    return members


def _copy_stream(source: IO[bytes], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024**2)


def _archive_rows(archives: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in archives:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"archive must be a regular file: {path.name}")
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        byte_size = path.stat().st_size
        completed = 0
        _emit_progress("archive_hash", completed=0, total=byte_size, item=path.name)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                md5.update(chunk)
                sha256.update(chunk)
                completed += len(chunk)
                if completed == byte_size or completed % (1024**3) < len(chunk):
                    _emit_progress(
                        "archive_hash", completed=completed, total=byte_size, item=path.name
                    )
        rows.append(
            {
                "filename": path.name,
                "byte_size": byte_size,
                "sha256": sha256.hexdigest(),
                "md5": md5.hexdigest(),
            }
        )
    return rows


def _emit_progress(
    phase: str, *, completed: int, total: int | None, item: str | None = None
) -> None:
    """Emit one flushed, machine-readable progress event for long Colab operations."""
    payload: dict[str, Any] = {
        "record_type": "edgeguard_dataset_preparation_progress",
        "phase": phase,
        "completed": completed,
        "total": total,
    }
    if total is not None and total > 0:
        payload["percent"] = round(100.0 * completed / total, 2)
    if item is not None:
        payload["item"] = item
    print(canonical_json(payload), flush=True)


def _require_archive_set(
    rows: Sequence[dict[str, Any]], expected: dict[str, str], *, verify_hashes: bool
) -> None:
    actual = {str(row["filename"]): str(row["sha256"]) for row in rows}
    if set(actual) != set(expected):
        raise ValueError(f"archive filenames must be exactly {sorted(expected)}")
    if verify_hashes:
        mismatches = [name for name, digest in expected.items() if actual[name] != digest]
        if mismatches:
            raise ValueError(f"archive SHA-256 mismatch: {mismatches}")


def _write_receipt(
    root: Path,
    *,
    dataset_id: str,
    source_profile: str,
    archives: Sequence[dict[str, Any]],
    counts: dict[str, int],
    scientific_eligible: bool,
    mapping_version: str,
    ontology_sha256: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_dataset_preparation_receipt",
        "dataset_id": dataset_id,
        "source_profile": source_profile,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_archives": list(archives),
        "counts": counts,
        "mapping_version": mapping_version,
        "ontology_sha256": ontology_sha256,
        "native_labels_preserved": True,
        "scientific_eligible": scientific_eligible,
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    (root / "preparation_receipt.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def _atomic_destination(destination: Path) -> Path:
    if destination.exists():
        raise ValueError("destination already exists; use --verify-only for prepared data")
    incoming = destination.with_name(f".{destination.name}.incoming")
    if incoming.exists():
        raise ValueError(f"stale incoming preparation must be inspected: {incoming}")
    incoming.mkdir(parents=True)
    return incoming


def _validate_counts(dataset_id: str, counts: dict[str, int], *, allow_fixture_count: bool) -> bool:
    expected = EXPECTED_COUNTS[dataset_id]
    valid = counts == expected
    if not valid and not allow_fixture_count:
        raise ValueError(f"{dataset_id} pair counts must be {expected}, found {counts}")
    return valid


def _image_mask_counts(root: Path, dataset_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        if dataset_id == "cityscapes":
            images = list((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
            masks = list((root / "gtFine" / split / "trainIds").glob("*/*_gtFine_trainIds.png"))
        elif dataset_id == "bdd100k":
            images = [
                *list((root / "images" / "10k" / split).glob("*.jpg")),
                *list((root / "images" / "10k" / split).glob("*.png")),
            ]
            masks = list((root / "labels" / "sem_seg" / "masks" / split).glob("*.png"))
        else:
            images = [
                *list((root / "leftImg8bit" / split).glob("**/*_leftImg8bit.png")),
                *list((root / "leftImg8bit" / split).glob("**/*_leftImg8bit.jpg")),
                *list((root / "leftImg8bit" / split).glob("**/*_leftImg8bit.jpeg")),
            ]
            masks = list((root / "gtFine" / split).glob("**/*_gtFine_labelids.png"))
        if len(images) != len(masks):
            raise ValueError(
                f"{dataset_id} {split} pairing mismatch: images={len(images)}, masks={len(masks)}"
            )
        counts[split] = len(images)
    return counts


def _extract_cityscapes(archives: Sequence[Path], staging: Path) -> None:
    by_name = {path.name: path for path in archives}
    for archive_name, kind in (
        ("leftImg8bit_trainvaltest.zip", "images"),
        ("gtFine_trainvaltest.zip", "labels"),
    ):
        with ZipFile(by_name[archive_name]) as archive:
            members = _validated_zip_members(archive)
            for name, info in sorted(members.items()):
                path = PurePosixPath(name)
                if info.is_dir() or len(path.parts) != 4:
                    continue
                if kind == "images":
                    selected = (
                        path.parts[0] == "leftImg8bit"
                        and path.parts[1] in {"train", "val"}
                        and path.name.endswith("_leftImg8bit.png")
                    )
                    target = staging / Path(*path.parts)
                else:
                    selected = (
                        path.parts[0] == "gtFine"
                        and path.parts[1] in {"train", "val"}
                        and path.name.endswith("_gtFine_labelIds.png")
                    )
                    target = (
                        staging / "gtFine" / path.parts[1] / "labelIds" / path.parts[2] / path.name
                    )
                if not selected:
                    continue
                with archive.open(info) as source:
                    _copy_stream(source, target)
    for split in ("train", "val"):
        for label_path in sorted((staging / "gtFine" / split / "labelIds").glob("*/*.png")):
            with Image.open(label_path) as opened:
                label_ids = np.asarray(opened, dtype=np.uint8)
            train_ids = label_ids_to_train_ids(label_ids)
            target = (
                staging
                / "gtFine"
                / split
                / "trainIds"
                / label_path.parent.name
                / label_path.name.replace("_labelIds.png", "_trainIds.png")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(train_ids, mode="L").save(target)


def _bdd_target(name: str, source_profile: str) -> tuple[Path, str] | None:
    path = PurePosixPath(name)
    parts = path.parts
    if source_profile == "official":
        for marker in (("images", "10k"), ("labels", "sem_seg", "masks")):
            for index in range(len(parts) - len(marker) + 1):
                if parts[index : index + len(marker)] == marker:
                    relative = parts[index:]
                    if len(relative) == len(marker) + 2 and relative[-2] in {"train", "val"}:
                        kind = "image" if marker[0] == "images" else "mask"
                        return Path(*relative), kind
        return None
    image_marker = ("bdd100k", "seg", "images")
    label_marker = ("bdd100k", "seg", "labels")
    for marker, kind in ((image_marker, "image"), (label_marker, "mask")):
        for index in range(len(parts) - len(marker) + 1):
            if parts[index : index + len(marker)] != marker:
                continue
            relative = parts[index + len(marker) :]
            if len(relative) != 2 or relative[0] not in {"train", "val"}:
                return None
            filename = relative[1]
            if kind == "mask":
                filename = filename.removesuffix("_train_id.png") + ".png"
                return Path("labels", "sem_seg", "masks", relative[0], filename), kind
            return Path("images", "10k", relative[0], filename), kind
    return None


def _extract_bdd(archives: Sequence[Path], staging: Path, source_profile: str) -> None:
    targets: set[str] = set()
    for archive_path in archives:
        try:
            with ZipFile(archive_path) as archive:
                members = _validated_zip_members(archive)
                for name, info in sorted(members.items()):
                    if info.is_dir():
                        continue
                    selected = _bdd_target(name, source_profile)
                    if selected is None:
                        continue
                    target, kind = selected
                    suffix = target.suffix.lower()
                    if (kind == "image" and suffix not in {".jpg", ".jpeg", ".png"}) or (
                        kind == "mask" and suffix != ".png"
                    ):
                        continue
                    key = target.as_posix()
                    if key in targets:
                        raise ValueError(f"BDD preparation target collision: {key}")
                    targets.add(key)
                    with archive.open(info) as source:
                        _copy_stream(source, staging / target)
        except BadZipFile as error:
            raise ValueError(f"invalid BDD archive: {archive_path.name}") from error


def _idd_relative(name: str) -> tuple[Path, str] | None:
    parts = PurePosixPath(name).parts
    for marker, kind in (("leftImg8bit", "image"), ("gtFine", "polygon")):
        if marker not in parts:
            continue
        index = parts.index(marker)
        relative = parts[index:]
        if len(relative) != 4 or relative[1] not in {"train", "val"}:
            return None
        if kind == "image" and relative[-1].lower().endswith(
            ("_leftimg8bit.png", "_leftimg8bit.jpg", "_leftimg8bit.jpeg")
        ):
            return Path(*relative), kind
        if kind == "polygon" and relative[-1].endswith("_gtFine_polygons.json"):
            return Path(*relative), kind
    return None


def _normalize_idd_label(name: str) -> str:
    if name in IDD_LABEL_IDS:
        return name
    if name.endswith("group") and name[: -len("group")] in IDD_LABEL_IDS:
        return name[: -len("group")]
    raise ValueError(f"IDD polygon contains unknown label: {name!r}")


def render_idd_source_mask(annotation: dict[str, Any]) -> Image.Image:
    """Render one IDD polygon JSON with the pinned regular source-ID encoding."""
    try:
        width = int(annotation["imgWidth"])
        height = int(annotation["imgHeight"])
        objects = annotation["objects"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("IDD polygon annotation header is invalid") from error
    if width <= 0 or height <= 0 or not isinstance(objects, list):
        raise ValueError("IDD polygon annotation dimensions or objects are invalid")
    output = Image.new("L", (width, height), color=IDD_LABEL_IDS["unlabeled"])
    drawer = ImageDraw.Draw(output)
    for record in objects:
        if not isinstance(record, dict):
            raise ValueError("IDD polygon object must be a mapping")
        if bool(record.get("deleted", False)):
            continue
        label = _normalize_idd_label(str(record.get("label", "")))
        polygon = record.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points: list[tuple[int, int]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("IDD polygon point is invalid")
            points.append((int(point[0]), int(point[1])))
        drawer.polygon(points, fill=IDD_LABEL_IDS[label])
    return output


def _idd_lut(ontology: dict[str, Any]) -> tuple[int, ...]:
    """Build the reviewed uint8 source-ID to Cityscapes19 lookup table."""
    source = ontology["sources"]["idd20k"]
    mapping = {int(key): int(value) for key, value in source["map"].items()}
    ignored = {int(value) for value in source["ignore_source_ids"]}
    reviewed = set(mapping) | ignored
    if reviewed != set(range(40)) | {255} or set(mapping) & ignored:
        raise ValueError("IDD ontology must review every source ID exactly once")
    lut = np.full(256, 255, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        lut[source_id] = target_id
    return tuple(int(value) for value in lut)


def _render_idd_mask_pair(polygon_path: Path, lut: tuple[int, ...]) -> None:
    """Render native and canonical masks for one validated IDD polygon file."""
    try:
        payload = json.loads(polygon_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid IDD polygon JSON: {polygon_path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid IDD polygon root: {polygon_path.name}")
    mask = render_idd_source_mask(payload)
    target = polygon_path.with_name(
        polygon_path.name.replace("_gtFine_polygons.json", "_gtFine_labelids.png")
    )
    mask.save(target, compress_level=1)
    canonical_target = polygon_path.with_name(
        polygon_path.name.replace("_gtFine_polygons.json", "_gtFine_labelTrainIds.png")
    )
    mask.point(lut).save(canonical_target, compress_level=1)


def _extract_idd(archives: Sequence[Path], staging: Path, ontology_path: Path) -> None:
    from edgeguard.rescue.multidomain import load_semantic_ontology

    ontology = load_semantic_ontology(ontology_path)
    lut = _idd_lut(ontology)
    targets: set[str] = set()
    polygons: list[Path] = []
    extracted = 0
    for archive_path in archives:
        member_names: set[str] = set()
        _emit_progress("idd_extract", completed=extracted, total=None, item=archive_path.name)
        # Streaming mode preserves physical member order. In contrast, seeking through a
        # gzip tar after sorting names can repeatedly decompress the archive from the start.
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for info in archive:
                member_path = _safe_member_name(info.name)
                if not (info.isfile() or info.isdir()) or info.issym() or info.islnk():
                    raise ValueError(f"unsafe archive member type: {info.name!r}")
                name = member_path.as_posix()
                if name in member_names:
                    raise ValueError(f"duplicate archive member: {name}")
                member_names.add(name)
                if not info.isfile():
                    continue
                selected = _idd_relative(name)
                if selected is None:
                    continue
                target, kind = selected
                key = target.as_posix()
                if key in targets:
                    raise ValueError(f"IDD Part I/II normalized-path collision: {key}")
                targets.add(key)
                stream = archive.extractfile(info)
                if stream is None:
                    raise ValueError(f"IDD archive member has no content: {name}")
                with stream:
                    _copy_stream(stream, staging / target)
                extracted += 1
                if kind == "polygon":
                    polygons.append(staging / target)
                if extracted % 250 == 0:
                    _emit_progress(
                        "idd_extract", completed=extracted, total=None, item=archive_path.name
                    )
        _emit_progress("idd_extract", completed=extracted, total=None, item=archive_path.name)

    total_polygons = len(polygons)
    workers = min(4, max(1, os.cpu_count() or 1))
    _emit_progress("idd_mask_render", completed=0, total=total_polygons)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="idd-mask") as executor:
        polygon_iterator = iter(polygons)
        pending = {
            executor.submit(_render_idd_mask_pair, polygon_path, lut)
            for polygon_path in islice(polygon_iterator, workers)
        }
        completed = 0
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                completed += 1
                next_polygon = next(polygon_iterator, None)
                if next_polygon is not None:
                    pending.add(executor.submit(_render_idd_mask_pair, next_polygon, lut))
                if completed == total_polygons or completed % 100 == 0:
                    _emit_progress("idd_mask_render", completed=completed, total=total_polygons)


def prepare_dataset(
    dataset_id: str,
    archives: Sequence[Path],
    destination: Path,
    *,
    source_profile: str = "official",
    allow_fixture_count: bool = False,
    verify_only: bool = False,
    verify_archive_hashes: bool = True,
    ontology_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare or verify one approved semantic dataset without rewriting native labels."""
    if dataset_id not in EXPECTED_COUNTS:
        raise ValueError(f"unsupported preparation dataset: {dataset_id}")
    if source_profile not in {"official", "kaggle_mirror"}:
        raise ValueError("source profile must be official or kaggle_mirror")
    if dataset_id != "bdd100k" and source_profile != "official":
        raise ValueError("only BDD100K may use the quarantined kaggle_mirror profile")
    archive_paths = tuple(path.resolve() for path in archives)
    resolved_ontology: Path | None = None
    if dataset_id == "idd20k":
        resolved_ontology = (
            ontology_path.resolve()
            if ontology_path is not None
            else Path(__file__).parents[3] / "configs/dataset/semantic_ontology_v2.yaml"
        )
        if not resolved_ontology.is_file():
            raise ValueError(f"IDD ontology contract is missing: {resolved_ontology}")
    if not archive_paths:
        raise ValueError("at least one archive is required")
    rows = _archive_rows(archive_paths)
    if dataset_id == "cityscapes":
        _require_archive_set(rows, CITYSCAPES_ARCHIVES, verify_hashes=verify_archive_hashes)
    elif dataset_id == "idd20k":
        _require_archive_set(rows, IDD_ARCHIVES, verify_hashes=verify_archive_hashes)
    elif source_profile == "official":
        if {path.name for path in archive_paths} != set(BDD_OFFICIAL_MD5):
            raise ValueError(f"official BDD archives must be exactly {sorted(BDD_OFFICIAL_MD5)}")
        if verify_archive_hashes:
            actual_md5 = {str(row["filename"]): str(row["md5"]) for row in rows}
            mismatches = [
                name for name, digest in BDD_OFFICIAL_MD5.items() if actual_md5[name] != digest
            ]
            if mismatches:
                raise ValueError(f"official BDD archive MD5 mismatch: {mismatches}")
    elif len(archive_paths) != 1:
        raise ValueError("BDD kaggle_mirror preparation accepts exactly one archive")

    if verify_only:
        receipt_path = destination / "preparation_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("prepared dataset receipt is missing or invalid") from error
        if receipt.get("source_archives") != rows or receipt.get("dataset_id") != dataset_id:
            raise ValueError("prepared dataset receipt does not match current inputs")
        if resolved_ontology is not None and receipt.get("ontology_sha256") != sha256_file(
            resolved_ontology
        ):
            raise ValueError("prepared IDD ontology hash does not match current contract")
        expected_hash = receipt.get("receipt_sha256")
        unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if expected_hash != sha256_payload(unhashed):
            raise ValueError("prepared dataset receipt hash mismatch")
        counts = _image_mask_counts(destination, dataset_id)
        if counts != receipt.get("counts"):
            raise ValueError("prepared dataset file counts changed")
        return receipt

    staging = _atomic_destination(destination)
    try:
        estimated = sum(row["byte_size"] for row in rows)
        ensure_disk_space(staging, estimated, reserve_bytes=5 * 1024**3)
        if dataset_id == "cityscapes":
            _extract_cityscapes(archive_paths, staging)
            mapping_version = "cityscapes-labelids-to-trainids-v1"
        elif dataset_id == "bdd100k":
            _extract_bdd(archive_paths, staging, source_profile)
            mapping_version = "bdd100k-cityscapes19-v1"
        else:
            assert resolved_ontology is not None
            _extract_idd(archive_paths, staging, resolved_ontology)
            mapping_version = "autonue-polygon-source-id-and-cityscapes19-v2"
        counts = _image_mask_counts(staging, dataset_id)
        counts_valid = _validate_counts(dataset_id, counts, allow_fixture_count=allow_fixture_count)
        receipt = _write_receipt(
            staging,
            dataset_id=dataset_id,
            source_profile=source_profile,
            archives=rows,
            counts=counts,
            scientific_eligible=counts_valid and source_profile == "official",
            mapping_version=mapping_version,
            ontology_sha256=(sha256_file(resolved_ontology) if resolved_ontology else None),
        )
        os.replace(staging, destination)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def preparation_storage_bytes(archives: Iterable[Path]) -> int:
    """Return a conservative raw-archive byte total for preflight display."""
    return sum(path.stat().st_size for path in archives if path.is_file())
