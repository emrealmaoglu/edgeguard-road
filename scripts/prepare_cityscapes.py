"""Verify and prepare only the approved Cityscapes validation split."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from edgeguard.data.cityscapes import build_cityscapes_val_manifest
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

LEFT_ARCHIVE_NAME = "leftImg8bit_trainvaltest.zip"
LEFT_ARCHIVE_SHA256 = "3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525"
LABEL_ARCHIVE_NAME = "gtFine_trainvaltest.zip"
LABEL_ARCHIVE_SHA256 = "40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884"
MIN_FREE_MARGIN_BYTES = 5 * 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-images-archive", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _verify_archive(path: Path, expected_name: str, expected_sha256: str) -> None:
    if path.name != expected_name:
        raise ValueError(f"archive filename mismatch: expected {expected_name}, got {path.name}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive SHA-256 mismatch for {expected_name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _validated_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos: dict[str, ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or name.startswith(("/", "\\"))
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(mode)
        ):
            raise ValueError(f"unsafe ZIP member: {name!r}")
        normalized = path.as_posix()
        if normalized in infos:
            raise ValueError(f"duplicate ZIP member: {normalized}")
        infos[normalized] = info
    return infos


def _selected_infos(
    infos: dict[str, ZipInfo], *, prefix: str, document_group: str
) -> list[tuple[ZipInfo, Path]]:
    selected: list[tuple[ZipInfo, Path]] = []
    for name, info in sorted(infos.items()):
        if name.startswith(prefix):
            selected.append((info, Path(name)))
        elif name.lower() in {"readme", "license.txt"}:
            selected.append((info, Path("archive_docs") / document_group / name))
    return selected


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError(f"no existing parent for destination: {path}")
        candidate = candidate.parent
    return candidate


def _extract_selected(
    archive: ZipFile, selected: list[tuple[ZipInfo, Path]], staging: Path
) -> None:
    for info, relative_path in selected:
        target = staging / relative_path
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)


def _validated_manifest(root: Path) -> dict[str, Any]:
    manifest = build_cityscapes_val_manifest(root)
    if manifest["image_count"] != 500 or manifest["label_count"] != 500:
        raise ValueError(
            "Cityscapes val count mismatch: "
            f"images={manifest['image_count']}, labels={manifest['label_count']}"
        )
    if manifest["city_count"] != 3:
        raise ValueError(f"Cityscapes val city count mismatch: {manifest['city_count']}")
    return manifest


def _prepared_manifest(root: Path) -> dict[str, Any]:
    manifest = _validated_manifest(root)
    manifest["archives"] = [
        {"filename": LEFT_ARCHIVE_NAME, "sha256": LEFT_ARCHIVE_SHA256},
        {"filename": LABEL_ARCHIVE_NAME, "sha256": LABEL_ARCHIVE_SHA256},
    ]
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_payload(manifest_without_hash)
    return manifest


def prepare_cityscapes_val(
    left_archive_path: Path,
    label_archive_path: Path,
    destination: Path,
    *,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Prepare or verify one immutable, paired Cityscapes val directory."""
    _verify_archive(left_archive_path, LEFT_ARCHIVE_NAME, LEFT_ARCHIVE_SHA256)
    _verify_archive(label_archive_path, LABEL_ARCHIVE_NAME, LABEL_ARCHIVE_SHA256)
    if verify_only:
        if not destination.is_dir():
            raise ValueError("--verify-only destination does not exist")
        manifest = _prepared_manifest(destination)
        manifest_path = destination / "dataset_manifest.json"
        try:
            recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("recorded Cityscapes val manifest is missing or invalid") from error
        if recorded_manifest != manifest:
            raise ValueError("recorded Cityscapes val manifest does not match extracted data")
        return manifest
    if destination.exists():
        raise ValueError("destination already exists; overwrite is not permitted")

    staging = destination.with_name(f"{destination.name}.staging")
    if staging.exists():
        raise ValueError("staging destination already exists; inspect it before retrying")

    try:
        with (
            ZipFile(left_archive_path) as left_archive,
            ZipFile(label_archive_path) as label_archive,
        ):
            left_infos = _validated_infos(left_archive)
            label_infos = _validated_infos(label_archive)
            selected_left = _selected_infos(
                left_infos, prefix="leftImg8bit/val/", document_group="leftImg8bit"
            )
            selected_labels = _selected_infos(
                label_infos, prefix="gtFine/val/", document_group="gtFine"
            )
            selected_bytes = sum(
                info.file_size for info, _path in (*selected_left, *selected_labels)
            )
            free_bytes = shutil.disk_usage(_existing_parent(destination.parent)).free
            if free_bytes < selected_bytes + MIN_FREE_MARGIN_BYTES:
                raise ValueError(
                    "insufficient disk space for Cityscapes val extraction plus 5 GiB margin"
                )

            staging.mkdir(parents=True)
            _extract_selected(left_archive, selected_left, staging)
            _extract_selected(label_archive, selected_labels, staging)
    except BadZipFile as error:
        raise ValueError(f"invalid Cityscapes archive: {error}") from error

    manifest = _prepared_manifest(staging)
    (staging / "dataset_manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema_version": "1.0",
        "record_type": "cityscapes_val_extraction_receipt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_uncompressed_bytes": selected_bytes,
    }
    (staging / "extraction_receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return manifest


def main() -> int:
    """Prepare Cityscapes val and emit one path-free canonical result."""
    args = _parser().parse_args()
    try:
        manifest = prepare_cityscapes_val(
            args.left_images_archive,
            args.labels_archive,
            args.destination,
            verify_only=args.verify_only,
        )
    except (OSError, ValueError) as error:
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(
        canonical_json(
            {
                "status": "ok",
                "manifest_sha256": manifest["manifest_sha256"],
                "image_count": manifest["image_count"],
                "label_count": manifest["label_count"],
                "city_count": manifest["city_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
