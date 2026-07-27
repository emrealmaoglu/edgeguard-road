"""Verify and prepare approved Cityscapes validation or Fine train data."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np
import PIL

from edgeguard.data.cityscapes import build_cityscapes_val_manifest
from edgeguard.data.cityscapes_train import (
    analyze_prepared_train,
    build_split_candidates,
    build_split_comparison,
    generate_train_id_masks,
    render_split_report,
)
from edgeguard.data.ontology import load_project_ontology
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

LEFT_ARCHIVE_NAME = "leftImg8bit_trainvaltest.zip"
LEFT_ARCHIVE_SHA256 = "3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525"
LABEL_ARCHIVE_NAME = "gtFine_trainvaltest.zip"
LABEL_ARCHIVE_SHA256 = "40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884"
MIN_FREE_MARGIN_BYTES = 5 * 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--left-images-archive", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifests-destination", type=Path)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--preparation-git-commit")
    parser.add_argument(
        "--ontology-config",
        type=Path,
        default=Path("configs/dataset/ontology_v1.yaml"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _verify_archive(path: Path, expected_name: str, expected_sha256: str) -> dict[str, Any]:
    if path.name != expected_name:
        raise ValueError(f"archive filename mismatch: expected {expected_name}, got {path.name}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"archive must be a regular non-symlink file: {expected_name}")
    byte_size = path.stat().st_size
    if byte_size <= 0:
        raise ValueError(f"archive is empty: {expected_name}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive SHA-256 mismatch for {expected_name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "filename": expected_name,
        "byte_size": byte_size,
        "sha256": actual_sha256,
        "source_reference": f"private_inputs/{expected_name}",
    }


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


def _selected_train_images(infos: dict[str, ZipInfo]) -> list[tuple[ZipInfo, Path]]:
    selected: list[tuple[ZipInfo, Path]] = []
    for name, info in sorted(infos.items()):
        path = PurePosixPath(name)
        if info.is_dir() or len(path.parts) < 2:
            continue
        if path.parts[:2] != ("leftImg8bit", "train"):
            continue
        if len(path.parts) != 4 or not name.endswith("_leftImg8bit.png"):
            raise ValueError(f"unexpected Cityscapes train image member: {name}")
        city = path.parts[2]
        if not path.name.startswith(f"{city}_"):
            raise ValueError(f"Cityscapes train image city mismatch: {name}")
        selected.append((info, Path(*path.parts)))
    if not selected:
        raise ValueError("Cityscapes image archive contains no Fine train images")
    return selected


def _selected_train_labels(infos: dict[str, ZipInfo]) -> list[tuple[ZipInfo, Path]]:
    selected: list[tuple[ZipInfo, Path]] = []
    for name, info in sorted(infos.items()):
        path = PurePosixPath(name)
        if info.is_dir() or len(path.parts) < 2:
            continue
        if path.parts[:2] != ("gtFine", "train"):
            continue
        if not name.endswith("_gtFine_labelIds.png"):
            continue
        if len(path.parts) != 4:
            raise ValueError(f"unexpected Cityscapes train label member: {name}")
        city = path.parts[2]
        if not path.name.startswith(f"{city}_"):
            raise ValueError(f"Cityscapes train label city mismatch: {name}")
        target = Path("gtFine", "train", "labelIds", city, path.name)
        selected.append((info, target))
    if not selected:
        raise ValueError("Cityscapes label archive contains no Fine train label-ID masks")
    return selected


def _validate_git_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("preparation Git commit must be a full lowercase 40-character SHA")
    return value


def _verify_execution_git_state(repository_root: Path, expected_commit: str) -> None:
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not verify preparation repository state") from error
    if actual_commit != expected_commit:
        raise ValueError("preparation Git commit does not match the current checkout")
    if status:
        raise ValueError("preparation requires a clean Git checkout")


def _write_json(path: Path, payload: Any) -> None:
    _assert_root_free(payload)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _assert_root_free(value: Any) -> None:
    if isinstance(value, str):
        if (
            value.startswith(("/", "~/"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or "/content/drive/MyDrive/" in value
        ):
            raise ValueError("serialized preparation records must be root-free")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_root_free(key)
            _assert_root_free(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_root_free(item)


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _manifest_payload(
    analysis: dict[str, Any],
    *,
    archive_identities: list[dict[str, Any]],
    preparation_git_commit: str,
    ontology_version: str,
    ontology_sha256: str,
) -> dict[str, Any]:
    samples = [
        {
            "sample_id": sample["sample_id"],
            "city": sample["city"],
            "sequence": sample["sequence"],
            "frame": sample["frame"],
            "group_id": sample["group_id"],
            "image_relative_path": sample["image_relative_path"],
            "label_id_relative_path": sample["label_id_relative_path"],
            "train_id_relative_path": sample["train_id_relative_path"],
            "train_id_bytes": sample["train_id_bytes"],
            "train_id_sha256": sample["train_id_sha256"],
            "pairing_sha256": sample["pairing_sha256"],
        }
        for sample in analysis["samples"]
    ]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_manifest",
        "dataset": "Cityscapes",
        "dataset_version": "Fine-v1",
        "dataset_roles": ["train_fit", "train_select", "train_calibration"],
        "official_val_role": "official_val_common_eval",
        "split": "train",
        "source_archives": archive_identities,
        "preparation_git_commit": preparation_git_commit,
        "ontology_version": ontology_version,
        "ontology_sha256": ontology_sha256,
        "image_count": len(samples),
        "label_id_count": len(samples),
        "generated_train_id_count": len(samples),
        "city_count": len(analysis["cities"]),
        "sequence_group_count": len(analysis["groups"]),
        "train_id_file_map_sha256": analysis["train_id_integrity"]["file_map_sha256"],
        "samples": samples,
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


def _compact_dataset_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_summary",
        "sample_count": analysis["global"]["sample_count"],
        "group_count": len(analysis["groups"]),
        "city_count": len(analysis["cities"]),
        "valid_pixel_count": analysis["global"]["valid_pixel_count"],
        "ignored_pixel_count": analysis["global"]["ignored_pixel_count"],
        "ignored_pixel_ratio": analysis["global"]["ignored_pixel_ratio"],
        "class_pixel_counts": analysis["global"]["class_pixel_counts"],
        "class_pixel_shares": analysis["global"]["class_pixel_shares"],
        "class_presence_counts": analysis["global"]["class_presence_counts"],
        "class_image_presence_shares": analysis["global"]["class_image_presence_shares"],
        "rare_class_definition_candidates": analysis["rare_class_definition_candidates"],
        "heuristic_rare_class_ids": analysis["heuristic_rare_class_ids"],
        "train_id_integrity": analysis["train_id_integrity"],
    }


def _group_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    groups = []
    for group_id, group in analysis["groups"].items():
        groups.append(
            {
                "group_id": group_id,
                "city": group["city"],
                "sequence": group["sequence"],
                "sample_count": group["sample_count"],
                "valid_pixel_count": group["valid_pixel_count"],
                "ignored_pixel_count": group["ignored_pixel_count"],
                "class_pixel_counts": group["class_pixel_counts"],
                "class_presence_counts": group["class_presence_counts"],
            }
        )
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_group_summary",
        "group_identity": "city+sequence",
        "group_count": len(groups),
        "groups": groups,
    }


def _aggregate_without_sample_ids(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in aggregate.items() if key != "sample_ids"}


def _write_candidate_manifests(manifests_root: Path, candidate_set: dict[str, Any]) -> list[str]:
    candidate_dir = manifests_root / "split_candidates"
    candidate_dir.mkdir(parents=True)
    relative_paths: list[str] = []
    for candidate in candidate_set["candidates"]:
        candidate_id = candidate["candidate_id"]
        sample_payload = {
            "schema_version": "1.0",
            "record_type": "cityscapes_fine_split_sample_manifest",
            "candidate_id": candidate_id,
            "candidate_sha256": candidate["candidate_sha256"],
            "status": candidate["status"],
            "samples": candidate["sample_manifest"],
        }
        group_payload = {
            "schema_version": "1.0",
            "record_type": "cityscapes_fine_split_group_manifest",
            "candidate_id": candidate_id,
            "candidate_sha256": candidate["candidate_sha256"],
            "status": candidate["status"],
            "groups": candidate["group_manifest"],
        }
        sample_path = candidate_dir / f"{candidate_id}.samples.json"
        group_path = candidate_dir / f"{candidate_id}.groups.json"
        _write_json(sample_path, sample_payload)
        _write_json(group_path, group_payload)
        relative_paths.extend(
            [
                sample_path.relative_to(manifests_root).as_posix(),
                group_path.relative_to(manifests_root).as_posix(),
            ]
        )
    return relative_paths


def _environment_summary() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_preparation_environment",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
    }


def _write_evidence_package(manifests_root: Path, relative_names: list[str]) -> dict[str, Any]:
    file_rows = [
        {
            "relative_path": name,
            "byte_size": (manifests_root / name).stat().st_size,
            "sha256": sha256_file(manifests_root / name),
        }
        for name in sorted(relative_names)
    ]
    evidence_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_evidence_manifest",
        "files": file_rows,
    }
    evidence_manifest["manifest_sha256"] = sha256_payload(evidence_manifest)
    evidence_manifest_path = manifests_root / "evidence_manifest.json"
    _write_json(evidence_manifest_path, evidence_manifest)

    package_path = manifests_root / "edgeguard-cityscapes-fine-train-evidence.zip"
    with ZipFile(package_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted([*relative_names, "evidence_manifest.json"]):
            data = (manifests_root / name).read_bytes()
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    receipt = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_evidence_package_receipt",
        "filename": package_path.name,
        "byte_size": package_path.stat().st_size,
        "sha256": sha256_file(package_path),
        "evidence_manifest_sha256": evidence_manifest["manifest_sha256"],
    }
    _write_json(manifests_root / "evidence_package_receipt.json", receipt)
    return receipt


def _verify_manifest_hash(payload: dict[str, Any], *, field: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"recorded {field} is missing or invalid")
    unhashed = {key: value for key, value in payload.items() if key != field}
    if sha256_payload(unhashed) != expected:
        raise ValueError(f"recorded {field} does not match canonical payload")


def _verify_prepared_train(
    destination: Path,
    manifests_destination: Path,
    *,
    archive_identities: list[dict[str, Any]],
    ontology_version: str,
    ontology_sha256: str,
) -> dict[str, Any]:
    if not destination.is_dir() or not manifests_destination.is_dir():
        raise ValueError("prepared dataset and manifest destinations must both exist")
    manifest = _load_json(
        destination / "dataset_manifest.json", description="prepared dataset manifest"
    )
    external_manifest = _load_json(
        manifests_destination / "dataset_manifest.json",
        description="external dataset manifest",
    )
    if manifest != external_manifest:
        raise ValueError("internal and external dataset manifests do not match")
    _verify_manifest_hash(manifest, field="manifest_sha256")
    if manifest.get("source_archives") != archive_identities:
        raise ValueError("prepared dataset archive identities do not match current inputs")
    if manifest.get("ontology_version") != ontology_version:
        raise ValueError("prepared dataset ontology version mismatch")
    if manifest.get("ontology_sha256") != ontology_sha256:
        raise ValueError("prepared dataset ontology SHA-256 mismatch")

    analysis = analyze_prepared_train(destination)
    recorded_samples = manifest.get("samples")
    if not isinstance(recorded_samples, list):
        raise ValueError("prepared dataset manifest samples are invalid")
    actual_samples = {sample["sample_id"]: sample for sample in analysis["samples"]}
    if len(recorded_samples) != len(actual_samples):
        raise ValueError("prepared dataset manifest sample count mismatch")
    for recorded in recorded_samples:
        actual = actual_samples.get(recorded.get("sample_id"))
        if actual is None:
            raise ValueError("prepared dataset manifest contains an unknown sample")
        for key in (
            "image_relative_path",
            "label_id_relative_path",
            "train_id_relative_path",
            "train_id_bytes",
            "train_id_sha256",
            "pairing_sha256",
        ):
            if recorded.get(key) != actual.get(key):
                raise ValueError(
                    f"prepared dataset sample field mismatch for {recorded.get('sample_id')}: {key}"
                )
    expected_counts = {
        "image_count": len(actual_samples),
        "label_id_count": len(actual_samples),
        "generated_train_id_count": len(actual_samples),
        "city_count": len(analysis["cities"]),
        "sequence_group_count": len(analysis["groups"]),
    }
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise ValueError(f"prepared dataset {key} mismatch")
    comparison = _load_json(
        manifests_destination / "split_candidate_comparison.json",
        description="split candidate comparison",
    )
    _verify_manifest_hash(comparison, field="comparison_sha256")
    if comparison.get("selection_status") != "recommended_pending_human_approval":
        raise ValueError("prepared split candidates must remain pending human approval")
    evidence_receipt = _load_json(
        manifests_destination / "evidence_package_receipt.json",
        description="evidence package receipt",
    )
    package_path = manifests_destination / evidence_receipt.get("filename", "")
    if sha256_file(package_path) != evidence_receipt.get("sha256"):
        raise ValueError("evidence package SHA-256 mismatch")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "image_count": len(actual_samples),
        "label_id_count": len(actual_samples),
        "generated_train_id_count": len(actual_samples),
        "city_count": len(analysis["cities"]),
        "sequence_group_count": len(analysis["groups"]),
        "recommended_candidate_id": comparison["recommended_candidate_id"],
        "selection_status": comparison["selection_status"],
        "evidence_package_filename": evidence_receipt["filename"],
        "evidence_package_sha256": evidence_receipt["sha256"],
    }


def prepare_cityscapes_train(
    left_archive_path: Path,
    label_archive_path: Path,
    destination: Path,
    manifests_destination: Path,
    work_directory: Path,
    *,
    preparation_git_commit: str,
    ontology_config: Path,
    verify_only: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prepare or idempotently verify Cityscapes Fine train and split candidates."""
    started = time.perf_counter()
    commit = _validate_git_commit(preparation_git_commit)
    archive_identities = [
        _verify_archive(left_archive_path, LEFT_ARCHIVE_NAME, LEFT_ARCHIVE_SHA256),
        _verify_archive(label_archive_path, LABEL_ARCHIVE_NAME, LABEL_ARCHIVE_SHA256),
    ]
    ontology = load_project_ontology(ontology_config)
    ontology_sha256 = sha256_payload(ontology.model_dump(mode="json"))
    if verify_only:
        return _verify_prepared_train(
            destination,
            manifests_destination,
            archive_identities=archive_identities,
            ontology_version=ontology.ontology_version,
            ontology_sha256=ontology_sha256,
        )
    if destination.exists():
        raise ValueError(
            "dataset destination already exists; use --verify-only for exact validation"
        )
    if manifests_destination.exists():
        raise ValueError("manifest destination already exists; overwrite is not permitted")

    staging = work_directory / "cityscapes-fine-train.staging"
    manifest_staging = work_directory / "cityscapes-fine-manifests.staging"
    destination_incoming = destination.with_name(f".{destination.name}.incoming")
    manifests_incoming = manifests_destination.with_name(f".{manifests_destination.name}.incoming")
    for path in (staging, manifest_staging, destination_incoming, manifests_incoming):
        if path.exists():
            raise ValueError(f"preparation collision; inspect existing path named {path.name!r}")

    with (
        ZipFile(left_archive_path) as left_archive,
        ZipFile(label_archive_path) as label_archive,
    ):
        left_infos = _validated_infos(left_archive)
        label_infos = _validated_infos(label_archive)
        selected_left = _selected_train_images(left_infos)
        selected_labels = _selected_train_labels(label_infos)
        if len(selected_left) != len(selected_labels):
            raise ValueError(
                "Cityscapes Fine train archive count mismatch: "
                f"images={len(selected_left)}, labels={len(selected_labels)}"
            )
        selected_bytes = sum(
            info.file_size for info, _relative in (*selected_left, *selected_labels)
        )
        free_bytes = shutil.disk_usage(_existing_parent(work_directory)).free
        if free_bytes < selected_bytes + MIN_FREE_MARGIN_BYTES:
            raise ValueError(
                "insufficient ephemeral disk for Cityscapes Fine train preparation "
                "plus 5 GiB margin"
            )
        staging.mkdir(parents=True)
        manifest_staging.mkdir(parents=True)
        _extract_selected(left_archive, selected_left, staging)
        _extract_selected(label_archive, selected_labels, staging)

    samples = generate_train_id_masks(staging)
    if len(samples) != len(selected_left):
        raise ValueError(
            "Cityscapes Fine train extracted pairing mismatch: "
            f"archive={len(selected_left)}, prepared={len(samples)}"
        )
    analysis = analyze_prepared_train(staging)
    candidate_set = build_split_candidates(analysis)
    comparison = build_split_comparison(candidate_set)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    manifest = _manifest_payload(
        analysis,
        archive_identities=archive_identities,
        preparation_git_commit=commit,
        ontology_version=ontology.ontology_version,
        ontology_sha256=ontology_sha256,
    )
    archive_receipt = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_archive_verification",
        "verified_at": timestamp,
        "archives": archive_identities,
        "selected_image_count": len(selected_left),
        "selected_label_id_count": len(selected_labels),
        "selected_uncompressed_bytes": selected_bytes,
    }
    ontology_identity = {
        "schema_version": "1.0",
        "record_type": "edgeguard_ontology_identity",
        "ontology_version": ontology.ontology_version,
        "ontology_status": ontology.ontology_status,
        "ontology_sha256": ontology_sha256,
    }
    preparation_receipt = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_preparation_receipt",
        "status": "prepared_pending_split_approval",
        "prepared_at": timestamp,
        "preparation_git_commit": commit,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "ontology_sha256": ontology_sha256,
        "recommended_candidate_id": comparison["recommended_candidate_id"],
        "selection_status": comparison["selection_status"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    dataset_summary = _compact_dataset_summary(analysis)
    group_summary = _group_summary(analysis)
    class_frequency = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_class_frequency",
        "global": _aggregate_without_sample_ids(analysis["global"]),
        "cities": {
            city: _aggregate_without_sample_ids(stats) for city, stats in analysis["cities"].items()
        },
        "rare_class_definition_candidates": analysis["rare_class_definition_candidates"],
        "heuristic_rare_class_ids": analysis["heuristic_rare_class_ids"],
    }

    _write_json(staging / "dataset_manifest.json", manifest)
    _write_json(staging / "preparation_receipt.json", preparation_receipt)
    report_files: dict[str, Any] = {
        "archive_verification_receipt.json": archive_receipt,
        "dataset_manifest.json": manifest,
        "dataset_summary.json": dataset_summary,
        "ontology_identity.json": ontology_identity,
        "preparation_receipt.json": preparation_receipt,
        "class_mapping_receipt.json": {
            "schema_version": "1.0",
            "record_type": "cityscapes_fine_train_class_mapping_receipt",
            "mappings": analysis["class_mapping_receipt"],
        },
        "class_frequency.json": class_frequency,
        "group_summary.json": group_summary,
        "split_candidate_comparison.json": comparison,
        "environment.json": _environment_summary(),
    }
    for name, payload in report_files.items():
        _write_json(manifest_staging / name, payload)
    (manifest_staging / "failures.jsonl").write_text("", encoding="utf-8")
    (manifest_staging / "split_candidate_report.md").write_text(
        render_split_report(comparison), encoding="utf-8"
    )
    candidate_paths = _write_candidate_manifests(manifest_staging, candidate_set)
    identity_names = sorted([*report_files, *candidate_paths, "split_candidate_report.md"])
    manifest_identities = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_manifest_identities",
        "files": [
            {
                "relative_path": name,
                "byte_size": (manifest_staging / name).stat().st_size,
                "sha256": sha256_file(manifest_staging / name),
            }
            for name in identity_names
        ],
    }
    manifest_identities["manifest_sha256"] = sha256_payload(manifest_identities)
    _write_json(manifest_staging / "manifest_identities.json", manifest_identities)
    evidence_names = [
        "archive_verification_receipt.json",
        "dataset_summary.json",
        "ontology_identity.json",
        "preparation_receipt.json",
        "class_mapping_receipt.json",
        "class_frequency.json",
        "group_summary.json",
        "split_candidate_comparison.json",
        "split_candidate_report.md",
        "manifest_identities.json",
        "environment.json",
        "failures.jsonl",
    ]
    evidence_receipt = _write_evidence_package(manifest_staging, evidence_names)

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifests_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, destination_incoming)
    shutil.copytree(manifest_staging, manifests_incoming)
    verification = _verify_prepared_train(
        destination_incoming,
        manifests_incoming,
        archive_identities=archive_identities,
        ontology_version=ontology.ontology_version,
        ontology_sha256=ontology_sha256,
    )
    destination_incoming.rename(destination)
    manifests_incoming.rename(manifests_destination)
    verification["evidence_package_filename"] = evidence_receipt["filename"]
    verification["evidence_package_sha256"] = evidence_receipt["sha256"]
    return verification


def main() -> int:
    """Prepare Cityscapes data and emit one path-free canonical result."""
    args = _parser().parse_args()
    try:
        if args.split == "train":
            if args.manifests_destination is None:
                raise ValueError("--manifests-destination is required for train preparation")
            if args.work_directory is None:
                raise ValueError("--work-directory is required for train preparation")
            if args.preparation_git_commit is None:
                raise ValueError("--preparation-git-commit is required for train preparation")
            _verify_execution_git_state(
                Path(__file__).resolve().parents[1], args.preparation_git_commit
            )
            result = prepare_cityscapes_train(
                args.left_images_archive,
                args.labels_archive,
                args.destination,
                args.manifests_destination,
                args.work_directory,
                preparation_git_commit=args.preparation_git_commit,
                ontology_config=args.ontology_config,
                verify_only=args.verify_only,
            )
        else:
            result = prepare_cityscapes_val(
                args.left_images_archive,
                args.labels_archive,
                args.destination,
                verify_only=args.verify_only,
            )
    except (BadZipFile, OSError, ValueError) as error:
        message = str(error)
        runtime_paths = (
            args.left_images_archive,
            args.labels_archive,
            args.destination,
            args.manifests_destination,
            args.work_directory,
            args.ontology_config,
        )
        for path in runtime_paths:
            if path is not None:
                message = message.replace(str(path), f"<{path.name or 'runtime-path'}>")
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": message}
            ),
            file=sys.stderr,
        )
        return 2
    if args.split == "train":
        print(canonical_json({"status": "ok", "split": "train", **result}))
    else:
        print(
            canonical_json(
                {
                    "status": "ok",
                    "split": "val",
                    "manifest_sha256": result["manifest_sha256"],
                    "image_count": result["image_count"],
                    "label_count": result["label_count"],
                    "city_count": result["city_count"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
