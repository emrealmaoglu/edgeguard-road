"""Colab/Drive data access, single-file bundling, and fail-closed staging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

GIB = 1024**3
SOURCE_DATASETS = ("cityscapes", "bdd100k", "idd20k")


def load_colab_data_access(path: Path) -> dict[str, Any]:
    """Load the strict storage/access contract used by both Colab notebooks."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("Colab data-access plan is missing or malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0"
        or payload.get("record_type") != "edgeguard_colab_data_access"
    ):
        raise ValueError("Colab data-access plan must use schema 1.0")
    storage = payload.get("storage")
    datasets = payload.get("datasets")
    if not isinstance(storage, dict) or not isinstance(datasets, dict):
        raise ValueError("Colab data-access plan requires storage and datasets mappings")
    if int(storage.get("maximum_staged_gib", 0)) + int(storage.get("reserve_gib", 0)) > int(
        storage.get("colab_ephemeral_limit_gib", 0)
    ):
        raise ValueError("staging budget and reserve exceed the Colab limit")
    for dataset_id, record in datasets.items():
        if not isinstance(record, dict):
            raise ValueError(f"{dataset_id} access record must be a mapping")
        for field in (
            "campaign_role",
            "activation_phase",
            "access_method",
            "official_url",
            "instructions",
            "packages",
            "prepared_subdirectory",
            "required_paths",
        ):
            if field not in record:
                raise ValueError(f"{dataset_id} is missing access field {field}")
        if not str(record["official_url"]).startswith("https://"):
            raise ValueError(f"{dataset_id} official URL must use HTTPS")
        if not isinstance(record["packages"], list) or not isinstance(
            record["required_paths"], list
        ):
            raise ValueError(f"{dataset_id} packages and required_paths must be lists")
        for relative in record["required_paths"]:
            path_value = Path(str(relative))
            if path_value.is_absolute() or ".." in path_value.parts:
                raise ValueError(f"{dataset_id} has an unsafe required path")
    if not set(SOURCE_DATASETS).issubset(datasets):
        raise ValueError("Colab access plan omits a required source domain")
    return payload


def initialize_drive_layout(plan: dict[str, Any], drive_root: Path) -> dict[str, str]:
    """Create the bounded project directory skeleton without acquiring any dataset."""
    storage = plan["storage"]
    project = drive_root / str(storage["drive_project_directory"])
    directories = {
        "project": project,
        "datasets": project / str(storage["dataset_directory"]),
        "archives": project / str(storage["archive_directory"]),
        "bundles": project / str(storage["bundle_directory"]),
        "artifacts": project / "artifacts",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for dataset_id in plan["datasets"]:
        (directories["archives"] / dataset_id).mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in directories.items()}


def _tree_inventory(root: Path) -> tuple[int, int]:
    files = 0
    byte_size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"dataset tree contains a symlink: {path}")
        if path.is_file():
            files += 1
            byte_size += path.stat().st_size
    return files, byte_size


def _file_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def inventory_colab_data(
    plan: dict[str, Any], drive_root: Path, *, hash_archives: bool = False
) -> dict[str, Any]:
    """Report exact local readiness without treating catalog facts as acquired data."""
    paths = initialize_drive_layout(plan, drive_root)
    dataset_root = Path(paths["datasets"])
    archive_root = Path(paths["archives"])
    rows: list[dict[str, Any]] = []
    for dataset_id, record in plan["datasets"].items():
        prepared = dataset_root / str(record["prepared_subdirectory"])
        required = [str(value) for value in record["required_paths"]]
        missing_paths = [relative for relative in required if not (prepared / relative).exists()]
        packages = []
        for package in record["packages"]:
            filename = str(package["filename"])
            candidate = archive_root / dataset_id / filename
            present = candidate.is_file()
            published_md5 = package.get("published_md5")
            sha256_value: str | None = None
            md5_value: str | None = None
            if present and hash_archives:
                sha256_value, md5_value = _file_digests(candidate)
            packages.append(
                {
                    "filename": filename,
                    "present": present,
                    "byte_size": candidate.stat().st_size if present else None,
                    "published_md5": published_md5,
                    "sha256": sha256_value,
                    "md5": md5_value,
                    "published_md5_matches": (
                        md5_value == published_md5
                        if md5_value is not None and published_md5 is not None
                        else None
                    ),
                }
            )
        if prepared.is_dir() and not missing_paths:
            file_count, byte_size = _tree_inventory(prepared)
            state = "prepared"
        else:
            file_count, byte_size = 0, 0
            state = "needs_manual_preparation"
        rows.append(
            {
                "dataset_id": dataset_id,
                "campaign_role": record["campaign_role"],
                "activation_phase": record["activation_phase"],
                "access_method": record["access_method"],
                "official_url": record["official_url"],
                "instructions": record["instructions"],
                "state": state,
                "prepared_root": str(prepared),
                "missing_required_paths": missing_paths,
                "prepared_file_count": file_count,
                "prepared_bytes": byte_size,
                "packages": packages,
            }
        )
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_data_inventory",
        "plan_sha256": sha256_payload(plan),
        "drive_paths": paths,
        "datasets": rows,
        "excluded_sources": plan.get("excluded_sources", []),
    }


def _bundle_receipt_path(bundle: Path) -> Path:
    return bundle.with_suffix(bundle.suffix + ".receipt.json")


def create_dataset_bundle(
    plan: dict[str, Any], drive_root: Path, dataset_id: str, *, replace: bool = False
) -> dict[str, Any]:
    """Create one deterministic uncompressed tar for fast, sequential Drive staging."""
    if dataset_id not in plan["datasets"]:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    inventory = inventory_colab_data(plan, drive_root)
    row = next(item for item in inventory["datasets"] if item["dataset_id"] == dataset_id)
    if row["state"] != "prepared":
        raise ValueError(f"{dataset_id} is not prepared: {row['missing_required_paths']}")
    paths = inventory["drive_paths"]
    source = Path(row["prepared_root"])
    bundle = Path(paths["bundles"]) / f"{dataset_id}.prepared.tar"
    receipt_path = _bundle_receipt_path(bundle)
    if bundle.exists() or receipt_path.exists():
        if not replace:
            if not bundle.is_file() or not receipt_path.is_file():
                raise ValueError(f"{dataset_id} bundle is partial")
            existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_hash = existing_receipt.pop("receipt_sha256", None)
            if receipt_hash != sha256_payload(existing_receipt):
                raise ValueError(f"{dataset_id} existing bundle receipt hash mismatch")
            existing_receipt["receipt_sha256"] = receipt_hash
            if existing_receipt.get("sha256") != sha256_file(bundle):
                raise ValueError(f"{dataset_id} existing bundle hash mismatch")
            return {**existing_receipt, "status": "reused"}
        bundle.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    incoming = bundle.with_name(f".{bundle.name}.incoming")
    if incoming.exists():
        raise ValueError(f"stale incoming bundle must be inspected: {incoming}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(incoming, mode="x") as archive:
        ordered_paths = sorted(
            source.rglob("*"), key=lambda value: value.relative_to(source).as_posix()
        )
        for path in ordered_paths:
            if path.is_symlink():
                raise ValueError(f"dataset tree contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    file_count, source_bytes = _tree_inventory(source)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_prepared_dataset_bundle",
        "dataset_id": dataset_id,
        "filename": bundle.name,
        "byte_size": incoming.stat().st_size,
        "source_bytes": source_bytes,
        "file_count": file_count,
        "sha256": sha256_file(incoming),
        "plan_sha256": sha256_payload(plan),
        "required_paths": plan["datasets"][dataset_id]["required_paths"],
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    os.replace(incoming, bundle)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return {**receipt, "status": "created"}


def _validate_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    names: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not member.isfile()
            or member.name in names
        ):
            raise ValueError(f"unsafe or duplicate dataset bundle member: {member.name}")
        names.add(member.name)
    return members


def stage_dataset_bundles(
    plan: dict[str, Any],
    drive_root: Path,
    local_root: Path,
    dataset_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Copy, hash, extract, and validate selected bundles within the 200 GiB budget."""
    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("dataset_ids must be non-empty and unique")
    unknown = set(dataset_ids) - set(plan["datasets"])
    if unknown:
        raise ValueError(f"unknown dataset ids: {sorted(unknown)}")
    paths = initialize_drive_layout(plan, drive_root)
    receipts: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        bundle = Path(paths["bundles"]) / f"{dataset_id}.prepared.tar"
        receipt_path = _bundle_receipt_path(bundle)
        if not bundle.is_file() or not receipt_path.is_file():
            raise ValueError(f"missing prepared bundle for {dataset_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt_hash = receipt.pop("receipt_sha256", None)
        if expected_receipt_hash != sha256_payload(receipt):
            raise ValueError(f"{dataset_id} bundle receipt hash mismatch")
        receipt["receipt_sha256"] = expected_receipt_hash
        if receipt.get("dataset_id") != dataset_id or receipt.get("plan_sha256") != sha256_payload(
            plan
        ):
            raise ValueError(f"{dataset_id} bundle identity mismatch")
        if bundle.stat().st_size != int(receipt["byte_size"]):
            raise ValueError(f"{dataset_id} bundle size mismatch")
        receipts.append(receipt)
    storage = plan["storage"]
    configured_limit = int(storage["colab_ephemeral_limit_gib"]) * GIB
    reserve = int(storage["reserve_gib"]) * GIB
    _actual_total, _used, actual_free = shutil.disk_usage(local_root.parent)
    usable_total = min(configured_limit, actual_free)
    if actual_free < reserve:
        raise OSError("Colab runtime has less than the required reserve")
    staged_bytes = 0
    peak_bytes = 0
    for receipt in receipts:
        peak_bytes = max(
            peak_bytes,
            staged_bytes + int(receipt["byte_size"]) + int(receipt["source_bytes"]) + reserve,
        )
        staged_bytes += int(receipt["source_bytes"])
    if staged_bytes > int(storage["maximum_staged_gib"]) * GIB or peak_bytes > usable_total:
        raise OSError(
            "selected datasets exceed the configured 175 GiB staging or 200 GiB peak budget"
        )
    local_root.mkdir(parents=True, exist_ok=True)
    cache = local_root.parent / ".edgeguard-bundle-cache"
    cache.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for dataset_id, receipt in zip(dataset_ids, receipts, strict=True):
        destination = local_root / str(plan["datasets"][dataset_id]["prepared_subdirectory"])
        required = [str(value) for value in receipt["required_paths"]]
        if destination.is_dir() and all((destination / value).exists() for value in required):
            results.append({"dataset_id": dataset_id, "status": "already_staged"})
            continue
        if destination.exists():
            raise ValueError(f"partial local dataset destination exists: {destination}")
        drive_bundle = Path(paths["bundles"]) / str(receipt["filename"])
        local_bundle = cache / drive_bundle.name
        partial = local_bundle.with_suffix(local_bundle.suffix + ".part")
        shutil.copyfile(drive_bundle, partial)
        if sha256_file(partial) != receipt["sha256"]:
            partial.unlink(missing_ok=True)
            raise ValueError(f"{dataset_id} local bundle SHA-256 mismatch")
        os.replace(partial, local_bundle)
        incoming = destination.with_name(f".{destination.name}.incoming")
        if incoming.exists():
            raise ValueError(f"stale extraction directory must be inspected: {incoming}")
        incoming.mkdir(parents=True)
        with tarfile.open(local_bundle, mode="r:") as archive:
            members = _validate_tar_members(archive)
            if len(members) != int(receipt["file_count"]):
                raise ValueError(f"{dataset_id} bundle file count mismatch")
            for member in members:
                source_stream = archive.extractfile(member)
                if source_stream is None:
                    raise ValueError(f"{dataset_id} bundle member has no file content")
                target = incoming / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source_stream, target.open("xb") as destination_stream:
                    shutil.copyfileobj(source_stream, destination_stream, length=8 * 1024**2)
        local_bundle.unlink()
        if not all((incoming / value).exists() for value in required):
            raise ValueError(f"{dataset_id} staged tree failed required-path validation")
        files, byte_size = _tree_inventory(incoming)
        if files != int(receipt["file_count"]) or byte_size != int(receipt["source_bytes"]):
            raise ValueError(f"{dataset_id} staged tree inventory mismatch")
        os.replace(incoming, destination)
        results.append({"dataset_id": dataset_id, "status": "staged_verified"})
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_staging_receipt",
        "datasets": results,
        "staged_bytes": staged_bytes,
        "planned_peak_bytes": peak_bytes,
        "configured_limit_bytes": configured_limit,
        "reserve_bytes": reserve,
    }
