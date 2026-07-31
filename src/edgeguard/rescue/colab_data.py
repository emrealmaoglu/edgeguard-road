"""Colab/Drive data access, single-file bundling, and fail-closed staging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import yaml

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

GIB = 1024**3
SOURCE_DATASETS = ("cityscapes", "bdd100k", "idd20k")
STORAGE_DIRECTORIES = {
    "datasets": "dataset_directory",
    "archives": "archive_directory",
    "bundles": "bundle_directory",
    "manifests": "manifest_directory",
    "campaigns": "campaign_directory",
    "downloads": "download_directory",
    "quarantine": "quarantine_directory",
    "source": "source_directory",
    "private_inputs": "private_input_directory",
    "failures": "failure_directory",
    "prepared": "prepared_directory",
    "runtime_cache": "runtime_cache_directory",
    "review_packages": "review_package_directory",
}


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
    if float(storage.get("preparation_archive_multiplier", 0)) < 1.0:
        raise ValueError("preparation archive multiplier must be at least 1.0")
    if int(storage.get("preparation_reserve_gib", -1)) < 0:
        raise ValueError("preparation reserve must be non-negative")
    required_storage = {"drive_project_directory", *STORAGE_DIRECTORIES.values()}
    missing_storage = sorted(required_storage - storage.keys())
    if missing_storage:
        raise ValueError(f"Colab storage contract is missing: {missing_storage}")
    for field in required_storage:
        relative = Path(str(storage[field]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"unsafe Colab storage directory: {field}")
    legacy = storage.get("legacy_compatibility", {})
    if not isinstance(legacy, dict):
        raise ValueError("legacy Drive compatibility must be a mapping")
    for dataset_id, record in legacy.items():
        if dataset_id not in datasets or not isinstance(record, dict):
            raise ValueError("legacy Drive compatibility names an invalid dataset")
        for field in (
            "archive_directory",
            "prepared_directory",
            "manifest_file",
            "split_file",
            "bundle_directory",
            "bundle_filename",
            "receipt_filename",
            "bundle_sha256",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "byte_size",
            "source_bytes",
            "file_count",
            "required_paths",
        ):
            if field not in record:
                raise ValueError(f"legacy {dataset_id} contract is missing {field}")
        for field in (
            "archive_directory",
            "prepared_directory",
            "manifest_file",
            "split_file",
            "bundle_directory",
            "bundle_filename",
            "receipt_filename",
        ):
            relative = Path(str(record[field]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"legacy {dataset_id} has an unsafe {field}")
        for field in ("bundle_sha256", "dataset_manifest_sha256", "split_manifest_sha256"):
            value = str(record[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"legacy {dataset_id} has an invalid {field}")
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
        engineering_packages = record.get("engineering_packages", [])
        if not isinstance(engineering_packages, list):
            raise ValueError(f"{dataset_id} engineering_packages must be a list")
        package_names: set[str] = set()
        for package in [*record["packages"], *engineering_packages]:
            if not isinstance(package, dict) or "filename" not in package:
                raise ValueError(f"{dataset_id} package record is malformed")
            filename = Path(str(package["filename"]))
            if filename.is_absolute() or ".." in filename.parts or len(filename.parts) != 1:
                raise ValueError(f"{dataset_id} has an unsafe package filename")
            if filename.name in package_names:
                raise ValueError(f"{dataset_id} repeats package filename {filename.name}")
            package_names.add(filename.name)
        for package in engineering_packages:
            if package.get("scientific_eligible") is not False:
                raise ValueError(f"{dataset_id} engineering package must be science-ineligible")
            if not str(package.get("source_profile", "")):
                raise ValueError(f"{dataset_id} engineering package needs a source profile")
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
    directories = {"project": project}
    directories.update(
        {name: project / str(storage[field]) for name, field in STORAGE_DIRECTORIES.items()}
    )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for dataset_id in plan["datasets"]:
        (directories["archives"] / dataset_id).mkdir(parents=True, exist_ok=True)
    for versioned in (directories["prepared"], directories["manifests"]):
        (versioned / "v2").mkdir(parents=True, exist_ok=True)
    (directories["quarantine"] / "kaggle" / "bdd100k").mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in directories.items()}


def preparation_disk_budget(
    plan: dict[str, Any], archives: tuple[Path, ...], filesystem_root: Path
) -> dict[str, Any]:
    """Fail closed before copying archives or expanding a dataset in ephemeral storage."""
    if not archives or len({path.resolve() for path in archives}) != len(archives):
        raise ValueError("preparation archives must be non-empty and unique")
    missing = [str(path) for path in archives if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError(f"preparation archives are missing or unsafe: {missing}")
    root = filesystem_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive_bytes = sum(path.stat().st_size for path in archives)
    storage = plan["storage"]
    multiplier = float(storage["preparation_archive_multiplier"])
    reserve_bytes = int(storage["preparation_reserve_gib"]) * GIB
    estimated_peak_bytes = int(archive_bytes * multiplier) + reserve_bytes
    usage = shutil.disk_usage(root)
    policy_limit_bytes = int(storage["colab_ephemeral_limit_gib"]) * GIB
    policy_available_bytes = max(0, policy_limit_bytes - min(usage.used, policy_limit_bytes))
    available_bytes = min(usage.free, policy_available_bytes)
    if estimated_peak_bytes > available_bytes:
        raise ValueError(
            "insufficient ephemeral storage for preparation: "
            f"required={estimated_peak_bytes}, available={available_bytes}"
        )
    return {
        "archive_bytes": archive_bytes,
        "archive_multiplier": multiplier,
        "reserve_bytes": reserve_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "filesystem_free_bytes": usage.free,
        "policy_available_bytes": policy_available_bytes,
        "available_bytes": available_bytes,
        "passes": True,
    }


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


def _cached_file_digests(path: Path, receipt_path: Path) -> tuple[str, str, str]:
    """Hash a Drive archive once and reuse the receipt while its stat identity is stable."""
    stat = path.stat()
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_hash = receipt.pop("receipt_sha256", None)
            valid = (
                receipt_hash == sha256_payload(receipt)
                and receipt.get("byte_size") == stat.st_size
                and receipt.get("mtime_ns") == stat.st_mtime_ns
                and isinstance(receipt.get("sha256"), str)
                and isinstance(receipt.get("md5"), str)
            )
            if valid:
                return str(receipt["sha256"]), str(receipt["md5"]), "cached"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    sha256_value, md5_value = _file_digests(path)
    receipt = {
        "schema_version": "2.0",
        "record_type": "edgeguard_archive_digest",
        "filename": path.name,
        "byte_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_value,
        "md5": md5_value,
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = receipt_path.with_name(f".{receipt_path.name}.incoming")
    incoming.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    os.replace(incoming, receipt_path)
    return sha256_value, md5_value, "computed"


class _HashingProgressWriter:
    """Hash a sequential tar while periodically proving forward progress."""

    def __init__(self, stream: BinaryIO, *, dataset_id: str, interval_bytes: int = 256 * 1024**2):
        self.stream = stream
        self.dataset_id = dataset_id
        self.interval_bytes = interval_bytes
        self.bytes_written = 0
        self._next_report = interval_bytes
        self._sha256 = hashlib.sha256()

    def write(self, payload: bytes) -> int:
        written = self.stream.write(payload)
        if written != len(payload):
            raise OSError("short write while creating dataset bundle")
        self._sha256.update(payload)
        self.bytes_written += written
        if self.bytes_written >= self._next_report:
            print(
                canonical_json(
                    {
                        "phase": "bundle-write",
                        "dataset_id": self.dataset_id,
                        "bytes_written": self.bytes_written,
                    }
                ),
                flush=True,
            )
            self._next_report = self.bytes_written + self.interval_bytes
        return written

    def hexdigest(self) -> str:
        """Return the digest of bytes accepted by the underlying stream."""
        return self._sha256.hexdigest()


class _HashingProgressReader:
    """Hash a sequential copy while periodically proving forward progress."""

    def __init__(
        self,
        stream: BinaryIO,
        *,
        label: str,
        phase: str,
        interval_bytes: int = 256 * 1024**2,
    ):
        self.stream = stream
        self.label = label
        self.phase = phase
        self.interval_bytes = interval_bytes
        self.bytes_read = 0
        self._next_report = interval_bytes
        self._sha256 = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        payload = self.stream.read(size)
        if payload:
            self._sha256.update(payload)
            self.bytes_read += len(payload)
            if self.bytes_read >= self._next_report:
                print(
                    canonical_json(
                        {
                            "phase": self.phase,
                            "label": self.label,
                            "bytes_read": self.bytes_read,
                        }
                    ),
                    flush=True,
                )
                self._next_report = self.bytes_read + self.interval_bytes
        return payload

    def hexdigest(self) -> str:
        """Return the digest of bytes returned by this reader."""
        return self._sha256.hexdigest()


def copy_archive_to_local(source: Path, destination: Path, *, attempts: int = 3) -> dict[str, Any]:
    """Copy one mounted-Drive archive with bounded retries and an atomic destination."""
    if attempts <= 0:
        raise ValueError("archive copy attempts must be positive")
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"archive source is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        try:
            with source.open("rb") as input_stream, partial.open("xb") as output_stream:
                progress = _HashingProgressReader(
                    input_stream,
                    label=source.name,
                    phase="archive-copy",
                )
                shutil.copyfileobj(progress, output_stream, length=8 * 1024**2)
            if partial.stat().st_size != source.stat().st_size:
                raise OSError("archive copy size mismatch")
            partial.replace(destination)
            return {
                "source": str(source),
                "destination": str(destination),
                "byte_size": destination.stat().st_size,
                "sha256": progress.hexdigest(),
                "attempts": attempt,
            }
        except OSError as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            partial.unlink(missing_ok=True)
    raise OSError(f"archive copy failed after {attempts} attempts: {source}; " + " | ".join(errors))


def inventory_colab_data(
    plan: dict[str, Any], drive_root: Path, *, hash_archives: bool = False
) -> dict[str, Any]:
    """Report exact local readiness without treating catalog facts as acquired data."""
    paths = initialize_drive_layout(plan, drive_root)
    dataset_root = Path(paths["datasets"])
    archive_root = Path(paths["archives"])
    private_input_root = Path(paths["private_inputs"])
    project_root = Path(paths["project"])
    digest_root = Path(paths["manifests"]) / "archive-digests"
    legacy_contracts = plan["storage"].get("legacy_compatibility", {})
    rows: list[dict[str, Any]] = []
    for dataset_id, record in plan["datasets"].items():
        legacy_record = legacy_contracts.get(dataset_id)
        prepared = dataset_root / str(record["prepared_subdirectory"])
        required = [str(value) for value in record["required_paths"]]
        missing_paths = [relative for relative in required if not (prepared / relative).exists()]
        packages = []
        for package in record["packages"]:
            filename = str(package["filename"])
            candidate = archive_root / dataset_id / filename
            location_profile = "canonical_archives"
            if not candidate.is_file() and isinstance(legacy_record, dict):
                legacy_candidate = project_root / str(legacy_record["archive_directory"]) / filename
                if legacy_candidate.is_file():
                    candidate = legacy_candidate
                    location_profile = "legacy_private_inputs"
            if not candidate.is_file():
                private_candidate = private_input_root / filename
                if private_candidate.is_file():
                    candidate = private_candidate
                    location_profile = "private_inputs"
            present = candidate.is_file()
            published_md5 = package.get("published_md5")
            published_sha256 = package.get("published_sha256")
            sha256_value: str | None = None
            md5_value: str | None = None
            hash_error: str | None = None
            hash_status = "not_requested" if present else "missing"
            if present and hash_archives:
                try:
                    sha256_value, md5_value, hash_status = _cached_file_digests(
                        candidate, digest_root / f"{dataset_id}-{filename}.json"
                    )
                except OSError as error:
                    hash_error = f"{type(error).__name__}: {error}"
                    hash_status = "read_error"
            packages.append(
                {
                    "filename": filename,
                    "path": str(candidate),
                    "present": present,
                    "byte_size": candidate.stat().st_size if present else None,
                    "published_md5": published_md5,
                    "published_sha256": published_sha256,
                    "sha256": sha256_value,
                    "md5": md5_value,
                    "hash_status": hash_status,
                    "hash_error": hash_error,
                    "published_md5_matches": (
                        md5_value == published_md5
                        if md5_value is not None and published_md5 is not None
                        else None
                    ),
                    "published_sha256_matches": (
                        sha256_value == published_sha256
                        if sha256_value is not None and published_sha256 is not None
                        else None
                    ),
                    "location_profile": location_profile if present else None,
                }
            )
        engineering_packages = []
        for package in record.get("engineering_packages", []):
            filename = str(package["filename"])
            candidates = (
                private_input_root / filename,
                Path(paths["quarantine"]) / str(package["source_profile"]) / dataset_id / filename,
                Path(paths["quarantine"]) / "kaggle" / dataset_id / filename,
            )
            candidate = next((value for value in candidates if value.is_file()), candidates[0])
            present = candidate.is_file()
            engineering_sha256: str | None = None
            engineering_md5: str | None = None
            engineering_hash_error: str | None = None
            engineering_hash_status = "not_requested" if present else "missing"
            if present and hash_archives:
                try:
                    (
                        engineering_sha256,
                        engineering_md5,
                        engineering_hash_status,
                    ) = _cached_file_digests(
                        candidate, digest_root / f"{dataset_id}-{filename}.engineering.json"
                    )
                except OSError as error:
                    engineering_hash_error = f"{type(error).__name__}: {error}"
                    engineering_hash_status = "read_error"
            engineering_packages.append(
                {
                    **package,
                    "path": str(candidate),
                    "present": present,
                    "byte_size": candidate.stat().st_size if present else None,
                    "sha256": engineering_sha256,
                    "md5": engineering_md5,
                    "hash_status": engineering_hash_status,
                    "hash_error": engineering_hash_error,
                    "location_profile": (
                        "private_inputs" if candidate.parent == private_input_root else "quarantine"
                    )
                    if present
                    else None,
                }
            )
        legacy_state: dict[str, Any] | None = None
        if isinstance(legacy_record, dict):
            legacy_bundle = (
                project_root
                / str(legacy_record["bundle_directory"])
                / str(legacy_record["bundle_filename"])
            )
            legacy_receipt = (
                project_root
                / str(legacy_record["bundle_directory"])
                / str(legacy_record["receipt_filename"])
            )
            legacy_archives = project_root / str(legacy_record["archive_directory"])
            legacy_prepared = project_root / str(legacy_record["prepared_directory"])
            legacy_state = {
                "bundle_present": legacy_bundle.is_file(),
                "receipt_present": legacy_receipt.is_file(),
                "bundle_path": str(legacy_bundle),
                "archive_root_present": legacy_archives.is_dir(),
                "prepared_root_present": legacy_prepared.is_dir(),
                "usable_for_training_staging": legacy_bundle.is_file() and legacy_receipt.is_file(),
            }
        canonical_bundle = Path(paths["bundles"]) / f"{dataset_id}.prepared.tar"
        canonical_receipt = _bundle_receipt_path(canonical_bundle)
        ineligible_bundle = (
            Path(paths["bundles"]) / "bdd100k.kaggle_mirror.prepared.tar"
            if dataset_id == "bdd100k"
            else None
        )
        ineligible_receipt = (
            _bundle_receipt_path(ineligible_bundle) if ineligible_bundle is not None else None
        )
        if prepared.is_dir() and not missing_paths:
            file_count, byte_size = _tree_inventory(prepared)
            state = "prepared"
        elif canonical_bundle.is_file() and canonical_receipt.is_file():
            file_count, byte_size = 0, 0
            state = "canonical_bundle_ready"
        elif legacy_state and legacy_state["usable_for_training_staging"]:
            assert isinstance(legacy_record, dict)
            file_count = int(legacy_record["file_count"])
            byte_size = int(legacy_record["source_bytes"])
            state = "legacy_bundle_ready"
        elif packages and all(package["present"] for package in packages):
            file_count, byte_size = 0, 0
            state = "archives_ready"
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
                "engineering_packages": engineering_packages,
                "engineering_state": (
                    "archives_ready"
                    if engineering_packages
                    and all(package["present"] for package in engineering_packages)
                    else "not_configured_or_missing"
                ),
                "legacy_compatibility": legacy_state,
                "canonical_bundle_usable": canonical_bundle.is_file()
                and canonical_receipt.is_file(),
                "ineligible_smoke_bundle_usable": bool(
                    ineligible_bundle
                    and ineligible_receipt
                    and ineligible_bundle.is_file()
                    and ineligible_receipt.is_file()
                ),
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
    plan: dict[str, Any],
    drive_root: Path,
    dataset_id: str,
    *,
    replace: bool = False,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Create one deterministic uncompressed tar for fast, sequential Drive staging."""
    if dataset_id not in plan["datasets"]:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    inventory = inventory_colab_data(plan, drive_root)
    row = next(item for item in inventory["datasets"] if item["dataset_id"] == dataset_id)
    source = source_root.resolve() if source_root is not None else Path(row["prepared_root"])
    required = [str(value) for value in plan["datasets"][dataset_id]["required_paths"]]
    missing = [relative for relative in required if not (source / relative).exists()]
    if not source.is_dir() or missing:
        raise ValueError(f"{dataset_id} is not prepared: {missing}")
    paths = inventory["drive_paths"]
    preparation_path = source / "preparation_receipt.json"
    preparation = (
        json.loads(preparation_path.read_text(encoding="utf-8"))
        if preparation_path.is_file()
        else {}
    )
    source_profile = str(preparation.get("source_profile", "official"))
    scientific_eligible = bool(preparation.get("scientific_eligible", True))
    if dataset_id == "bdd100k" and source_profile == "kaggle_mirror":
        bundle_name = "bdd100k.kaggle_mirror.prepared.tar"
    else:
        bundle_name = f"{dataset_id}.prepared.tar"
    bundle = Path(paths["bundles"]) / bundle_name
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
            if bundle.stat().st_size != int(existing_receipt.get("byte_size", -1)):
                raise ValueError(f"{dataset_id} existing bundle size mismatch")
            return {**existing_receipt, "status": "reused"}
        bundle.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    incoming = bundle.with_name(f".{bundle.name}.incoming")
    if incoming.exists():
        raise ValueError(f"stale incoming bundle must be inspected: {incoming}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    ordered_paths: list[Path] = []
    source_bytes = 0
    for path in sorted(source.rglob("*"), key=lambda value: value.relative_to(source).as_posix()):
        if path.is_symlink():
            raise ValueError(f"dataset tree contains a symlink: {path}")
        if path.is_file():
            ordered_paths.append(path)
            source_bytes += path.stat().st_size
    file_count = len(ordered_paths)
    with incoming.open("xb") as raw_stream:
        writer = _HashingProgressWriter(raw_stream, dataset_id=dataset_id)
        with tarfile.open(fileobj=cast(Any, writer), mode="w|") as archive:
            for file_index, path in enumerate(ordered_paths, start=1):
                if file_index == 1 or file_index % 1000 == 0 or file_index == file_count:
                    print(
                        canonical_json(
                            {
                                "phase": "bundle-files",
                                "dataset_id": dataset_id,
                                "completed": file_index,
                                "total": file_count,
                            }
                        ),
                        flush=True,
                    )
                relative = path.relative_to(source).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
        bundle_sha256 = writer.hexdigest()
        bundle_byte_size = writer.bytes_written
    if incoming.stat().st_size != bundle_byte_size:
        raise OSError("dataset bundle size changed after close")
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_prepared_dataset_bundle",
        "dataset_id": dataset_id,
        "filename": bundle.name,
        "byte_size": bundle_byte_size,
        "source_bytes": source_bytes,
        "file_count": file_count,
        "sha256": bundle_sha256,
        "plan_sha256": sha256_payload(plan),
        "required_paths": plan["datasets"][dataset_id]["required_paths"],
        "source_profile": source_profile,
        "scientific_eligible": scientific_eligible,
        "source_preparation_receipt_sha256": (
            sha256_file(source / "preparation_receipt.json")
            if (source / "preparation_receipt.json").is_file()
            else None
        ),
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


def _load_idd_shard_index(project: Path) -> tuple[Path, dict[str, Any]] | None:
    shard_root = project / "prepared/v2/idd20k/shards"
    index_path = shard_root / "idd20k.shards.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("IDD shard index is malformed") from error
    receipt_hash = index.pop("receipt_sha256", None)
    if (
        receipt_hash != sha256_payload(index)
        or index.get("record_type") != "edgeguard_idd_shard_index"
        or index.get("dataset_id") != "idd20k"
        or index.get("counts") != {"train": 14_027, "val": 2_036}
        or int(index.get("sample_count", -1)) != 16_063
        or sum(int(row.get("sample_count", -1)) for row in index.get("shards", [])) != 16_063
        or index.get("scientific_eligible") is not True
    ):
        raise ValueError("IDD shard index identity/count contract failed")
    index["receipt_sha256"] = receipt_hash
    return shard_root, index


def _stage_idd_shards(plan: dict[str, Any], project: Path, local_root: Path) -> dict[str, Any]:
    loaded = _load_idd_shard_index(project)
    if loaded is None:
        raise ValueError("IDD shard staging requested without a shard index")
    shard_root, index = loaded
    destination = local_root / str(plan["datasets"]["idd20k"]["prepared_subdirectory"])
    required = [str(value) for value in plan["datasets"]["idd20k"]["required_paths"]]
    if destination.is_dir() and all((destination / value).exists() for value in required):
        return {"dataset_id": "idd20k", "status": "already_staged_shards"}
    if destination.exists():
        raise ValueError(f"partial local IDD destination exists: {destination}")
    reserve = int(plan["storage"]["reserve_gib"]) * GIB
    source_bytes = sum(int(row["source_bytes"]) for row in index["shards"])
    if source_bytes + reserve > shutil.disk_usage(local_root.parent).free:
        raise OSError("IDD shards and required reserve exceed current Colab free space")
    incoming = destination.with_name(f".{destination.name}.incoming")
    if incoming.exists():
        raise ValueError(f"stale IDD shard extraction exists: {incoming}")
    incoming.mkdir(parents=True)
    cache = local_root.parent / ".edgeguard-bundle-cache/idd20k-shards"
    cache.mkdir(parents=True, exist_ok=True)
    total = len(index["shards"])
    extracted_files = 0
    for shard_number, row in enumerate(index["shards"], start=1):
        shard = shard_root / str(row["filename"])
        receipt_path = shard_root / Path(str(row["filename"])).with_suffix(".receipt.json")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"IDD shard receipt missing: {receipt_path.name}") from error
        receipt_hash = receipt.pop("receipt_sha256", None)
        if (
            receipt_hash != sha256_payload(receipt)
            or receipt.get("sha256") != row["sha256"]
            or shard.stat().st_size != int(row["byte_size"])
        ):
            raise ValueError(f"IDD shard identity mismatch: {shard.name}")
        local_shard = cache / shard.name
        partial = local_shard.with_suffix(".part")
        partial.unlink(missing_ok=True)
        with shard.open("rb") as input_stream, partial.open("xb") as output_stream:
            progress = _HashingProgressReader(
                input_stream, label=shard.name, phase="stage-idd-shard-copy"
            )
            shutil.copyfileobj(progress, output_stream, length=8 * 1024**2)
        if progress.hexdigest() != row["sha256"]:
            partial.unlink(missing_ok=True)
            raise ValueError(f"IDD shard copy hash mismatch: {shard.name}")
        os.replace(partial, local_shard)
        with tarfile.open(local_shard, "r:") as archive:
            members = _validate_tar_members(archive)
            if len(members) != int(row["file_count"]):
                raise ValueError(f"IDD shard file-count mismatch: {shard.name}")
            for member in members:
                target = incoming / member.name
                if target.exists():
                    raise ValueError(
                        f"IDD shards contain a normalized-path collision: {member.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"IDD shard member has no content: {member.name}")
                with source, target.open("xb") as output_stream:
                    shutil.copyfileobj(source, output_stream, length=8 * 1024**2)
                extracted_files += 1
        local_shard.unlink()
        print(
            canonical_json(
                {
                    "phase": "stage-idd-shards",
                    "completed": shard_number,
                    "total": total,
                    "files": extracted_files,
                }
            ),
            flush=True,
        )
    if not all((incoming / value).exists() for value in required):
        raise ValueError("staged IDD shards failed required-path validation")
    preparation: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_preparation_receipt",
        "dataset_id": "idd20k",
        "source_profile": "official",
        "source_archives": index["source_archives"],
        "counts": index["counts"],
        "mapping_version": "autonue-polygon-to-cityscapes19-canonical-shards-v2",
        "ontology_sha256": index["ontology_sha256"],
        "native_labels_preserved": True,
        "prepared_payload": "images_and_cityscapes19_canonical_masks_only",
        "scientific_eligible": True,
        "shard_index_sha256": sha256_file(shard_root / "idd20k.shards.json"),
    }
    preparation["receipt_sha256"] = sha256_payload(preparation)
    (incoming / "preparation_receipt.json").write_text(
        canonical_json(preparation) + "\n", encoding="utf-8"
    )
    os.replace(incoming, destination)
    return {
        "dataset_id": "idd20k",
        "status": "staged_verified_shards",
        "shard_count": total,
        "sample_count": int(index["sample_count"]),
        "file_count": extracted_files + 1,
    }


def _canonical_bundle_receipt(
    plan: dict[str, Any],
    paths: dict[str, str],
    dataset_id: str,
    *,
    allow_ineligible: bool,
) -> tuple[dict[str, Any], Path, str]:
    candidates = [Path(paths["bundles"]) / f"{dataset_id}.prepared.tar"]
    if dataset_id == "bdd100k" and allow_ineligible:
        candidates.append(Path(paths["bundles"]) / "bdd100k.kaggle_mirror.prepared.tar")
    partial: list[str] = []
    for bundle in candidates:
        receipt_path = _bundle_receipt_path(bundle)
        if not bundle.is_file() and not receipt_path.is_file():
            continue
        if not bundle.is_file() or not receipt_path.is_file():
            partial.append(bundle.name)
            continue
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
        if receipt.get("scientific_eligible") is False and not allow_ineligible:
            raise ValueError(f"{dataset_id} ineligible bundle is blocked from scientific staging")
        profile = str(receipt.get("source_profile", "official"))
        return receipt, bundle, f"canonical_v1:{profile}"
    if partial:
        raise ValueError(f"{dataset_id} canonical bundle is partial: {partial}")
    if dataset_id == "bdd100k" and not allow_ineligible:
        smoke_bundle = Path(paths["bundles"]) / "bdd100k.kaggle_mirror.prepared.tar"
        if smoke_bundle.is_file() and _bundle_receipt_path(smoke_bundle).is_file():
            raise ValueError("BDD Kaggle bundle exists but is blocked from scientific staging")

    legacy_contracts = plan["storage"].get("legacy_compatibility", {})
    legacy = legacy_contracts.get(dataset_id)
    if not isinstance(legacy, dict):
        raise ValueError(f"missing prepared bundle for {dataset_id}")
    project = Path(paths["project"])
    legacy_bundle = project / str(legacy["bundle_directory"]) / str(legacy["bundle_filename"])
    legacy_receipt_path = (
        project / str(legacy["bundle_directory"]) / str(legacy["receipt_filename"])
    )
    if not legacy_bundle.is_file() or not legacy_receipt_path.is_file():
        raise ValueError(f"missing canonical and legacy prepared bundle for {dataset_id}")
    legacy_receipt = json.loads(legacy_receipt_path.read_text(encoding="utf-8"))
    expected = {
        "record_type": "cityscapes_training_bundle_receipt",
        "filename": legacy["bundle_filename"],
        "byte_size": int(legacy["byte_size"]),
        "sha256": legacy["bundle_sha256"],
        "dataset_manifest_sha256": legacy["dataset_manifest_sha256"],
        "split_manifest_sha256": legacy["split_manifest_sha256"],
        "file_count": int(legacy["file_count"]),
    }
    mismatches = [field for field, value in expected.items() if legacy_receipt.get(field) != value]
    if mismatches or legacy_bundle.stat().st_size != int(legacy["byte_size"]):
        raise ValueError(f"{dataset_id} legacy bundle identity mismatch: {mismatches}")
    normalized = {
        "dataset_id": dataset_id,
        "filename": legacy["bundle_filename"],
        "byte_size": int(legacy["byte_size"]),
        "source_bytes": int(legacy["source_bytes"]),
        "file_count": int(legacy["file_count"]),
        "sha256": legacy["bundle_sha256"],
        "required_paths": legacy["required_paths"],
        "legacy_dataset_manifest_sha256": legacy["dataset_manifest_sha256"],
        "legacy_split_manifest_sha256": legacy["split_manifest_sha256"],
    }
    return normalized, legacy_bundle, "legacy_cityscapes_v1"


def stage_dataset_bundles(
    plan: dict[str, Any],
    drive_root: Path,
    local_root: Path,
    dataset_ids: tuple[str, ...],
    *,
    allow_ineligible: bool = False,
) -> dict[str, Any]:
    """Copy, hash, extract, and validate selected bundles within the 200 GiB budget."""
    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("dataset_ids must be non-empty and unique")
    unknown = set(dataset_ids) - set(plan["datasets"])
    if unknown:
        raise ValueError(f"unknown dataset ids: {sorted(unknown)}")
    paths = initialize_drive_layout(plan, drive_root)
    shard_index = _load_idd_shard_index(Path(paths["project"]))
    if "idd20k" in dataset_ids and shard_index is not None:
        other_ids = tuple(value for value in dataset_ids if value != "idd20k")
        shard_results: list[dict[str, Any]] = []
        if other_ids:
            shard_results.extend(
                stage_dataset_bundles(
                    plan,
                    drive_root,
                    local_root,
                    other_ids,
                    allow_ineligible=allow_ineligible,
                )["datasets"]
            )
        shard_results.append(_stage_idd_shards(plan, Path(paths["project"]), local_root))
        return {
            "schema_version": "2.0",
            "record_type": "edgeguard_colab_dataset_staging",
            "datasets": shard_results,
            "local_root": str(local_root),
        }
    receipts: list[dict[str, Any]] = []
    drive_bundles: list[Path] = []
    bundle_profiles: list[str] = []
    for dataset_id in dataset_ids:
        receipt, bundle, bundle_profile = _canonical_bundle_receipt(
            plan, paths, dataset_id, allow_ineligible=allow_ineligible
        )
        receipts.append(receipt)
        drive_bundles.append(bundle)
        bundle_profiles.append(bundle_profile)
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
    for dataset_id, receipt, drive_bundle, bundle_profile in zip(
        dataset_ids, receipts, drive_bundles, bundle_profiles, strict=True
    ):
        destination = local_root / str(plan["datasets"][dataset_id]["prepared_subdirectory"])
        required = [str(value) for value in receipt["required_paths"]]
        if destination.is_dir() and all((destination / value).exists() for value in required):
            results.append({"dataset_id": dataset_id, "status": "already_staged"})
            continue
        if destination.exists():
            raise ValueError(f"partial local dataset destination exists: {destination}")
        local_bundle = cache / drive_bundle.name
        partial = local_bundle.with_suffix(local_bundle.suffix + ".part")
        partial.unlink(missing_ok=True)
        with drive_bundle.open("rb") as input_stream, partial.open("xb") as output_stream:
            progress = _HashingProgressReader(
                input_stream,
                label=dataset_id,
                phase="stage-bundle-copy",
            )
            shutil.copyfileobj(progress, output_stream, length=8 * 1024**2)
        if progress.hexdigest() != receipt["sha256"]:
            partial.unlink(missing_ok=True)
            raise ValueError(f"{dataset_id} local bundle SHA-256 mismatch")
        os.replace(partial, local_bundle)
        incoming = destination.with_name(f".{destination.name}.incoming")
        if incoming.exists():
            raise ValueError(f"stale extraction directory must be inspected: {incoming}")
        incoming.mkdir(parents=True)
        with tarfile.open(local_bundle, mode="r:*") as archive:
            members = _validate_tar_members(archive)
            if len(members) != int(receipt["file_count"]):
                raise ValueError(f"{dataset_id} bundle file count mismatch")
            for member_index, member in enumerate(members, start=1):
                if member_index == 1 or member_index % 1000 == 0 or member_index == len(members):
                    print(
                        canonical_json(
                            {
                                "phase": "stage-bundle-extract",
                                "dataset_id": dataset_id,
                                "completed": member_index,
                                "total": len(members),
                            }
                        ),
                        flush=True,
                    )
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
        results.append(
            {
                "dataset_id": dataset_id,
                "status": "staged_verified",
                "bundle_profile": bundle_profile,
            }
        )
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_staging_receipt",
        "datasets": results,
        "staged_bytes": staged_bytes,
        "planned_peak_bytes": peak_bytes,
        "configured_limit_bytes": configured_limit,
        "reserve_bytes": reserve,
    }
