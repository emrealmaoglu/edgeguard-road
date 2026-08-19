"""Exhaustive, dataset-name-agnostic inventory of raw files in a Drive
`private_inputs/`-style folder.

Every file is dispatched purely by *format* -- zip, tar(.gz), or a standalone file --
and, within an archive, every entry is classified purely by extension and (for images)
decoded PIL mode. Nothing here special-cases a dataset by name: a file this module has
never heard of gets exactly the same depth of inspection as Cityscapes or ACDC. Known
dataset metadata (declared role, ontology class count) is attached afterward, purely as
annotation on top of an already-fully-scanned result -- it never gates or shortens the
scan itself.

Image inspection is exhaustive (every image entry), not sampled: this is deliberately
more expensive than a bounded-sample design, chosen because the whole point of this
report is to support real downstream decisions (e.g. class-imbalance-driven ontology
changes), which a shortcut sample cannot honestly support. `inventory_identity()` lets a
caller skip a full re-scan when the folder's contents have not changed since the last
run (see `scripts/inventory_private_inputs.py`).
"""

from __future__ import annotations

import io
import tarfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from edgeguard.rescue.stall_guard import DEFAULT_STALL_TIMEOUT_SECONDS, stall_guard
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".webp"}
LABEL_LIKE_MODES = {"L", "P", "I", "I;16", "1"}
DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".yaml", ".yml", ".license", ".rst"}
SCRIPT_EXTENSIONS = {".py", ".sh"}
STRUCTURED_EXTENSIONS = {".json"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".ckpt", ".safetensors"}


def _format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _format_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN or negative
        return "hesaplanıyor"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}sa {minutes}dk"
    if minutes:
        return f"{minutes}dk {secs}sn"
    return f"{secs}sn"


def _print_entry_progress(filename: str, completed: int, total: int, elapsed: float) -> None:
    percent = (completed / total * 100) if total else 100.0
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = _format_duration((total - completed) / rate) if rate > 0 else "hesaplanıyor"
    print(
        f"    ... {filename}: {completed}/{total} giriş tarandı "
        f"(%{percent:.0f}) — tahmini kalan: {eta}",
        flush=True,
    )


def classify_entry(name: str) -> str:
    """Classify one archive member (or a standalone file) purely by extension."""
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in SCRIPT_EXTENSIONS:
        return "script"
    if suffix in STRUCTURED_EXTENSIONS:
        return "structured_data"
    if suffix in WEIGHT_EXTENSIONS:
        return "model_weights"
    return "other"


@dataclass
class PathInventory:
    """Fully scanned result for one file found directly under `private_inputs/`."""

    filename: str
    file_type: str  # "zip" | "tar" | "standalone"
    byte_size: int
    sha256: str
    entry_count: int
    uncompressed_total_bytes: int
    classification_counts: dict[str, int] = field(default_factory=dict)
    image_count: int = 0
    corrupt_count: int = 0
    corrupt_entries: list[dict[str, str]] = field(default_factory=list)
    resolution_histogram: dict[str, int] = field(default_factory=dict)
    format_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    label_like_image_count: int = 0
    class_pixel_histogram: dict[str, int] = field(default_factory=dict)
    class_image_histogram: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    known_role: dict[str, Any] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "entry_count": self.entry_count,
            "uncompressed_total_bytes": self.uncompressed_total_bytes,
            "classification_counts": dict(sorted(self.classification_counts.items())),
            "image_count": self.image_count,
            "corrupt_count": self.corrupt_count,
            "corrupt_entries": self.corrupt_entries,
            "resolution_histogram": dict(sorted(self.resolution_histogram.items())),
            "format_counts": dict(sorted(self.format_counts.items())),
            "mode_counts": dict(sorted(self.mode_counts.items())),
            "label_like_image_count": self.label_like_image_count,
            "class_pixel_histogram_measured": dict(sorted(self.class_pixel_histogram.items())),
            "class_image_histogram_measured": dict(sorted(self.class_image_histogram.items())),
            "elapsed_seconds": self.elapsed_seconds,
            "known_role_declared": self.known_role,
        }


@dataclass
class _ScanAccumulator:
    """Mutable counters filled in while walking one archive's entries."""

    classification_counts: Counter[str] = field(default_factory=Counter)
    resolution_histogram: Counter[str] = field(default_factory=Counter)
    format_counts: Counter[str] = field(default_factory=Counter)
    mode_counts: Counter[str] = field(default_factory=Counter)
    class_pixel_histogram: Counter[int] = field(default_factory=Counter)
    class_image_histogram: Counter[int] = field(default_factory=Counter)
    corrupt_entries: list[dict[str, str]] = field(default_factory=list)
    image_count: int = 0
    label_like_count: int = 0

    def record_image_bytes(self, data: bytes, *, entry_name: str) -> None:
        self.image_count += 1
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                self.resolution_histogram[f"{width}x{height}"] += 1
                self.format_counts[str(image.format)] += 1
                self.mode_counts[image.mode] += 1
                if image.mode not in LABEL_LIKE_MODES:
                    return
                self.label_like_count += 1
                values, counts = np.unique(np.asarray(image), return_counts=True)
                for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                    self.class_pixel_histogram[int(value)] += int(count)
                    self.class_image_histogram[int(value)] += 1
        except (OSError, UnidentifiedImageError, ValueError) as error:
            self.corrupt_entries.append({"name": entry_name, "error": str(error)})


def _inspect_zip(
    archive_path: Path, *, stall_timeout_seconds: int | None
) -> tuple[int, int, _ScanAccumulator]:
    accumulator = _ScanAccumulator()
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        entry_count = len(infos)
        uncompressed_total = sum(info.file_size for info in infos)
        started = time.perf_counter()
        last_print = started
        for entry_index, info in enumerate(infos, start=1):
            classification = classify_entry(info.filename)
            accumulator.classification_counts[classification] += 1
            if classification == "image":
                try:
                    with stall_guard(stall_timeout_seconds), archive.open(info) as stream:
                        data = stream.read()
                except OSError as error:
                    accumulator.image_count += 1
                    accumulator.corrupt_entries.append({"name": info.filename, "error": str(error)})
                else:
                    accumulator.record_image_bytes(data, entry_name=info.filename)
            now = time.perf_counter()
            if entry_index == 1 or entry_index == entry_count or now - last_print >= 2.0:
                last_print = now
                _print_entry_progress(archive_path.name, entry_index, entry_count, now - started)
    return entry_count, uncompressed_total, accumulator


def _tar_read_mode(archive_path: Path) -> Literal["r:gz", "r:bz2", "r:"]:
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        return "r:gz"
    if name.endswith(".tar.bz2"):
        return "r:bz2"
    return "r:"


def _inspect_tar(
    archive_path: Path, *, stall_timeout_seconds: int | None
) -> tuple[int, int, _ScanAccumulator]:
    accumulator = _ScanAccumulator()
    with tarfile.open(archive_path, _tar_read_mode(archive_path)) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        entry_count = len(members)
        uncompressed_total = sum(member.size for member in members)
        started = time.perf_counter()
        last_print = started
        for entry_index, member in enumerate(members, start=1):
            classification = classify_entry(member.name)
            accumulator.classification_counts[classification] += 1
            if classification == "image":
                try:
                    with stall_guard(stall_timeout_seconds):
                        extracted = archive.extractfile(member)
                        data = extracted.read() if extracted is not None else b""
                except OSError as error:
                    accumulator.image_count += 1
                    accumulator.corrupt_entries.append({"name": member.name, "error": str(error)})
                else:
                    accumulator.record_image_bytes(data, entry_name=member.name)
            now = time.perf_counter()
            if entry_index == 1 or entry_index == entry_count or now - last_print >= 2.0:
                last_print = now
                _print_entry_progress(archive_path.name, entry_index, entry_count, now - started)
    return entry_count, uncompressed_total, accumulator


def inspect_path(
    path: Path, *, stall_timeout_seconds: int | None = DEFAULT_STALL_TIMEOUT_SECONDS
) -> PathInventory:
    """Fully, exhaustively inspect one file -- archive or standalone -- by format only."""
    started = time.perf_counter()
    byte_size = path.stat().st_size
    digest = sha256_file(path)
    if zipfile.is_zipfile(path):
        file_type = "zip"
        entry_count, uncompressed_total, accumulator = _inspect_zip(
            path, stall_timeout_seconds=stall_timeout_seconds
        )
    elif tarfile.is_tarfile(path):
        file_type = "tar"
        entry_count, uncompressed_total, accumulator = _inspect_tar(
            path, stall_timeout_seconds=stall_timeout_seconds
        )
    else:
        file_type = "standalone"
        entry_count = 1
        uncompressed_total = byte_size
        accumulator = _ScanAccumulator()
        accumulator.classification_counts[classify_entry(path.name)] += 1
    return PathInventory(
        filename=path.name,
        file_type=file_type,
        byte_size=byte_size,
        sha256=digest,
        entry_count=entry_count,
        uncompressed_total_bytes=uncompressed_total,
        classification_counts=dict(accumulator.classification_counts),
        image_count=accumulator.image_count,
        corrupt_count=len(accumulator.corrupt_entries),
        corrupt_entries=accumulator.corrupt_entries,
        resolution_histogram=dict(accumulator.resolution_histogram),
        format_counts=dict(accumulator.format_counts),
        mode_counts=dict(accumulator.mode_counts),
        label_like_image_count=accumulator.label_like_count,
        class_pixel_histogram={str(k): v for k, v in accumulator.class_pixel_histogram.items()},
        class_image_histogram={str(k): v for k, v in accumulator.class_image_histogram.items()},
        elapsed_seconds=time.perf_counter() - started,
    )


def known_role_for_filename(filename: str, access_plan: dict[str, Any]) -> dict[str, Any] | None:
    """Annotate a filename with its declared role, if any. Never gates the scan."""
    for dataset_id, record in access_plan.get("datasets", {}).items():
        for kind in ("packages", "engineering_packages"):
            for package in record.get(kind, []):
                if str(package.get("filename")) == filename:
                    return {
                        "dataset_id": dataset_id,
                        "campaign_role": record.get("campaign_role"),
                        "package_kind": kind,
                        "purpose": package.get("purpose"),
                        "scientific_eligible": package.get("scientific_eligible", True),
                    }
    for excluded in access_plan.get("excluded_sources", []):
        source_id = str(excluded.get("source_id", ""))
        if source_id and source_id.replace("_", "").replace("-", "") in filename.replace(
            "_", ""
        ).replace("-", "").replace(".", ""):
            return {
                "dataset_id": None,
                "excluded": True,
                "reason": excluded.get("reason"),
            }
    return None


def inventory_identity(private_inputs_root: Path) -> str:
    """Hash the current file listing (names + sizes + mtimes) for cache-skip checks."""
    entries = []
    for path in sorted(private_inputs_root.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append({"name": path.name, "byte_size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return sha256_payload({"schema_version": "1.0", "entries": entries})


def build_inventory_report(
    private_inputs_root: Path,
    output_root: Path,
    *,
    access_plan_path: Path | None = None,
    stall_timeout_seconds: int | None = DEFAULT_STALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Exhaustively inventory every file in `private_inputs_root`; write JSON + Markdown."""
    if output_root.exists():
        raise FileExistsError(f"data-inventory output already exists: {output_root}")
    output_root.mkdir(parents=True)
    access_plan: dict[str, Any] = {}
    if access_plan_path is not None and access_plan_path.is_file():
        access_plan = yaml.safe_load(access_plan_path.read_text(encoding="utf-8")) or {}
    started = time.perf_counter()
    identity = inventory_identity(private_inputs_root)
    files = [path for path in sorted(private_inputs_root.iterdir()) if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"private_inputs taraması başlıyor: {len(files)} dosya, "
        f"toplam {_format_bytes(total_bytes)}",
        flush=True,
    )
    archives: list[dict[str, Any]] = []
    bytes_done = 0
    for file_index, path in enumerate(files, start=1):
        byte_size = path.stat().st_size
        print(
            f"[{file_index}/{len(files)}] taranıyor: {path.name} ({_format_bytes(byte_size)})",
            flush=True,
        )
        file_started = time.perf_counter()
        result = inspect_path(path, stall_timeout_seconds=stall_timeout_seconds)
        result.known_role = known_role_for_filename(path.name, access_plan)
        archives.append(result.to_jsonable())
        bytes_done += byte_size
        elapsed_so_far = time.perf_counter() - started
        rate = bytes_done / elapsed_so_far if elapsed_so_far > 0 else 0.0
        remaining_bytes = max(0, total_bytes - bytes_done)
        overall_eta = _format_duration(remaining_bytes / rate) if rate > 0 else "hesaplanıyor"
        print(
            f"[{file_index}/{len(files)}] tamamlandı: {path.name} — {result.file_type}, "
            f"{result.image_count} görüntü, {result.corrupt_count} bozuk "
            f"({time.perf_counter() - file_started:.1f}sn) "
            f"— genel tahmini kalan süre: {overall_eta}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0",
        "record_type": "raw_archive_inventory",
        "private_inputs_root": str(private_inputs_root),
        "inventory_identity_sha256": identity,
        "archive_count": len(archives),
        "total_byte_size": sum(int(item["byte_size"]) for item in archives),
        "total_image_count": sum(int(item["image_count"]) for item in archives),
        "total_corrupt_count": sum(int(item["corrupt_count"]) for item in archives),
        "elapsed_seconds": elapsed,
        "sampling": "exhaustive: every archive entry was inspected, none were sampled",
        "archives": archives,
    }
    (output_root / "dataset_inventory.json").write_text(canonical_json(report), encoding="utf-8")
    (output_root / "dataset_inventory.md").write_text(
        _render_markdown_report(report), encoding="utf-8"
    )
    print(
        f"private_inputs taraması tamamlandı: {report['archive_count']} dosya, "
        f"{report['total_image_count']} görüntü, {report['total_corrupt_count']} bozuk giriş, "
        f"toplam süre {_format_duration(elapsed)}",
        flush=True,
    )
    return report


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Raw archive inventory (`private_inputs/`)",
        "",
        f"- Archives inspected: {report['archive_count']}",
        f"- Total on-disk bytes: {report['total_byte_size']:,}",
        f"- Total images inspected: {report['total_image_count']:,}",
        f"- Total corrupt/unreadable entries: {report['total_corrupt_count']}",
        f"- Elapsed: {report['elapsed_seconds']:.1f}s",
        f"- Sampling: {report['sampling']}",
        "",
        "| File | Type | Bytes | Entries | Images | Corrupt | Declared role |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for archive in report["archives"]:
        role = archive.get("known_role_declared") or {}
        role_label = role.get("dataset_id") or ("excluded: " + str(role.get("reason", "")))[:40]
        lines.append(
            f"| {archive['filename']} | {archive['file_type']} | {archive['byte_size']:,} "
            f"| {archive['entry_count']:,} | {archive['image_count']:,} "
            f"| {archive['corrupt_count']} | {role_label or '—'} |"
        )
    lines.append("")
    for archive in report["archives"]:
        if not archive["class_pixel_histogram_measured"]:
            continue
        lines.append(f"## {archive['filename']} — measured pixel-value histogram")
        lines.append("")
        lines.append("| Pixel value | Pixel count | Image count |")
        lines.append("| --- | --- | --- |")
        for value, count in sorted(
            archive["class_pixel_histogram_measured"].items(), key=lambda item: -item[1]
        ):
            image_count = archive["class_image_histogram_measured"].get(value, 0)
            lines.append(f"| {value} | {count:,} | {image_count:,} |")
        lines.append("")
    return "\n".join(lines) + "\n"
