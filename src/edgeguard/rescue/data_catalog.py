"""Verified semantic-source catalog, bounded probes, and phase-two RGB mapping."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import yaml
from PIL import Image

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

ACTIVE_DATASET_IDS = {
    "cityscapes",
    "bdd100k",
    "idd20k",
    "acdc",
    "wilddash2",
    "muses",
    "kitti",
    "mapillary_vistas",
    "a2d2",
}
SEALED_DATASET_IDS = {"wilddash2", "muses"}
_HEX_COLOR = re.compile(r"^#[0-9a-f]{6}$")


def _verified_ssl_context() -> ssl.SSLContext:
    """Use an explicit Mozilla CA bundle across macOS, Colab, and Linux."""
    return ssl.create_default_context(cafile=certifi.where())


def load_dataset_catalog(path: Path) -> dict[str, Any]:
    """Load the authoritative catalog and reject unsafe or incomplete records."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("dataset catalog is missing or malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "2.0"
        or payload.get("record_type") != "edgeguard_semantic_dataset_catalog"
    ):
        raise ValueError("dataset catalog must use the semantic schema 2.0 contract")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset catalog has no records")
    identities: set[str] = set()
    for record in datasets:
        if not isinstance(record, dict):
            raise ValueError("dataset catalog records must be mappings")
        dataset_id = record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset catalog record has no dataset_id")
        if dataset_id in identities:
            raise ValueError(f"duplicate dataset_id: {dataset_id}")
        identities.add(dataset_id)
        for field in (
            "name",
            "portfolio_role",
            "official_source",
            "access",
            "license",
            "statistics",
            "label_contract",
            "merge_contract",
            "allowed_stages",
            "prohibited_stages",
        ):
            if field not in record:
                raise ValueError(f"{dataset_id} is missing catalog field {field}")
        if not str(record["official_source"]).startswith("https://"):
            raise ValueError(f"{dataset_id} official source must use HTTPS")
        if not isinstance(record["allowed_stages"], list) or not isinstance(
            record["prohibited_stages"], list
        ):
            raise ValueError(f"{dataset_id} stage policies must be lists")
        access = record["access"]
        if not isinstance(access, dict) or access.get("mode") not in {
            "registered_manual",
            "account_manual",
            "public_direct",
            "public_server_submission",
        }:
            raise ValueError(f"{dataset_id} access mode is unsupported")
        sample_bundle = record.get("sample_bundle")
        if dataset_id in SEALED_DATASET_IDS:
            if access.get("sealed") is not True or sample_bundle is not None:
                raise ValueError(f"sealed dataset {dataset_id} cannot expose sample downloads")
        if sample_bundle is not None:
            _validate_sample_bundle(dataset_id, sample_bundle, access)
    missing = ACTIVE_DATASET_IDS - identities
    if missing:
        raise ValueError(f"active semantic portfolio is incomplete: {sorted(missing)}")
    return payload


def _validate_sample_bundle(dataset_id: str, sample_bundle: Any, access: dict[str, Any]) -> None:
    if access.get("mode") != "public_direct" or access.get("sealed") is True:
        raise ValueError(f"{dataset_id} sample bundle is not public and unsealed")
    if not isinstance(sample_bundle, dict) or sample_bundle.get("usage_scope") != (
        "engineering_probe_only"
    ):
        raise ValueError(f"{dataset_id} sample bundle must be engineering-only")
    files = sample_bundle.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{dataset_id} sample bundle has no files")
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_id} sample file must be a mapping")
        filename = item.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename or filename in names:
            raise ValueError(f"{dataset_id} sample filenames must be unique and flat")
        names.add(filename)
        if not str(item.get("url", "")).startswith("https://"):
            raise ValueError(f"{dataset_id} sample URL must use HTTPS")
        size = item.get("byte_size")
        digest = item.get("sha256")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"{dataset_id} sample byte size must be positive")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{dataset_id} sample SHA-256 is invalid")


def render_catalog_markdown(catalog: dict[str, Any]) -> str:
    """Render a concise human review view from the machine-readable catalog."""
    lines = [
        "# EdgeGuard semantic dataset catalog",
        "",
        (
            "`catalog.json` is authoritative. Counts describe official package contracts, "
            "not locally acquired evidence. `verified_local` remains false until a "
            "hash-bound receipt exists."
        ),
        "",
        "| Dataset | Portfolio role | Official count | Native labels | Canonical merge | Access |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in catalog["datasets"]:
        statistics = row["statistics"]
        labels = row["label_contract"]
        merge = row["merge_contract"]
        count = statistics.get("annotated_images")
        count_text = f"{count:,}" if isinstance(count, int) else "server-defined"
        lines.append(
            "| {name} | `{role}` | {count} | {labels} | `{merge}` | `{access}` |".format(
                name=row["name"],
                role=row["portfolio_role"],
                count=count_text,
                labels=labels["native_summary"],
                merge=merge["status"],
                access=row["access"]["mode"],
            )
        )
    lines.extend(
        [
            "",
            "## Non-negotiable merge rules",
            "",
            "- Native masks are preserved; generated Cityscapes19 masks are separate artifacts.",
            "- Unknown or semantically ambiguous labels become `255`, never background.",
            (
                "- Source domains are sampled uniformly in the primary experiment; physical "
                "file concatenation is not the sampling policy."
            ),
            (
                "- Official validation, adverse-domain, and sealed external records never "
                "enter training, calibration, preprocessing fitting, HPO, or debugging."
            ),
            (
                "- Dataset version, license receipt, source hash, mapping hash, split hash, "
                "exact hash, and perceptual-hash evidence are required before scientific use."
            ),
        ]
    )
    probes = catalog.get("integration_probes", [])
    if probes:
        lines.extend(("", "## Engineering probes", ""))
        for probe in probes:
            lines.append(f"- `{probe['probe_id']}`: {probe['status']}; {probe['scientific_role']}.")
    return "\n".join(lines) + "\n"


def probe_official_sources(
    catalog: dict[str, Any], *, timeout_seconds: float = 15.0
) -> list[dict[str, Any]]:
    """Issue bounded HEAD requests to official landing pages without acquiring data."""
    results: list[dict[str, Any]] = []
    for record in catalog["datasets"]:
        request = urllib.request.Request(
            str(record["official_source"]),
            method="HEAD",
            headers={"User-Agent": "EdgeGuard-Road-source-probe/1.0"},
        )
        result: dict[str, Any] = {
            "dataset_id": record["dataset_id"],
            "url": record["official_source"],
            "reachable": False,
        }
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds, context=_verified_ssl_context()
            ) as response:
                result.update(
                    {
                        "reachable": 200 <= int(response.status) < 400,
                        "http_status": int(response.status),
                        "resolved_url": response.url,
                    }
                )
        except urllib.error.HTTPError as error:
            result.update({"http_status": int(error.code), "error": "HTTPError"})
        except (OSError, urllib.error.URLError) as error:
            result.update({"error": type(error).__name__})
        results.append(result)
    return results


def download_public_sample_bundle(
    catalog: dict[str, Any],
    dataset_id: str,
    output_root: Path,
    *,
    maximum_total_bytes: int = 8 * 1024**2,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Download one catalog-pinned public sample bundle with strict size/hash gates."""
    records = {str(row["dataset_id"]): row for row in catalog["datasets"]}
    if dataset_id not in records:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    record = records[dataset_id]
    bundle = record.get("sample_bundle")
    if bundle is None:
        raise PermissionError(f"{dataset_id} has no approved public engineering sample bundle")
    _validate_sample_bundle(dataset_id, bundle, record["access"])
    total = sum(int(item["byte_size"]) for item in bundle["files"])
    if total > maximum_total_bytes:
        raise ValueError("sample bundle exceeds the caller's maximum byte budget")
    output_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for item in bundle["files"]:
        destination = output_root / str(item["filename"])
        if not destination.is_file() or (
            destination.stat().st_size != int(item["byte_size"])
            or sha256_file(destination) != item["sha256"]
        ):
            _download_verified_file(item, destination, timeout_seconds=timeout_seconds)
        files.append(
            {
                "filename": destination.name,
                "byte_size": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "role": item["role"],
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_public_dataset_probe_receipt",
        "dataset_id": dataset_id,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "license_id": record["license"]["id"],
        "usage_scope": "engineering_probe_only",
        "scientific_evidence": False,
        "files": files,
    }
    if dataset_id == "a2d2":
        mapping_path = Path(__file__).parents[3] / str(bundle["mapping_file"])
        receipt["inspection"] = inspect_a2d2_bundle(output_root, mapping_path)
    receipt["receipt_sha256"] = sha256_payload(receipt)
    (output_root / "probe_receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


def _download_verified_file(
    item: dict[str, Any], destination: Path, *, timeout_seconds: float
) -> None:
    partial = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(
        str(item["url"]), headers={"User-Agent": "EdgeGuard-Road-sample-probe/1.0"}
    )
    received = 0
    try:
        with (
            urllib.request.urlopen(
                request, timeout=timeout_seconds, context=_verified_ssl_context()
            ) as response,
            partial.open("wb") as stream,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != int(item["byte_size"]):
                raise ValueError("public sample Content-Length differs from the catalog")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > int(item["byte_size"]):
                    raise ValueError("public sample exceeded its catalog byte limit")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if received != int(item["byte_size"]) or sha256_file(partial) != item["sha256"]:
            raise ValueError("public sample size or SHA-256 mismatch")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def load_a2d2_mapping(path: Path) -> dict[str, Any]:
    """Load the complete 55-color A2D2 proposal and validate every disposition."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("A2D2 mapping is missing or malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0"
        or payload.get("source_dataset") != "a2d2"
        or payload.get("unknown_color_policy") != "reject"
    ):
        raise ValueError("invalid A2D2 mapping contract")
    rows = payload.get("classes")
    if not isinstance(rows, list) or len(rows) != 55:
        raise ValueError("A2D2 mapping must classify all 55 official colors")
    colors: set[str] = set()
    for row in rows:
        color = str(row.get("color", ""))
        target = row.get("target_id")
        if not _HEX_COLOR.fullmatch(color) or color in colors:
            raise ValueError("A2D2 mapping colors must be unique lowercase hex values")
        colors.add(color)
        if target is None:
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                raise ValueError(f"ignored A2D2 color {color} needs a reason")
        elif not isinstance(target, int) or target not in range(19):
            raise ValueError(f"A2D2 color {color} has an invalid Cityscapes19 target")
    return payload


def map_a2d2_rgb_mask(mask: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    """Map an A2D2 RGB mask to Cityscapes19 and reject undeclared colors."""
    if mask.ndim != 3 or mask.shape[2] != 3 or mask.dtype != np.uint8:
        raise ValueError("A2D2 semantic masks must be uint8 RGB arrays")
    lut: dict[int, int] = {}
    for row in mapping["classes"]:
        color = str(row["color"])
        packed = int(color[1:], 16)
        target = row.get("target_id")
        lut[packed] = 255 if target is None else int(target)
    packed_mask = (
        mask[:, :, 0].astype(np.uint32) << 16
        | mask[:, :, 1].astype(np.uint32) << 8
        | mask[:, :, 2].astype(np.uint32)
    )
    unknown = sorted(int(value) for value in np.unique(packed_mask) if int(value) not in lut)
    if unknown:
        colors = [f"#{value:06x}" for value in unknown]
        raise ValueError(f"A2D2 mask contains undeclared RGB colors: {colors}")
    result = np.full(packed_mask.shape, 255, dtype=np.uint8)
    for source_color, target_id in lut.items():
        result[packed_mask == source_color] = target_id
    return result


def inspect_a2d2_bundle(output_root: Path, mapping_path: Path) -> dict[str, Any]:
    """Inspect geometry, native-color presence, and canonical usable-pixel coverage."""
    mapping = load_a2d2_mapping(mapping_path)
    class_file = json.loads((output_root / "class_list.json").read_text(encoding="utf-8"))
    official = {str(row["color"]): str(row["name"]) for row in mapping["classes"]}
    if class_file != official:
        raise ValueError("downloaded A2D2 class list differs from the reviewed mapping")
    with Image.open(output_root / "image.png") as image:
        image.load()
        image_size = image.size
        image_mode = image.mode
    with Image.open(output_root / "label.png") as label_image:
        label = np.asarray(label_image.convert("RGB"), dtype=np.uint8)
        label_size = label_image.size
    if image_size != label_size:
        raise ValueError("A2D2 public sample image/mask geometry mismatch")
    canonical = map_a2d2_rgb_mask(label, mapping)
    colors, counts = np.unique(label.reshape(-1, 3), axis=0, return_counts=True)
    source_counts = {
        f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}": int(count)
        for color, count in zip(colors, counts, strict=True)
    }
    canonical_counts = Counter(int(value) for value in canonical.ravel())
    return {
        "geometry": [image_size[1], image_size[0]],
        "image_mode": image_mode,
        "mask_mode": "RGB",
        "official_color_count": len(class_file),
        "sample_unique_color_count": len(source_counts),
        "sample_source_pixel_counts": source_counts,
        "canonical_pixel_counts": {
            str(key): value for key, value in sorted(canonical_counts.items())
        },
        "usable_pixel_ratio": float(np.mean(canonical != 255)),
        "mapping_sha256": sha256_file(mapping_path),
    }
