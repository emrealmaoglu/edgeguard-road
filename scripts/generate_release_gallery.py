"""Generate a measured accepted-model gallery from official source validation samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.rescue.inference import predict_onnx
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.rescue.visualization import colorize_mask, overlay_mask
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

SMALL_CLASSES = {6, 7, 11, 12, 17, 18}


def _colorize_ground_truth(target: np.ndarray) -> Image.Image:
    """Render ignored pixels neutrally instead of mislabeling them as road."""
    rendered = np.asarray(colorize_mask(np.where(target == 255, 0, target))).copy()
    rendered[target == 255] = (96, 96, 96)
    return Image.fromarray(rendered)


def _sample_paths(manifest: dict[str, Any], record: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(str(manifest["dataset_root"]))
    prepared = Path(str(manifest["prepared_root"]))
    image = root / str(record["image"])
    canonical = str(record["canonical_mask"])
    mask = (prepared if manifest["dataset_id"] == "idd20k" else root) / canonical
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError("gallery source image or canonical mask is missing")
    return image, mask


def _frame_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    confusion = SemanticConfusionMatrix()
    confusion.update(prediction.astype(np.int64), target.astype(np.int64))
    result = confusion.result()
    valid = (target >= 0) & (target < 19)
    horizontal = np.count_nonzero(valid[:, 1:] & (target[:, 1:] != target[:, :-1]))
    vertical = np.count_nonzero(valid[1:, :] & (target[1:, :] != target[:-1, :]))
    small = np.count_nonzero(np.isin(target, tuple(SMALL_CLASSES)))
    return {
        "mIoU": float(result["mean_iou"] or 0.0),
        "pixel_accuracy": float(result["pixel_accuracy"] or 0.0),
        "boundary_density": float((horizontal + vertical) / max(1, np.count_nonzero(valid))),
        "small_class_pixel_ratio": float(small / max(1, np.count_nonzero(valid))),
    }


def generate_gallery(
    *,
    accepted_release: Path,
    work_root: Path,
    manifests: tuple[Path, ...],
    output_root: Path,
) -> dict[str, Any]:
    """Select best/worst/small-object/boundary examples without inventing labels."""
    release = json.loads(accepted_release.read_text(encoding="utf-8"))
    release_id = str(release["release_id"])
    model = str(release["recommended_model"])
    onnx = work_root / "exports/export" / release_id / model / f"{model}.onnx"
    identity = {
        "accepted_release_sha256": sha256_file(accepted_release),
        "onnx_sha256": sha256_file(onnx),
        "manifest_sha256s": [sha256_file(path) for path in manifests],
    }
    completion = output_root / "gallery_manifest.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("input_identity") == identity:
            for artifact in payload.get("artifacts", []):
                path = output_root / str(artifact["path"])
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    raise ValueError("existing gallery artifact identity mismatch")
            return payload
        raise FileExistsError("gallery output exists with another identity")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("incomplete gallery output requires review")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        ort = __import__("onnxruntime")
    except ModuleNotFoundError as error:
        raise RuntimeError("gallery generation requires the locked ONNX runtime") from error
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    selected_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    for manifest_path in manifests:
        manifest = validate_dataset_manifest(manifest_path, allowed_roles=("official_source_val",))
        dataset = str(manifest["dataset_id"])
        records = manifest["roles"]["official_source_val"]
        if not isinstance(records, list) or not records:
            raise ValueError("gallery manifest has no official validation records")
        count = min(24, len(records))
        indices = sorted({round(value) for value in np.linspace(0, len(records) - 1, count)})
        measured: list[dict[str, Any]] = []
        for index in indices:
            record = records[index]
            image_path, mask_path = _sample_paths(manifest, record)
            image = Image.open(image_path).convert("RGB")
            target = np.asarray(Image.open(mask_path), dtype=np.uint8)
            if target.shape != (image.height, image.width):
                target = np.asarray(
                    Image.fromarray(target).resize(image.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            result = predict_onnx(image, onnx, session=session)
            measured.append(
                {
                    "record": record,
                    "image": image,
                    "target": target,
                    "prediction": result.mask,
                    "metrics": _frame_metrics(result.mask, target),
                    "latency_ms": result.latency_ms,
                }
            )
        choices = (
            ("highest_miou", max(measured, key=lambda row: row["metrics"]["mIoU"])),
            ("lowest_miou", min(measured, key=lambda row: row["metrics"]["mIoU"])),
            (
                "small_object_challenge",
                max(measured, key=lambda row: row["metrics"]["small_class_pixel_ratio"]),
            ),
            (
                "boundary_challenge",
                max(measured, key=lambda row: row["metrics"]["boundary_density"]),
            ),
        )
        for category, row in choices:
            root = output_root / dataset / category
            root.mkdir(parents=True, exist_ok=True)
            image = row["image"]
            image.save(root / "input.png")
            _colorize_ground_truth(row["target"]).save(root / "ground_truth.png")
            colorize_mask(row["prediction"]).save(root / "prediction.png")
            overlay_mask(image, row["prediction"], 0.55).save(root / "overlay.png")
            artifact_paths.extend(sorted(root.glob("*.png")))
            selected_rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "sample_id": row["record"].get("sample_id"),
                    **row["metrics"],
                    "onnxruntime_cpu_latency_ms": row["latency_ms"],
                    "scientific_status": "accepted",
                }
            )
    table = output_root / "gallery_metrics.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    artifact_paths.append(table)
    artifacts = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in sorted(artifact_paths)
    ]
    payload = {
        "schema_version": "1.0",
        "record_type": "edgeguard_measured_failure_gallery",
        "release_id": release_id,
        "recommended_model": model,
        "input_identity": identity,
        "selection_note": (
            "Categories are measured extrema within a deterministic 24-sample subset per "
            "source; lowest_miou is a failure-analysis example, not a population estimate."
        ),
        "scientific_status": "accepted",
        "rows": selected_rows,
        "artifacts": artifacts,
        "gallery_identity": sha256_payload(artifacts),
    }
    completion.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-release", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate_gallery(
        accepted_release=args.accepted_release.resolve(),
        work_root=args.work_root.resolve(),
        manifests=tuple(path.resolve() for path in args.evaluation_manifest),
        output_root=args.output_root.resolve(),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
