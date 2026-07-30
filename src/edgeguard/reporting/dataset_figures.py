"""Measured-only dataset tables and thesis-ready multi-domain figures."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.serialization import canonical_json, sha256_file


def _canonical_path(manifest: dict[str, Any], record: dict[str, Any]) -> Path:
    root = (
        Path(manifest["prepared_root"])
        if manifest["dataset_id"] == "idd20k"
        else Path(manifest["dataset_root"])
    )
    return root / str(record["canonical_mask"])


def _save_figure(figure: Any, root: Path, stem: str) -> list[Path]:
    outputs = [root / f"{stem}.png", root / f"{stem}.pdf"]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    return outputs


def build_dataset_figures(
    manifests: list[dict[str, Any]],
    *,
    per_domain_counts: dict[str, list[int]],
    pooled_counts: list[int],
    weights: list[float],
    rare_ids: list[int],
    output_root: Path,
) -> dict[str, Any]:
    """Create CSV/PNG/PDF evidence directly from frozen manifest measurements."""
    try:
        pyplot = __import__("matplotlib.pyplot", fromlist=["subplots", "close"])
    except ModuleNotFoundError:
        result = {
            "schema_version": "1.0",
            "record_type": "edgeguard_dataset_figure_report",
            "figures_generated": False,
            "reason": "matplotlib_not_installed",
        }
        (output_root / "thesis_figures.json").write_text(
            canonical_json(result) + "\n", encoding="utf-8"
        )
        return result
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    pooled = np.asarray(pooled_counts, dtype=np.float64)
    pooled_ratio = pooled / pooled.sum()
    rows: list[dict[str, Any]] = []
    for dataset_id, counts in sorted(per_domain_counts.items()):
        values = np.asarray(counts, dtype=np.float64)
        ratios = values / values.sum()
        for class_id, (count, ratio) in enumerate(zip(values, ratios, strict=True)):
            rows.append(
                {
                    "dataset": dataset_id,
                    "class_id": class_id,
                    "class_name": CITYSCAPES_CLASSES[class_id],
                    "pixel_count": int(count),
                    "pixel_ratio": float(ratio),
                    "pooled_pixel_ratio": float(pooled_ratio[class_id]),
                    "median_frequency_weight": float(weights[class_id]),
                    "rare_class": class_id in rare_ids,
                }
            )
    csv_path = output_root / "class_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    created: list[Path] = [csv_path]
    positions = np.arange(19)
    figure, axis = pyplot.subplots(figsize=(14, 6))
    for dataset_id, counts in sorted(per_domain_counts.items()):
        values = np.asarray(counts, dtype=np.float64)
        axis.plot(
            positions,
            np.maximum(values / values.sum(), 1.0e-12),
            marker="o",
            linewidth=1.5,
            label=dataset_id,
        )
    axis.set_yscale("log")
    axis.set_ylabel("train_fit pixel ratio (log scale)")
    axis.set_xticks(positions, CITYSCAPES_CLASSES, rotation=65, ha="right")
    axis.set_title("Canonical class distribution by source domain")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    created.extend(_save_figure(figure, figure_root, "class_distribution_by_domain"))
    pyplot.close(figure)

    rare_mask = np.asarray([index in rare_ids for index in range(19)])
    figure, (frequency_axis, weight_axis) = pyplot.subplots(2, 1, figsize=(14, 9))
    colors = ["tab:red" if rare else "tab:blue" for rare in rare_mask]
    frequency_axis.bar(positions, pooled_ratio * 100.0, color=colors)
    frequency_axis.set_yscale("log")
    frequency_axis.set_ylabel("pooled train_fit pixels (%)")
    frequency_axis.set_title("Pooled class imbalance; rare-five classes highlighted")
    weight_axis.bar(positions, weights, color=colors)
    weight_axis.axhline(1.0, color="black", linewidth=1.0)
    weight_axis.set_ylabel("weighted-CE multiplier")
    weight_axis.set_xticks(positions, CITYSCAPES_CLASSES, rotation=65, ha="right")
    figure.tight_layout()
    created.extend(_save_figure(figure, figure_root, "pooled_imbalance_and_weights"))
    pyplot.close(figure)

    role_names = ("train_fit", "train_select", "train_calibration")
    dataset_ids = [str(manifest["dataset_id"]) for manifest in manifests]
    figure, axis = pyplot.subplots(figsize=(10, 5))
    width = 0.24
    base = np.arange(len(dataset_ids))
    for index, role in enumerate(role_names):
        axis.bar(
            base + (index - 1) * width,
            [len(manifest["roles"].get(role, [])) for manifest in manifests],
            width=width,
            label=role,
        )
    axis.set_xticks(base, dataset_ids)
    axis.set_ylabel("image count")
    axis.set_title("Frozen group-atomic source roles")
    axis.legend()
    figure.tight_layout()
    created.extend(_save_figure(figure, figure_root, "source_split_sizes"))
    pyplot.close(figure)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for manifest in manifests:
        records = list(manifest["roles"].get("train_fit", []))
        records.sort(key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest())
        if records:
            selected.append((manifest, records[0]))
    if selected:
        figure, axes = pyplot.subplots(len(selected), 2, figsize=(12, 4 * len(selected)))
        axes_array = np.asarray(axes, dtype=object).reshape(len(selected), 2)
        color_map = pyplot.get_cmap("tab20", 19)
        for row_index, (manifest, record) in enumerate(selected):
            with Image.open(Path(manifest["dataset_root"]) / str(record["image"])) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            with Image.open(_canonical_path(manifest, record)) as opened:
                mask = np.asarray(opened, dtype=np.uint8)
            axes_array[row_index, 0].imshow(rgb)
            axes_array[row_index, 0].set_title(f"{manifest['dataset_id']} · {record['sample_id']}")
            visible = np.ma.masked_where(mask == 255, mask)
            axes_array[row_index, 1].imshow(rgb)
            axes_array[row_index, 1].imshow(visible, cmap=color_map, vmin=0, vmax=18, alpha=0.55)
            axes_array[row_index, 1].set_title("Cityscapes19 canonical overlay")
            for column in range(2):
                axes_array[row_index, column].axis("off")
        figure.tight_layout()
        created.extend(_save_figure(figure, figure_root, "source_domain_examples"))
        pyplot.close(figure)

    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_dataset_figure_report",
        "figures_generated": True,
        "source_role": "train_fit",
        "dataset_manifest_sha256s": [str(row["manifest_sha256"]) for row in manifests],
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(created)
        ],
        "sample_images_for_thesis_only": True,
        "dataset_redistribution_authorized": False,
    }
    (output_root / "thesis_figures.json").write_text(
        canonical_json(result) + "\n", encoding="utf-8"
    )
    return result
