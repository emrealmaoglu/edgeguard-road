"""Deterministic, data-backed thesis figure factory without placeholder metrics."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, ImageDraw

from edgeguard.campaign.state import Campaign
from edgeguard.serialization import sha256_file, sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json

FIGURE_LABEL = "NON-SCIENTIFIC PIPELINE VALIDATION"


def _svg_line_chart(title: str, series: dict[str, list[float]]) -> str:
    width, height = 900, 500
    all_values = [value for values in series.values() for value in values]
    maximum = max(all_values, default=1.0)
    minimum = min(all_values, default=0.0)
    span = max(maximum - minimum, 1e-9)
    colors = ("#54a6ff", "#f5be3c", "#60d394", "#ff6b6b", "#b48ef7")
    elements = [
        f'<rect width="{width}" height="{height}" fill="#141c26"/>',
        f'<text x="30" y="35" fill="white" font-size="22">{title}</text>',
        f'<text x="30" y="65" fill="#f5be3c" font-size="15">{FIGURE_LABEL}</text>',
    ]
    for index, (name, values) in enumerate(sorted(series.items())):
        points = []
        for step, value in enumerate(values):
            x = 70 + step * 750 / max(len(values) - 1, 1)
            y = 430 - (value - minimum) / span * 320
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[index % len(colors)]
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="700" y="{95 + index * 22}" fill="{color}" font-size="14">{name}</text>'
        )
    body = "".join(elements)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{body}</svg>'


def _svg_bar_chart(title: str, labels: list[str], values: list[float]) -> str:
    width, height = 900, 500
    maximum = max(values, default=1.0) or 1.0
    elements = [
        f'<rect width="{width}" height="{height}" fill="#141c26"/>',
        f'<text x="30" y="35" fill="white" font-size="22">{title}</text>',
        f'<text x="30" y="65" fill="#f5be3c" font-size="15">{FIGURE_LABEL}</text>',
    ]
    bar_width = 760 / max(len(values), 1)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_height = value / maximum * 300
        x = 70 + index * bar_width
        y = 420 - bar_height
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.7:.2f}" '
            f'height="{bar_height:.2f}" fill="#54a6ff"/>'
        )
        elements.append(f'<text x="{x:.2f}" y="445" fill="white" font-size="11">{label}</text>')
    body = "".join(elements)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{body}</svg>'


def _svg_flow(title: str, labels: list[str]) -> str:
    width, height = 1000, 300
    elements = [
        f'<rect width="{width}" height="{height}" fill="#141c26"/>',
        f'<text x="30" y="35" fill="white" font-size="22">{title}</text>',
        f'<text x="30" y="65" fill="#f5be3c" font-size="15">{FIGURE_LABEL}</text>',
    ]
    box_width = 850 / max(len(labels), 1)
    for index, label in enumerate(labels):
        x = 40 + index * box_width
        elements.append(
            f'<rect x="{x:.1f}" y="115" width="{box_width * 0.82:.1f}" height="70" '
            'rx="8" fill="#243447" stroke="#54a6ff"/>'
        )
        elements.append(f'<text x="{x + 8:.1f}" y="154" fill="white" font-size="12">{label}</text>')
        if index < len(labels) - 1:
            elements.append(
                f'<line x1="{x + box_width * 0.82:.1f}" y1="150" '
                f'x2="{x + box_width:.1f}" y2="150" stroke="#f5be3c" stroke-width="3"/>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}">{"".join(elements)}</svg>'
    )


def _svg_to_png_fallback(svg: str, path: Path, title: str) -> None:
    """Write a deterministic raster companion without requiring an SVG renderer."""
    del svg
    image = Image.new("RGB", (900, 500), (20, 28, 38))
    draw = ImageDraw.Draw(image)
    draw.text((30, 35), title, fill="white")
    draw.text((30, 65), FIGURE_LABEL, fill=(245, 190, 60))
    draw.text(
        (30, 110), "See SVG and CSV/JSON for the complete plotted data.", fill=(145, 200, 255)
    )
    image.save(path, format="PNG", optimize=True)


def _write_figure(
    root: Path,
    *,
    figure_id: str,
    title: str,
    svg: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    caption_tr: str,
    caption_en: str,
) -> dict[str, Any]:
    directory = root / figure_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / f"{figure_id}.svg").write_text(svg, encoding="utf-8")
    _svg_to_png_fallback(svg, directory / f"{figure_id}.png", title)
    atomic_write_json(directory / "data.json", {"rows": rows})
    if rows:
        fields = sorted({key for row in rows for key in row})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        (directory / "data.csv").write_text(buffer.getvalue(), encoding="utf-8")
    atomic_write_json(directory / "provenance.json", provenance)
    (directory / "caption_tr.txt").write_text(caption_tr + "\n", encoding="utf-8")
    (directory / "caption_en.txt").write_text(caption_en + "\n", encoding="utf-8")
    return {
        "figure_id": figure_id,
        "title": title,
        "status": "generated",
        "formats": ["png", "svg", "json", "csv"],
        "classification": FIGURE_LABEL,
        "data_sha256": sha256_payload(rows),
    }


def _zip(source: Path, destination: Path) -> None:
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = ZipInfo(path.relative_to(source).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def generate_thesis_figures(
    campaign: Campaign, *, receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Generate only figures supported by promoted campaign receipts."""
    manifest = campaign.load_manifest()
    build = campaign.root / "reports" / ".thesis-figure-build"
    if build.exists():
        shutil = __import__("shutil")
        shutil.rmtree(build)
    build.mkdir(parents=True)
    by_stage = {receipt["stage_id"]: receipt for receipt in receipts}
    provenance = {
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git_commit"],
        "profile": manifest["profile"],
        "scientific_evidence": False,
    }
    figures: list[dict[str, Any]] = []
    architecture_labels = ["data", "semantic", "OOD", "detection", "risk", "temporal", "export"]
    figures.append(
        _write_figure(
            build,
            figure_id="EG-FIG-ARCH-001",
            title="EdgeGuard-Road local validation architecture",
            svg=_svg_flow("EdgeGuard-Road local validation architecture", architecture_labels),
            rows=[
                {"position": index, "component": label}
                for index, label in enumerate(architecture_labels)
            ],
            provenance=provenance,
            caption_tr="EdgeGuard-Road sentetik yerel doğrulama bileşenleri.",
            caption_en="EdgeGuard-Road synthetic local validation components.",
        )
    )
    pipeline_labels = ["prepare", "train", "score", "calibrate", "fuse", "report"]
    figures.append(
        _write_figure(
            build,
            figure_id="EG-FIG-PIPE-001",
            title="Campaign experiment pipeline",
            svg=_svg_flow("Campaign experiment pipeline", pipeline_labels),
            rows=[
                {"position": index, "stage": label} for index, label in enumerate(pipeline_labels)
            ],
            provenance=provenance,
            caption_tr="Kampanya deney hattının sentetik yürütme sırası.",
            caption_en="Synthetic execution order of the campaign experiment pipeline.",
        )
    )
    if "semantic_smoke" in by_stage:
        models = by_stage["semantic_smoke"].get("models", [])
        series = {model["model_family"]: model["train_losses"] for model in models}
        rows = [
            {"model": name, "step": step, "loss": value}
            for name, values in sorted(series.items())
            for step, value in enumerate(values, start=1)
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-TRAIN-001",
                title="Local-mini training curves",
                svg=_svg_line_chart("Local-mini training curves", series),
                rows=rows,
                provenance=provenance,
                caption_tr=(
                    "Beş aday için sentetik local-mini eğitim kaybı; bilimsel sonuç değildir."
                ),
                caption_en=(
                    "Synthetic local-mini training loss for five candidates; "
                    "not a scientific result."
                ),
            )
        )
        if models:
            ious = models[0]["validation"]["per_class_iou"]
            values = [float(value or 0.0) for value in ious]
            rows = [{"class_id": index, "iou": ious[index]} for index in range(len(ious))]
            figures.append(
                _write_figure(
                    build,
                    figure_id="EG-FIG-IOU-001",
                    title="Synthetic per-class IoU plumbing",
                    svg=_svg_bar_chart(
                        "Synthetic per-class IoU plumbing", [str(i) for i in range(19)], values
                    ),
                    rows=rows,
                    provenance=provenance,
                    caption_tr="Sentetik veri üzerinde sınıf başına IoU raporlama hattı.",
                    caption_en="Per-class IoU reporting path on synthetic data.",
                )
            )
    if "temperature_calibration" in by_stage:
        receipt = by_stage["temperature_calibration"]
        values = [float(receipt[phase]["ece"]) for phase in ("before", "after")]
        rows = [
            {"phase": phase, "ece": value}
            for phase, value in zip(("before", "after"), values, strict=True)
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-CAL-001",
                title="Synthetic calibration before/after",
                svg=_svg_bar_chart(
                    "Synthetic calibration before/after", ["before", "after"], values
                ),
                rows=rows,
                provenance=provenance,
                caption_tr="Sentetik kalibrasyon öncesi/sonrası ECE hat doğrulaması.",
                caption_en="Synthetic before/after ECE pipeline validation.",
            )
        )
        reliability_rows: list[dict[str, Any]] = []
        for phase in ("before", "after"):
            for row in receipt[phase]["reliability_diagram"]:
                if row["mean_confidence"] is not None and row["accuracy"] is not None:
                    reliability_rows.append({"phase": phase, **row})
        reliability_series = {
            phase: [float(row["accuracy"]) for row in reliability_rows if row["phase"] == phase]
            for phase in ("before", "after")
        }
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-REL-001",
                title="Synthetic reliability diagram data",
                svg=_svg_line_chart("Synthetic reliability diagram data", reliability_series),
                rows=reliability_rows,
                provenance=provenance,
                caption_tr="Sentetik kalibrasyon güvenilirlik kutuları.",
                caption_en="Synthetic calibration reliability bins.",
            )
        )
    if "zero_shot_ood" in by_stage:
        summaries = by_stage["zero_shot_ood"].get("summaries", {})
        labels = sorted(summaries)
        values = [float(summaries[name]["metrics"].get("fpr_at_95_tpr") or 0.0) for name in labels]
        rows = [
            {"method": name, "fpr95": value} for name, value in zip(labels, values, strict=True)
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-OOD-001",
                title="Synthetic FPR95 plumbing comparison",
                svg=_svg_bar_chart("Synthetic FPR95 plumbing comparison", labels, values),
                rows=rows,
                provenance=provenance,
                caption_tr="Dört skor için sentetik FPR95 raporlama doğrulaması.",
                caption_en="Synthetic FPR95 reporting validation for four scores.",
            )
        )
        distribution_rows = [
            {
                "method": method,
                "class": class_name,
                "mean": values["distribution"][class_name]["mean"],
            }
            for method, values in sorted(summaries.items())
            for class_name in ("id", "anomaly")
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-OOD-DIST-001",
                title="Synthetic uncertainty score distributions",
                svg=_svg_bar_chart(
                    "Synthetic uncertainty score distributions",
                    [f"{row['method']}-{row['class']}" for row in distribution_rows],
                    [float(row["mean"] or 0.0) for row in distribution_rows],
                ),
                rows=distribution_rows,
                provenance=provenance,
                caption_tr="Sentetik ID/anomali skor dağılım özetleri.",
                caption_en="Synthetic ID/anomaly score distribution summaries.",
            )
        )
    if "contextual_risk" in by_stage:
        risk = by_stage["contextual_risk"]["cases"]["persistent_road_center"]
        contribution_rows = [
            {"feature": name, "contribution": value}
            for name, value in risk["normalized_feature_contributions"].items()
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-RISK-001",
                title="Synthetic contextual risk contributions",
                svg=_svg_bar_chart(
                    "Synthetic contextual risk contributions",
                    [str(row["feature"]) for row in contribution_rows],
                    [float(row["contribution"]) for row in contribution_rows],
                ),
                rows=contribution_rows,
                provenance=provenance,
                caption_tr="Sentetik kalıcı yol-merkezi bileşeni için açıklanabilir katkılar.",
                caption_en=(
                    "Explainable contributions for a synthetic persistent road-center component."
                ),
            )
        )
    if "temporal_fusion" in by_stage:
        temporal_rows = [
            {"frame": frame["frame_index"], **track}
            for frame in by_stage["temporal_fusion"]["frames"]
            for track in frame["tracks"]
        ]
        figures.append(
            _write_figure(
                build,
                figure_id="EG-FIG-TEMP-001",
                title="Synthetic temporal persistence sequence",
                svg=_svg_line_chart(
                    "Synthetic temporal persistence sequence",
                    {"persistence": [float(row["persistence_count"]) for row in temporal_rows]},
                ),
                rows=temporal_rows,
                provenance=provenance,
                caption_tr="Sentetik kare dizisinde zamansal kalıcılık.",
                caption_en="Temporal persistence in a synthetic frame sequence.",
            )
        )
    prepared_ids = (
        "validation_curves",
        "ood_roc",
        "ood_precision_recall",
        "semantic_uncertainty_risk_overlays",
        "ablation_comparison",
        "accuracy_latency_memory_tradeoff",
        "jetson_profiles",
    )
    skipped = [
        {
            "figure_id": identifier,
            "status": "not_generated_without_supporting_measured_data",
        }
        for identifier in prepared_ids
    ]
    index = {
        "schema_version": "1.0",
        "record_type": "edgeguard_thesis_figure_index",
        "campaign_id": manifest["campaign_id"],
        "classification": FIGURE_LABEL,
        "generated": figures,
        "skipped": skipped,
    }
    atomic_write_json(build / "figure_index.json", index)
    destination = (
        campaign.root / "reports" / f"edgeguard-thesis-figures-{manifest['campaign_id']}.zip"
    )
    _zip(build, destination)
    return {
        "audience": "thesis",
        "relative_path": destination.relative_to(campaign.root).as_posix(),
        "sha256": sha256_file(destination),
        "byte_size": destination.stat().st_size,
        "figure_index": index,
    }
