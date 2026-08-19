"""Reproducible thesis bundles assembled only from accepted measured artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.rescue.colab_recovery import atomic_json
from edgeguard.serialization import sha256_file

CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

ALLOWED_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".png", ".svg", ".txt", ".yaml", ".yml"}
THESIS_COVERAGE = (
    "dataset_distribution",
    "training_curves",
    "miou_latency_pareto",
    "class_iou_heatmap",
    "confusion_matrix",
    "reliability_ece_nll_brier",
    "domain_gap_robustness",
    "loss_and_resolution_ablation",
    "onnx_equivalence",
    "jetson_latency_power_thermal",
    "failure_gallery",
)


def _relative(value: object) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("accepted release contains an unsafe artifact path")
    return relative


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _csv_to_latex(source: Path, destination: Path) -> int:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return 0
    columns = len(rows[0])
    if columns == 0 or any(len(row) != columns for row in rows):
        raise ValueError(f"thesis CSV has inconsistent columns: {source}")
    lines = [r"\begin{tabular}{" + "l" * columns + "}", r"\toprule"]
    for index, row in enumerate(rows):
        lines.append(" & ".join(_escape_latex(value) for value in row) + r" \\")
        if index == 0:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return max(0, len(rows) - 1)


def _zip_directory(source: Path, destination: Path) -> None:
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = ZipInfo(path.relative_to(source).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _save_figure(figure: Any, root: Path, name: str) -> list[Path]:
    paths = [root / f"{name}.png", root / f"{name}.pdf", root / f"{name}.svg"]
    root.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], bbox_inches="tight")
    return paths


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _measured_evidence(
    evidence_root: Path,
    release: dict[str, Any],
    output_root: Path,
    coverage: dict[str, str],
) -> list[Path]:
    """Generate thesis tables and figures only from hash-bound measured outputs."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("measured thesis figures require matplotlib and numpy") from error
    generated: list[Path] = []
    release_id = str(release["release_id"])
    checkpoint_hashes = {
        str(record["model"]): str(record["checkpoint"]["sha256"])
        for record in release.get("models", [])
    }
    evaluations: list[dict[str, Any]] = []
    evaluation_root = evidence_root / "evaluation/evaluate" / release_id
    for path in sorted(evaluation_root.glob("*/*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", ""))
        if (
            payload.get("scientific_evidence") is not True
            or payload.get("role") != "official_source_val"
            or payload.get("checkpoint_sha256") != checkpoint_hashes.get(model)
            or payload.get("official_val_used_for_selection") is not False
        ):
            raise ValueError(f"official thesis evidence identity mismatch: {path}")
        evaluations.append(payload)
    figures = output_root / "figures"
    tables = output_root / "tables"

    distribution_rows: list[tuple[Any, ...]] = []
    class_distribution: list[list[float]] = []
    class_distribution_labels: list[str] = []
    for path in sorted((evidence_root / "manifests/training").glob("*.frozen.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(manifest.get("dataset_id", path.stem))
        roles = manifest.get("roles")
        if not isinstance(roles, dict):
            continue
        totals = np.zeros(19, dtype=np.int64)
        for role, records in sorted(roles.items()):
            if not isinstance(records, list):
                continue
            distribution_rows.append((dataset, role, len(records)))
            for record in records:
                counts = record.get("class_pixel_counts") if isinstance(record, dict) else None
                if isinstance(counts, list) and len(counts) == 19:
                    totals += np.asarray(counts, dtype=np.int64)
        if bool(totals.any()):
            class_distribution.append((totals / totals.sum()).tolist())
            class_distribution_labels.append(dataset)
    if distribution_rows:
        generated.append(
            _write_csv(
                tables / "dataset_role_distribution.csv",
                ("dataset", "role", "sample_count"),
                distribution_rows,
            )
        )
        datasets = sorted({str(row[0]) for row in distribution_rows})
        roles = sorted({str(row[1]) for row in distribution_rows})
        lookup = {(str(row[0]), str(row[1])): int(row[2]) for row in distribution_rows}
        figure, axis = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(datasets))
        for role in roles:
            values = np.asarray([lookup.get((dataset, role), 0) for dataset in datasets])
            axis.bar(datasets, values, bottom=bottom, label=role)
            bottom += values
        axis.set_ylabel("Sample count")
        axis.set_title("Frozen training-manifest role distribution")
        axis.legend()
        generated.extend(_save_figure(figure, figures, "dataset_role_distribution"))
        plt.close(figure)
        coverage["dataset_distribution"] = "accepted"
    if class_distribution:
        generated.append(
            _write_csv(
                tables / "dataset_class_pixel_distribution.csv",
                ("dataset", *CLASS_NAMES),
                [
                    (label, *values)
                    for label, values in zip(
                        class_distribution_labels, class_distribution, strict=True
                    )
                ],
            )
        )
        figure, axis = plt.subplots(figsize=(14, 3.5))
        image = axis.imshow(np.asarray(class_distribution), aspect="auto", cmap="Blues")
        axis.set_xticks(range(19), CLASS_NAMES, rotation=55, ha="right")
        axis.set_yticks(range(len(class_distribution_labels)), class_distribution_labels)
        axis.set_title("Class-pixel proportions in frozen source manifests")
        figure.colorbar(image, ax=axis, label="Pixel proportion")
        generated.extend(_save_figure(figure, figures, "dataset_class_distribution"))
        plt.close(figure)

    history_rows: list[tuple[Any, ...]] = []
    for model in checkpoint_hashes:
        history = evidence_root / "runs/final" / model / "ce/metrics_history.jsonl"
        if not history.is_file():
            continue
        for line in history.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            scalars = record.get("scalars")
            if not isinstance(scalars, dict):
                continue
            loss = scalars.get("loss")
            if loss is None:
                loss_values = [
                    float(value) for name, value in scalars.items() if "loss" in str(name)
                ]
                loss = sum(loss_values) if loss_values else None
            if loss is not None:
                history_rows.append((model, record["optimizer_step"], float(loss)))
    if history_rows:
        generated.append(
            _write_csv(
                tables / "final_training_loss_history.csv",
                ("model", "optimizer_step", "loss"),
                history_rows,
            )
        )
        figure, axis = plt.subplots(figsize=(10, 5))
        for model in checkpoint_hashes:
            rows = [row for row in history_rows if row[0] == model]
            if rows:
                axis.plot([row[1] for row in rows], [row[2] for row in rows], label=model)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Training loss")
        axis.set_title("Final-run measured training curves")
        axis.legend()
        generated.extend(_save_figure(figure, figures, "final_training_curves"))
        plt.close(figure)
        coverage["training_curves"] = "accepted"

    hpo_rows: list[tuple[Any, ...]] = []
    for path in sorted((evidence_root / "runs/hpo").glob("*/trials.snapshot.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trial in payload.get("trials", []):
            hpo_rows.append(
                (
                    payload.get("model"),
                    trial.get("number"),
                    trial.get("state"),
                    trial.get("value"),
                )
            )
    if hpo_rows:
        generated.append(
            _write_csv(
                tables / "hpo_trials.csv",
                ("model", "trial", "state", "domain_macro_mIoU"),
                hpo_rows,
            )
        )
        figure, axis = plt.subplots(figsize=(8, 5))
        for model in sorted({str(row[0]) for row in hpo_rows}):
            rows = [row for row in hpo_rows if row[0] == model and row[3] is not None]
            if rows:
                axis.scatter([row[1] for row in rows], [row[3] for row in rows], label=model)
        axis.set_xlabel("Trial number")
        axis.set_ylabel("Train-select domain-macro mIoU")
        axis.set_title("Measured HPO trial outcomes")
        axis.legend()
        generated.extend(_save_figure(figure, figures, "hpo_trial_outcomes"))
        plt.close(figure)
    if evaluations:
        metric_rows = [
            (
                item["model"],
                item["dataset"],
                item["metrics"]["mIoU"],
                item.get("rare_class_mIoU"),
                item["metrics"].get("mAcc"),
                item["metrics"].get("aAcc"),
            )
            for item in evaluations
        ]
        generated.append(
            _write_csv(
                tables / "official_source_metrics.csv",
                ("model", "dataset", "mIoU", "rare_class_mIoU", "mAcc", "aAcc"),
                metric_rows,
            )
        )
        models = list(dict.fromkeys(str(item["model"]) for item in evaluations))
        evaluation_datasets = ("cityscapes", "idd20k")
        metric_values = {
            (str(item["model"]), str(item["dataset"])): float(item["metrics"]["mIoU"])
            for item in evaluations
        }
        figure, axis = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(models))
        width = 0.36
        for offset, dataset in enumerate(evaluation_datasets):
            axis.bar(
                positions + (offset - 0.5) * width,
                [metric_values[(model, dataset)] for model in models],
                width,
                label=dataset,
            )
        axis.set_ylim(0, 1)
        axis.set_ylabel("mIoU")
        axis.set_xticks(positions, models, rotation=20, ha="right")
        axis.set_title("Official source-domain semantic accuracy")
        axis.legend()
        generated.extend(_save_figure(figure, figures, "official_source_model_comparison"))
        plt.close(figure)
        coverage["domain_gap_robustness"] = "accepted"

        heat_rows: list[list[float]] = []
        heat_labels: list[str] = []
        for item in evaluations:
            per_class = item.get("classwise_metrics", {}).get("per_class_iou")
            if isinstance(per_class, list) and len(per_class) == 19:
                heat_rows.append([np.nan if value is None else float(value) for value in per_class])
                heat_labels.append(f"{item['model']} · {item['dataset']}")
        if heat_rows:
            figure, axis = plt.subplots(figsize=(14, max(5, len(heat_rows) * 0.45)))
            image = axis.imshow(
                np.asarray(heat_rows), aspect="auto", vmin=0, vmax=1, cmap="viridis"
            )
            axis.set_xticks(range(19), CLASS_NAMES, rotation=55, ha="right")
            axis.set_yticks(range(len(heat_labels)), heat_labels)
            axis.set_title("Per-class IoU on official source validation")
            figure.colorbar(image, ax=axis, label="IoU")
            generated.extend(_save_figure(figure, figures, "model_class_iou_heatmap"))
            plt.close(figure)
            coverage["class_iou_heatmap"] = "accepted"

        recommended = str(release.get("recommended_model"))
        confusion = next(
            (
                item.get("classwise_metrics", {}).get("confusion_matrix")
                for item in evaluations
                if item["model"] == recommended and item["dataset"] == "idd20k"
            ),
            None,
        )
        if isinstance(confusion, list) and len(confusion) == 19:
            matrix = np.asarray(confusion, dtype=np.float64)
            denominator = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                denominator,
                out=np.zeros_like(matrix),
                where=denominator > 0,
            )
            figure, axis = plt.subplots(figsize=(10, 9))
            image = axis.imshow(normalized, vmin=0, vmax=1, cmap="magma")
            axis.set_xticks(range(19), CLASS_NAMES, rotation=55, ha="right")
            axis.set_yticks(range(19), CLASS_NAMES)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("Ground-truth class")
            axis.set_title(f"Normalized confusion matrix · {recommended} · IDD20K")
            figure.colorbar(image, ax=axis)
            generated.extend(_save_figure(figure, figures, "recommended_confusion_matrix"))
            plt.close(figure)
            coverage["confusion_matrix"] = "accepted"

        reliability_rows: list[tuple[Any, ...]] = []
        for item in evaluations:
            after = (item.get("reliability") or {}).get("after")
            if isinstance(after, dict):
                reliability_rows.append(
                    (
                        item["model"],
                        item["dataset"],
                        after.get("ece"),
                        after.get("nll"),
                        after.get("brier_score"),
                    )
                )
        if reliability_rows:
            generated.append(
                _write_csv(
                    tables / "calibration_metrics.csv",
                    ("model", "dataset", "ECE", "NLL", "Brier"),
                    reliability_rows,
                )
            )
            coverage["reliability_ece_nll_brier"] = "accepted"
            figure, axis = plt.subplots(figsize=(7, 7))
            axis.plot((0, 1), (0, 1), linestyle="--", color="black", label="ideal")
            plotted = False
            for item in evaluations:
                after = (item.get("reliability") or {}).get("after")
                bins = after.get("reliability_diagram") if isinstance(after, dict) else None
                if not isinstance(bins, list):
                    continue
                points = [
                    (row["mean_confidence"], row["accuracy"])
                    for row in bins
                    if row.get("mean_confidence") is not None and row.get("accuracy") is not None
                ]
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=f"{item['model']} · {item['dataset']}",
                    )
                    plotted = True
            if plotted:
                axis.set_xlabel("Mean confidence")
                axis.set_ylabel("Accuracy")
                axis.set_title("Post-temperature reliability diagram")
                axis.legend(fontsize=7)
                generated.extend(_save_figure(figure, figures, "reliability_diagram"))
            plt.close(figure)

    selection_table = evidence_root / "reports/selection/candidate_table.json"
    if selection_table.is_file():
        payload = json.loads(selection_table.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", [])
        if isinstance(candidates, list) and len(candidates) == 5:
            figure, axis = plt.subplots(figsize=(7, 5))
            for row in candidates:
                axis.scatter(
                    float(row["onnx_median_latency_ms"]),
                    float(row["domain_macro_mIoU"]),
                    s=70,
                )
                axis.annotate(
                    str(row["model"]),
                    (row["onnx_median_latency_ms"], row["domain_macro_mIoU"]),
                )
            axis.set_xlabel("ONNX Runtime CPU median latency (ms)")
            axis.set_ylabel("Train-select domain-macro mIoU")
            axis.set_title("Accuracy–latency comparison (not Jetson evidence)")
            generated.extend(_save_figure(figure, figures, "miou_onnxruntime_pareto"))
            plt.close(figure)
            coverage["miou_latency_pareto"] = "accepted"

    ablation_rows: list[tuple[Any, ...]] = []
    for path in sorted((evidence_root / "evaluation/ablation").glob("*/*/*/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        ablation_rows.append(
            (
                payload.get("model"),
                path.parent.parent.name,
                payload.get("dataset"),
                payload.get("metrics", {}).get("mIoU"),
                payload.get("rare_class_mIoU"),
            )
        )
    if ablation_rows:
        generated.append(
            _write_csv(
                tables / "loss_resolution_ablation.csv",
                ("model", "variant", "dataset", "mIoU", "rare_class_mIoU"),
                ablation_rows,
            )
        )
        coverage["loss_and_resolution_ablation"] = "accepted"

    validation_rows: list[tuple[Any, ...]] = []
    for path in sorted((evidence_root / "exports/export" / release_id).glob("*/*.validation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation_rows.append(
            (
                path.parent.name,
                payload.get("max_absolute_difference"),
                payload.get("mean_absolute_difference"),
                payload.get("allclose_atol_1e_4_rtol_1e_4"),
                payload.get("onnx_bytes"),
            )
        )
    if validation_rows:
        generated.append(
            _write_csv(
                tables / "onnx_equivalence.csv",
                ("model", "max_abs_diff", "mean_abs_diff", "allclose", "onnx_bytes"),
                validation_rows,
            )
        )
        coverage["onnx_equivalence"] = "accepted"
    gallery = evidence_root / "reports/gallery" / release_id
    gallery_manifest = gallery / "gallery_manifest.json"
    if gallery_manifest.is_file():
        gallery_payload = json.loads(gallery_manifest.read_text(encoding="utf-8"))
        gallery_output = figures / "failure_gallery"
        for artifact in gallery_payload.get("artifacts", []):
            source = gallery / str(artifact["path"])
            if not source.is_file() or sha256_file(source) != artifact.get("sha256"):
                raise ValueError("measured gallery artifact identity mismatch")
            destination = gallery_output / str(artifact["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            generated.append(destination)
        coverage["failure_gallery"] = "accepted"
    return generated


def build_thesis_bundle(
    release_manifest_path: Path,
    output_root: Path,
    *,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Collect accepted evidence, source tables, vectors, and claim provenance."""
    if output_root.exists() or output_root.with_suffix(".zip").exists():
        raise ValueError("thesis bundle overwrite is not permitted")
    release = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release.get("record_type") != "edgeguard_accepted_release":
        raise ValueError("invalid accepted release manifest")
    if release.get("status") != "accepted":
        raise ValueError("thesis bundle requires an accepted release")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("accepted release has no artifacts")
    output_root.mkdir(parents=True)
    copied: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []
    coverage = {name: "not_run" for name in THESIS_COVERAGE}
    index_lines = ["# EdgeGuard thesis evidence index", ""]
    for item in artifacts:
        if not isinstance(item, dict) or item.get("scientific_status") != "accepted":
            raise ValueError("thesis source artifact is not accepted")
        relative = _relative(item.get("path"))
        source = release_manifest_path.parent.joinpath(*relative.parts)
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if not source.is_file() or sha256_file(source) != item.get("sha256"):
            raise ValueError(f"accepted thesis artifact identity mismatch: {relative}")
        destination = output_root / "sources" / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        target = destination.relative_to(output_root).as_posix()
        record = {
            "source": relative.as_posix(),
            "bundle_path": target,
            "sha256": sha256_file(destination),
            "run_id": item.get("run_id"),
            "claim_ids": item.get("claim_ids", []),
            "scientific_status": "accepted",
        }
        copied.append(record)
        index_lines.append(
            f"- `{relative.as_posix()}` → `{target}`; run `{item.get('run_id')}`; "
            f"SHA-256 `{record['sha256']}`."
        )
        coverage_name = item.get("coverage")
        if coverage_name in coverage:
            coverage[str(coverage_name)] = "accepted"
        if source.name == "environment.json":
            environments.append(json.loads(source.read_text(encoding="utf-8")))
        if source.suffix.lower() == ".csv":
            latex = output_root / "latex" / Path(*relative.with_suffix(".tex").parts)
            row_count = _csv_to_latex(source, latex)
            copied.append(
                {
                    "source": relative.as_posix(),
                    "bundle_path": latex.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(latex),
                    "row_count": row_count,
                    "scientific_status": "accepted",
                }
            )
    if evidence_root is not None:
        generated = _measured_evidence(evidence_root, release, output_root, coverage)
        for path in generated:
            copied.append(
                {
                    "source": "measured_campaign_evidence",
                    "bundle_path": path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(path),
                    "run_id": release.get("release_id"),
                    "scientific_status": "accepted",
                }
            )
            index_lines.append(
                f"- Measured campaign evidence → "
                f"`{path.relative_to(output_root).as_posix()}`; release "
                f"`{release.get('release_id')}`; SHA-256 `{sha256_file(path)}`."
            )
            if path.suffix.lower() == ".csv":
                latex = output_root / "latex/generated" / f"{path.stem}.tex"
                row_count = _csv_to_latex(path, latex)
                copied.append(
                    {
                        "source": "measured_campaign_evidence",
                        "bundle_path": latex.relative_to(output_root).as_posix(),
                        "sha256": sha256_file(latex),
                        "row_count": row_count,
                        "scientific_status": "accepted",
                    }
                )
    (output_root / "thesis_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    atomic_json(
        output_root / "software_provenance.json",
        {
            "schema_version": "2.0",
            "record_type": "edgeguard_thesis_software_provenance",
            "release_id": release.get("release_id"),
            "environments": environments,
            "source_manifest_sha256": sha256_file(release_manifest_path),
        },
    )
    atomic_json(
        output_root / "coverage.json",
        {
            "schema_version": "2.0",
            "record_type": "edgeguard_thesis_coverage",
            "coverage": coverage,
            "note": "Missing accepted measurements remain not_run; no placeholder is generated.",
        },
    )
    bundle_manifest = {
        "schema_version": "2.0",
        "record_type": "edgeguard_thesis_bundle",
        "release_id": release.get("release_id"),
        "scientific_status": "accepted",
        "source_release_sha256": sha256_file(release_manifest_path),
        "artifacts": copied,
    }
    atomic_json(output_root / "bundle_manifest.json", bundle_manifest)
    archive = output_root.with_suffix(".zip")
    _zip_directory(output_root, archive)
    return {
        "status": "created",
        "scientific_status": "accepted",
        "artifact_count": len(copied),
        "output_root": output_root.name,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "coverage": coverage,
    }
