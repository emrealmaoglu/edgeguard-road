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


def build_thesis_bundle(release_manifest_path: Path, output_root: Path) -> dict[str, Any]:
    """Collect accepted evidence, source tables, vectors, and claim provenance."""
    if output_root.exists() or output_root.with_suffix(".zip").exists():
        raise ValueError("thesis bundle overwrite is not permitted")
    release = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if release.get("record_type") != "edgeguard_accepted_release":
        raise ValueError("invalid accepted release manifest")
    if release.get("status") != "accepted":
        raise ValueError("thesis bundle requires a human-accepted release")
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
