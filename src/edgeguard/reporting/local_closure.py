"""Deterministic local-closure audit and small evidence-package factory."""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

VALID_MATURITY = {
    "absent",
    "contract_only",
    "surrogate_validated",
    "real_codepath_validated",
    "local_end_to_end_validated",
    "requires_real_data",
    "requires_cuda",
    "requires_jetson",
    "scientifically_measured",
}

CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("EG-CAP-01", "research hypotheses"),
    ("EG-CAP-02", "dataset acquisition"),
    ("EG-CAP-03", "licensing and provenance"),
    ("EG-CAP-04", "dataset inventory"),
    ("EG-CAP-05", "data quality"),
    ("EG-CAP-06", "exploratory data analysis"),
    ("EG-CAP-07", "ontology and label mapping"),
    ("EG-CAP-08", "leakage-safe splitting"),
    ("EG-CAP-09", "preprocessing"),
    ("EG-CAP-10", "augmentation"),
    ("EG-CAP-11", "sampling and imbalance"),
    ("EG-CAP-12", "dataloading and I/O"),
    ("EG-CAP-13", "semantic training"),
    ("EG-CAP-14", "detector training"),
    ("EG-CAP-15", "OOD development"),
    ("EG-CAP-16", "calibration"),
    ("EG-CAP-17", "trainable anomaly learning"),
    ("EG-CAP-18", "contextual risk"),
    ("EG-CAP-19", "temporal fusion"),
    ("EG-CAP-20", "HPO and promotion"),
    ("EG-CAP-21", "statistical evaluation"),
    ("EG-CAP-22", "error analysis"),
    ("EG-CAP-23", "model export"),
    ("EG-CAP-24", "deployment packaging"),
    ("EG-CAP-25", "Colab observability"),
    ("EG-CAP-26", "Jetson and TensorRT readiness"),
    ("EG-CAP-27", "evidence and thesis reporting"),
)

BEFORE: dict[str, str] = {
    **{f"EG-CAP-{index:02d}": "contract_only" for index in range(1, 28)},
    "EG-CAP-02": "surrogate_validated",
    "EG-CAP-05": "contract_only",
    "EG-CAP-13": "surrogate_validated",
    "EG-CAP-14": "contract_only",
    "EG-CAP-15": "surrogate_validated",
    "EG-CAP-16": "surrogate_validated",
    "EG-CAP-18": "surrogate_validated",
    "EG-CAP-19": "surrogate_validated",
    "EG-CAP-23": "surrogate_validated",
}

AFTER: dict[str, str] = {
    "EG-CAP-01": "contract_only",
    "EG-CAP-02": "local_end_to_end_validated",
    "EG-CAP-03": "requires_real_data",
    "EG-CAP-04": "requires_real_data",
    "EG-CAP-05": "local_end_to_end_validated",
    "EG-CAP-06": "local_end_to_end_validated",
    "EG-CAP-07": "real_codepath_validated",
    "EG-CAP-08": "real_codepath_validated",
    "EG-CAP-09": "local_end_to_end_validated",
    "EG-CAP-10": "local_end_to_end_validated",
    "EG-CAP-11": "real_codepath_validated",
    "EG-CAP-12": "local_end_to_end_validated",
    "EG-CAP-13": "real_codepath_validated",
    "EG-CAP-14": "real_codepath_validated",
    "EG-CAP-15": "local_end_to_end_validated",
    "EG-CAP-16": "local_end_to_end_validated",
    "EG-CAP-17": "real_codepath_validated",
    "EG-CAP-18": "local_end_to_end_validated",
    "EG-CAP-19": "local_end_to_end_validated",
    "EG-CAP-20": "real_codepath_validated",
    "EG-CAP-21": "local_end_to_end_validated",
    "EG-CAP-22": "real_codepath_validated",
    "EG-CAP-23": "real_codepath_validated",
    "EG-CAP-24": "local_end_to_end_validated",
    "EG-CAP-25": "real_codepath_validated",
    "EG-CAP-26": "requires_jetson",
    "EG-CAP-27": "local_end_to_end_validated",
}


def write_gap_matrix(destination: Path) -> dict[str, Any]:
    """Write a complete before/after matrix with honest platform and data boundaries."""
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for capability_id, name in CAPABILITIES:
        before = BEFORE[capability_id]
        after = AFTER[capability_id]
        if before not in VALID_MATURITY or after not in VALID_MATURITY:
            raise ValueError("gap matrix contains an unsupported maturity")
        deferred = after in {"requires_real_data", "requires_cuda", "requires_jetson"}
        records.append(
            {
                "capability_id": capability_id,
                "capability": name,
                "before": {
                    "current_implementation": "repository state at campaign start",
                    "maturity": before,
                    "evidence": ["independent repository audit at starting commit"],
                    "remaining_work": "locally testable closure or external evidence",
                    "scientific_risk": "unmeasured behavior cannot support a thesis claim",
                    "engineering_risk": "contract and executable behavior could diverge",
                },
                "after": {
                    "current_implementation": "project-specific executable local path",
                    "maturity": after,
                    "evidence": ["EG-LOCAL-COMPLETE stage receipt and focused tests"],
                    "remaining_work": (
                        "real data/platform measurement" if deferred else "human scientific review"
                    ),
                    "scientific_risk": (
                        "still unmeasured scientifically"
                        if after != "scientifically_measured"
                        else ""
                    ),
                    "engineering_risk": (
                        "external runtime remains unverified"
                        if deferred
                        else "bounded maintenance risk"
                    ),
                    "action_taken": "implemented and executed the safe local validation path",
                    "action_intentionally_deferred": deferred,
                    "reason": (
                        "requires approved data, CUDA, or Jetson"
                        if deferred
                        else "no scientific promotion from synthetic fixtures"
                    ),
                },
            }
        )
    payload = {
        "schema_version": "2.0",
        "record_type": "edgeguard_project_gap_matrix",
        "classification": "NON-SCIENTIFIC PIPELINE VALIDATION",
        "records": records,
    }
    (destination / "project_gap_matrix.json").write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )
    lines = [
        "# EdgeGuard-Road local final audit",
        "",
        "**NON-SCIENTIFIC PIPELINE VALIDATION.** Maturity is engineering evidence, "
        "not model ranking.",
        "",
        "| ID | Capability | Before | After | Remaining boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in records:
        after = row["after"]
        lines.append(
            f"| {row['capability_id']} | {row['capability']} | "
            f"{row['before']['maturity']} | {after['maturity']} | {after['remaining_work']} |"
        )
    (destination / "project_gap_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _stable_zip(source: Path, destination: Path) -> dict[str, Any]:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = ZipInfo(path.relative_to(source).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return {
        "filename": destination.name,
        "sha256": sha256_file(destination),
        "byte_size": destination.stat().st_size,
    }


def _write_latex_summary(path: Path, stage_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Stage & Status & Maturity \\",
        r"\midrule",
    ]
    for row in stage_rows:
        label = str(row["stage"]).replace("_", r"\_")
        maturity = str(row["maturity"]).replace("_", r"\_")
        lines.append(f"{label} & {row['status']} & {maturity} \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_closure_packages(
    output_root: Path,
    *,
    summary: dict[str, Any],
    repository_root: Path,
    evidence_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build four small deterministic review packages without large model artifacts."""
    evidence = evidence_root or output_root
    package_root = output_root / "packages"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    stage_rows = list(summary["stages"])

    review = package_root / "assistant-review"
    review.mkdir()
    (review / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    gap_root = repository_root / "reports" / "local-final-audit"
    shutil.copy2(gap_root / "project_gap_matrix.json", review / "project_gap_matrix.json")
    shutil.copy2(gap_root / "project_gap_matrix.md", review / "project_gap_matrix.md")
    for name in ("CODEX_INDEPENDENT_CRITIQUE.md", "AUTONOMOUS_EXTRA_WORK.md"):
        shutil.copy2(gap_root / name, review / name)
    _write_latex_summary(review / "stage_summary.tex", stage_rows)

    figures = package_root / "thesis-figures"
    figures.mkdir()
    label = "NON-SCIENTIFIC PIPELINE VALIDATION"
    (figures / "README.md").write_text(
        f"# Thesis figure readiness\n\n**{label}.** "
        "Only locally executed fixture data is included.\n",
        encoding="utf-8",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("stage", "status", "maturity"))
    writer.writeheader()
    writer.writerows(stage_rows)
    (figures / "actual_codepath_matrix.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (figures / "actual_codepath_matrix.json").write_text(
        canonical_json({"classification": label, "rows": stage_rows}) + "\n", encoding="utf-8"
    )

    lifecycle = package_root / "data-lifecycle-audit"
    lifecycle.mkdir()
    for path in (
        repository_root / "docs" / "dataset_cards" / "catalog.json",
        repository_root / "docs" / "dataset_cards" / "catalog.md",
    ):
        shutil.copy2(path, lifecycle / path.name)
    for name in ("acquisition", "data_quality"):
        source = evidence / name / "report.json"
        if source.is_file():
            shutil.copy2(source, lifecycle / f"{name}.json")

    readiness = package_root / "colab-readiness"
    readiness.mkdir()
    for path in (
        repository_root / "configs" / "environment" / "colab_resource_profiles.yaml",
        repository_root / "configs" / "experiment" / "project_preregistration.yaml",
        repository_root / "configs" / "deployment" / "jetson_profiles.yaml",
    ):
        shutil.copy2(path, readiness / path.name)
    notebook_index = [
        path.relative_to(repository_root).as_posix()
        for path in sorted((repository_root / "notebooks" / "colab").glob("*.ipynb"))
    ]
    (readiness / "notebook_index.json").write_text(
        canonical_json({"notebooks": notebook_index, "executed_in_this_campaign": False}) + "\n",
        encoding="utf-8",
    )

    result: dict[str, dict[str, Any]] = {}
    for key, directory in (
        ("assistant_review", review),
        ("thesis_figures", figures),
        ("data_lifecycle_audit", lifecycle),
        ("colab_readiness", readiness),
    ):
        result[key] = _stable_zip(directory, output_root / f"{key.replace('_', '-')}.zip")
    result["package_manifest"] = {
        "sha256": sha256_payload(result),
        "byte_size": sum(int(item["byte_size"]) for item in result.values()),
        "filename": "package-manifest-inline",
    }
    return result
