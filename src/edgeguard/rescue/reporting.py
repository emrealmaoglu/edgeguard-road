"""Evidence-only tables and presentation material from measured run records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from edgeguard.rescue.selection import select_top_two
from edgeguard.serialization import canonical_json


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    for key, value in metrics.items():
        if key == name or key.endswith(f"/{name}"):
            return float(value)
    return None


def build_evidence_report(
    evaluation_root: Path, export_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Build week-3/final-safe tables without estimating missing measurements."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output_dir}")
    output_dir.mkdir(parents=True)
    evaluations: list[dict[str, Any]] = []
    for path in sorted(evaluation_root.glob("**/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluations.append(
            {
                "model": payload.get("model"),
                "dataset": payload.get("dataset"),
                "role": payload.get("role"),
                "condition": payload.get("condition"),
                "mIoU": _metric(payload.get("metrics", {}), "mIoU"),
                "rare_class_mIoU": payload.get("rare_class_mIoU"),
                "ece_before": (
                    payload.get("reliability", {}).get("before", {}).get("ece")
                    if payload.get("reliability")
                    else None
                ),
                "ece_after": (
                    payload.get("reliability", {}).get("after", {}).get("ece")
                    if payload.get("reliability")
                    else None
                ),
                "mean_confidence_before": (
                    payload.get("reliability", {})
                    .get("confidence_entropy_before", {})
                    .get("mean_max_softmax_confidence")
                    if payload.get("reliability")
                    else None
                ),
                "mean_entropy_before": (
                    payload.get("reliability", {})
                    .get("confidence_entropy_before", {})
                    .get("mean_normalized_entropy")
                    if payload.get("reliability")
                    else None
                ),
                "mean_confidence_after": (
                    payload.get("reliability", {})
                    .get("confidence_entropy_after", {})
                    .get("mean_max_softmax_confidence")
                    if payload.get("reliability")
                    else None
                ),
                "mean_entropy_after": (
                    payload.get("reliability", {})
                    .get("confidence_entropy_after", {})
                    .get("mean_normalized_entropy")
                    if payload.get("reliability")
                    else None
                ),
                "source": path.relative_to(evaluation_root).as_posix(),
            }
        )
    exports: dict[str, dict[str, Any]] = {}
    for path in sorted(export_root.glob("**/*.validation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        exports[path.name.removesuffix(".validation.json")] = payload
    candidates: list[dict[str, Any]] = []
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for row in evaluations:
        if row["role"] == "train_select" and row["mIoU"] is not None:
            source_rows.setdefault(str(row["model"]), []).append(row)
    for model, rows in sorted(source_rows.items()):
        if model not in exports:
            continue
        export = exports[model]
        domain_values = {str(row["dataset"]): row["mIoU"] for row in rows}
        if set(domain_values) != {"cityscapes", "bdd100k", "idd20k"}:
            continue
        onnx_validated = bool(
            export.get("shape_equal") and export.get("allclose_atol_1e_4_rtol_1e_4")
        )
        if not onnx_validated:
            continue
        macro_miou = sum(float(row["mIoU"]) for row in rows) / len(rows)
        rare_values = [
            float(row["rare_class_mIoU"]) for row in rows if row["rare_class_mIoU"] is not None
        ]
        candidates.append(
            {
                "model": model,
                "domain_macro_mIoU": macro_miou,
                "mIoU": macro_miou,
                "source_domain_mIoU": domain_values,
                "rare_class_mIoU": (sum(rare_values) / len(rare_values) if rare_values else None),
                "onnx_validated": onnx_validated,
                "screening_valid": True,
                "onnx_median_latency_ms": export["onnxruntime_cpu"]["median_latency_ms"],
                "onnx_bytes": export["onnx_bytes"],
            }
        )
    table_fields = [
        "model",
        "dataset",
        "role",
        "condition",
        "mIoU",
        "rare_class_mIoU",
        "ece_before",
        "ece_after",
        "mean_confidence_before",
        "mean_entropy_before",
        "mean_confidence_after",
        "mean_entropy_after",
        "source",
    ]
    with (output_dir / "metrics_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(evaluations)
    baselines = {
        str(row["model"]): row
        for row in evaluations
        if row["dataset"] == "cityscapes" and row["role"] == "official_val_common_eval"
    }
    shift_rows: list[dict[str, Any]] = []
    for row in evaluations:
        baseline = baselines.get(str(row["model"]))
        if baseline is None or row["role"] not in {
            "domain_shift_val",
            "sealed_external_test",
            "official_source_val",
        }:
            continue
        shift_rows.append(
            {
                "model": row["model"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "mIoU": row["mIoU"],
                "mIoU_drop_from_cityscapes": (
                    float(baseline["mIoU"]) - float(row["mIoU"])
                    if baseline["mIoU"] is not None and row["mIoU"] is not None
                    else None
                ),
                "rare_class_mIoU_drop": (
                    float(baseline["rare_class_mIoU"]) - float(row["rare_class_mIoU"])
                    if baseline["rare_class_mIoU"] is not None
                    and row["rare_class_mIoU"] is not None
                    else None
                ),
                "confidence_change": (
                    float(row["mean_confidence_after"]) - float(baseline["mean_confidence_after"])
                    if baseline["mean_confidence_after"] is not None
                    and row["mean_confidence_after"] is not None
                    else None
                ),
                "entropy_change": (
                    float(row["mean_entropy_after"]) - float(baseline["mean_entropy_after"])
                    if baseline["mean_entropy_after"] is not None
                    and row["mean_entropy_after"] is not None
                    else None
                ),
            }
        )
    with (output_dir / "domain_shift_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model",
            "dataset",
            "condition",
            "mIoU",
            "mIoU_drop_from_cityscapes",
            "rare_class_mIoU_drop",
            "confidence_change",
            "entropy_change",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shift_rows)
    candidate_payload = {"schema_version": "1.0", "candidates": candidates}
    (output_dir / "candidate_table.json").write_text(
        canonical_json(candidate_payload) + "\n", encoding="utf-8"
    )
    selection: dict[str, Any] | None = None
    if len(candidates) >= 2:
        selection = select_top_two(candidates)
        (output_dir / "top_two.json").write_text(canonical_json(selection) + "\n", encoding="utf-8")
    summary_lines = [
        "# EdgeGuard-Road measured evidence summary",
        "",
        f"- Evaluation records: {len(evaluations)}",
        f"- ONNX-linked screening candidates: {len(candidates)}",
        "- Top-two status: "
        + ("candidate generated; human acceptance required" if selection else "pending"),
        "",
        "Only values present in immutable evaluation/export records appear in the tables.",
        "Missing measurements remain blank and no synthetic fixture is promoted as "
        "scientific evidence.",
    ]
    (output_dir / "week3_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    outline = """# Presentation outline

1. Multi-domain generalization question and edge target
2. Cityscapes, BDD100K, and IDD20K audits, ontology, split roles, and leakage controls
3. Domain-uniform sampling, class imbalance, and frozen rare-class definition
4. Five lightweight MMSeg models under one optimizer-step protocol
5. Measured three-domain pilot/screening and bounded top-two HPO
6. Dataset-composition and CrossEntropy/weighted-CrossEntropy ablations
7. Equal-source global temperature scaling and reliability
8. Source validation, ACDC domain shift, and sealed WildDash/MUSES evidence
9. ONNX agreement, CPU latency, and optional Jetson evidence
10. Streamlit demonstration
11. Limitations, negative results, and prohibited post-external tuning
"""
    (output_dir / "presentation_outline.md").write_text(outline, encoding="utf-8")
    return {
        "schema_version": "1.0",
        "record_type": "semantic_evidence_report",
        "evaluation_count": len(evaluations),
        "candidate_count": len(candidates),
        "selection_generated": selection is not None,
        "scientific_evidence": bool(evaluations),
    }
