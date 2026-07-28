"""Deterministic top-two model selection for the frozen rescue protocol."""

from __future__ import annotations

from typing import Any


def select_top_two(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose accuracy and edge finalists without collapsing evidence to one score."""
    if len(candidates) < 2:
        raise ValueError("top-two selection requires at least two candidates")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        record = dict(candidate)
        if "domain_macro_mIoU" not in record:
            record["domain_macro_mIoU"] = record.get("mIoU")
        for field in ("domain_macro_mIoU", "onnx_median_latency_ms", "onnx_bytes"):
            value = float(record[field])
            if value < 0:
                raise ValueError(f"candidate field {field} cannot be negative")
            record[field] = value
        if not bool(record.get("onnx_validated")):
            continue
        normalized.append(record)
    if len(normalized) < 2:
        raise ValueError("at least two ONNX-validated candidates are required")
    best_macro = max(row["domain_macro_mIoU"] for row in normalized)
    scientific_ties = [row for row in normalized if best_macro - row["domain_macro_mIoU"] <= 0.002]
    scientific = sorted(
        scientific_ties,
        key=lambda row: (
            -float(row.get("rare_class_mIoU") or -1.0),
            -row["domain_macro_mIoU"],
            str(row["model"]),
        ),
    )[0]
    ranked = sorted(
        normalized,
        key=lambda row: (-row["domain_macro_mIoU"], str(row["model"])),
    )
    eligible_edge = [
        row for row in ranked if scientific["domain_macro_mIoU"] - row["domain_macro_mIoU"] <= 0.03
    ]
    edge = min(
        eligible_edge,
        key=lambda row: (
            row["onnx_median_latency_ms"],
            row["onnx_bytes"],
            -row["domain_macro_mIoU"],
            str(row["model"]),
        ),
    )
    if edge["model"] == scientific["model"]:
        edge = next(row for row in ranked if row["model"] != scientific["model"])
    return {
        "schema_version": "1.0",
        "record_type": "semantic_top_two_selection",
        "scientific_candidate": scientific["model"],
        "edge_candidate": edge["model"],
        "accuracy_rule": "highest_source_domain_macro_train_select_mIoU_then_rare_mIoU",
        "edge_rule": "lowest_ONNX_latency_within_0.03_domain_macro_mIoU_then_size",
        "eligible_candidates": normalized,
        "human_acceptance_required": True,
    }
