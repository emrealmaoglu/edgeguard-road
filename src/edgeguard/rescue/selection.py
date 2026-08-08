"""Deterministic model selection for the frozen rescue protocol."""

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


def select_recommended_model(
    candidates: list[dict[str, Any]], *, expected_models: tuple[str, ...]
) -> dict[str, Any]:
    """Select one deployment recommendation without consulting final-only data."""
    by_model: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        model = str(candidate.get("model", ""))
        if model not in expected_models or model in by_model:
            continue
        source_domains = candidate.get("source_domain_mIoU")
        if (
            candidate.get("screening_valid") is not True
            or candidate.get("onnx_validated") is not True
            or not isinstance(source_domains, dict)
            or set(source_domains) != {"cityscapes", "idd20k"}
        ):
            continue
        macro = sum(float(value) for value in source_domains.values()) / len(source_domains)
        recorded_value = candidate.get("domain_macro_mIoU", candidate.get("mIoU"))
        if not isinstance(recorded_value, int | float):
            raise ValueError("recommended-model candidate has no numeric domain macro")
        recorded_macro = float(recorded_value)
        if abs(macro - recorded_macro) > 1.0e-12:
            raise ValueError("recommended-model domain macro does not match source metrics")
        by_model[model] = {
            **candidate,
            "domain_macro_mIoU": macro,
            "rare_class_mIoU": float(candidate.get("rare_class_mIoU") or -1.0),
            "onnx_bytes": int(candidate["onnx_bytes"]),
        }
    if set(by_model) != set(expected_models):
        missing = sorted(set(expected_models) - set(by_model))
        raise ValueError(f"recommended-model selection lacks complete evidence: {missing}")
    ranked = sorted(
        by_model.values(),
        key=lambda row: (
            -float(row["domain_macro_mIoU"]),
            -float(row["rare_class_mIoU"]),
            int(row["onnx_bytes"]),
            str(row["model"]),
        ),
    )
    winner = ranked[0]
    return {
        "schema_version": "3.0",
        "record_type": "edgeguard_recommended_model_selection",
        "recommended_model": winner["model"],
        "selection_role": "train_select",
        "selection_domains": ["cityscapes", "idd20k"],
        "selection_rule": [
            "highest_domain_macro_mIoU",
            "highest_rare_class_mIoU",
            "smallest_ONNX_bytes",
            "lexical_model_name",
        ],
        "official_validation_used_for_selection": False,
        "ranked_candidates": ranked,
    }
