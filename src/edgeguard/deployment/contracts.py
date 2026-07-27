"""Validate executable experiment and Jetson deployment contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import sha256_payload


def _mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError("deployment/preregistration config must be a YAML mapping")
    return payload


def validate_preregistration(path: Path) -> dict[str, Any]:
    """Validate the frozen-role and campaign-order parts of the project protocol."""
    payload = _mapping(path)
    expected_sequence = [
        "five_architecture_smoke",
        "five_short_screening",
        "top_three_medium",
        "top_two_limited_hpo",
        "three_project_owned_final_runs",
        "ood_calibration_development",
        "context_temporal_ablations",
        "frozen_holdout",
        "sealed_final",
        "export",
        "jetson",
    ]
    roles = payload.get("dataset_roles")
    if not isinstance(roles, dict) or roles.get("fishyscapes_lost_and_found") != "frozen_holdout":
        raise ValueError("pre-registration must preserve the Lost & Found frozen holdout")
    if roles.get("smiyc") != "sealed_final":
        raise ValueError("pre-registration must preserve the SMIYC sealed boundary")
    if payload.get("semantic_sequence") != expected_sequence:
        raise ValueError("pre-registration semantic campaign order is invalid")
    final = payload.get("final_run_requirements")
    if not isinstance(final, dict) or final.get("minimum_random_initialization_runs", 0) < 1:
        raise ValueError("at least one final random-initialization run is required")
    if payload.get("scientific_decisions_automated") is not False:
        raise ValueError("scientific decisions cannot be automated")
    return {"status": "valid", "config_sha256": sha256_payload(payload), "payload": payload}


def validate_jetson_profiles(path: Path) -> dict[str, Any]:
    """Validate three unassigned profiles and the measurement/INT8 gates."""
    payload = _mapping(path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "A_accuracy",
        "B_balanced",
        "C_edge_real_time",
    }:
        raise ValueError("Jetson profiles A/B/C are required")
    assigned = any(
        not isinstance(value, dict) or value.get("model_assignment") is not None
        for value in profiles.values()
    )
    if assigned:
        raise ValueError("Jetson models cannot be assigned before platform measurement")
    int8 = payload.get("int8")
    if not isinstance(int8, dict) or int8.get("enabled") is not False:
        raise ValueError("INT8 must remain disabled pending accuracy-retention evidence")
    common = payload.get("common")
    if not isinstance(common, dict) or not common.get("tensorrt_parser_required"):
        raise ValueError("TensorRT parser validation is a deployment gate")
    return {"status": "valid", "config_sha256": sha256_payload(payload), "payload": payload}
