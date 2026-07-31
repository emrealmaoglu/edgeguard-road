"""Fail-closed phase-two RTMDet-Tiny activation gate."""

from __future__ import annotations

from typing import Any

REQUIRED_PHASE_ONE = (
    "scientific_experiments_complete",
    "external_evaluation_complete",
    "semantic_tensorrt_fp16_complete",
)


def evaluate_detection_gate(
    phase_one: dict[str, Any],
    jetson_benchmark: dict[str, Any],
    bdd_detection_provenance: dict[str, Any],
    *,
    remaining_gpu_hours: float,
    remaining_calendar_days: int,
) -> dict[str, Any]:
    """Return an auditable decision; absence or ambiguity always closes the gate."""
    phase_checks = {name: phase_one.get(name) is True for name in REQUIRED_PHASE_ONE}
    end_to_end = jetson_benchmark.get("end_to_end_latency_ms", {})
    telemetry = jetson_benchmark.get("telemetry", {})
    total_ram = telemetry.get("ram_total_mib")
    peak_ram = telemetry.get("peak_ram_used_mib")
    safe_memory_mib = (
        float(total_ram) - float(peak_ram)
        if isinstance(total_ram, (int, float)) and isinstance(peak_ram, (int, float))
        else None
    )
    p95 = end_to_end.get("p95")
    checks = {
        **phase_checks,
        "jetson_25w_measured": (
            jetson_benchmark.get("record_type") == "edgeguard_jetson_tensorrt_sustained_benchmark"
            and jetson_benchmark.get("scientific_measurement") is True
            and jetson_benchmark.get("power_profile_declared") == "25W"
        ),
        "semantic_p95_le_40ms": isinstance(p95, (int, float)) and float(p95) <= 40.0,
        "safe_memory_at_least_2gib": (safe_memory_mib is not None and safe_memory_mib >= 2048.0),
        "remaining_gpu_hours_at_least_20": remaining_gpu_hours >= 20.0,
        "remaining_calendar_days_at_least_14": remaining_calendar_days >= 14,
        "official_bdd_detection_labels": (
            bdd_detection_provenance.get("source_profile") == "official"
            and bdd_detection_provenance.get("license_recorded") is True
            and isinstance(bdd_detection_provenance.get("archive_sha256"), str)
            and len(bdd_detection_provenance["archive_sha256"]) == 64
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_phase_two_detection_gate",
        "checks": checks,
        "passed": passed,
        "allowed_detector": "rtmdet_tiny" if passed else None,
        "second_detector_family_allowed": False,
        "safe_memory_mib": safe_memory_mib,
        "semantic_p95_ms": float(p95) if isinstance(p95, (int, float)) else None,
    }
