from __future__ import annotations

from edgeguard.rescue.detection_gate import evaluate_detection_gate


def _benchmark(*, p95: float = 39.0, peak_ram: int = 5000) -> dict[str, object]:
    return {
        "record_type": "edgeguard_jetson_tensorrt_sustained_benchmark",
        "scientific_measurement": True,
        "power_profile_declared": "25W",
        "end_to_end_latency_ms": {"p95": p95},
        "telemetry": {"ram_total_mib": 8000, "peak_ram_used_mib": peak_ram},
    }


def test_detection_gate_passes_only_complete_measured_contract() -> None:
    result = evaluate_detection_gate(
        {
            "scientific_experiments_complete": True,
            "external_evaluation_complete": True,
            "semantic_tensorrt_fp16_complete": True,
        },
        _benchmark(),
        {"source_profile": "official", "license_recorded": True, "archive_sha256": "a" * 64},
        remaining_gpu_hours=20.0,
        remaining_calendar_days=14,
    )
    assert result["passed"] is True
    assert result["allowed_detector"] == "rtmdet_tiny"
    assert result["second_detector_family_allowed"] is False


def test_detection_gate_fails_closed_on_missing_memory_and_slow_p95() -> None:
    result = evaluate_detection_gate(
        {},
        _benchmark(p95=41.0, peak_ram=7000),
        {"source_profile": "kaggle_mirror", "license_recorded": False},
        remaining_gpu_hours=100.0,
        remaining_calendar_days=30,
    )
    assert result["passed"] is False
    assert result["allowed_detector"] is None
    assert result["checks"]["semantic_p95_le_40ms"] is False
    assert result["checks"]["safe_memory_at_least_2gib"] is False
