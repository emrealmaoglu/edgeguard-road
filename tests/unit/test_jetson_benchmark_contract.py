from __future__ import annotations

from pathlib import Path

from scripts.jetson.benchmark import parse_tegrastats, realtime_acceptance


def test_tegrastats_parser_uses_only_present_measurements(tmp_path: Path) -> None:
    log = tmp_path / "tegrastats.log"
    log.write_text(
        "RAM 2100/7620MB CPU@45.0C GPU@51.5C VDD_IN 12000mW\n"
        "RAM 2200/7620MB CPU@46.0C GPU@53.0C VDD_IN 14000mW\n",
        encoding="utf-8",
    )
    result = parse_tegrastats(log)
    assert result["mean_input_power_w"] == 13.0
    assert result["peak_ram_used_mib"] == 2200
    assert result["temperature_c"]["GPU"]["peak"] == 53.0
    assert result["throttling_warning_detected"] is False
    assert result["telemetry_complete"] is True


def test_realtime_gate_requires_latency_fps_and_complete_telemetry() -> None:
    telemetry = {"telemetry_complete": True, "throttling_warning_detected": False}
    passing = realtime_acceptance(median_ms=40.0, p95_ms=60.0, fps=22.0, telemetry=telemetry)
    assert passing["passed"] is True
    failing = realtime_acceptance(median_ms=40.0, p95_ms=70.0, fps=22.0, telemetry=telemetry)
    assert failing["passed"] is False
