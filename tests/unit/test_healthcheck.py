"""Tests for failure-tolerant optional package discovery."""

import json
import subprocess
from typing import Any

import pytest

import edgeguard.healthcheck as healthcheck
from edgeguard import cli
from scripts import collect_env


def test_missing_optional_package_is_reported_without_failure() -> None:
    report = healthcheck.inspect_optional_package(
        "edgeguard_package_that_does_not_exist",
        ("edgeguard-distribution-that-does-not-exist",),
    )

    assert report["present"] is False
    assert report["probe_status"] == "not_found"


def test_broken_isolated_import_is_reported(monkeypatch: Any) -> None:
    monkeypatch.setattr(healthcheck, "_module_present", lambda _name: (True, None))
    monkeypatch.setattr(healthcheck, "_distribution_version", lambda _names: None)
    monkeypatch.setattr(
        healthcheck,
        "_isolated_probe",
        lambda _name, include_cuda, *, probe_timeout_seconds: {
            "probe_status": "error",
            "error": "native failure",
        },
    )

    report = healthcheck.inspect_optional_package("broken", ("broken",))

    assert report["present"] is True
    assert report["probe_status"] == "error"
    assert report["error"] == "native failure"


def test_doctor_report_survives_optional_probe_errors(monkeypatch: Any) -> None:
    def failed_probe(
        _module: str,
        _distributions: tuple[str, ...],
        *,
        probe_timeout_seconds: float,
    ) -> dict[str, Any]:
        assert probe_timeout_seconds == healthcheck.DEFAULT_PROBE_TIMEOUT_SECONDS
        return {
            "present": True,
            "version": None,
            "probe_status": "error",
            "error": "simulated",
        }

    monkeypatch.setattr(healthcheck, "inspect_optional_package", failed_probe)

    report = healthcheck.doctor_report()

    assert report["status"] == "ok_with_optional_errors"
    assert set(report["packages"]) == set(healthcheck.OPTIONAL_PACKAGES)


def test_custom_timeout_reaches_optional_package_probes(monkeypatch: Any) -> None:
    observed: list[float] = []

    def inspect(
        _module: str,
        _distributions: tuple[str, ...],
        *,
        probe_timeout_seconds: float,
    ) -> dict[str, Any]:
        observed.append(probe_timeout_seconds)
        return {
            "present": False,
            "version": None,
            "probe_status": "not_found",
            "error": None,
        }

    monkeypatch.setattr(healthcheck, "inspect_optional_package", inspect)

    healthcheck.doctor_report(probe_timeout_seconds=160.0)

    assert observed == [160.0] * len(healthcheck.OPTIONAL_PACKAGES)


def test_optional_package_forwards_custom_timeout_to_isolated_probe(monkeypatch: Any) -> None:
    observed: list[float] = []

    def probe(
        _module: str,
        include_cuda: bool,
        *,
        probe_timeout_seconds: float,
    ) -> dict[str, Any]:
        observed.append(probe_timeout_seconds)
        return {"probe_status": "timeout", "error": "simulated timeout"}

    monkeypatch.setattr(healthcheck, "_module_present", lambda _name: (True, None))
    monkeypatch.setattr(healthcheck, "_distribution_version", lambda _names: None)
    monkeypatch.setattr(healthcheck, "_isolated_probe", probe)

    healthcheck.inspect_optional_package(
        "torch",
        ("torch",),
        probe_timeout_seconds=160.0,
    )

    assert observed == [160.0]


def test_isolated_probe_uses_custom_subprocess_timeout(monkeypatch: Any) -> None:
    observed: list[float] = []

    def run(
        command: list[str],
        *,
        capture_output: bool,
        check: bool,
        text: bool,
        timeout: float,
    ) -> None:
        observed.append(timeout)
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(healthcheck.subprocess, "run", run)

    result = healthcheck._isolated_probe(
        "torch",
        include_cuda=True,
        probe_timeout_seconds=160.0,
    )

    assert observed == [160.0]
    assert result["probe_status"] == "timeout"


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_doctor_report_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        healthcheck.doctor_report(probe_timeout_seconds=timeout)


@pytest.mark.parametrize(
    ("arguments", "expected_timeout"),
    [([], 20.0), (["--probe-timeout-seconds", "160"], 160.0)],
)
def test_collect_env_outputs_json_and_forwards_timeout(
    arguments: list[str],
    expected_timeout: float,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    observed: list[float] = []

    def report(*, probe_timeout_seconds: float) -> dict[str, Any]:
        observed.append(probe_timeout_seconds)
        return {"schema_version": "1.0", "status": "ok"}

    monkeypatch.setattr(collect_env, "doctor_report", report)

    assert collect_env.main(arguments) == 0
    assert observed == [expected_timeout]
    assert json.loads(capsys.readouterr().out) == {"schema_version": "1.0", "status": "ok"}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_collect_env_rejects_invalid_timeout(value: str, capsys: Any) -> None:
    with pytest.raises(SystemExit) as error:
        collect_env.main(["--probe-timeout-seconds", value])

    assert error.value.code == 2
    assert "positive finite number" in capsys.readouterr().err


def test_doctor_cli_keeps_json_parseable_on_unexpected_error(monkeypatch: Any, capsys: Any) -> None:
    def fail() -> dict[str, Any]:
        raise RuntimeError("simulated doctor failure")

    monkeypatch.setattr(cli, "doctor_report", fail)

    assert cli.main(["doctor", "--json"]) == 1
    assert '"status":"error"' in capsys.readouterr().out
