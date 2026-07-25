"""Tests for failure-tolerant optional package discovery."""

from typing import Any

import edgeguard.healthcheck as healthcheck
from edgeguard import cli


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
        lambda _name, include_cuda: {"probe_status": "error", "error": "native failure"},
    )

    report = healthcheck.inspect_optional_package("broken", ("broken",))

    assert report["present"] is True
    assert report["probe_status"] == "error"
    assert report["error"] == "native failure"


def test_doctor_report_survives_optional_probe_errors(monkeypatch: Any) -> None:
    def failed_probe(_module: str, _distributions: tuple[str, ...]) -> dict[str, Any]:
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


def test_doctor_cli_keeps_json_parseable_on_unexpected_error(monkeypatch: Any, capsys: Any) -> None:
    def fail() -> dict[str, Any]:
        raise RuntimeError("simulated doctor failure")

    monkeypatch.setattr(cli, "doctor_report", fail)

    assert cli.main(["doctor", "--json"]) == 1
    assert '"status":"error"' in capsys.readouterr().out
