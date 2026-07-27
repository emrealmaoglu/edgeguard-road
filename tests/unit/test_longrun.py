"""Tests for minimal live progress and secret-safe subprocess logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, require_finite


def test_long_run_status_is_atomic_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "run_status.json"
    status = LongRunStatus(path, heartbeat_seconds=0.01)
    status.update(
        phase="hashing",
        phase_index=1,
        phase_total=2,
        completed=5,
        total=10,
        speed_per_second=2.0,
        force=True,
    )
    status.complete(last_checkpoint="checkpoint.pth")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["percent"] == 50.0
    assert payload["eta_seconds"] == 2.5
    assert payload["last_checkpoint"] == "checkpoint.pth"
    assert not path.with_name(".run_status.json.incoming").exists()


def test_live_command_redacts_runtime_url_from_console_and_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sensitive_url = "https://example.invalid/archive?signature=private-value"
    status = LongRunStatus(tmp_path / "run_status.json")
    runner = LiveCommandRunner(tmp_path / "logs", status)

    runner.run(
        "redaction-probe",
        (
            sys.executable,
            "-c",
            f"import sys; print({sensitive_url!r}); print({sensitive_url!r}, file=sys.stderr)",
        ),
        display_command=(sys.executable, "-c", "<runtime-url>"),
        stage_index=1,
        stage_total=1,
        redact_values=(sensitive_url,),
    )

    combined = capsys.readouterr().out + capsys.readouterr().err
    logs = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").iterdir())
    assert sensitive_url not in combined
    assert sensitive_url not in logs
    assert "<redacted>" in logs


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_require_finite_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(FloatingPointError):
        require_finite(value, "test value")
