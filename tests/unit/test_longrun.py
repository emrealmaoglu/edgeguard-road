"""Tests for minimal live progress and secret-safe subprocess logging."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from edgeguard.serialization import sha256_file
from edgeguard.telemetry.longrun import (
    LiveCommandRunner,
    LongRunStatus,
    atomic_copy_verified,
    ensure_disk_space,
    require_finite,
    require_fresh_heartbeat,
)


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


def test_stale_heartbeat_is_rejected() -> None:
    now = datetime(2026, 7, 27, tzinfo=timezone.utc)
    with pytest.raises(TimeoutError, match="stale"):
        require_fresh_heartbeat(
            {"heartbeat_utc": (now - timedelta(seconds=61)).isoformat()},
            now=now,
            maximum_age_seconds=60,
        )
    require_fresh_heartbeat(
        {"heartbeat_utc": (now - timedelta(seconds=30)).isoformat()},
        now=now,
        maximum_age_seconds=60,
    )


def test_disk_exhaustion_prediction_fails_before_write(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="insufficient disk"):
        ensure_disk_space(tmp_path, 2**63, reserve_bytes=0)


def test_failed_artifact_copy_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"validated")

    def fail_copy(_source: Path, _destination: Path) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr("shutil.copy2", fail_copy)
    with pytest.raises(OSError, match="injected"):
        atomic_copy_verified(source, destination, expected_sha256=sha256_file(source))
    assert not destination.exists()
    assert not destination.with_name(".destination.bin.incoming").exists()
