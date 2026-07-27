"""Minimal observable long-run primitives for Colab scripts and training."""

from __future__ import annotations

import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from edgeguard.serialization import canonical_json

ProgressProbe = Callable[[], tuple[int, int | None]]


def utc_now() -> str:
    """Return one timezone-explicit UTC heartbeat value."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON record atomically within one filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_name(f".{path.name}.incoming")
    incoming.write_text(canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one canonical line and flush it for interruption tolerance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(payload)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ensure_disk_space(path: Path, required_bytes: int, *, reserve_bytes: int = 2 * 1024**3) -> None:
    """Fail before a write when usable disk cannot cover data plus reserve."""
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError("could not resolve an existing filesystem for disk check")
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if required_bytes < 0 or free < required_bytes + reserve_bytes:
        raise OSError(
            f"insufficient disk: required={required_bytes}, reserve={reserve_bytes}, free={free}"
        )


class LongRunStatus:
    """Atomic heartbeat and progress record for one script-first operation."""

    def __init__(self, path: Path, *, heartbeat_seconds: float = 30.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.path = path
        self.heartbeat_seconds = heartbeat_seconds
        self.started = time.monotonic()
        self.last_write = 0.0
        self.payload: dict[str, Any] = {
            "schema_version": "1.0",
            "record_type": "edgeguard_long_run_status",
            "status": "starting",
            "phase": "preflight",
            "phase_index": 0,
            "phase_total": 1,
            "completed": 0,
            "total": None,
            "percent": None,
            "speed_per_second": None,
            "eta_seconds": None,
            "heartbeat_utc": utc_now(),
            "last_checkpoint": None,
            "last_error": None,
        }
        self.update(force=True)

    def update(
        self,
        *,
        force: bool = False,
        status: str | None = None,
        phase: str | None = None,
        phase_index: int | None = None,
        phase_total: int | None = None,
        completed: int | None = None,
        total: int | None = None,
        speed_per_second: float | None = None,
        last_checkpoint: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        """Update status and persist at the heartbeat cadence or when forced."""
        if status is not None:
            self.payload["status"] = status
        if phase is not None:
            self.payload["phase"] = phase
        if phase_index is not None:
            self.payload["phase_index"] = phase_index
        if phase_total is not None:
            self.payload["phase_total"] = phase_total
        if completed is not None:
            self.payload["completed"] = completed
        if total is not None:
            self.payload["total"] = total
        if speed_per_second is not None:
            self.payload["speed_per_second"] = speed_per_second
        if last_checkpoint is not None:
            self.payload["last_checkpoint"] = last_checkpoint
        if last_error is not None:
            self.payload["last_error"] = last_error
        completed_value = int(self.payload["completed"] or 0)
        total_value = self.payload["total"]
        speed_value = self.payload["speed_per_second"]
        if isinstance(total_value, int) and total_value > 0:
            self.payload["percent"] = min(100.0, 100.0 * completed_value / total_value)
            remaining = max(0, total_value - completed_value)
            self.payload["eta_seconds"] = (
                remaining / speed_value
                if isinstance(speed_value, (int, float)) and speed_value > 0
                else None
            )
        else:
            self.payload["percent"] = None
            self.payload["eta_seconds"] = None
        now = time.monotonic()
        if force or now - self.last_write >= self.heartbeat_seconds:
            self.payload["heartbeat_utc"] = utc_now()
            atomic_write_json(self.path, self.payload)
            self.last_write = now
            print(
                canonical_json(
                    {
                        key: self.payload[key]
                        for key in (
                            "phase",
                            "phase_index",
                            "phase_total",
                            "completed",
                            "total",
                            "percent",
                            "speed_per_second",
                            "eta_seconds",
                            "heartbeat_utc",
                            "last_checkpoint",
                            "last_error",
                        )
                    }
                ),
                flush=True,
            )
        return dict(self.payload)

    def fail(self, error: BaseException | str) -> None:
        """Persist a terminal, concise failure without losing previous progress."""
        message = str(error)
        self.update(status="failed", last_error=message[:1000], force=True)

    def complete(self, *, last_checkpoint: str | None = None) -> None:
        """Persist terminal completion and a final heartbeat."""
        self.update(status="completed", last_checkpoint=last_checkpoint, force=True)


def _stream_reader(stream: TextIO, name: str, output: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put((name, line))
    finally:
        output.put((name, None))


def _failure_classification(return_code: int, tail: str) -> str | None:
    if return_code == 0:
        return None
    lowered = tail.lower()
    if "out of memory" in lowered:
        return "cuda_out_of_memory"
    if "no space left" in lowered or "disk quota" in lowered:
        return "disk_exhaustion"
    if "401" in lowered or "403" in lowered or "unauthorized" in lowered:
        return "authentication_or_terms_gate"
    if "timed out" in lowered:
        return "network_or_process_timeout"
    return "command_failed"


class LiveCommandRunner:
    """Run child commands with live output, heartbeat, and durable logs."""

    def __init__(self, log_directory: Path, status: LongRunStatus) -> None:
        self.log_directory = log_directory
        self.status = status
        self.log_directory.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        stage_index: int,
        stage_total: int,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        display_command: Sequence[str] | None = None,
        progress_probe: ProgressProbe | None = None,
        redact_values: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Stream one command and return a path-free execution receipt."""
        if not command:
            raise ValueError("live command cannot be empty")
        safe_name = "".join(character if character.isalnum() else "-" for character in stage).strip(
            "-"
        )
        stdout_path = self.log_directory / f"{stage_index:02d}-{safe_name}.stdout.log"
        stderr_path = self.log_directory / f"{stage_index:02d}-{safe_name}.stderr.log"
        shown = tuple(display_command or command)
        started_utc = utc_now()
        started = time.monotonic()
        print(
            canonical_json(
                {
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_total": stage_total,
                    "command": shlex.join(shown),
                    "started_at": started_utc,
                    "stdout_log": stdout_path.name,
                    "stderr_log": stderr_path.name,
                }
            ),
            flush=True,
        )
        self.status.update(
            status="running",
            phase=stage,
            phase_index=stage_index,
            phase_total=stage_total,
            completed=0,
            force=True,
        )
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None and process.stderr is not None
        output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        threads = [
            threading.Thread(
                target=_stream_reader,
                args=(stream, name, output_queue),
                daemon=True,
            )
            for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
        ]
        for thread in threads:
            thread.start()
        closed: set[str] = set()
        tail: list[str] = []
        with (
            stdout_path.open("a", encoding="utf-8") as stdout_log,
            stderr_path.open("a", encoding="utf-8") as stderr_log,
        ):
            logs = {"stdout": stdout_log, "stderr": stderr_log}
            while len(closed) < 2 or process.poll() is None:
                try:
                    stream_name, line = output_queue.get(timeout=1.0)
                except queue.Empty:
                    line = None
                    stream_name = ""
                if stream_name and line is None:
                    closed.add(stream_name)
                elif line is not None:
                    for secret in redact_values:
                        if secret:
                            line = line.replace(secret, "<redacted>")
                    logs[stream_name].write(line)
                    logs[stream_name].flush()
                    destination = sys.stdout if stream_name == "stdout" else sys.stderr
                    print(line, end="", file=destination, flush=True)
                    tail.append(line)
                    tail = tail[-50:]
                elapsed = max(time.monotonic() - started, 1e-9)
                completed, total = progress_probe() if progress_probe is not None else (0, None)
                speed = completed / elapsed if completed > 0 else None
                self.status.update(
                    completed=completed,
                    total=total,
                    speed_per_second=speed,
                )
        return_code = process.wait()
        elapsed = time.monotonic() - started
        classification = _failure_classification(return_code, "".join(tail))
        receipt = {
            "stage": stage,
            "stage_index": stage_index,
            "stage_total": stage_total,
            "started_at": started_utc,
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "stdout_log": stdout_path.name,
            "stderr_log": stderr_path.name,
            "failure_classification": classification,
        }
        print(canonical_json(receipt), flush=True)
        if return_code != 0:
            self.status.update(last_error=classification or "command_failed", force=True)
            raise subprocess.CalledProcessError(return_code, list(shown))
        self.status.update(
            completed=1, total=1, speed_per_second=1 / max(elapsed, 1e-9), force=True
        )
        return receipt


def require_finite(value: float, description: str) -> float:
    """Reject NaN/Inf before metrics or checkpoint state can be promoted."""
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite {description}")
    return value
