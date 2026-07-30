"""Persistent, redacted Colab failure evidence for notebook and subprocess errors."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

_SAFE_STAGE = re.compile(r"[^a-z0-9._-]+")
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s'\"]+"),
    re.compile(r"(?i)((?:token|password|passwd|secret|api[_-]?key)\s*[:=]\s*)[^\s,;'\"]+"),
    re.compile(r"(?i)([?&](?:token|key|signature|x-goog-signature)=)[^&\s]+"),
    re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{12,}\b"),
)
_ALLOWED_DIAGNOSTIC_SUFFIXES = {".json", ".log", ".txt", ".csv", ".yaml", ".yml"}


def redact_failure_text(value: str) -> str:
    """Remove common credential forms without hiding actionable error context."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(lambda match: f"{match.group(1)}<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


def run_logged_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    log_root: Path,
    stage: str,
    check: bool = True,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream a subprocess to the notebook and a bounded, redacted diagnostic log."""
    rendered = [os.fspath(value) for value in command]
    safe_stage = _safe_stage(stage)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{safe_stage}-{uuid.uuid4().hex[:8]}.log"
    tail: deque[str] = deque(maxlen=80)
    with log_path.open("x", encoding="utf-8") as log:
        header = redact_failure_text("COMMAND: " + " ".join(rendered))
        print(header, flush=True)
        log.write(header + "\n")
        process = subprocess.Popen(
            rendered,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = redact_failure_text(line.rstrip("\n"))
            print(safe_line, flush=True)
            log.write(safe_line + "\n")
            log.flush()
            tail.append(safe_line)
        return_code = process.wait()
        footer = f"RETURN_CODE: {return_code}"
        print(footer, flush=True)
        log.write(footer + "\n")
    completed = subprocess.CompletedProcess(rendered, return_code, "\n".join(tail), None)
    if check and return_code != 0:
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"Subprocess failed with exit code {return_code}; log={log_path}\n"
            f"Last output lines:\n{tail_text}"
        )
    return completed


def _safe_stage(value: str) -> str:
    rendered = _SAFE_STAGE.sub("-", value.strip().lower()).strip("-._")
    return rendered[:80] or "unknown-stage"


def _redact_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_failure_text(value)
    if isinstance(value, dict):
        return {str(key): _redact_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_payload(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_failure_text(repr(value))


class ColabFailureReporter:
    """Create append-only failure folders and install an IPython exception hook."""

    def __init__(
        self,
        output_root: Path,
        *,
        notebook: str,
        project_commit: str,
        context: dict[str, Any] | None = None,
        max_diagnostic_bytes: int = 5 * 1024**2,
        max_total_diagnostic_bytes: int = 25 * 1024**2,
    ) -> None:
        if not notebook.strip() or not project_commit.strip():
            raise ValueError("failure reporter requires notebook and project commit identity")
        if max_diagnostic_bytes <= 0 or max_total_diagnostic_bytes <= 0:
            raise ValueError("failure diagnostic limits must be positive")
        self.output_root = output_root
        self.notebook = notebook
        self.project_commit = project_commit
        self.context = dict(context or {})
        self.max_diagnostic_bytes = max_diagnostic_bytes
        self.max_total_diagnostic_bytes = max_total_diagnostic_bytes
        self.stage = "notebook-bootstrap"
        self._diagnostic_roots: list[tuple[str, Path]] = []

    def set_stage(self, stage: str) -> None:
        self.stage = _safe_stage(stage)

    def add_diagnostic_root(self, label: str, root: Path) -> None:
        safe_label = _safe_stage(label)
        resolved = root.resolve()
        item = (safe_label, resolved)
        if item not in self._diagnostic_roots:
            self._diagnostic_roots.append(item)

    def _disk_rows(self) -> list[dict[str, Any]]:
        rows = []
        seen: set[Path] = set()
        for label, root in [("failure-output", self.output_root), *self._diagnostic_roots]:
            probe = root
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            resolved = probe.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            usage = shutil.disk_usage(resolved)
            rows.append(
                {
                    "label": label,
                    "path": str(resolved),
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                }
            )
        return rows

    def _diagnostic_files(self) -> list[tuple[str, Path]]:
        selected: list[tuple[str, Path]] = []
        total = 0
        for label, root in self._diagnostic_roots:
            if not root.is_dir() or root.is_symlink():
                continue
            for source in sorted(root.rglob("*")):
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or source.suffix.lower() not in _ALLOWED_DIAGNOSTIC_SUFFIXES
                ):
                    continue
                size = source.stat().st_size
                if (
                    size > self.max_diagnostic_bytes
                    or total + size > self.max_total_diagnostic_bytes
                ):
                    continue
                relative = source.relative_to(root).as_posix()
                selected.append((f"diagnostics/{label}/{relative}", source))
                total += size
        return selected

    def capture(
        self,
        error_type: type[BaseException],
        error: BaseException,
        trace: TracebackType | None,
    ) -> dict[str, Any]:
        """Persist one immutable JSON/ZIP report and return its public-safe receipt."""
        timestamp = datetime.now(timezone.utc)
        stage = _safe_stage(self.stage)
        failure_id = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{stage}-{uuid.uuid4().hex[:8]}"
        destination = self.output_root / failure_id
        destination.mkdir(parents=True, exist_ok=False)
        formatted_traceback = "".join(traceback.format_exception(error_type, error, trace))
        diagnostics = self._diagnostic_files()
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "record_type": "edgeguard_colab_failure",
            "failure_id": failure_id,
            "failed_at": timestamp.isoformat(),
            "stage": stage,
            "notebook": self.notebook,
            "project_commit": self.project_commit,
            "error_type": error_type.__name__,
            "message": redact_failure_text(str(error)),
            "traceback": redact_failure_text(formatted_traceback),
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "process": {"pid": os.getpid(), "cwd": str(Path.cwd())},
            "disk": self._disk_rows(),
            "context": _redact_payload(self.context),
            "diagnostics": [
                {
                    "archive_path": archive_path,
                    "byte_size": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
                for archive_path, source in diagnostics
            ],
        }
        payload["failure_sha256"] = sha256_payload(payload)
        report_path = destination / "failure.json"
        report_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        package_path = destination / "failure-report.zip"
        with ZipFile(package_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            files = [("failure.json", report_path), *diagnostics]
            for archive_path, source in files:
                info = ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                data = source.read_bytes()
                if source.suffix.lower() in _ALLOWED_DIAGNOSTIC_SUFFIXES:
                    data = redact_failure_text(data.decode("utf-8", errors="replace")).encode(
                        "utf-8"
                    )
                archive.writestr(info, data)
        receipt = {
            "failure_id": failure_id,
            "stage": stage,
            "report": str(report_path),
            "package": str(package_path),
            "package_sha256": sha256_file(package_path),
        }
        (destination / "receipt.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "LATEST.txt").write_text(failure_id + "\n", encoding="utf-8")
        return receipt

    def install_ipython_hook(self) -> bool:
        """Install a notebook-wide hook; return False in ordinary Python/local tests."""
        try:
            ipython: Any = import_module("IPython")
            get_ipython = ipython.get_ipython
        except (ImportError, AttributeError):
            return False
        shell = get_ipython()
        if shell is None:
            return False

        def handler(
            active_shell: Any,
            error_type: type[BaseException],
            error: BaseException,
            trace: TracebackType,
            tb_offset: int | None = None,
        ) -> None:
            try:
                receipt = self.capture(error_type, error, trace)
                print("EDGEGUARD FAILURE REPORT:", canonical_json(receipt))
            except BaseException as reporting_error:
                print(
                    "EDGEGUARD FAILURE REPORTER FAILED:", redact_failure_text(str(reporting_error))
                )
            active_shell.showtraceback((error_type, error, trace), tb_offset=tb_offset)

        shell.set_custom_exc((Exception,), handler)
        return True

    def latest_package(self) -> Path | None:
        """Resolve the latest report ZIP without accepting path traversal in the pointer."""
        pointer = self.output_root / "LATEST.txt"
        if not pointer.is_file():
            return None
        failure_id = pointer.read_text(encoding="utf-8").strip()
        if not failure_id or Path(failure_id).name != failure_id:
            raise ValueError("failure LATEST pointer is unsafe")
        package = self.output_root / failure_id / "failure-report.zip"
        return package if package.is_file() else None
