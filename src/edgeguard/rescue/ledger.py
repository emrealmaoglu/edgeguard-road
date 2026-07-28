"""Append-only provenance ledger for scientific and deployment operations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgeguard.provenance import detect_git_provenance
from edgeguard.serialization import sha256_payload
from edgeguard.telemetry.longrun import append_jsonl


def append_run_ledger(
    path: Path,
    *,
    operation: str,
    result: Mapping[str, Any],
    repository: Path | None = None,
) -> dict[str, Any]:
    """Append one immutable, Git-aware result envelope and return the exact row."""
    git = detect_git_provenance((repository or Path.cwd()).resolve())
    payload = dict(result)
    row = {
        "schema_version": "2.0",
        "record_type": "edgeguard_run_ledger_entry",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "git": {
            "commit": git.commit,
            "state": git.state.value,
            "dirty": git.dirty,
        },
        "result_sha256": sha256_payload(payload),
        "result": payload,
    }
    append_jsonl(path, row)
    return row
