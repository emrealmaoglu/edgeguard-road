"""Minimal collision-safe JSONL experiment registry."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from edgeguard.serialization import canonical_json
from edgeguard.training.contracts import ExperimentRegistryRecord


def load_registry(path: Path) -> tuple[ExperimentRegistryRecord, ...]:
    """Read and validate a JSONL registry without repairing malformed state."""
    if not path.exists():
        return ()
    records: list[ExperimentRegistryRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"registry contains an empty line at {line_number}")
            payload = json.loads(line)
            records.append(ExperimentRegistryRecord.model_validate(payload))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("experiment registry is missing or malformed") from error
    experiment_ids = [record.experiment_id for record in records]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("experiment registry contains duplicate experiment IDs")
    return tuple(records)


def append_registry(path: Path, record: ExperimentRegistryRecord) -> None:
    """Append one canonical row while refusing identity collisions."""
    records = load_registry(path)
    if any(existing.experiment_id == record.experiment_id for existing in records):
        raise ValueError(f"experiment registry collision: {record.experiment_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(record.model_dump(mode="json")) + "\n")
