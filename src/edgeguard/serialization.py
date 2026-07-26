"""Canonical JSON and SHA-256 helpers used by configs and records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert supported application values to JSON-compatible primitives."""
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a value using the project's stable canonical JSON settings."""
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def sha256_payload(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file as raw bytes without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: npt.NDArray[Any]) -> str:
    """Hash array dtype, shape, and C-order bytes deterministically."""
    contiguous = np.ascontiguousarray(array)
    descriptor = {
        "dtype": contiguous.dtype.str,
        "shape": [int(dimension) for dimension in contiguous.shape],
    }
    digest = hashlib.sha256(canonical_json(descriptor).encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
