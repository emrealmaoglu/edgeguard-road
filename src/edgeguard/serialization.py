"""Canonical JSON and SHA-256 helpers used by configs and records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

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


def write_verified_zip(destination: Path, members: dict[str, Path | bytes]) -> None:
    """Build a deterministic zip, then reopen and re-hash every member to confirm it
    wrote back exactly what was asked, catching silent disk/filesystem corruption."""
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(members.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("zip member path is unsafe")
            payload = source if isinstance(source, bytes) else source.read_bytes()
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"zip CRC verification failed: {destination.name}")
        if set(archive.namelist()) != set(members):
            raise ValueError(f"zip member verification failed: {destination.name}")
        for name, source in members.items():
            expected = source if isinstance(source, bytes) else source.read_bytes()
            if hashlib.sha256(archive.read(name)).digest() != hashlib.sha256(expected).digest():
                raise ValueError(f"zip payload verification failed: {name}")
