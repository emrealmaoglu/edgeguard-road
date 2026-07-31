"""Hash-verified resumable downloader used only with approved sources."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload


def _redacted_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    retries: int = 3,
    backoff_seconds: float = 0.05,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resume to `.part`, verify size/hash, then atomically promote."""
    if expected_size <= 0 or retries < 1 or backoff_seconds < 0:
        raise ValueError("download size/retry/backoff contract is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        if (
            destination.stat().st_size == expected_size
            and sha256_file(destination) == expected_sha256
        ):
            return {
                "status": "reused",
                "byte_size": expected_size,
                "sha256": expected_sha256,
                "source": _redacted_url(url),
            }
        raise ValueError("existing download destination does not match the requested identity")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    completed = offset
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        completed += len(chunk)
                        if progress is not None:
                            elapsed = max(time.monotonic() - started, 1e-9)
                            speed = (completed - offset) / elapsed
                            progress(
                                {
                                    "completed_bytes": completed,
                                    "total_bytes": expected_size,
                                    "speed_bytes_per_second": speed,
                                    "eta_seconds": max(0, expected_size - completed)
                                    / max(speed, 1e-9),
                                    "attempt": attempt,
                                }
                            )
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != expected_size:
                raise ValueError("download byte size mismatch")
            actual_sha = sha256_file(partial)
            if actual_sha != expected_sha256:
                raise ValueError("download SHA-256 mismatch")
            os.replace(partial, destination)
            return {
                "status": "downloaded",
                "byte_size": expected_size,
                "sha256": actual_sha,
                "source": _redacted_url(url),
                "attempts": attempt,
            }
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def dataset_artifact_readiness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require every declared artifact identity before a dataset becomes ready."""
    if not records:
        raise ValueError("dataset readiness requires declared artifacts")
    missing = [str(record.get("artifact_id")) for record in records if not record.get("verified")]
    return {
        "artifact_count": len(records),
        "missing_or_unverified": missing,
        "ready": not missing,
    }


def generate_fixture_bundle(
    output_root: Path,
    *,
    generator_config: dict[str, Any],
    interrupt_after_files: int | None = None,
) -> dict[str, Any]:
    """Generate/resume a deterministic synthetic acquisition bundle and manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_payload(generator_config)
    state_path = output_root / "generator_state.json"
    if state_path.is_file():
        import json

        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("generator_config_sha256") != config_sha:
            raise ValueError("fixture generator resume identity mismatch")
    else:
        state = {"generator_config_sha256": config_sha, "completed_files": []}
    count = int(generator_config.get("file_count", 0))
    seed = int(generator_config.get("seed", -1))
    if not 1 <= count <= 100 or seed < 0:
        raise ValueError("fixture generator config is invalid")
    generated = 0
    for index in range(count):
        filename = f"fixture-{index:03d}.bin"
        path = output_root / filename
        payload = sha256_payload({"seed": seed, "index": index}).encode("ascii")
        if filename in state["completed_files"]:
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError("fixture generator completed file is corrupt")
            continue
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(payload)
        os.replace(partial, path)
        state["completed_files"].append(filename)
        state_path.write_text(canonical_json(state) + "\n", encoding="utf-8")
        generated += 1
        if interrupt_after_files is not None and generated >= interrupt_after_files:
            raise InterruptedError("bounded fixture generator interruption")
    files = [
        {
            "relative_path": name,
            "byte_size": (output_root / name).stat().st_size,
            "sha256": sha256_file(output_root / name),
        }
        for name in sorted(state["completed_files"])
    ]
    manifest = {
        "schema_version": "1.0",
        "record_type": "synthetic_acquisition_generator_manifest",
        "generator_config_sha256": config_sha,
        "files": files,
        "scientific_evidence": False,
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    archive = output_root / "fixture-bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for record in files:
            info = ZipInfo(record["relative_path"], (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, (output_root / record["relative_path"]).read_bytes())
    return {
        "manifest": manifest,
        "archive_filename": archive.name,
        "archive_sha256": sha256_file(archive),
        "resumed": generated < count,
    }


def copy_with_progress(
    source: Path,
    destination: Path,
    *,
    chunk_size: int = 1024,
    delay_seconds: float = 0.0,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Simulate a slow Drive-like destination with atomic verified promotion."""
    if chunk_size <= 0 or delay_seconds < 0 or destination.exists():
        raise ValueError("copy destination/chunk/delay contract is invalid")
    partial = destination.with_name(destination.name + ".part")
    completed = 0
    with source.open("rb") as input_stream, partial.open("xb") as output_stream:
        while chunk := input_stream.read(chunk_size):
            output_stream.write(chunk)
            completed += len(chunk)
            if progress is not None:
                progress({"completed_bytes": completed, "total_bytes": source.stat().st_size})
            if delay_seconds:
                time.sleep(delay_seconds)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if sha256_file(source) != sha256_file(partial):
        raise RuntimeError("slow destination copy failed SHA-256 verification")
    os.replace(partial, destination)
    return {"byte_size": completed, "sha256": sha256_file(destination)}
