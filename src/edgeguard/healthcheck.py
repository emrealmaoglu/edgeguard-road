"""Failure-tolerant local environment inventory for the doctor command."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from typing import Any

OPTIONAL_PACKAGES: dict[str, tuple[str, ...]] = {
    "numpy": ("numpy",),
    "torch": ("torch",),
    "onnx": ("onnx",),
    "onnxruntime": ("onnxruntime",),
    "tensorrt": ("tensorrt",),
    "cv2": ("opencv-python", "opencv-python-headless"),
}
DEFAULT_PROBE_TIMEOUT_SECONDS = 20.0


def _validated_probe_timeout(probe_timeout_seconds: float) -> float:
    if not math.isfinite(probe_timeout_seconds) or probe_timeout_seconds <= 0:
        raise ValueError("probe_timeout_seconds must be a positive finite number")
    return probe_timeout_seconds


def _safe_error(error: object, limit: int = 300) -> str:
    text = " ".join(str(error).split())
    return text[:limit]


def _module_present(module_name: str) -> tuple[bool, str | None]:
    try:
        return importlib.util.find_spec(module_name) is not None, None
    except Exception as error:  # optional packages can have broken import hooks
        return False, _safe_error(error)


def _distribution_version(distributions: tuple[str, ...]) -> str | None:
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _isolated_probe(
    module_name: str,
    include_cuda: bool,
    *,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    probe_timeout_seconds = _validated_probe_timeout(probe_timeout_seconds)
    script = (
        "import importlib,json;"
        f"m=importlib.import_module({module_name!r});"
        "r={'module_version':str(getattr(m,'__version__','unknown'))};"
        + ("r['cuda_available']=bool(m.cuda.is_available());" if include_cuda else "")
        + "print(json.dumps(r,sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=probe_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"probe_status": "timeout", "error": "isolated import probe timed out"}
    except OSError as error:
        return {"probe_status": "error", "error": _safe_error(error)}

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "probe failed"
        return {"probe_status": "error", "error": _safe_error(detail)}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"probe_status": "error", "error": _safe_error(error)}
    if not isinstance(payload, dict):
        return {"probe_status": "error", "error": "probe returned non-object JSON"}
    return {"probe_status": "ok", "runtime": payload, "error": None}


def inspect_optional_package(
    module_name: str,
    distributions: tuple[str, ...],
    *,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Inspect module metadata before any isolated import probe."""
    probe_timeout_seconds = _validated_probe_timeout(probe_timeout_seconds)
    present, discovery_error = _module_present(module_name)
    version = _distribution_version(distributions)
    result: dict[str, Any] = {
        "present": present,
        "version": version,
        "probe_status": "not_found" if not present else "metadata_only",
        "error": discovery_error,
    }
    if not present:
        return result

    if module_name == "torch" or version is None:
        probe = _isolated_probe(
            module_name,
            include_cuda=module_name == "torch",
            probe_timeout_seconds=probe_timeout_seconds,
        )
        result.update(probe)
        runtime = result.get("runtime")
        if version is None and isinstance(runtime, dict):
            runtime_version = runtime.get("module_version")
            if isinstance(runtime_version, str):
                result["version"] = runtime_version
    return result


def _ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(physical_pages, int):
        return None
    return page_size * physical_pages


def doctor_report(
    *, probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Return a best-effort environment report without requiring ML packages."""
    probe_timeout_seconds = _validated_probe_timeout(probe_timeout_seconds)
    packages = {
        module: inspect_optional_package(
            module,
            distributions,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        for module, distributions in OPTIONAL_PACKAGES.items()
    }
    optional_errors = any(
        details.get("probe_status") in {"error", "timeout"} for details in packages.values()
    )
    return {
        "schema_version": "1.0",
        "status": "ok_with_optional_errors" if optional_errors else "ok",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "ram_bytes": _ram_bytes(),
        },
        "packages": packages,
    }


def format_doctor_text(report: dict[str, Any]) -> str:
    """Format the structured doctor report for human-readable CLI output."""
    python_info = report["python"]
    platform_info = report["platform"]
    lines = [
        f"status: {report['status']}",
        f"python: {python_info['version']} ({python_info['implementation']})",
        f"platform: {platform_info['system']} {platform_info['release']}",
        f"machine: {platform_info['machine']}",
        f"ram_bytes: {platform_info['ram_bytes']}",
        "optional packages:",
    ]
    packages = report["packages"]
    for name in sorted(packages):
        details = packages[name]
        lines.append(
            f"  {name}: present={details['present']} version={details['version']} "
            f"probe={details['probe_status']}"
        )
        if details.get("error"):
            lines.append(f"    error: {details['error']}")
    return "\n".join(lines)
