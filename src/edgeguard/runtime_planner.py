"""Assumption-only Colab resource planning and post-run calibration."""

from __future__ import annotations

from typing import Any


def resource_plan(
    profile: dict[str, Any], *, observed: dict[str, float] | None = None
) -> dict[str, Any]:
    """Return device/sample assumptions or measured duration after a real probe."""
    required = {
        "preferred_accelerator",
        "device_batch",
        "gradient_accumulation",
        "effective_batch",
        "workers",
        "prefetch",
    }
    if not required <= set(profile):
        raise ValueError("resource profile is incomplete")
    if profile["device_batch"] * profile["gradient_accumulation"] != profile["effective_batch"]:
        raise ValueError("resource profile changes effective batch")
    result: dict[str, Any] = {
        "profile": profile,
        "estimated_wall_time_seconds": None,
        "estimate_source": "unmeasured_assumptions_only",
        "gpu_hours": None,
    }
    if observed is not None:
        images_per_second = observed.get("images_per_second")
        samples = observed.get("planned_samples")
        if images_per_second is None or samples is None or images_per_second <= 0 or samples <= 0:
            raise ValueError("observed resource calibration is invalid")
        seconds = samples / images_per_second
        result.update(
            {
                "estimated_wall_time_seconds": seconds,
                "estimate_source": "measured_first_real_run",
                "gpu_hours": seconds / 3600,
            }
        )
    return result
