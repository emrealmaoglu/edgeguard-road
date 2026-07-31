"""Measured Colab environment signatures and conservative training profiles."""

from __future__ import annotations

import hashlib
import os
import platform
from typing import Any

from edgeguard.serialization import canonical_json


def environment_signature(torch: Any) -> dict[str, Any]:
    """Bind reusable runtime evidence to binary compatibility inputs."""
    cuda = str(getattr(torch.version, "cuda", None))
    available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if available else "cpu"
    capability = list(torch.cuda.get_device_capability(0)) if available else None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": cuda,
        "gpu": gpu,
        "capability": capability,
    }
    payload["signature"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload


def select_training_profile(torch: Any, *, cpu_count: int | None = None) -> dict[str, Any]:
    """Choose a safe effective-batch-four profile without changing science settings."""
    cpu_count = max(1, cpu_count or os.cpu_count() or 1)
    available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if available else "cpu"
    memory = int(torch.cuda.get_device_properties(0).total_memory) if available else 0
    gib = memory / 1024**3
    if not available:
        device_batch, precision, profile = 1, "fp32", "cpu-debug"
    elif "T4" in gpu or gib < 20:
        device_batch, precision, profile = 1, "fp16", "t4-safe"
    elif gib < 35:
        device_batch, profile = 2, "l4-balanced"
        precision = "bf16" if bool(torch.cuda.is_bf16_supported()) else "fp16"
    else:
        device_batch, profile = 4, "a100-h100-throughput"
        precision = "bf16" if bool(torch.cuda.is_bf16_supported()) else "fp16"
    effective_batch = 4
    workers = 0 if not available else min(4, max(2, cpu_count // 2))
    return {
        "schema_version": "2.0",
        "record_type": "edgeguard_colab_training_profile",
        "profile": profile,
        "gpu": gpu,
        "gpu_memory_bytes": memory,
        "device_batch": device_batch,
        "gradient_accumulation": effective_batch // device_batch,
        "effective_batch": effective_batch,
        "workers": workers,
        "prefetch_factor": 2 if workers else None,
        "pin_memory": available,
        "precision": precision,
        "cudnn_benchmark": available,
        "tf32": available and "T4" not in gpu,
        "torch_compile": False,
    }
