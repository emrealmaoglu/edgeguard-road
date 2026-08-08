"""Resolve the exact successful hermetic Colab runtime from its evidence receipt."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from edgeguard.serialization import sha256_file


def resolve_colab_runtime(
    receipt_path: Path,
    *,
    expected_project_commit: str,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Validate and resolve the one hash-locked interpreter and checkout."""
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("record_type") != "semantic_hermetic_runtime_receipt":
        raise ValueError("invalid semantic hermetic runtime receipt")
    if payload.get("project_commit") != expected_project_commit:
        raise ValueError("runtime receipt belongs to another project commit")
    if payload.get("runtime_profile") != "py311-cu121":
        raise ValueError("runtime receipt has no supported profile")
    contract = payload.get("runtime_contract")
    environment = payload.get("environment")
    if not isinstance(contract, dict) or not isinstance(environment, dict):
        raise ValueError("runtime receipt omits runtime or environment evidence")
    runtime_root = Path(str(contract["runtime_root"]))
    checkout_root = Path(str(contract["checkout_root"]))
    interpreter = runtime_root / "bin/python"
    checkout = checkout_root / "mmsegmentation"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise FileNotFoundError(f"selected semantic interpreter is missing: {interpreter}")
    if not checkout.is_dir():
        raise FileNotFoundError(f"selected MMSeg checkout is missing: {checkout}")
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != payload.get("framework_commit"):
        raise ValueError("selected MMSeg checkout commit does not match receipt")
    if require_cuda and environment.get("cuda_available") is not True:
        raise ValueError("selected semantic runtime has no verified CUDA")
    probe = payload.get("core_model_probe")
    if not isinstance(probe, dict) or (
        probe.get("model_count") != 5
        or probe.get("checkpoint_resume_verified") is not True
        or probe.get("fp16_finite_model_count") != 5
    ):
        raise ValueError("selected runtime lacks the five-model reload gate")
    return {
        "schema_version": "2.0",
        "record_type": "edgeguard_resolved_colab_runtime",
        "runtime_profile": "py311-cu121",
        "interpreter": str(interpreter),
        "mmseg_root": str(checkout),
        "project_commit": expected_project_commit,
        "framework_commit": commit,
        "environment": environment,
        "runtime_receipt_sha256": sha256_file(receipt_path),
    }
