"""Install the pinned OpenMMLab stack into an isolated Colab runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json
from edgeguard.training.config import load_semantic_framework_config
from edgeguard.training.contracts import SemanticFrameworkConfig


def build_install_commands(
    config: SemanticFrameworkConfig, checkout: Path
) -> tuple[tuple[str, ...], ...]:
    """Build exact runtime-aware commands without a hard-coded CUDA wheel URL."""
    return (
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            f"openmim=={config.openmim_version}",
        ),
        (
            sys.executable,
            "-m",
            "mim",
            "install",
            f"mmengine=={config.mmengine_version}",
            f"mmcv=={config.mmcv_version}",
        ),
        (
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            config.repository_url,
            str(checkout),
        ),
        ("git", "-C", str(checkout), "checkout", "--detach", config.commit),
        (sys.executable, "-m", "pip", "install", "-e", str(checkout)),
    )


def _runtime_summary() -> dict[str, Any]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Colab runtime must provide PyTorch before stack installation"
        ) from error
    import torch

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _run(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


def install_stack(config_path: Path, checkout: Path) -> dict[str, Any]:
    """Install, verify exact checkout identity, and return a path-free receipt."""
    config = load_semantic_framework_config(config_path)
    runtime = _runtime_summary()
    if not runtime["cuda_available"]:
        raise RuntimeError("semantic stack probe requires a CUDA Colab runtime")
    if checkout.exists():
        raise ValueError("MMSegmentation checkout destination already exists")
    commands = build_install_commands(config, checkout)
    for command in commands:
        _run(command)
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != config.commit:
        raise RuntimeError("MMSegmentation checkout commit mismatch")
    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ("torch", "mmengine", "mmcv", "mmsegmentation", "openmim")
    }
    expected = {
        "mmengine": config.mmengine_version,
        "mmcv": config.mmcv_version,
        "mmsegmentation": "1.2.2",
        "openmim": config.openmim_version,
    }
    mismatches = {
        key: {"expected": value, "actual": versions[key]}
        for key, value in expected.items()
        if versions[key] != value
    }
    if mismatches:
        raise RuntimeError(f"resolved training stack mismatch: {mismatches}")
    return {
        "schema_version": "1.0",
        "record_type": "semantic_stack_install_receipt",
        "framework_commit": actual_commit,
        "versions": versions,
        "runtime": runtime,
        "cuda_wheel_policy": config.cuda_wheel_policy,
        "install_scope": config.install_scope,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_semantic_framework_config(args.config)
    if args.execute:
        result = install_stack(args.config, args.checkout)
    else:
        result = {
            "schema_version": "1.0",
            "record_type": "semantic_stack_install_plan",
            "framework_commit": config.commit,
            "commands": [
                list(command) for command in build_install_commands(config, args.checkout)
            ],
            "executes": False,
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
