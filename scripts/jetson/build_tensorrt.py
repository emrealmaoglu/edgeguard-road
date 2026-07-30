"""Review or explicitly build one static TensorRT FP16 engine on the target Jetson."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="without this flag only print the reviewed command; never changes power mode",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the fixed FP16/static-shape trtexec command."""
    if args.workspace_mib <= 0:
        raise ValueError("TensorRT workspace must be positive")
    return [
        str(args.trtexec),
        f"--onnx={args.onnx.resolve()}",
        f"--saveEngine={args.engine.resolve()}",
        "--fp16",
        f"--memPoolSize=workspace:{args.workspace_mib}",
        "--skipInference",
        "--verbose",
    ]


def main() -> int:
    """Keep engine creation explicit, non-overwriting, and device-bound."""
    args = _parser().parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.engine.exists() or args.manifest.exists():
        raise FileExistsError("engine or manifest already exists; overwrite is prohibited")
    command = build_command(args)
    review = {
        "schema_version": "1.0",
        "record_type": "edgeguard_tensorrt_fp16_build_review",
        "onnx_sha256": sha256_file(args.onnx),
        "engine_filename": args.engine.name,
        "command": command,
        "fp16": True,
        "int8": False,
        "static_shape": True,
        "power_mode_changed": False,
        "system": {"platform": platform.platform(), "machine": platform.machine()},
    }
    if not args.execute:
        print(canonical_json({**review, "status": "dry_run_review"}))
        return 0
    executable = (
        shutil.which(str(args.trtexec)) if not args.trtexec.is_file() else str(args.trtexec)
    )
    if executable is None:
        raise FileNotFoundError(f"trtexec not found: {args.trtexec}")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0 or not args.engine.is_file():
        raise RuntimeError(f"TensorRT build failed: {process.stderr[-2000:]}")
    manifest = {
        **review,
        "status": "built_on_target_device",
        "engine_sha256": sha256_file(args.engine),
        "trtexec_stdout_tail": process.stdout[-4000:],
        "trtexec_stderr_tail": process.stderr[-4000:],
        "numerical_equivalence_pending": True,
        "benchmark_pending": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
