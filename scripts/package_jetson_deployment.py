"""Build or verify a Colab-produced Jetson deployment handoff bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edgeguard.deployment.jetson_bundle import (
    build_jetson_deployment_bundle,
    verify_jetson_deployment_bundle,
)
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--onnx", type=Path, required=True)
    build.add_argument("--ontology", type=Path, required=True)
    build.add_argument("--preprocessing", type=Path, required=True)
    build.add_argument("--model-family", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--config-sha256", required=True)
    build.add_argument("--checkpoint-sha256", required=True)
    build.add_argument("--golden-input", type=Path, required=True)
    build.add_argument("--golden-output", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "verify":
        result = verify_jetson_deployment_bundle(args.package.resolve())
    else:
        preprocessing = json.loads(args.preprocessing.read_text(encoding="utf-8"))
        result = build_jetson_deployment_bundle(
            args.output.resolve(),
            onnx_path=args.onnx.resolve(),
            ontology_path=args.ontology.resolve(),
            preprocessing=preprocessing,
            model_family=args.model_family,
            source_commit=args.source_commit,
            config_sha256=args.config_sha256,
            checkpoint_sha256=args.checkpoint_sha256,
            golden_input=np.load(args.golden_input, allow_pickle=False),
            golden_output=np.load(args.golden_output, allow_pickle=False),
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
