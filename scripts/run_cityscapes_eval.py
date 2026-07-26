"""Run the minimal PIDNet-S Cityscapes validation evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from edgeguard.evaluation.cityscapes_runner import run_cityscapes_evaluation
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subset-size", type=int)
    selection.add_argument("--subset-manifest", type=Path)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Parse runtime paths and emit one canonical completion record."""
    args = _parser().parse_args()
    try:
        result = run_cityscapes_evaluation(
            config_path=args.config,
            dataset_root=args.dataset_root,
            checkpoint_path=args.checkpoint,
            upstream_checkout=args.upstream_checkout,
            subset_size=args.subset_size,
            subset_manifest_path=args.subset_manifest,
            select_all=args.all,
            device=args.device,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
