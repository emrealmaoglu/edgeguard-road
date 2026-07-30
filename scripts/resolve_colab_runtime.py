"""Resolve the successful semantic Colab interpreter and MMSeg checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_runtime import resolve_colab_runtime
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    """Validate the compatibility evidence and emit the selected paths."""
    args = _parser().parse_args()
    result = resolve_colab_runtime(
        args.receipt.resolve(),
        expected_project_commit=args.project_commit,
        require_cuda=not args.allow_cpu,
    )
    rendered = canonical_json(result) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
