"""Command-line interface for repository diagnostics and smoke verification."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from edgeguard.config import load_smoke_config
from edgeguard.healthcheck import doctor_report, format_doctor_text
from edgeguard.serialization import canonical_json
from edgeguard.smoke import build_smoke_result, write_smoke_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgeguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report environment capabilities")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    smoke = subparsers.add_parser("smoke", help="run the CPU-only synthetic pipeline")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    smoke.add_argument("--output", type=Path, default=Path("artifacts/dev/smoke.jsonl"))
    smoke.add_argument("--deterministic", action="store_true")
    return parser


def _run_doctor(as_json: bool) -> int:
    try:
        report = doctor_report()
    except Exception as error:  # preserve parseable output for unexpected local failures
        report = {
            "schema_version": "1.0",
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    if as_json:
        print(canonical_json(report))
    elif report.get("status") == "error":
        print(f"status: error\nerror: {report.get('error', 'unknown error')}")
    else:
        print(format_doctor_text(report))
    return 0 if report.get("status") != "error" else 1


def _run_smoke(config_path: Path, output_path: Path, deterministic: bool) -> int:
    try:
        config = load_smoke_config(config_path)
        result = build_smoke_result(
            config,
            config_path=config_path,
            deterministic=deterministic,
        )
        write_smoke_result(result, output_path)
    except (OSError, ValueError, ValidationError) as error:
        print(f"smoke failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    print(f"wrote: {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the EdgeGuard-Road command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(args.as_json)
    if args.command == "smoke":
        return _run_smoke(args.config, args.output, args.deterministic)
    raise AssertionError(f"unhandled command: {args.command}")
