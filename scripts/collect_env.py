"""Dependency-light environment inventory that emits deterministic JSON."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence

from edgeguard.healthcheck import DEFAULT_PROBE_TIMEOUT_SECONDS, doctor_report


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect an EdgeGuard environment inventory")
    parser.add_argument(
        "--probe-timeout-seconds",
        type=_positive_timeout,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="per-package isolated import timeout (default: 20 seconds)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Collect and print one parseable environment report."""
    args = _parser().parse_args(argv)
    report = doctor_report(probe_timeout_seconds=args.probe_timeout_seconds)
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
