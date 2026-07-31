"""Strictly verify the approved PIDNet-S checkpoint without running inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from edgeguard.config import load_pidnet_spike_config
from edgeguard.models.pidnet_spike import PIDNetSpikeError, verify_pidnet_checkpoint_layout
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pidnet_spike.yaml"))
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main() -> int:
    """Validate source, checkpoint identity, layout, shapes, and strict load."""
    args = _parser().parse_args()
    try:
        config = load_pidnet_spike_config(args.config)
        report = verify_pidnet_checkpoint_layout(
            checkout=args.upstream_checkout,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=config.checkpoint.sha256,
            config=config,
        )
    except (OSError, ValueError, ValidationError, PIDNetSpikeError) as error:
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json({"status": "ok", "checkpoint": report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
