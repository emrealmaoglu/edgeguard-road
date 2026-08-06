"""Build a thesis evidence bundle from one accepted release manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.reporting.thesis_bundle import build_thesis_bundle
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_thesis_bundle(
        args.accepted_release.resolve(),
        args.output_root.resolve(),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
