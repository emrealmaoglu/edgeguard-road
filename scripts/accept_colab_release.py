"""Promote a Colab final-model candidate using an explicit human review receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.release_acceptance import accept_release_candidate
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = accept_release_candidate(
        args.candidate.resolve(),
        args.review_receipt.resolve(),
        args.output.resolve(),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
