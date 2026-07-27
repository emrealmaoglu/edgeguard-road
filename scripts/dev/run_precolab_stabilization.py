"""Build the deterministic deployment fixture and final pre-Colab review evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.deployment.precolab import build_precolab_evidence
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_precolab_evidence(
        args.repository.resolve(),
        closure_root=args.closure_root.resolve(),
        equivalence_report_path=args.equivalence_report.resolve(),
        output_root=args.output_root.resolve(),
        expected_commit=args.expected_commit,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
