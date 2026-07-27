"""Check the exact clean commit and all bounded pre-Colab evidence gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.deployment.precolab import check_precolab_readiness
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--closure-summary", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--deployment-validation", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    args = parser.parse_args()
    report = check_precolab_readiness(
        args.repository.resolve(),
        expected_commit=args.expected_commit,
        closure_summary_path=args.closure_summary.resolve(),
        equivalence_report_path=args.equivalence_report.resolve(),
        deployment_validation_path=args.deployment_validation.resolve(),
        minimum_free_gib=args.minimum_free_gib,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
