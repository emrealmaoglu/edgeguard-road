"""Build the final five-model Colab delivery packages."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.deployment.colab_release import build_colab_release_packages
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-release", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_colab_release_packages(
        accepted_release=args.accepted_release.resolve(),
        work_root=args.work_root.resolve(),
        project_root=args.project_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
