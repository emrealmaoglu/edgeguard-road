#!/usr/bin/env python3
"""Record the active Colab GPU/runtime signature and bounded training profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_performance import environment_signature, select_training_profile
from edgeguard.serialization import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch = __import__("torch")
    payload = {
        "environment": environment_signature(torch),
        "training": select_training_profile(torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
