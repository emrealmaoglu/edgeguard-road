#!/usr/bin/env python3
"""Inventory, bundle, and safely stage approved datasets for Colab."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_data import (
    create_dataset_bundle,
    initialize_drive_layout,
    inventory_colab_data,
    load_colab_data_access,
    stage_dataset_bundles,
)
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/dataset/colab_data_access_v1.yaml"),
    )
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("initialize")
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--hash-archives", action="store_true")
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--dataset", action="append", required=True)
    bundle.add_argument("--replace", action="store_true")
    stage = subparsers.add_parser("stage")
    stage.add_argument("--dataset", action="append", required=True)
    stage.add_argument("--local-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = load_colab_data_access(args.plan)
    if args.command == "initialize":
        result: object = initialize_drive_layout(plan, args.drive_root)
    elif args.command == "inventory":
        result = inventory_colab_data(plan, args.drive_root, hash_archives=args.hash_archives)
    elif args.command == "bundle":
        result = {
            "bundles": [
                create_dataset_bundle(plan, args.drive_root, dataset, replace=args.replace)
                for dataset in args.dataset
            ]
        }
    else:
        result = stage_dataset_bundles(
            plan,
            args.drive_root,
            args.local_root,
            tuple(args.dataset),
        )
    rendered = canonical_json(result) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
