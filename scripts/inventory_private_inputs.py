"""Exhaustively inventory every raw archive in a Drive `private_inputs/` folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeguard.rescue.archive_inventory import build_inventory_report, inventory_identity
from edgeguard.serialization import canonical_json, write_verified_zip

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-inputs-root", type=Path, required=True)
    parser.add_argument(
        "--output-parent",
        type=Path,
        required=True,
        help=(
            "parent directory; the report is written to "
            "<output-parent>/inventory-<identity prefix>/, so an unchanged "
            "private_inputs/ folder naturally reuses the same report across sessions"
        ),
    )
    parser.add_argument(
        "--access-plan",
        type=Path,
        default=REPOSITORY_ROOT / "configs/dataset/colab_data_access_v1.yaml",
        help="declared package/role metadata, used only to annotate results",
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=int,
        default=120,
        help="best-effort per-read guard against a stalled Drive/FUSE mount",
    )
    parser.add_argument(
        "--zip-out",
        type=Path,
        help="if set, also bundle the report into a hash-verified zip",
    )
    return parser


def main() -> int:
    """Run the exhaustive archive inventory and print the resulting report."""
    args = _parser().parse_args()
    private_inputs_root = args.private_inputs_root.resolve()
    output_parent = args.output_parent.resolve()

    identity = inventory_identity(private_inputs_root)
    output_root = output_parent / f"inventory-{identity[:16]}"
    report_path = output_root / "dataset_inventory.json"

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cache_hit = report.get("inventory_identity_sha256") == identity
    else:
        cache_hit = False

    if not cache_hit:
        report = build_inventory_report(
            private_inputs_root,
            output_root,
            access_plan_path=args.access_plan.resolve() if args.access_plan else None,
            stall_timeout_seconds=args.stall_timeout_seconds,
        )

    if args.zip_out is not None:
        members: dict[str, Path | bytes] = {
            "dataset_inventory.json": output_root / "dataset_inventory.json",
            "dataset_inventory.md": output_root / "dataset_inventory.md",
        }
        if not args.zip_out.exists():
            write_verified_zip(args.zip_out.resolve(), members)

    print(canonical_json({**report, "cache_hit": cache_hit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
