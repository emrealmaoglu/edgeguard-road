#!/usr/bin/env python3
"""Validate, render, probe, and safely sample the semantic dataset portfolio."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.data_catalog import (
    download_public_sample_bundle,
    load_dataset_catalog,
    probe_official_sources,
    render_catalog_markdown,
)
from edgeguard.serialization import canonical_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded dataset-catalog command surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/dataset_cards/catalog.json"),
    )
    parser.add_argument("--write-markdown", type=Path)
    parser.add_argument("--probe-sources", action="store_true")
    parser.add_argument("--download-public-sample", choices=("a2d2",))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--maximum-sample-bytes", type=int, default=8 * 1024**2)
    return parser


def main() -> int:
    """Run only explicitly requested network or write operations."""
    args = build_parser().parse_args()
    catalog = load_dataset_catalog(args.catalog)
    result: dict[str, object] = {
        "catalog": str(args.catalog),
        "catalog_sha256": sha256_file(args.catalog),
        "dataset_count": len(catalog["datasets"]),
        "validated": True,
    }
    if args.write_markdown is not None:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.write_markdown.write_text(render_catalog_markdown(catalog), encoding="utf-8")
        result["markdown"] = str(args.write_markdown)
    if args.probe_sources:
        result["source_probes"] = probe_official_sources(
            catalog, timeout_seconds=args.timeout_seconds
        )
    if args.download_public_sample is not None:
        if args.output_root is None:
            raise SystemExit("--output-root is required for public sample acquisition")
        result["sample_receipt"] = download_public_sample_bundle(
            catalog,
            args.download_public_sample,
            args.output_root,
            maximum_total_bytes=args.maximum_sample_bytes,
            timeout_seconds=args.timeout_seconds,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
