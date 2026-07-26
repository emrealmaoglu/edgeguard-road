"""Package one completed evaluation bundle without datasets or model files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from edgeguard.serialization import canonical_json, sha256_file

REQUIRED_FILES = {
    "artifact_manifest.json",
    "dataset_manifest.json",
    "environment.json",
    "failures.jsonl",
    "run_metadata.json",
    "selection_manifest.json",
    "semantic_metrics.json",
    "uncertainty_summary.json",
}


def package_eval_artifacts(input_dir: Path, output_zip: Path) -> dict[str, object]:
    """Create a deterministic-path ZIP from one completed output directory."""
    if not input_dir.is_dir():
        raise ValueError("evaluation output directory does not exist")
    missing = sorted(name for name in REQUIRED_FILES if not (input_dir / name).is_file())
    if missing:
        raise ValueError(f"evaluation output is incomplete: missing={missing}")
    if output_zip.exists():
        raise ValueError("artifact ZIP already exists; overwrite is not permitted")
    resolved_input = input_dir.resolve()
    if output_zip.resolve().is_relative_to(resolved_input):
        raise ValueError("artifact ZIP must be outside the evaluation output directory")

    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("evaluation output must not contain symlinks")
    with ZipFile(output_zip, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(input_dir).as_posix())
    return {
        "status": "ok",
        "filename": output_zip.name,
        "size_bytes": output_zip.stat().st_size,
        "sha256": sha256_file(output_zip),
        "file_count": len(files),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser


def main() -> int:
    """Package one bundle and emit a path-free canonical result."""
    args = _parser().parse_args()
    try:
        result = package_eval_artifacts(args.input_dir, args.output_zip)
    except (OSError, ValueError) as error:
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
