"""Assemble the results tree the presentation panel reads on the Jetson.

The panel takes one directory and renders whatever records it finds. Getting that
directory onto the device was an ad-hoc `tar` invocation typed out each time, which is
fine until a new measurement lands the night before a recording and the copy silently
misses it -- the panel does not fail in that case, it just quietly shows one fewer table,
which is the worst way to lose a result.

So the bundle is built from a declared layout: every group the panel loads is named here,
the manifest records what was found and what was not, and `--require` turns a specific
absence into an error instead of a silent gap.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from edgeguard.serialization import canonical_json, sha256_file

# Every directory `presentation_app.load_group` reads, plus the loose records and the
# figure and video trees the pages show. Names here are the names the panel expects.
GROUPS = (
    "accuracy",
    "acdc",
    "drivable",
    "figures",
    "jetson",
    "open_set",
    "profile",
    "qualitative",
    "risk",
)
LOOSE_RECORDS = (
    "candidate_table.json",
    "cityscapes_trainid_prep.json",
    "logit_stride_tradeoff.json",
    "shift_response.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--figures",
        type=Path,
        help="tree of rendered thesis figures, copied in as `thesis_figures`",
    )
    parser.add_argument("--video", type=Path, help="demo video tree, copied in as `video`")
    parser.add_argument("--output", type=Path, required=True, help="bundle directory to build")
    parser.add_argument("--archive", type=Path, help="also write a .tgz of the bundle")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "fail if this group is empty or missing. Use it for the results the "
            "presentation cannot be given without."
        ),
    )
    return parser


def _copy_tree(source: Path, destination: Path) -> int:
    """Copy a tree, dropping the AppleDouble sidecars macOS archives carry."""
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir() or path.name.startswith("._") or path.name == ".DS_Store":
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def main() -> int:
    args = _parser().parse_args()
    results = args.results.resolve()
    if not results.is_dir():
        raise ValueError(f"results tree not found: {results}")
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    contents: dict[str, int] = {}
    for group in GROUPS:
        contents[group] = _copy_tree(results / group, output / group)
    for name in LOOSE_RECORDS:
        source = results / name
        if source.is_file():
            shutil.copy2(source, output / name)
            contents[name] = 1
        else:
            contents[name] = 0
    for option, destination in ((args.figures, "thesis_figures"), (args.video, "video")):
        contents[destination] = _copy_tree(option.resolve(), output / destination) if option else 0

    missing = sorted(name for name in args.require if not contents.get(name))
    manifest = {
        "schema_version": "1.0",
        "record_type": "edgeguard_presentation_bundle_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(results),
        "file_counts": contents,
        "present": sorted(name for name, count in contents.items() if count),
        # Named explicitly: a panel with a missing group renders a smaller page rather
        # than failing, so the absence has to be visible somewhere that is read.
        "absent": sorted(name for name, count in contents.items() if not count),
        "required_and_missing": missing,
    }
    (output / "bundle_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    if missing:
        raise ValueError(f"required results are missing from the bundle: {', '.join(missing)}")

    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            archive.add(output, arcname=output.name)

    total = sum(contents.values())
    print(f"paket: {output}  ({total} dosya)")
    for name in sorted(contents):
        mark = "·" if contents[name] else "—"
        print(f"  {mark} {name:26s} {contents[name] or ''}")
    if manifest["absent"]:
        print("\nyok: " + ", ".join(manifest["absent"]))
    if args.archive:
        print(f"\narşiv: {args.archive}  sha256 {sha256_file(args.archive)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
