"""Derive Cityscapes `labelTrainIds` masks from the `labelIds` the archive ships.

`gtFine_trainvaltest.zip` contains 34-class `labelIds`, while every dataset path in this
project reads the 19-class `labelTrainIds` that `cityscapesscripts` normally generates as
a separate preparation step. Getting that mapping wrong silently corrupts every mIoU it
touches, so it is taken from `cityscapesscripts.helpers.labels` rather than transcribed,
and checked against real data before a single mask is written: ACDC ships `labelIds` and
`labelTrainIds` side by side for the same frames, so applying this mapping to its
`labelIds` must reproduce its `labelTrainIds` exactly.

Without a validation set the conversion still runs, but it says so.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.serialization import canonical_json

TRAIN_ID_SUFFIX = "_labelTrainIds.png"
LABEL_ID_SUFFIX = "_labelIds.png"


def build_lookup() -> np.ndarray:
    """Return a 256-entry id -> train id table straight from cityscapesscripts."""
    try:
        from cityscapesscripts.helpers.labels import labels
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "cityscapesscripts is required so the mapping is authoritative rather than "
            "transcribed: pip install cityscapesscripts"
        ) from error
    lookup = np.full(256, 255, dtype=np.uint8)
    for label in labels:
        if 0 <= label.id < 256:
            lookup[label.id] = 255 if label.trainId in (255, -1) else label.trainId
    return lookup


def validate_against(reference_root: Path, lookup: np.ndarray, limit: int) -> dict[str, object]:
    """Prove the table on frames that already carry both encodings."""
    pairs = sorted(reference_root.rglob(f"*{LABEL_ID_SUFFIX}"))[:limit]
    checked = 0
    for path in pairs:
        expected_path = Path(str(path).replace(LABEL_ID_SUFFIX, TRAIN_ID_SUFFIX))
        if not expected_path.is_file():
            continue
        source = np.array(Image.open(path))
        expected = np.array(Image.open(expected_path))
        if not np.array_equal(lookup[source], expected):
            raise ValueError(f"label mapping disagrees with shipped train ids: {path}")
        checked += 1
    return {"reference_root": str(reference_root), "frames_checked": checked}


def convert(root: Path, split: str, lookup: np.ndarray, *, overwrite: bool) -> int:
    written = 0
    for path in sorted((root / "gtFine" / split).rglob(f"*{LABEL_ID_SUFFIX}")):
        target = Path(str(path).replace(LABEL_ID_SUFFIX, TRAIN_ID_SUFFIX))
        if target.is_file() and not overwrite:
            continue
        source = np.array(Image.open(path))
        if source.ndim != 2:
            raise ValueError(f"label id mask must be single channel: {path}")
        Image.fromarray(lookup[source], mode="L").save(target)
        written += 1
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--validate-against",
        type=Path,
        help="a root whose masks carry both encodings (an ACDC gt tree does)",
    )
    parser.add_argument("--validation-frames", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    lookup = build_lookup()

    validation: dict[str, object]
    if args.validate_against is not None:
        validation = validate_against(
            args.validate_against.resolve(), lookup, args.validation_frames
        )
        print(f"mapping validated on {validation['frames_checked']} frames carrying both encodings")
    else:
        validation = {"frames_checked": 0, "note": "no reference tree supplied"}
        print("WARNING: mapping not cross-checked against data")

    written = convert(args.cityscapes_root.resolve(), args.split, lookup, overwrite=args.overwrite)
    destination = args.cityscapes_root / "gtFine" / args.split
    print(f"wrote {written} {TRAIN_ID_SUFFIX} masks under {destination}")

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_cityscapes_trainid_preparation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cityscapes_root": str(args.cityscapes_root.resolve()),
        "split": args.split,
        "masks_written": written,
        "mapping_source": "cityscapesscripts.helpers.labels",
        "mapping_validated": validation,
        "trainid_classes": 19,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")
        print(f"record: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
