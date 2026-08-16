"""Look for near-duplicate frames across splits, which would inflate every number above.

Research document 18 is blunt about this: driving footage runs at 15-30 FPS, so frames a
third of a second apart are visually almost identical, and a random split puts one in
train and its neighbour in validation. The model then scores well by recognising the
background it memorised, not by segmenting. Every accuracy figure in this thesis rests on
the assumption that this did not happen.

`data/quality.py::bounded_perceptual_duplicate_pairs` was written for exactly this check
and has never been called. It hashes each image down to 64 bits of coarse structure and
reports pairs within a Hamming radius -- diagnostic, not proof of identity, which is why
the record says so and prints the pairs rather than a verdict.

Two questions are asked separately, because they have different answers:

* *within* a split -- consecutive frames of the same drive, which is expected in
  Cityscapes val (it is sequence-sampled) and harmless on its own;
* *across* splits -- the same scene appearing in both, which is the leak that would
  invalidate the comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.data.quality import bounded_perceptual_duplicate_pairs
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=ROOT",
        help="repeatable, as `val=/path/to/leftImg8bit/val`",
    )
    parser.add_argument("--per-split", type=int, default=120)
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="glob for frames; datasets do not agree on naming (Cityscapes uses "
        "`*_leftImg8bit.png`, ACDC `*_rgb_anon.png`), so a plain `*.png` is the "
        "default and the record stores what was used",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=4,
        help="Hamming radius on the 64-bit average hash. 0 finds byte-identical structure; "
        "a small radius finds the near-duplicates that matter here.",
    )
    parser.add_argument(
        "--sweep",
        default="0,2,4,6",
        help="radii to report alongside the main one. A single radius cannot be "
        "interpreted on its own: real duplicates appear at 0 and stay, while a count "
        "that only grows with the radius is the hash running out of signal.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(root: Path, limit: int, prefix: str, pattern: str) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    # Evenly spaced rather than the first N: consecutive frames of one city would
    # otherwise fill the sample and answer a question nobody asked.
    candidates = sorted(root.rglob(pattern))
    if not candidates:
        raise ValueError(f"no frames under {root}")
    step = max(1, len(candidates) // limit)
    for path in candidates[::step][:limit]:
        with Image.open(path) as opened:
            images[f"{prefix}::{path.stem}"] = np.asarray(
                opened.convert("RGB").resize((256, 128), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
    return images


def main() -> int:
    args = _parser().parse_args()
    splits: dict[str, Path] = {}
    for entry in args.split:
        name, _, root = str(entry).partition("=")
        if not root:
            raise ValueError(f"--split expects NAME=ROOT, got {entry!r}")
        splits[name] = Path(root).expanduser().resolve()

    images: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, root in splits.items():
        loaded = _load(root, args.per_split, name, args.pattern)
        counts[name] = len(loaded)
        images.update(loaded)

    def _split_of(name: object) -> str:
        return str(name).split("::", 1)[0]

    sweep: dict[str, dict[str, Any]] = {}
    for radius in sorted({int(v) for v in args.sweep.split(",") if v.strip()} | {args.distance}):
        found = bounded_perceptual_duplicate_pairs(
            images, maximum_samples=len(images), distance=radius
        )
        crossing = [row for row in found if _split_of(row["left"]) != _split_of(row["right"])]
        per_split: dict[str, int] = dict.fromkeys(splits, 0)
        for row in found:
            if _split_of(row["left"]) == _split_of(row["right"]):
                per_split[_split_of(row["left"])] += 1
        sweep[str(radius)] = {
            "within_split_pairs": per_split,
            "across_split_pairs": len(crossing),
        }
        if radius == args.distance:
            pairs = found
    within: dict[str, list[dict[str, Any]]] = defaultdict(list)
    across: list[dict[str, Any]] = []
    for pair in pairs:
        left_split = str(pair["left"]).split("::", 1)[0]
        right_split = str(pair["right"]).split("::", 1)[0]
        if left_split == right_split:
            within[left_split].append(pair)
        else:
            across.append(pair)

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_split_leakage_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            name: {"root": str(root), "sampled": counts[name]} for name, root in splits.items()
        },
        "sampled_total": len(images),
        "hamming_distance": args.distance,
        "hash": "64-bit average hash on a 256x128 downscale",
        "sampling": "evenly spaced across the sorted frame list, not the first N",
        "pattern": args.pattern,
        "within_split_pairs": {name: len(rows) for name, rows in within.items()},
        "sweep": sweep,
        "across_split_pairs": len(across),
        "across_split_examples": across[:20],
        # Stated so the record is not read as a clean bill of health: a perceptual hash
        # cannot prove two frames are different scenes, only that they are not near
        # duplicates at this radius.
        "identity_proof": False,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{len(images)} kare · {len(splits)} split · Hamming ≤ {args.distance}")
    for name, count in counts.items():
        print(f"  {name:12s} {count} kare · split içi yakın-kopya {len(within.get(name, []))}")
    print("\n  yarıçap taraması (gerçek kopya 0'da görünür ve kalır):")
    for radius, row in sorted(sweep.items(), key=lambda item: int(item[0])):
        inner = " ".join(f"{k}={v}" for k, v in row["within_split_pairs"].items())
        print(f"    d≤{radius}: splitler arası {row['across_split_pairs']:4d} · {inner}")
    print(f"\n  SPLİTLER ARASI yakın-kopya (d≤{args.distance}): {len(across)}")
    for pair in across[:10]:
        print(f"    {pair['left']}  ↔  {pair['right']}  (d={pair['hamming_distance']})")
    if not across:
        print("    yok — bu radyusta splitler arası yakın-kopya bulunmadı")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
