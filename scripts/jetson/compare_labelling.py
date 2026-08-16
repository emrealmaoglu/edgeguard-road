"""Time both connected-component implementations on the target device.

The shipped `_label_components` is a breadth-first search. A vectorised label-propagation
version was written, measured on the development Mac at 0.4-0.7x its speed, and reverted.
That decision was made on the wrong machine: a Jetson's ARM CPU runs the Python
interpreter far slower relative to NumPy than an M-series Mac does, so a per-pixel Python
loop and a whole-array kernel do not keep their ranking across the two.

This decides it where it matters, on the masks the deployed model actually produces.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Running this as `python scripts/jetson/<name>.py` puts *this directory* on the import
# path, not the repository root, so the shared runner below is not importable. The runbook
# documents exactly that invocation, so the script makes it work rather than asking the
# reader to know about PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgeguard.rescue.inference import preprocess_image
from edgeguard.rescue.perception import REGION_CLASS_IDS, _label_components
from scripts.jetson.benchmark import TensorRTTorchRunner, _images


def propagation_label_components(binary: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The reverted vectorised variant, kept here only so it can be timed on target."""
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32), []
    seeds = np.arange(1, binary.size + 1, dtype=np.int64).reshape(binary.shape)
    labels = np.where(binary, seeds, 0)
    while True:
        merged = labels.copy()
        merged[1:, :] = np.maximum(merged[1:, :], labels[:-1, :])
        merged[:-1, :] = np.maximum(merged[:-1, :], labels[1:, :])
        merged[:, 1:] = np.maximum(merged[:, 1:], labels[:, :-1])
        merged[:, :-1] = np.maximum(merged[:, :-1], labels[:, 1:])
        merged[~binary] = 0
        if np.array_equal(merged, labels):
            break
        labels = merged

    flat = labels.reshape(-1)
    foreground = np.flatnonzero(flat)
    unique, inverse = np.unique(flat[foreground], return_inverse=True)
    first_index = np.full(unique.size, flat.size, dtype=np.int64)
    np.minimum.at(first_index, inverse, foreground)
    ranking = np.empty(unique.size, dtype=np.int32)
    ranking[np.argsort(first_index)] = np.arange(1, unique.size + 1, dtype=np.int32)
    renumbered = ranking[inverse]
    ordered = np.zeros(flat.size, dtype=np.int32)
    ordered[foreground] = renumbered
    grouped = np.argsort(renumbered, kind="stable")
    boundaries = np.searchsorted(renumbered[grouped], np.arange(1, unique.size + 2))
    width = binary.shape[1]
    components = [
        np.stack(
            (foreground[grouped[a:b]] // width, foreground[grouped[a:b]] % width), axis=1
        ).astype(np.int32)
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return ordered.reshape(binary.shape), components


def bfs_label_components(binary: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Call the shipped implementation under an explicit name."""
    return _label_components(binary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    images = _images(args.image_root)
    runner = TensorRTTorchRunner(args.engine)
    input_height, input_width = runner.input_shape[2:]

    masks: list[np.ndarray] = []
    for index in range(args.frames):
        with Image.open(images[index % len(images)]) as opened:
            tensor = preprocess_image(opened.convert("RGB"), (input_height, input_width))
        logits, _ = runner.infer(tensor)
        masks.append(np.argmax(logits[0], axis=0).astype(np.uint8))

    # `derive_perception` labels the road plus every attention class, so time the same set.
    binaries = [mask == 0 for mask in masks]
    binaries += [mask == class_id for mask in masks for class_id in REGION_CLASS_IDS]

    results: dict[str, float] = {}
    implementations = (
        ("bfs (shipped)", bfs_label_components),
        ("propagation", propagation_label_components),
    )
    for name, function in implementations:
        function(binaries[0])
        start = time.perf_counter()
        for binary in binaries:
            function(binary)
        results[name] = (time.perf_counter() - start) / len(binaries) * 1000.0

    # Equal output is the precondition for the timing to mean anything.
    for binary in binaries[:50]:
        shipped_labels, shipped_components = bfs_label_components(binary)
        other_labels, other_components = propagation_label_components(binary)
        assert np.array_equal(shipped_labels, other_labels), "labellings diverge"
        assert len(shipped_components) == len(other_components), "component counts diverge"

    print(f"{len(binaries)} maske, {masks[0].shape[0]}x{masks[0].shape[1]}")
    for name, milliseconds in results.items():
        print(f"  {name:16s} {milliseconds:7.3f} ms/çağrı")
    fastest = min(results, key=lambda key: results[key])
    slowest = max(results, key=lambda key: results[key])
    print(f"\nkazanan: {fastest}  ({results[slowest] / results[fastest]:.2f}x)")
    per_frame = len(REGION_CLASS_IDS) + 1
    print(f"kare başına {per_frame} çağrı -> {results[fastest] * per_frame:.1f} ms (kazanan)")
    print(f"                            -> {results[slowest] * per_frame:.1f} ms (kaybeden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
