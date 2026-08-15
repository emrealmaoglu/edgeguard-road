"""Dependency-light connected components and road-aware statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ComponentRecord:
    """One four-connected anomaly component with explainable geometry."""

    component_id: int
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    mean_score: float
    max_score: float
    road_overlap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def road_mask_from_semantics(
    semantic_mask: npt.NDArray[np.integer], *, road_class_ids: tuple[int, ...] = (0, 1)
) -> npt.NDArray[np.bool_]:
    """Create an explicit boolean drivable mask from configured semantic IDs."""
    if not isinstance(semantic_mask, np.ndarray) or semantic_mask.ndim != 2:
        raise ValueError("semantic mask must be a two-dimensional NumPy array")
    if not np.issubdtype(semantic_mask.dtype, np.integer) or not road_class_ids:
        raise ValueError("semantic mask must be integer and road IDs cannot be empty")
    return np.isin(semantic_mask, np.asarray(road_class_ids, dtype=np.int64))


def _run_starts(counts: np.ndarray) -> np.ndarray:
    """Offset of each element within its own run, for `np.repeat`-expanded arrays."""
    total = int(counts.sum())
    ends = np.cumsum(counts)
    return np.arange(total) - np.repeat(ends - counts, counts)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Label four-connected components, numbered by raster order of first pixel.

    Deliberately a different algorithm from `rescue.perception._label_components`, which
    resolves the same problem by propagating labels through whole-array maxima. That one
    takes as many passes as a component is wide, which is cheap on the stride-8 masks the
    deployment path derives perception from (64x128, where it is the measured-faster
    option on the Jetson) and ruinous at evaluation resolution: a Cityscapes road spans
    the image, and a single 2048x1024 frame costs 76 s.

    This runs over horizontal runs instead of pixels -- a few thousand of them in a road
    mask rather than two million -- unioning runs that overlap in adjacent rows. The cost
    follows the number of runs, not the width of a component, so it does not care about
    resolution. Output is identical to the propagation labeller, which
    `test_labellers_agree` pins across random masks.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), []
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    transitions = np.diff(padded, axis=1)
    row, start = np.nonzero(transitions == 1)
    end = np.nonzero(transitions == -1)[1]

    # Runs arrive in raster order and never overlap within a row, so both key arrays are
    # globally sorted and a search for a row's neighbours cannot stray outside its block.
    stride = width + 2
    end_keys = row * stride + end
    start_keys = row * stride + start
    above = (row - 1) * stride
    first = np.searchsorted(end_keys, above + start, side="right")
    last = np.searchsorted(start_keys, above + end, side="left")
    overlaps = np.maximum(last - first, 0)
    neighbour = np.repeat(first, overlaps) + _run_starts(overlaps)
    current = np.repeat(np.arange(row.size), overlaps)

    parent = np.arange(row.size)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for below, above_run in zip(current.tolist(), neighbour.tolist(), strict=True):
        left, right = find(below), find(above_run)
        if left != right:
            parent[max(left, right)] = min(left, right)

    roots = np.array([find(index) for index in range(row.size)], dtype=np.int64)
    unique, inverse = np.unique(roots, return_inverse=True)
    # Runs are in raster order, so a component's lowest run index is its first pixel.
    first_run = np.full(unique.size, row.size, dtype=np.int64)
    np.minimum.at(first_run, inverse, np.arange(row.size))
    ranking = np.empty(unique.size, dtype=np.int32)
    ranking[np.argsort(first_run)] = np.arange(1, unique.size + 1, dtype=np.int32)

    lengths = end - start
    pixels = np.repeat(row * width + start, lengths) + _run_starts(lengths)
    flat = np.zeros(mask.size, dtype=np.int32)
    flat[pixels] = np.repeat(ranking[inverse], lengths)

    foreground = np.flatnonzero(flat)
    grouped = np.argsort(flat[foreground], kind="stable")
    boundaries = np.searchsorted(flat[foreground][grouped], np.arange(1, unique.size + 2))
    components = [
        np.stack(
            (foreground[grouped[begin:finish]] // width, foreground[grouped[begin:finish]] % width),
            axis=1,
        ).astype(np.int32)
        for begin, finish in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return flat.reshape(mask.shape), components


def connected_components(
    anomaly_mask: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.floating],
    *,
    road_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[ComponentRecord, ...]:
    """Extract deterministic four-connected components and compact statistics."""
    if not isinstance(anomaly_mask, np.ndarray) or anomaly_mask.ndim != 2:
        raise ValueError("anomaly mask must be a two-dimensional NumPy array")
    if anomaly_mask.dtype != np.bool_:
        raise TypeError("anomaly mask must use bool dtype")
    if not isinstance(scores, np.ndarray) or scores.shape != anomaly_mask.shape:
        raise ValueError("component scores must match anomaly-mask geometry")
    if not np.issubdtype(scores.dtype, np.floating) or not bool(np.isfinite(scores).all()):
        raise ValueError("component scores must be finite floating values")
    if road_mask is None:
        road = np.zeros_like(anomaly_mask)
    elif road_mask.shape != anomaly_mask.shape or road_mask.dtype != np.bool_:
        raise ValueError("road mask must be bool and match anomaly-mask geometry")
    else:
        road = road_mask
    # Every statistic below is order-invariant -- area, bbox extremes, centroid, mean and
    # max score, road fraction -- so only which pixels share a component matters, not the
    # order the search would have visited them in.
    _, components = label_components(anomaly_mask)
    records: list[ComponentRecord] = []
    for pixels in components:
        ys, xs = pixels[:, 0].astype(np.int64), pixels[:, 1].astype(np.int64)
        values = scores[ys, xs]
        records.append(
            ComponentRecord(
                component_id=len(records) + 1,
                area=int(pixels.shape[0]),
                bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                mean_score=float(values.mean()),
                max_score=float(values.max()),
                road_overlap=float(np.count_nonzero(road[ys, xs]) / pixels.shape[0]),
            )
        )
    return tuple(records)
