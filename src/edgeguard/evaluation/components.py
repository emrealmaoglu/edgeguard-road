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
    visited = np.zeros_like(anomaly_mask)
    height, width = anomaly_mask.shape
    records: list[ComponentRecord] = []
    for y in range(height):
        for x in range(width):
            if not anomaly_mask[y, x] or visited[y, x]:
                continue
            queue = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                current_y, current_x = queue.pop()
                pixels.append((current_y, current_x))
                for next_y, next_x in (
                    (current_y - 1, current_x),
                    (current_y + 1, current_x),
                    (current_y, current_x - 1),
                    (current_y, current_x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and anomaly_mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            ys = np.asarray([item[0] for item in pixels], dtype=np.int64)
            xs = np.asarray([item[1] for item in pixels], dtype=np.int64)
            values = scores[ys, xs]
            records.append(
                ComponentRecord(
                    component_id=len(records) + 1,
                    area=len(pixels),
                    bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                    centroid_xy=(float(xs.mean()), float(ys.mean())),
                    mean_score=float(values.mean()),
                    max_score=float(values.max()),
                    road_overlap=float(np.count_nonzero(road[ys, xs]) / len(pixels)),
                )
            )
    return tuple(records)
