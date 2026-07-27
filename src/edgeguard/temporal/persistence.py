"""Deterministic lightweight temporal component persistence."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from edgeguard.evaluation.components import ComponentRecord


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    track_id: int
    component: ComponentRecord
    persistence: int
    smoothed_score: float
    missed_frames: int = 0


class TemporalPersistence:
    """Associate adjacent components by IoU or centroid distance with bounded gaps."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.2,
        centroid_distance: float = 20.0,
        smoothing: float = 0.6,
        missed_frame_tolerance: int = 1,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0 or centroid_distance < 0.0:
            raise ValueError("temporal association thresholds are invalid")
        if not 0.0 <= smoothing < 1.0 or missed_frame_tolerance < 0:
            raise ValueError("temporal smoothing and missed-frame tolerance are invalid")
        self.iou_threshold = iou_threshold
        self.centroid_distance = centroid_distance
        self.smoothing = smoothing
        self.missed_frame_tolerance = missed_frame_tolerance
        self._sequence_id: str | None = None
        self._frame_index: int | None = None
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def reset(self, sequence_id: str) -> None:
        """Start a new sequence and deterministically discard earlier tracks."""
        if not sequence_id:
            raise ValueError("sequence_id cannot be empty")
        self._sequence_id = sequence_id
        self._frame_index = None
        self._next_track_id = 1
        self._tracks.clear()

    def update(
        self, sequence_id: str, frame_index: int, components: tuple[ComponentRecord, ...]
    ) -> tuple[dict[str, Any], ...]:
        """Update tracks and return explanation records for visible components."""
        if self._sequence_id != sequence_id:
            self.reset(sequence_id)
        if frame_index < 0 or (self._frame_index is not None and frame_index <= self._frame_index):
            raise ValueError("frame indices must increase strictly within a sequence")
        gap = 1 if self._frame_index is None else frame_index - self._frame_index
        for track in self._tracks.values():
            track.missed_frames += gap
        unmatched_tracks = set(self._tracks)
        records: list[dict[str, Any]] = []
        for component in sorted(components, key=lambda item: item.component_id):
            candidates: list[tuple[float, float, int]] = []
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                iou = _box_iou(track.component.bbox_xyxy, component.bbox_xyxy)
                distance = hypot(
                    track.component.centroid_xy[0] - component.centroid_xy[0],
                    track.component.centroid_xy[1] - component.centroid_xy[1],
                )
                if iou >= self.iou_threshold or distance <= self.centroid_distance:
                    candidates.append((-iou, distance, track_id))
            if candidates:
                _negative_iou, _distance, track_id = min(candidates)
                track = self._tracks[track_id]
                track.persistence += 1
                track.smoothed_score = (
                    self.smoothing * track.smoothed_score
                    + (1.0 - self.smoothing) * component.mean_score
                )
                track.component = component
                track.missed_frames = 0
                unmatched_tracks.remove(track_id)
                event = "matched"
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(track_id, component, 1, component.mean_score)
                self._tracks[track_id] = track
                event = "appeared"
            records.append(
                {
                    "track_id": track_id,
                    "component_id": component.component_id,
                    "persistence_count": track.persistence,
                    "smoothed_score": track.smoothed_score,
                    "event": event,
                    "association": "iou_or_centroid",
                }
            )
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.missed_frame_tolerance
        ]
        for track_id in expired:
            del self._tracks[track_id]
        self._frame_index = frame_index
        return tuple(records)

    def snapshot(self) -> dict[str, Any]:
        """Return enough state for exact local interruption recovery."""
        return {
            "sequence_id": self._sequence_id,
            "frame_index": self._frame_index,
            "next_track_id": self._next_track_id,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "component": track.component.to_dict(),
                    "persistence": track.persistence,
                    "smoothed_score": track.smoothed_score,
                    "missed_frames": track.missed_frames,
                }
                for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            ],
        }
