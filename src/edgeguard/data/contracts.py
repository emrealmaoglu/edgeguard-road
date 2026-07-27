"""Source-aware perception sample contracts shared by local validation paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SemanticSample:
    """One semantic sample without a private filesystem root."""

    image_id: str
    image: npt.NDArray[np.uint8]
    train_ids: npt.NDArray[np.uint8]
    ignore_mask: npt.NDArray[np.bool_]
    source: str
    group_id: str
    split_role: str
    city: str | None = None
    sequence_id: str | None = None

    def validate(self) -> None:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("semantic image must be positive HWC uint8 RGB")
        if any(size <= 0 for size in self.image.shape):
            raise ValueError("semantic image dimensions must be positive")
        if self.train_ids.dtype != np.uint8 or self.train_ids.shape != self.image.shape[:2]:
            raise ValueError("semantic train-ID mask must be HW uint8 and match the image")
        if self.ignore_mask.dtype != np.bool_ or self.ignore_mask.shape != self.train_ids.shape:
            raise ValueError("semantic ignore mask must be bool and match the train-ID mask")
        observed = set(int(value) for value in np.unique(self.train_ids))
        if not observed <= {*range(19), 255}:
            raise ValueError("semantic mask contains values outside 0..18 and 255")
        if not np.array_equal(self.ignore_mask, self.train_ids == 255):
            raise ValueError("semantic ignore mask does not preserve train ID 255")
        if self.split_role not in {
            "train_fit",
            "train_select",
            "train_calibration",
            "official_val_common_eval",
            "synthetic_fixture_only",
        }:
            raise ValueError("semantic split role is not approved")


@dataclass(frozen=True)
class DetectionSample:
    """One source-aware detection sample in original image coordinates."""

    image_id: str
    image: npt.NDArray[np.uint8]
    boxes_xyxy: npt.NDArray[np.float32]
    source_boxes_xyxy: npt.NDArray[np.float32]
    class_ids: npt.NDArray[np.int64]
    class_names: tuple[str, ...]
    crowd_or_ignore: npt.NDArray[np.bool_]
    source: str
    original_size_hw: tuple[int, int]
    model_input_size_hw: tuple[int, int]
    transform_receipt: dict[str, object]

    def validate(self) -> None:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("detection image must be HWC uint8 RGB")
        count = self.boxes_xyxy.shape[0]
        if self.boxes_xyxy.dtype != np.float32 or self.boxes_xyxy.shape != (count, 4):
            raise ValueError("detection boxes must be Kx4 float32")
        if self.source_boxes_xyxy.dtype != np.float32 or self.source_boxes_xyxy.shape != (count, 4):
            raise ValueError("source boxes must be Kx4 float32")
        if self.class_ids.dtype != np.int64 or self.class_ids.shape != (count,):
            raise ValueError("detection class IDs must be K int64")
        if len(self.class_names) != count or self.crowd_or_ignore.shape != (count,):
            raise ValueError("detection metadata count mismatch")
        height, width = self.original_size_hw
        boxes = self.source_boxes_xyxy
        if count and (
            not np.isfinite(boxes).all()
            or np.any(boxes[:, 2] <= boxes[:, 0])
            or np.any(boxes[:, 3] <= boxes[:, 1])
            or np.any(boxes[:, (0, 2)] < 0)
            or np.any(boxes[:, (1, 3)] < 0)
            or np.any(boxes[:, (0, 2)] > width)
            or np.any(boxes[:, (1, 3)] > height)
        ):
            raise ValueError("detection source boxes are invalid or outside the image")


@dataclass(frozen=True)
class OODSample:
    """One binary OOD evaluation sample with an explicit valid region."""

    image_id: str
    anomaly_target: npt.NDArray[np.uint8]
    valid_mask: npt.NDArray[np.bool_]
    source: str
    split_role: str
    component_metadata: tuple[dict[str, object], ...] = ()

    def validate(self) -> None:
        if self.anomaly_target.dtype != np.uint8 or self.anomaly_target.ndim != 2:
            raise ValueError("OOD target must be HW uint8")
        if self.valid_mask.dtype != np.bool_ or self.valid_mask.shape != self.anomaly_target.shape:
            raise ValueError("OOD valid mask must be bool and match the target")
        if not set(int(value) for value in np.unique(self.anomaly_target)) <= {0, 1, 255}:
            raise ValueError("OOD target must contain only 0, 1, and 255")
        if np.any(self.valid_mask & (self.anomaly_target == 255)):
            raise ValueError("ignored OOD pixels cannot be in the valid region")


@dataclass(frozen=True)
class TemporalFrame:
    """One ordered frame identity with explicit continuity metadata."""

    image_id: str
    sequence_id: str
    frame_index: int
    timestamp_seconds: float | None
    gap_from_previous: int

    def validate(self) -> None:
        if self.frame_index < 0 or self.gap_from_previous < 0:
            raise ValueError("temporal frame indices and gaps must be non-negative")
        if self.timestamp_seconds is not None and not np.isfinite(self.timestamp_seconds):
            raise ValueError("temporal timestamp must be finite")
