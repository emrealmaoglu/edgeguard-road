"""Dependency-neutral validation for framework native-logit probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def validate_native_logits_tensor(
    value: Any,
    *,
    is_tensor: Callable[[Any], bool],
) -> tuple[int, int, int, int]:
    """Require a real backend tensor with direct NCHW 19-class output."""
    if not is_tensor(value):
        raise ValueError("native logits must be a real framework tensor")
    shape = tuple(int(dimension) for dimension in value.shape)
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"native logits must have positive NCHW shape, got {shape}")
    if shape[1] != 19:
        raise ValueError(f"native logits class dimension must be 19, got {shape[1]}")
    return shape[0], shape[1], shape[2], shape[3]
