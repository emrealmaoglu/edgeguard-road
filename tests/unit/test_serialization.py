"""Tests for canonical JSON and hashing."""

import json
from pathlib import Path

import numpy as np
import pytest

from edgeguard.serialization import canonical_json, sha256_array, sha256_file, sha256_payload


def test_canonical_json_has_stable_order_and_separators() -> None:
    payload = {"z": 1, "a": "ç", "nested": {"b": 2, "a": 1}}

    encoded = canonical_json(payload)

    assert encoded == '{"a":"ç","nested":{"a":1,"b":2},"z":1}'
    assert json.loads(encoded) == payload
    assert sha256_payload(payload) == sha256_payload({"nested": {"a": 1, "b": 2}, "a": "ç", "z": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": value})


def test_file_and_array_hashes_are_stable_and_content_sensitive(tmp_path: Path) -> None:
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"approved-checkpoint-fixture")
    array = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)

    assert sha256_file(file_path) == sha256_file(file_path)
    assert sha256_array(array) == sha256_array(array.copy())
    changed = array.copy()
    changed[0, 0, 0, 0] = 99.0
    assert sha256_array(array) != sha256_array(changed)
