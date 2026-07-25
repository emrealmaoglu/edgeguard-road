"""Tests for canonical JSON and hashing."""

import json

import pytest

from edgeguard.serialization import canonical_json, sha256_payload


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
