from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgeguard.rescue.data_catalog import (
    _download_verified_file,
    download_public_sample_bundle,
    load_a2d2_mapping,
    load_dataset_catalog,
    map_a2d2_rgb_mask,
    render_catalog_markdown,
)
from edgeguard.serialization import canonical_json

CATALOG = Path("docs/dataset_cards/catalog.json")
A2D2_MAPPING = Path("configs/dataset/a2d2_to_cityscapes19_v1.yaml")


def test_catalog_covers_active_portfolio_and_generated_markdown_is_current() -> None:
    catalog = load_dataset_catalog(CATALOG)
    assert len(catalog["datasets"]) == 9
    records = {row["dataset_id"]: row for row in catalog["datasets"]}
    assert records["wilddash2"]["access"]["sealed"] is True
    assert "sample_bundle" not in records["wilddash2"]
    assert records["a2d2"]["merge_contract"]["status"].startswith("phase2")
    assert render_catalog_markdown(catalog) == Path("docs/dataset_cards/catalog.md").read_text(
        encoding="utf-8"
    )


def test_catalog_rejects_a_sample_bundle_on_a_sealed_dataset(tmp_path: Path) -> None:
    catalog = load_dataset_catalog(CATALOG)
    mutated = deepcopy(catalog)
    records = {row["dataset_id"]: row for row in mutated["datasets"]}
    records["wilddash2"]["sample_bundle"] = records["a2d2"]["sample_bundle"]
    path = tmp_path / "catalog.json"
    path.write_text(canonical_json(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed dataset"):
        load_dataset_catalog(path)


def test_public_sample_byte_budget_blocks_before_network(tmp_path: Path) -> None:
    catalog = load_dataset_catalog(CATALOG)
    with pytest.raises(ValueError, match="maximum byte budget"):
        download_public_sample_bundle(catalog, "a2d2", tmp_path, maximum_total_bytes=1)
    with pytest.raises(PermissionError, match="no approved public"):
        download_public_sample_bundle(catalog, "cityscapes", tmp_path)


def test_verified_download_enforces_content_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"bounded-public-sample"

    class FakeResponse:
        status = 200
        url = "https://example.test/sample.bin"
        headers = {"Content-Length": str(len(payload))}

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("edgeguard.rescue.data_catalog.urllib.request.urlopen", fake_urlopen)
    item = {
        "url": "https://example.test/sample.bin",
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    destination = tmp_path / "sample.bin"
    _download_verified_file(item, destination, timeout_seconds=1.0)
    assert destination.read_bytes() == payload

    item["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="size or SHA-256"):
        _download_verified_file(item, tmp_path / "bad.bin", timeout_seconds=1.0)


def test_a2d2_mapping_is_complete_loss_aware_and_fail_closed() -> None:
    mapping = load_a2d2_mapping(A2D2_MAPPING)
    rows = mapping["classes"]
    assert len(rows) == 55
    assert sum(row["target_id"] is not None for row in rows) == 31
    mask = np.asarray([[[255, 0, 0], [255, 0, 255], [255, 0, 128]]], dtype=np.uint8)
    mapped = map_a2d2_rgb_mask(mask, mapping)
    assert mapped.tolist() == [[13, 0, 255]]
    unknown = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    with pytest.raises(ValueError, match="#010203"):
        map_a2d2_rgb_mask(unknown, mapping)
