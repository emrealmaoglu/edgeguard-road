"""Tests for independent base and smoke configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeguard.config import (
    config_sha256,
    load_base_config,
    load_pidnet_spike_config,
    load_smoke_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_base_and_smoke_configs_validate_independently() -> None:
    base = load_base_config(REPO_ROOT / "configs/base.yaml")
    smoke = load_smoke_config(REPO_ROOT / "configs/smoke.yaml")

    assert base.project_name == "edgeguard-road"
    assert smoke.input.channels == 3
    assert smoke.model.num_classes == 4


def test_config_hash_is_stable() -> None:
    config = load_smoke_config(REPO_ROOT / "configs/smoke.yaml")

    assert config_sha256(config) == config_sha256(config.model_copy(deep=True))
    assert len(config_sha256(config)) == 64


def test_pidnet_spike_config_is_complete_and_pinned() -> None:
    config = load_pidnet_spike_config(REPO_ROOT / "configs/pidnet_spike.yaml")

    assert config.upstream.commit == "4c158cf24ce432f0a8cb43364fae38d93cee0dc3"
    assert config.checkpoint.filename == "PIDNet_S_Cityscapes_val.pt"
    assert config.checkpoint.sha256 == (
        "b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01"
    )
    assert config.checkpoint.source_url.startswith("https://drive.google.com/drive/folders/")
    assert config.checkpoint.license_status == "OPEN QUESTION"
    assert config.sample.primary.relative_path == (
        "samples/frankfurt_000000_002196_leftImg8bit.png"
    )
    assert config.sample.primary.sha256 == (
        "78c65d3055fbd62e41d066813132c971a85dcdea4e5ef5459bad410bccead246"
    )
    assert config.sample.dataset_role == "plumbing_only"
    assert config.input.model_dump() == {
        "batch_size": 1,
        "height": 512,
        "width": 1024,
        "channels": 3,
        "color_space": "RGB",
    }
    assert config.model.augment is False
    assert config.scorers == ("msp", "predictive_entropy")


def test_smoke_config_rejects_duplicate_root_seed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-root.yaml"
    source = (REPO_ROOT / "configs/smoke.yaml").read_text(encoding="utf-8")
    duplicate.write_text(source + "seed: 7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate YAML key 'seed'"):
        load_smoke_config(duplicate)


def test_smoke_config_rejects_duplicate_nested_height(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-nested.yaml"
    source = (REPO_ROOT / "configs/smoke.yaml").read_text(encoding="utf-8")
    duplicate.write_text(
        source.replace("  height: 32\n", "  height: 32\n  height: 64\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key 'height'"):
        load_smoke_config(duplicate)


def test_smoke_config_rejects_inheritance_field(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """\
schema_version: "1.0"
project_name: edgeguard-road
contract_version: "1.0"
record_schema_version: "1.0"
extends: base.yaml
pipeline_name: dummy-smoke
seed: 1
input: {batch_size: 1, height: 8, width: 8, channels: 3}
model: {backend: dummy, num_classes: 2}
scorer: {name: dummy-normalized-magnitude, epsilon: 1.0e-6}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_smoke_config(invalid)


def test_invalid_config_fails_fast(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: '1.0'\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_smoke_config(invalid)
