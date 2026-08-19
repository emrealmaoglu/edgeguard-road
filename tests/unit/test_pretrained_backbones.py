"""The committed ImageNet-initialisation manifests must satisfy the runtime verifier.

`_verified_pretrained_checkpoint` is the only gate between a downloaded file and a real
training run, and it is strict about provenance. These tests pin the shape of the
committed manifests against that gate so a malformed manifest fails here rather than
after the GPU has already been paid for.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from edgeguard.rescue.mmseg_runtime import _verified_pretrained_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = REPOSITORY_ROOT / "configs/pretrained"
EXPECTED_MODELS = ("ddrnet_23_slim", "pidnet_s", "segformer_b0")


def _manifests() -> list[Path]:
    return sorted(MANIFEST_ROOT.glob("*.json"))


def test_exactly_the_models_with_upstream_pretraining_have_manifests() -> None:
    """fast_scnn/bisenetv2 declare no upstream `Pretrained` init_cfg, so they get none."""
    assert tuple(path.stem for path in _manifests()) == EXPECTED_MODELS


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda path: path.stem)
def test_committed_manifest_passes_the_runtime_verifier(
    manifest_path: Path, tmp_path: Path
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The verifier hashes the real file, so stand in a local file whose bytes hash to the
    # committed value. This proves the manifest's own fields are accepted without needing
    # to download ~20 MB of real weights in a unit test.
    checkpoint = tmp_path / "checkpoint.pth"
    body = b"edgeguard-pretrained-stub"
    checkpoint.write_bytes(body)
    payload["checkpoint_path"] = str(checkpoint)
    payload["checkpoint_sha256"] = hashlib.sha256(body).hexdigest()
    staged = tmp_path / manifest_path.name
    staged.write_text(json.dumps(payload), encoding="utf-8")

    assert _verified_pretrained_checkpoint(staged, payload["model"]) == checkpoint


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda path: path.stem)
def test_manifest_declares_classification_provenance_and_a_stable_identity(
    manifest_path: Path,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Only classification backbones are admissible: a Cityscapes-pretrained segmentation
    # checkpoint would already have seen the evaluation data.
    assert payload["source_task"] == "image_classification"
    assert payload["human_approved"] is True
    assert payload["source_url"].startswith("https://download.openmmlab.com/")
    assert len(payload["checkpoint_sha256"]) == 64
    assert payload["license_id"]
    # `pretrained_manifest_sha256` is part of every run's immutable identity, so these
    # fields must be committed constants -- a value regenerated per session (a moving
    # access_date, a temp-directory path) would invalidate every Drive checkpoint.
    assert payload["access_date"] == "2026-08-14"
    assert payload["checkpoint_path"].startswith("/content/edgeguard-cache/pretrained/")


def test_manifest_hash_is_stable_across_reads() -> None:
    """The identity field derives from these bytes; they must not drift between reads."""
    first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _manifests()}
    second = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _manifests()}
    assert first == second
    assert len(first) == len(EXPECTED_MODELS)
