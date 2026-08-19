"""Tests for `migrate_recovery_identity`: the verify-before-republish tool that lets a
real Drive checkpoint survive a commit that changed nothing about how it was trained.

See `docs/AI_USAGE_LOG.md`'s 2026-08-13 entries for the real incident this exists to
fix: a screening-model-scope commit invalidated its own prior fix, because
project_commit is baked into every run's immutable identity hash.
"""

from __future__ import annotations

from pathlib import Path

from edgeguard.rescue.colab_recovery import peek_recovery_receipt, publish_recovery_file
from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.mmseg_runtime import compute_run_identity, migrate_recovery_identity
from edgeguard.serialization import sha256_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/rescue/semantic_first.yaml"


def _fake_mmseg_root(tmp_path: Path) -> Path:
    mmseg_root = tmp_path / "mmsegmentation"
    upstream = mmseg_root / "configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(
        "optim_wrapper = dict(optimizer=dict(type='AdamW', lr=6e-05, weight_decay=0.01))\n",
        encoding="utf-8",
    )
    return mmseg_root


def _manifest(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text('{"dataset_id": "' + name + '"}\n', encoding="utf-8")
    return path


def _common(tmp_path: Path) -> dict:
    protocol = load_rescue_config(REAL_CONFIG)
    manifests = (_manifest(tmp_path, "cityscapes.json"), _manifest(tmp_path, "idd20k.json"))
    return {
        "protocol": protocol,
        "mmseg_root": _fake_mmseg_root(tmp_path),
        "manifests": manifests,
        "datasets": ["cityscapes", "idd20k"],
        "model_name": "segformer_b0",
        "stage_name": "screening",
        "loss": "ce",
        "max_steps": 6000,
        "precision": "bf16",
    }


def _publish_fake_checkpoint(
    tmp_path: Path, recovery_root: Path, *, project_commit: str, common: dict
) -> str:
    """Publish a fake checkpoint under `project_commit`, with a genuinely correct
    identity_sha256 (as a real EdgeGuardRecoveryHook publish would)."""
    identity = compute_run_identity(
        common["protocol"],
        model_name=common["model_name"],
        stage_name=common["stage_name"],
        mmseg_root=common["mmseg_root"],
        loss=common["loss"],
        audit_report=None,
        split_manifest=None,
        manifests=common["manifests"],
        datasets=common["datasets"],
        learning_rate=None,
        weight_decay=None,
        scheduler="poly",
        warmup_ratio=0.03,
        initialization="random",
        pretrained_manifest=None,
        precision=common["precision"],
        max_steps=common["max_steps"],
        scheduler_steps=common["max_steps"],
        intentional_interrupt_optimizer_step=None,
        project_commit=project_commit,
    )
    identity_sha256 = sha256_payload(identity)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"real-enough-checkpoint-bytes")
    publish_recovery_file(
        checkpoint,
        recovery_root,
        artifact_id="screening-segformer-b0-ce",
        campaign_id="semantic-cs-idd-v3",
        project_commit=project_commit,
        metadata={"identity_sha256": identity_sha256},
    )
    return identity_sha256


def test_migrate_reports_no_existing_pointer_when_nothing_was_ever_published(
    tmp_path: Path,
) -> None:
    result = migrate_recovery_identity(
        recovery_root=tmp_path / "recovery",
        campaign_id="semantic-cs-idd-v3",
        new_project_commit="b" * 40,
        execute=True,
        **_common(tmp_path),
    )
    assert result["status"] == "no_existing_pointer"


def test_migrate_reports_already_current_when_commit_matches(tmp_path: Path) -> None:
    common = _common(tmp_path)
    recovery_root = tmp_path / "recovery"
    _publish_fake_checkpoint(tmp_path, recovery_root, project_commit="a" * 40, common=common)
    result = migrate_recovery_identity(
        recovery_root=recovery_root,
        campaign_id="semantic-cs-idd-v3",
        new_project_commit="a" * 40,
        execute=True,
        **common,
    )
    assert result["status"] == "already_current"


def test_migrate_dry_run_verifies_without_publishing(tmp_path: Path) -> None:
    common = _common(tmp_path)
    recovery_root = tmp_path / "recovery"
    _publish_fake_checkpoint(tmp_path, recovery_root, project_commit="a" * 40, common=common)
    result = migrate_recovery_identity(
        recovery_root=recovery_root,
        campaign_id="semantic-cs-idd-v3",
        new_project_commit="b" * 40,
        execute=False,
        **common,
    )
    assert result["status"] == "verified_dry_run"
    # Still points at the old commit -- dry run must not have published anything.
    receipt = peek_recovery_receipt(recovery_root, artifact_id="screening-segformer-b0-ce")
    assert receipt is not None
    assert receipt["project_commit"] == "a" * 40


def test_migrate_execute_republishes_the_same_bytes_under_the_new_commit(tmp_path: Path) -> None:
    common = _common(tmp_path)
    recovery_root = tmp_path / "recovery"
    old_identity_sha256 = _publish_fake_checkpoint(
        tmp_path, recovery_root, project_commit="a" * 40, common=common
    )
    original_receipt = peek_recovery_receipt(recovery_root, artifact_id="screening-segformer-b0-ce")
    assert original_receipt is not None
    original_object_sha256 = original_receipt["sha256"]

    result = migrate_recovery_identity(
        recovery_root=recovery_root,
        campaign_id="semantic-cs-idd-v3",
        new_project_commit="b" * 40,
        execute=True,
        **common,
    )
    assert result["status"] == "migrated"
    assert result["old_identity_sha256"] == old_identity_sha256
    receipt = peek_recovery_receipt(recovery_root, artifact_id="screening-segformer-b0-ce")
    assert receipt is not None
    assert receipt["project_commit"] == "b" * 40
    assert receipt["metadata"]["identity_sha256"] == result["new_identity_sha256"]
    assert receipt["metadata"]["migrated_from_project_commit"] == "a" * 40
    # Content-addressed and byte-identical: republishing must reuse the same real object.
    assert receipt["sha256"] == original_object_sha256


def test_migrate_refuses_when_recorded_identity_does_not_match_reality(tmp_path: Path) -> None:
    common = _common(tmp_path)
    recovery_root = tmp_path / "recovery"
    checkpoint = tmp_path / "tampered_checkpoint.pth"
    checkpoint.write_bytes(b"tampered")
    publish_recovery_file(
        checkpoint,
        recovery_root,
        artifact_id="screening-segformer-b0-ce",
        campaign_id="semantic-cs-idd-v3",
        project_commit="a" * 40,
        metadata={"identity_sha256": "not-the-real-hash"},
    )
    result = migrate_recovery_identity(
        recovery_root=recovery_root,
        campaign_id="semantic-cs-idd-v3",
        new_project_commit="b" * 40,
        execute=True,
        **common,
    )
    assert result["status"] == "verification_failed"
    receipt = peek_recovery_receipt(recovery_root, artifact_id="screening-segformer-b0-ce")
    assert receipt is not None
    assert receipt["project_commit"] == "a" * 40
