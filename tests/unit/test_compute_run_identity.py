"""Tests for `compute_run_identity`, the pure function factored out of `train_model` so
recovery-migration tooling can recompute a run's identity without running training.

Context: a Colab session had to be force-stopped entirely; the next real session pushed
a new commit that only changed orchestration (which models a phase loops over), but the
Drive recovery store binds every checkpoint's identity to `project_commit`, so the
already-completed real screening runs looked "stale" and would have been retrained from
scratch (~20+ GPU-hours wasted). `compute_run_identity` lets a migration tool verify that
project_commit really is the only thing that changed before re-publishing the same real
checkpoint under the new commit's identity -- see scripts/migrate_recovery_identity.py.
"""

from __future__ import annotations

from pathlib import Path

from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.mmseg_runtime import compute_run_identity
from edgeguard.serialization import sha256_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/rescue/semantic_first.yaml"


def _fake_mmseg_root(tmp_path: Path) -> Path:
    """A minimal fake mmseg checkout: just enough of segformer_b0's real upstream config
    path, with a real optimizer block, for resolve_model_optimizer_defaults to read."""
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


def _identity_kwargs(tmp_path: Path, *, project_commit: str) -> dict:
    protocol = load_rescue_config(REAL_CONFIG)
    manifests = (_manifest(tmp_path, "cityscapes.json"), _manifest(tmp_path, "idd20k.json"))
    return {
        "protocol": protocol,
        "model_name": "segformer_b0",
        "stage_name": "screening",
        "mmseg_root": _fake_mmseg_root(tmp_path),
        "loss": "ce",
        "audit_report": None,
        "split_manifest": None,
        "manifests": manifests,
        "datasets": ["cityscapes", "idd20k"],
        "learning_rate": None,
        "weight_decay": None,
        "scheduler": "poly",
        "warmup_ratio": 0.03,
        "initialization": "random",
        "pretrained_manifest": None,
        "precision": "bf16",
        "max_steps": 6000,
        "scheduler_steps": 6000,
        "intentional_interrupt_optimizer_step": None,
        "project_commit": project_commit,
    }


def test_compute_run_identity_uses_the_real_native_optimizer(tmp_path: Path) -> None:
    identity = compute_run_identity(**_identity_kwargs(tmp_path, project_commit="a" * 40))
    assert identity["optimizer_type"] == "AdamW"
    assert identity["learning_rate"] == 6e-05
    assert identity["weight_decay"] == 0.01
    assert identity["model"] == "segformer_b0"
    assert identity["stage"] == "screening"
    assert identity["max_steps"] == 6000
    assert identity["project_commit"] == "a" * 40
    assert len(identity["dataset_manifest_sha256s"]) == 2


def test_compute_run_identity_changes_only_project_commit_across_a_pure_recommit(
    tmp_path: Path,
) -> None:
    """The exact real-world migration precondition: recomputing identity for two
    commits that changed nothing training-relevant must differ in project_commit only."""
    old = compute_run_identity(**_identity_kwargs(tmp_path, project_commit="a" * 40))
    new = compute_run_identity(**_identity_kwargs(tmp_path, project_commit="b" * 40))
    diff_keys = sorted(key for key in old if old[key] != new[key])
    assert diff_keys == ["project_commit"]
    assert sha256_payload(old) != sha256_payload(new)


def test_compute_run_identity_is_deterministic(tmp_path: Path) -> None:
    kwargs = _identity_kwargs(tmp_path, project_commit="c" * 40)
    first = compute_run_identity(**kwargs)
    second = compute_run_identity(**kwargs)
    assert sha256_payload(first) == sha256_payload(second)
