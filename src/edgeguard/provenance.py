"""Git source-state detection and deterministic experiment fingerprints."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from edgeguard.contracts import GitState
from edgeguard.serialization import sha256_payload


@dataclass(frozen=True)
class GitProvenance:
    """Normalized Git state for run and artifact records."""

    commit: str | None
    state: GitState
    dirty: bool | None


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )


def detect_git_provenance(repo_path: Path) -> GitProvenance:
    """Detect unborn, clean, dirty, or unavailable Git state."""
    try:
        inside = _git(repo_path, "rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return GitProvenance(None, GitState.UNAVAILABLE, None)

        status = _git(repo_path, "status", "--porcelain=v1", "--untracked-files=normal")
        if status.returncode != 0:
            return GitProvenance(None, GitState.UNAVAILABLE, None)
        has_changes = bool(status.stdout.strip())

        head = _git(repo_path, "rev-parse", "--verify", "HEAD")
        if head.returncode != 0:
            return GitProvenance(None, GitState.UNBORN, has_changes)

        commit = head.stdout.strip()
        state = GitState.DIRTY if has_changes else GitState.CLEAN
        return GitProvenance(commit, state, has_changes)
    except (OSError, subprocess.SubprocessError):
        return GitProvenance(None, GitState.UNAVAILABLE, None)


def experiment_fingerprint(
    *,
    config_sha256: str,
    contract_version: str,
    pipeline_name: str,
    backend: str,
    scorer: str,
    git: GitProvenance,
    dataset_manifest_sha256: str | None = None,
    model_artifact_sha256: str | None = None,
) -> str:
    """Hash deterministic scientific inputs and explicit Git provenance."""
    payload = {
        "backend": backend,
        "config_sha256": config_sha256,
        "contract_version": contract_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "git": {
            "commit": git.commit,
            "dirty": git.dirty,
            "state": git.state.value,
        },
        "model_artifact_sha256": model_artifact_sha256,
        "pipeline_name": pipeline_name,
        "scorer": scorer,
    }
    return sha256_payload(payload)
