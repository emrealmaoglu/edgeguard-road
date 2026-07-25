"""Tests for Git provenance in temporary repositories."""

import subprocess
from pathlib import Path

from edgeguard.contracts import GitState
from edgeguard.provenance import detect_git_provenance, experiment_fingerprint


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


def test_git_state_unavailable(tmp_path: Path) -> None:
    provenance = detect_git_provenance(tmp_path)

    assert provenance.state is GitState.UNAVAILABLE
    assert provenance.commit is None
    assert provenance.dirty is None


def test_git_states_unborn_clean_and_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "test")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")

    unborn = detect_git_provenance(repo)
    assert unborn.state is GitState.UNBORN
    assert unborn.commit is None
    assert unborn.dirty is True

    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=EdgeGuard Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    clean = detect_git_provenance(repo)
    assert clean.state is GitState.CLEAN
    assert clean.commit is not None
    assert len(clean.commit) == 40
    assert clean.dirty is False

    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    dirty = detect_git_provenance(repo)
    assert dirty.state is GitState.DIRTY
    assert dirty.commit == clean.commit
    assert dirty.dirty is True


def test_fingerprint_includes_explicit_git_state(tmp_path: Path) -> None:
    unavailable = detect_git_provenance(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "test")
    unborn = detect_git_provenance(repo)

    common = {
        "config_sha256": "a" * 64,
        "contract_version": "1.0",
        "pipeline_name": "dummy-smoke",
        "backend": "dummy",
        "scorer": "dummy-normalized-magnitude",
    }
    assert experiment_fingerprint(**common, git=unavailable) != experiment_fingerprint(
        **common, git=unborn
    )
