"""Atomic, lock-protected state for one EdgeGuard campaign."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from edgeguard.campaign.contracts import PROFILES, STAGE_DEPENDENCIES, topological_stages
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(repository: Path) -> tuple[str | None, str]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, "dirty" if dirty else "clean"
    except (OSError, subprocess.CalledProcessError):
        return None, "unavailable"


class Campaign:
    """Manage one bounded campaign whose JSON state is the source of truth."""

    def __init__(self, root: Path, repository: Path) -> None:
        self.root = root.resolve()
        self.repository = repository.resolve()

    @property
    def manifest_path(self) -> Path:
        return self.root / "campaign_manifest.json"

    @property
    def state_path(self) -> Path:
        return self.root / "pipeline_state.json"

    @property
    def artifact_index_path(self) -> Path:
        return self.root / "artifact_index.json"

    @property
    def decision_path(self) -> Path:
        return self.root / "decisions.jsonl"

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize all state transitions with one campaign-local file lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".campaign.lock"
        with lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def initialize(self, *, campaign_id: str | None, profile: str) -> dict[str, Any]:
        """Create the fixed directory contract and initial stage states."""
        if profile not in PROFILES:
            raise ValueError(f"unknown campaign profile: {profile}")
        with self.locked():
            existing = [path for path in self.root.iterdir() if path.name != ".campaign.lock"]
            if existing:
                raise ValueError("campaign root must be empty before initialization")
            for name in ("stages", "checkpoints", "reports", "recovery"):
                (self.root / name).mkdir()
            commit, git_state = _git_identity(self.repository)
            identifier = campaign_id or f"eg-{uuid4().hex[:12]}"
            profile_payload = PROFILES[profile].__dict__
            manifest = {
                "schema_version": "1.0",
                "record_type": "edgeguard_campaign_manifest",
                "campaign_id": identifier,
                "profile": profile,
                "git_commit": commit,
                "git_state_at_initialization": git_state,
                "profile_sha256": sha256_payload(profile_payload),
                "stage_graph_sha256": sha256_payload(STAGE_DEPENDENCIES),
                "created_at": _utc_now(),
                "scientific_evidence": not PROFILES[profile].synthetic,
            }
            stages = {
                stage: self._new_stage(identifier, stage, profile, commit)
                for stage in topological_stages()
            }
            stages["preflight"]["status"] = "ready"
            state = {
                "schema_version": "1.0",
                "record_type": "edgeguard_pipeline_state",
                "campaign_id": identifier,
                "revision": 1,
                "updated_at": _utc_now(),
                "stages": stages,
            }
            artifacts = {
                "schema_version": "1.0",
                "record_type": "edgeguard_artifact_index",
                "campaign_id": identifier,
                "artifacts": [],
            }
            atomic_write_json(self.manifest_path, manifest)
            atomic_write_json(self.state_path, state)
            atomic_write_json(self.artifact_index_path, artifacts)
            self.decision_path.touch()
            return manifest

    def _new_stage(
        self, campaign_id: str, stage_id: str, profile: str, commit: str | None
    ) -> dict[str, Any]:
        return {
            "campaign_id": campaign_id,
            "stage_id": stage_id,
            "status": "pending",
            "attempt_number": 0,
            "profile": profile,
            "git_commit": commit,
            "config_hashes": [],
            "input_artifact_hashes": [],
            "output_artifact_hashes": [],
            "environment_identity": None,
            "started_at": None,
            "ended_at": None,
            "progress": {"completed": 0, "total": 1, "percent": 0.0},
            "recovery_identity": None,
            "failure_classification": None,
            "last_error": None,
            "next_eligible_stages": [],
        }

    def load_manifest(self) -> dict[str, Any]:
        return self._read_json(self.manifest_path, "campaign manifest")

    def load_state(self) -> dict[str, Any]:
        state = self._read_json(self.state_path, "pipeline state")
        if set(state.get("stages", {})) != set(STAGE_DEPENDENCIES):
            raise ValueError("pipeline state stage graph is incompatible")
        return state

    @staticmethod
    def _read_json(path: Path, description: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{description} is missing or corrupt") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{description} must be a JSON object")
        return payload

    def verify_identity(self) -> dict[str, Any]:
        """Reject a different commit or profile contract before execution."""
        manifest = self.load_manifest()
        expected = manifest["git_commit"]
        actual, _state = _git_identity(self.repository)
        if expected is not None and actual != expected:
            raise ValueError("campaign Git commit is incompatible with the repository")
        profile = str(manifest["profile"])
        if manifest["profile_sha256"] != sha256_payload(PROFILES[profile].__dict__):
            raise ValueError("campaign profile contract is incompatible")
        return manifest

    def refresh_readiness(self, state: dict[str, Any]) -> None:
        """Derive ready stages and next-stage hints from completed dependencies."""
        stages = state["stages"]
        for stage_id, dependencies in STAGE_DEPENDENCIES.items():
            record = stages[stage_id]
            if record["status"] == "pending" and all(
                stages[dependency]["status"] == "completed" for dependency in dependencies
            ):
                record["status"] = "ready"
            record["next_eligible_stages"] = [
                candidate
                for candidate, candidate_dependencies in STAGE_DEPENDENCIES.items()
                if stage_id in candidate_dependencies
                and all(
                    dependency == stage_id or stages[dependency]["status"] == "completed"
                    for dependency in candidate_dependencies
                )
            ]

    def save_state(self, state: dict[str, Any]) -> None:
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_at"] = _utc_now()
        atomic_write_json(self.state_path, state)

    def begin_stage(self, state: dict[str, Any], stage_id: str) -> dict[str, Any]:
        record = state["stages"][stage_id]
        if record["status"] not in {"ready", "failed"}:
            raise ValueError(f"stage {stage_id} is not executable: {record['status']}")
        record.update(
            {
                "status": "running",
                "attempt_number": int(record["attempt_number"]) + 1,
                "started_at": _utc_now(),
                "ended_at": None,
                "progress": {"completed": 0, "total": 1, "percent": 0.0},
                "failure_classification": None,
                "last_error": None,
            }
        )
        record["environment_identity"] = sha256_payload(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
            }
        )
        self.save_state(state)
        return record

    def complete_stage(
        self, state: dict[str, Any], stage_id: str, receipt_path: Path
    ) -> dict[str, Any]:
        record = state["stages"][stage_id]
        digest = sha256_file(receipt_path)
        record.update(
            {
                "status": "completed",
                "ended_at": _utc_now(),
                "progress": {"completed": 1, "total": 1, "percent": 100.0},
                "output_artifact_hashes": [digest],
                "recovery_identity": None,
            }
        )
        self._index_artifact(stage_id, receipt_path, digest)
        self.refresh_readiness(state)
        self.save_state(state)
        return record

    def fail_stage(
        self,
        state: dict[str, Any],
        stage_id: str,
        classification: str,
        error: str,
        *,
        recovery_identity: str | None = None,
    ) -> None:
        record = state["stages"][stage_id]
        record.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "failure_classification": classification,
                "last_error": error[:1000],
                "recovery_identity": recovery_identity,
            }
        )
        self.save_state(state)

    def verify_completed(self, state: dict[str, Any], stage_id: str) -> bool:
        """Permit reuse only when every indexed stage artifact still matches."""
        record = state["stages"][stage_id]
        if record["status"] != "completed":
            return False
        index = self._read_json(self.artifact_index_path, "artifact index")
        entries = [item for item in index["artifacts"] if item["stage_id"] == stage_id]
        if not entries:
            raise ValueError(f"completed stage {stage_id} has no indexed artifact")
        for item in entries:
            path = self.root / item["relative_path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"completed stage artifact is missing or corrupt: {stage_id}")
        return True

    def _index_artifact(self, stage_id: str, path: Path, digest: str) -> None:
        index = self._read_json(self.artifact_index_path, "artifact index")
        relative = path.resolve().relative_to(self.root).as_posix()
        index["artifacts"] = [
            item
            for item in index["artifacts"]
            if not (item["stage_id"] == stage_id and item["relative_path"] == relative)
        ]
        index["artifacts"].append(
            {
                "stage_id": stage_id,
                "relative_path": relative,
                "sha256": digest,
                "byte_size": path.stat().st_size,
            }
        )
        index["artifacts"].sort(key=lambda item: (item["stage_id"], item["relative_path"]))
        atomic_write_json(self.artifact_index_path, index)

    def record_decision(self, stage_id: str, decision: str, rationale: str) -> None:
        """Append one non-blocking engineering decision with portable content."""
        payload = {
            "schema_version": "1.0",
            "record_type": "campaign_decision",
            "stage_id": stage_id,
            "decision": decision,
            "rationale": rationale,
            "recorded_at": _utc_now(),
        }
        descriptor = os.open(self.decision_path, os.O_APPEND | os.O_WRONLY)
        try:
            os.write(descriptor, (canonical_json(payload) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
