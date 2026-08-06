"""Human-gated promotion of a hash-bound Colab final-model release candidate."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from edgeguard.rescue.colab_recovery import atomic_json
from edgeguard.serialization import sha256_file


def _verify_artifact(root: Path, payload: object, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"release candidate {label} artifact is invalid")
    relative = PurePosixPath(str(payload.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"release candidate {label} artifact path is unsafe")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError(f"release candidate {label} artifact identity mismatch")


def accept_release_candidate(
    candidate_path: Path,
    review_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Promote one exact candidate only when a human receipt approves its SHA-256."""
    if output_path.exists():
        raise FileExistsError("accepted release overwrite is not permitted")
    if output_path.parent.resolve() != candidate_path.parent.resolve():
        raise ValueError("accepted release must remain beside its release candidate")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("schema_version") != "2.0"
        or candidate.get("record_type") != "edgeguard_release_candidate"
        or candidate.get("status") != "candidate_requires_human_acceptance"
    ):
        raise ValueError("invalid Colab release candidate")
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release candidate has no artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or artifact.get("scientific_status") != "measured":
            raise ValueError("release candidate contains a non-measured artifact")
        _verify_artifact(candidate_path.parent, artifact, f"artifact[{index}]")
    models = candidate.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("release candidate has no models")
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError("release candidate model record is invalid")
        _verify_artifact(candidate_path.parent, model.get("checkpoint"), f"model[{index}]")
        _verify_artifact(
            candidate_path.parent,
            model.get("resolved_config"),
            f"model[{index}].resolved_config",
        )
    receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "2.0"
        or receipt.get("record_type") != "edgeguard_release_review_receipt"
    ):
        raise ValueError("invalid release review receipt")
    if receipt.get("decision") != "accept_release" or receipt.get("human_approved") is not True:
        raise PermissionError("release review receipt does not approve acceptance")
    for field in ("campaign_id", "project_commit"):
        if receipt.get(field) != candidate.get(field):
            raise ValueError(f"release review receipt {field} mismatch")
    if receipt.get("candidate_release_sha256") != sha256_file(candidate_path):
        raise ValueError("release review receipt candidate hash mismatch")
    reviewer = receipt.get("reviewer")
    release_id = str(receipt.get("release_id", ""))
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("release review receipt must identify the human reviewer")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", release_id) is None:
        raise ValueError("release review receipt has an unsafe release_id")
    accepted = {
        **candidate,
        "record_type": "edgeguard_accepted_release",
        "status": "accepted",
        "release_id": release_id,
        "accepted_candidate_sha256": sha256_file(candidate_path),
        "human_review_receipt_sha256": sha256_file(review_receipt_path),
        "human_reviewer": reviewer,
        "artifacts": [{**artifact, "scientific_status": "accepted"} for artifact in artifacts],
    }
    atomic_json(output_path, accepted)
    return accepted
