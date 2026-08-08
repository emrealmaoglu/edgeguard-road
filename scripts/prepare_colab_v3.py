"""Stage current Drive data and import the exact owner-approved v2 audit candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from edgeguard.rescue.colab_data import load_colab_data_access
from edgeguard.rescue.multidomain import (
    freeze_candidate_manifest_by_policy,
    validate_dataset_manifest,
    write_multidomain_statistics,
)
from edgeguard.serialization import canonical_json, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _stage_data(drive_root: Path, local_root: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/prepare_colab_data.py"),
        "--drive-root",
        str(drive_root),
        "stage",
        "--local-root",
        str(local_root),
        "--dataset",
        "cityscapes",
        "--dataset",
        "idd20k",
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def _verify_staged_data(local_root: Path) -> dict[str, list[str]]:
    """Fail before training unless every frozen train/val source directory exists."""
    plan = load_colab_data_access(REPOSITORY_ROOT / "configs/dataset/colab_data_access_v1.yaml")
    verified: dict[str, list[str]] = {}
    for dataset in ("cityscapes", "idd20k"):
        definition = plan["datasets"][dataset]
        root = local_root / str(definition["prepared_subdirectory"])
        required = [str(value) for value in definition["required_paths"]]
        missing = [relative for relative in required if not (root / relative).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"staged {dataset} is missing required train/val directories: {missing}"
            )
        verified[dataset] = required
    return verified


def _candidate_from_review_packages(
    review_root: Path, *, dataset: str, expected_sha256: str, destination: Path
) -> bool:
    for package in sorted(review_root.glob("*.zip"), reverse=True):
        try:
            with ZipFile(package) as archive:
                for name in archive.namelist():
                    if not name.endswith("dataset_manifest.candidate.json"):
                        continue
                    if f"/{dataset}/" not in f"/{name}" and not (
                        dataset == "cityscapes" and "/cityscapes/" in f"/{name}"
                    ):
                        continue
                    payload = archive.read(name)
                    if hashlib.sha256(payload).hexdigest() != expected_sha256:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    return True
        except (BadZipFile, OSError):
            continue
    return False


def _restore_v2_state(drive_root: Path, import_root: Path) -> None:
    if import_root.exists():
        return
    store = drive_root / "EdgeGuard/campaigns/semantic-cs-idd-v2/recovery/v2"
    if not (store / "pointers/campaign-state.json").is_file():
        return
    archive = import_root.parent / ".semantic-cs-idd-v2-state.tar.gz"
    restore = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/manage_colab_recovery.py"),
        "restore",
        "--store-root",
        str(store),
        "--artifact-id",
        "campaign-state",
        "--destination",
        str(archive),
    ]
    subprocess.run(restore, cwd=REPOSITORY_ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/manage_colab_recovery.py"),
            "restore-state",
            "--archive",
            str(archive),
            "--destination",
            str(import_root),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def _find_candidate(
    *,
    dataset: str,
    expected_sha256: str,
    drive_root: Path,
    import_root: Path,
    destination: Path,
) -> Path:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination
    destination.unlink(missing_ok=True)
    _restore_v2_state(drive_root, import_root)
    if import_root.is_dir():
        for candidate in sorted(import_root.glob("**/dataset_manifest.candidate.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("dataset_id") == dataset and sha256_file(candidate) == expected_sha256:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(candidate.read_bytes())
                return destination
    if _candidate_from_review_packages(
        drive_root / "EdgeGuard/review_packages",
        dataset=dataset,
        expected_sha256=expected_sha256,
        destination=destination,
    ):
        return destination
    raise FileNotFoundError(
        f"approved {dataset} audit candidate {expected_sha256} was not found in v2 recovery "
        "or Drive review packages"
    )


def prepare(
    *,
    drive_root: Path,
    local_data_root: Path,
    work_root: Path,
    policy_path: Path,
    project_commit: str,
) -> dict[str, object]:
    """Return frozen training manifests and pooled statistics without re-auditing pixels."""
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    campaign_id = str(policy["campaign_id"])
    _stage_data(drive_root, local_data_root)
    staged_required_paths = _verify_staged_data(local_data_root)
    import_root = work_root / "imports/semantic-cs-idd-v2"
    candidate_root = work_root / "imports/approved-audit-candidates"
    manifest_root = work_root / "manifests/training"
    manifests: list[Path] = []
    for dataset in ("cityscapes", "idd20k"):
        expected = policy["training_manifest_candidates"][dataset]
        candidate = _find_candidate(
            dataset=dataset,
            expected_sha256=str(expected["file_sha256"]),
            drive_root=drive_root,
            import_root=import_root,
            destination=candidate_root / dataset / "dataset_manifest.candidate.json",
        )
        frozen = manifest_root / f"{dataset}.frozen.json"
        if frozen.is_file():
            payload = validate_dataset_manifest(frozen)
            if payload.get("approved_candidate_sha256") != sha256_file(candidate):
                raise ValueError(f"existing {dataset} frozen manifest has another candidate")
        else:
            freeze_candidate_manifest_by_policy(
                candidate,
                frozen,
                policy_path=policy_path,
                campaign_id=campaign_id,
                project_commit=project_commit,
            )
        manifests.append(frozen)
    statistics = work_root / "multidomain-statistics"
    if statistics.exists():
        expected_hashes = sorted(sha256_file(path) for path in manifests)
        for name in ("class_weights.json", "rare_classes.json"):
            payload = json.loads((statistics / name).read_text(encoding="utf-8"))
            if sorted(payload.get("dataset_manifest_sha256s", [])) != expected_hashes:
                raise ValueError("existing multidomain statistics belong to other manifests")
    else:
        write_multidomain_statistics(tuple(manifests), statistics)
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_v3_data_preparation",
        "campaign_id": campaign_id,
        "data_roots": {
            dataset: str(local_data_root / dataset) for dataset in ("cityscapes", "idd20k")
        },
        "training_manifests": [str(path) for path in manifests],
        "training_manifest_sha256s": [sha256_file(path) for path in manifests],
        "rare_classes": str(statistics / "rare_classes.json"),
        "class_weights": str(statistics / "class_weights.json"),
        "reused_pixel_audits": True,
        "staged_required_paths": staged_required_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--local-data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--authorization-policy", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    args = parser.parse_args()
    result = prepare(
        drive_root=args.drive_root.resolve(),
        local_data_root=args.local_data_root.resolve(),
        work_root=args.work_root.resolve(),
        policy_path=args.authorization_policy.resolve(),
        project_commit=args.project_commit,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
