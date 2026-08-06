"""Execute the complete resumable EdgeGuard Colab v3 campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from edgeguard.rescue.colab_recovery import restore_recovery_file, restore_state_archive
from edgeguard.serialization import canonical_json, sha256_file


def _run(
    command: list[str],
    *,
    project_root: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source = str(project_root / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment["UV_CACHE_DIR"] = str(project_root.parent / "edgeguard-cache/uv")
    print("COMMAND:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    if capture_output and completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}\n"
            + (completed.stdout or "")[-12000:]
        )
    return completed


def _resource_gate(content_root: Path) -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, memory_text = [value.strip() for value in query.split(",", 1)]
    memory_mib = int(memory_text)
    allowed = any(token in name.upper() for token in ("L4", "A100", "H100"))
    if not allowed or memory_mib < 22_000:
        raise RuntimeError(
            f"production requires L4/A100/H100 with at least 22 GiB VRAM; received {query}"
        )
    memory_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
    if memory_kib < 38 * 1024**2:
        raise RuntimeError("production requires Colab high-RAM (at least 38 GiB)")
    free_bytes = shutil.disk_usage(content_root).free
    if free_bytes < 80 * 1024**3:
        raise RuntimeError("production requires at least 80 GiB free ephemeral disk")
    return {
        "gpu": name,
        "gpu_memory_mib": memory_mib,
        "system_memory_gib": round(memory_kib / 1024**2, 2),
        "free_disk_gib": round(free_bytes / 1024**3, 2),
    }


def _restore_state(
    *, store_root: Path, work_root: Path, content_root: Path, project_commit: str
) -> bool:
    if work_root.exists() and any(work_root.iterdir()):
        return False
    pointer = store_root / "pointers/campaign-state.json"
    if not pointer.is_file():
        return False
    archive = content_root / "edgeguard-campaign-state-v3.tar.gz"
    result = restore_recovery_file(
        store_root,
        artifact_id="campaign-state",
        destination=archive,
    )
    if result.get("project_commit") != project_commit:
        archive.unlink(missing_ok=True)
        raise ValueError("v3 campaign state belongs to another pinned application commit")
    restore_state_archive(archive, work_root)
    return True


def _publish_deliveries(source_root: Path, destination_root: Path) -> dict[str, str]:
    destination_root.mkdir(parents=True, exist_ok=True)
    published: dict[str, str] = {}
    for source in sorted(path for path in source_root.iterdir() if path.is_file()):
        destination = destination_root / source.name
        digest = sha256_file(source)
        if destination.is_file():
            if sha256_file(destination) != digest:
                raise FileExistsError(f"Drive delivery exists with another identity: {destination}")
        else:
            incoming = destination.with_name(f".{destination.name}.incoming")
            incoming.unlink(missing_ok=True)
            shutil.copy2(source, incoming)
            if sha256_file(incoming) != digest:
                incoming.unlink(missing_ok=True)
                raise OSError(f"Drive delivery copy hash mismatch: {source.name}")
            os.replace(incoming, destination)
        published[source.name] = str(destination)
    return published


def execute(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve()
    content_root = args.content_root.resolve()
    drive_root = args.drive_root.resolve()
    work_root = content_root / "edgeguard-work"
    data_root = content_root / "edgeguard-data"
    campaign_root = drive_root / "EdgeGuard/campaigns/semantic-cs-idd-v3"
    recovery_root = campaign_root / "recovery/v3"
    policy = project_root / "configs/campaign/semantic_cs_idd_v3_authorization.json"
    resources = (
        _resource_gate(content_root)
        if args.execution_mode == "production"
        else {"acceptance_mode": True}
    )
    restored = _restore_state(
        store_root=recovery_root,
        work_root=work_root,
        content_root=content_root,
        project_commit=args.project_commit,
    )
    prepared = _run(
        [
            sys.executable,
            str(project_root / "scripts/prepare_colab_v3.py"),
            "--drive-root",
            str(drive_root),
            "--local-data-root",
            str(data_root),
            "--work-root",
            str(work_root),
            "--authorization-policy",
            str(policy),
            "--project-commit",
            args.project_commit,
        ],
        project_root=project_root,
        capture_output=True,
    )
    prepared_payload = json.loads(prepared.stdout.splitlines()[-1])

    evidence_root = content_root / "edgeguard-evidence"
    runtime_root = content_root / "edgeguard-runtime"
    checkout_root = content_root / "edgeguard-checkouts"
    _run(
        [
            sys.executable,
            str(project_root / "scripts/train/install_semantic_stack.py"),
            "--config",
            str(project_root / "configs/training/segmentation/framework_mmseg.yaml"),
            "--project-root",
            str(project_root),
            "--project-commit",
            args.project_commit,
            "--config-root",
            str(project_root / "configs/training/segmentation"),
            "--runtime-root",
            str(runtime_root),
            "--checkout-root",
            str(checkout_root),
            "--evidence-root",
            str(evidence_root),
            "--log-root",
            str(content_root / "edgeguard-logs"),
            "--cache-root",
            str(content_root / "edgeguard-cache"),
            "--data-root",
            str(data_root / "cityscapes"),
            "--execute",
        ],
        project_root=project_root,
    )
    resolved_runtime = evidence_root / "resolved-runtime.json"
    _run(
        [
            sys.executable,
            str(project_root / "scripts/resolve_colab_runtime.py"),
            "--receipt",
            str(evidence_root / "runtime_receipt.json"),
            "--project-commit",
            args.project_commit,
            "--output",
            str(resolved_runtime),
        ],
        project_root=project_root,
    )
    runtime = json.loads(resolved_runtime.read_text(encoding="utf-8"))
    runtime_python = str(runtime["interpreter"])
    command = [
        runtime_python,
        str(project_root / "scripts/colab_pipeline.py"),
        "run",
        "--target",
        "all",
        "--execution-mode",
        args.execution_mode,
        "--campaign-id",
        "semantic-cs-idd-v3",
        "--project-root",
        str(project_root),
        "--project-commit",
        args.project_commit,
        "--runtime-receipt",
        str(evidence_root / "runtime_receipt.json"),
        "--mmseg-root",
        str(runtime["mmseg_root"]),
        "--work-root",
        str(work_root),
        "--recovery-root",
        str(recovery_root),
        "--state-store-root",
        str(recovery_root),
        "--config",
        str(project_root / "configs/rescue/semantic_first.yaml"),
        "--authorization-policy",
        str(policy),
        "--release-policy",
        str(policy),
        "--rare-classes-file",
        str(work_root / "multidomain-statistics/rare_classes.json"),
        "--class-weights-file",
        str(work_root / "multidomain-statistics/class_weights.json"),
        "--data-root",
        f"cityscapes={data_root / 'cityscapes'}",
        "--data-root",
        f"idd20k={data_root / 'idd20k'}",
    ]
    for manifest in prepared_payload["training_manifests"]:
        command.extend(("--data-manifest", str(manifest)))
    _run(command, project_root=project_root)
    accepted = json.loads((work_root / "accepted_release.json").read_text(encoding="utf-8"))
    release_id = str(accepted["release_id"])
    local_delivery = work_root / "deliveries" / release_id
    drive_delivery = campaign_root / "releases" / release_id
    published = _publish_deliveries(local_delivery, drive_delivery)
    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_master_result",
        "status": "completed",
        "campaign_id": "semantic-cs-idd-v3",
        "release_id": release_id,
        "recommended_model": accepted["recommended_model"],
        "project_commit": args.project_commit,
        "resources": resources,
        "restored_campaign_state": restored,
        "data": prepared_payload,
        "drive_deliveries": published,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=Path("/content"))
    parser.add_argument(
        "--execution-mode", choices=("production", "acceptance"), default="production"
    )
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(execute(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
