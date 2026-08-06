"""Execute the complete resumable EdgeGuard Colab v3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_name(f".{path.name}.incoming")
    incoming.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def _tail(path: Path, limit: int = 24000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - limit))
        return source.read().decode("utf-8", errors="replace")


def _run(
    command: list[str],
    *,
    project_root: Path,
    child_log: Path,
    environment: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Stream child output while also retaining it for failure diagnostics."""
    resolved_environment = environment or os.environ.copy()
    print("COMMAND:", " ".join(command), flush=True)
    child_log.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with child_log.open("a", encoding="utf-8") as sink:
        sink.write("\nCOMMAND: " + " ".join(command) + "\n")
        sink.flush()
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=resolved_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
            if capture_output:
                captured.append(line)
        return_code = process.wait()
        sink.write(f"RETURN_CODE: {return_code}\n")
    output = "".join(captured) if capture_output else None
    completed = subprocess.CompletedProcess(command, return_code, output, None)
    if return_code:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {command}\n{_tail(child_log)}"
        )
    return completed


def _last_json(output: str | None) -> dict[str, Any]:
    for line in reversed((output or "").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("child command completed without a machine-readable JSON result")


def _resource_gate(content_root: Path) -> dict[str, object]:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "GPU okunamadı. Colab'da Çalışma zamanı > Çalışma zamanı türünü değiştir > L4 seçin."
        ) from error
    candidates: list[tuple[str, int]] = []
    for raw_line in query.splitlines():
        name, separator, memory_text = raw_line.partition(",")
        digits = re.sub(r"[^0-9]", "", memory_text)
        if separator and name.strip() and digits:
            candidates.append((name.strip(), int(digits)))
    selected = next(
        (
            item
            for item in candidates
            if any(token in item[0].upper() for token in ("L4", "A100", "H100"))
            and item[1] >= 22_000
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            "Üretim için en az 22 GiB VRAM'li L4/A100/H100 gerekli; "
            f"Colab şu GPU bilgisini verdi: {query.strip() or 'boş'}"
        )
    memory_kib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
    free_bytes = shutil.disk_usage(content_root).free
    observed = {
        "gpu": selected[0],
        "gpu_memory_mib": selected[1],
        "system_memory_gib": round(memory_kib / 1024**2, 2),
        "free_disk_gib": round(free_bytes / 1024**3, 2),
    }
    if memory_kib < 38 * 1024**2:
        raise RuntimeError(
            "Üretim için Yüksek RAM gerekli (en az 38 GiB); "
            f"mevcut toplam RAM {observed['system_memory_gib']} GiB."
        )
    if free_bytes < 80 * 1024**3:
        raise RuntimeError(
            "Üretim için /content altında en az 80 GiB boş alan gerekli; "
            f"mevcut boş alan {observed['free_disk_gib']} GiB."
        )
    return observed


def _runtime_environment(
    *, project_root: Path, cache_root: Path, runtime_python: Path, uv_executable: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = str(uv_executable.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["UV_CACHE_DIR"] = str(cache_root / "uv")
    environment["UV_PYTHON_INSTALL_DIR"] = str(cache_root / "python")
    environment["UV_PYTHON_PREFERENCE"] = "only-managed"
    environment["EDGEGUARD_RUNTIME_PYTHON"] = str(runtime_python)
    return environment


def _restore_state(
    *,
    runtime_python: Path,
    environment: dict[str, str],
    project_root: Path,
    child_log: Path,
    store_root: Path,
    work_root: Path,
    content_root: Path,
    project_commit: str,
) -> bool:
    if work_root.exists() and any(work_root.iterdir()):
        return False
    pointer = store_root / "pointers/campaign-state.json"
    if not pointer.is_file():
        return False
    archive = content_root / "edgeguard-campaign-state-v3.tar.gz"
    restored = _run(
        [
            str(runtime_python),
            str(project_root / "scripts/manage_colab_recovery.py"),
            "restore",
            "--store-root",
            str(store_root),
            "--artifact-id",
            "campaign-state",
            "--destination",
            str(archive),
        ],
        project_root=project_root,
        child_log=child_log,
        environment=environment,
        capture_output=True,
    )
    result = _last_json(restored.stdout)
    if result.get("project_commit") != project_commit:
        archive.unlink(missing_ok=True)
        print(
            "Drive'daki eski v3 state farklı uygulama commit'ine ait; korunup atlandı.",
            flush=True,
        )
        return False
    _run(
        [
            str(runtime_python),
            str(project_root / "scripts/manage_colab_recovery.py"),
            "restore-state",
            "--archive",
            str(archive),
            "--destination",
            str(work_root),
        ],
        project_root=project_root,
        child_log=child_log,
        environment=environment,
    )
    return True


def _quarantine_local_commit_drift(root: Path, project_commit: str) -> Path | None:
    """Preserve, but never resume, ephemeral evidence bound to another source commit."""
    if not root.is_dir():
        return None
    candidates = (
        root / "runtime_receipt.json",
        root / "accepted_release.json",
        *sorted(root.glob("manifests/training/*.frozen.json")),
        *sorted(root.glob("pipeline-v3/*/run_manifest.json")),
    )
    observed: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("project_commit", "git_commit", "approved_project_commit"):
            value = payload.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
                observed.add(value)
    incompatible = sorted(value for value in observed if value != project_commit)
    if not incompatible:
        return None
    suffix = incompatible[0][:12]
    destination = root.with_name(f"{root.name}.incompatible-{suffix}")
    counter = 1
    while destination.exists():
        destination = root.with_name(f"{root.name}.incompatible-{suffix}-{counter}")
        counter += 1
    os.replace(root, destination)
    print(f"Eski yerel state korumaya alındı: {destination}", flush=True)
    return destination


def _publish_deliveries(source_root: Path, destination_root: Path) -> dict[str, str]:
    destination_root.mkdir(parents=True, exist_ok=True)
    published: dict[str, str] = {}
    for source in sorted(path for path in source_root.iterdir() if path.is_file()):
        destination = destination_root / source.name
        digest = _sha256(source)
        if destination.is_file():
            if _sha256(destination) != digest:
                raise FileExistsError(f"Drive delivery exists with another identity: {destination}")
        else:
            incoming = destination.with_name(f".{destination.name}.incoming")
            incoming.unlink(missing_ok=True)
            shutil.copy2(source, incoming)
            if _sha256(incoming) != digest:
                incoming.unlink(missing_ok=True)
                raise OSError(f"Drive delivery copy hash mismatch: {source.name}")
            os.replace(incoming, destination)
        published[source.name] = str(destination)
    return published


def execute(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve()
    content_root = args.content_root.resolve()
    drive_root = args.drive_root.resolve()
    work_root = content_root / "edgeguard-work-v3"
    data_root = content_root / "edgeguard-data"
    cache_root = content_root / "edgeguard-cache"
    runtime_root = content_root / "edgeguard-runtime"
    checkout_root = content_root / "edgeguard-checkouts"
    evidence_root = content_root / "edgeguard-evidence"
    child_log = content_root / "edgeguard-master-child.log"
    stage_path = content_root / "edgeguard-master-stage.json"
    campaign_root = drive_root / "EdgeGuard/campaigns/semantic-cs-idd-v3"
    recovery_root = campaign_root / "recovery/v3"
    policy = project_root / "configs/campaign/semantic_cs_idd_v3_authorization.json"

    def stage(name: str, **extra: object) -> None:
        payload = {
            "schema_version": "1.0",
            "record_type": "edgeguard_colab_master_stage",
            "stage": name,
            "project_commit": args.project_commit,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        _atomic_json(stage_path, payload)
        print(f"\nEDGEGUARD STAGE: {name}", flush=True)

    child_log.unlink(missing_ok=True)
    stage("resource-gate")
    resources = (
        _resource_gate(content_root)
        if args.execution_mode == "production"
        else {"acceptance_mode": True}
    )
    quarantined_work = _quarantine_local_commit_drift(work_root, args.project_commit)
    quarantined_evidence = _quarantine_local_commit_drift(evidence_root, args.project_commit)
    stage("stdlib-hermetic-bootstrap", resources=resources)
    bootstrap_receipt = content_root / "edgeguard-bootstrap-receipt.json"
    _run(
        [
            sys.executable,
            str(project_root / "scripts/bootstrap_colab_runtime.py"),
            "--project-root",
            str(project_root),
            "--project-commit",
            args.project_commit,
            "--content-root",
            str(content_root),
            "--runtime-root",
            str(runtime_root),
            "--checkout-root",
            str(checkout_root),
            "--cache-root",
            str(cache_root),
            "--log",
            str(content_root / "edgeguard-logs/bootstrap.log"),
            "--output",
            str(bootstrap_receipt),
        ],
        project_root=project_root,
        child_log=child_log,
    )
    bootstrap_payload = json.loads(bootstrap_receipt.read_text(encoding="utf-8"))
    runtime_python = Path(str(bootstrap_payload["interpreter"]))
    uv_executable = Path(str(bootstrap_payload["uv"]["path"]))
    if not uv_executable.is_file():
        raise FileNotFoundError("verified private uv executable disappeared after bootstrap")
    environment = _runtime_environment(
        project_root=project_root,
        cache_root=cache_root,
        runtime_python=runtime_python,
        uv_executable=uv_executable,
    )

    stage("five-model-runtime-canary")
    _run(
        [
            str(runtime_python),
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
            str(cache_root),
            "--data-root",
            str(data_root / "cityscapes"),
            "--execute",
        ],
        project_root=project_root,
        child_log=child_log,
        environment=environment,
    )
    resolved_runtime = evidence_root / "resolved-runtime.json"
    _run(
        [
            str(runtime_python),
            str(project_root / "scripts/resolve_colab_runtime.py"),
            "--receipt",
            str(evidence_root / "runtime_receipt.json"),
            "--project-commit",
            args.project_commit,
            "--output",
            str(resolved_runtime),
        ],
        project_root=project_root,
        child_log=child_log,
        environment=environment,
    )
    runtime = json.loads(resolved_runtime.read_text(encoding="utf-8"))
    if Path(str(runtime["interpreter"])).resolve() != runtime_python.resolve():
        raise ValueError("resolved runtime differs from the standard-library bootstrap")

    stage(
        "restore",
        quarantined_work=str(quarantined_work) if quarantined_work else None,
        quarantined_evidence=str(quarantined_evidence) if quarantined_evidence else None,
    )
    restored = _restore_state(
        runtime_python=runtime_python,
        environment=environment,
        project_root=project_root,
        child_log=child_log,
        store_root=recovery_root,
        work_root=work_root,
        content_root=content_root,
        project_commit=args.project_commit,
    )
    stage("data", restored_campaign_state=restored)
    prepared = _run(
        [
            str(runtime_python),
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
        child_log=child_log,
        environment=environment,
        capture_output=True,
    )
    prepared_payload = _last_json(prepared.stdout)

    stage("production-pipeline")
    command = [
        str(runtime_python),
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
    manifests = prepared_payload.get("training_manifests")
    if not isinstance(manifests, list) or len(manifests) != 2:
        raise ValueError("data preparation did not return both frozen training manifests")
    for manifest in manifests:
        command.extend(("--data-manifest", str(manifest)))
    _run(
        command,
        project_root=project_root,
        child_log=child_log,
        environment=environment,
    )

    stage("publish-deliveries")
    accepted = json.loads((work_root / "accepted_release.json").read_text(encoding="utf-8"))
    release_id = str(accepted["release_id"])
    local_delivery = work_root / "deliveries" / release_id
    drive_delivery = campaign_root / "releases" / release_id
    published = _publish_deliveries(local_delivery, drive_delivery)
    required = {
        "EdgeGuard_Jetson_Release.zip",
        "EdgeGuard_Thesis_Bundle.zip",
        "EdgeGuard_Streamlit_Demo.zip",
        "release_index.json",
    }
    if not required.issubset(published):
        missing = sorted(required - published.keys())
        raise FileNotFoundError(f"release is missing deliveries: {missing}")
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
    _atomic_json(args.result.resolve(), result)
    stage("completed", release_id=release_id, deliveries=published)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=Path("/content"))
    parser.add_argument(
        "--execution-mode", choices=("production", "acceptance"), default="production"
    )
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = execute(args)
    except BaseException as error:
        content_root = args.content_root.resolve()
        stage_path = content_root / "edgeguard-master-stage.json"
        try:
            current_stage = json.loads(stage_path.read_text(encoding="utf-8")).get("stage")
        except (OSError, json.JSONDecodeError):
            current_stage = "master-entry"
        failure = {
            "schema_version": "1.0",
            "record_type": "edgeguard_colab_master_child_failure",
            "status": "failed",
            "stage": current_stage,
            "project_commit": args.project_commit,
            "error_type": type(error).__name__,
            "message": str(error)[-24000:],
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-30000:],
            "child_log": str(content_root / "edgeguard-master-child.log"),
            "safe_restart": "fix the reported stage if needed, then run all again",
        }
        _atomic_json(content_root / "edgeguard-master-child-failure.json", failure)
        print("\nEDGEGUARD ROOT FAILURE:", _canonical_json(failure), file=sys.stderr, flush=True)
        raise
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
