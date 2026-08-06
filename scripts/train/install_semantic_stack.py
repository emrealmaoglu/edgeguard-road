"""Install and prove the single hash-locked Colab semantic runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from packaging.version import InvalidVersion, Version

from edgeguard.runtime import RuntimePathContract
from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, atomic_write_json
from edgeguard.training.config import load_semantic_framework_config
from edgeguard.training.contracts import SemanticFrameworkConfig

UV_VERSION = "0.8.8"
CORE_CANARY_MODELS = ("segformer_b0", "fast_scnn", "pidnet_s")
OPENMMLAB_LOCKFILE = Path("requirements/colab-openmmlab.lock")
OPENMMLAB_REQUIREMENTS = {
    "mmengine": (
        "0.10.7",
        "262ac976a925562f78cd5fd14dd1bc9b680ed0aa81f0d85b723ef782f99c54ee",
    ),
    "mmcv-lite": (
        "2.1.0",
        "1d9913c35f793de4a3a022b93cecb712e1e7262eb4704eb8cd15e623dd375000",
    ),
}
OWNED_RUNTIME_NAMES = {"edgeguard-runtime", "runtime"}
OWNED_CHECKOUT_NAMES = {"mmsegmentation"}
OWNED_PROBE_NAMES = {"hermetic-core-model-probe"}

WhichFunction = Callable[[str], str | None]
VersionProbe = Callable[[Path], str]


class BootstrapError(RuntimeError):
    """Classified uv bootstrap failure with one exact terminal stage."""

    def __init__(self, stage: str, classification: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.classification = classification


def _python(root: Path) -> Path:
    return root / "bin" / "python"


def _uv_version(executable: Path) -> str:
    return subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_uv_version(version_output: str) -> Version | None:
    match = re.fullmatch(r"uv\s+([^\s]+)(?:\s+.*)?", version_output)
    try:
        return Version(match.group(1)) if match else None
    except InvalidVersion:
        return None


def _resolve_uv_executable(
    runner: LiveCommandRunner,
    *,
    which: WhichFunction = shutil.which,
    scripts_directory: Path | None = None,
    bootstrap_root: Path | None = None,
    version_probe: VersionProbe = _uv_version,
) -> tuple[Path, str, list[dict[str, Any]]]:
    """Resolve exact uv, privately bootstrapping it when the host version differs."""
    receipts: list[dict[str, Any]] = []
    scripts_root = scripts_directory or (
        bootstrap_root / "bin" if bootstrap_root else Path(sysconfig.get_path("scripts"))
    )
    resolved = which("uv")
    if resolved is not None:
        hosted = Path(resolved).resolve()
        if hosted.is_file() and os.access(hosted, os.X_OK):
            try:
                hosted_output = version_probe(hosted)
            except (OSError, subprocess.CalledProcessError):
                hosted_output = ""
            hosted_version = _parse_uv_version(hosted_output)
            if hosted_version == Version(UV_VERSION):
                return hosted, str(hosted_version), receipts

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        "--force-reinstall",
    ]
    if bootstrap_root is not None:
        bootstrap_root.mkdir(parents=True, exist_ok=True)
        command.extend(["--prefix", str(bootstrap_root)])
    command.append(f"uv=={UV_VERSION}")
    try:
        receipts.append(
            runner.run(
                "bootstrap-uv",
                tuple(command),
                stage_index=1,
                stage_total=1,
            )
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BootstrapError(
            "uv_install",
            "uv_install_failed",
            "pinned uv installation command failed",
        ) from error

    executable = (scripts_root / "uv").resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise BootstrapError(
            "uv_resolution",
            "uv_executable_not_found",
            "uv installation completed but the private prefix has no uv executable",
        )
    try:
        version_output = version_probe(executable)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BootstrapError(
            "uv_version_validation",
            "uv_version_probe_failed",
            "uv version probe failed",
        ) from error
    actual_version = _parse_uv_version(version_output)
    if actual_version != Version(UV_VERSION):
        raise BootstrapError(
            "uv_version_validation",
            "uv_version_mismatch",
            f"expected uv {UV_VERSION}, found: {version_output}",
        )
    return executable, str(actual_version), receipts


def _checkout_head(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _repair_runtime_target(runtime_root: Path) -> dict[str, str] | None:
    if not runtime_root.exists():
        return None
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise ValueError("owned runtime target is not a regular directory")
    if runtime_root.name not in OWNED_RUNTIME_NAMES:
        raise ValueError("refusing repair outside the bounded EdgeGuard runtime names")
    interpreter = _python(runtime_root)
    if interpreter.is_file() and os.access(interpreter, os.X_OK):
        return {"target": runtime_root.name, "action": "reuse_resumable_environment"}
    recognizable = not any(runtime_root.iterdir()) or (runtime_root / "pyvenv.cfg").exists()
    if not recognizable:
        raise ValueError("refusing to remove an unrecognized existing runtime directory")
    shutil.rmtree(runtime_root)
    return {"target": runtime_root.name, "action": "removed_incomplete_environment"}


def _repair_checkout_target(checkout: Path, expected_commit: str) -> dict[str, str] | None:
    if not checkout.exists():
        return None
    if checkout.is_symlink() or not checkout.is_dir() or checkout.name not in OWNED_CHECKOUT_NAMES:
        raise ValueError("refusing repair outside the bounded EdgeGuard checkout names")
    if _checkout_head(checkout) == expected_commit:
        return {"target": checkout.name, "action": "reuse_exact_checkout"}
    if (checkout / ".git").exists() or not any(checkout.iterdir()):
        shutil.rmtree(checkout)
        return {"target": checkout.name, "action": "removed_incomplete_checkout"}
    raise ValueError("refusing to remove an unrecognized existing checkout directory")


def _repair_probe_target(probe: Path) -> dict[str, str] | None:
    if not probe.exists() or (probe / "completion.json").is_file():
        return None
    if probe.is_symlink() or not probe.is_dir() or probe.name not in OWNED_PROBE_NAMES:
        raise ValueError("refusing to remove an unrecognized probe target")
    shutil.rmtree(probe)
    return {"target": probe.name, "action": "removed_incomplete_probe"}


def repair_owned_path(
    *,
    runtime_root: Path,
    checkout: Path,
    probe: Path,
    expected_commit: str,
) -> list[dict[str, str]]:
    """Apply the bounded repair policy for the one owned hermetic runtime."""
    actions = (
        _repair_runtime_target(runtime_root),
        _repair_checkout_target(checkout, expected_commit),
        _repair_probe_target(probe),
    )
    return [action for action in actions if action is not None]


def _requirement_present(lock_text: str, distribution: str, version: str) -> bool:
    return (
        re.search(
            rf"(?m)^{re.escape(distribution)}=={re.escape(version)}(?:\s|\\|$)",
            lock_text,
        )
        is not None
    )


def _validate_lock_contract(main_lock: Path, openmmlab_lock: Path) -> None:
    """Reject dependency layouts that can silently recreate the Colab ABI failure."""
    main_text = main_lock.read_text(encoding="utf-8")
    openmmlab_text = openmmlab_lock.read_text(encoding="utf-8")
    if re.search(r"(?m)^opencv-python==", main_text):
        raise ValueError("main Colab lock must not contain GUI opencv-python")
    if not _requirement_present(main_text, "opencv-python-headless", "4.10.0.84"):
        raise ValueError("main Colab lock must pin opencv-python-headless==4.10.0.84")
    if not _requirement_present(main_text, "rich", "15.0.0"):
        raise ValueError("main Colab lock must directly pin rich==15.0.0")
    for forbidden in ("mmengine", "mmcv-lite"):
        if re.search(rf"(?m)^{re.escape(forbidden)}==", main_text):
            raise ValueError(f"{forbidden} must live only in the OpenMMLab lock")
    for distribution, (version, expected_hash) in OPENMMLAB_REQUIREMENTS.items():
        if not _requirement_present(openmmlab_text, distribution, version):
            raise ValueError(f"OpenMMLab lock must pin {distribution}=={version}")
        if f"--hash=sha256:{expected_hash}" not in openmmlab_text:
            raise ValueError(f"OpenMMLab lock has an unexpected {distribution} wheel hash")
    extra = {
        match.group(1).lower() for match in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==", openmmlab_text)
    } - set(OPENMMLAB_REQUIREMENTS)
    if extra:
        raise ValueError(f"OpenMMLab lock contains unexpected distributions: {sorted(extra)}")


def _lock_paths(config: SemanticFrameworkConfig, project_root: Path) -> tuple[Path, Path]:
    main_lock = (project_root / config.lockfile).resolve()
    openmmlab_lock = (project_root / OPENMMLAB_LOCKFILE).resolve()
    for lock in (main_lock, openmmlab_lock):
        if not lock.is_file():
            raise FileNotFoundError(f"required Colab lock file is missing: {lock}")
    _validate_lock_contract(main_lock, openmmlab_lock)
    return main_lock, openmmlab_lock


def build_hermetic_commands(
    config: SemanticFrameworkConfig,
    checkout: Path,
    *,
    uv_executable: Path,
    project_root: Path,
    runtime_root: Path,
) -> tuple[tuple[str, ...], ...]:
    """Build the one lock-sync command sequence; no dependency fallback is permitted."""
    uv = str(uv_executable)
    interpreter = _python(runtime_root)
    main_lock, openmmlab_lock = _lock_paths(config, project_root)
    return (
        (uv, "python", "install", config.python_version),
        (uv, "venv", "--python", config.python_version, str(runtime_root)),
        (
            uv,
            "pip",
            "sync",
            "--python",
            str(interpreter),
            "--strict",
            "--require-hashes",
            str(main_lock),
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "--require-hashes",
            "-r",
            str(openmmlab_lock),
        ),
        (
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            config.repository_url,
            str(checkout),
        ),
        ("git", "-C", str(checkout), "checkout", "--detach", config.commit),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(checkout),
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(project_root),
        ),
    )


def _environment_probe(interpreter: Path, checkout: Path) -> dict[str, Any]:
    script = """
import importlib.metadata, json, platform
import cv2, matplotlib, mmcv, mmengine, mmseg, numpy, onnx, onnxruntime, optuna, streamlit
import torch, torchaudio, torchvision
def maybe_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
payload = {
  'python_version': platform.python_version(),
  'numpy_version': numpy.__version__,
  'torch_version': importlib.metadata.version('torch'),
  'torchvision_version': importlib.metadata.version('torchvision'),
  'torchaudio_version': importlib.metadata.version('torchaudio'),
  'cuda_runtime': torch.version.cuda,
  'cuda_available': bool(torch.cuda.is_available()),
  'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'mmengine_version': importlib.metadata.version('mmengine'),
  'mmcv_distribution': 'mmcv-lite',
  'mmcv_version': importlib.metadata.version('mmcv-lite'),
  'mmsegmentation_version': importlib.metadata.version('mmsegmentation'),
  'opencv_headless_version': importlib.metadata.version('opencv-python-headless'),
  'opencv_gui_version': maybe_version('opencv-python'),
  'onnx_version': importlib.metadata.version('onnx'),
  'onnxruntime_version': importlib.metadata.version('onnxruntime'),
  'optuna_version': importlib.metadata.version('optuna'),
  'streamlit_version': importlib.metadata.version('streamlit'),
  'matplotlib_version': importlib.metadata.version('matplotlib'),
  'delivery_imports_verified': True,
}
print(json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [str(interpreter), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("environment probe did not return an object")
    payload["mmsegmentation_commit"] = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return payload


def _base_version(value: object) -> str:
    return str(value).split("+")[0]


def _validate_identity(identity: dict[str, Any], config: SemanticFrameworkConfig) -> None:
    exact = {
        "python_version": config.python_version,
        "numpy_version": config.numpy_version,
        "mmengine_version": config.mmengine_version,
        "mmcv_version": config.mmcv_version,
        "opencv_headless_version": config.opencv_headless_version,
        "mmsegmentation_commit": config.commit,
    }
    for field, expected in exact.items():
        if identity.get(field) != expected:
            raise ValueError(f"hermetic environment identity mismatch for {field}")
    wheel_versions = {
        "torch_version": config.torch_version,
        "torchvision_version": config.torchvision_version,
        "torchaudio_version": config.torchaudio_version,
    }
    for field, expected in wheel_versions.items():
        if _base_version(identity.get(field)) != expected:
            raise ValueError(f"hermetic CUDA wheel identity mismatch for {field}")
    if identity.get("mmcv_distribution") != config.preferred_mmcv_distribution:
        raise ValueError("MMCV distribution mismatch")
    if identity.get("opencv_gui_version") is not None:
        raise ValueError("opencv-python must not coexist with opencv-python-headless")
    if identity.get("cuda_runtime") != "12.1" or identity.get("cuda_available") is not True:
        raise ValueError("hermetic runtime does not expose the locked CUDA 12.1 stack")
    if identity.get("delivery_imports_verified") is not True:
        raise ValueError("Colab delivery dependency imports were not verified")


def _probe_command(
    interpreter: Path,
    *,
    project_root: Path,
    project_commit: str,
    config_root: Path,
    checkout: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    command = [
        str(interpreter),
        str(project_root / "scripts/train/train_semantic.py"),
        "stack-probe",
        "--config-root",
        str(config_root),
        "--mmseg-checkout",
        str(checkout),
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(project_root),
        "--project-commit",
        project_commit,
        "--device",
        "cuda",
    ]
    for model in CORE_CANARY_MODELS:
        command.extend(("--model", model))
    return tuple(command)


def _execute_runtime(
    commands: tuple[tuple[str, ...], ...],
    *,
    interpreter: Path,
    checkout: Path,
    config: SemanticFrameworkConfig,
    project_root: Path,
    project_commit: str,
    config_root: Path,
    evidence_root: Path,
    cache_root: Path,
    runner: LiveCommandRunner,
) -> dict[str, Any]:
    started = time.perf_counter()
    command_receipts: list[dict[str, Any]] = []
    cache_environment = {**os.environ, "UV_CACHE_DIR": str(cache_root / "uv")}
    for index, command in enumerate(commands, start=1):
        is_venv = len(command) > 1 and command[1] == "venv"
        is_clone = len(command) > 1 and command[0] == "git" and command[1] == "clone"
        is_checkout = len(command) > 3 and command[0] == "git" and command[3] == "checkout"
        if is_venv and interpreter.is_file():
            continue
        if (is_clone or is_checkout) and _checkout_head(checkout) == config.commit:
            continue
        command_receipts.append(
            runner.run(
                f"hermetic-install-{index}",
                command,
                stage_index=index,
                stage_total=len(commands) + 1,
                cwd=project_root,
                env=cache_environment,
            )
        )
    identity = _environment_probe(interpreter, checkout)
    _validate_identity(identity, config)
    probe_output = evidence_root / "hermetic-core-model-probe"
    if probe_output.exists() and not (probe_output / "completion.json").is_file():
        raise ValueError("incomplete prior core-model probe must be inspected")
    if not probe_output.exists():
        command_receipts.append(
            runner.run(
                "hermetic-core-model-probe",
                _probe_command(
                    interpreter,
                    project_root=project_root,
                    project_commit=project_commit,
                    config_root=config_root,
                    checkout=checkout,
                    output_dir=probe_output,
                ),
                stage_index=len(commands) + 1,
                stage_total=len(commands) + 1,
                cwd=project_root,
            )
        )
    completion = json.loads((probe_output / "completion.json").read_text(encoding="utf-8"))
    summary = json.loads((probe_output / "stack_probe_summary.json").read_text(encoding="utf-8"))
    fp16_models = [
        model
        for model in summary.get("models", [])
        if isinstance(model, dict) and model.get("fp16_finite_verified") is True
    ]
    if (
        completion.get("model_count") != len(CORE_CANARY_MODELS)
        or completion.get("checkpoint_resume_verified") is not True
        or completion.get("checkpoint_resume_model_count") != len(CORE_CANARY_MODELS)
        or len(fp16_models) != len(CORE_CANARY_MODELS)
    ):
        raise ValueError("three-model hermetic canary is incomplete")
    completion["fp16_finite_model_count"] = len(fp16_models)
    return {
        "runtime_profile": "py311-cu121",
        "interpreter": str(interpreter),
        "environment": identity,
        "framework_commit": config.commit,
        "install_duration_seconds": time.perf_counter() - started,
        "commands": [list(command) for command in commands],
        "command_receipts": command_receipts,
        "core_model_probe": completion,
    }


def _evidence_zip(evidence_root: Path, receipt: dict[str, Any]) -> Path:
    receipt_path = evidence_root / "runtime_receipt.json"
    atomic_write_json(receipt_path, receipt)
    package = evidence_root / "semantic-hermetic-runtime-evidence.zip"
    files = [receipt_path]
    for candidate in sorted(evidence_root.rglob("*.json")):
        if candidate not in files and candidate.stat().st_size <= 2 * 1024**2:
            files.append(candidate)
    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(files):
            relative = source.relative_to(evidence_root).as_posix()
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return package


def _reuse_completed_environment(
    evidence_root: Path,
    checkout_root: Path,
    config: SemanticFrameworkConfig,
    project_root: Path,
    project_commit: str,
) -> dict[str, Any] | None:
    receipt_path = evidence_root / "runtime_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("record_type") != "semantic_hermetic_runtime_receipt":
        raise ValueError("completed runtime receipt has an invalid record type")
    if receipt.get("project_commit") != project_commit:
        raise ValueError("completed runtime receipt belongs to another project commit")
    lock_paths = _lock_paths(config, project_root)
    expected_locks = {path.name: sha256_file(path) for path in lock_paths}
    if receipt.get("lock_sha256") != expected_locks:
        raise ValueError("completed runtime receipt belongs to another package lock")
    interpreter = Path(str(receipt.get("interpreter")))
    checkout = checkout_root / "mmsegmentation"
    identity = _environment_probe(interpreter, checkout)
    _validate_identity(identity, config)
    probe = receipt.get("core_model_probe")
    if not isinstance(probe, dict) or (
        probe.get("model_count") != len(CORE_CANARY_MODELS)
        or probe.get("checkpoint_resume_verified") is not True
        or probe.get("fp16_finite_model_count") != len(CORE_CANARY_MODELS)
    ):
        raise ValueError("completed runtime receipt lacks the three-model acceptance")
    return {**receipt, "reused_completed_environment": True}


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None and len(chain) < 5:
        chain.append({"error_type": type(current).__name__, "message": str(current)[:1000]})
        current = current.__cause__ or current.__context__
    return chain


def _record_failure(
    error: BaseException,
    *,
    paths: RuntimePathContract,
    status: LongRunStatus,
    failed_stage: str,
) -> None:
    classification = (
        error.classification if isinstance(error, BootstrapError) else "runtime_install_failed"
    )
    stage = error.stage if isinstance(error, BootstrapError) else failed_stage
    diagnostics = {
        "schema_version": "1.0",
        "record_type": "semantic_runtime_failure",
        "status": "failed",
        "failed_stage": stage,
        "failure_classification": classification,
        "exception_chain": _exception_chain(error),
        "runtime_contract": paths.receipt(),
        "safe_restart": "rerun_install_target",
    }
    paths.log_root.mkdir(parents=True, exist_ok=True)
    paths.evidence_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.evidence_root / "failure.json", diagnostics)
    status.update(phase=stage, last_error=classification, force=True)
    status.fail(error)


def install_hermetic_runtime(
    config_path: Path,
    *,
    paths: RuntimePathContract,
    project_root: Path,
    project_commit: str,
    config_root: Path,
) -> dict[str, Any]:
    """Install the sole published runtime; a failure never selects another matrix."""
    config = load_semantic_framework_config(config_path)
    paths = paths.validated()
    paths.evidence_root.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(paths.evidence_root / "run_status.json")
    try:
        completed = _reuse_completed_environment(
            paths.evidence_root,
            paths.checkout_root,
            config,
            project_root,
            project_commit,
        )
        if completed is not None:
            return completed
        runner = LiveCommandRunner(paths.log_root, status)
        uv_executable, uv_version, bootstrap_receipts = _resolve_uv_executable(
            runner,
            bootstrap_root=paths.cache_root / "bootstrap" / f"uv-{UV_VERSION}",
        )
        checkout = paths.checkout_root / "mmsegmentation"
        probe = paths.evidence_root / "hermetic-core-model-probe"
        cleanup_actions = repair_owned_path(
            runtime_root=paths.runtime_root,
            checkout=checkout,
            probe=probe,
            expected_commit=config.commit,
        )
        receipt = _execute_runtime(
            build_hermetic_commands(
                config,
                checkout,
                uv_executable=uv_executable,
                project_root=project_root,
                runtime_root=paths.runtime_root,
            ),
            interpreter=_python(paths.runtime_root),
            checkout=checkout,
            config=config,
            project_root=project_root,
            project_commit=project_commit,
            config_root=config_root,
            evidence_root=paths.evidence_root,
            cache_root=paths.cache_root,
            runner=runner,
        )
        lock_paths = _lock_paths(config, project_root)
        receipt.update(
            {
                "schema_version": "2.0",
                "record_type": "semantic_hermetic_runtime_receipt",
                "project_commit": project_commit,
                "lock_sha256": {path.name: sha256_file(path) for path in lock_paths},
                "cleanup_actions": cleanup_actions,
                "uv": {
                    "version": uv_version,
                    "executable_basename": uv_executable.name,
                    "bootstrap_commands": bootstrap_receipts,
                },
                "runtime_contract": {key: str(value) for key, value in paths.as_dict().items()},
            }
        )
        package = _evidence_zip(paths.evidence_root, receipt)
        receipt["evidence_package"] = package.name
        receipt["evidence_package_sha256"] = sha256_file(package)
        atomic_write_json(paths.evidence_root / "runtime_receipt.json", receipt)
        status.complete(last_checkpoint=receipt["core_model_probe"].get("evidence_package"))
        return receipt
    except BaseException as error:
        _record_failure(error, paths=paths, status=status, failed_stage="hermetic_runtime")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_semantic_framework_config(args.config)
    paths = RuntimePathContract(
        runtime_root=args.runtime_root,
        checkout_root=args.checkout_root,
        evidence_root=args.evidence_root,
        log_root=args.log_root,
        cache_root=args.cache_root,
        data_root=args.data_root,
    ).validated()
    if args.execute:
        result = install_hermetic_runtime(
            args.config,
            paths=paths,
            project_root=args.project_root,
            project_commit=args.project_commit,
            config_root=args.config_root,
        )
    else:
        result = {
            "schema_version": "2.0",
            "record_type": "semantic_hermetic_runtime_plan",
            "framework_commit": config.commit,
            "runtime_profile": "py311-cu121",
            "commands": [
                list(command)
                for command in build_hermetic_commands(
                    config,
                    paths.checkout_root / "mmsegmentation",
                    uv_executable=Path("resolved-uv"),
                    project_root=args.project_root,
                    runtime_root=paths.runtime_root,
                )
            ],
            "lock_sha256": {
                path.name: sha256_file(path) for path in _lock_paths(config, args.project_root)
            },
            "runtime_contract": paths.receipt(),
            "executes": False,
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
