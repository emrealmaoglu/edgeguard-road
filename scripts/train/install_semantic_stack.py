"""Install and prove the two-path OpenMIM-free Colab semantic stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, atomic_write_json
from edgeguard.training.config import load_semantic_framework_config
from edgeguard.training.contracts import SemanticFrameworkConfig

PURE_RUNTIME_PINS = (
    "addict==2.4.0",
    "packaging==24.2",
    "prettytable==3.16.0",
    "termcolor==3.1.0",
    "tqdm==4.67.1",
    "yapf==0.43.0",
    "tensorboard==2.19.0",
    "Pillow==11.3.0",
    "pydantic==2.11.7",
    "PyYAML==6.0.2",
    "matplotlib==3.10.5",
    "scipy==1.16.1",
    "terminaltables==3.1.10",
)
UV_VERSION = "0.8.8"
PATH_A_ROOT = Path("/content/edgeguard-runtime-current")
PATH_B_ROOT = Path("/content/edgeguard-runtime-py311")
DEFAULT_LOG_ROOT = Path("/content/edgeguard-logs")
OWNED_RUNTIME_NAMES = {"edgeguard-runtime-current", "edgeguard-runtime-py311"}
OWNED_CHECKOUT_NAMES = {"mmseg-path-a", "mmseg-path-b"}


def _python(root: Path) -> Path:
    return root / "bin" / "python"


def _resolve_uv_executable(runner: LiveCommandRunner) -> tuple[Path, str, list[dict[str, Any]]]:
    """Bootstrap bounded uv once and resolve its actual executable from PATH."""
    receipts: list[dict[str, Any]] = []
    resolved = shutil.which("uv")
    if resolved is None:
        receipts.append(
            runner.run(
                "bootstrap-uv",
                (sys.executable, "-m", "pip", "install", f"uv=={UV_VERSION}"),
                stage_index=1,
                stage_total=1,
            )
        )
        resolved = shutil.which("uv")
    if resolved is None:
        raise RuntimeError("uv bootstrap completed but no executable was found on PATH")
    executable = Path(resolved).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("resolved uv path is not an executable regular file")
    version_output = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version_output != f"uv {UV_VERSION}":
        raise RuntimeError(f"unexpected uv version: {version_output}")
    return executable, UV_VERSION, receipts


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
    """Preserve resumable owned venvs and remove only recognizable incomplete ones."""
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
    """Reuse the exact checkout or remove only an incomplete owned clone."""
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
    if probe.is_symlink() or not probe.is_dir() or not probe.name.endswith("-five-model-probe"):
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
    """Apply the bounded, testable repair policy for one compatibility path."""
    actions = (
        _repair_runtime_target(runtime_root),
        _repair_checkout_target(checkout, expected_commit),
        _repair_probe_target(probe),
    )
    return [action for action in actions if action is not None]


def build_path_a_commands(
    config: SemanticFrameworkConfig,
    checkout: Path,
    *,
    uv_executable: Path,
    project_root: Path,
    runtime_root: Path = PATH_A_ROOT,
) -> tuple[tuple[str, ...], ...]:
    """Build a hosted-stack-preserving Python-3.12-compatible command sequence."""
    interpreter = _python(runtime_root)
    uv = str(uv_executable)
    return (
        (
            uv,
            "venv",
            "--system-site-packages",
            "--python",
            sys.executable,
            str(runtime_root),
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--upgrade-strategy",
            "only-if-needed",
            *PURE_RUNTIME_PINS,
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            f"mmengine=={config.mmengine_version}",
            f"{config.preferred_mmcv_distribution}=={config.mmcv_version}",
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
        (uv, "pip", "install", "--python", str(interpreter), "--no-deps", "-e", str(checkout)),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "-e",
            str(project_root),
        ),
    )


def build_path_b_commands(
    config: SemanticFrameworkConfig,
    checkout: Path,
    *,
    uv_executable: Path,
    project_root: Path,
    runtime_root: Path = PATH_B_ROOT,
) -> tuple[tuple[str, ...], ...]:
    """Build the isolated Python 3.11/CUDA 12.1 fallback sequence."""
    uv = str(uv_executable)
    interpreter = _python(runtime_root)
    torch_index = "https://download.pytorch.org/whl/cu121"
    mmcv_index = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
    return (
        (uv, "python", "install", config.fallback_python_version),
        (uv, "venv", "--python", config.fallback_python_version, str(runtime_root)),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            f"numpy=={config.fallback_numpy_version}",
            *PURE_RUNTIME_PINS,
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--index-url",
            torch_index,
            f"torch=={config.fallback_torch_version}",
            f"torchvision=={config.fallback_torchvision_version}",
            f"torchaudio=={config.fallback_torchaudio_version}",
        ),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            f"mmengine=={config.mmengine_version}",
            f"mmcv=={config.mmcv_version}",
            "--find-links",
            mmcv_index,
            "--only-binary=:all:",
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
        (uv, "pip", "install", "--python", str(interpreter), "--no-deps", "-e", str(checkout)),
        (
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "-e",
            str(project_root),
        ),
    )


def _hosted_runtime_summary() -> dict[str, Any]:
    try:
        torch_version = importlib.metadata.version("torch")
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("hosted Colab must provide Torch and TorchVision") from error
    import torch

    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _environment_probe(interpreter: Path, checkout: Path) -> dict[str, Any]:
    script = """
import importlib.metadata, json, platform, subprocess, torch
try:
    mmcv_distribution = 'mmcv'
    mmcv_version = importlib.metadata.version(mmcv_distribution)
except importlib.metadata.PackageNotFoundError:
    mmcv_distribution = 'mmcv-lite'
    mmcv_version = importlib.metadata.version(mmcv_distribution)
payload = {
  'python_version': platform.python_version(),
  'torch_version': importlib.metadata.version('torch'),
  'torchvision_version': importlib.metadata.version('torchvision'),
  'cuda_runtime': torch.version.cuda,
  'cuda_available': bool(torch.cuda.is_available()),
  'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'mmengine_version': importlib.metadata.version('mmengine'),
  'mmcv_distribution': mmcv_distribution,
  'mmcv_version': mmcv_version,
  'mmsegmentation_version': importlib.metadata.version('mmsegmentation'),
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
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload["mmsegmentation_commit"] = commit
    return payload


def _validate_identity(
    identity: dict[str, Any], config: SemanticFrameworkConfig, *, path_name: str
) -> None:
    if identity.get("mmsegmentation_commit") != config.commit:
        raise ValueError("MMSegmentation checkout identity mismatch")
    if identity.get("mmengine_version") != config.mmengine_version:
        raise ValueError("MMEngine version mismatch")
    if identity.get("mmcv_version") != config.mmcv_version:
        raise ValueError("MMCV version mismatch")
    expected_distribution = "mmcv-lite" if path_name == "hosted_current" else "mmcv"
    if identity.get("mmcv_distribution") != expected_distribution:
        raise ValueError("MMCV distribution mismatch")
    if identity.get("cuda_available") is not True:
        raise ValueError("selected compatibility path has no CUDA")
    if path_name == "isolated_py311" and identity.get("python_version", "").split(".")[:2] != [
        "3",
        "11",
    ]:
        raise ValueError("fallback interpreter is not Python 3.11")


def _preflight_path_a(
    config: SemanticFrameworkConfig, runner: LiveCommandRunner, directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--no-deps",
        "--dest",
        str(directory),
        f"mmengine=={config.mmengine_version}",
        f"{config.preferred_mmcv_distribution}=={config.mmcv_version}",
    )
    runner.run("path-a-wheel-preflight", command, stage_index=1, stage_total=1)


def _probe_command(
    interpreter: Path,
    *,
    project_root: Path,
    project_commit: str,
    config_root: Path,
    checkout: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
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
    )


def _execute_path(
    path_name: str,
    commands: tuple[tuple[str, ...], ...],
    *,
    interpreter: Path,
    checkout: Path,
    config: SemanticFrameworkConfig,
    project_root: Path,
    project_commit: str,
    config_root: Path,
    evidence_root: Path,
    runner: LiveCommandRunner,
) -> dict[str, Any]:
    started = time.perf_counter()
    command_receipts = []
    for index, command in enumerate(commands, start=1):
        if command[0] == "git" and checkout.is_dir():
            if (
                subprocess.run(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                == config.commit
            ):
                continue
        if len(command) > 1 and command[1] == "venv" and interpreter.is_file():
            continue
        command_receipts.append(
            runner.run(
                f"{path_name}-install-{index}",
                command,
                stage_index=index,
                stage_total=len(commands) + 1,
                cwd=project_root,
            )
        )
    identity = _environment_probe(interpreter, checkout)
    _validate_identity(identity, config, path_name=path_name)
    probe_output = evidence_root / f"{path_name}-five-model-probe"
    if probe_output.exists() and not (probe_output / "completion.json").is_file():
        raise ValueError("incomplete prior compatibility probe must be inspected")
    if not probe_output.exists():
        command_receipts.append(
            runner.run(
                f"{path_name}-five-model-probe",
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
    if (
        completion.get("model_count") != 5
        or completion.get("checkpoint_resume_verified") is not True
    ):
        raise ValueError("five-model compatibility probe is incomplete")
    return {
        "selected_path": path_name,
        "interpreter": str(interpreter),
        "environment": identity,
        "framework_commit": config.commit,
        "install_duration_seconds": time.perf_counter() - started,
        "commands": [list(command) for command in commands],
        "command_receipts": command_receipts,
        "five_model_probe": completion,
    }


def _evidence_zip(evidence_root: Path, receipt: dict[str, Any]) -> Path:
    receipt_path = evidence_root / "compatibility_receipt.json"
    atomic_write_json(receipt_path, receipt)
    package = evidence_root / "semantic-compatibility-evidence.zip"
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
    project_commit: str,
) -> dict[str, Any] | None:
    """Return an already completed compatible receipt or reject corrupt completion."""
    receipt_path = evidence_root / "compatibility_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected = receipt.get("selected_path")
    if selected not in {"hosted_current", "isolated_py311"}:
        raise ValueError("completed compatibility receipt has an invalid selected path")
    interpreter_value = receipt.get("interpreter")
    if not isinstance(interpreter_value, str):
        raise ValueError("completed compatibility receipt has no interpreter identity")
    interpreter = Path(interpreter_value)
    checkout_name = "mmseg-path-a" if selected == "hosted_current" else "mmseg-path-b"
    checkout = checkout_root / checkout_name
    identity = _environment_probe(interpreter, checkout)
    _validate_identity(identity, config, path_name=selected)
    probe = receipt.get("five_model_probe")
    if not isinstance(probe, dict) or probe.get("project_commit") != project_commit:
        raise ValueError("completed compatibility receipt belongs to another project commit")
    if probe.get("model_count") != 5 or probe.get("checkpoint_resume_verified") is not True:
        raise ValueError("completed compatibility receipt lacks the five-model acceptance")
    return {**receipt, "reused_completed_environment": True}


def install_compatibility_cascade(
    config_path: Path,
    *,
    project_root: Path,
    project_commit: str,
    config_root: Path,
    checkout_root: Path,
    evidence_root: Path,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> dict[str, Any]:
    """Select the first path that passes all five model and resume checks."""
    config = load_semantic_framework_config(config_path)
    hosted = _hosted_runtime_summary()
    if hosted["cuda_available"] is not True:
        raise RuntimeError("compatibility cascade requires a CUDA Colab runtime")
    evidence_root.mkdir(parents=True, exist_ok=True)
    cleanup_actions: list[dict[str, str]] = []
    try:
        completed = _reuse_completed_environment(
            evidence_root, checkout_root, config, project_commit
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        if evidence_root.name != "edgeguard-compatibility":
            raise
        (evidence_root / "compatibility_receipt.json").unlink(missing_ok=True)
        (evidence_root / "semantic-compatibility-evidence.zip").unlink(missing_ok=True)
        cleanup_actions.append(
            {
                "target": evidence_root.name,
                "action": "removed_invalid_completion_receipt",
            }
        )
        completed = None
    if completed is not None:
        return completed
    stale_failure = evidence_root / "compatibility_failures.json"
    if stale_failure.is_file():
        if evidence_root.name != "edgeguard-compatibility":
            raise ValueError("refusing to remove failure evidence outside the owned root")
        stale_failure.unlink()
        cleanup_actions.append(
            {
                "target": stale_failure.name,
                "action": "removed_stale_failure_receipt_before_retry",
            }
        )
    status = LongRunStatus(evidence_root / "run_status.json")
    runner = LiveCommandRunner(log_root, status)
    failures: list[dict[str, str]] = []
    uv_executable, uv_version, bootstrap_receipts = _resolve_uv_executable(runner)

    path_a_checkout = checkout_root / "mmseg-path-a"
    cleanup_actions.extend(
        repair_owned_path(
            runtime_root=PATH_A_ROOT,
            checkout=path_a_checkout,
            probe=evidence_root / "hosted_current-five-model-probe",
            expected_commit=config.commit,
        )
    )
    try:
        _preflight_path_a(config, runner, evidence_root / "path-a-wheel-preflight")
        receipt = _execute_path(
            "hosted_current",
            build_path_a_commands(
                config,
                path_a_checkout,
                uv_executable=uv_executable,
                project_root=project_root,
            ),
            interpreter=_python(PATH_A_ROOT),
            checkout=path_a_checkout,
            config=config,
            project_root=project_root,
            project_commit=project_commit,
            config_root=config_root,
            evidence_root=evidence_root,
            runner=runner,
        )
        if receipt["environment"].get("torch_version") != hosted["torch_version"]:
            raise ValueError("hosted Path A did not preserve the hosted Torch version")
        if receipt["environment"].get("torchvision_version") != hosted["torchvision_version"]:
            raise ValueError("hosted Path A did not preserve the hosted TorchVision version")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        failures.append({"path": "hosted_current", "error": str(error)[:1000]})
        path_b_checkout = checkout_root / "mmseg-path-b"
        cleanup_actions.extend(
            repair_owned_path(
                runtime_root=PATH_B_ROOT,
                checkout=path_b_checkout,
                probe=evidence_root / "isolated_py311-five-model-probe",
                expected_commit=config.commit,
            )
        )
        try:
            receipt = _execute_path(
                "isolated_py311",
                build_path_b_commands(
                    config,
                    path_b_checkout,
                    uv_executable=uv_executable,
                    project_root=project_root,
                ),
                interpreter=_python(PATH_B_ROOT),
                checkout=path_b_checkout,
                config=config,
                project_root=project_root,
                project_commit=project_commit,
                config_root=config_root,
                evidence_root=evidence_root,
                runner=runner,
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as fallback_error:
            failures.append({"path": "isolated_py311", "error": str(fallback_error)[:1000]})
            status.fail("both compatibility paths failed")
            atomic_write_json(
                evidence_root / "compatibility_failures.json",
                {
                    "failures": failures,
                    "cleanup_actions": cleanup_actions,
                    "uv": {"version": uv_version, "executable_basename": uv_executable.name},
                },
            )
            raise RuntimeError(f"both compatibility paths failed: {failures}") from fallback_error
    receipt.update(
        {
            "schema_version": "1.0",
            "record_type": "semantic_compatibility_cascade_receipt",
            "hosted_runtime_before_install": hosted,
            "failed_paths": failures,
            "cleanup_actions": cleanup_actions,
            "uv": {
                "version": uv_version,
                "executable_basename": uv_executable.name,
                "bootstrap_commands": bootstrap_receipts,
            },
            "project_commit": project_commit,
        }
    )
    package = _evidence_zip(evidence_root, receipt)
    receipt["evidence_package"] = package.name
    receipt["evidence_package_sha256"] = sha256_file(package)
    atomic_write_json(evidence_root / "compatibility_receipt.json", receipt)
    status.complete(last_checkpoint=receipt["five_model_probe"].get("evidence_package"))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_semantic_framework_config(args.config)
    if args.execute:
        result = install_compatibility_cascade(
            args.config,
            project_root=args.project_root,
            project_commit=args.project_commit,
            config_root=args.config_root,
            checkout_root=args.checkout_root,
            evidence_root=args.evidence_root,
            log_root=args.log_root,
        )
    else:
        result = {
            "schema_version": "1.0",
            "record_type": "semantic_compatibility_cascade_plan",
            "framework_commit": config.commit,
            "path_a": [
                list(command)
                for command in build_path_a_commands(
                    config,
                    args.checkout_root / "mmseg-path-a",
                    uv_executable=Path("<resolved-uv>"),
                    project_root=args.project_root,
                )
            ],
            "path_b": [
                list(command)
                for command in build_path_b_commands(
                    config,
                    args.checkout_root / "mmseg-path-b",
                    uv_executable=Path("<resolved-uv>"),
                    project_root=args.project_root,
                )
            ],
            "executes": False,
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
