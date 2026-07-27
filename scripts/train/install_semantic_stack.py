"""Install and prove the two-path OpenMIM-free Colab semantic stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
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


def _python(root: Path) -> Path:
    return root / "bin" / "python"


def _pip(interpreter: Path, *arguments: str) -> tuple[str, ...]:
    return (str(interpreter), "-m", "pip", *arguments)


def build_path_a_commands(
    config: SemanticFrameworkConfig,
    checkout: Path,
    *,
    project_root: Path,
    runtime_root: Path = PATH_A_ROOT,
) -> tuple[tuple[str, ...], ...]:
    """Build a hosted-stack-preserving Python-3.12-compatible command sequence."""
    interpreter = _python(runtime_root)
    return (
        (sys.executable, "-m", "venv", "--system-site-packages", str(runtime_root)),
        _pip(interpreter, "install", "--upgrade-strategy", "only-if-needed", *PURE_RUNTIME_PINS),
        _pip(
            interpreter,
            "install",
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
        _pip(interpreter, "install", "--no-deps", "-e", str(checkout)),
        _pip(interpreter, "install", "--no-deps", "-e", str(project_root)),
    )


def build_path_b_commands(
    config: SemanticFrameworkConfig,
    checkout: Path,
    *,
    project_root: Path,
    runtime_root: Path = PATH_B_ROOT,
    uv_root: Path = Path("/content/edgeguard-uv"),
) -> tuple[tuple[str, ...], ...]:
    """Build the isolated Python 3.11/CUDA 12.1 fallback sequence."""
    uv = uv_root / "bin" / "uv"
    interpreter = _python(runtime_root)
    torch_index = "https://download.pytorch.org/whl/cu121"
    mmcv_index = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
    return (
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--prefix",
            str(uv_root),
            f"uv=={UV_VERSION}",
        ),
        (str(uv), "venv", "--python", config.fallback_python_version, str(runtime_root)),
        (
            str(uv),
            "pip",
            "install",
            "--python",
            str(interpreter),
            f"numpy=={config.fallback_numpy_version}",
            *PURE_RUNTIME_PINS,
        ),
        (
            str(uv),
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
            str(uv),
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
        (str(uv), "pip", "install", "--python", str(interpreter), "--no-deps", "-e", str(checkout)),
        (
            str(uv),
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
    command = _pip(
        Path(sys.executable),
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
        if command[0] == sys.executable and "venv" in command and interpreter.is_file():
            continue
        if command[0].endswith("/uv") and "venv" in command and interpreter.is_file():
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
    status = LongRunStatus(evidence_root / "run_status.json")
    runner = LiveCommandRunner(log_root, status)
    failures: list[dict[str, str]] = []

    path_a_checkout = checkout_root / "mmseg-path-a"
    try:
        _preflight_path_a(config, runner, evidence_root / "path-a-wheel-preflight")
        receipt = _execute_path(
            "hosted_current",
            build_path_a_commands(config, path_a_checkout, project_root=project_root),
            interpreter=_python(PATH_A_ROOT),
            checkout=path_a_checkout,
            config=config,
            project_root=project_root,
            project_commit=project_commit,
            config_root=config_root,
            evidence_root=evidence_root,
            runner=runner,
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        failures.append({"path": "hosted_current", "error": str(error)[:1000]})
        path_b_checkout = checkout_root / "mmseg-path-b"
        try:
            receipt = _execute_path(
                "isolated_py311",
                build_path_b_commands(config, path_b_checkout, project_root=project_root),
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
            atomic_write_json(evidence_root / "compatibility_failures.json", {"failures": failures})
            raise RuntimeError(f"both compatibility paths failed: {failures}") from fallback_error
    receipt.update(
        {
            "schema_version": "1.0",
            "record_type": "semantic_compatibility_cascade_receipt",
            "hosted_runtime_before_install": hosted,
            "failed_paths": failures,
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
                    config, args.checkout_root / "mmseg-path-a", project_root=args.project_root
                )
            ],
            "path_b": [
                list(command)
                for command in build_path_b_commands(
                    config, args.checkout_root / "mmseg-path-b", project_root=args.project_root
                )
            ],
            "executes": False,
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
