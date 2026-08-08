#!/usr/bin/env python3
"""Bootstrap the EdgeGuard Colab runtime without importing hosted packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UV_VERSION = "0.8.8"
PYTHON_VERSION = "3.11.13"
MMSEG_REPOSITORY = "https://github.com/open-mmlab/mmsegmentation.git"
MMSEG_COMMIT = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"
MAIN_LOCK = "requirements/colab-py311-cu121.lock"
OPENMMLAB_LOCK = "requirements/colab-openmmlab.lock"
_HOST_ENVIRONMENT_KEYS = (
    "CONDA_PREFIX",
    "PIP_PREFIX",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TARGET",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
)


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


def _tail(path: Path, limit: int = 16000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - limit))
        return source.read().decode("utf-8", errors="replace")


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    """Stream one command to Colab and retain the complete bootstrap log."""
    print("BOOTSTRAP COMMAND:", " ".join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as sink:
        sink.write("\nCOMMAND: " + " ".join(command) + "\n")
        sink.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
        return_code = process.wait()
        sink.write(f"RETURN_CODE: {return_code}\n")
    if return_code:
        raise RuntimeError(
            f"bootstrap command failed with exit code {return_code}: {command}\n{_tail(log)}"
        )


def _require_bounded_target(path: Path, *, content_root: Path, name: str) -> Path:
    resolved = path.resolve()
    expected = (content_root / name).resolve()
    if resolved != expected or path.is_symlink():
        raise ValueError(f"unsafe Colab bootstrap target rejected: {path}")
    return resolved


def _validate_source(project_root: Path, project_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ValueError("project commit must be a full lowercase Git SHA")
    head = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != project_commit:
        raise ValueError("checked-out source does not match the pinned project commit")


def _validate_contract(project_root: Path) -> tuple[Path, Path, Path]:
    config = project_root / "configs/training/segmentation/framework_mmseg.yaml"
    main_lock = project_root / MAIN_LOCK
    openmmlab_lock = project_root / OPENMMLAB_LOCK
    for path in (config, main_lock, openmmlab_lock):
        if not path.is_file():
            raise FileNotFoundError(f"required runtime input is missing: {path}")
    config_text = config.read_text(encoding="utf-8")
    expected = {
        "uv_version": UV_VERSION,
        "python_version": PYTHON_VERSION,
        "commit": MMSEG_COMMIT,
        "lockfile": MAIN_LOCK,
    }
    for key, value in expected.items():
        pattern = rf'(?m)^{re.escape(key)}:\s*["\']?{re.escape(value)}["\']?\s*$'
        if re.search(pattern, config_text) is None:
            raise ValueError(f"runtime config does not match bootstrap contract: {key}")
    main_text = main_lock.read_text(encoding="utf-8")
    openmmlab_text = openmmlab_lock.read_text(encoding="utf-8")
    if re.search(r"(?m)^opencv-python==", main_text):
        raise ValueError("GUI OpenCV is forbidden in the Colab lock")
    for required in (
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "setuptools==80.9.0",
        "torch-2.1.1%2Bcu121-cp311-cp311-linux_x86_64.whl",
    ):
        if required not in main_text:
            raise ValueError(f"main Colab lock is missing the pinned input: {required}")
    for required in ("mmengine==0.10.7", "mmcv-lite==2.1.0", "--hash=sha256:"):
        if required not in openmmlab_text:
            raise ValueError(f"OpenMMLab lock is missing the pinned input: {required}")
    return config, main_lock, openmmlab_lock


def _uv_version(uv: Path) -> str | None:
    try:
        completed = subprocess.run([str(uv), "--version"], capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode:
        return None
    match = re.fullmatch(r"uv\s+([^\s]+)(?:\s+.*)?", completed.stdout.strip())
    return match.group(1) if match else None


def _find_private_uv(prefix: Path) -> Path | None:
    """Resolve both POSIX prefix layouts used by hosted Colab pip builds."""
    candidates = [prefix / "bin/uv", prefix / "local/bin/uv"]
    candidates.extend(path for path in sorted(prefix.glob("*/bin/uv")) if path not in candidates)
    for candidate in candidates:
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and os.access(candidate, os.X_OK)
            and _uv_version(candidate) == UV_VERSION
        ):
            return candidate.resolve()
    return None


def _private_uv(
    *, cache_root: Path, project_root: Path, environment: dict[str, str], log: Path
) -> Path:
    prefix = cache_root / "bootstrap" / f"uv-{UV_VERSION}"
    executable = _find_private_uv(prefix)
    if executable is not None:
        return executable
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            "--force-reinstall",
            "--prefix",
            str(prefix),
            f"uv=={UV_VERSION}",
        ],
        cwd=project_root,
        environment=environment,
        log=log,
    )
    executable = _find_private_uv(prefix)
    if executable is None:
        discovered = sorted(
            path.relative_to(prefix).as_posix() for path in prefix.rglob("uv") if path.is_file()
        )
        raise RuntimeError(
            "private uv bootstrap did not produce an executable exact uv 0.8.8; "
            f"discovered candidates: {discovered}"
        )
    return executable


def _checkout_head(checkout: Path) -> str | None:
    if not (checkout / ".git").is_dir():
        return None
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _prepare_checkout(
    checkout: Path,
    *,
    project_root: Path,
    environment: dict[str, str],
    log: Path,
) -> None:
    head = _checkout_head(checkout)
    if head == MMSEG_COMMIT:
        return
    if checkout.exists():
        if checkout.is_symlink() or checkout.name != "mmsegmentation":
            raise ValueError("unsafe MMSegmentation checkout target rejected")
        if (checkout / ".git").is_dir() or not any(checkout.iterdir()):
            shutil.rmtree(checkout)
        else:
            raise ValueError("unrecognized MMSegmentation checkout cannot be repaired safely")
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            MMSEG_REPOSITORY,
            str(checkout),
        ],
        cwd=project_root,
        environment=environment,
        log=log,
    )
    _run(
        ["git", "-C", str(checkout), "checkout", "--detach", MMSEG_COMMIT],
        cwd=project_root,
        environment=environment,
        log=log,
    )


def _bootstrap_environment(cache_root: Path) -> dict[str, str]:
    """Isolate the stdlib bootstrap from hosted Python and notebook state."""
    environment = os.environ.copy()
    for key in _HOST_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    for key in tuple(environment):
        if key.startswith("UV_"):
            environment.pop(key, None)
    cache_directories = {
        "HF_HOME": cache_root / "huggingface",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for path in cache_directories.values():
        path.mkdir(parents=True, exist_ok=True)
    environment.update({key: str(path) for key, path in cache_directories.items()})
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONNOUSERSITE": "1",
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(cache_root / "python"),
            "UV_PYTHON_PREFERENCE": "only-managed",
        }
    )
    return environment


def bootstrap(args: argparse.Namespace) -> dict[str, object]:
    """Create the locked runtime before any EdgeGuard module is imported."""
    project_root = args.project_root.resolve()
    content_root = args.content_root.resolve()
    runtime_root = _require_bounded_target(
        args.runtime_root, content_root=content_root, name="edgeguard-runtime"
    )
    checkout_root = _require_bounded_target(
        args.checkout_root, content_root=content_root, name="edgeguard-checkouts"
    )
    cache_root = _require_bounded_target(
        args.cache_root, content_root=content_root, name="edgeguard-cache"
    )
    log = args.log.resolve()
    _validate_source(project_root, args.project_commit)
    _, main_lock, openmmlab_lock = _validate_contract(project_root)
    environment = _bootstrap_environment(cache_root)
    uv = _private_uv(
        cache_root=cache_root,
        project_root=project_root,
        environment=environment,
        log=log,
    )
    interpreter = runtime_root / "bin/python"
    _run(
        [str(uv), "python", "install", PYTHON_VERSION],
        cwd=project_root,
        environment=environment,
        log=log,
    )
    if not interpreter.is_file():
        if runtime_root.exists():
            if runtime_root.is_symlink() or runtime_root.name != "edgeguard-runtime":
                raise ValueError("unsafe incomplete runtime target rejected")
            if (runtime_root / "pyvenv.cfg").is_file() or not any(runtime_root.iterdir()):
                shutil.rmtree(runtime_root)
            else:
                raise ValueError("unrecognized incomplete runtime cannot be repaired safely")
        _run(
            [str(uv), "venv", "--python", PYTHON_VERSION, str(runtime_root)],
            cwd=project_root,
            environment=environment,
            log=log,
        )
    _run(
        [
            str(uv),
            "pip",
            "sync",
            "--python",
            str(interpreter),
            "--strict",
            "--require-hashes",
            str(main_lock),
        ],
        cwd=project_root,
        environment=environment,
        log=log,
    )
    _run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "--require-hashes",
            "-r",
            str(openmmlab_lock),
        ],
        cwd=project_root,
        environment=environment,
        log=log,
    )
    checkout = checkout_root / "mmsegmentation"
    checkout_root.mkdir(parents=True, exist_ok=True)
    _prepare_checkout(
        checkout,
        project_root=project_root,
        environment=environment,
        log=log,
    )
    for editable in (checkout, project_root):
        _run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--no-build-isolation",
                "--no-deps",
                "-e",
                str(editable),
            ],
            cwd=project_root,
            environment=environment,
            log=log,
        )
    probe = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import json,platform; print(json.dumps({'python': platform.python_version()}))",
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    identity = json.loads(probe.stdout)
    if identity.get("python") != PYTHON_VERSION:
        raise RuntimeError("runtime interpreter does not match Python 3.11.13")
    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_stdlib_colab_bootstrap",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": args.project_commit,
        "interpreter": str(interpreter),
        "uv": {"version": UV_VERSION, "path": str(uv)},
        "mmseg_root": str(checkout),
        "mmseg_commit": MMSEG_COMMIT,
        "lock_sha256": {
            main_lock.name: _sha256(main_lock),
            openmmlab_lock.name: _sha256(openmmlab_lock),
        },
        "log": str(log),
    }
    _atomic_json(args.output.resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = bootstrap(args)
    except BaseException as error:
        failure = {
            "schema_version": "1.0",
            "record_type": "edgeguard_stdlib_colab_bootstrap_failure",
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error)[-20000:],
            "log": str(args.log.resolve()),
        }
        _atomic_json(args.output.resolve().with_name("bootstrap-failure.json"), failure)
        print(_canonical_json(failure), file=sys.stderr, flush=True)
        raise
    print(_canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
