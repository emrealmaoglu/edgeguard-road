from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bootstrap_colab_runtime, run_colab_master

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    ("scripts/bootstrap_colab_runtime.py", "scripts/run_colab_master.py"),
)
def test_host_entrypoints_import_only_the_standard_library(relative: str) -> None:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    assert roots.isdisjoint(
        {"edgeguard", "numpy", "packaging", "pydantic", "torch", "yaml", "mmcv", "mmseg"}
    )


def test_bootstrap_cli_loads_when_site_packages_are_disabled() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts/bootstrap_colab_runtime.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--runtime-root" in completed.stdout


def test_bootstrap_validates_the_checked_in_lock_contract() -> None:
    config, main_lock, openmmlab_lock = bootstrap_colab_runtime._validate_contract(ROOT)
    assert config.name == "framework_mmseg.yaml"
    assert main_lock.name == "colab-py311-cu121.lock"
    assert openmmlab_lock.name == "colab-openmmlab.lock"


@pytest.mark.parametrize("relative", ("bin/uv", "local/bin/uv"))
def test_private_uv_discovery_accepts_both_posix_prefix_layouts(
    tmp_path: Path, relative: str
) -> None:
    executable = tmp_path / relative
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\necho 'uv 0.8.8'\n", encoding="utf-8")
    executable.chmod(0o755)
    assert bootstrap_colab_runtime._find_private_uv(tmp_path) == executable.resolve()


def test_runtime_path_uses_the_discovered_local_bin_layout(tmp_path: Path) -> None:
    private_uv = tmp_path / "cache/bootstrap/uv-0.8.8/local/bin/uv"
    private_uv.parent.mkdir(parents=True)
    private_uv.touch()
    environment = run_colab_master._runtime_environment(
        project_root=ROOT,
        cache_root=tmp_path / "cache",
        runtime_python=tmp_path / "runtime/bin/python",
        uv_executable=private_uv,
    )
    assert environment["PATH"].split(os.pathsep)[0] == str(private_uv.parent)


def test_runtime_environment_prioritizes_wheel_cuda_libraries(tmp_path: Path) -> None:
    runtime_python = tmp_path / "runtime/bin/python"
    torch_lib = tmp_path / "runtime/lib/python3.11/site-packages/torch/lib"
    nvidia_lib = tmp_path / "runtime/lib/python3.11/site-packages/nvidia/cudnn/lib"
    torch_lib.mkdir(parents=True)
    nvidia_lib.mkdir(parents=True)
    uv = tmp_path / "uv-prefix/local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch()
    previous = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib64-nvidia"
    try:
        environment = run_colab_master._runtime_environment(
            project_root=ROOT,
            cache_root=tmp_path / "cache",
            runtime_python=runtime_python,
            uv_executable=uv,
        )
    finally:
        if previous is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous
    entries = environment["LD_LIBRARY_PATH"].split(os.pathsep)
    assert entries[:2] == [str(torch_lib), str(nvidia_lib)]
    assert entries[-2:] == ["/usr/local/cuda/lib64", "/usr/lib64-nvidia"]
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_runtime_environment_firewalls_hosted_colab_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hostile = {
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        "PYTHONHOME": "/host/python",
        "PYTHONSTARTUP": "/host/startup.py",
        "PYTHONUSERBASE": "/host/userbase",
        "VIRTUAL_ENV": "/host/venv",
        "CONDA_PREFIX": "/host/conda",
        "PIP_TARGET": "/host/target",
        "PIP_PREFIX": "/host/prefix",
        "PIP_REQUIRE_VIRTUALENV": "1",
        "UV_CACHE_DIR": "/host/uv-cache",
        "UV_PYTHON": "/host/python",
        "CUDA_VISIBLE_DEVICES": "0",
        "HTTPS_PROXY": "http://proxy.invalid",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    uv = tmp_path / "uv-prefix/local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch()

    environment = run_colab_master._runtime_environment(
        project_root=ROOT,
        cache_root=tmp_path / "cache",
        runtime_python=tmp_path / "runtime/bin/python",
        uv_executable=uv,
    )

    assert environment["MPLBACKEND"] == "Agg"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HTTPS_PROXY"] == "http://proxy.invalid"
    assert environment["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "cache/uv")
    assert re.fullmatch(r"[0-9a-f]{64}", environment["EDGEGUARD_ENVIRONMENT_CONTRACT_SHA256"])
    for key in (
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_REQUIRE_VIRTUALENV",
        "UV_PYTHON",
    ):
        assert key not in environment
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME"):
        assert Path(environment[key]).is_dir()


def test_bootstrap_environment_ignores_host_uv_and_inline_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.setenv("UV_CACHE_DIR", "/host/uv-0.11.19")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    environment = bootstrap_colab_runtime._bootstrap_environment(tmp_path / "cache")
    assert environment["MPLBACKEND"] == "Agg"
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "cache/uv")
    assert environment["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert "VIRTUAL_ENV" not in environment


def test_runtime_environment_renders_headless_png_under_hostile_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    uv = tmp_path / "uv-prefix/local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch()
    environment = run_colab_master._runtime_environment(
        project_root=ROOT,
        cache_root=tmp_path / "cache",
        runtime_python=Path(sys.executable),
        uv_executable=uv,
    )
    script = """
import json, tempfile
from pathlib import Path
import matplotlib
from matplotlib.figure import Figure
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'probe.png'
    figure = Figure(figsize=(1, 1))
    figure.subplots().plot([0, 1], [0, 1])
    figure.savefig(output)
    print(json.dumps({'backend': matplotlib.get_backend().lower(), 'bytes': output.stat().st_size}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["backend"] == "agg"
    assert payload["bytes"] > 0


def test_master_child_command_streams_and_preserves_failure_tail(tmp_path: Path) -> None:
    log = tmp_path / "master.log"
    with pytest.raises(RuntimeError, match="visible-root-cause"):
        run_colab_master._run(
            [sys.executable, "-c", "print('visible-root-cause'); raise SystemExit(7)"],
            project_root=ROOT,
            child_log=log,
        )
    assert "visible-root-cause" in log.read_text(encoding="utf-8")
    assert "RETURN_CODE: 7" in log.read_text(encoding="utf-8")


def test_master_json_parser_ignores_human_progress_lines() -> None:
    payload = run_colab_master._last_json('progress\n{"status":"completed","count":2}\n')
    assert payload == {"status": "completed", "count": 2}


def test_resource_gate_accepts_colab_gpu_output_with_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="NVIDIA L4, 23034 MiB\n")

    original_read_text = Path.read_text

    def fake_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/meminfo"):
            return "MemTotal:       52428800 kB\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(
        run_colab_master.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * 1024**3),
    )
    result = run_colab_master._resource_gate(tmp_path)
    assert result["gpu"] == "NVIDIA L4"
    assert result["gpu_memory_mib"] == 23034


def test_master_uses_v3_work_root_and_bootstraps_before_project_imports() -> None:
    source = (ROOT / "scripts/run_colab_master.py").read_text(encoding="utf-8")
    assert 'work_root = content_root / "edgeguard-work-v3"' in source
    assert "scripts/bootstrap_colab_runtime.py" in source
    assert "from edgeguard" not in source
    assert source.index('stage("stdlib-hermetic-bootstrap"') < source.index(
        'stage("five-model-runtime-canary")'
    )


def test_failed_canary_evidence_is_preserved_outside_the_next_attempt(tmp_path: Path) -> None:
    evidence = tmp_path / "edgeguard-evidence"
    evidence.mkdir()
    (evidence / "failure.json").write_text('{"status":"failed"}\n', encoding="utf-8")
    (evidence / "partial.txt").write_text("keep", encoding="utf-8")

    quarantined = run_colab_master._quarantine_failed_runtime_evidence(evidence)

    assert quarantined is not None
    assert quarantined.name.startswith("edgeguard-evidence.failed-")
    assert (quarantined / "failure.json").is_file()
    assert (quarantined / "partial.txt").read_text(encoding="utf-8") == "keep"
    assert not evidence.exists()


def test_completed_runtime_evidence_is_not_quarantined(tmp_path: Path) -> None:
    evidence = tmp_path / "edgeguard-evidence"
    evidence.mkdir()
    (evidence / "failure.json").write_text('{"status":"old"}\n', encoding="utf-8")
    (evidence / "runtime_receipt.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    assert run_colab_master._quarantine_failed_runtime_evidence(evidence) is None
    assert evidence.is_dir()
