from __future__ import annotations

import ast
import os
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
