from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from edgeguard.rescue.colab_runtime import resolve_colab_runtime


def _receipt(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    interpreter = runtime / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    checkout_root = tmp_path / "checkouts"
    checkout = checkout_root / "mmsegmentation"
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "fixture@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Fixture"], check=True)
    (checkout / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = tmp_path / "runtime_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "record_type": "semantic_hermetic_runtime_receipt",
                "runtime_profile": "py311-cu121",
                "project_commit": "a" * 40,
                "framework_commit": commit,
                "runtime_contract": {
                    "runtime_root": str(runtime),
                    "checkout_root": str(checkout_root),
                },
                "environment": {"cuda_available": True},
                "core_model_probe": {
                    "model_count": 5,
                    "checkpoint_resume_verified": True,
                    "checkpoint_resume_model_count": 5,
                    "fp16_finite_model_count": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_colab_runtime_resolves_the_hermetic_profile(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    result = resolve_colab_runtime(receipt, expected_project_commit="a" * 40)
    assert Path(result["interpreter"]).is_file()
    assert Path(result["mmseg_root"]).name == "mmsegmentation"
    assert result["runtime_profile"] == "py311-cu121"


def test_colab_runtime_rejects_commit_drift(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    with pytest.raises(ValueError, match="another project commit"):
        resolve_colab_runtime(receipt, expected_project_commit="b" * 40)
