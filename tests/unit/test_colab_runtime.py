from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from edgeguard.rescue.colab_runtime import resolve_colab_runtime


def _receipt(tmp_path: Path, selected: str) -> Path:
    runtime_current = tmp_path / "runtime-current"
    runtime_py311 = tmp_path / "runtime-py311"
    selected_runtime = runtime_current if selected == "hosted_current" else runtime_py311
    interpreter = selected_runtime / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    checkout_root = tmp_path / "checkouts"
    checkout = checkout_root / ("mmseg-path-a" if selected == "hosted_current" else "mmseg-path-b")
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
    receipt = tmp_path / "compatibility_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "record_type": "semantic_compatibility_cascade_receipt",
                "selected_path": selected,
                "project_commit": "a" * 40,
                "framework_commit": commit,
                "runtime_contract": {
                    "runtime_current_root": str(runtime_current),
                    "runtime_py311_root": str(runtime_py311),
                    "checkout_root": str(checkout_root),
                },
                "environment": {"cuda_available": True},
                "five_model_probe": {"model_count": 5, "checkpoint_resume_verified": True},
            }
        ),
        encoding="utf-8",
    )
    return receipt


@pytest.mark.parametrize("selected", ["hosted_current", "isolated_py311"])
def test_colab_runtime_resolves_the_selected_path(tmp_path: Path, selected: str) -> None:
    receipt = _receipt(tmp_path, selected)
    result = resolve_colab_runtime(receipt, expected_project_commit="a" * 40)
    expected = "mmseg-path-a" if selected == "hosted_current" else "mmseg-path-b"
    assert Path(result["interpreter"]).is_file()
    assert Path(result["mmseg_root"]).name == expected


def test_colab_runtime_rejects_commit_drift(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, "hosted_current")
    with pytest.raises(ValueError, match="another project commit"):
        resolve_colab_runtime(receipt, expected_project_commit="b" * 40)
