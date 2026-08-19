"""Compile and execute the single Colab notebook in claim-safe local mode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file

NOTEBOOKS = ("EdgeGuard_Master_Colab.ipynb",)


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def execute_notebook(path: Path, environment: dict[str, str]) -> dict[str, Any]:
    """Compile and execute every cell without Drive, network, data, or GPU access."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__edgeguard_notebook_harness__"}
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        count = 0
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{path.name}:cell-{count}", "exec"), namespace)
            count += 1
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result = namespace.get("MASTER_RESULT")
    if not isinstance(result, dict) or result.get("scientific_status") != "not_run":
        raise ValueError("master notebook local contract did not preserve the claim boundary")
    return {
        "notebook": path.name,
        "code_cells": count,
        "sha256": sha256_file(path),
        "contract": result,
    }


def run_harness(repository: Path, campaign_root: Path) -> dict[str, Any]:
    """Run the one master wrapper with local filesystem adapters."""
    campaign_root.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(repository)
    environment = {
        "EDGEGUARD_NOTEBOOK_LOCAL_TEST": "1",
        "EDGEGUARD_PROJECT_ROOT": str(repository.resolve()),
        "EDGEGUARD_TEST_DRIVE_ROOT": str((campaign_root / "drive").resolve()),
        "EDGEGUARD_TEST_CONTENT_ROOT": str((campaign_root / "content").resolve()),
    }
    notebook_root = repository / "notebooks"
    receipts = [execute_notebook(notebook_root / name, environment) for name in NOTEBOOKS]
    return {
        "schema_version": "2.0",
        "record_type": "campaign_notebook_harness_receipt",
        "status": "passed",
        "git_commit": commit,
        "notebooks": receipts,
        "drive_adapter": "local_temporary_directory",
        "colab_adapter": "local_python_process",
        "scientific_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(run_harness(args.repository, args.campaign_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
