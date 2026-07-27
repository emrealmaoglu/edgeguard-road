"""Compile and execute the thin campaign notebooks with local adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from edgeguard.campaign.contracts import topological_stages
from edgeguard.serialization import canonical_json, sha256_file

NOTEBOOKS = (
    "00_campaign_control.ipynb",
    "10_semantic_campaign.ipynb",
    "20_ood_calibration_risk.ipynb",
    "30_detection_temporal_fusion.ipynb",
    "40_export_and_reporting.ipynb",
)


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def execute_notebook(path: Path, environment: dict[str, str]) -> dict[str, Any]:
    """Compile and execute code cells without a notebook-only runtime dependency."""
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
            compiled = compile(source, f"{path.name}:cell-{count}", "exec")
            exec(compiled, namespace)
            count += 1
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"notebook": path.name, "code_cells": count, "sha256": sha256_file(path)}


def run_harness(repository: Path, campaign_root: Path) -> dict[str, Any]:
    """Run all wrappers over one source-of-truth local-mini campaign."""
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise ValueError("notebook harness campaign root must be absent or empty")
    commit = _git_commit(repository)
    environment = {
        "EDGEGUARD_PROJECT_ROOT": str(repository.resolve()),
        "EDGEGUARD_CAMPAIGN_ROOT": str(campaign_root.resolve()),
        "EDGEGUARD_PROJECT_COMMIT": commit,
        "EDGEGUARD_CAMPAIGN_ID": "eg-notebook-local-mini",
        "EDGEGUARD_CAMPAIGN_PROFILE": "local-mini",
        "EDGEGUARD_AUTO_CONTINUE": "1",
    }
    notebook_root = repository / "notebooks" / "colab"
    receipts = [execute_notebook(notebook_root / name, environment) for name in NOTEBOOKS]
    state = json.loads((campaign_root / "pipeline_state.json").read_text(encoding="utf-8"))
    completed = [
        stage for stage in topological_stages() if state["stages"][stage]["status"] == "completed"
    ]
    if completed != list(topological_stages()):
        raise RuntimeError("notebook handoff did not complete the local-mini campaign")
    reports = sorted(path.name for path in (campaign_root / "reports").glob("*.zip"))
    return {
        "schema_version": "1.0",
        "record_type": "campaign_notebook_harness_receipt",
        "status": "passed",
        "git_commit": commit,
        "notebooks": receipts,
        "completed_stages": completed,
        "reports": reports,
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
