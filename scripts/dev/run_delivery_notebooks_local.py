#!/usr/bin/env python3
"""Execute all delivery notebook code cells in safe local contract-test mode."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json

ROOT = Path(__file__).parents[2]
DEFAULT_NOTEBOOKS = (
    ROOT / "notebooks/EdgeGuard_Data_Preflight_Colab.ipynb",
    ROOT / "notebooks/EdgeGuard_Road_Colab.ipynb",
)


def execute_notebook_contract(path: Path, drive_root: Path, content_root: Path) -> dict[str, Any]:
    """Execute code cells without Drive, network, package installation, data, or GPU work."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__edgeguard_notebook_contract__"}
    rows: list[dict[str, Any]] = []
    previous = {
        key: os.environ.get(key)
        for key in (
            "EDGEGUARD_NOTEBOOK_LOCAL_TEST",
            "EDGEGUARD_PROJECT_ROOT",
            "EDGEGUARD_TEST_DRIVE_ROOT",
            "EDGEGUARD_TEST_CONTENT_ROOT",
        )
    }
    os.environ.update(
        {
            "EDGEGUARD_NOTEBOOK_LOCAL_TEST": "1",
            "EDGEGUARD_PROJECT_ROOT": str(ROOT),
            "EDGEGUARD_TEST_DRIVE_ROOT": str(drive_root),
            "EDGEGUARD_TEST_CONTENT_ROOT": str(content_root),
        }
    )
    try:
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            started = time.perf_counter()
            compile(source, f"{path.name}:cell-{index}", "exec")
            exec(source, namespace)
            rows.append(
                {
                    "cell_index": index,
                    "status": "passed",
                    "duration_seconds": round(time.perf_counter() - started, 6),
                }
            )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {
        "notebook": path.name,
        "code_cell_count": len(rows),
        "status": "passed",
        "cells": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebook", type=Path, action="append")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="edgeguard-notebook-contract-") as temporary:
        root = Path(temporary)
        reports = [
            execute_notebook_contract(path, root / "drive", root / "content")
            for path in (tuple(args.notebook) if args.notebook else DEFAULT_NOTEBOOKS)
        ]
    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_delivery_notebook_local_contract",
        "status": "passed",
        "notebooks": reports,
    }
    rendered = canonical_json(result) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
