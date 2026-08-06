"""Static checks for the one-button Colab master notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/EdgeGuard_Master_Colab.ipynb"


def _source(payload: dict[str, object]) -> str:
    cells = payload["cells"]
    assert isinstance(cells, list)
    return "\n".join("".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict))


def test_master_notebook_is_the_only_active_notebook_and_is_output_free() -> None:
    notebooks = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.glob("notebooks/**/*.ipynb")
    )
    assert notebooks == ["notebooks/EdgeGuard_Master_Colab.ipynb"]
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    for index, cell in enumerate(payload["cells"]):
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"master-cell-{index}", "exec")


def test_master_notebook_is_thin_immutable_and_run_all_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _source(payload)
    assert re.search(r'EXPECTED_PROJECT_COMMIT = "[0-9a-f]{40}"', source)
    assert 'CAMPAIGN_ID = "semantic-cs-idd-v3"' in source
    assert 'BRANCH = "stabilize/colab-v2"' in source
    assert "scripts/run_colab_master.py" in source
    assert '"--execution-mode", "production"' in source
    assert '"--target", "all"' not in source  # orchestration belongs to versioned Python
    assert "scripts/audit_dataset.py" not in source
    assert "pip install -e" not in source
    assert "AUTO_DOWNLOAD_JETSON_RELEASE = True" in source
    assert "EdgeGuard_Jetson_Release.zip" in source
    assert "persist_failure" in source
    assert "safe_restart" in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered
