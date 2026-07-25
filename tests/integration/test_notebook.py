"""Static checks for the thin Colab execution wrapper."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/colab/00_environment_and_smoke.ipynb"


def test_notebook_is_valid_thin_wrapper_without_secret_markers() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", [])).lower()
    assert "git clone" in source
    assert "pip install -e ." in source
    assert "python -m edgeguard doctor" in source
    assert "python -m edgeguard smoke" in source
    assert "pytest -q" in source
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in source
