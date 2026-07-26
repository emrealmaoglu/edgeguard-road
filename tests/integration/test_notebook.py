"""Static checks for the thin Colab execution wrapper."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/colab/00_environment_and_smoke.ipynb"
PIDNET_NOTEBOOK = REPO_ROOT / "notebooks/colab/01_pidnet_single_image_spike.ipynb"


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


def test_pidnet_notebook_is_valid_execution_only_wrapper() -> None:
    payload = json.loads(PIDNET_NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert all(not cell.get("outputs") for cell in payload["cells"])
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))
    assert "4c158cf24ce432f0a8cb43364fae38d93cee0dc3" in source
    assert "PIDNet_S_Cityscapes_val.pt" in source
    assert "REPLACE_WITH_DOWNLOAD_DATE_YYYY_MM_DD" in source
    assert "REPLACE_WITH_CHECKOUT_ACCESS_DATE_YYYY_MM_DD" in source
    assert "frankfurt_000000_002196_leftImg8bit.png" not in source
    assert "files.upload()" in source
    assert "spike_config.checkpoint.filename" in source
    assert "spike_config.checkpoint.sha256" in source
    assert "actual_checkpoint_sha256 != spike_config.checkpoint.sha256" in source
    assert "--sample-access-date" in source
    assert "--expected-checkpoint-sha256" not in source
    assert "--image-root" not in source
    assert "scripts/run_pidnet_spike.py" in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered
