"""Static checks for the one-button Colab master notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.dev.build_colab_notebooks import PHASE_SECTIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/EdgeGuard_Master_Colab.ipynb"


def _source(payload: dict[str, object]) -> str:
    cells = payload["cells"]
    assert isinstance(cells, list)
    return "\n".join("".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict))


def test_every_checked_in_notebook_is_generated_output_free_and_commit_pinned() -> None:
    """The repository carries the master notebook plus exactly one notebook per campaign
    phase, and nothing else. The point of the rule is not the count: a hand-edited or
    stale notebook is a second, unversioned copy of the orchestration that can drift from
    the Python it is supposed to pin, so every file here must be one the builder emits,
    output-free, and pinned to a full commit SHA.
    """
    notebooks = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.glob("notebooks/**/*.ipynb")
    )
    expected = ["notebooks/EdgeGuard_Master_Colab.ipynb"] + [
        f"notebooks/phases/EdgeGuard_{index:02d}_{target}.ipynb"
        for index, (target, _) in enumerate(PHASE_SECTIONS, start=1)
    ]
    assert notebooks == sorted(expected)
    for relative in notebooks:
        payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["nbformat_minor"] >= 5
        assert re.search(r'EXPECTED_PROJECT_COMMIT = "[0-9a-f]{40}"', _source(payload)), relative
        for index, cell in enumerate(payload["cells"]):
            assert not cell.get("outputs"), relative
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{relative}-cell-{index}", "exec")


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
    assert "GERÇEK DURMA AŞAMASI" in source
    assert "edgeguard-master-child-failure.json" in source
    assert "edgeguard-master-child.log" in source
    # The shared `environment` dict feeding the main run_colab_master.py subprocess must
    # stay PYTHONPATH-free: that host process is stdlib-only by design, and PYTHONPATH for
    # the actually-isolated locked training runtime is set exclusively inside
    # run_colab_master.py's own _runtime_environment(). A scoped, differently-named dict
    # (e.g. the private_inputs inventory step's own `inventory_environment`) setting
    # PYTHONPATH for its own unrelated, non-training subprocess call is fine and expected.
    assert re.search(r'(?<!\w)environment\["PYTHONPATH"\]', source) is None
    assert 'inventory_environment["PYTHONPATH"]' in source
    assert "scripts/inventory_private_inputs.py" in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered
