from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev.build_colab_notebooks import PHASE_SECTIONS
from scripts.dev.run_delivery_notebooks_local import execute_notebook_contract

PHASE_ROOT = Path("notebooks/phases")


def test_delivery_notebooks_execute_all_code_cells_in_local_mode(tmp_path: Path) -> None:
    name = "EdgeGuard_Master_Colab.ipynb"
    result = execute_notebook_contract(
        Path("notebooks") / name,
        tmp_path / name / "drive",
        tmp_path / name / "content",
    )
    assert result["status"] == "passed"
    # Setup/bootstrap/inventory/helpers, one cell per campaign phase, and the summary.
    # Phases run as separate cells so a dropped Colab session resumes from the cell it
    # died in rather than replaying the whole campaign.
    assert result["code_cell_count"] == 15
    assert all(row["status"] == "passed" for row in result["cells"])


def _phase_notebook(index: int, target: str) -> Path:
    return PHASE_ROOT / f"EdgeGuard_{index:02d}_{target}.ipynb"


def test_every_phase_has_a_standalone_notebook_pinned_to_the_master_commit() -> None:
    """Colab gives each notebook its own VM, so a per-phase notebook cannot inherit a
    wedged runtime or a half-written directory from the phase before it, and it hands back
    exactly one log bundle. They must stay pinned to the same commit as the master
    notebook: a phase notebook running older code against a campaign state store written
    by newer code is precisely the kind of mismatch the recovery identity refuses.
    """
    master = json.loads(Path("notebooks/EdgeGuard_Master_Colab.ipynb").read_text())
    master_source = "".join("".join(cell["source"]) for cell in master["cells"])
    pin = next(
        line for line in master_source.splitlines() if line.startswith("EXPECTED_PROJECT_COMMIT")
    )

    assert [target for target, _ in PHASE_SECTIONS] == [
        "smoke",
        "pilot",
        "screening",
        "hpo",
        "final",
        "evaluate",
        "export",
        "report",
        "package",
    ]
    for index, (target, _) in enumerate(PHASE_SECTIONS, start=1):
        path = _phase_notebook(index, target)
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "".join("".join(cell["source"]) for cell in payload["cells"])
        assert pin in source, path
        assert f'eg_phase("{target}")' in source, path
        # Each notebook drives exactly one phase, so the operator can never lose track of
        # which output belongs to which stage.
        driven = {other for other, _ in PHASE_SECTIONS if f'eg_phase("{other}")' in source}
        assert driven == {target}, (path, driven)
        assert all(
            cell.get("outputs") == [] and cell.get("execution_count") is None
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        ), path


@pytest.mark.parametrize("index,target", [(3, "screening"), (9, "package")])
def test_phase_notebooks_execute_their_code_cells_in_local_mode(
    tmp_path: Path, index: int, target: str
) -> None:
    path = _phase_notebook(index, target)
    result = execute_notebook_contract(
        path, tmp_path / target / "drive", tmp_path / target / "content"
    )
    assert result["status"] == "passed"
    # The shared prelude plus this phase's single run cell.
    assert result["code_cell_count"] == 6
    assert all(row["status"] == "passed" for row in result["cells"])
