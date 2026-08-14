from __future__ import annotations

from pathlib import Path

from scripts.dev.run_delivery_notebooks_local import execute_notebook_contract


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
