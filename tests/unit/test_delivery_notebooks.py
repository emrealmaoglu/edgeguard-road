from __future__ import annotations

from pathlib import Path

from scripts.dev.run_delivery_notebooks_local import execute_notebook_contract


def test_delivery_notebooks_execute_all_code_cells_in_local_mode(tmp_path: Path) -> None:
    for name, expected_cells in (
        ("EdgeGuard_Data_Preflight_Colab.ipynb", 5),
        ("EdgeGuard_Road_Colab.ipynb", 11),
    ):
        result = execute_notebook_contract(
            Path("notebooks") / name,
            tmp_path / name / "drive",
            tmp_path / name / "content",
        )
        assert result["status"] == "passed"
        assert result["code_cell_count"] == expected_cells
        assert all(row["status"] == "passed" for row in result["cells"])
