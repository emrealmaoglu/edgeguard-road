"""Claim-safe local execution of the one master notebook."""

from pathlib import Path

from scripts.dev.run_campaign_notebook_harness import NOTEBOOKS, run_harness

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_campaign_notebooks_complete_one_shared_local_mini_state(tmp_path: Path) -> None:
    result = run_harness(PROJECT_ROOT, tmp_path / "campaign")

    assert result["status"] == "passed"
    assert [notebook["notebook"] for notebook in result["notebooks"]] == list(NOTEBOOKS)
    assert result["scientific_evidence"] is False
    assert result["notebooks"][0]["contract"]["scientific_status"] == "not_run"
