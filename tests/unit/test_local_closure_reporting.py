import json
import zipfile
from pathlib import Path

from edgeguard.reporting.local_closure import build_closure_packages, write_gap_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_gap_matrix_has_all_capabilities_and_complete_fields(tmp_path: Path) -> None:
    payload = write_gap_matrix(tmp_path)
    assert len(payload["records"]) == 27
    required = {
        "current_implementation",
        "maturity",
        "evidence",
        "remaining_work",
        "scientific_risk",
        "engineering_risk",
    }
    assert all(required <= set(row["before"]) for row in payload["records"])
    assert all(
        required | {"action_taken", "action_intentionally_deferred", "reason"} <= set(row["after"])
        for row in payload["records"]
    )
    assert "NON-SCIENTIFIC" in (tmp_path / "project_gap_matrix.md").read_text()


def test_closure_packages_are_small_sanitized_and_parseable(tmp_path: Path) -> None:
    audit_root = PROJECT_ROOT / "reports" / "local-final-audit"
    write_gap_matrix(audit_root)
    evidence = tmp_path / "evidence"
    for name in ("acquisition", "data_quality"):
        (evidence / name).mkdir(parents=True)
        (evidence / name / "report.json").write_text(
            json.dumps({"scientific_evidence": False}), encoding="utf-8"
        )
    summary = {
        "stages": [
            {
                "stage": "data_quality",
                "status": "completed",
                "maturity": "local_end_to_end_validated",
            }
        ]
    }
    result = build_closure_packages(
        tmp_path / "output",
        summary=summary,
        repository_root=PROJECT_ROOT,
        evidence_root=evidence,
    )
    for name in (
        "assistant_review",
        "thesis_figures",
        "data_lifecycle_audit",
        "colab_readiness",
    ):
        path = tmp_path / "output" / result[name]["filename"]
        assert path.stat().st_size < 5 * 1024**2
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
