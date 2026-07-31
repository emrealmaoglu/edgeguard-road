"""Integration coverage for the path-free synthetic OOD/calibration evidence."""

import json
from pathlib import Path
from zipfile import ZipFile

from scripts.dev.generate_ood_calibration_evidence import generate_evidence


def test_ood_calibration_evidence_is_small_path_free_and_non_scientific(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence"

    result = generate_evidence(output)
    repeated = generate_evidence(tmp_path / "evidence-repeat")

    assert result["status"] == "passed"
    assert result["scientific_evidence"] is False
    assert repeated["evidence_package_sha256"] == result["evidence_package_sha256"]
    package = output / "ood-calibration-evidence.zip"
    assert package.stat().st_size < 1_000_000
    with ZipFile(package) as archive:
        assert set(archive.namelist()) == {
            "calibration_before_after.json",
            "scoring_contract.json",
            "synthetic_score_statistics.json",
            "test_identity.json",
        }
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(output.glob("*.json"))
    )
    assert str(tmp_path) not in serialized
    assert '"scientific_evidence":true' not in serialized
    calibration = json.loads((output / "calibration_before_after.json").read_text())
    assert calibration["fit"]["final_nll"] <= calibration["fit"]["initial_nll"] + 1.0e-12
    assert calibration["input_logits_preserved"] is True
