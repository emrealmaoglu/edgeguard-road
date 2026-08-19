"""Tests for explicit runtime paths shared by local and Colab wrappers."""

from pathlib import Path

import pytest

from edgeguard.runtime import RuntimePathContract


def test_local_runtime_contract_is_distinct_path_free_evidence(tmp_path: Path) -> None:
    contract = RuntimePathContract.from_workspace(tmp_path / "readiness").validated(
        forbid_content=True
    )

    assert len(set(contract.as_dict().values())) == 6
    assert "/content" not in str(contract.as_dict())
    assert str(tmp_path) not in str(contract.receipt())


def test_runtime_contract_rejects_aliases_and_content_for_local_profile(tmp_path: Path) -> None:
    duplicated = RuntimePathContract(
        runtime_root=tmp_path / "same",
        checkout_root=tmp_path / "same",
        evidence_root=tmp_path / "evidence",
        log_root=tmp_path / "logs",
        cache_root=tmp_path / "cache",
        data_root=tmp_path / "data",
    )
    with pytest.raises(ValueError, match="distinct"):
        duplicated.validated()

    content = RuntimePathContract.from_workspace(Path("/content/local-readiness"))
    with pytest.raises(ValueError, match="cannot point to /content"):
        content.validated(forbid_content=True)
