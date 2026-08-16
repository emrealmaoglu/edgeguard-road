"""Run the reported-number check as a test, so a stale figure fails before a defence does.

This is the check itself, not a test of the checker: the assertion is that the thesis
documents currently contain no four-decimal number absent from the measurement records. It
fails when someone edits a number by hand, when a measurement is re-run and a document is
not updated, and when a value is derived without being declared.

The unit tests below it pin the checker's own behaviour, since a check that silently stops
checking is worse than none.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.verify_reported_numbers import (  # noqa: E402
    DERIVED,
    known_values,
    reported_numbers,
)

DOCUMENTS = (
    ROOT / "docs/THESIS_EVIDENCE.md",
    ROOT / "docs/THESIS_SECTIONS.md",
    ROOT / "docs/LITERATURE_PLAN.md",
)


@pytest.fixture(scope="module")
def corpus() -> set[float]:
    return known_values(ROOT / "reports/measurements") | set(DERIVED)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_every_reported_number_exists_in_a_record(document: Path, corpus: set[float]) -> None:
    if not document.is_file():
        pytest.skip(f"{document.name} is not present")

    stale = [
        f"{document.name}:{line} {text}"
        for line, text, value in reported_numbers(document)
        if not any(abs(value - candidate) <= 5e-4 for candidate in corpus)
    ]

    assert not stale, (
        "these numbers appear in no measurement record. Each is stale, mistyped, or "
        "derived; if derived, add it to DERIVED with its derivation: " + ", ".join(stale)
    )


def test_the_checker_finds_a_number_that_is_not_measured(tmp_path: Path) -> None:
    """Guard against the check quietly passing everything."""
    document = tmp_path / "claim.md"
    document.write_text("Model reached 0,7391 mIoU.\n", encoding="utf-8")

    found = reported_numbers(document)

    assert found and found[0][2] == pytest.approx(0.7391)


def test_an_arxiv_identifier_is_not_a_measurement(tmp_path: Path) -> None:
    """`arXiv:2607.04304` has the shape of a four-decimal number and is not one; treating
    it as a claim produced a false report before this was handled.
    """
    document = tmp_path / "cite.md"
    document.write_text("See `arXiv:2607.04304` for the related work.\n", encoding="utf-8")

    assert reported_numbers(document) == []


def test_a_percentage_of_a_measured_value_is_accepted(tmp_path: Path) -> None:
    """Documents state 0.5007 as 50.07%; that is the same measurement in another unit."""
    records = tmp_path / "records"
    records.mkdir()
    (records / "a.json").write_text(json.dumps({"fraction": 0.5007}), encoding="utf-8")

    assert 50.07 in {round(value, 4) for value in known_values(records)}


def test_every_declared_derivation_says_where_it_came_from() -> None:
    """A number allowed through without an explanation is an allowlist entry that will
    outlive the reason for it.
    """
    for value, reason in DERIVED.items():
        assert len(reason) > 40, f"{value} is allowed without a stated derivation"
