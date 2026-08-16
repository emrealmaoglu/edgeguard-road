"""Pin the one property a review flow must have: its stages subtract to their own total.

The figure this replaces carried hand-written counts that did not balance -- 912 records
minus 34 minus 661 leaves 217, and the box said 251 -- and claimed more included sources
than the bibliography held. Both are the kind of error a reader finds by doing arithmetic
the author did not, which is why the check belongs in code rather than in review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audit_literature_corpus import classify, main  # noqa: E402


def _corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "01_search.md").write_text(
        "Paper https://arxiv.org/abs/1811.10200 and https://openaccess.thecvf.com/x.html .\n"
        "Code https://github.com/open-mmlab/mmsegmentation and https://pypi.org/project/x/\n"
        "Thread https://stackoverflow.com/questions/1 plus https://example.invalid/page\n",
        encoding="utf-8",
    )
    (root / "02_search.md").write_text(
        "Duplicate https://arxiv.org/abs/1811.10200 again, and https://doi.org/10.1000/x\n",
        encoding="utf-8",
    )
    return root


def _bibliography(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "**[1]** FIRST, A. *Title.* 2020.\n\n**[2]** SECOND, B. *Other.* 2021.\n",
        encoding="utf-8",
    )
    return path


def _run(argv: list[str]) -> None:
    original = sys.argv
    sys.argv = ["audit_literature_corpus.py", *argv]
    try:
        main()
    finally:
        sys.argv = original


def test_the_stages_balance_and_the_record_says_so(tmp_path: Path) -> None:
    research = _corpus(tmp_path / "researchs")
    bibliography = _bibliography(tmp_path / "docs" / "BIBLIOGRAPHY.md")
    output = tmp_path / "literature_audit.json"

    _run(
        [
            "--research-root",
            str(research),
            "--bibliography",
            str(bibliography),
            "--output",
            str(output),
        ]
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    removed = sum(record["screening_removed"].values())
    assert record["identification"] - removed == record["eligibility"]


def test_a_repeated_url_is_one_record(tmp_path: Path) -> None:
    """Identification counts records, not mentions: the same paper cited in two search
    documents is one source, and counting it twice would inflate the top of the flow.
    """
    research = _corpus(tmp_path / "researchs")
    output = tmp_path / "literature_audit.json"

    _run(
        [
            "--research-root",
            str(research),
            "--bibliography",
            str(_bibliography(tmp_path / "docs" / "BIBLIOGRAPHY.md")),
            "--output",
            str(output),
        ]
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["records_total"] > record["identification"]


def test_the_included_count_comes_from_the_bibliography(tmp_path: Path) -> None:
    """So the figure cannot claim more sources than the reference list holds, which is the
    second error the previous figure made.
    """
    research = _corpus(tmp_path / "researchs")
    output = tmp_path / "literature_audit.json"

    _run(
        [
            "--research-root",
            str(research),
            "--bibliography",
            str(_bibliography(tmp_path / "docs" / "BIBLIOGRAPHY.md")),
            "--output",
            str(output),
        ]
    )

    assert json.loads(output.read_text(encoding="utf-8"))["included"] == 2


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/1811.10200", "academic_venue"),
        ("https://www.researchgate.net/publication/1", "academic_venue"),
        ("https://ecva.net/papers/x.pdf", "academic_venue"),
        ("https://github.com/open-mmlab/mmsegmentation", "tool_or_documentation"),
        ("https://docs.nvidia.com/deeplearning/tensorrt/", "tool_or_documentation"),
        ("https://stackoverflow.com/questions/1", "non_scholarly"),
        ("https://example.invalid/page", "unclassified"),
    ],
)
def test_classification_of_representative_hosts(url: str, expected: str) -> None:
    assert classify(url) == expected


def test_an_empty_corpus_is_an_error_not_an_empty_flow(tmp_path: Path) -> None:
    """A figure showing zeros would look like a finished review of nothing."""
    empty = tmp_path / "researchs"
    empty.mkdir()

    with pytest.raises(ValueError, match="no search documents"):
        _run(
            [
                "--research-root",
                str(empty),
                "--bibliography",
                str(_bibliography(tmp_path / "docs" / "BIBLIOGRAPHY.md")),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
