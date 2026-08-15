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


def test_presentation_notebook_is_pinned_and_executes_its_cells_in_local_mode(
    tmp_path: Path,
) -> None:
    """The presentation notebook renders artefacts from finished screening checkpoints.

    It must stay pinned to the same commit as the master notebook for the same reason the
    phase notebooks do -- it reads a campaign state store written by that code -- and it
    must drive `screening` and nothing later, since the whole point is that the rest of
    the chain is not a prerequisite for any presentation output.
    """
    path = Path("notebooks/EdgeGuard_10_Sunum_Ciktilari.ipynb")
    master = json.loads(Path("notebooks/EdgeGuard_Master_Colab.ipynb").read_text())
    master_source = "".join("".join(cell["source"]) for cell in master["cells"])
    pin = next(
        line for line in master_source.splitlines() if line.startswith("EXPECTED_PROJECT_COMMIT")
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in payload["cells"])
    assert pin in source
    driven = {target for target, _ in PHASE_SECTIONS if f'eg_phase("{target}")' in source}
    assert driven == {"screening"}
    assert "build_presentation_outputs.py" in source
    assert all(
        cell.get("outputs") == [] and cell.get("execution_count") is None
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )

    result = execute_notebook_contract(path, tmp_path / "drive", tmp_path / "content")
    assert result["status"] == "passed"
    assert all(row["status"] == "passed" for row in result["cells"])


def test_presentation_bundle_keeps_images_unlike_the_text_only_phase_bundle() -> None:
    """`eg_bundle` filters to text suffixes so weights never bloat a log archive.

    The presentation outputs are almost entirely PNG/PDF figures, so reusing that bundler
    would hand back an archive with every figure stripped out. The presentation notebook
    therefore carries its own archiver, which must not inherit the suffix filter and must
    keep the pre-1980 mtime clamp that the log bundler needed.
    """
    payload = json.loads(Path("notebooks/EdgeGuard_10_Sunum_Ciktilari.ipynb").read_text())
    source = "".join(
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
    )
    header = "def eg_sunum_bundle():"
    start = source.index(header)
    # Search past the definition line itself, which also contains the call spelling.
    end = source.index("\neg_sunum_bundle()", start + len(header))
    bundle_source = source[start:end]
    assert "shutil.make_archive" in bundle_source
    assert "keep_suffixes" not in bundle_source
    assert "os.utime(" in bundle_source
    assert "1980" in bundle_source


def test_eg_bundle_source_clamps_pre_1980_mtimes_before_archiving() -> None:
    """`eg_bundle` never raises (`try`/`except BaseException`), which is exactly why this
    defect was invisible until now: on 2026-08-14 a real failed HPO phase printed "'hpo-FAILED'
    log paketi oluşturulamadı: ValueError('ZIP does not support timestamps before 1980')"
    and produced no archive at all -- silently, at exactly the moment a failure diagnostic
    was most needed. `shutil.copy2` preserves each source file's mtime, and a file
    restored from a Drive tar/zip with no timestamp metadata lands at the Unix epoch
    (1970), which the DOS-era ZIP date field cannot represent. The generated notebook's
    `eg_bundle` cell source must therefore clamp forward before calling
    `shutil.make_archive`.
    """
    master = json.loads(Path("notebooks/EdgeGuard_Master_Colab.ipynb").read_text())
    source = "".join(
        "".join(cell["source"]) for cell in master["cells"] if cell["cell_type"] == "code"
    )
    assert "def eg_bundle(label):" in source
    start = source.index("def eg_bundle(label):")
    end = source.index("files.download(archive)", start)
    bundle_source = source[start:end]
    assert "os.utime(" in bundle_source
    assert "1980" in bundle_source


def test_pre_1980_mtimes_break_zip_archiving_and_the_clamp_fixes_it(tmp_path: Path) -> None:
    """Reproduces the causal mechanism with the same stdlib calls `eg_bundle` uses --
    independent proof that the defect is real and that clamping mtimes forward to the
    DOS-zip epoch (1980-01-01) before archiving is a correct, sufficient fix. This does
    not execute the notebook cell itself (its archiving path only runs outside
    LOCAL_TEST_MODE); `test_eg_bundle_source_clamps_pre_1980_mtimes_before_archiving`
    checks the generated cell source carries the fix.
    """
    import os
    import shutil
    from datetime import datetime

    source_file = tmp_path / "source" / "run_identity.json"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("{}", encoding="utf-8")
    os.utime(source_file, (0, 0))  # the Unix epoch: 1970-01-01, before DOS zip's 1980 floor

    staging = tmp_path / "staging"
    staging.mkdir()
    copied = staging / source_file.name
    shutil.copy2(source_file, copied)
    assert copied.stat().st_mtime == 0

    with pytest.raises(ValueError, match="1980"):
        shutil.make_archive(str(tmp_path / "unfixed"), "zip", staging)

    dos_epoch = datetime(1980, 1, 1).timestamp()
    for path in staging.rglob("*"):
        if path.is_file() and path.stat().st_mtime < dos_epoch:
            os.utime(path, (dos_epoch, dos_epoch))
    archive = shutil.make_archive(str(tmp_path / "fixed"), "zip", staging)
    assert Path(archive).is_file()
