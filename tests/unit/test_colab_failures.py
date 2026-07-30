from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest

from edgeguard.rescue.colab_failures import ColabFailureReporter, redact_failure_text


def test_failure_report_is_redacted_hashed_and_downloadable(tmp_path: Path) -> None:
    diagnostics = tmp_path / "logs"
    diagnostics.mkdir()
    (diagnostics / "install.log").write_text(
        "Authorization: Bearer secret-value\ntoken=another-secret\nreal error\n",
        encoding="utf-8",
    )
    reporter = ColabFailureReporter(
        tmp_path / "failures",
        notebook="EdgeGuard_Road_Colab.ipynb",
        project_commit="a" * 40,
        context={"api_key": "api_key=never-store-this", "stage_kind": "fixture"},
    )
    reporter.set_stage("runtime install")
    reporter.add_diagnostic_root("runtime-logs", diagnostics)
    try:
        raise RuntimeError("token=message-secret installation failed")
    except RuntimeError as error:
        error_type, _value, trace = sys.exc_info()
        assert error_type is RuntimeError
        receipt = reporter.capture(error_type, error, trace)

    report = json.loads(Path(receipt["report"]).read_text(encoding="utf-8"))
    assert report["stage"] == "runtime-install"
    assert report["failure_sha256"]
    assert "message-secret" not in json.dumps(report)
    with ZipFile(receipt["package"]) as archive:
        assert "failure.json" in archive.namelist()
        log = archive.read("diagnostics/runtime-logs/install.log").decode()
    assert "secret-value" not in log and "another-secret" not in log
    assert "real error" in log
    assert (tmp_path / "failures/LATEST.txt").read_text().strip() == receipt["failure_id"]
    assert reporter.latest_package() == Path(receipt["package"])


def test_failure_report_skips_large_or_unsupported_diagnostics(tmp_path: Path) -> None:
    diagnostics = tmp_path / "logs"
    diagnostics.mkdir()
    (diagnostics / "large.log").write_bytes(b"x" * 64)
    (diagnostics / "model.pth").write_bytes(b"weights")
    reporter = ColabFailureReporter(
        tmp_path / "failures",
        notebook="preflight.ipynb",
        project_commit="b" * 40,
        max_diagnostic_bytes=32,
    )
    reporter.add_diagnostic_root("logs", diagnostics)
    try:
        raise ValueError("fixture")
    except ValueError as error:
        error_type, _value, trace = sys.exc_info()
        assert error_type is ValueError
        receipt = reporter.capture(error_type, error, trace)
    with ZipFile(receipt["package"]) as archive:
        assert archive.namelist() == ["failure.json"]


def test_redaction_preserves_actionable_text() -> None:
    rendered = redact_failure_text(
        "pip failed; password=hunter2; github_pat_ABCDEF1234567890; retry wheel"
    )
    assert "hunter2" not in rendered
    assert "github_pat_" not in rendered
    assert "pip failed" in rendered and "retry wheel" in rendered


def test_latest_failure_pointer_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "failures"
    root.mkdir()
    (root / "LATEST.txt").write_text("../escape\n", encoding="utf-8")
    reporter = ColabFailureReporter(root, notebook="n.ipynb", project_commit="d" * 40)
    with pytest.raises(ValueError, match="unsafe"):
        reporter.latest_package()


def test_ipython_hook_records_unhandled_notebook_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeShell:
        handler: object | None = None
        shown = False

        def set_custom_exc(
            self, exception_types: tuple[type[Exception], ...], handler: object
        ) -> None:
            assert exception_types == (Exception,)
            self.handler = handler

        def showtraceback(self, _exception: object, *, tb_offset: int | None = None) -> None:
            assert tb_offset == 0
            self.shown = True

    shell = FakeShell()
    fake_ipython = ModuleType("IPython")
    fake_ipython.get_ipython = lambda: shell  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    reporter = ColabFailureReporter(
        tmp_path / "failures",
        notebook="notebook.ipynb",
        project_commit="c" * 40,
    )
    assert reporter.install_ipython_hook() is True
    assert callable(shell.handler)
    try:
        raise RuntimeError("hook fixture")
    except RuntimeError as error:
        error_type, _value, trace = sys.exc_info()
        assert error_type is RuntimeError and trace is not None
        shell.handler(shell, error_type, error, trace, 0)  # type: ignore[operator]
    assert shell.shown is True
    assert (tmp_path / "failures/LATEST.txt").is_file()
