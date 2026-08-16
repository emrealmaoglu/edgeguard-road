"""Run each device script the way the runbook says to run it.

Three of them import a shared TensorRT runner from a sibling module. Invoked as
`python scripts/jetson/<name>.py`, Python puts *that directory* on the import path rather
than the repository root, so the import fails before any argument is parsed -- which is
what happened on the device, on the last measurement left in the project, after the data
had already been copied across.

Nothing caught it because the tests import these modules through the package path, where
the problem does not exist. So the check has to be the documented invocation itself: a
subprocess, from the repository root, with `--help`. That reaches the imports and stops
before doing any work, and it needs no TensorRT, so it runs anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    path.relative_to(ROOT)
    for path in (ROOT / "scripts" / "jetson").glob("*.py")
    if path.name != "__init__.py"
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_the_documented_invocation_reaches_argument_parsing(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"`python {script} --help` failed, so the runbook's own invocation does not work:\n"
        + result.stderr[-1500:]
    )
    assert "usage:" in result.stdout


def test_there_is_something_to_check() -> None:
    """A glob that matches nothing would make every test above vacuously pass."""
    assert len(SCRIPTS) >= 3
