"""Every script must survive being started the way the docs say to start it.

The bug this exists for: `python scripts/jetson/evaluate_engine.py` puts *that* directory
on the import path rather than the repository root, so a module imported from a sibling
file is unresolvable. It surfaced on the device, on the last measurement in the project,
after 239 MB of evaluation data had already been copied across -- and it was present in
two other scripts documented the same way.

The property checked is import survival, not `--help` success. Some scripts legitimately
take no arguments (`run_video_dashboard.py` launches Streamlit from an environment
variable), so demanding a usage message would fail them for being correctly designed. What
must never happen is `ImportError` or `ModuleNotFoundError`: those mean the file cannot be
run at all, whatever it was invoked with.

Nothing else catches this. The test suite imports these modules through the package path,
where the problem does not exist -- so the check has to be a subprocess, from the
repository root, exactly as a reader would type it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    path.relative_to(ROOT)
    for path in ROOT.glob("scripts/**/*.py")
    if path.name != "__init__.py" and "dev/" not in str(path.relative_to(ROOT))
)

# Failures that mean the file could not be loaded, as opposed to a script that loaded
# fine and then objected to being given no configuration.
IMPORT_FAILURES = ("ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: str(path))
def test_the_documented_invocation_loads_the_file(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    broken = [name for name in IMPORT_FAILURES if name in result.stderr]
    assert not broken, (
        f"`python {script} --help` fails to load ({broken[0]}), so the invocation the "
        "documentation gives does not work:\n" + result.stderr[-1200:]
    )


def test_there_is_something_to_check() -> None:
    """A glob matching nothing would make every case above pass vacuously."""
    assert len(SCRIPTS) >= 40
