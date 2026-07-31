"""Thin Streamlit entrypoint for a prerecorded EdgeGuard report."""

from __future__ import annotations

import os
from pathlib import Path

from edgeguard.demo.video_dashboard import run_streamlit


def main() -> None:
    """Render the runtime-supplied report without embedding private paths."""
    value = os.environ.get("EDGEGUARD_VIDEO_REPORT")
    if not value:
        raise ValueError("EDGEGUARD_VIDEO_REPORT must identify a local report.json")
    run_streamlit(Path(value))


if __name__ == "__main__":
    main()
