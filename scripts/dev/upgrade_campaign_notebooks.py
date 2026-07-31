"""One-time deterministic injection of compact overview guidance into campaign notebooks."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = (
    "00_campaign_control.ipynb",
    "10_semantic_campaign.ipynb",
    "20_ood_calibration_risk.ipynb",
    "30_detection_temporal_fusion.ipynb",
    "40_export_and_reporting.ipynb",
)

MARKER = "## Compact campaign status"
SOURCE = [
    f"{MARKER}\n",
    "\n",
    "Each stage prints a compact overview, periodic progress rows, and a post-stage summary. "
    "Full logs remain in files. Failures produce an `edgeguard-failure-<campaign>-<stage>.zip` "
    "with environment, state, artifact identities, recovery state, traceback, and log tails.\n",
]


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "notebooks" / "colab"
    for name in NOTEBOOKS:
        path = root / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if any(MARKER in "".join(cell.get("source", [])) for cell in notebook["cells"]):
            continue
        notebook["cells"].insert(
            1,
            {"cell_type": "markdown", "metadata": {}, "source": SOURCE},
        )
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
