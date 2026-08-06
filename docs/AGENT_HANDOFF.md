# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `1da25ef405fcf36180d0b223973ff768296a228f`
- **Campaign:** `semantic-cs-idd-v3`
- **Notebook:** `notebooks/EdgeGuard_Master_Colab.ipynb`
- **Classification:** locally verified engineering delivery; real Colab GPU/training and
  Jetson evidence remain external.

## What changed

- Replaced fourteen entry notebooks with one generated Run-all notebook.
- Added the `all` production orchestrator from preflight through final delivery packages.
- Reuses exact v2 Cityscapes/IDD audit candidates and verified Drive data without rescans.
- Requires all five canary, screening, and final models; HPO remains top-two.
- Added forced smoke interruption/resume evidence and general immutable checkpoint restore.
- Added owner-preauthorized exact-source/release policy while keeping official validation
  after model selection.
- Added five-model ONNX/golden-vector Jetson packaging, accepted Streamlit packaging, and
  measured thesis tables/figures/galleries.
- Added L4/High-RAM/disk gates and atomic Drive publication of final ZIPs.
- Replaced the hosted-Python project import with a standard-library-only runtime bootstrap.
- Runs restore and data preparation only after the verified Python 3.11 environment exists.
- Isolates v3 work state and preserves/skips incompatible commit-bound local/Drive state.
- Streams the real child error and packages stage, bootstrap and log-tail diagnostics.

## Local gates

- Ruff and format checks pass for the full repository.
- Mypy passes for all 116 configured source modules.
- Full pytest passes: 462 passed, 2 environment-gated skipped.
- Both hosted entrypoints load with site packages disabled (`python -S`).
- Master notebook generation is byte-identical across two runs.
- All master notebook code cells execute in local claim-safe mode with
  `scientific_status=not_run`.
- The notebook SHA-256 after pinning is
  `1f7e4a7d7a65f2d921e4ed34f395029c59a20917b1744e54f4d69b694accb0e0`.
- ZIP writer verifies CRC, member set and every member payload after creation.

## Next external action

Open the master notebook from the pushed branch in a fresh Colab L4 + High-RAM runtime and
use Run all. If the session ends, repeat Run all in a new compliant runtime. Do not change
the notebook or select stages manually.

Do not create `colab-v0.1.0-rc1` until two independent clean L4 sessions pass the exact
lock/five-model FP32/AMP canary and the real 50-step interruption/resume proof. After the
campaign completes, build TensorRT only on the target Jetson and attach actual 25W
telemetry. Do not merge, tag, open sealed datasets, upgrade JetPack, or change Jetson power
mode without a separate explicit decision.
