# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `2495354d02e45cc4f8748e94cfcf1862ae48a295`
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

## Local gates

- Ruff and mypy pass.
- Full pytest passes with only environment-gated skips.
- Master notebook generation is byte-identical across two runs.
- All master notebook code cells execute in local claim-safe mode with
  `scientific_status=not_run`.
- The notebook SHA-256 after pinning is
  `09f06e9124868ee2dad6619cf1eb4bdd1f254b6bc886e769f1c79ed82b4d6f46`.
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
