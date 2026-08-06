# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `3f3ef8f7d065f739650d75f7921fd8c7e748fe81`
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
- Resolves both `bin/uv` and Colab system-pip `local/bin/uv` private-prefix layouts and
  carries the verified executable path into the runtime installer.
- Prioritizes the pinned wheel's CUDA libraries over Colab's mutable toolkit paths while
  preserving host driver discovery.
- Reuses the hash-verified bootstrap receipt for canary execution instead of performing a
  second destructive package sync.
- Reports the exact failed module or CUDA initialization stderr and retains MMEngine's
  required `pkg_resources` path through the Setuptools 80.9.0 lock.
- Preserves failed canary evidence under a timestamped quarantine root before retry, so
  repeated Run-all attempts start with clean runtime evidence without deleting history.
- Firewalls Colab's hosted Matplotlib/Python/virtualenv/pip/uv state before bootstrap and
  every locked-runtime child; forces `Agg` and isolated cache roots.
- Runs a real headless PNG probe before the five-model canary and records the non-secret
  environment-contract identity in runtime evidence.
- Resumes correctly when the one permitted OOM batch reduction is followed by the forced
  smoke interruption; the resumed command and checkpoint identity are both verified.
- Marks every acceptance-mode phase `not_run`, preventing fixture execution from becoming
  measured or accepted scientific evidence.

## Local gates

- Ruff and format checks pass for the full repository.
- Mypy passes for all 116 configured source modules.
- Full pytest passes: 477 passed, 2 environment-gated skipped.
- Mypy passes for all 120 configured source modules and changed runtime entrypoints;
  full-repository Ruff lint/format passes.
- Both hosted entrypoints load with site packages disabled (`python -S`).
- Master notebook generation is byte-identical across two runs.
- All master notebook code cells execute in local claim-safe mode with
  `scientific_status=not_run`.
- The notebook SHA-256 after pinning is
  `2d06b7afdfd75561ddcc6409d476351a6469113b813900f5f70dad7b6e1a1e9a`.
- ZIP writer verifies CRC, member set and every member payload after creation.
- Remote Linux run `31128538612` passed the exact 92-package Colab lock imports,
  `pkg_resources` and MMEngine Runner import, five-model CPU probe, real-codepath closure,
  ONNX classification and pre-Colab deployment-evidence build.

## Next external action

Open the master notebook from the pushed branch in a fresh Colab L4 + High-RAM runtime and
use Run all. If the session ends, repeat Run all in a new compliant runtime. Do not change
the notebook or select stages manually.

Do not create `colab-v0.1.0-rc1` until two independent clean L4 sessions pass the exact
lock/five-model FP32/AMP canary and the real 50-step interruption/resume proof. After the
campaign completes, build TensorRT only on the target Jetson and attach actual 25W
telemetry. Do not merge, tag, open sealed datasets, upgrade JetPack, or change Jetson power
mode without a separate explicit decision.
