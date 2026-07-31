# Agent Handoff

- **Milestone:** updated phase-one road perception and pre-Colab implementation.
- **Branch:** `rescue/semantic-first`; exact-commit Colab delivery publication.
- **Pinned Colab implementation:** `3134d3f1e6d3bf23ede14f7a29b0adbeb51e0e89`.
- **Classification:** locally tested engineering implementation; no new scientific or
  Jetson measurement.
- **Pin enforcement:** strict in Colab; the no-clone local harness only reports a later
  checkout so branch-tip CI can validate the immutable notebook payload.
- **Observed Colab incident:** the first real inventory child process resolved its relative
  config default from `/content`, not the checkout. Active CLI defaults and every notebook
  subprocess working directory are repository-root anchored. Subprocess-tail logging exposed
  the exact cause; hash-read resilience and mandatory post-copy verification remain active.
- **Observed IDD incident:** the official Part I/II preparation produced no output for over
  12 hours. The old gzip seek pattern, duplicate hash reads, serial dual-mask rendering,
  missing liveness, and orphanable child process are corrected locally. Do not rerun the
  old `6745106` payload; publish and pin this implementation first.

## Delivered

- Safe archive preparation for Cityscapes, official/quarantined BDD100K, and official
  IDD20K Part I/II, including published identities, safe extraction, collision checks,
  pinned IDD polygon rendering, Part II JPG support, and immutable native labels.
- Kaggle BDD is fail-closed as `scientific_eligible=false`; active scientific HPO and
  final training use Cityscapes + IDD20K.
- Drive/Colab notebooks prepare and bundle one dataset at a time in ephemeral storage,
  enforce the 175 GiB + 25 GiB policy, and avoid Mac/Drive small-file extraction.
- IDD preparation now reads each gzip TAR forward once, hashes archives once, renders
  canonical masks through a reviewed LUT with bounded workers, and reports extraction,
  mask, bundle, staging and disk progress. Verified archive cache survives a failed retry.
- Interrupting a notebook command now terminates its complete subprocess group with a
  bounded TERM-to-KILL escalation, preventing hidden duplicate preparation processes.
- Colab runtime selection is receipt-driven for both hosted-current and isolated-Python
  paths; the previous hard-coded fallback interpreter/checkout mismatch is removed.
- Notebook-wide failure reporting captures bootstrap, Python and failed subprocess
  errors into redacted JSON plus a small downloadable ZIP in Drive. It records stage,
  commit, platform, disk and hashed bounded logs; data/model payloads are excluded.
- Selective campaign snapshots exclude staged data. Small review ZIPs expose reports,
  manifests and thesis figures for browser download while excluding models/data.
- Real frozen manifests generate class-distribution CSV and hashed 300-DPI PNG/PDF
  figures; notebook local mode executed every delivery code cell without external work.
- Connected Drive was inspected read-only. The implementation now supports the observed
  legacy/current Cityscapes paths and exact 6.99 GB bundle without moving or recompressing
  them; new BDD/IDD storage remains additive.
- The repeated Drive inspection confirmed IDD Part I/II and `bdd100k.zip` are now in
  `private_inputs/`; local archives were intentionally removed for space. Inventory and
  notebook preparation use this placement directly without a Drive migration.
- Semantic-derived road/corridor, regions, confidence, entropy, unreliable pixels, and
  deterministic operational-attention outputs in prediction and Streamlit.
- Road/component evaluation metrics and explicit non-instance/non-physical-risk bounds.
- Per-frame MSP, normalized entropy, maximum logit, energy, source-only shift reference,
  source/external AUROC/AP and alert-rate evidence.
- Dry-run-first TensorRT FP16 build and target-only sustained Jetson benchmark contracts;
  neither changes power mode or overwrites evidence.
- Fail-closed RTMDet-Tiny activation gate for measured phase-one, 25W p95, memory,
  remaining budget, and official BDD detection provenance.
- Charter, state, tasks, runbook, experiment and claim matrices now describe one current
  semantic thesis rather than the legacy detection/OOD/temporal campaign.

## Verification

- `ruff format --check .`: 288 files formatted.
- `ruff check .`: passed.
- `mypy src/edgeguard`: 108 source files passed.
- `pytest -q`: 419 passed, 10 skipped.
- Both delivery notebooks regenerate, contain no outputs, and all 16 code cells execute
  in the external-action-free local contract runner.
- All active command help surfaces and Python compile-all passed.
- `git diff --check` passed; no machine-local path or credential marker was found in
  active code, configuration, documentation, or notebooks.

## Evidence boundary and next action

No archive was extracted locally, no Drive write/Colab/GPU/official-validation/ACDC/sealed
run occurred, and no ONNX/TensorRT/Jetson artifact was created. The user-managed archive
store contains official Cityscapes, official IDD20K I/II, and a provisional Kaggle BDD
mirror. Run `notebooks/EdgeGuard_Data_Preflight_Colab.ipynb` against the existing
`private_inputs/` uploads. Then audit/freeze Cityscapes + IDD scientific manifests;
audit BDD separately without promoting it into HPO or the main thesis table.

Do not open ACDC or sealed external data before the final model/preprocessing/reliability
freeze. Do not start RTMDet-Tiny unless `scripts/check_detection_gate.py` returns a
fully evidenced pass. No dataset license acceptance, sealed submission, scientific-result
approval or privileged Jetson operation is implied by this handoff.
