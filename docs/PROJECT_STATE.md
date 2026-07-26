# Project State

- **Project phase:** WP-03/WP-04/first WP-05 vertical slice — Stage 2 prepared, real Colab execution pending
- **Active work package:** Isolated PIDNet-S single-image spike
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; no new environment probe was run in this task
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; fixed upstream checkout and approved sample provenance verified
- **Stage 2 real-forward status:** Pending; checkpoint bytes are unavailable and Codex did not run a Colab session
- **Later stages:** Stage 3 Cityscapes integration, Stage 4 Fishyscapes development adapter, and Stage 5 ONNX pilot not started
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; real-forward and visual-sanity gate — Pending
- **Last verified commit:** `67ceeff0d3b97994800077684305274c1f7ed1cc`
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Fixed official PIDNet checkout at the approved commit; primary/fallback upstream sample identity, SHA-256, shape, and use-boundary verification; deterministic upstream-sample provenance; exact upstream/checkpoint guards; safe `weights_only=True` and strict state-dict key/shape loading; explicit native/aligned logits path; aligned-grid semantic mask, MSP, and predictive entropy; repeat diagnostics; ignored artifact writer; execution-only Colab notebook with human checkpoint upload; focused tests
- **Checkpoint access probe:** Official individual file reference returned HTTP 404 and gdown could not resolve it; official README replacement folder returned HTTP 200; no checkpoint file was created
- **Current blockers:** The official checkpoint bytes, byte size, and reviewed SHA-256 are not present; the replacement folder requires human-controlled retrieval/upload; no directly controlled Colab runtime is available to Codex
- **Next single action:** The human owner retrieves `PIDNet_S_Cityscapes_val.pt` from the official README replacement folder and runs `notebooks/colab/01_pidnet_single_image_spike.ipynb`, reviewing the recorded byte size and SHA-256 before model load
- **Pending later human decisions:** Visual sanity and spike acceptance; controlled vendoring versus fixed external checkout; Cityscapes/Fishyscapes access; ONNX numerical gates; artifact promotion
- **Last local verification:** 2026-07-26T02:23:55Z — Ruff check passed; Ruff format check passed; mypy passed for 22 source files; pytest passed 98/98; `git diff --check` passed
- **Artifact status:** Ignored fixed PIDNet source checkout exists under `artifacts/external`; no checkpoint, dataset download, redistributed sample, logits, PNG output, metric, or ONNX artifact was created
- **Git action status:** No staging, commit, push, merge, tag, release, or promotion performed

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
