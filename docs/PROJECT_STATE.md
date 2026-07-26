# Project State

- **Project phase:** WP-03/WP-04/first WP-05 vertical slice — Stage 2 prepared, real Colab execution pending
- **Active work package:** Isolated PIDNet-S single-image spike
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; no new environment probe was run in this task
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; fixed upstream checkout and approved sample provenance verified
- **Stage 2 real-forward status:** Pending; human-verified checkpoint identity is pinned, but Codex did not open the file or run a Colab session
- **Later stages:** Stage 3 Cityscapes integration, Stage 4 Fishyscapes development adapter, and Stage 5 ONNX pilot not started
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; real-forward and visual-sanity gate — Pending
- **Last verified commit:** `67ceeff0d3b97994800077684305274c1f7ed1cc`
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Fixed official PIDNet checkout at the approved commit; primary/fallback upstream sample identity, SHA-256, shape, and use-boundary verification; deterministic upstream-sample provenance; exact upstream/checkpoint guards; safe `weights_only=True` and strict state-dict key/shape loading; explicit native/aligned logits path; aligned-grid semantic mask, MSP, and predictive entropy; repeat diagnostics; ignored artifact writer; execution-only Colab notebook with human checkpoint upload; focused tests
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, license `OPEN QUESTION`
- **Current blockers:** The pinned checkpoint still requires human upload into a real Colab session; no directly controlled Colab runtime is available to Codex
- **Next single action:** The human owner runs `notebooks/colab/01_pidnet_single_image_spike.ipynb`; the notebook automatically rejects filename/hash mismatches before model load
- **Pending later human decisions:** Visual sanity and spike acceptance; controlled vendoring versus fixed external checkout; Cityscapes/Fishyscapes access; ONNX numerical gates; artifact promotion
- **Last local verification:** 2026-07-26T02:42:58Z — Ruff check passed; Ruff format check passed; mypy passed for 22 source files; pytest passed 98/98; `git diff --check` passed
- **Artifact status:** Ignored fixed PIDNet source checkout exists under `artifacts/external`; no checkpoint was copied into the repository, and no dataset download, redistributed sample, logits, PNG output, metric, or ONNX artifact was created
- **Git action status:** No staging, commit, push, merge, tag, release, or promotion performed

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
