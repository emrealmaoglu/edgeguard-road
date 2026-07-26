# Project State

- **Project phase:** Local-first vertical slice — local PIDNet and Cityscapes val foundation ready for Commit B review
- **Active work package:** Real-checkpoint local validation and Cityscapes val foundation
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; local macOS arm64 inventory refreshed with PyTorch/MPS and disk evidence
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; checkpoint loader hardened for the reviewed official training layout and Colab wrapper made reproducible
- **Stage 2 real-forward status:** T4/CUDA feasibility and local CPU/MPS development forwards succeeded; clean committed rerun remains pending
- **Local foundation status:** Real checkpoint opt-in validation and Cityscapes val preparation/adapter complete; Commit B review pending
- **Later blocks:** Semantic metrics, four scores, minimal 1/5/10-image runner, and full-resolution Colab preparation not started
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; development feasibility — Demonstrated; clean-run and visual-sanity gate — Pending
- **Last verified commit:** `9a734358332e8221af940d323e103b3adda376ad` (Commit A)
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Manually installed PyTorch 2.13.0 only in `.venv`; added path-free strict checkpoint verification, explicit CPU/MPS/CUDA selection, PyTorch/device run metadata, and upstream bytecode protection. Verified real CPU and MPS forwards. Added project-specific val-only Cityscapes archive verification/extraction, deterministic manifest, explicit labelId-to-trainId LUT, RGB/label loading, geometry checks, and nearest-neighbor mask resizing.
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, license `OPEN QUESTION`
- **First T4 development evidence:** CUDA forward succeeded with native logits `[1,19,64,128]` and aligned logits `[1,19,512,1024]`; two consecutive forwards were byte-identical. The run was Git-dirty because a temporary loader patch was applied in Colab, so it is feasibility evidence rather than the final clean reproducibility artifact.
- **Local environment evidence:** Python 3.11.9, PyTorch 2.13.0, Apple arm64, 8 GiB RAM; MPS built and available, CUDA unavailable. PyTorch was not added to project dependencies or `LICENSES.md`.
- **Local forward evidence:** CPU and MPS both produced finite native `[1,19,64,128]` and aligned `[1,19,512,1024]` logits. Both were byte-identical within backend across two repeats; CPU/MPS semantic masks matched. Cross-device native-logit maximum absolute difference was approximately `1.85e-5` and is diagnostic, not a scientific gate.
- **Cityscapes val evidence:** Approved archives matched SHA-256 pins; val-only extraction produced 500 image/label pairs across 3 cities, no train/test extraction, and deterministic manifest SHA-256 `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`.
- **Scientific claim boundary:** MSP and predictive entropy are uncertainty/anomaly scores, not anomaly probabilities. No OOD performance, accuracy, or benchmark claim was made.
- **Current blocker:** The local checkpoint and Cityscapes val foundation diff needs human review and explicit Commit B approval.
- **Next single action:** Human reviews the Commit B diff and approves the commit; after that, rerun the clean CPU single-image verification and begin metrics, four scores, and the minimal runner.
- **Pending later human decisions:** Commit B approval; clean-run visual sanity; exact Colab commit; ONNX numerical gates; artifact promotion
- **Last local verification:** 2026-07-26T05:37:44Z — Ruff check passed; Ruff format check passed for 79 files; mypy passed for 23 source files; pytest passed 131 with 2 expected opt-in skips; real checkpoint and Cityscapes opt-in tests passed 2/2; `git diff --check` passed
- **Artifact status:** Checkpoint, upstream checkout, CPU/MPS tensors, environment report, and extracted Cityscapes val remain outside Git. No dataset, checkpoint, tensor, generated image, or other large/binary artifact was added to the repository.
- **Git action status:** Commit A created locally; Commit B changes are unstaged; nothing pushed, merged, tagged, released, or promoted

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
