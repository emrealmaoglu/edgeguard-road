# Project State

- **Project phase:** Local-first vertical slice — Commit C locally approved; push and Colab campaign remain gated
- **Active work package:** Local 1/5/10-image Cityscapes development campaign and full-resolution Colab preparation
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; local macOS arm64 inventory refreshed with PyTorch/MPS and disk evidence
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; checkpoint loader hardened for the reviewed official training layout and Colab wrapper made reproducible
- **Stage 2 real-forward status:** T4/CUDA feasibility and local CPU/MPS development forwards succeeded; the CPU forward was repeated from clean Commit B
- **Local foundation status:** Real checkpoint validation, Cityscapes val preparation/adapter, semantic metrics, four aligned-logit scores, and the mandatory 1/5/10-image local campaign are complete
- **Colab preparation status:** Thin full-resolution runner notebook and deterministic artifact packager are locally validated; no full 500-image Colab result exists yet
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; local foundation, 1/5/10 development campaign, and Commit C review — Passed; push and full Colab campaign — Pending
- **Last verified change set:** Commit C, `feat(eval): add Cityscapes metrics and uncertainty runner`
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Added streaming 19-class semantic metrics; stable MSP, predictive entropy, MaxLogit, and Energy scores on `aligned_logits`; a minimal path-free PIDNet/Cityscapes evaluator; local and Colab configs; deterministic subset selection; compact visual/artifact output; an artifact packager; and a thin full-resolution Colab execution notebook. Completed the mandatory 1/5/10-image local campaign.
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, license `OPEN QUESTION`
- **First T4 development evidence:** CUDA forward succeeded with native logits `[1,19,64,128]` and aligned logits `[1,19,512,1024]`; two consecutive forwards were byte-identical. The run was Git-dirty because a temporary loader patch was applied in Colab, so it is feasibility evidence rather than the final clean reproducibility artifact.
- **Local environment evidence:** Python 3.11.9, PyTorch 2.13.0, Apple arm64, 8 GiB RAM; MPS built and available, CUDA unavailable. PyTorch was not added to project dependencies or `LICENSES.md`.
- **Local forward evidence:** CPU and MPS both produced finite native `[1,19,64,128]` and aligned `[1,19,512,1024]` logits. Both were byte-identical within backend across two repeats; CPU/MPS semantic masks matched. Cross-device native-logit maximum absolute difference was approximately `1.85e-5` and is diagnostic, not a scientific gate.
- **Cityscapes val evidence:** Approved archives matched SHA-256 pins; val-only extraction produced 500 image/label pairs across 3 cities, no train/test extraction, and deterministic manifest SHA-256 `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`.
- **Local campaign evidence:** The deterministic 1-image CPU, 5-image MPS, and 10-image MPS runs completed with zero failures. Evaluation times were approximately 0.777 s, 3.945 s, and 5.710 s respectively. Development-grid mIoU values were approximately 0.6632, 0.6121, and 0.6246; pixel accuracies were approximately 0.9637, 0.9514, and 0.9565. These small resized-grid development measurements are not a paper-protocol reproduction or a final scientific result.
- **Score evidence:** All four aligned-logit score summaries were finite. They are ID-only development summaries; no threshold, OOD performance, or anomaly-probability claim was made.
- **Artifact package evidence:** The final 10-image external evidence package contains 15 files, is 1,132,620 bytes, and has SHA-256 `40c0033c13de1a6641f3cf73fca317d695a2a79cc51d7832b3789362ede1d408`.
- **Current blocker:** Push remains unauthorized; the full Colab campaign requires the exact pushed Commit C SHA and separate human approval.
- **Next single action:** Human reviews the reported Commit C SHA and authorizes or declines push.
- **Pending later human decisions:** Commit C push; exact Colab commit; full-run interpretation; ONNX numerical gates; artifact promotion
- **Last local verification:** 2026-07-26T06:00:09Z — Ruff check passed; Ruff format check passed for 87 files; mypy passed for 25 source files; pytest passed 155 with 2 expected opt-in skips; notebook JSON and six code cells compiled; path/secret and large-file scans were clean; `git diff --check` passed
- **Artifact status:** Checkpoint, upstream checkout, CPU/MPS tensors, extracted Cityscapes val, local evaluator outputs, and the ZIP evidence package remain outside Git. No dataset, checkpoint, tensor, generated visual, archive, or other large/binary artifact was added to the repository.
- **Git action status:** Commits A, B, and C exist locally; the branch is three commits ahead of origin; nothing pushed, merged, tagged, released, or promoted

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
