# Project State

- **Project phase:** Local-first vertical slice — Stage 2 closure ready for Commit A review
- **Active work package:** Isolated PIDNet-S single-image spike
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; no new environment probe was run in this task
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; checkpoint loader hardened for the reviewed official training layout and Colab wrapper made reproducible
- **Stage 2 real-forward status:** Development run succeeded on T4/CUDA; final clean reproducibility artifact remains pending
- **Later blocks:** Local checkpoint/CPU forward, Cityscapes val foundation, metrics/scores, and the minimal 1/5/10-image runner not started
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; development feasibility — Demonstrated; clean-run and visual-sanity gate — Pending
- **Last verified commit:** `4ffba90f1f27723c525a74d61ec3eff7f6951468`
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Added a narrow loader policy for the reviewed 481-key official training checkpoint: exactly 453 inference keys retained, exactly 13 `seghead_p.*` and 13 `seghead_d.*` keys excluded, and only the two reviewed `[19]` loss roots ignored; strict loading and exact shape validation remain mandatory. Hardened the Colab branch/commit, repository origin, kernel import path, bytecode, interpreter, and failure-output behavior. Replaced raw CLI argument capture with path-free command metadata.
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, license `OPEN QUESTION`
- **First T4 development evidence:** CUDA forward succeeded with native logits `[1,19,64,128]` and aligned logits `[1,19,512,1024]`; two consecutive forwards were byte-identical. The run was Git-dirty because a temporary loader patch was applied in Colab, so it is feasibility evidence rather than the final clean reproducibility artifact.
- **Scientific claim boundary:** MSP and predictive entropy are uncertainty/anomaly scores, not anomaly probabilities. No OOD performance, accuracy, or benchmark claim was made.
- **Current blocker:** The bounded Stage 2 closure diff needs human review and explicit Commit A approval.
- **Next single action:** Human reviews the Stage 2 closure diff and approves Commit A; only then begin the manual local PyTorch inventory, real checkpoint verification, and CPU forward block.
- **Pending later human decisions:** Visual sanity and spike acceptance; controlled vendoring versus fixed external checkout; Cityscapes/Fishyscapes access; ONNX numerical gates; artifact promotion
- **Last local verification:** 2026-07-26T05:20:08Z — Ruff check passed; Ruff format check passed for 72 files; mypy passed for 22 source files; pytest passed 116/116; notebook JSON/structure, path/secret scan, artifact/binary scan, and `git diff --check` passed
- **Artifact status:** No checkpoint, dataset, logits, generated image, or other large/binary artifact was added to Git. The T4 development outputs were external/ephemeral evidence and are not repository content.
- **Git action status:** No staging, commit, push, merge, tag, release, or promotion performed

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
