# Project State

- **Project phase:** Cityscapes vertical slice complete; public Fishyscapes development foundation under human review
- **Active work package:** Verified full-resolution Cityscapes evidence, selection provenance correction, and Fishyscapes AP/FPR95 foundation
- **WP-00 status:** Complete
- **WP-01 status:** Complete
- **WP-02 status:** Environment inventory tooling complete; local macOS arm64 inventory refreshed with PyTorch/MPS and disk evidence
- **Stage 1 status:** Complete and human-approved on 2026-07-26
- **Stage 2 local preparation status:** Complete; checkpoint loader hardened for the reviewed official training layout and Colab wrapper made reproducible
- **Stage 2 real-forward status:** T4/CUDA feasibility and local CPU/MPS development forwards succeeded; the CPU forward was repeated from clean Commit B
- **Local foundation status:** Real checkpoint validation, Cityscapes val preparation/adapter, semantic metrics, four aligned-logit scores, and the mandatory 1/5/10-image local campaign are complete
- **Full Cityscapes campaign status:** Complete and independently verified from the external evidence ZIP; 500/500 samples succeeded from clean pushed Commit C
- **Fishyscapes foundation status:** Public Lost & Found validation adapter contract and NumPy AP/FPR95 metrics are locally complete; no Fishyscapes data was accessed
- **Gate status:** PIDNet commit — Approved; dataset roles — Approved; restricted checkpoint use — Approved with checkpoint license remaining `OPEN QUESTION`; local foundation, Commit C push, and full Cityscapes campaign evidence — Passed; Fishyscapes data acquisition and real development evaluation — Pending human action
- **Python compatibility status:** Local Python 3.11.9 validation passed. An isolated Python 3.10.11 environment using the CI dependency combination NumPy 2.2.6 and mypy 1.20.2 also passed after making `predictive_entropy` produce an explicit float32 ndarray. Remote Python 3.10 CI remains pending until a reviewed commit is pushed.
- **Last verified commit:** `aa8803e8060af8cd704f81fb7c6903d0d48e2a6e` (Commit C, pushed)
- **Branch:** `feat/first-vertical-slice`
- **Completed in this task:** Independently verified the external full-validation evidence chain; corrected selection strategy provenance and cross-city visual selection without changing evaluation order; added a narrow manual-only Fishyscapes Lost & Found validation adapter/manifest contract; added tested NumPy pixel AP and FPR95 metrics; and documented the manual Lost & Found and FS Static preparation gates.
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, license `OPEN QUESTION`
- **First T4 development evidence:** CUDA forward succeeded with native logits `[1,19,64,128]` and aligned logits `[1,19,512,1024]`; two consecutive forwards were byte-identical. The run was Git-dirty because a temporary loader patch was applied in Colab, so it is feasibility evidence rather than the final clean reproducibility artifact.
- **Local environment evidence:** Python 3.11.9, PyTorch 2.13.0, Apple arm64, 8 GiB RAM; MPS built and available, CUDA unavailable. PyTorch was not added to project dependencies or `LICENSES.md`.
- **Local forward evidence:** CPU and MPS both produced finite native `[1,19,64,128]` and aligned `[1,19,512,1024]` logits. Both were byte-identical within backend across two repeats; CPU/MPS semantic masks matched. Cross-device native-logit maximum absolute difference was approximately `1.85e-5` and is diagnostic, not a scientific gate.
- **Cityscapes val evidence:** Approved archives matched SHA-256 pins; val-only extraction produced 500 image/label pairs across 3 cities, no train/test extraction, and deterministic manifest SHA-256 `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`.
- **Local campaign evidence:** The deterministic 1-image CPU, 5-image MPS, and 10-image MPS runs completed with zero failures. Evaluation times were approximately 0.777 s, 3.945 s, and 5.710 s respectively. Development-grid mIoU values were approximately 0.6632, 0.6121, and 0.6246; pixel accuracies were approximately 0.9637, 0.9514, and 0.9565. These small resized-grid development measurements are not a paper-protocol reproduction or a final scientific result.
- **Score evidence:** All four aligned-logit score summaries were finite. They are ID-only development summaries; no threshold, OOD performance, or anomaly-probability claim was made.
- **Artifact package evidence:** The final 10-image external evidence package contains 15 files, is 1,132,620 bytes, and has SHA-256 `40c0033c13de1a6641f3cf73fca317d695a2a79cc51d7832b3789362ede1d408`.
- **Full Cityscapes evidence:** External ZIP SHA-256 `756abf1a983b8eed11b22f0c10b3cabf093d6e614a4bec2a6d223c41202132b7`, 19,966,292 bytes, 43 CRC-valid entries, and 42/42 verified internal file hashes. Dataset and selection manifest hashes independently reproduced; semantic pixel totals were consistent; path/secret checks were clean.
- **Full Cityscapes measurements:** 500 selected, 500 successful, 0 failures; input `[1,3,1024,2048]`, native logits `[1,19,128,256]`, aligned logits `[1,19,1024,2048]`; mIoU `0.7875813077220126`, pixel accuracy `0.9619008903101843`, and mean class accuracy `0.8618737663500519`.
- **Timing boundary:** `827.00028256` seconds and mean `1.6539497549533844` seconds/sample measure the end-to-end evaluation pipeline, not pure inference latency or Jetson FPS. Peak PyTorch CUDA allocated memory was 217,726,976 bytes.
- **Scientific claim boundary:** This is the EdgeGuard-Road single-scale PIDNet-S evaluation, not a guaranteed reproduction of the official PIDNet paper protocol. All four score summaries were finite, but no OOD, threshold, calibration, or anomaly-probability claim was made.
- **Selection correction:** The completed campaign's 500 sorted samples and measurements remain valid, but its strategy label should have been `all_sorted_v1`, and its five visuals were all Frankfurt. Future `--all`, subset-size, and manifest modes now record honest strategies; visuals use independent deterministic city round-robin without changing evaluation order or measurements.
- **Fishyscapes boundary:** Lost & Found validation is `ood_development`; its underlying image terms and manual data preparation remain human gates. FS Static is generation-preparation only from human-authorized Cityscapes inputs. No automatic download, real Fishyscapes run, SMIYC access, threshold, training, calibration, context, morphology, or temporal work occurred.
- **Current blocker:** Fishyscapes Lost & Found images/annotations and any FS Static generator inputs require manual human acquisition, terms review, provenance recording, and manifest approval.
- **Next single action:** Human reviews the unstaged evidence/selection/Fishyscapes foundation diff and decides whether to approve a local commit.
- **Pending later human decisions:** Fishyscapes source terms and manifest; FS Static generator pin and generation; real OOD-development run interpretation; ONNX numerical gates; artifact promotion
- **Last local Python 3.11 verification:** 2026-07-26T23:18:43Z — Ruff check passed; Ruff format check passed for 93 files; mypy passed for 27 source files; pytest passed 174 with 2 expected opt-in skips; `git diff --check` passed
- **Python 3.10 compatibility verification:** Python 3.10.11, NumPy 2.2.6, mypy 1.20.2 — editable dev install, Ruff check, Ruff format check, mypy, and pytest passed; pytest reported 174 passed with 2 expected opt-in skips
- **Artifact status:** Checkpoint, upstream checkout, CPU/MPS tensors, extracted Cityscapes val, local evaluator outputs, full-validation ZIP, and smoke outputs remain outside Git. No dataset, checkpoint, tensor, generated visual, archive, or other large/binary artifact was added to the repository.
- **Git action status:** Commit C is pushed and local HEAD matches origin; the current follow-up diff is unstaged; nothing new was committed, pushed, merged, tagged, released, or promoted

This file is updated at the end of every material agent task. Measured results only;
never replace missing evidence with estimates.
