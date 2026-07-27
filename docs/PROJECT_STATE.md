# Project State

- **Repository/branch:** `.` on `feat/first-vertical-slice`.
- **Latest executable milestone:** campaign implementation through `add61f7`; a final
  documentation-only handoff commit follows this measured local run.
- **Campaign orchestration:** `python -m edgeguard.campaign` provides `init`, `plan`,
  `run`, `resume`, `status`, and `report` over one atomically written, lock-protected
  campaign state. Completed stages are reused only after identity and artifact-hash
  verification.
- **Clean local-mini result:** Campaign `eg-local-mini-add61f7` completed all 19
  bounded stages at clean commit `add61f7703110e3a901976ceda8a89139cafa7bb`.
  It exercised project-owned synthetic data, semantic training/checkpoint/resume,
  OOD/calibration, a trainable anomaly-head candidate, detection contracts,
  contextual risk, temporal persistence, export feasibility, and reporting.
- **Five-model Mac result:** The pinned MMSeg CPU path constructed Fast-SCNN,
  BiSeNetV2, PIDNet-S, DDRNet-23-Slim, and SegFormer-B0 and completed finite
  forward/backward plus model/optimizer/scheduler checkpoint-resume checks for each.
  These are random-weight compatibility results, not model quality measurements.
- **Reuse and recovery:** A second run reused 19/19 completed stages and executed
  none. A bounded interruption after `semantic_smoke` produced a failed terminal
  record; compatible resume completed the campaign on attempt two. Corrupt recovery,
  commit/profile mismatch, stale heartbeat, copy failure, and artifact hash mismatch
  are covered by tests.
- **Notebook handoff:** Five thin campaign notebooks completed one shared 19-stage
  local-mini state under the local harness. Notebook order is not the source of truth.
- **Export boundary:** Five project-owned tiny export surrogates passed ONNX checker
  and ONNX Runtime numerical comparison on the Mac; maximum absolute difference was
  approximately `2.98e-08`. This is plumbing evidence, not export evidence for the
  five production architectures and not Jetson performance.
- **Reports:** The clean campaign generated an assistant pack and a thesis-figure
  pack under `.local/edgeguard-campaign-final-add61f7/reports/`. Every generated
  local figure is marked `NON-SCIENTIFIC PIPELINE VALIDATION`; unsupported figures
  are explicitly skipped rather than populated with placeholder measurements.
- **Storage:** `.local` grew from `381060` KiB to `411740` KiB during the clean
  campaign, including five-model and report artifacts: `30680` KiB additional local
  cache, below the 2 GiB campaign limit.
- **Protected boundaries:** No real dataset or pretrained weight was downloaded; no
  Drive mutation, Colab session, official Cityscapes-val access, Fishyscapes access,
  Lost & Found holdout opening, SMIYC access, TensorRT build, Jetson run, scientific
  training, model ranking, threshold promotion, or scientific claim occurred.
- **Remote gates:** Normal Python 3.10/3.11 CI and the Linux x86 CPU framework workflow
  must pass on the final branch revision before remote verification is complete.
- **Next execution boundary:** Only platform/data-gated work remains: execute the
  exact-commit Colab profiles with approved private data, perform production-model
  export validation, and later run separately authorized Jetson/TensorRT checks.

Planned, implemented, locally tested, Colab measured, Jetson measured, human accepted,
and remotely verified remain distinct states.
