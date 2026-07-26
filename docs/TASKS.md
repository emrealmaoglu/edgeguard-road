# Tasks

## Backlog

- P0 after Commit B: semantic metrics, MSP/entropy/MaxLogit/Energy, the minimal
  path-free runner, deterministic 1/5/10-image local campaign, and thin full-resolution
  Colab preparation.
- P1 after the core runner: optional 20-image MPS run, minimal identity-protected
  resume, boundary/interior analysis, correlations, and wider percentile reports.
- P2 after the first full Colab campaign: ONNX/ORT and Fishyscapes development work.

## Ready

- Human review and explicit Commit B approval for the local checkpoint and
  Cityscapes val foundation diff.
- After Commit B, rerun a clean CPU single-image verification, then begin metrics,
  four scores, and the minimal runner.

## In Progress

- Commit B review: real checkpoint verification, local CPU/MPS forward, val-only
  Cityscapes preparation, deterministic 500/500/3 manifest, adapter, and LUT are
  complete and awaiting human review.

## Done

- WP-00 governance foundation.
- WP-01 repository, package, contract, smoke, doctor, test, and CI foundation.
- WP-02 environment inventory tooling and configurable probe timeout.
- Stage 1 source/evidence review and human approval of the PIDNet commit, dataset
  roles, sealed SMIYC boundary, and restricted academic checkpoint usage.
- Stage 2 local preparation: legal-image loader/manifest, preprocessing, strict
  checkpoint guards, PIDNet runner, native/aligned output metadata, MSP/entropy,
  repeat diagnostics, Colab execution notebook, fixed-checkout primary/fallback
  sample provenance, official checkpoint access probe, and hardened loader/notebook
  validation with 116 passing local tests.
- Stage 2 development feasibility: T4/CUDA produced native logits
  `[1,19,64,128]` and aligned logits `[1,19,512,1024]`; two consecutive forwards
  were byte-identical. This is not a clean reproducibility artifact, an OOD
  performance result, or evidence that MSP/entropy are anomaly probabilities.
- Commit A `9a734358332e8221af940d323e103b3adda376ad`: Stage 2 loader,
  notebook reproducibility, path sanitization, and evidence closure.
- Local environment/checkpoint foundation: Python 3.11.9 and environment-only
  PyTorch 2.13.0; real 481-key checkpoint validated with strict load; CPU and MPS
  forwards finite and repeatable within backend.
- Cityscapes val foundation: approved archive hashes, safe val-only extraction,
  500/500 pairs across 3 cities, deterministic manifest, explicit label LUT, and
  real-root opt-in validation.
