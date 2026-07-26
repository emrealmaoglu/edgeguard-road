# Tasks

## Backlog

- P0 after Commit C: human-approved push of the exact reviewed commit, followed by
  the clean single-image preflight and full 500-image Colab campaign.
- P1 after the full Colab campaign: optional 20-image MPS run, minimal identity-protected
  resume, boundary/interior analysis, correlations, and wider percentile reports.
- P2 after the first full Colab campaign: ONNX/ORT and Fishyscapes development work.

## Ready

- Human review of the reported Commit C SHA and a separate push decision.
- After push and exact-commit approval, run the clean commit in Colab.

## In Progress

- No implementation block is in progress. The full-resolution Colab runner is ready,
  but push and campaign execution remain human-gated.

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
- Commit B `05f72b4bdd8c3a69bba62a9023f61ceb2504c355`: local strict checkpoint
  validation, clean CPU verification, Cityscapes val-only preparation, deterministic
  500/500/3 manifest, adapter, and label LUT.
- Local environment/checkpoint foundation: Python 3.11.9 and environment-only
  PyTorch 2.13.0; real 481-key checkpoint validated with strict load; CPU and MPS
  forwards finite and repeatable within backend.
- Cityscapes val foundation: approved archive hashes, safe val-only extraction,
  500/500 pairs across 3 cities, deterministic manifest, explicit label LUT, and
  real-root opt-in validation.
- Core local evaluator: streaming semantic metrics and finite MSP, predictive entropy,
  MaxLogit, and Energy summaries derived from `aligned_logits`; deterministic subset
  selection; path-free minimal output; output-collision protection; and small visuals.
- Mandatory local campaign: 1-image CPU, 5-image MPS, and 10-image MPS runs completed
  with zero failures. Results are resized-grid development measurements, not a
  reproduction of the official PIDNet paper protocol.
- Colab preparation: full-resolution config, thin execution notebook, required
  single-image preflight, full 500-image invocation, and deterministic artifact
  packaging are locally validated. No Colab campaign result has been claimed.
- Commit C human review and local commit approval. The commit remains local until a
  separate push authorization is given.
