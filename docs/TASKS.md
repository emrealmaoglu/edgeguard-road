# Tasks

## Backlog

- Stage 3: only after a successful human-reviewed real forward and source-integration
  decision, implement the chosen PIDNet path and minimum Cityscapes val adapter.
- Stage 4: only after scoring works and license/access approval, implement the
  Fishyscapes Lost & Found validation development adapter.
- Stage 5: run a native-logit ONNX measurement pilot without pre-freezing numerical
  gates.

## Ready

- Human review and explicit Commit A approval for the bounded Stage 2 closure diff.
- After Commit A, begin manual local PyTorch inventory, real checkpoint verification,
  and the `512×1024` CPU forward. PyTorch remains environment-only.

## In Progress

- Stage 2 closure review: the first T4/CUDA development forward proved feasibility
  but used a temporary Git-dirty loader patch. The bounded loader, notebook,
  repository-origin, and path-sanitization diff is awaiting human review.

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
