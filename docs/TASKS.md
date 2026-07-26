# Tasks

## Backlog

- Stage 3: only after a successful human-reviewed real forward and source-integration
  decision, implement the chosen PIDNet path and minimum Cityscapes val adapter.
- Stage 4: only after scoring works and license/access approval, implement the
  Fishyscapes Lost & Found validation development adapter.
- Stage 5: run a native-logit ONNX measurement pilot without pre-freezing numerical
  gates.

## Ready

- Human upload of the config-pinned `PIDNet_S_Cityscapes_val.pt` and Colab execution
  of `notebooks/colab/01_pidnet_single_image_spike.ipynb`; filename and hash are
  checked automatically before load.

## In Progress

- Stage 2 real-forward gate: upstream sample and checkpoint identity are pinned;
  awaiting human Colab upload/execution and visual sanity review.

## Done

- WP-00 governance foundation.
- WP-01 repository, package, contract, smoke, doctor, test, and CI foundation.
- WP-02 environment inventory tooling and configurable probe timeout.
- Stage 1 source/evidence review and human approval of the PIDNet commit, dataset
  roles, sealed SMIYC boundary, and restricted academic checkpoint usage.
- Stage 2 local preparation: legal-image loader/manifest, preprocessing, strict
  checkpoint guards, PIDNet runner, native/aligned output metadata, MSP/entropy,
  repeat diagnostics, Colab execution notebook, fixed-checkout primary/fallback
  sample provenance, official checkpoint access probe, and 98 passing local tests.
