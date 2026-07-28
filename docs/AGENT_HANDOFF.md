# Agent Handoff

- **Milestone:** multi-domain semantic rescue implementation ready for human diff review.
- **Branch:** local `rescue/semantic-first`, uncommitted on base `a786522`.
- **Classification:** implementation plus local engineering validation; no new
  scientific result.

## Completed

- Cityscapes19 ontology v2 with strict BDD train-ID and loss-aware IDD source-ID mapping.
- BDD100K/IDD20K train and withheld official-validation audits, hash/provenance records,
  group-atomic split candidates, explicit freeze, and source/external overlap gates.
- Three approved dataset combinations, equal-domain sampler, pooled rare-five set, and
  train-fit-only weighted CrossEntropy contract.
- Common five-model, optimizer-step MMSeg path and separately verified classification
  pretraining manifest for later finalist-only comparison.
- Measured-screening-only top-two selection and resumable bounded Optuna HPO.
- Equal-domain global temperature calibration and reliability/domain-shift tables.
- Sealed WildDash/MUSES/KITTI release, prediction archive, and server-result binding.
- ONNX validation/benchmark, letterboxed inference, Streamlit demo, Colab orchestration,
  evidence report generator, and append-only Git-aware run ledger.

## Verification

- Focused Ruff and mypy: passed for the rescue runtime and public scripts.
- Focused pytest: `30 passed`.
- Full pytest: `363 passed, 10 skipped`.
- All-tree Ruff format/lint: passed for 248 files; mypy passed for 97 source files.
- Five public CLI help smokes and `git diff --check`: passed.
- Notebook JSON: valid; all eight code cells compile; stored outputs are empty.
- Real MMSeg/CUDA multi-domain execution: not run on this host.
- Real dataset audit/training/evaluation: not run; no scientific metric produced.

## External execution order

1. Mount licensed Cityscapes, BDD100K, and IDD20K roots; run audits without fixture flags.
2. Review audit/duplicate/mapping/split evidence and explicitly freeze all manifests.
3. Run five-model 50-step smoke, 2,000-step pilot, and 6,000-step screening on CUDA.
4. Produce three-domain evaluation + ONNX records, review top two, then run bounded HPO.
5. Run dataset-composition and CE/weighted-CE ablations; freeze finalist checkpoints.
6. Fit one global source calibration temperature; open official source validation and ACDC.
7. Author the hash-bound sealed release and perform one WildDash/MUSES submission.
8. Build final tables/demo; run Jetson only if approved hardware is available.

No commit, push, PR, data acquisition, Drive mutation, model promotion, external
submission, or device action was performed. Commit/push still requires explicit approval.
