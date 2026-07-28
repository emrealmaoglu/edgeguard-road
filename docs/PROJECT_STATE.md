# Project State

- **Repository/branch:** local `rescue/semantic-first` from `a786522`; working tree
  remains uncommitted because no commit or push was authorized.
- **Active work package:** `EG-MULTIDOMAIN-001`.
- **Research scope:** domain generalization for Cityscapes19 semantic segmentation with
  Cityscapes, BDD100K, and IDD20K source data; five lightweight MMSeg models; bounded
  HPO; class imbalance; reliability; ACDC; sealed WildDash 2/MUSES; ONNX/demo.
- **Implemented source-data path:** strict versioned ontology; BDD100K/IDD20K native
  adapters; official-count, corrupt, geometry, label, exact-hash and perceptual-hash
  audit; group-atomic 80/15/5 candidates; explicit human freeze; cross-domain duplicate
  evidence; domain-uniform distributed sampler; pooled train-fit rare classes and
  bounded mean-one median-frequency weights.
- **Validation separation:** BDD100K/IDD20K `--source-split val` creates only
  `official_source_val`; it requires frozen source manifests for overlap checking and
  cannot enter training/HPO. Cityscapes official validation remains final-only.
- **Implemented experiment path:** SegFormer-B0, PIDNet-S, DDRNet-23-Slim,
  BiSeNetV2, and Fast-SCNN use common random initialization, AdamW, effective batch,
  augmentation, 512x1024 geometry, and optimizer-step budgets. Only the three approved
  dataset compositions are accepted.
- **Implemented HPO path:** measured three-domain screening + validated ONNX gate;
  top-two selection; 12-trial seeded Optuna TPE per model; 1,500/3,000-step halving;
  interruption closure/resume; duplicate-trial pruning; domain-macro objective with
  rare-class tie-break.
- **Implemented reliability/external path:** equal-pixel, manifest-bound global
  temperature scaling; ECE/NLL/Brier/confidence/entropy; ACDC/source-domain reporting;
  hash-bound sealed release; WildDash/MUSES prediction package; official server-score
  record that is never renamed Cityscapes19 mIoU.
- **Implemented delivery path:** static ONNX export and numerical/latency validation;
  aspect-preserving inference; Streamlit demo; one Colab notebook; evidence-only tables;
  Git-aware append-only run ledgers; external artifact schema examples.
- **Local quality:** on 2026-07-28, Python 3.11 all-tree Ruff format/lint passed for
  248 files, mypy passed for 97 source files, five public CLI help smokes passed,
  notebook JSON/eight code cells compiled without stored outputs, and the full suite
  passed with `363 passed, 10 skipped`.
- **Scientific evidence:** no new multi-domain real-data training, HPO, calibration,
  ACDC, sealed external, or ONNX model result exists. Local fixture tests are engineering
  evidence only and `--allow-fixture-count` manifests are rejected by training.
- **Local data/compute:** no licensed Cityscapes/BDD100K/IDD20K/ACDC/WildDash/MUSES
  corpus is available in this workspace; this Apple M1 host has no CUDA training path.
- **External gates:** licensed data access; three real source audits; human manifest
  freezes; CUDA smoke/pilot/screening; human top-two/config freeze; final training;
  source/ACDC evaluation; sealed release and official submission. Jetson remains optional.
- **Protected boundaries:** no dataset/model download, real-data processing, Drive
  mutation, GPU training, official validation, sealed-data inference, submission,
  checkpoint/ONNX artifact, Jetson action, commit, push, or PR occurred.
- **Next action:** execute the notebook in audit-only mode on licensed source roots,
  review/freeze all three training manifests, then run the five 50-step CUDA smokes.

Implemented, locally verified, externally executed, scientifically measured, and human
accepted remain separate states.
