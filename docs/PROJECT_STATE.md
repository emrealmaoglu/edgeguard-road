# Project State

- **Branch:** `feat/first-vertical-slice`
- **EG-THESIS-001 base revision:**
  `345d9fd1dcff0a7aa9c54c6f3929c2c751c24c7c` (Commit D)
- **Starting repository state for EG-THESIS-001:** clean; branch and commit matched
  the required gate
- **EG-THESIS-001 review state:** The complete 17-file documentation block was
  accepted by the human owner for one coherent local commit; remote CI remains a
  separate verification gate
- **EG-OOD-001:** Complete and remotely verified; GitHub Actions passed on Python
  3.10 and Python 3.11
- **EG-THESIS-001:** Documentation, claim boundary, agent memory, experiment
  matrix, and task-system migration implemented, locally validated, and accepted
  for commit; remote verification is pending
- **Formal title status:** The expanded English and Turkish titles are proposals
  pending human and university approval; the prior title remains the fallback
- **Current implemented baseline:** Strict PIDNet-S checkpoint validation,
  native/aligned-logit inference, four uncertainty scores, Cityscapes evaluation,
  a manual-only Fishyscapes adapter foundation, and NumPy AP/FPR95 metrics
- **Measured Cityscapes baseline:** 500 selected, 500 successful, 0 failures;
  mIoU `0.7875813077220126`, pixel accuracy `0.9619008903101843`, mean class
  accuracy `0.8618737663500519`
- **Baseline provenance:** The full evaluation used clean Commit C
  `aa8803e8060af8cd704f81fb7c6903d0d48e2a6e`; external evidence ZIP SHA-256
  `756abf1a983b8eed11b22f0c10b3cabf093d6e614a4bec2a6d223c41202132b7`
- **Claim boundary:** The measured run is the project's single-scale PIDNet-S
  Cityscapes-val evaluation, not a guaranteed reproduction of the official
  PIDNet paper protocol. It provides no OOD, calibration, temporal, detector,
  Jetson, or anomaly-probability result.
- **Fishyscapes state:** Manual-only adapter and metric foundation exists; no real
  Fishyscapes inference has run. FS Static is planned for development/HPO, and
  the complete Lost & Found validation set is planned as a one-time frozen
  holdout after all relevant decisions are frozen.
- **Training and acquisition state:** No new dataset acquisition, framework
  installation, model training, HPO, detector run, calibration run, synthetic
  data generation, ONNX/TensorRT export, or Jetson benchmark has started
- **Sealed boundary:** SMIYC RoadObstacle21 and RoadAnomaly21 remain unaccessed
  sealed-final datasets
- **Storage state:** Approximately 5 TB of private Google Drive capacity is
  available for canonical project storage; the local Apple M1 Mac has
  approximately 18 GiB free and must not hold expanded datasets
- **Compute state:** Week-5 required queue is provisionally 70–130 measured GPU
  hours; the full aggressive queue adds 90–220 GPU hours. Estimates must be
  recalibrated after each experiment family's first real throughput run.
- **Immediate next task after human acceptance:** `EG-DATA-001 — Storage, access
  and ontology gate`; it has not started
- **Human gates:** Proposed thesis title and university process; private Drive
  root/naming; dataset terms and acquisition; semantic and detector ontology;
  Cityscapes group split; framework/model source pins; HPO budgets; calibration
  and threshold protocol; Lost & Found holdout opening; SMIYC sealed evaluation;
  artifact promotion; Jetson access and deployment decisions
- **Validation:** 2026-07-26T23:49Z — Ruff check passed; Ruff format check passed
  for 98 files; mypy passed for 27 source files; pytest passed 174 with 2 expected
  opt-in skips; `git diff --check` passed. All 3 notebooks parsed and their 3
  focused integration tests passed. Changed-file path, secret, >1 MiB, binary,
  and forbidden-artifact scans were clean.
- **Publication at this record's authoring gate:** Human approved one
  documentation-only commit and push to `origin/feat/first-vertical-slice`;
  resulting Git/CI state must be read from repository history and the delivery
  report rather than inferred in advance

This file records measured facts and explicit planning state. Planned,
implemented, locally tested, Colab measured, Jetson measured, and human accepted
must remain distinct.
