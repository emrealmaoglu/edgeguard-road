# Project State

- **Branch:** `feat/first-vertical-slice`
- **Current committed revision:**
  `ee4460bda9b518a4e784cd43ad23d043ad15cd7b`
- **Starting state for EG-DATA-001:** Clean and synchronized with
  `origin/feat/first-vertical-slice`
- **EG-OOD-001:** Complete and remotely verified at Commit D
  `345d9fd1dcff0a7aa9c54c6f3929c2c751c24c7c`; Python 3.10 and 3.11 CI passed
- **EG-THESIS-001:** Complete and remotely verified at
  `ee4460bda9b518a4e784cd43ad23d043ad15cd7b`; Python 3.10 and 3.11 CI passed
- **EG-DATA-001:** Runtime storage-root proposal, lifecycle, dataset access matrix,
  four-namespace ontology, BDD100K mappings, Cityscapes split-analysis method, and
  human gate checklist are locally implemented and tested; human freeze is pending
- **Storage proposal:** `EDGEGUARD_EXTERNAL_ROOT` will point at the human-approved
  `EdgeGuard/` project root. Existing `private_inputs/` remains a legacy/current
  child whose migration or reuse is deferred. No Drive directory was created or
  moved.
- **Resource boundary:** Approximately 5 TB private Drive is available for canonical
  external storage; the local Apple M1 has approximately 18 GiB free and is not a
  heavy-data store or transfer relay
- **Ontology proposal:** Provisional `edgeguard-ontology-v1` preserves separate
  `semantic_cityscapes19`, `known_detection10`, `ood_binary`, and
  `risk_operational` namespaces. Ten reviewed BDD100K names map explicitly; unknown
  source classes fail closed. Mapillary mapping remains deferred.
- **Current implemented baseline:** Strict PIDNet-S checkpoint validation,
  native/aligned-logit inference, four uncertainty scores, Cityscapes evaluation,
  manual-only Fishyscapes adapter foundation, and NumPy AP/FPR95 metrics
- **Measured Cityscapes baseline:** 500 selected, 500 successful, 0 failures; mIoU
  `0.7875813077220126`, pixel accuracy `0.9619008903101843`, mean class accuracy
  `0.8618737663500519`
- **Forward dataset roles:** Cityscapes Fine train awaits distribution-based
  `train_fit`/`train_select`/`train_calibration` analysis; official val is
  `official_val_common_eval`, used for common final-model evaluation but not HPO,
  `train_select`, temperature fitting, or sealed/unseen claims; Fishyscapes Static
  is OOD development/HPO; full Lost & Found is a one-time frozen holdout; SMIYC
  remains the actual unaccessed sealed-final boundary
- **Acquisition/training state:** No dataset was downloaded, extracted, inspected,
  copied, or marked acquired. No framework was installed; no training, HPO, Colab
  campaign, export, Jetson access, or Drive mutation occurred.
- **Exact next implementation task:** `EG-DATA-002 — Cityscapes Fine train
  preparation`; it must not start before human approval of the applicable Drive,
  Cityscapes access, ontology, and task gates
- **Parallel detector state:** `EG-DET-001` remains blocked on BDD100K
  access/terms/package decisions and the proposed YOLO path's AGPL decision
- **Human gates:** Drive root convention; Cityscapes and BDD100K access/terms;
  Fishyscapes sources; synthetic object-mask sources; temporal dataset; YOLO AGPL;
  title/university decision; actual Colab accelerator availability; Jetson storage
  inventory; ontology and first acquisition freeze
- **Validation:** 2026-07-27T00:19Z — Ruff check passed; Ruff format check passed for
  100 files; mypy passed for 28 source files; pytest passed 183 with 2 expected
  opt-in skips; `git diff --check` passed. Ontology tests passed 9/9; all 3 notebooks
  parsed and their 3 focused integration tests passed; path, secret, >1 MiB, binary,
  and forbidden-artifact scans were clean. Ontology canonical payload SHA-256 is
  `24fd17b54a4aa461e004eaf8c5feebe7b3115c0906559ff24f3bd7f2e1510a10`.
- **Git action:** All EG-DATA-001 changes remain unstaged; no commit or push is
  authorized

Planned, implemented, locally tested, acquired, Colab measured, Jetson measured, and
human accepted remain distinct states.
