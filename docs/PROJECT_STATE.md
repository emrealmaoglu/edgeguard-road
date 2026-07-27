# Project State

- **Branch:** `feat/first-vertical-slice`
- **Current committed revision:**
  `0114d4e4778c1d6e53b6359e0a11f71eb15d2fb4`
- **Starting state for EG-DATA-002:** Clean and synchronized with
  `origin/feat/first-vertical-slice`
- **EG-OOD-001:** Complete and remotely verified at Commit D
  `345d9fd1dcff0a7aa9c54c6f3929c2c751c24c7c`
- **EG-THESIS-001:** Complete and remotely verified at
  `ee4460bda9b518a4e784cd43ad23d043ad15cd7b`
- **EG-DATA-001:** Complete and remotely verified at
  `0114d4e4778c1d6e53b6359e0a11f71eb15d2fb4`; Python 3.10 and 3.11 CI passed
- **EG-DATA-002:** Cityscapes Fine train archive validation, train-only staging,
  deterministic train-ID generation, streaming class/group analysis, three
  group-atomic split candidates, idempotent verification, evidence packaging, and a
  thin Colab wrapper are implemented and locally tested with synthetic ZIP fixtures
- **EG-DATA-002 real-data state:** The local runtime has no access to the approved
  private Drive mount. Neither approved archive was opened; no dataset or Drive
  directory was created, no real count/frequency/candidate was produced, and no
  evidence package was promoted
- **Approved runtime storage:** `EDGEGUARD_EXTERNAL_ROOT` resolves to the private
  `EdgeGuard/` root. Existing archives remain immutable under relative
  `private_inputs/`; prepared data and manifests target
  `datasets/cityscapes/fine/v1/` and `manifests/cityscapes/fine/v1/`
- **Archive identities awaiting real verification:**
  `leftImg8bit_trainvaltest.zip` →
  `3ccff9ac1fa1d80a6a064407e589d747ed0657aac7dc495a4403ae1235a37525`;
  `gtFine_trainvaltest.zip` →
  `40461a50097844f400fef147ecaf58b18fd99e14e4917fb7c3bf9c0d87d95884`
- **Ontology:** `edgeguard-ontology-v1` remains provisional with canonical SHA-256
  `24fd17b54a4aa461e004eaf8c5feebe7b3115c0906559ff24f3bd7f2e1510a10`;
  local validation does not freeze it
- **Split status:** `CSF-SPLIT-A/B/C` are implemented candidate definitions only.
  A real run must produce measured manifests and a
  `recommended_pending_human_approval` comparison; no candidate is selected or
  frozen
- **Official val:** `official_val_common_eval`; excluded from `train_select`, routine
  HPO, and temperature fitting; not sealed or previously unseen
- **Protected boundaries:** Cityscapes test labels and SMIYC remain unaccessed. No
  BDD100K, Mapillary, SOS, or Fishyscapes data was accessed. No training framework,
  model training, Colab campaign, or Jetson action occurred
- **Validation:** Ruff check passed; Ruff format check passed for 103 files; mypy
  passed for 29 source files; pytest passed 201 with 2 expected opt-in skips;
  `git diff --check` passed. Synthetic preparation tests cover unsafe/absolute/
  symlink/duplicate archive members, train-only selection, pairs, geometry, corrupt
  PNG, unknown IDs, deterministic masks/manifests/candidates, leakage, partial output,
  exact verification, and destination collision
- **Exact next action:** Human reviews this unstaged diff and authorizes a commit.
  After that exact commit is pushed, the human runs
  `notebooks/colab/03_prepare_cityscapes_fine_train.ipynb`, verifies the real evidence
  package, and selects or rejects one measured split candidate
- **Training gate:** `EG-SEG-001` remains blocked until the real preparation passes
  and the human explicitly freezes a split
- **Git action:** All EG-DATA-002 changes remain unstaged; no commit or push is
  authorized

Planned, implemented, locally tested, acquired, prepared, Colab measured, human
accepted, and remotely verified remain distinct states.
