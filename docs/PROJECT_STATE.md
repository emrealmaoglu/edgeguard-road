# Project State

- **Repository/branch:** `.` on `feat/first-vertical-slice`
- **Current committed revision:**
  `9d269c35f0adc08be193ec3ee50b7c505c485fae`, synchronized with
  `origin/feat/first-vertical-slice` at task start
- **EG-OOD-001, EG-THESIS-001, EG-DATA-001:** Completed and remotely verified
- **EG-DATA-002 repository implementation:** Committed, pushed, and remotely verified
  at the current revision; Python 3.10 and 3.11 CI passed
- **EG-DATA-002 measured split evidence:** Human-supplied preparation evidence records
  2,975 samples, 18 cities, and 1,885 city+sequence groups. A/B/C are rejected:
  measured candidate B placed 196 Hanover samples from one group in calibration and
  248 of 438 selection samples in Hamburg
- **Diversity split correction:** The split-only D/E policy implementation is locally
  tested. It consumes the existing dataset manifest and group summary without image
  extraction or mask regeneration, enforces the recorded diversity constraints, and
  emits one cryptographically identified `policy_selected` result. The real manifest
  rebuild has not run in this action, so its selected candidate/hash remain pending
- **EG-SEG-001 repository implementation:** Pinned framework/common/five-model
  configs, dependency-light experiment/data/checkpoint/registry/logit contracts,
  runtime install/probe/verifier commands, and thin Colab notebook are implemented;
  dependency-free local validation passed
- **Framework proposal:** MMSegmentation `v1.2.2` at
  `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`, MMEngine `0.10.7`, MMCV `2.1.0`,
  and OpenMIM `0.3.9`. Status remains `proposal_pending_colab_probe`; no training
  framework was installed locally
- **Five model specs:** Fast-SCNN, BiSeNetV2, PIDNet-S, DDRNet-23-Slim, and
  SegFormer-B0 each declare 19 classes, ignore 255, baseline `512×1024`, random
  no-download stack-probe initialization, and a direct native-logit contract.
  Fast-SCNN and BiSeNetV2 use random project-training baselines. Other pretrained
  source identities remain unresolved human inputs
- **Dataset handoff:** Real training requires the EG-DATA-002 dataset manifest and a
  `policy_selected` D/E manifest binding policy version/config hash, candidate hash,
  dataset-manifest hash, ontology hash, and all sample roles. No separate
  human-selected flag is required. Only `train_fit` updates gradients and
  `train_select` selects models;
  `train_calibration` and `official_val_common_eval` are excluded from routine model
  selection
- **Ontology:** `edgeguard-ontology-v1` remains provisional with SHA-256
  `24fd17b54a4aa461e004eaf8c5feebe7b3115c0906559ff24f3bd7f2e1510a10`
- **Protected boundaries:** No private Drive access, real Cityscapes processing,
  official-val model selection, Fishyscapes/SMIYC access, scientific training, GPU
  probe, framework installation, Jetson action, stage, commit, or push occurred
- **Local validation:** Ruff check passed; Ruff format check passed for 120 files;
  mypy passed for 37 source files; pytest passed 227 with 2 expected opt-in skips;
  `git diff --check` passed. The Cityscapes-split/training-focused subset passed 39
  tests, and a synthetic 1,890-group rebuild completed in 15.298 seconds
- **Validation-interval evidence:** The real runner requires and records train loss,
  `train_select` loss, `train_select` mIoU, all 19 per-class IoUs, learning rate, and
  explicit generalization-gap inputs at every validation interval
- **EG-SEG-002 gate:** Blocked on the real D/E policy rebuild, successful exact-commit
  five-model Colab stack probe, and human framework/pretrained-source approval
- **Exact next action:** After this milestone is committed and remotely verified, run
  `notebooks/colab/04_semantic_training_stack_probe.ipynb` at the exact clean commit;
  no real semantic training starts in that notebook

Planned, implemented, locally tested, Colab measured, human accepted, and remotely
verified remain distinct states. Synthetic stack results can never support semantic
accuracy or scientific throughput claims.
