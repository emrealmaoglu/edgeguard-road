# Project State

- **Repository/branch:** `.` on `feat/first-vertical-slice`
- **Current committed revision:** `8343ed582c03be69a1fd753617a1a3c2566ab20b`,
  synchronized with `origin/feat/first-vertical-slice` at task start
- **Completed foundation:** EG-OOD-001, EG-THESIS-001, EG-DATA-001 and the
  EG-DATA-002 repository implementation are remotely verified. EG-SEG-001 is the
  current committed autonomous semantic training laboratory.
- **EG-SEG-002 repository implementation:** OpenMIM-free current-Colab compatibility
  cascade, deterministic Drive-to-`/content` Cityscapes staging, persistent approved
  acquisition queue, long-run telemetry/recovery, group-aware 25/50/100% fractions,
  and the five-model 100-step smoke runner are implemented and locally tested.
- **Compatibility state:** Path A preserves the hosted Python/Torch/TorchVision/CUDA
  stack and uses MMEngine 0.10.7 plus `mmcv-lite` 2.1.0. Path B is an automatic
  isolated Python 3.11 fallback with NumPy 1.26.4, Torch 2.1.1 CUDA 12.1 wheels and
  full prebuilt MMCV 2.1.0. Both use MMSeg commit
  `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`; neither uses OpenMIM or permits an
  MMCV source build. The selected path remains pending the real Colab probe.
- **Dataset state:** The real split rebuild consumes only the existing dataset
  manifest and group summary and writes `split-policy-v1`. Staging creates/reuses one
  identity-bound Drive bundle, copies and verifies it under `/content`, and trains
  from ephemeral storage. No archive was extracted and no mask was regenerated in
  this repository task.
- **Acquisition boundary:** BDD100K, Fishyscapes Static, conditional Cityscapes
  coarse/trainextra, an unresolved temporal choice and optional approved demo videos
  are represented. Missing runtime access is a structured block; the temporal entry
  is policy-blocked. SMIYC and full Fishyscapes Lost & Found are excluded.
- **Observability/resume:** Long jobs emit atomic `run_status.json`, live stage output,
  JSONL training/validation records, TensorBoard events where MMEngine supports them,
  stdout/stderr logs, finite/disk/stall checks and SHA-verified Drive recovery. Resume
  validates config, Git, framework, dataset, split, initialization and precision
  identities before loading.
- **Scientific status:** No real compatibility path, D/E candidate, Cityscapes smoke
  loss, semantic metric, accelerator throughput or promotion result was produced
  locally. EG-SEG-002 is not `colab_measured`; common screening has not started.
- **Protected boundaries:** No official Cityscapes val, Fishyscapes, Lost & Found,
  SMIYC, detector, Jetson or training-framework execution occurred in this task.
- **Next execution:** At the exact remotely verified EG-SEG-002 commit, run
  `notebooks/colab/05_semantic_five_model_smoke.ipynb`. Separately,
  `notebooks/colab/06_acquire_edgeguard_datasets.ipynb` may process only an approved
  acquisition queue item with runtime-only access values.

Planned, implemented, locally tested, Colab measured, human accepted and remotely
verified remain distinct states.
