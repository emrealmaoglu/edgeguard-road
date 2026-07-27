# Project State

- **Repository/branch:** `.` on `feat/first-vertical-slice`
- **Current committed revision:** `f9b489338517e32dcabe0110614993f046af43ed`,
  synchronized with `origin/feat/first-vertical-slice` at task start
- **Completed foundation:** EG-OOD-001, EG-THESIS-001, EG-DATA-001 and the
  EG-DATA-002 repository implementation are remotely verified. EG-SEG-001 and the
  prior EG-SEG-002 automation/hotfix milestones are committed on this branch.
- **Local-first readiness gate:** Explicit runtime roots, terminal uv-bootstrap failure
  evidence, an 11-phase synthetic readiness command, read-only storage inventory,
  zero-redownload staging dry-run, split compatibility/smoke notebooks, and a manual
  Linux x86 CPU workflow are implemented and locally tested.
- **Compatibility state:** Path A preserves the hosted Python/Torch/TorchVision/CUDA
  stack and uses MMEngine 0.10.7 plus `mmcv-lite` 2.1.0. Path B is an automatic
  isolated Python 3.11 fallback with NumPy 1.26.4, Torch 2.1.1 CUDA 12.1 wheels and
  full prebuilt MMCV 2.1.0. Both use MMSeg commit
  `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`; neither uses OpenMIM or permits an
  MMCV source build. Bootstrap failures now terminate with structured status, logs,
  classification and interpreter-script/PATH diagnostics.
- **Dataset state:** The real split rebuild consumes only the existing dataset
  manifest and group summary and writes `split-policy-v1`. Staging reuses a verified
  identity-bound Drive bundle by default; creating one requires an explicit command.
  It copies and verifies active data under `/content`. No archive was extracted and no
  mask was regenerated in this repository task.
- **Acquisition boundary:** BDD100K, Fishyscapes Static, conditional Cityscapes
  coarse/trainextra, an unresolved temporal choice and optional approved demo videos
  are represented. Missing runtime access is a structured block; the temporal entry
  is policy-blocked. SMIYC and full Fishyscapes Lost & Found are excluded.
- **Observability/resume:** Long jobs emit atomic `run_status.json`, live stage output,
  JSONL training/validation records, TensorBoard events where MMEngine supports them,
  stdout/stderr logs, finite/disk/stall checks and SHA-verified Drive recovery. Resume
  validates config, Git, framework, dataset, split, initialization and precision
  identities before loading.
- **Local validation:** Python 3.11.9 on Apple M1 completed the project-owned fixture,
  split/stage/load/loss/metric/checkpoint-resume flow and all five random-weight MMSeg
  CPU forward/backward/checkpoint-resume probes. Ruff, formatting, mypy and pytest
  passed (`260 passed`, `2` expected skips). These are compatibility results, not
  scientific accuracy or performance evidence.
- **Protected boundaries:** No official Cityscapes val, Fishyscapes, Lost & Found,
  SMIYC, detector, Jetson, CUDA or real-data training execution occurred in this task.
- **Next execution:** Create and push the single reviewed local-first commit, verify
  normal Python 3.10/3.11 CI, then pass the manual Linux x86 five-model CPU workflow.
  Colab remains blocked until those remote gates pass.

Planned, implemented, locally tested, Colab measured, human accepted and remotely
verified remain distinct states.
