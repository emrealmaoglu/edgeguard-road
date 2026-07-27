# Agent Handoff

- **Task:** Local-first Colab readiness gate
- **Branch/base:** `feat/first-vertical-slice` at task-start commit
  `f9b489338517e32dcabe0110614993f046af43ed`
- **Repository result:** Implemented and locally tested; remote CI and manual Linux CPU
  verification pending
- **No scientific result claimed:** Only synthetic, random-weight compatibility probes
  ran. No Drive data, real Cityscapes sample, pretrained weight, training campaign or
  accelerator performance measurement was used.

## Colab execution

Do not open Colab yet. First push the single local-first commit, pass normal Python
3.10/3.11 CI, and manually pass `.github/workflows/semantic-framework-cpu-probe.yml`.
After those gates, `notebooks/colab/04_colab_semantic_compatibility_probe.ipynb` runs
only the clean compatibility install and five synthetic CUDA probes. It does not mount
Drive. `notebooks/colab/05_semantic_five_model_smoke.ipynb` accepts only a successful
same-runtime compatibility receipt before it may mount Drive or stage data.

Persistent outputs are external to Git:

- recovery: `checkpoints/segmentation/<model>/recovery/attempt-*/`
- small smoke evidence: `experiments/segmentation/EG-SEG-002/`
- split policy: `manifests/cityscapes/fine/v1/split-policy-v1/`
- reusable bundle: `datasets/cityscapes/fine/bundles/`

`notebooks/colab/06_acquire_edgeguard_datasets.ipynb` lists and processes the approved
queue. BDD100K, Fishyscapes Static, conditional Cityscapes coarse/trainextra and most
external assets still require an official login, terms decision, signed URL or exact
runtime filename/size/SHA-256. The temporal source remains blocked until selected.

## Evidence contract

The local command runs 11 visible phases and emits an atomic status plus a small,
path-sanitized evidence ZIP. The real Mac CPU probe constructed all five pinned MMSeg
models and verified finite forward/backward plus model/optimizer/scheduler resume for
each. The storage inventory is read-only; the empty local check predicted zero
Cityscapes download bytes and correctly blocked on missing verified private assets.

Do not access official Cityscapes val, full Fishyscapes Lost & Found or either SMIYC
set in this milestone. Do not infer training quality from compatibility smoke results.
