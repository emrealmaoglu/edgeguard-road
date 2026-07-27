# Agent Handoff

- **Task:** `EG-SEG-002 — Current-Colab compatibility, persistent acquisition and
  five-model Cityscapes smoke`
- **Branch/base:** `feat/first-vertical-slice` at
  `8343ed582c03be69a1fd753617a1a3c2566ab20b`
- **Repository result:** Implemented and locally tested; real Colab measurement pending
- **No scientific result claimed:** No framework path was selected locally, no Drive
  manifest was rebuilt, no Cityscapes training sample was read, and no smoke model was
  trained.

## Colab execution

Use `notebooks/colab/05_semantic_five_model_smoke.ipynb` at the exact pushed commit.
It verifies a clean checkout, mounts the private `EdgeGuard/` root, selects the first
five-model-compatible OpenMIM-free runtime, rebuilds `split-policy-v1` from existing
manifests, stages an identity-bound Cityscapes bundle to `/content`, then runs the five
random-initialized 100-step smoke jobs. Batch 2 with accumulation 2 adapts once on CUDA
OOM to batch 1 with accumulation 4. At least four passes make the milestone ready for
common screening; the notebook does not start screening.

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

The runner records progress every 25 optimizer steps, validation loss/mIoU and all 19
class IoUs, config/data/split/framework identities, finite loss/gradient checks,
checkpoint hashes and exact resume. Active work is under `/content`; mounted Drive is
used for bounded recovery sync and verified final evidence, not sample-by-sample
training.

Do not access official Cityscapes val, full Fishyscapes Lost & Found or either SMIYC
set in this milestone. Do not infer training quality from compatibility smoke results.
