# Agent Handoff

- **Task:** Local OOD uncertainty scoring and semantic calibration foundation
- **Branch/base:** `feat/first-vertical-slice` after operational CI commit
  `ee2a292e00d0d14711163ba037b4a35dacd1f0d1`
- **Repository result:** Bounded CPU-only implementation and local validation complete;
  scientific commit and remote verification pending
- **No scientific result claimed:** Only synthetic, random-weight compatibility probes
  ran. No Drive data, real Cityscapes sample, pretrained weight, training campaign or
  accelerator performance measurement was used.

## Colab execution

The operational workflow activation passed normal Python 3.10/3.11 CI and Linux x86
Python 3.12 five-model CPU run `30275255124`. Do not open or use Colab in this task.
The future compatibility-only notebook remains separate from the Drive-backed smoke
notebook, and the latter still requires a successful same-runtime receipt.

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

The new OOD/calibration command uses six deterministic synthetic fixture masks and
produces only four small JSON records plus a deterministic ZIP. Temperature fitting is
bounded in log space, preserves input logits, fails on an empty valid region, and
records before/after NLL. Calibration outputs describe semantic class confidence;
MSP, entropy, MaxLogit and Energy remain uncertainty/OOD scores, not anomaly
probabilities.

Do not access official Cityscapes val, full Fishyscapes Lost & Found or either SMIYC
set in this milestone. Do not infer training quality from compatibility smoke results.
