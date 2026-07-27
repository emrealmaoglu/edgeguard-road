# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `.`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `0114d4e4778c1d6e53b6359e0a11f71eb15d2fb4`
- **Task:** `EG-DATA-002 — Cityscapes Fine train preparation`
- **State:** Implemented and locally tested with synthetic fixtures; human diff
  review and real Colab preparation pending
- **Excluded:** No real archive read, dataset extraction, Drive mutation, split
  freeze, framework installation, training, Cityscapes test-label access, SMIYC,
  unrelated dataset access, Jetson action, stage, commit, or push

## Preparation architecture

The existing `scripts/prepare_cityscapes.py` now has a narrow `--split train` path.
It reuses the existing pinned archive constants and ZIP safety validation and the
shared Cityscapes label mapping; it does not introduce a generalized extraction or
dataset framework.

The real flow is:

```text
immutable private_inputs archives
→ filename/size/SHA and complete ZIP safety validation
→ train-only RGB/native-label extraction into Colab-local staging
→ deterministic 0..18/255 train-ID PNG generation
→ streaming geometry/source-ID/class/city/group analysis
→ CSF-SPLIT-A/B/C group-atomic candidates
→ temporary-result validation
→ incoming copy and second exact validation
→ canonical dataset/manifests promotion
→ small independently verifiable evidence ZIP
```

Original `labelIds` remain external and unchanged. Every generated train-ID mask is
byte-deterministically encoded and hashed. Original image/label identity is anchored
to the verified immutable archive hashes and root-free paths. The real run records
generated-mask hashing duration and bytes so broader hashing can be judged from
measurement.

Unknown source label IDs hard-fail; they are not converted silently to ignore. The
class-mapping receipt includes every reviewed source ID, its Cityscapes name, mapped
project train ID or ignore action, and observed pixel count. Image/label/train-ID
geometry is checked while streaming; full-resolution masks are not retained in
memory as a dataset-wide collection.

## Split boundary

Each candidate keeps `city+sequence` groups atomic and records sample/city/group
counts, class pixels and presence, rare-class coverage, sample deviation,
class-coverage penalty, pixel-distribution divergence, rare-class absence, and
leakage validation. Candidate targets are near `85/10/5`, `80/15/5`, and `90/5/5`,
but no ratio or candidate is approved. The compact comparison status is always
`recommended_pending_human_approval`.

Official Cityscapes val remains `official_val_common_eval`: not routine HPO,
`train_select`, temperature fitting, sealed, or unseen. Cityscapes test labels and
SMIYC are inaccessible to this workflow. No semantic training may begin before the
human accepts a measured candidate.

## Command and external outputs

The notebook calls the repository entry point with runtime-only paths:

```text
python scripts/prepare_cityscapes.py
  --split train
  --left-images-archive <external-root>/private_inputs/leftImg8bit_trainvaltest.zip
  --labels-archive <external-root>/private_inputs/gtFine_trainvaltest.zip
  --destination <external-root>/datasets/cityscapes/fine/v1
  --manifests-destination <external-root>/manifests/cityscapes/fine/v1
  --work-directory /content/edgeguard-work/cityscapes-fine-v1
  --preparation-git-commit <reviewed-commit-sha>
  --ontology-config configs/dataset/ontology_v1.yaml
```

`--verify-only` idempotently verifies an existing exact prepared dataset. Normal mode
refuses any existing dataset, manifest, staging, or incoming destination.

Full manifests, prepared data, masks, and evidence remain outside Git. The evidence
ZIP contains only archive/preparation receipts, ontology identity, aggregate class
and group summaries, split comparison/report, environment, manifest identities, and
failures; it contains no image or mask.

## Local result and validation

- Private Drive root at the approved Colab mount: unavailable in this local runtime
- Real archives processed: no
- Real archive hash status: pending Colab verification
- Real sample/group/city counts: unavailable
- Real class-frequency and split summaries: unavailable
- Evidence package: unavailable until real preparation
- Ontology: provisional; SHA-256
  `24fd17b54a4aa461e004eaf8c5feebe7b3115c0906559ff24f3bd7f2e1510a10`
- Ruff check: passed
- Ruff format check: 103 files passed
- mypy: 29 source files passed
- pytest: 201 passed, 2 expected opt-in skips
- `git diff --check`: passed

## Human gates and exact next action

1. Review this complete unstaged EG-DATA-002 diff.
2. Authorize a coherent commit and push; do not run the notebook from an unstaged or
   dirty checkout.
3. Run `notebooks/colab/03_prepare_cityscapes_fine_train.ipynb` at that exact commit.
4. Verify archive identities, counts, class/group summaries, manifest/evidence hashes,
   and absence of failures.
5. Inspect all three candidates and explicitly select, revise, or reject one.

`EG-SEG-001` remains blocked. Current changes are unstaged; publication is not
authorized in this task.
