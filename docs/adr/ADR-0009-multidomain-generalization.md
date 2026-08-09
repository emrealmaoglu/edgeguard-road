# ADR-0009: Make multi-domain generalization the scientific core

## Status

Accepted by the project owner on 2026-07-28, acquisition-amended on 2026-07-30, and
role-amended on 2026-08-09.

## Decision

EdgeGuard-Road will study lightweight semantic segmentation under controlled
multi-domain training. The active scientific sources are Cityscapes and IDD20K mapped
to Cityscapes19. The available Kaggle BDD100K mirror is provisional audit/smoke data and
cannot enter scientific manifests; official BDD packages may later support a separate
ablation. ACDC is a frozen adverse-condition diagnostic. WildDash 2 and MUSES
remain sealed external evaluations; KITTI is the documented access fallback.

Five existing MMSeg models use common random initialization, input geometry,
optimizer family, augmentation, effective batch, evaluator, and optimizer-step
budgets. Dataset size does not determine exposure: multi-source training uses a
domain-uniform sampler. Only the top two screening models enter the fixed 12-trial
TPE/successive-halving HPO budget.

Detection and temporal fusion remain outside the thesis-critical path. External
predictions require a human-approved release bound to the frozen model and manifest.

**2026-08-09 role amendment.** Four newly-acquired auxiliary sources are assigned roles,
under an explicit, narrow, in-session delegation from the project owner covering only
these four sources (the `train_fit`/`train_select`/official-validation roles for
Cityscapes/IDD20K remain reserved and unchanged). WildDash 2's `wd_both_02.zip` (812
images, no ground truth) activates the `primary_sealed_external` role already named above.
WildDash 2's `wd_public_v2p0.zip` (4256 images, with panoptic ground truth) is
`engineering_packages`-only, `scientific_eligible: false`: it is a diagnostic/visual
resource, never a metric source, since decoding this GT would silently break the sealed
claim on the actual submission. WildDash's pickup/van/autorickshaw ontology-extension
script (`wd_add_pickupvan.zip`) is excluded — applying it would be an ontology-version
change mid-campaign, invalidating every already-computed manifest/split hash; retained as
a future v2-ontology candidate only. RailSem19 (`rs19_val.zip`) is excluded from
training/eval scope as off-domain (rail-scene imagery, weakly-supervised road labels);
noted only as a possible future zero-cost OOD-diagnostic stress input to the existing
uncertainty/OOD stack, not scheduled.

## Consequences

- Dataset manifests, ontology mappings, source roles, and cross-domain duplicate
  evidence become part of every scientific run identity.
- Source-domain selection uses macro mIoU across domains, not pooled sample-weighted
  mIoU.
- Epoch budgets are replaced with optimizer-step budgets.
- IDD fallback/ambiguous classes are ignored, never relabeled as background.
- Sealed-server results cannot cause a model, threshold, calibration, or preprocessing
  change.
