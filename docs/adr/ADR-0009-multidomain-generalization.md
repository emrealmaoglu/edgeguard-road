# ADR-0009: Make multi-domain generalization the scientific core

## Status

Accepted by the project owner on 2026-07-28 and acquisition-amended on 2026-07-30.

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

## Consequences

- Dataset manifests, ontology mappings, source roles, and cross-domain duplicate
  evidence become part of every scientific run identity.
- Source-domain selection uses macro mIoU across domains, not pooled sample-weighted
  mIoU.
- Epoch budgets are replaced with optimizer-step budgets.
- IDD fallback/ambiguous classes are ignored, never relabeled as background.
- Sealed-server results cannot cause a model, threshold, calibration, or preprocessing
  change.
