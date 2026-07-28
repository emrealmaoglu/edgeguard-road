# ADR-0009: Make multi-domain generalization the scientific core

## Status

Accepted by the project owner on 2026-07-28 through the approved implementation plan.

## Decision

EdgeGuard-Road will study lightweight semantic segmentation under controlled
multi-domain training. Cityscapes, BDD100K, and IDD20K are source domains mapped to
Cityscapes19. ACDC is a frozen adverse-condition diagnostic. WildDash 2 and MUSES
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
