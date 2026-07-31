# Multi-source semantic merge protocol

## Decision

EdgeGuard does not physically merge source trees and does not let the largest dataset
define the training distribution. Every native dataset remains immutable. A prepared
record points to the native RGB image, a separately generated Cityscapes19 mask, the
source dataset, sequence/group, condition, source identity, mapping hash and split hash.

The primary scientific comparison remains:

1. Cityscapes;
2. Cityscapes + IDD20K.

The available BDD100K Kaggle mirror is prepared and audited only as provenance-limited
engineering evidence. It cannot enter HPO, calibration, model selection or the primary
comparison. Official BDD packages may later add a separately frozen third-source ablation.

A2D2 and Mapillary are phase-two candidates, not silently added to this comparison.
Their value is geographic and capture diversity; their cost is lossy ontology mapping.
Adding them before the two primary rows are measured would confound the thesis result.

## Label harmonization

The target is `Cityscapes19 + ignore=255`. The conversion rules are:

- exact semantic equivalent: map to the canonical class;
- source subclass whose pixels belong unambiguously to one canonical surface: map;
- ambiguous superclass, source-only object, privacy/sensor artifact or Cityscapes-ignored
  class: map to `255`;
- unseen native ID/RGB color: fail the audit; never infer background;
- native label files are retained and hashed; generated masks never overwrite them;
- categorical masks use nearest-neighbor interpolation only.

This creates partial-label training for IDD and future A2D2/Mapillary candidates. The
loss is computed only where the mapped mask is not `255`. Per-dataset usable-pixel ratio,
per-class presence and ignored-pixel ratio must be reported so that a dataset with little
canonical supervision cannot masquerade as a full source domain.

## Sampling and loss

The primary multi-domain sampler is domain-uniform (`alpha=0`): each source has equal
expected optimizer-step mass regardless of image count. Physical `ConcatDataset`
proportional sampling is rejected because IDD would dominate Cityscapes.

Dataset-size power sampling is reserved for a post-baseline data ablation:

`p(domain=d) = n_d^alpha / sum(n_j^alpha)`, for `alpha in {0, 0.5, 1}`.

- `alpha=0`: equal domains, primary protocol;
- `alpha=0.5`: compromise candidate;
- `alpha=1`: proportional sampling control.

This is not an HPO variable. It is tested only after one architecture and all source
splits are frozen. Class weighting is also separate from HPO: standard CE versus
train-fit-only median-frequency weighted CE, clipped to `[0.5, 5.0]` and mean-normalized.

## Leakage and fairness gates

- Sequence/video groups are atomic across fit/select/calibration.
- Exact SHA-256 and perceptual hashes are compared across all source and external sets.
- Official source validation is opened only for frozen models.
- ACDC is evaluation-only; WildDash 2 and MUSES remain sealed external tests.
- Normalization, augmentation, crop, optimizer, effective batch, step budget and seed
  policy are common across model comparisons.
- Rare classes are frozen from pooled `train_fit` pixels before model comparison.
- Domain-macro mIoU is primary; pooled pixel/sample weighting is supplementary only.

## HPO decision

The 12-trial top-two Optuna search remains intentionally narrow: learning rate, weight
decay, scheduler and warm-up ratio. Dataset composition, domain-sampling alpha,
augmentation, loss, initialization, resolution and external results are excluded. This
prevents a small GPU budget from turning scientific ablations into hidden tuning.

After the primary campaign, the only justified expansion order is:

1. A2D2 controlled source ablation after its RGB mapping and sequence split are frozen;
2. Mapillary Vistas after an exact release/version-specific ontology review;
3. synthetic GTA5/SYNTHIA pretraining only as a separately labeled initialization study;
4. off-road or anomaly datasets only under a different research question.

CamVid is too small and leakage-prone for a new core source; ApolloScape is valuable for
geographic diversity but requires another large ontology/license integration; Dark
Zurich/Foggy Zurich are better adverse-domain diagnostics than source domains. None is
allowed to weaken the sealed external boundary.
