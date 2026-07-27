# EdgeGuard-Road Dataset Cards

The machine-readable source of truth is `catalog.json`. Counts are never inferred
from dataset names. `null` means that the required real artifact is unavailable in
this checkout and the value remains `unavailable_pending_access`.

| Dataset | Role | Access | Count evidence | Protected boundary |
| --- | --- | --- | --- | --- |
| Cityscapes Fine train | fit/select/calibration | External, not locally present | Human-provided measured evidence: 2,975 samples, 18 cities, 1,885 city+sequence groups, 19 train classes | Exact D/E selected split identity and class-frequency artifact still unavailable locally |
| Cityscapes official val | common final-model evaluation | Historical external evaluation; data absent locally | 500 images/annotations, 19 classes | Not HPO, `train_select`, temperature fitting, sealed, or unseen |
| Cityscapes Coarse/trainextra | aggressive coarse-to-fine candidate | unavailable pending access | Not recorded | No acquisition without a justified experiment |
| BDD100K detection | detection train/development | unavailable pending access | Not recorded | Real package/source IDs must be inspected before mapping freeze |
| Fishyscapes Static | OOD development/HPO | unavailable pending access | Not recorded | Development only |
| Fishyscapes Lost & Found | one-time frozen holdout | unavailable pending access | Not recorded | No routine tuning, HPO, threshold selection, or debugging |
| SMIYC RoadObstacle21 | sealed final | deliberately inaccessible | Not recorded | No development-time access |
| SMIYC RoadAnomaly21 | sealed final | deliberately inaccessible | Not recorded | No development-time access |
| Temporal dataset | role and dataset pending human selection | unavailable pending access | Not recorded | Synthetic fixtures validate only plumbing |
| Approved demo assets | prerecorded demonstration | unavailable pending asset approval | Not recorded | No scientific metric claim |

## Existing Cityscapes evidence boundary

The repository records the human-provided measured Fine-train totals above and a
historical 500-image PIDNet-S official-val baseline. The real Fine-train dataset
manifest, its per-class pixel frequencies, image-level presence frequencies,
ignored-pixel ratio, and the cryptographically selected D/E split are not available
in this local checkout. They are therefore not fabricated or marked verified here.
The historical official-val artifact identity remains unchanged.
