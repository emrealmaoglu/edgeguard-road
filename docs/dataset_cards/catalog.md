# EdgeGuard semantic dataset catalog

`catalog.json` is authoritative. Counts describe official package contracts, not locally acquired evidence. `verified_local` remains false until a hash-bound receipt exists.

| Dataset | Portfolio role | Official count | Native labels | Canonical merge | Access |
| --- | --- | ---: | --- | --- | --- |
| Cityscapes Fine | `core_source_domain` | 5,000 | 34 semantic labels; 19 train/evaluation classes | `direct_canonical` | `registered_manual` |
| BDD100K semantic segmentation | `core_source_domain` | 10,000 | Cityscapes-compatible 19-class semantic masks | `direct_canonical` | `account_manual` |
| IDD20K | `controlled_lossy_source_ablation` | 20,101 | hierarchical ontology; 26 level-3 evaluation classes | `partial_exact_mapping` | `account_manual` |
| ACDC | `public_adverse_domain_shift` | 4,006 | 19 Cityscapes semantic classes plus invalid masks | `evaluation_only_direct` | `registered_manual` |
| WildDash 2 | `primary_sealed_external_test` | 5,032 | Cityscapes labels plus six WildDash-specific classes and negative/void policy | `sealed_server_only` | `public_server_submission` |
| MUSES | `secondary_sealed_external_test` | 2,500 | 19 Cityscapes evaluation classes with panoptic and difficulty annotations | `sealed_evaluation_only` | `public_direct` |
| KITTI semantic segmentation | `external_access_fallback` | 400 | Cityscapes-compatible semantic format | `fallback_evaluation_only` | `registered_manual` |
| Mapillary Vistas | `phase_two_broad_domain_candidate` | 25,000 | version-sensitive taxonomy; V1 paper reports 66 classes | `blocked_pending_versioned_mapping_review` | `account_manual` |
| A2D2 semantic segmentation | `phase_two_geographic_source_candidate` | 41,277 | 55 official RGB colors representing 38 semantic concepts/variants | `phase2_partial_mapping_proposal` | `public_direct` |

## Non-negotiable merge rules

- Native masks are preserved; generated Cityscapes19 masks are separate artifacts.
- Unknown or semantically ambiguous labels become `255`, never background.
- Source domains are sampled uniformly in the primary experiment; physical file concatenation is not the sampling policy.
- Official validation, adverse-domain, and sealed external records never enter training, calibration, preprocessing fitting, HPO, or debugging.
- Dataset version, license receipt, source hash, mapping hash, split hash, exact hash, and perceptual-hash evidence are required before scientific use.
