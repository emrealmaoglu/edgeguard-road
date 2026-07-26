# WP-03 / WP-05 Source and Evidence Review

## Scope

This is the single evidence review for the first PIDNet-S scientific vertical
slice. It records official sources, open questions, the proposed pinned upstream
commit, and the frozen dataset roles. It does not approve a checkpoint license,
download data or model artifacts, integrate upstream source, or close a
scientific gate.

All sources below were accessed on 2026-07-26. Status labels mean:

- **VERIFIED FACT:** directly supported by the linked official source.
- **ENGINEERING INFERENCE:** a proposed implementation consequence of verified
  facts; it still requires validation.
- **OPEN QUESTION:** evidence or human approval is still missing.

## Source and evidence table

| Topic | Finding | Status | Official source | Supported decision | Open risk |
| --- | --- | --- | --- | --- | --- |
| PIDNet paper and model family | PIDNet is a real-time semantic segmentation architecture published at CVPR 2023; the official repository presents PIDNet-S Cityscapes configurations and checkpoints. | VERIFIED FACT | [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html); [official PIDNet repository](https://github.com/XuJiacong/PIDNet) | Use PIDNet-S as the proposed first spike model, without treating upstream metrics as EdgeGuard measurements. | Runtime and checkpoint compatibility remain unverified in the target Colab environment. |
| PIDNet source license | The official PIDNet source repository carries the MIT License. | VERIFIED FACT | [PIDNet LICENSE at the pinned commit](https://raw.githubusercontent.com/XuJiacong/PIDNet/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/LICENSE) | Source vendoring may be evaluated only after a successful inference spike and human review. | The source license alone does not establish checkpoint usage rights. |
| PIDNet checkpoint | The official README publishes Cityscapes checkpoint links and notes replacement links for previously broken links. No explicit checkpoint-specific license was identified in the reviewed official material. | OPEN QUESTION | [official PIDNet repository](https://github.com/XuJiacong/PIDNet) | The human owner approved only the officially referenced PIDNet-S Cityscapes checkpoint for non-commercial academic thesis research, with filename, source URL, access date, and SHA-256 recording and no redistribution. | The checkpoint-specific license remains unresolved; actual bytes and hash are unverified until the human-run Colab step. |
| PIDNet prediction output | With `augment=False`, the official model returns one semantic tensor; the augmented training/evaluation path returns auxiliary semantic and boundary outputs as well. | VERIFIED FACT | [PIDNet model implementation at the pinned commit](https://raw.githubusercontent.com/XuJiacong/PIDNet/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/models/pidnet.py) | The isolated spike must use `augment=False` and inspect the actual returned object rather than guessing an output index. | A future upstream change or incompatible checkpoint may alter observed behavior. |
| PIDNet raw logits | The official segmentation head returns its final convolution result without applying softmax. | VERIFIED FACT | [PIDNet model utilities at the pinned commit](https://raw.githubusercontent.com/XuJiacong/PIDNet/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/models/model_utils.py) | Preserve the direct semantic head result as `native_logits`. | A real forward is still required to prove shape, dtype, finiteness, and class count for the pinned revision. |
| PIDNet alignment and preprocessing | The official custom inference converts BGR to RGB, divides by 255, applies ImageNet mean/std normalization, uses NCHW input, and bilinearly resizes semantic output to the input grid. | VERIFIED FACT | [PIDNet custom inference at the pinned commit](https://raw.githubusercontent.com/XuJiacong/PIDNet/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/tools/custom.py) | Test channel order and normalization; preserve the resized derivative separately as `aligned_logits` with recorded alignment settings. | The project must not mislabel aligned logits as the model's direct output. |
| PIDNet PyTorch compatibility | The official repository does not define a currently verified PyTorch compatibility range for this project environment. | OPEN QUESTION | [official PIDNet repository](https://github.com/XuJiacong/PIDNet) | Validate import, strict checkpoint load, and real forward in the isolated Colab spike before any vendoring decision. | Modern PyTorch behavior may require a bounded compatibility correction. |
| PIDNet ONNX support | No official, project-validated PIDNet ONNX export recipe was identified in the reviewed repository. | OPEN QUESTION | [official PIDNet repository](https://github.com/XuJiacong/PIDNet) | Treat ONNX as a later measurement pilot, not as a precondition for the PyTorch vertical slice. | Export may fail or require work outside the allowed pilot scope. |
| PyTorch ONNX exporter | The official PyTorch tutorial recommends the `dynamo=True` exporter path beginning with PyTorch 2.5. | VERIFIED FACT | [PyTorch ONNX tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) | Prefer the dynamo exporter only when the measured runtime supports it; record the actual exporter and opset. | No numerical acceptance thresholds are frozen before pilot measurements. |
| Cityscapes | Cityscapes requires registration and imposes non-commercial and redistribution restrictions; the fine validation split contains 500 images. | VERIFIED FACT | [Cityscapes license](https://www.cityscapes-dataset.com/license/); [official cityscapesScripts](https://github.com/mcordts/cityscapesScripts) | Limit this slice to a minimum external Cityscapes validation adapter after human acceptance of the terms. | Access and license acceptance are human decisions; data must remain outside Git. |
| Fishyscapes Lost & Found | Fishyscapes offers Lost & Found validation data for development. Its Zenodo annotation record is CC BY 4.0, while underlying image terms must be tracked separately. | VERIFIED FACT / OPEN QUESTION | [Fishyscapes dataset page](https://fishyscapes.com/dataset); [Fishyscapes Lost & Found annotations](https://zenodo.org/records/6511227) | Add only a development adapter after the raw-logit/scoring path works and both annotation and image terms receive human approval. | Annotation licensing does not by itself resolve the underlying image license. |
| SMIYC | RoadObstacle21 and RoadAnomaly21 are official Segment Me If You Can benchmark datasets. | VERIFIED FACT | [SMIYC datasets](https://segmentmeifyoucan.com/datasets); [official benchmark repository](https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark) | Freeze both datasets as `sealed_final`; create no loader, root, config, manifest, or test and access no files in this slice. | Any access would violate the sealed-test boundary and invalidate affected results. |
| SOS | The official SOS record describes 20 real driving sequences and 1,129 annotated frames and publishes the dataset under CC BY 4.0. | VERIFIED FACT | [SOS Zenodo record](https://zenodo.org/records/7144906); [ACCV 2022 paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Maag_Two_Video_Data_Sets_for_Tracking_and_Retrieval_of_Out_ACCV_2022_paper.pdf) | Record SOS only as a future temporal-unseen role. | SOS is not accessed or implemented in this vertical slice. |
| DDRNet fallback | The official DDRNet repository is MIT licensed and publishes a Cityscapes DDRNet-23-Slim checkpoint link. | VERIFIED FACT | [official DDRNet repository](https://github.com/ydhongHIT/DDRNet) | Consider DDRNet only after the bounded PIDNet stop criteria and a separate human decision. | DDRNet checkpoint rights and compatibility would need their own review; no parallel integration is allowed. |

## Proposed PIDNet upstream pin

- **Repository:** `https://github.com/XuJiacong/PIDNet.git`
- **Resolved ref:** `refs/heads/main`
- **Proposed commit:** `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Resolution evidence:** The full SHA was read directly from the official remote
  with `git ls-remote` on 2026-07-26. No checkout was performed.
- **Status:** **HUMAN APPROVED on 2026-07-26** for one fixed external checkout
  used by the isolated spike. This does not authorize vendoring.

The approved commit must remain immutable for the isolated spike. A later
upstream `main` value must not silently replace it.

## Dataset role matrix

**Status:** **HUMAN APPROVED on 2026-07-26.** The sealed SMIYC boundary is
unchanged and mandatory.

| Dataset | Frozen role | Required in this slice | Access and license state | Allowed use | Boundary |
| --- | --- | --- | --- | --- | --- |
| Human-approved single road image | `plumbing_only` | Yes, Stage 2 | A specific image and its usage rights have not yet been supplied or approved. | One-image preprocessing, inference, artifact, and visual-plumbing checks only. | Must not support a metric, accuracy, or anomaly-probability claim. |
| Cityscapes train | `id_train` | No | Terms recorded; no access is needed in this slice. | None in this slice. | No training, fine-tuning, or train manifest. |
| Cityscapes val | `id_validation` / `semantic_development` | Conditional, Stage 3 | Human registration, access, and terms acceptance pending. | Minimum validation adapter and semantic-development plumbing. | Not threshold selection or final testing; data stays outside Git. |
| Fishyscapes Lost & Found validation | `ood_development` | Conditional, Stage 4 | Annotation license recorded; underlying image terms and access require separate human approval. | Qualitative MSP/entropy plumbing after the scoring path works. | No threshold, AP, FPR95, or final benchmark claim. |
| SMIYC RoadObstacle21 | `sealed_final` | No | Deliberately not accessed. | None. | No file access and no loader, root, config, manifest, or test. |
| SMIYC RoadAnomaly21 | `sealed_final` | No | Deliberately not accessed. | None. | No file access and no loader, root, config, manifest, or test. |
| SOS | `temporal_unseen` | No | License evidence recorded; no access is requested. | Role record only. | No temporal implementation in this slice. |
| BDD100K / ACDC | `optional_unseen` | No | Not reviewed for this slice. | None. | Out of scope. |

## Human decision record and remaining execution gate

On 2026-07-26 the project owner:

1. Approved commit `4c158cf24ce432f0a8cb43364fae38d93cee0dc3` as the
   immutable official upstream source for the isolated spike.
2. Approved the dataset role matrix, including no access of any kind to the two
   `sealed_final` SMIYC datasets.
3. Approved the official repository-referenced Cityscapes PIDNet-S checkpoint
   only for non-commercial academic thesis research. The checkpoint must not be
   committed, redistributed, or included in the thesis delivery package. Its
   license remains **OPEN QUESTION**, and the MIT source license must not be
   presented as applying automatically to checkpoint weights.

Before a real forward, the official file must be identified as
`PIDNet_S_Cityscapes_val.pt`, its source URL and access date recorded, and its
downloaded SHA-256 reviewed and supplied to the runner. A usage-approved road
image and a Colab execution path are also required. No dataset, checkpoint,
upstream checkout, or model artifact was downloaded during the initial evidence
review; the later fixed-checkout verification is recorded below.

## Stage 2 upstream sample decision and verification

On 2026-07-26 the human owner approved only the sample images contained in the
fixed official PIDNet checkout for internal plumbing. The checkout was then
created under ignored `artifacts/external`, and the following source state was
measured before reading either image:

- **Upstream repository:** `https://github.com/XuJiacong/PIDNet.git`
- **Upstream commit / actual HEAD:**
  `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Checkout state:** clean

| Selection | Relative path | Filename | SHA-256 | Byte size | Original shape | Access date | Usage scope | Dataset role | Image license status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| Primary | `samples/frankfurt_000000_002196_leftImg8bit.png` | `frankfurt_000000_002196_leftImg8bit.png` | `78c65d3055fbd62e41d066813132c971a85dcdea4e5ef5459bad410bccead246` | 2,306,975 | `1024×2048×3` RGB | 2026-07-26 | `noncommercial_internal_plumbing` | `plumbing_only` | **OPEN QUESTION** |
| Fallback | `samples/frankfurt_000000_003025_leftImg8bit.png` | `frankfurt_000000_003025_leftImg8bit.png` | `acaf8970f2595b39678af6b38e9483384c2582b35750fe87c142cacd318b005e` | 2,522,812 | `1024×2048×3` RGB | 2026-07-26 | `noncommercial_internal_plumbing` | `plumbing_only` | **OPEN QUESTION** |

The source-code MIT license is not asserted to cover these images. Neither image
nor any derivative may be committed or redistributed. These files cannot support
a metric, Cityscapes validation claim, or scientific performance result. Actual
Cityscapes dataset access remains deferred to Stage 3.

## Stage 2 checkpoint access probe

The pinned README identifies the Cityscapes validation checkpoint filename as
`PIDNet_S_Cityscapes_val.pt` and links the PIDNet-S validation entry to the
configured Google Drive file reference. The same README warns that the individual
links no longer work and directs users to one replacement Drive folder.

Measured on 2026-07-26:

- The individual official file reference returned HTTP 404 to a non-interactive
  request.
- `gdown` 6.1.0 could not retrieve a public file URL from that same official
  reference.
- No alternative or generated direct URL was attempted.
- The official replacement folder landing page returned HTTP 200, but its file
  contents and checkpoint bytes were not machine-verifiable in this environment.
- No checkpoint file was created; byte size and SHA-256 therefore remain pending.

**Decision:** Use a human-controlled Colab upload from the official
repository-directed replacement folder. The notebook accepts only the exact
expected filename, records source references, access date, byte size, and SHA-256,
then stops until the hash is reviewed. Model loading cannot occur before that
review, and checkpoint license status remains **OPEN QUESTION**.
