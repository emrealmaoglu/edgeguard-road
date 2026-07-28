# Colab data access and storage runbook

Verified on 2026-07-28. The machine-readable authority is
`configs/dataset/colab_data_access_v1.yaml`; this page explains the human steps. Dataset
files, login data, cookies, temporary signed URLs, and license receipts never enter Git.

## Storage decision

- Google Drive is the durable 5 TiB store: official archives, prepared roots, one-file
  prepared bundles, manifests, and experiment snapshots.
- A Colab session is treated as a hard 200 GiB volume even when the runtime reports more.
  Staged datasets may use at most 175 GiB; 25 GiB remains reserved for environments,
  checkpoints, ONNX exports, and temporary files.
- Training reads from `/content/edgeguard-data`, never directly from mounted Drive.
- Each prepared dataset is packed once as an uncompressed deterministic tar. The training
  notebook copies one tar at a time, verifies SHA-256, extracts it safely, deletes the local
  tar, and then starts audit/training. This follows Colab's recommendation to avoid many
  small Drive reads.

Run `notebooks/EdgeGuard_Data_Preflight_Colab.ipynb` first. It creates this layout:

```text
MyDrive/EdgeGuard/
├── archives/<dataset_id>/       # untouched official downloads
├── datasets/<dataset_id>/       # manually extracted/prepared native trees
├── bundles/                     # *.prepared.tar + hash receipt
├── manifests/                   # access/inventory evidence
└── artifacts/                   # immutable experiment snapshots
```

## Phase-one source datasets

### Cityscapes Fine — required, manual account

Register, accept the official terms, and download only:

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Extract to `MyDrive/EdgeGuard/datasets/cityscapes/`. The required train and val paths
are checked before a bundle can be created. Official page:
<https://www.cityscapes-dataset.com/downloads/>.

### BDD100K semantic — required, official release

Download only the 10K image and semantic-segmentation packages. Do not acquire the 100K
image/video/detection corpus; it is irrelevant and wastes storage.

- `bdd100k_images_10k.zip` — published as 1.1 GB, MD5
  `08f26aecceda982568063d3d5873378e`
- `bdd100k_sem_seg_labels_trainval.zip` — published as 419 MB, MD5
  `9a2968dde3345eeb689cffb1e26f9c78`

The source of record is the BDD100K repository's download documentation:
<https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst>. If the official
ETH host is temporarily unreachable, wait or use the official browser flow; do not replace
it with Kaggle, Hugging Face, Google Drive, or another mirror. Extract so the root contains
`images/10k/{train,val}` and `labels/sem_seg/masks/{train,val}`.

### IDD20K — required controlled ablation, manual account

Register for the AutoNUE event, open Dataset → Download, acquire IDD20K Part I and Part II,
and extract both archives into the same `MyDrive/EdgeGuard/datasets/idd20k/` root. The
official instructions explicitly require the two-part merge:
<https://idd.insaan.iiit.ac.in/evaluation/autonue19/>. Preserve native source-ID masks;
the project maps only exact ontology matches and sends ambiguous classes to ignore `255`.

## Datasets deliberately deferred until model freeze

- **ACDC:** registered manual access. Only `rgb_anon_trainvaltest.zip` (15.6 GB) and
  `gt_trainval.zip` (127 MB) are needed. It is domain-shift evaluation, never training.
  <https://acdc.vision.ee.ethz.ch/download>
- **WildDash 2:** register with an academic email at least one week early. Manual approval
  and the official resumable downloader are expected. It remains the primary sealed server
  test and is not a development set. <https://www.wilddash.cc/submit>
- **MUSES:** after the sealed protocol gate, acquire only
  `frame_camera_trainvaltest.zip` (5.3 GB) and `gt_semantic_trainval.zip` (84 MB) from the
  public official package directory. Lidar, radar, event, panoptic, and detection packages
  are excluded. <https://muses.ethz.ch/MUSES_packages/>
- **KITTI semantic:** registered access is now required. Its 200 labeled training images
  are used only as a predeclared fallback when both WildDash and MUSES are unavailable.
  <https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics>

## Removed from the active Colab campaign

- The Hugging Face `segments/sidewalk-semantic` probe is removed because anonymous access
  returned 401 and no reproducible official acquisition route was verified.
- Every third-party dataset mirror is rejected because release identity, license,
  completeness, and checksum provenance cannot be defended.
- Mapillary Vistas and A2D2 remain future catalog candidates, not phase-one data. Mapillary
  lacks a frozen release-specific ontology mapping; A2D2 needs its own lossy RGB-mapping
  ablation. Neither may silently enter the current source pool.

## Execution order

1. Place official archives under `archives/<dataset_id>/` and keep their official names.
2. Run the preflight notebook with archive hashing enabled. Preserve the JSON inventory.
3. Extract/normalize native trees under `datasets/<dataset_id>/`; never rewrite masks.
4. Rerun inventory. Only a `prepared` state may enter bundling.
5. Set `CREATE_BUNDLES=True` once. Do not replace a bundle unless its source tree changed
   intentionally and the scientific manifests will be regenerated.
6. Open `EdgeGuard_Road_Colab.ipynb`. Its stage command refuses missing receipts, altered
   hashes, partial destinations, unsafe tar members, or storage plans over budget.
7. Leave `RUN_TRAINING=False` until all three audit reports and candidate splits are
   reviewed and frozen.
