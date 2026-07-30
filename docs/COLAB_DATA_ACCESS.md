# Colab data access and storage runbook

Verified against the connected Drive on 2026-07-30. The machine-readable authority is
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
├── archives/                    # untouched official downloads
│   ├── cityscapes/
│   ├── bdd100k/
│   ├── idd20k/
│   ├── acdc/
│   ├── wilddash2/
│   ├── muses/
│   └── kitti/
├── quarantine/kaggle/bdd100k/  # smoke-only BDD mirror; never final science
├── private_inputs/              # current immutable uploads; no migration required
├── bundles/                     # *.prepared.tar + hash receipt
├── manifests/                   # acquisition and inventory evidence
├── campaigns/<id>/<commit>/     # verified resumable campaign.latest.tar.gz
├── downloads/                   # small review ZIPs for local inspection
├── source/                      # optional immutable source-code handoff
└── datasets/<dataset_id>/       # legacy prepared roots; not the default path
```

The live read-only Drive audit found an older, valid project structure rather than an
empty target. It is preserved in place. In particular, all current archives remain under
`private_inputs/`, while Cityscapes prepared data and its 6.99 GB verified bundle remain under
`datasets/cityscapes/fine/`, and `campaigns/EG-REAL-001` remains historical evidence.
The notebook detects and reuses the exact hash-pinned Cityscapes bundle; it creates only
missing BDD/IDD and review directories. See `DRIVE_LAYOUT_AUDIT_2026-07-30.md`.

The campaign snapshot includes only named experiment-state roots. Staged datasets are
never recompressed into it. The review ZIP additionally excludes datasets, canonical
masks, checkpoints, ONNX and TensorRT payloads; it contains human-readable JSON/CSV/logs
and thesis PNG/PDF figures. Set `DOWNLOAD_REVIEW_PACKAGE=True` in the training notebook
to download that small ZIP through the browser. Full state remains in Drive.

## Phase-one source datasets

### Cityscapes Fine — required, manual account

Register, accept the official terms, and download only:

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Do not extract it manually. The preflight notebook prepares the required train/val tree
in ephemeral Colab storage and writes a verified bundle. Official page:
<https://www.cityscapes-dataset.com/downloads/>.

### BDD100K semantic — provisional now; official packages required for scientific use

Download only the 10K image and semantic-segmentation packages. Do not acquire the 100K
image/video/detection corpus; it is irrelevant and wastes storage.

- `bdd100k_images_10k.zip` — published as 1.1 GB, MD5
  `08f26aecceda982568063d3d5873378e`
- `bdd100k_sem_seg_labels_trainval.zip` — published as 419 MB, MD5
  `9a2968dde3345eeb689cffb1e26f9c78`

The source of record is the BDD100K repository's download documentation:
<https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst>. If the official
ETH host is temporarily unreachable, wait or use the official browser flow; do not replace
it with Kaggle, Hugging Face, Google Drive, or another mirror for final science. The
uploaded `private_inputs/bdd100k.zip` is retained for audit/smoke only and is marked
scientifically ineligible by code.

Kaggle and official BDD can never share a bundle filename. The quarantined output is
`bdd100k.kaggle_mirror.prepared.tar`; the official output is
`bdd100k.prepared.tar`. Scientific staging refuses the mirror bundle. An explicit
`ALLOW_INELIGIBLE_BDD_SMOKE=True` can stage it for plumbing and provisional audit, and
its preparation receipt still prevents a scientific manifest. Main HPO/training uses
Cityscapes + IDD20K. Official BDD may return later as a separately frozen ablation.

### IDD20K — required controlled ablation, manual account

Register for the AutoNUE event, open Dataset → Download, acquire IDD20K Part I and Part II,
and leave both archives untouched. The notebook merges both into one ephemeral prepared
root only after path-collision checks. The official instructions require the two-part merge:
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

1. Leave current uploads under `private_inputs/`; future official archives may use
   `archives/<dataset_id>/`. Do not move or extract source archives in Drive.
2. Run the preflight notebook with archive hashing enabled. Preserve the JSON inventory.
3. Set `RUN_ARCHIVE_PREPARATION=True`; keep `CREATE_BUNDLES=True`. The notebook reuses
   Cityscapes, prepares BDD then IDD one at a time, enforces a conservative 3× archive
   working-space estimate plus 25 GiB reserve, bundles directly, and removes temporary
   trees. Repeat runs reuse verified bundles.
4. Do not replace a bundle unless its archives/source profile changed intentionally and
   all scientific manifests will be regenerated.
5. Open `EdgeGuard_Road_Colab.ipynb`. Its stage command refuses missing receipts, altered
   hashes, partial destinations, unsafe tar members, or storage plans over budget.
6. Leave `RUN_TRAINING=False` until Cityscapes and IDD scientific candidate manifests
   are reviewed and frozen. Review the BDD audit separately; it cannot be frozen for science.
7. After each stage, inspect `downloads/*-review.zip`. Class distribution, pooled
   imbalance/weights, split sizes and deterministic source examples are generated only
   from measured `train_fit` manifests; fixture plots are never thesis evidence.

## Colab runtime rule

The current hosted Colab stack is not assumed. Colab announced Python 3.12 and Torch 2.8
for the hosted image, while the pinned MMSeg stack predates that image. The notebook first
attempts hosted Torch unchanged and then falls back to isolated Python 3.11, Torch 2.1.1
and CUDA 12.1. Both paths run dependency checks, delivery-package imports, five-model
forward/backward and checkpoint reload. Training no longer uses a hard-coded interpreter:
it resolves the selected Python and MMSeg checkout from the verified compatibility
receipt and rejects project/framework commit drift.

References: <https://github.com/googlecolab/colabtools/issues/5483>,
<https://mmsegmentation.readthedocs.io/en/main/notes/faq.html>, and
<https://mmcv.readthedocs.io/en/2.x/get_started/installation.html>.

Any unhandled notebook/subprocess error creates a redacted JSON and downloadable ZIP
under `EdgeGuard/failures/`. Set `DOWNLOAD_LATEST_FAILURE_REPORT=True` and rerun the last
notebook cell to download it. The exact contents and recovery procedure are defined in
`docs/COLAB_FAILURE_REPORTING.md`.
