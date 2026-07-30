# Semantic-First Execution Runbook

This is the canonical execution order. Replace example paths; never put licensed data,
credentials, checkpoints, generated models, or result ledgers in Git.

## 1. Prepare Drive data

Upload untouched archives under `MyDrive/EdgeGuard/archives/<dataset>/`. Run
`notebooks/EdgeGuard_Data_Preflight_Colab.ipynb`, inspect inventory, then set
`RUN_ARCHIVE_PREPARATION=True`. It copies and prepares one source at a time under
`/content`, writes `bundles/<dataset>.prepared.tar` plus a receipt, and cleans temporary
data. Do not extract archives manually on the Mac or into Drive.

Required final-science packages:

- Cityscapes: `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`.
- BDD100K: `bdd100k_images_10k.zip`, `bdd100k_sem_seg_labels_trainval.zip`.
- IDD20K: official `idd-20k-I.tar.gz` and `idd-20k-II.tar.gz`.

The existing Kaggle BDD zip may use `BDD_SOURCE_PROFILE="kaggle_mirror"` for smoke only.
Its receipt and every downstream manifest remain scientifically ineligible.

Keep Drive outputs separated: `campaigns/` is the resumable full experiment state,
`downloads/` is the bounded human-review ZIP, and `manifests/` is acquisition evidence.
The notebook never snapshots `/content/edgeguard-data`.

## 2. Audit and freeze source manifests

After the training notebook stages verified bundles into `/content/edgeguard-data`:

```bash
python scripts/audit_dataset.py --dataset cityscapes --dataset-root /content/edgeguard-data/cityscapes --output-root /content/work/audit/cityscapes
python scripts/audit_dataset.py --dataset bdd100k --dataset-root /content/edgeguard-data/bdd100k --output-root /content/work/audit/bdd100k
python scripts/audit_dataset.py --dataset idd20k --dataset-root /content/edgeguard-data/idd20k --output-root /content/work/audit/idd20k
```

Review corrupt/geometry/unknown-label/ignore/class/group/duplicate evidence. Freeze only
group-atomic candidates with no cross-role leakage. Generate rare-five classes and
median-frequency weights from the three `train_fit` roles only. Official validation,
ACDC, and sealed datasets remain inaccessible to training, selection, calibration, HPO,
and threshold fitting.

The statistics command also writes thesis-ready 300-DPI PNG/PDF figures and their CSV:
per-domain class distribution, pooled imbalance/CE weights, frozen split sizes, and
deterministically selected source examples. Every file is hash-listed in
`thesis_figures.json`; use only real frozen-manifest output in the thesis.

## 3. Model campaign

Use repeated `--data-manifest` arguments for the two frozen scientific sources; never
concatenate native folders. Run one-batch validation, then stages `smoke`, `pilot`, and `screening`
for each model in `configs/rescue/semantic_first.yaml`. Source batches are domain-uniform.
Record every failure and allow at most two substantial integration repairs/model.

After screening, export checkpoints early. Freeze top two from measured source-select
results and run:

```bash
python scripts/train.py --stage hpo --candidate-table /content/work/reports/screening/candidate_table.json --data-manifest /content/work/manifests/cityscapes.frozen.json --data-manifest /content/work/manifests/idd20k.frozen.json --rare-classes-file /content/work/statistics/rare_classes.json --output-root /content/work/runs --mmseg-root /content/mmsegmentation
```

Then execute Cityscapes versus Cityscapes+IDD source-composition ablations,
CE/weighted-CE comparison, and at
most 40,000-step finals. Random initialization is the primary table; pretrained models
remain a separate reference table.

## 4. Reliability and frozen evaluation

Fit one global temperature using equal valid-pixel contributions from the two scientific
source calibration roles. Report ECE/NLL/Brier before and after. Create source frame summaries
and freeze the shift reference:

```bash
python scripts/evaluate.py calibrate-shift --help
python scripts/evaluate.py evaluate-shift --help
```

Only after checkpoint, preprocessing, ontology, temperature, and shift thresholds are
frozen may official source validation and ACDC run. Report mIoU, 19 class IoUs,
rare-class mIoU, confidence/entropy/logit/energy, reliability, frame-shift AUROC/AP and
alert rate. Open MUSES/WildDash only after the sealed release record; do not iterate on
its result. Use KITTI only if both preferred external routes are unavailable.

## 5. Prediction and demo

```bash
python scripts/predict.py --help
python scripts/predict.py --emit-regions --emit-risk --shift-reference /artifacts/shift-reference.json ...
streamlit run app.py
```

The JSON output explicitly labels regions as semantic connected components and attention
as a heuristic operational score. Validate missing-checkpoint messaging and CPU fallback.

## 6. ONNX and target-only TensorRT

Export static batch-one `1×3×512×1024` raw logits and require PyTorch/ONNX shape,
19-class, finiteness, and numerical agreement before target transfer.

On the Jetson, review before execution:

```bash
python scripts/jetson/build_tensorrt.py --onnx model.onnx --engine model.fp16.engine --manifest model.fp16.build.json
python scripts/jetson/build_tensorrt.py --onnx model.onnx --engine model.fp16.engine --manifest model.fp16.build.json --execute
```

The first command is dry-run. The script does not change `nvpmodel`. Start `tegrastats`
separately, select and record the intended 25W or MAXN SUPER mode manually, then run:

```bash
python scripts/jetson/benchmark.py --engine model.fp16.engine --engine-manifest model.fp16.build.json --image-root /data/benchmark-images --telemetry-log /artifacts/tegrastats.log --output /artifacts/jetson-25w.json --power-profile 25W
```

Repeat for MAXN SUPER only as a secondary thermal/performance comparison. Preserve
engine/build/benchmark hashes. UI timing is excluded from the hardware acceptance gate.

## 7. Conditional detector decision

Do not start detection until every gate in `PROJECT_CHARTER.md` is evidenced. If all
pass, add only RTMDet-Tiny with official BDD detection labels and a separate metrics table.

```bash
python scripts/check_detection_gate.py --help
```
