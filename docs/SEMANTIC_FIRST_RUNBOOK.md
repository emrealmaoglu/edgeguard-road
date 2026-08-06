# Semantic-First Execution Runbook

This is the canonical execution order. Replace example paths; never put licensed data,
credentials, checkpoints, generated models, or result ledgers in Git.

## 1. Prepare Drive data

Upload untouched archives under `MyDrive/EdgeGuard/archives/<dataset>/`. Run
`notebooks/EdgeGuard_Data_Preflight_Colab.ipynb` with **Run all**. The default prepares
only missing scientific sources. Cityscapes is one verified tar; IDD is published as
500-sample canonical shards so a reset can lose only the open shard. `DEEP_VERIFY_ARCHIVES`
stays false after the first pinned digest receipt. Do not extract archives on the Mac or Drive.

Required final-science packages:

- Cityscapes: `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`.
- IDD20K: official `idd-20k-I.tar.gz` and `idd-20k-II.tar.gz`.

The existing Kaggle BDD zip may use `BDD_SOURCE_PROFILE="kaggle_mirror"` for smoke only.
Its receipt and every downstream manifest remain scientifically ineligible.

BDD is provisional and excluded from the active campaign. Drive separates immutable
recovery objects, small campaign state, HPO SQLite backups, prepared shards, runtime cache,
failure packages and `review_packages/`. Staged datasets are never snapshotted.

## 2. Audit and freeze source manifests

After the training notebook stages verified bundles into `/content/edgeguard-data`:

```bash
python scripts/audit_dataset.py --dataset cityscapes --dataset-root /content/edgeguard-data/cityscapes --output-root /content/work/audit/cityscapes
python scripts/audit_dataset.py --dataset idd20k --dataset-root /content/edgeguard-data/idd20k --output-root /content/work/audit/idd20k --checkpoint-root /content/drive/MyDrive/EdgeGuard/campaigns/semantic-cs-idd-v2/state/audit-catalog
```

Review corrupt/geometry/unknown-label/ignore/class/group/duplicate evidence. Freeze only
group-atomic candidates with no cross-role leakage. `--freeze-approved` also requires a
human-authored `--review-receipt`, `--campaign-id`, and full `--project-commit`; choosing a
training target is never approval. The receipt must bind the candidate file SHA-256,
dataset, campaign, commit, reviewer and `freeze_approved` decision. Generate rare-five
classes and median-frequency weights from the two scientific `train_fit` roles only. Official validation,
ACDC, and sealed datasets remain inaccessible to training, selection, calibration, HPO,
and threshold fitting.

The statistics command also writes thesis-ready 300-DPI PNG/PDF figures and their CSV:
per-domain class distribution, pooled imbalance/CE weights, frozen split sizes, and
deterministically selected source examples. Every file is hash-listed in
`thesis_figures.json`; use only real frozen-manifest output in the thesis.

## 3. Model campaign

Use repeated `--data-manifest` arguments for the two frozen scientific sources; never
concatenate native folders. The hermetic canary and first smoke/pilot run only SegFormer-B0,
Fast-SCNN and PIDNet-S. DDRNet-23-Slim and BiSeNetV2 enter extension smoke only after the
core pilot, then all five enter screening. Source batches are domain-uniform.

Use `CAMPAIGN_TARGET` and Run all in this order: `audit`, `smoke`, `pilot`, `screening`,
`hpo`, `final`. Every later target runs only missing prerequisites. HPO rungs and training
checkpoints survive resets; final training uses frozen HPO parameters when available. The
notebook delegates training state to one command. The equivalent screening target is:

```bash
python scripts/colab_pipeline.py run --target screening --project-root /content/edgeguard-road --project-commit "$EDGEGUARD_COMMIT" --runtime-receipt /content/edgeguard-evidence/runtime_receipt.json --mmseg-root /content/edgeguard-checkouts/mmsegmentation --work-root /content/edgeguard-work --recovery-root /content/drive/MyDrive/EdgeGuard/campaigns/semantic-cs-idd-v2/recovery --config /content/edgeguard-road/configs/rescue/semantic_first.yaml --data-manifest /content/edgeguard-work/manifests/cityscapes.frozen.json --data-manifest /content/edgeguard-work/manifests/idd20k.frozen.json
```

HPO requires the reviewed screening candidate table. Final requires three explicitly frozen
models; its first listed finalist receives the separate CE/weighted-CE final ablation.
Random initialization is the primary table; pretrained models remain a separate reference.

## 4. Reliability and frozen evaluation

Fit one global temperature using equal valid-pixel contributions from the two scientific
source calibration roles. Report ECE/NLL/Brier before and after. Create source frame summaries
and freeze the shift reference:

```bash
python scripts/evaluate.py calibrate-shift --help
python scripts/evaluate.py evaluate-shift --help
```

First complete `CAMPAIGN_TARGET="final"`. Only in a later run set
`CAMPAIGN_TARGET="evaluate"` and `ALLOW_FINAL_DATA=True`. Final writes
`accepted_release.candidate.json`; it does not accept itself. After human review, promote
that exact candidate and only then run the later targets:

```bash
python scripts/accept_colab_release.py --candidate /content/edgeguard-work/accepted_release.candidate.json --review-receipt /content/edgeguard-work/reviews/release.review.json --output /content/edgeguard-work/accepted_release.json
```

The public continuation is `evaluate`, then `export`, then `report`; each requires the
same accepted-release hash. Optionally set `RUN_ACDC=True` after preparing its validation
bundle. Report mIoU, 19 class IoUs,
rare-class mIoU, confidence/entropy/logit/energy, reliability, frame-shift AUROC/AP and
alert rate. Open MUSES/WildDash only after the sealed release record; do not iterate on
its result. Use KITTI only if both preferred external routes are unavailable.

## 5. Prediction and demo

```bash
python scripts/predict.py --help
python scripts/predict.py --emit-regions --emit-risk --shift-reference /artifacts/shift-reference.json ...
EDGEGUARD_ACCEPTED_BUNDLE_ROOT=/accepted/release streamlit run app.py
```

The JSON output explicitly labels regions as semantic connected components and attention
as a heuristic operational score. Validate missing-checkpoint messaging and CPU fallback.

## 6. ONNX and target-only TensorRT

Export static batch-one `1×3×512×1024` raw logits and require PyTorch/ONNX shape,
19-class, finiteness, and numerical agreement before target transfer.

Build and verify the device-neutral transfer ZIP in Colab. It contains no TensorRT engine:

```bash
python scripts/package_jetson_deployment.py build --help
python scripts/package_jetson_deployment.py verify --package jetson-deployment.zip
```

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
