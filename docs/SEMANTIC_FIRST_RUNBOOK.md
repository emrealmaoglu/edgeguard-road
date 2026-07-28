# Semantic-first execution runbook

This is the canonical rescue path. All paths below are examples and must point to
approved external storage. Git never contains datasets, checkpoints, ONNX graphs, or
generated scientific evidence.

## 0. Multi-domain canonical path

ADR-0009 supersedes the old three-model/Cityscapes-only experiment ladder while
preserving its command compatibility. The scientific source domains are Cityscapes,
BDD100K, and IDD20K. Every audit first creates a candidate; training accepts only
schema-2 manifests explicitly copied to `split_state=frozen` by the reviewed
`--freeze-approved` action.

```bash
python scripts/audit_dataset.py --dataset cityscapes \
  --dataset-root "$CITYSCAPES_ROOT" --output-root "$OUTPUT_ROOT/audit/cityscapes"
python scripts/audit_dataset.py --dataset bdd100k \
  --dataset-root "$BDD100K_ROOT" --output-root "$OUTPUT_ROOT/audit/bdd100k"
python scripts/audit_dataset.py --dataset idd20k \
  --dataset-root "$IDD20K_ROOT" --output-root "$OUTPUT_ROOT/audit/idd20k"
```

Review label coverage, invalid/corrupt files, sequence groups, exact duplicates,
near-duplicate candidates, and role counts before freezing each candidate. IDD input
masks must be the official raw source-ID `labelids` representation; fallback,
autorickshaw, animal, curb, billboard, bridge, tunnel, group, and void classes become
ignore `255` rather than background or an approximate Cityscapes class.

`--allow-fixture-count` exists only for unit fixtures. A manifest created with that
escape hatch is marked `scientific_eligible=false` and is rejected by training.
Freeze reviewed candidates explicitly; supplying a candidate to training is never
enough:

```bash
python scripts/audit_dataset.py --dataset bdd100k --freeze-approved \
  --split-manifest "$OUTPUT_ROOT/audit/bdd100k/bdd100k_audit/dataset_manifest.candidate.json" \
  --output-root "$OUTPUT_ROOT/manifests"
```

After all three frozen manifests exist, compute the cross-domain duplicate gate,
pooled train-fit weights, and the bottom-five rare-class set:

```bash
python scripts/audit_dataset.py --output-root "$OUTPUT_ROOT/statistics" \
  --data-manifest "$OUTPUT_ROOT/manifests/cityscapes.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/bdd100k.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/idd20k.frozen.json"
```

All five models use optimizer-step budgets: smoke 50, pilot 2,000, screening 6,000,
HPO 6,000 per trial, and final 40,000. Multi-domain training is invoked by repeating
`--data-manifest`; the custom distributed sampler gives every domain equal expected
exposure.

```bash
python scripts/train.py --model segformer_b0 --stage smoke \
  --data-manifest "$OUTPUT_ROOT/manifests/cityscapes.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/bdd100k.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/idd20k.frozen.json" \
  --output-root "$OUTPUT_ROOT/runs" --mmseg-root "$MMSEG_ROOT"
```

Screening evaluation is run once per source `train_select`; reporting computes the
unweighted domain-macro mIoU. A screening candidate is valid only when all three
source-domain results and a numerically validated ONNX record exist. Only the two
highest valid candidates enter 12-trial TPE HPO with 1,500/3,000-step
successive-halving rungs. Interrupted trials are closed as failed, exact duplicate
parameter trials are pruned, and the SQLite study resumes without exceeding twelve
terminal trials. Resolution, loss, initialization, augmentation, and domain sampling
are fixed during HPO.

```bash
python scripts/train.py --stage hpo \
  --candidate-table "$OUTPUT_ROOT/reports/screening/candidate_table.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/cityscapes.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/bdd100k.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/idd20k.frozen.json" \
  --rare-classes-file "$OUTPUT_ROOT/statistics/rare_classes.json" \
  --output-root "$OUTPUT_ROOT/runs" --mmseg-root "$MMSEG_ROOT"
```

The dataset-composition ablation uses exactly Cityscapes, Cityscapes+BDD100K, and
Cityscapes+BDD100K+IDD20K. The class-imbalance ablation remains outside HPO and uses
the pooled train-fit statistics:

```bash
python scripts/train.py --model "$FINAL_MODEL" --stage final \
  --loss median_frequency --audit-report "$OUTPUT_ROOT/statistics/class_weights.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/cityscapes.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/bdd100k.frozen.json" \
  --data-manifest "$OUTPUT_ROOT/manifests/idd20k.frozen.json" \
  --output-root "$OUTPUT_ROOT/runs" --mmseg-root "$MMSEG_ROOT"
```

Global calibration first saves sampled logits separately from each frozen
`train_calibration`, then fits one temperature with equal valid-pixel counts:

```bash
python scripts/evaluate.py calibrate-global \
  --evidence "$OUTPUT_ROOT/calibration/cityscapes.npz" \
  --evidence "$OUTPUT_ROOT/calibration/bdd100k.npz" \
  --evidence "$OUTPUT_ROOT/calibration/idd20k.npz" \
  --output "$OUTPUT_ROOT/calibration/global-temperature.json"
```

ACDC, source official validations, and sealed external evaluation open only after
checkpoint and preprocessing freeze. WildDash 2/MUSES audit requires an exact pairs
file, license/access record, and all source manifests for overlap exclusion. A
human-authored release JSON must bind the frozen manifest SHA and model SHA before
`evaluate.py package-external` creates a server archive. The server result is never
fed back into training, HPO, calibration, or threshold selection.
The pairs file declares the release-specific submission encoding. WildDash 2 is
fail-closed to regular Cityscapes label IDs; internal train IDs are never uploaded.

BDD100K/IDD20K official validation data is audited separately and never partitioned
into training roles. It must be checked against the frozen training manifests before
freeze:

```bash
python scripts/audit_dataset.py --dataset bdd100k --source-split val \
  --dataset-root "$BDD100K_ROOT" \
  --source-manifest "$OUTPUT_ROOT/manifests/cityscapes.frozen.json" \
  --source-manifest "$OUTPUT_ROOT/manifests/bdd100k.frozen.json" \
  --source-manifest "$OUTPUT_ROOT/manifests/idd20k.frozen.json" \
  --output-root "$OUTPUT_ROOT/audit/bdd100k-val"
```

Training, HPO, evaluation, calibration, ONNX export, and sealed external operations
append Git-aware, result-hashed rows to external `run_ledger.jsonl` files. These
ledgers are evidence artifacts and must not be committed.

## 1. Environment and audit gate

Use `notebooks/EdgeGuard_Road_Colab.ipynb` for the pinned CUDA/MMCV/MMSeg setup. For
an already prepared runtime, set `MMSEG_ROOT` to the pinned MMSeg checkout.

```bash
python scripts/audit_dataset.py \
  --dataset-root "$CITYSCAPES_ROOT" \
  --output-root "$OUTPUT_ROOT/audit"
```

Training remains blocked unless `summary.json` reports `audit_passed=true`. Review
the generated `CSF-SPLIT-D.json` before treating it as human-frozen. The audit emits:

```text
dataset_audit/
├── summary.json
├── dataset_audit.md
├── CSF-SPLIT-D.json
├── class_pixel_frequency.csv
├── class_image_frequency.csv
├── class_cooccurrence.csv
├── crop_survival.csv
├── city_distribution.csv
├── split_comparison.csv
├── split_summary.json
├── class_weights.json
├── rare_classes.json
├── corrupt_files.csv
├── invalid_labels.csv
├── duplicates.csv
├── near_duplicates.csv
├── black_images.csv
├── low_information_images.csv
├── all_ignore_masks.csv
└── figures/
```

Class weights and rare/medium/frequent groups are recomputed only from `train_fit`
after the split passes validation. Training also checks that `summary.json` is a
passing 2,975-pair audit receipt beside the split; copying an isolated split file is
not a valid training handoff.

## 2. Fixed experiment ladder

The legacy Cityscapes-only command below remains supported for recovery. New
scientific comparisons run all five models through the multi-domain command in
Section 0 and never skip directly to final training.

```bash
for model in segformer_b0 fast_scnn pidnet_s ddrnet_23_slim bisenetv2; do
  python scripts/train.py \
    --model "$model" \
    --stage smoke \
    --dataset-root "$CITYSCAPES_ROOT" \
    --split-manifest "$OUTPUT_ROOT/audit/dataset_audit/CSF-SPLIT-D.json" \
    --output-root "$OUTPUT_ROOT/runs" \
    --mmseg-root "$MMSEG_ROOT"
done
```

Stage budgets are frozen in `configs/rescue/semantic_first.yaml`:

- `smoke`: 50 optimizer steps
- `pilot`: 2,000 optimizer steps
- `screening`: 6,000 optimizer steps
- `hpo`: 6,000 optimizer steps per trial
- `final`: 40,000 optimizer steps

The common effective batch is four. Fast-SCNN therefore avoids the known batch-one
PPM BatchNorm failure. Every model uses stock MMSeg datasets, Runner, IoUMetric,
checkpoint hook, and PolyLR with positive terminal learning rate.

After the final scientific candidate is accepted, run the loss ablation without
changing another protocol field:

```bash
python scripts/train.py \
  --model "$FINAL_MODEL" --stage final --loss median_frequency \
  --audit-report "$OUTPUT_ROOT/audit/dataset_audit/class_weights.json" \
  --dataset-root "$CITYSCAPES_ROOT" \
  --split-manifest "$OUTPUT_ROOT/audit/dataset_audit/CSF-SPLIT-D.json" \
  --output-root "$OUTPUT_ROOT/runs" --mmseg-root "$MMSEG_ROOT"
```

## 3. Evaluation and calibration

Routine screening evaluates only `train_select`:

```bash
python scripts/evaluate.py run \
  --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
  --dataset cityscapes --dataset-root "$CITYSCAPES_ROOT" \
  --split-manifest "$OUTPUT_ROOT/audit/dataset_audit/CSF-SPLIT-D.json" \
  --role train_select \
  --rare-classes-file "$OUTPUT_ROOT/audit/dataset_audit/rare_classes.json" \
  --output-dir "$OUTPUT_ROOT/evaluation/train-select/$MODEL"
```

Fit temperature only on `train_calibration`:

```bash
python scripts/evaluate.py run \
  --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
  --dataset cityscapes --dataset-root "$CITYSCAPES_ROOT" \
  --split-manifest "$OUTPUT_ROOT/audit/dataset_audit/CSF-SPLIT-D.json" \
  --role train_calibration --fit-temperature \
  --output-dir "$OUTPUT_ROOT/evaluation/calibration/$MODEL"
```

Only after model/config freeze, evaluate official Cityscapes validation with the
frozen temperature:

```bash
python scripts/evaluate.py run \
  --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
  --dataset cityscapes --dataset-root "$CITYSCAPES_ROOT" \
  --role official_val_common_eval \
  --temperature-file "$OUTPUT_ROOT/evaluation/calibration/$MODEL/temperature.json" \
  --rare-classes-file "$OUTPUT_ROOT/audit/dataset_audit/rare_classes.json" \
  --output-dir "$OUTPUT_ROOT/evaluation/official-val/$MODEL"
```

Evaluate ACDC conditions independently with the same frozen checkpoint and
temperature. `ACDC_ROOT` must preserve the official extracted
`rgb_anon/{condition}/val/{sequence}` and `gt/{condition}/val/{sequence}` layout:

```bash
for condition in fog night rain snow; do
  python scripts/evaluate.py run \
    --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
    --dataset acdc --dataset-root "$ACDC_ROOT" --role domain_shift_val \
    --condition "$condition" \
    --temperature-file "$OUTPUT_ROOT/evaluation/calibration/$MODEL/temperature.json" \
    --rare-classes-file "$OUTPUT_ROOT/audit/dataset_audit/rare_classes.json" \
    --output-dir "$OUTPUT_ROOT/evaluation/acdc/$condition/$MODEL"
done
```

If ACDC access is unavailable, do not rename a synthetic corruption test to ACDC or
external OOD. Label it `synthetic robustness stress test` and keep the ACDC cells blank.

The deterministic fallback is created and evaluated explicitly:

```bash
python scripts/evaluate.py stress \
  --cityscapes-root "$CITYSCAPES_ROOT" \
  --condition fog --severity 0.6 \
  --output-root "$OUTPUT_ROOT/stress/fog"

python scripts/evaluate.py run \
  --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
  --dataset cityscapes_stress --dataset-root "$OUTPUT_ROOT/stress/fog" \
  --role synthetic_stress_test \
  --output-dir "$OUTPUT_ROOT/evaluation/stress/fog/$MODEL"
```

## 4. ONNX, selection, and report package

```bash
python scripts/export_onnx.py \
  --resolved-config "$RUN_DIR/resolved.py" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_ROOT/exports/$MODEL.onnx" --device cuda

python scripts/evaluate.py summarize \
  --evaluation-root "$OUTPUT_ROOT/evaluation" \
  --export-root "$OUTPUT_ROOT/exports" \
  --output-dir "$OUTPUT_ROOT/reports/week3"
```

The report command creates measured tables, presentation outline, and a top-two
candidate only when at least two models have all three source-domain selection mIoUs
and validated ONNX records. The scientific candidate has the highest source-domain
macro mIoU. The edge candidate has the lowest ONNX latency among models within 0.03
mIoU of the best; if both roles select the same model, the next-highest-mIoU model
becomes the second finalist. Human acceptance remains mandatory.

## 5. Demo and deployment gate

```bash
export EDGEGUARD_RUN_ROOT="$OUTPUT_ROOT"
streamlit run app.py
```

The demo discovers only complete ONNX graphs or checkpoint/`resolved.py` pairs. It
shows raw probabilities by default; an optional global-temperature file is applied
only when its checkpoint hash matches the selected checkpoint (or the ONNX validation
record). Missing CUDA falls back visibly to CPU. UI time is separate from model latency.

Jetson execution is manual and non-privileged:

```bash
python scripts/jetson/benchmark_onnx.py \
  --model "$OUTPUT_ROOT/exports/$MODEL.onnx" \
  --output "$OUTPUT_ROOT/jetson/$MODEL.json" \
  --provider CUDAExecutionProvider --warmup 50 --iterations 500
```

The script does not change power mode, install packages, build engines, or infer
power/thermal claims. A reviewed telemetry log may be attached for later human
interpretation. Without a real device run, the thesis must state that Jetson latency,
power, temperature, and throttling remain unmeasured.

## Stop and fallback rules

- After two unexpected integration failures in one model, audit the entire lifecycle
  and record a structured negative result before continuing other models.
- Never reuse a non-identical checkpoint with `--resume`.
- Never overwrite a non-empty run, evaluation, report, export, or benchmark output.
- If fewer than two models complete screening, finish one reliable baseline and
  report the failures; do not fabricate a comparison.
- Missing ACDC or Jetson access narrows claims but does not block the Cityscapes,
  ONNX, demo, Colab, and thesis-evidence deliverables.
