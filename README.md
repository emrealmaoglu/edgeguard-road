# EdgeGuard-Road

EdgeGuard-Road is an undergraduate road-perception research prototype that compares
lightweight semantic models under one leakage-safe protocol and deploys the selected
accuracy–speed–reliability trade-off on NVIDIA Jetson Orin Nano Super.

The active delivery scope is multi-domain but deliberately single-task:

- Cityscapes + IDD20K scientific manifests under Cityscapes19, plus a provenance-limited
  BDD100K mirror audit that is barred from HPO/main claims
- SegFormer-B0, Fast-SCNN, PIDNet-S, DDRNet-23-Slim, and BiSeNetV2
- domain-uniform source sampling and source-domain macro model selection
- bounded top-two Optuna HPO at fixed `512×1024`
- CrossEntropy versus train-fit-only median-frequency weighting
- equal-domain temperature calibration, ACDC, and sealed WildDash 2/MUSES evaluation
- static ONNX/TensorRT FP16 validation and sustained 25W/MAXN SUPER Jetson measurement
- road and ego-reachable drivable corridor extraction
- semantic connected-component regions with confidence and entropy summaries
- source-calibrated frame shift alerts and explainable operational-attention maps

Detection, temporal fusion, learned anomaly heads, INT8, and advanced tracking remain
experimental/legacy. RTMDet-Tiny is the only conditional phase-two detector and cannot
start until every gate in `PROJECT_CHARTER.md` passes.

## Current evidence boundary

- The pre-rescue feature branch had passing local/remote engineering probes for five
  random-weight semantic architectures, but those probes are non-scientific.
- An external pretrained PIDNet-S reference completed a 500-image Cityscapes
  validation run with measured mIoU `0.7875813077220126`; its claim boundary remains
  in `docs/research/CITYSCAPES_FULL_VAL_EVIDENCE.md`.
- No new rescue-path Cityscapes training, ACDC evaluation, final ONNX checkpoint, or
  Jetson measurement is claimed until real external artifacts are supplied and the
  corresponding commands complete.

## Active commands

Python 3.10 or newer is required. Install local development tools with:

```bash
python -m pip install -e '.[dev,rescue]'
```

Audit Cityscapes before any training:

```bash
python scripts/audit_dataset.py \
  --dataset cityscapes \
  --dataset-root /path/to/cityscapes \
  --output-root /path/to/edgeguard-output/audit
```

Run the first real-data gate after the audit passes and the split is reviewed:

```bash
python scripts/train.py \
  --model segformer_b0 \
  --stage smoke \
  --dataset-root /path/to/cityscapes \
  --split-manifest /path/to/edgeguard-output/audit/dataset_audit/CSF-SPLIT-D.json \
  --output-root /path/to/edgeguard-output/runs \
  --mmseg-root /path/to/mmsegmentation
```

BDD100K and IDD20K use the same command with `--dataset bdd100k` or
`--dataset idd20k`. Each generated candidate manifest must be reviewed and explicitly
frozen with `--freeze-approved`. Multi-domain runs then receive one repeated
`--data-manifest` argument per source domain; raw data roots are never combined or
copied into Git. `--source-split val` creates a separate official-validation candidate
that cannot appear in training/HPO roles. Test-only fixture-count manifests are
explicitly non-scientific and rejected by training.

The HPO entry point selects its two models from a measured screening table:

```bash
python scripts/train.py \
  --stage hpo \
  --candidate-table /output/reports/screening/candidate_table.json \
  --data-manifest /output/manifests/cityscapes.frozen.json \
  --data-manifest /output/manifests/bdd100k.frozen.json \
  --data-manifest /output/manifests/idd20k.frozen.json \
  --rare-classes-file /output/statistics/rare_classes.json \
  --output-root /output/runs \
  --mmseg-root /runtime/mmsegmentation
```

The other public commands are:

```text
python scripts/prepare_dataset.py --help
python scripts/evaluate.py --help
python scripts/predict.py --emit-regions --emit-risk --help
python scripts/export_onnx.py --help
python scripts/jetson/build_tensorrt.py --help
python scripts/jetson/benchmark.py --help
python scripts/check_detection_gate.py --help
streamlit run app.py
```

Use `notebooks/EdgeGuard_Master_Colab.ipynb` as the only Colab entry point. Select an
L4 GPU and High-RAM, then choose **Runtime → Run all**. The notebook checks out its
immutable application commit and runs the `semantic-cs-idd-v3` pipeline through verified
Drive staging, five-model training/HPO/final selection, official-source evaluation, ONNX
export, thesis figures, Streamlit packaging, and Jetson handoff. If Colab disconnects,
open the same notebook in a new L4 High-RAM runtime and choose **Run all** again; only
hash-verified phases are skipped and incomplete training resumes from Drive checkpoints.

The notebook refuses an unsuitable GPU/RAM/disk allocation. TensorRT engines and real
Jetson power/latency/thermal measurements are never fabricated in Colab; they remain
`not_run` until produced on the target device. See `docs/SEMANTIC_FIRST_RUNBOOK.md`.

## Scientific boundaries

- `train_fit`, `train_select`, `train_calibration`, and official validation are
  disjoint inside every source domain; sequences cannot cross roles.
- Official validation is final common evaluation only; it is never used for model
  selection, loss design, or temperature fitting.
- ACDC is domain-shift evaluation only. If unavailable, a synthetic stress test must
  be labeled synthetic and cannot support an external-OOD claim.
- WildDash 2, MUSES, and fallback KITTI manifests require a hash-bound human release
  after checkpoint/model-selection freeze. Their results cannot be used for iteration.
- Datasets, checkpoints, ONNX graphs, logs, and generated figures remain outside Git.
- Failed runs and failed exports remain evidence; missing measurements are never estimated.
- Scientific operations append Git-aware, hash-bound rows to external
  `run_ledger.jsonl`; generated ledgers are never committed.

The prototype is not a safety-certified ADAS product or vehicle controller.
Semantic connected components are not instance detections, and the deterministic
attention score is not collision probability or learned physical risk.
