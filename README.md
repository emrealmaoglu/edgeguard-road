# EdgeGuard-Road

EdgeGuard-Road is an undergraduate semantic-segmentation research prototype for
comparing edge-oriented road-scene models under one leakage-safe protocol and
preparing the selected model for ONNX/Jetson deployment.

The active delivery scope is multi-domain but deliberately single-task:

- Cityscapes, BDD100K, and IDD20K audit/frozen manifests under Cityscapes19
- SegFormer-B0, Fast-SCNN, PIDNet-S, DDRNet-23-Slim, and BiSeNetV2
- domain-uniform source sampling and source-domain macro model selection
- bounded top-two Optuna HPO at fixed `512×1024`
- CrossEntropy versus train-fit-only median-frequency weighting
- equal-domain temperature calibration, ACDC, and sealed WildDash 2/MUSES evaluation
- static ONNX export, numerical agreement, Streamlit image demo, and optional Jetson benchmark

Detection, temporal fusion, learned anomaly heads, INT8, and
advanced tracking remain experimental/legacy code. They are not dependencies of the
scientific delivery path and their fixture results are not model-quality evidence.

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
python scripts/evaluate.py --help
python scripts/predict.py --help
python scripts/export_onnx.py --help
streamlit run app.py
```

Use `notebooks/EdgeGuard_Road_Colab.ipynb` for the complete Drive-backed Colab
workflow. It defaults to audit-only and will not start GPU training until
`RUN_TRAINING` is explicitly enabled.

See `docs/SEMANTIC_FIRST_RUNBOOK.md` for calibration, ACDC, weighted-loss, reporting,
ONNX, and Jetson commands.

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
