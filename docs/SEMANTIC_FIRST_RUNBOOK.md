# Semantic-first production runbook

## One-button Colab campaign

Open `notebooks/EdgeGuard_Master_Colab.ipynb`, select **L4 GPU** and **High-RAM**, and
choose **Runtime → Run all**. There are no manual stage, finalist, review-receipt, or
accepted-release controls in the notebook. The committed owner policy binds the exact
Cityscapes/IDD audit candidates, five-model order, train-select selection rule, and the
post-acceptance official-source evaluation gate.

The campaign ID is `semantic-cs-idd-v3`. The master runner performs:

```text
preflight → restore → data → canary → smoke → pilot → screening → HPO → final →
selection → ablations → acceptance → evaluation → export → thesis → package
```

Cityscapes 2,975 accepted training samples and IDD20K 14,018 accepted plus nine quarantined
samples are reused from their exact v2 audit candidates. Cityscapes train/val and IDD20K
train/val directories must all exist after local staging. A changed candidate hash, count,
or quarantine identity stops before training.

## Training protocol

- Five-model canary: SegFormer-B0, Fast-SCNN, PIDNet-S, DDRNet-23-Slim, BiSeNetV2.
- Core smoke: 50 steps with a deliberate interruption at optimizer step 25 and verified
  resume from the same checkpoint identity.
- Core pilot: 2,000 optimizer steps.
- Five-model screening: 6,000 optimizer steps.
- Top-two HPO: 12 trials per model, 1,500/3,000-step pruning, 6,000-step ceiling.
- Five-model final: 40,000 optimizer steps. HPO winners use their selected parameters;
  the remaining models use the frozen common protocol.
- Recommendation order: Cityscapes–IDD train-select macro mIoU, rare-class mIoU, ONNX
  bytes, then fixed model name.
- Recommended-model ablations: weighted CE and 256×512. The deployment model remains
  512×1024.

Device batch may be reduced once after CUDA OOM only when gradient accumulation preserves
effective batch four. Crop size and scientific configuration do not change silently.
Training state is atomically published every 500 optimizer steps or ten minutes, including
optimizer, scheduler, AMP scaler, RNG/sampler identity and immutable input hashes.

## Outputs

The completed release directory in Drive contains:

- `EdgeGuard_Jetson_Release.zip`: five checkpoints/configs/ONNX graphs, golden vectors,
  preprocessing, ontology, ONNX validation, recommendation, and Jetson build/benchmark
  tools. No TensorRT engine is included.
- `EdgeGuard_Thesis_Bundle.zip`: source CSV/JSON, LaTeX tables, 300-DPI PNG and PDF/SVG
  figures, model/class/ablation/calibration/domain comparisons, measured gallery, and a
  hash-bound `thesis_index.md`.
- `EdgeGuard_Streamlit_Demo.zip`: accepted five-model demo, comparison data, calibration,
  overlays and honest Jetson `not_run` status.
- `release_index.json`: SHA-256 and byte size for every ZIP.

## Jetson

Extract the Jetson release on the target device. Record JetPack/L4T/CUDA/TensorRT versions,
then run `scripts/jetson/build_tensorrt.py` and `scripts/jetson/benchmark.py`. Do not build
the engine in Colab and do not copy an engine between platforms. The 25W acceptance run
uses 200 warm-ups, at least 5,000 frames and 600 seconds; UI/network/video encoding time is
excluded. No automatic JetPack upgrade or power-mode change is authorized.

## Acceptance status

Local pytest/Ruff/mypy, deterministic generation and claim-safe notebook execution are
engineering gates. The branch is not `colab-v0.1.0-rc1` eligible until two independent
clean L4 sessions pass the five-model FP32/AMP canary and a real 50-step interruption/resume
smoke. Real training metrics, release ZIPs, TensorRT and Jetson telemetry do not exist until
those external runs actually produce them.
