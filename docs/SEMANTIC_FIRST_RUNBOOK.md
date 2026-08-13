# Semantic-first production runbook

## One-button Colab campaign

Open `notebooks/EdgeGuard_Master_Colab.ipynb`, select **L4 GPU** and **High-RAM**, and
choose **Runtime → Run all**. There are no manual stage, finalist, review-receipt, or
accepted-release controls in the notebook. The committed owner policy binds the exact
Cityscapes/IDD audit candidates, model order, train-select selection rule, and the
post-acceptance official-source evaluation gate.

**2026-08-13 model-scope decisions.** Real per-iteration throughput showed `fast_scnn`
(~5.3 s/iter) and `bisenetv2` (~6.5-7 s/iter) are 15-25x slower than `segformer_b0`/
`pidnet_s`/`ddrnet_23_slim`; a full 40,000-step final run on all five would cost roughly
155 GPU-hours, incompatible with a 4-day presentation deadline. Under CLAUDE.md's
2026-08-12 broad scientific-decision delegation, `final_models` in
`configs/campaign/semantic_cs_idd_v3_authorization.json` is narrowed to `segformer_b0`,
`pidnet_s`, `ddrnet_23_slim` (~24 GPU-hours total). Separately, `bisenetv2`'s own
screening run was interrupted live mid-session (owner-directed stop, since resuming it
would cost ~8 more hours for a result that cannot change `final_models` either way);
`screening_models` in the same policy file is narrowed to the four models that actually
completed the full 6,000-step screening ceiling
(`segformer_b0`/`fast_scnn`/`pidnet_s`/`ddrnet_23_slim`), so a fresh Colab session does
not automatically try to resume and finish `bisenetv2`'s screening. `fast_scnn`
(screening mIoU 20.10) and `bisenetv2` (interrupted, smoke-only mIoU 6.57) keep their
real evidence in the report with an explicit exclusion note — never silently dropped.
See `docs/AI_USAGE_LOG.md` for the full evidence and decision record.

**2026-08-13 `fast_scnn` also excluded from `screening_models`.** The automatic
recovery-identity migration above could not migrate `fast_scnn`'s real, complete
6000-step screening checkpoint (mIoU 20.10) to the current commit — see the migration
finding in `docs/AI_USAGE_LOG.md`. Real pilot-stage throughput measured on both L4
(~5.2 s/iter) and A100 (~5.3-5.6 s/iter — no speedup for this model at the frozen
`device_batch: 4`, this workload appears CPU/dataloader-bound rather than
GPU-compute-bound at this batch size) puts a fresh 6000-step run at ~9 GPU-hours.
`fast_scnn` is already excluded from `final_models`, so rerunning its screening cannot
change any downstream decision; its real evidence already exists and is preserved.
`screening_models` is narrowed to `segformer_b0`/`pidnet_s`/`ddrnet_23_slim`.

**2026-08-13 automatic recovery-identity migration.** A commit that only changes
orchestration code (e.g. the `screening_models`/`final_models` decision above) still
changes `project_commit`, which is baked into every training run's immutable identity
hash — so, without this step, resuming on a new commit would make every real,
already-completed checkpoint look "stale" and get silently retrained from scratch. The
master runner now runs `scripts/migrate_recovery_identity.py` automatically, right
after data staging and before the production pipeline starts, for every model in
`screening_models`: it recomputes each model's identity under the commit actually
recorded on its existing Drive receipt and verifies that matches the real recorded
hash, recomputes it again under the current commit and verifies `project_commit` is
the *only* field that differs, and only then republishes the same real checkpoint
bytes under the new identity. It refuses outright (never retrains, never fabricates)
if either check fails, or does nothing if there is nothing to migrate — so this is
always safe to run, unattended, on every session. **No manual step is needed**; this
happens automatically inside **Runtime → Run all**.

The campaign ID is `semantic-cs-idd-v3`. The master runner performs:

```text
preflight → restore → data → recovery-identity-migration → canary → smoke → pilot →
screening → HPO → final → selection → ablations → acceptance → evaluation → export →
thesis → package
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
- Screening (segformer_b0, fast_scnn, pidnet_s, ddrnet_23_slim as of 2026-08-13; see the
  model-scope decision above): 6,000 optimizer steps. `bisenetv2`'s screening is
  intentionally abandoned at its interrupted checkpoint.
- Top-two HPO: 12 trials per model, 1,500/3,000-step pruning, 6,000-step ceiling.
- Final (segformer_b0, pidnet_s, ddrnet_23_slim as of 2026-08-13; see the model-scope
  decision above): 40,000 optimizer steps. HPO winners use their selected parameters;
  the remaining final-set models use the frozen common protocol.
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

- `EdgeGuard_Jetson_Release.zip`: one checkpoint/config/ONNX graph per accepted final
  model (see the model-scope decision above), golden vectors,
  preprocessing, ontology, ONNX validation, recommendation, and Jetson build/benchmark
  tools. No TensorRT engine is included.
- `EdgeGuard_Thesis_Bundle.zip`: source CSV/JSON, LaTeX tables, 300-DPI PNG and PDF/SVG
  figures, model/class/ablation/calibration/domain comparisons, measured gallery, and a
  hash-bound `thesis_index.md`.
- `EdgeGuard_Streamlit_Demo.zip`: accepted final-model-set demo, comparison data, calibration,
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
