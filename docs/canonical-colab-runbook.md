# Canonical Colab runbook

The sole active notebook is `notebooks/EdgeGuard_Master_Colab.ipynb`. The two former
delivery notebooks and all numbered notebooks were removed from the working tree; Git
history remains the recovery path for them.

## Before every push

Run the real local CPU rehearsal before pushing any change that touches
`src/edgeguard/rescue/mmseg_runtime.py`, `mmseg_components.py`, or `colab_pipeline.py`:

```bash
EDGEGUARD_MMSEG_CHECKOUT=.local/mmsegmentation-cpu \
  pytest tests/integration/test_colab_pipeline_cpu_rehearsal.py -v
```

This drives the real `ColabPipeline` orchestrator (real subprocesses, real
`EdgeGuardRecoveryHook` interrupt+resume, real `val_dataloader` build) against tiny
synthetic fixture data on CPU. It is not the same thing as "claim-safe local cell
execution" (`scripts/dev/run_campaign_notebook_harness.py`), which only proves the
generated notebook's cells import and execute correctly — the real training call there is
stubbed behind a hardcoded `{"scientific_status": "not_run"}` dict and never touches
`Runner.train()`. Seven real bugs (a hardcoded AMP dtype, a wrong `Pad` transform keyword, a
bare-filename `last_checkpoint` marker, a `Pad` size-argument dimension-order bug, a stale
cross-commit Drive recovery pointer crashing the whole campaign before any training step
ran, a checkpoint RNG-state device mismatch, and a PIDNet `BoundaryLoss` bf16 dtype
mismatch) were each discovered one at a time on a real Colab L4 run before this rehearsal
existed. Five would have been caught by this rehearsal once run against a real foreign
device — the stale-recovery scenario needed a dedicated new test that seeds the recovery
store with a mismatched identity, since every other rehearsal test starts from a clean,
empty recovery store; the RNG-state bug needed the harness to actually see a non-CPU device
during `Runner.resume()`. The seventh (`BoundaryLoss`) is structurally uncatchable by this
CPU-only rehearsal even in principle: it only manifests under real bf16 autocast on real
CUDA, and `resolve_auto_precision` always resolves to `fp32` without CUDA. This is the
project's one remaining fully CPU-blind bug class (shared with the original AMP-dtype bug)
— treat any future real-Colab crash that only reproduces under `precision: bf16`/`fp16` the
same way: fix it, write the strongest CPU-checkable regression test possible (config shape,
post-fix invariants, numerical no-op-at-matching-dtype), and say plainly in the commit that
real L4 confirmation is the only real test.

**Do not force `mmengine.device.utils.DEVICE = "cpu"`** (e.g. via a local
`sitecustomize.py`) as a blanket workaround for MPS/CPU quirks on Apple Silicon dev
machines. This was done in an earlier session to route around unrelated MPS operator gaps
(`Adaptive pool MPS: ... non-divisible input sizes`, `view size is not compatible with
input tensor's size and stride`) and it worked — but it also silently hid the RNG-state
device-mismatch bug, since `mmengine.device.get_device()` genuinely returns `"mps"` on
these machines, which is itself a real foreign device relative to a CPU-saved tensor. If a
CUDA-only or CPU/GPU-transition class of bug is suspected, run the rehearsal (or a
narrower reproduction) against the real device this machine reports — only fall back to
forcing `DEVICE=cpu` for unrelated operator-support gaps, and say explicitly in the test or
commit message that the run cannot see device-class bugs when you do. A `torch.load`
`weights_only` default patch (needed separately, for this machine's newer local torch
versus the pinned Colab/CI torch) does not have this problem and can stay.

## Run

1. Open the master notebook in Colab.
2. Select an L4 GPU and High-RAM runtime.
3. Choose **Runtime → Run all** once.
4. Leave the tab running, or use Colab Pro/Pro+ **background execution** (Runtime menu)
   so the session keeps running after the browser tab closes. Background execution
   extends how long one session can run unattended; it does not change what happens
   after the session itself ends.
5. After a disconnect (session limit, idle timeout, or Colab-side interruption), a human
   must open a new L4 High-RAM runtime and choose **Run all** again. Hash-verified
   completed phases are skipped automatically and training resumes from the last
   published checkpoint — but there is no way, from inside the notebook or this
   codebase, to make a new Colab session start itself after the previous one dies. Full
   unattended multi-session autonomy is a Colab platform limit, not an engineering gap.
6. On completion, use the three Drive ZIPs and `release_index.json` printed by the final
   cell. The Jetson ZIP is also requested as a browser download. All three ZIPs
   (checkpoints/configs, thesis figures/tables, Streamlit demo bundle) are meant to be
   downloaded and kept outside Drive for thesis writing and Jetson deployment.

The public orchestrator sequence is:

```text
preflight → restore → stage-data → canary → smoke → pilot → extension-smoke →
screening → hpo → final → selection → ablation → accept → validation-data →
evaluate → export → report → package
```

The notebook checks out application commit `4917c49` (see `git log` for the full SHA).
It does not use the hosted Python,
NumPy, Torch, or uv for training. The managed environment is Python 3.11.13, uv 0.8.8,
NumPy 1.26.4, PyTorch 2.1.1/cu121, MMEngine 0.10.7, mmcv-lite 2.1.0,
headless OpenCV 4.10.0.84, and the pinned MMSegmentation v1.2.2 commit.

## Boundaries

- Cityscapes and IDD20K are staged to local `/content`; training never samples mounted
  Drive files directly.
- The exact approved training-manifest hashes and counts are fail-closed.
- Model selection uses only `train_select`; official source validation is opened after
  the five-model release is accepted and cannot alter the recommendation.
- A smoke/canary/acceptance fixture is not a thesis result.
- TensorRT is built on the real Jetson. Device benchmarks remain `not_run` until measured.
- Do not tag the notebook Colab-ready until two clean L4 canaries and the intentional
  50-step interruption/resume gate have succeeded.
