# Canonical Colab runbook

The sole active notebook is `notebooks/EdgeGuard_Master_Colab.ipynb`. The two former
delivery notebooks and all numbered notebooks were removed from the working tree; Git
history remains the recovery path for them.

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

The notebook checks out application commit
`b22fd123e46478c1d7d368b8fbf50a28dbe28fdd`. It does not use the hosted Python,
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
