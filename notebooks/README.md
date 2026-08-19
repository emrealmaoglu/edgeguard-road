# Notebook index

There is one supported Colab entry point:

- `EdgeGuard_Master_Colab.ipynb`

Select **L4 GPU** and **High-RAM**, then use **Runtime → Run all**. Do not edit cells or
choose stages manually. The notebook is generated from versioned Python and pins the exact
application commit `2495354d02e45cc4f8748e94cfcf1862ae48a295`.

The pipeline uses the verified Cityscapes bundle and IDD20K shards already in Drive. It
copies them to `/content`, verifies train/validation paths and frozen audit identities,
then runs canary → smoke/resume → pilot → screening → HPO → all five final models →
selection/ablations → policy acceptance → official-source evaluation → ONNX → thesis,
Streamlit and Jetson packages.

After a runtime reset, select L4 + High-RAM and use **Run all** again. Completed stages are
accepted only when every indexed artifact hash matches. Checkpoints are published every
500 optimizer steps or ten minutes. No dataset, prior campaign, or Drive source is deleted.

`EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` executes all four code cells without Drive, network,
GPU, licensed data, or scientific claims. It validates the wrapper contract only. Real
Colab CUDA acceptance still requires two independent clean L4 sessions.
