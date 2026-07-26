# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `4ffba90f1f27723c525a74d61ec3eff7f6951468`
- **Scope:** Close Stage 2 loader/notebook reproducibility and path sanitization only; no local PyTorch, Cityscapes, metrics, Fishyscapes, license, or later-block work
- **Approved upstream:** `https://github.com/XuJiacong/PIDNet.git` at `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Checkpoint pin:** Only official-repository-referenced `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, non-commercial academic thesis research, no redistribution, license `OPEN QUESTION`
- **Upstream sample result:** Primary `samples/frankfurt_000000_002196_leftImg8bit.png` verified at 2,306,975 bytes, RGB `1024×2048×3`, SHA-256 `78c65d3055fbd62e41d066813132c971a85dcdea4e5ef5459bad410bccead246`; fallback was also verified and remains unused
- **Checkpoint evidence:** Safe `weights_only=True` load returned a root mapping with 481 entries: 479 `model.` keys, including all 453 exact-shape `augment=False` inference keys plus 13 `seghead_p.*` and 13 `seghead_d.*` keys; the only non-model roots were `sem_loss.criterion.weight` and `sb_loss.criterion.weight`, both tensors shaped `[19]`.
- **Implementation summary:** The loader accepts exact inference mappings, exact supported uniform prefixes, and only that reviewed official training layout. It never performs arbitrary matching/filtering, `strict=False`, or a `weights_only=False` retry. The notebook pins the real repository origin, spike branch, and a human-entered exact commit; repairs the running kernel import path; prevents bytecode dirtiness; uses `sys.executable`; and exposes runner output on failure. Spike metadata records a fixed path-free command identity instead of raw CLI arguments.
- **Native/aligned contract:** `native_logits` is the direct `augment=False` model result; `aligned_logits` is the separate bilinear derivative. Semantic mask, MSP, and entropy metadata all state `aligned_logits`
- **Dependency change:** Added Pillow for RGB decoding and types-Pillow for development typing. No PyTorch, CUDA, ONNX, TensorRT, model, or dataset dependency was added
- **Real execution result:** The first human-run T4/CUDA development execution succeeded: native logits `[1,19,64,128]`, aligned logits `[1,19,512,1024]`, and two consecutive forwards byte-identical. It was Git-dirty due to a temporary Colab loader patch, so it is feasibility evidence only and must not be promoted as the final clean artifact.
- **Scientific claim boundary:** MSP and predictive entropy are uncertainty/anomaly scores, not anomaly probabilities. No OOD performance, accuracy, or qualitative correctness claim was made.
- **Test result:** At 2026-07-26T05:20:08Z, Ruff check passed, Ruff format check passed for 72 files, mypy passed for 22 source files, pytest passed 116/116, notebook JSON/structure and path/secret checks passed, artifact/binary scan passed, and `git diff --check` passed.
- **Sealed boundary:** No SMIYC files were opened, downloaded, configured, manifested, or used for debugging
- **Repository artifact boundary:** No checkpoint, dataset, tensor, generated image, or large/binary artifact was added to Git; T4 development outputs remained external to the repository.
- **Publication:** Nothing staged, committed, pushed, merged, tagged, released, or promoted
- **Blocker:** The Stage 2 closure diff requires human review and explicit Commit A approval.
- **Next action:** Human reviews this diff and approves Commit A. The next implementation block is manual local PyTorch inventory, real checkpoint verification, and a `512×1024` CPU forward; the clean Colab campaign remains later.

## Changed file inventory

- Governance/evidence: `docs/research/WP03_WP05_SOURCE_REVIEW.md`,
  `docs/PROJECT_STATE.md`, `docs/TASKS.md`, `docs/AGENT_HANDOFF.md`, and
  `docs/AI_USAGE_LOG.md`.
- Implementation: `src/edgeguard/models/pidnet_spike.py` and
  `scripts/run_pidnet_spike.py`.
- Execution wrapper: `notebooks/colab/01_pidnet_single_image_spike.ipynb`.
- Tests: `tests/unit/test_pidnet_spike.py` and
  `tests/integration/test_notebook.py`.
