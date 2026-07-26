# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `67ceeff0d3b97994800077684305274c1f7ed1cc`
- **Scope:** Stage 1 decision recording and Stage 2 fixed-source/sample/checkpoint preparation through the real Colab execution gate
- **Approved upstream:** `https://github.com/XuJiacong/PIDNet.git` at `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Checkpoint policy:** Only official-repository-referenced `PIDNet_S_Cityscapes_val.pt`, non-commercial academic thesis research, no redistribution, checkpoint license `OPEN QUESTION`, explicit filename/source/access-date/SHA-256 required before safe load
- **Upstream sample result:** Primary `samples/frankfurt_000000_002196_leftImg8bit.png` verified at 2,306,975 bytes, RGB `1024×2048×3`, SHA-256 `78c65d3055fbd62e41d066813132c971a85dcdea4e5ef5459bad410bccead246`; fallback was also verified and remains unused
- **Checkpoint probe result:** Official individual file reference returned HTTP 404 and gdown 6.1.0 could not retrieve it; the official README replacement folder returned HTTP 200. No alternative URL was generated or tried, and no checkpoint file exists
- **Implementation summary:** Added deterministic fixed-checkout sample provenance, exact sample identity/hash/shape guards, exact checkout/checkpoint guards, `weights_only=True` strict state-dict loading, direct `native_logits`, separately bilinear-derived `aligned_logits`, aligned-grid semantic/MSP/entropy outputs, repeat diagnostics, ignored artifact writing, and an execution-only Colab notebook with exact-filename human upload
- **Native/aligned contract:** `native_logits` is the direct `augment=False` model result; `aligned_logits` is the separate bilinear derivative. Semantic mask, MSP, and entropy metadata all state `aligned_logits`
- **Dependency change:** Added Pillow for RGB decoding and types-Pillow for development typing. No PyTorch, CUDA, ONNX, TensorRT, model, or dataset dependency was added
- **Test result:** At 2026-07-26T02:23:55Z, Ruff check passed, Ruff format check passed, mypy passed for 22 source files, pytest passed 98/98, and `git diff --check` passed
- **Real execution result:** Not run. The approved source/sample checkout is locally verified, but no checkpoint was downloaded and Codex has no directly controlled Colab session
- **Sealed boundary:** No SMIYC files were opened, downloaded, configured, manifested, or used for debugging
- **Artifacts:** Ignored fixed PIDNet source checkout exists under `artifacts/external`; no checkpoint, copied image, derivative, logits, metric, or ONNX output exists
- **Publication:** Nothing staged, committed, pushed, merged, tagged, released, or promoted
- **Blocker:** Human retrieval of the exact checkpoint from the official replacement folder, exact-filename upload, byte-size/SHA-256 review, and human Colab execution
- **Next action:** Run `notebooks/colab/01_pidnet_single_image_spike.ipynb`, approve the displayed checkpoint hash before load, then return its JSON metadata and visual outputs for human sanity review and the vendoring/external-checkout decision

## Changed file inventory

- Governance/evidence: `LICENSES.md`, `docs/research/WP03_WP05_SOURCE_REVIEW.md`,
  `docs/PROJECT_STATE.md`, `docs/TASKS.md`, `docs/AGENT_HANDOFF.md`,
  `docs/AI_USAGE_LOG.md`.
- Config/dependencies: `pyproject.toml`, `configs/pidnet_spike.yaml`,
  `src/edgeguard/config.py`.
- Minimal implementation: `src/edgeguard/serialization.py`,
  `src/edgeguard/data/single_image.py`, `src/edgeguard/models/pidnet_spike.py`,
  `src/edgeguard/scoring/uncertainty.py`, `scripts/run_pidnet_spike.py`.
- Execution wrapper: `notebooks/colab/01_pidnet_single_image_spike.ipynb`.
- Tests: `tests/unit/test_config.py`, `tests/unit/test_serialization.py`,
  `tests/unit/test_single_image.py`, `tests/unit/test_pidnet_spike.py`,
  `tests/unit/test_uncertainty.py`, `tests/integration/test_notebook.py`.
