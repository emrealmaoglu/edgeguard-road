# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `9a734358332e8221af940d323e103b3adda376ad` (Commit A)
- **Scope:** Local environment/checkpoint validation plus project-specific Cityscapes val preparation and adapter; no metrics runner, Fishyscapes, ONNX, license change, vendoring, or dependency declaration
- **Approved upstream:** `https://github.com/XuJiacong/PIDNet.git` at `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Checkpoint pin:** Only official-repository-referenced `PIDNet_S_Cityscapes_val.pt`, SHA-256 `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`, source set to the official README replacement Drive folder, non-commercial academic thesis research, no redistribution, license `OPEN QUESTION`
- **Upstream sample result:** Primary `samples/frankfurt_000000_002196_leftImg8bit.png` verified at 2,306,975 bytes, RGB `1024×2048×3`, SHA-256 `78c65d3055fbd62e41d066813132c971a85dcdea4e5ef5459bad410bccead246`; fallback was also verified and remains unused
- **Checkpoint evidence:** Safe `weights_only=True` load returned a root mapping with 481 entries: 479 `model.` keys, including all 453 exact-shape `augment=False` inference keys plus 13 `seghead_p.*` and 13 `seghead_d.*` keys; the only non-model roots were `sem_loss.criterion.weight` and `sb_loss.criterion.weight`, both tensors shaped `[19]`.
- **Implementation summary:** The loader accepts exact inference mappings, exact supported uniform prefixes, and only that reviewed official training layout. It never performs arbitrary matching/filtering, `strict=False`, or a `weights_only=False` retry. The notebook pins the real repository origin, spike branch, and a human-entered exact commit; repairs the running kernel import path; prevents bytecode dirtiness; uses `sys.executable`; and exposes runner output on failure. Spike metadata records a fixed path-free command identity instead of raw CLI arguments.
- **Native/aligned contract:** `native_logits` is the direct `augment=False` model result; `aligned_logits` is the separate bilinear derivative. Semantic mask, MSP, and entropy metadata all state `aligned_logits`
- **Environment result:** PyTorch 2.13.0 was manually installed only in the local `.venv`; it was not added to `pyproject.toml`, requirements, or `LICENSES.md`. Python is 3.11.9 on Apple arm64 with 8 GiB RAM; MPS is built/available and CUDA is unavailable.
- **Local checkpoint result:** The new path-free verification CLI and opt-in integration test passed against the approved checkpoint. The report confirms 481 raw entries, 453 exact inference entries, 13+13 reviewed auxiliary entries, two reviewed loss roots, `torch.is_tensor`, exact shapes, `weights_only=True`, and `strict=True`.
- **Local forward result:** CPU and MPS both produced finite native `[1,19,64,128]` and aligned `[1,19,512,1024]` logits. Two repeats were byte-identical within each backend, and CPU/MPS semantic masks matched. Cross-device native-logit max absolute difference was approximately `1.85e-5`, recorded only as a diagnostic.
- **Cityscapes result:** Both official archives matched their pinned hashes. The project-specific script extracted only val plus archive docs into external storage, verified 500 image/label pairs across 3 cities, and produced manifest SHA-256 `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`. Verify-only reproduces the same hash.
- **Real execution result:** The first human-run T4/CUDA development execution succeeded: native logits `[1,19,64,128]`, aligned logits `[1,19,512,1024]`, and two consecutive forwards byte-identical. It was Git-dirty due to a temporary Colab loader patch, so it is feasibility evidence only and must not be promoted as the final clean artifact.
- **Scientific claim boundary:** MSP and predictive entropy are uncertainty/anomaly scores, not anomaly probabilities. No OOD performance, accuracy, or qualitative correctness claim was made.
- **Test result:** At 2026-07-26T05:37:44Z, Ruff check passed, Ruff format check passed for 79 files, mypy passed for 23 source files, pytest passed 131 with 2 expected opt-in skips, real checkpoint/Cityscapes opt-in tests passed 2/2, and `git diff --check` passed.
- **Sealed boundary:** No SMIYC files were opened, downloaded, configured, manifested, or used for debugging
- **Repository artifact boundary:** Checkpoint, fixed upstream checkout, CPU/MPS tensors, environment inventory, and extracted Cityscapes val remain external/ignored. No model, dataset, tensor, or large/binary artifact was added to Git.
- **Upstream checkout note:** A local preflight created untracked Python 3.11 bytecode. Tracked upstream Python 3.8 bytecode accidentally removed during cleanup was immediately restored from the pinned checkout HEAD; import-time bytecode writing is now disabled and the immutable checkout is clean.
- **Publication:** Commit A exists locally. Commit B changes are unstaged; nothing pushed, merged, tagged, released, or promoted.
- **Blocker:** The Commit B diff requires human review and explicit commit approval.
- **Next action:** Human reviews and approves Commit B. Then rerun the CPU single-image verification from the clean commit and start metrics, four scores, and the minimal runner.

## Changed file inventory

- Environment/runtime: `.env.example`, `src/edgeguard/healthcheck.py`,
  `src/edgeguard/models/pidnet_spike.py`, `scripts/run_pidnet_spike.py`, and
  `scripts/verify_pidnet_checkpoint.py`.
- Cityscapes: `src/edgeguard/data/__init__.py`,
  `src/edgeguard/data/cityscapes.py`, and `scripts/prepare_cityscapes.py`.
- Tests: `tests/unit/test_healthcheck.py`, `tests/unit/test_pidnet_spike.py`,
  `tests/unit/test_cityscapes.py`, `tests/unit/test_prepare_cityscapes.py`,
  `tests/integration/test_pidnet_real_checkpoint.py`, and
  `tests/integration/test_cityscapes_real_val.py`.
- State/evidence: `docs/PROJECT_STATE.md`, `docs/TASKS.md`,
  `docs/AGENT_HANDOFF.md`, and `docs/AI_USAGE_LOG.md`.
