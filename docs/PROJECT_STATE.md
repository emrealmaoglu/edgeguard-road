# Project State

- **Repository/branch:** `.` on `feat/first-vertical-slice`.
- **Latest local milestone:** `EG-LOCAL-COMPLETE`, comprehensive non-scientific local closure.
- **Quality:** Python 3.11 local gate passes with Ruff, format, mypy, and 328 pytest tests; two optional tests are skipped. Python 3.10 normal CI remains a remote gate for the final revision.
- **Real semantic code paths:** Fast-SCNN, BiSeNetV2, PIDNet-S, DDRNet-23-Slim, and SegFormer-B0 each complete actual random-weight MMSeg construction, fixture loading, augmentation, loss/backward, two optimizer steps with accumulation, validation, 19-channel native logits, feature extraction, anomaly-head optimization, checkpoint, and exact resume. No synthetic ranking is permitted.
- **Real detector code paths:** YOLO11n and RT-DETR-R18 each complete authoritative-library random-weight construction, actual detector loss/backward, optimizer/scheduler step, checkpoint/resume, prediction decoding, common box contract, and ONNX feasibility. No pretrained weight was downloaded.
- **Actual architecture export:** all five semantic architectures and both detectors complete ONNX checker and ONNX Runtime comparison locally. ONNX files remain ignored external artifacts. Fast-SCNN uses the bounded dynamo fallback; production/pretrained and Jetson equivalence remain unmeasured.
- **OOD/calibration/statistics:** MSP, entropy, MaxLogit, Energy, AUROC, AP, FPR95, three development-only threshold policies, valid/ignored regions, per-source summaries, deterministic bootstrap, component metrics, scalar temperature fitting, NLL, ECE, and Brier are locally executable. Raw OOD scores are not probabilities.
- **HPO and recovery:** the actual two-model/four-trial mini HPO supports deterministic identity, hard gates, Pareto reporting, interrupted-trial resume, failed-trial isolation, and human-only final promotion. Separate actual semantic and detector campaign interruption/resume probes pass; temporal state is restored mid-sequence.
- **Full video/UI:** a six-frame actual random-weight semantic + feature-head + YOLO path completes score maps, components, contextual risk, temporal tracking, missed-frame handling, one isolated detector failure, overlays, lineage, deterministic replay, and Streamlit headless execution. UI time is excluded from latency.
- **Data lifecycle:** authoritative path-free catalog, task contracts, quality/EDA, transformations, optional class-aware crop, rare-class sampling, source-aware loader, resumable acquisition, retry/range/hash/atomic promotion, generator resume, and slow-destination copy are locally tested with fixtures.
- **Lineage:** every closure receipt records config identity, exact dependency artifact hashes, output hashes, and maturity. Real file corruption and config-identity mismatch are detected; only transitive consumers are invalidated and unrelated detection evidence is preserved.
- **Evidence packages:** assistant review, thesis figure, data-lifecycle audit, and Colab-readiness ZIPs are generated under `.local/EG-LOCAL-COMPLETE/reporting/`. Synthetic outputs are visibly `NON-SCIENTIFIC PIPELINE VALIDATION`.
- **Protected boundaries:** no Drive access/mutation, real-data download, pretrained-weight download, official-val access, Fishyscapes access, Lost & Found opening, SMIYC access, CUDA claim, TensorRT engine, Jetson run, or scientific model selection occurred.
- **Remote gates:** normal Python 3.10/3.11 CI and the expanded Linux x86 actual-codepath closure workflow must pass on the final pushed revision.
- **Next execution boundary:** approved real-data acquisition/staging and exact-commit Colab throughput/smoke execution; subsequent scientific choices remain human-owned.

Planned, implemented, locally tested, remotely verified, Colab measured, Jetson measured,
scientifically measured, and human accepted remain distinct states.
