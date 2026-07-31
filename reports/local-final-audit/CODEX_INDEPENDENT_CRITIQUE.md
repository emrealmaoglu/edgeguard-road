# Codex independent critique

1. **Ten weakest parts:** no real train-data rerun, no pretrained comparison, no real BDD path, no Static OOD measurement, no frozen-holdout result, no CUDA campaign, no TensorRT parser result, no Jetson sustained result, no real temporal-domain result, and no examiner-reviewed statistical protocol.
2. **Unnecessary complexity:** the campaign graph and evidence machinery are close to the maximum justified complexity; new orchestration layers should not be added.
3. **Technically weak choices:** no accepted model or dataset is rejected locally, but random-weight fixture behavior cannot validate scientific suitability.
4. **Scope:** the complete roadmap is broad for a thesis. Semantic + zero-shot OOD + calibration + one anomaly head + one detector + lightweight temporal + one Jetson profile is the defensible core.
5. **Essential claim components:** leakage-safe data roles, reproducible semantic baseline, OOD score comparison, calibration, contextual/temporal false-alarm treatment, and measured edge trade-off.
6. **Stretch goals:** second detector HPO, depth, advanced tracking, INT8, Mapillary, and learned fusion.
7. **Likely examiner criticism:** selection bias, too many loosely measured modules, synthetic plumbing confused with evidence, incomplete license provenance, and insufficient uncertainty around limited seeds.
8. **Potential invalidators:** tuning on official val/holdout/sealed data, inconsistent preprocessing, hidden sample overlap, incomparable budgets, partial checkpoints, and silent backend numerical drift.
9. **Leakage risks:** Cityscapes sequence/group leakage, calibration reuse, repeated Lost & Found inspection, synthetic generator overlap, and post-hoc threshold selection.
10. **Jetson risks:** unsupported operators, memory pressure, thermal throttling, power-mode drift, preprocessing bottlenecks, and desktop/Jetson package mismatch.
11. **Additional local work found:** actual detector construction/training/export, real semantic feature taps, bounded HPO resume, acquisition retry/resume, compact notebook diagnostics, and selective lineage invalidation.
12. **Intentionally not implemented:** real-data measurement, CUDA claims, TensorRT engines, Jetson profiles, sealed evaluation, pretrained downloads, and scientific promotion because they require access, hardware, or human scientific authorization.
