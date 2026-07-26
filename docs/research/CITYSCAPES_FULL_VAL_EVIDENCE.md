# Cityscapes Full-Validation Evidence — 2026-07-26

## Claim boundary

This record covers the **EdgeGuard-Road single-scale PIDNet-S Cityscapes-val
evaluation**. It is not guaranteed to reproduce the official PIDNet paper
protocol. The reported runtime is end-to-end evaluation-pipeline timing: image
loading, preprocessing, model execution, alignment, semantic and uncertainty
processing, and artifact work inside the runner. It is not pure inference
latency and must not be converted into Jetson FPS.

No OOD performance, threshold, calibration, or anomaly-probability claim is
made from this Cityscapes ID-development run.

## External evidence identity

- External location: `~/edgeguard-data/evidence/cityscapes/full-val/2026-07-26/edgeguard-cityscapes-eval.zip`
- ZIP SHA-256: `756abf1a983b8eed11b22f0c10b3cabf093d6e614a4bec2a6d223c41202132b7`
- ZIP size: 19,966,292 bytes
- Archive entries: 43
- CRC result: all entries passed
- Artifact file-map SHA-256: `a10b66ae0108c838c0f9c76970af47108d2e9629d2602229fc6bca6591c83d7d`
- Internal verification: all 42 files named by `artifact_manifest.json`
  matched their recorded SHA-256 values
- Dataset manifest SHA-256:
  `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`
- Selection manifest SHA-256:
  `93efa0d5a4ab9c91c0f8ccf3865252646a22f4d41605032d003c6a524091d2a9`
- Run ID: `7505687f-1180-4f27-9ab3-09b50dd81222`
- Config SHA-256:
  `b576c2445d51b13ea1eab7992ea26d458b3a6ae0d7add5577bef46dbce32f3aa`
- Experiment fingerprint:
  `cf09f266564db661f29bcdac74592ed78d38fbfed6ff983d82d12be4621f8b77`

The ZIP remains external. No archive member, PNG, tensor, model, dataset, or
generated output was copied into Git.

## Source and execution provenance

- Git commit: `aa8803e8060af8cd704f81fb7c6903d0d48e2a6e`
- Git state: `clean`
- Git dirty: `false`
- Device: CUDA
- Python: 3.12.13
- PyTorch: `2.11.0+cu128`
- System: Linux x86_64
- Selected/successful/failed: 500 / 500 / 0
- Model input: `[1,3,1024,2048]`
- Native logits: `[1,19,128,256]`
- Aligned logits: `[1,19,1024,2048]`

## Measured results

- Cityscapes mIoU: `0.7875813077220126`
- Pixel accuracy: `0.9619008903101843`
- Mean class accuracy: `0.8618737663500519`
- Evaluated pixels: 917,018,489
- Ignored pixels: 131,557,511
- Total pixels: 1,048,576,000
- End-to-end evaluation-pipeline time: `827.00028256` seconds
- Mean end-to-end sample time: `1.6539497549533844` seconds
- Peak PyTorch CUDA allocated memory: 217,726,976 bytes

The confusion-matrix sum equals the evaluated-pixel count. Evaluated plus
ignored pixels equals `500 × 1024 × 2048`. MSP, predictive entropy, MaxLogit,
and Energy each contain 1,048,576,000 finite score values and retain the
direction “higher means more anomalous.”

## Independent verification performed locally

The external archive was independently checked for:

- exact ZIP SHA-256 and byte size;
- CRC integrity, required files, safe relative member names, and unique paths;
- exact internal file-hash map and its canonical SHA-256;
- independently recomputed dataset and selection manifest hashes;
- matching manifest references in run and artifact metadata;
- exact commit, clean Git state, 500 successes, and empty failures file;
- semantic pixel-count consistency and finite semantic/score values;
- absence of absolute user/Drive paths and secret-like values in textual members.

All checks passed.

## Non-blocking selection provenance defect

The completed run evaluated all 500 deterministically sorted samples, but its
selection manifest recorded `city_round_robin_v1`. The truthful strategy for
that mode is `all_sorted_v1`. In addition, the original visual rule selected
the first five evaluation samples, so all five visual groups came from
Frankfurt.

This defect does not alter the evaluated samples, their order, semantic metrics,
or score summaries. It affects only the strategy label and visual diversity.
The follow-up implementation records `all_sorted_v1` for `--all`, preserves
`city_round_robin_v1` for subset size, records
`subset_manifest_preserved_v1` for explicit manifests, and selects visual
samples independently with deterministic city round-robin. The 500-image
campaign was not rerun.
