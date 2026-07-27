# Experiment Protocol Invariants

## Dataset roles and leakage

- Cityscapes train is inspected before deterministic group-level
  `train_fit`/`train_select`/`train_calibration` candidates are proposed; no fixed
  percentage is assumed in advance.
- Official Cityscapes val is excluded from per-trial tuning and used only for frozen
  final semantic confirmation.
- Project synthetic anomaly data trains OOD methods. Fishyscapes Static is OOD
  development/HPO. Full Fishyscapes Lost & Found is a one-time frozen holdout. SMIYC
  is sealed final and unavailable to automated development.
- Video frames remain in one sequence/group role. Missing labels are never converted
  silently to background.

## Fair model comparison

- The five semantic candidates share ontology, split, seed policy, augmentation
  baseline, effective global batch, `512×1024` HPO crop, evaluator, and metric rules.
- The sequence is smoke → short screening → early export feasibility → top-three
  medium training → top-two HPO → three final project-owned runs.
- Screening, HPO, final confirmation, export, and Jetson results remain separate.
- Resolution is a separate controlled ablation (`512×1024`, `768×1536`, optional
  `1024×2048` evaluation), not an initial HPO dimension.

## Scores, calibration, and claims

- Semantic calibration and OOD score normalization/thresholding are distinct.
- All anomaly scores use “higher means more anomalous” and are not anomaly
  probabilities.
- Trainable feature taps and BCE+Dice are candidates, selected only after the winning
  semantic model and export constraints are known.
- Threshold, holdout, and sealed-test protocols are frozen by the human before use.

## Compute and interruption safety

- Every real run records accelerator, precision, samples, optimizer steps, effective
  batch, wall time, accelerator-hours, interruption overhead, and failed-run compute.
- FP32 is the correctness baseline; FP16/BF16 require finite loss/gradient evidence.
- Expensive Colab jobs persist resolved config, Git/data/model hashes, epoch/step,
  optimizer, scheduler, AMP scaler, recoverable seeds, best/last metrics, trial ID,
  environment, and logs. Resume refuses incompatible identity or overwrite.
- Training reads active data from Colab ephemeral storage, not repeated mounted-Drive
  sample I/O. Final sync is atomic and hash-verified.

## Provenance and status

- Run IDs are unique; config hashes and experiment fingerprints exclude volatile
  metadata.
- Promotable evidence records a clean Git commit and external artifact identities.
- Status vocabulary is: `planned`, `implemented`, `locally_tested`,
  `colab_measured`, `jetson_measured`, `human_accepted`, `blocked`, `failed`.
- Negative results and failed export evidence remain in the audit trail.

## Benchmark versus demo

- Benchmark mode disables UI/recording, fixes input, warms up, and records stage and
  end-to-end timing with telemetry.
- Streamlit rendering and video encode/decode time are reported separately and never
  relabeled as model or Jetson inference latency.
