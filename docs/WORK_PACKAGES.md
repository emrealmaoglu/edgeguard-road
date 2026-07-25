# Work Packages

| WP | Purpose | Depends on | Primary deliverable | Gate |
| --- | --- | --- | --- | --- |
| WP-00 | Governance and durable agent memory | — | Charter, rules, state, protocol, decisions | G0: scope and authority are unambiguous |
| WP-01 | CPU-only repository foundation | WP-00 | Package, config, dummy smoke, tests, CI | G1: install, lint, type-check, tests, doctor, smoke pass |
| WP-02 | Measured environment inventory | WP-01 partial | Local/Colab/Jetson environment matrix | G2: machine-produced inventories exist |
| WP-03 | Dataset access and lineage | WP-00 | Role/license matrices and checksummed manifests | G3: roles and sealed-test boundary are locked |
| WP-04 | Dataset adapters and QA | WP-01, WP-03 | Common sample contract and adapters | G4: adapters pass geometry/label/leakage audits |
| WP-05 | Model screening and pretrained baseline | WP-01, WP-02 | Primary/comparison choice and early ONNX evidence | G5: raw logits and baseline path work |
| WP-06 | Colab execution, training, and bounded HPO | G1, G3–G5 | Reproducible runs and frozen model decision | G6: resume/provenance/budget controls pass |
| WP-07 | Calibration and OOD scoring | WP-05, optionally WP-06 | MSP/MaxLogit/Entropy/Energy evaluator | G7: direction, numerical safety, split policy pass |
| WP-08 | Context, components, and temporal decisions | WP-07 | Stable frame/component/event pipeline | G8: ablations, state reset, logging pass |
| WP-09 | Development and sealed evaluation | WP-07, WP-08 | Reproducible metric and failure reports | G9: dev/final separation and traceability pass |
| WP-10 | ONNX/TensorRT and equivalence | WP-05 spike; WP-06/07 final | Export, engine manifest, equivalence report | G10: task-level FP16 equivalence accepted |
| WP-11 | Jetson pipeline and benchmark | WP-02, WP-10 | End-to-end benchmark and telemetry | G11: sustained latency/resource evidence exists |
| WP-12 | Offline demo and fallback material | WP-08, WP-11 | HUD, launcher, recorded/static fallbacks | G12: offline demo and disaster chain pass |
| WP-13 | Thesis evidence and closure | Continuous | Cards, raw results, figures, archive | G13: every claim is reproducible and attributable |

Human approval is required to close every gate and promote every artifact.
