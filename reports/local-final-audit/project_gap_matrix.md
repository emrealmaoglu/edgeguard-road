# EdgeGuard-Road local final audit

**NON-SCIENTIFIC PIPELINE VALIDATION.** Maturity is engineering evidence, not model ranking.

| ID | Capability | Before | After | Remaining boundary |
| --- | --- | --- | --- | --- |
| EG-CAP-01 | research hypotheses | contract_only | contract_only | human scientific review |
| EG-CAP-02 | dataset acquisition | surrogate_validated | local_end_to_end_validated | human scientific review |
| EG-CAP-03 | licensing and provenance | contract_only | requires_real_data | real data/platform measurement |
| EG-CAP-04 | dataset inventory | contract_only | requires_real_data | real data/platform measurement |
| EG-CAP-05 | data quality | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-06 | exploratory data analysis | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-07 | ontology and label mapping | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-08 | leakage-safe splitting | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-09 | preprocessing | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-10 | augmentation | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-11 | sampling and imbalance | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-12 | dataloading and I/O | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-13 | semantic training | surrogate_validated | real_codepath_validated | human scientific review |
| EG-CAP-14 | detector training | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-15 | OOD development | surrogate_validated | local_end_to_end_validated | human scientific review |
| EG-CAP-16 | calibration | surrogate_validated | local_end_to_end_validated | human scientific review |
| EG-CAP-17 | trainable anomaly learning | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-18 | contextual risk | surrogate_validated | local_end_to_end_validated | human scientific review |
| EG-CAP-19 | temporal fusion | surrogate_validated | local_end_to_end_validated | human scientific review |
| EG-CAP-20 | HPO and promotion | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-21 | statistical evaluation | contract_only | local_end_to_end_validated | human scientific review |
| EG-CAP-22 | error analysis | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-23 | model export | surrogate_validated | real_codepath_validated | human scientific review |
| EG-CAP-24 | deployment packaging | contract_only | contract_only | human scientific review |
| EG-CAP-25 | Colab observability | contract_only | real_codepath_validated | human scientific review |
| EG-CAP-26 | Jetson and TensorRT readiness | contract_only | requires_jetson | real data/platform measurement |
| EG-CAP-27 | evidence and thesis reporting | contract_only | local_end_to_end_validated | human scientific review |
