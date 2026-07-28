# Dataset configs

`ontology_v1.yaml` is the compact, path-free provisional source of truth for the four
separate project namespaces and the reviewed BDD100K detection-name mappings. Local
validation does not freeze it; human acceptance is still required. It does not
authorize acquisition, create a dataset split, or provide a private storage root.

Runtime dataset roots remain external. Dataset-specific acquisition and split
configs are deferred until their human access and terms gates close.

The active semantic portfolio is recorded in `docs/dataset_cards/catalog.json` and
validated with `scripts/catalog_datasets.py`. `semantic_ontology_v2.yaml` is the active
Cityscapes19 contract for Cityscapes/BDD/IDD and evaluation datasets. The A2D2 mapping
is a phase-two proposal only; it is intentionally not a training adapter or frozen
scientific ontology decision.
