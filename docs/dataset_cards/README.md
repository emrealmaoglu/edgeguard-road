# Dataset cards

`catalog.json` is the machine-readable source of truth for the active semantic
portfolio. Validate it and regenerate the review table with:

```bash
python3 scripts/catalog_datasets.py \
  --write-markdown docs/dataset_cards/catalog.md
```

Official package facts do not imply local acquisition. Only an external hash-bound
receipt can mark a runtime dataset ready. Licensed images, masks and archives never
enter Git.
