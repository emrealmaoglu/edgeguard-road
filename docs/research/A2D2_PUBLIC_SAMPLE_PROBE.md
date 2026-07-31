# A2D2 public sample probe

This is engineering evidence only. It is not a scientific dataset audit, training run,
validation result or model metric. The downloaded files remain under ignored
`data/cache/` and are not committed.

## Verified on 2026-07-28

- Provider: official public A2D2 AWS S3 extracted dataset.
- License file: CC BY-ND 4.0, 117 bytes,
  SHA-256 `7dcd2819b7724bbb95b2b180f45f5c67c7f8aa2d120c97a1e192eaa025f2d013`.
- Native class list: 55 RGB colors, 1,704 bytes,
  SHA-256 `985606925addb513e3d4164e1bb2c6b2b46f1bb23dc44f8fcb922d8af99ad6d1`.
- Front-center RGB sample: 3,817,263 bytes,
  SHA-256 `1ea14036742fe8b453add67ce6a09b7f306de21d2a0906c90568e1e7b7ec15cc`.
- Matching RGB label: 71,403 bytes,
  SHA-256 `43a7a453fe632228d181f06673fc7c9adfd16f19060f70ef31161138d2c89666`.
- Image and mask geometry: `1208 x 1920`, RGB/RGB.
- Mask colors present in this sample: 21 of the 55 declared colors; unknown colors: 0.
- Exact-only Cityscapes19 proposal retained 96.7306% of pixels; 75,829 pixels became
  `ignore=255`. This ratio is sample-specific and must not be generalized to A2D2.

The reviewed phase-two mapping maps 31 source colors and explicitly ignores 24. A2D2
cannot enter training until full-corpus audit, sequence grouping, cross-dataset duplicate
checks, usable-pixel/class coverage, mapping review and split freeze are complete.

Reproduce the bounded probe:

```bash
python3 scripts/catalog_datasets.py \
  --download-public-sample a2d2 \
  --output-root data/cache/a2d2_public_probe \
  --maximum-sample-bytes 8000000
```
