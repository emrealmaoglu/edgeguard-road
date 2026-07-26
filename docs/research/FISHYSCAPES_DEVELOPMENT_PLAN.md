# Fishyscapes Public-Development Foundation

## Scope and role record

This foundation is restricted to public validation/development data. It does
not access SMIYC, select thresholds, train or fine-tune PIDNet, perform
calibration, or add context, morphology, or temporal processing.

| Dataset | Project role | Current state | Allowed use | Redistribution |
| --- | --- | --- | --- | --- |
| Fishyscapes Lost & Found validation | `ood_development` | Adapter and metrics ready for manually supplied data; real data not accessed | Pixel-level AP and FPR95 development evaluation with higher scores meaning more anomalous | Images and derived visuals remain external and are not redistributed |
| Fishyscapes Static validation | `ood_development` | Generation preparation only; no generation performed | Future development evaluation generated from legally available Cityscapes inputs using a reviewed official generator pin | Generated images remain external and are not redistributed |

The official Fishyscapes page states that Lost & Found validation is public and
that FS Static cannot be distributed as a ZIP because it depends on Cityscapes;
the official integration generates it from Cityscapes inputs:
<https://fishyscapes.com/dataset>. The public annotation record describes 100
Lost & Found validation images: <https://zenodo.org/records/6511227>.

The annotation record and underlying Lost & Found images have distinct
provenance. The annotation license must not be claimed to cover the underlying
images automatically.

## Implemented local contract

`edgeguard.data.fishyscapes` supports only the manually prepared public Lost &
Found validation layout:

```text
<external-root>/
  fishyscapes_lostandfound/
    0000_<scene>_<sequence>_<frame>_labels.png
  lostandfound/leftImg8bit/
    train|test/<scene>/<scene>_<sequence>_<frame>_leftImg8bit.png
```

The adapter performs deterministic annotation sorting, exact train/test image
pairing, RGB decoding, image-mask geometry validation, and native mask-value
validation. The project contract is:

- `0`: ID pixel
- `1`: anomaly pixel
- `255`: ignored/void pixel

The deterministic root-free manifest records relative paths, SHA-256 values,
shape, anomaly/ignore pixel counts, `dataset_role=ood_development`,
`source_mode=manual_only`, and `higher_means_more_anomalous`.

Pixel AP and FPR95 are implemented with NumPy. They operate over all valid
pixels and do not choose or persist a deployment threshold. Undefined cases are
explicit: AP and FPR95 are `null` with no anomaly pixels; FPR95 is `null` with
no negative ID pixels.

## Exact manual acquisition steps

No automatic download or restricted-data access is implemented.

### Lost & Found validation

1. Human reviews and accepts the underlying Lost & Found image terms from the
   original dataset provider.
2. Human downloads the public Fishyscapes Lost & Found validation annotation
   archive only from the official Fishyscapes-linked Zenodo record
   `6511227` and obtains the required underlying Lost & Found images from their
   official provider.
3. Human records for each archive: source URL, filename, access date, byte size,
   SHA-256, and applicable license/terms. The published annotation MD5 may be
   recorded as additional evidence but does not replace SHA-256.
4. Human extracts both sources outside the repository into the layout above.
   No image, mask, generated visual, archive, or manifest containing absolute
   roots is added to Git.
5. Before a real evaluation, run the adapter manifest builder against the
   external root, confirm the expected 100 public-validation pairs, inspect any
   zero-anomaly sample explicitly, and have the human approve the resulting
   manifest hash.

### FS Static validation

1. Reuse only a human-authorized external Cityscapes validation root; do not
   download or redistribute Cityscapes through this project.
2. Before generator execution, human reviews and pins the exact official
   Fishyscapes/BDL generator source commit, its license, dependencies, and the
   FS Static version to generate.
3. Human supplies any additional legally available generator inputs outside Git
   and records their source identities and hashes.
4. Generation runs outside Git with outputs in external storage. Generated
   images are not redistributed.
5. Record the Cityscapes manifest hash, generator commit/config, input hashes,
   generated manifest hash, and access/generation date before any evaluation.

This task stops before steps requiring data acquisition or FS Static generation.
A real Fishyscapes run remains blocked on those human actions and manifest
approval.
