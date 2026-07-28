# Runtime artifacts

Actual run outputs are ignored by Git. This directory stores only the artifact
policy and small schema examples.

Every promoted artifact must be addressed by SHA-256 and linked to source state,
validated config, dataset/model provenance, environment, and source run. `unborn` or
`dirty` Git states are development evidence and are not eligible for promotion.

`external_pairs.example.json` records exact vendor-relative files and official
submission names without guessing a downloaded layout. `sealed_release.example.json`
is copied outside Git and filled only after model, preprocessing, checkpoint, and
external-manifest freeze. Changing either hash invalidates the release.
`pretrained_initialization.example.json` prevents an upstream config from silently
downloading or reusing a segmentation checkpoint: only a hash-verified,
classification-task initialization approved for the named finalist is accepted.
The external-pairs record must also state the vendor encoding. WildDash 2 requires
regular Cityscapes label IDs, never internal train IDs; MUSES/KITTI use the encoding
documented by the exact acquired release rather than an inferred default.
Scientific commands also append `run_ledger.jsonl` outside Git. Each row binds the
operation, exact result payload hash, UTC time, and Git commit/dirty state; the file is
append-only evidence rather than a mutable registry.
