# Google Drive layout audit — 2026-07-30

This is a read-only inventory of the connected `MyDrive/EdgeGuard` folder. No Drive file
was moved, renamed, copied, deleted, or uploaded during the audit. Drive file IDs and
temporary download URLs are intentionally excluded from Git.

## Observed structure

```text
EdgeGuard/
├── archives/                         # currently empty
├── campaigns/
│   └── EG-REAL-001/                  # existing legacy campaign; preserve unchanged
├── checkpoints/                      # currently empty
├── datasets/
│   └── cityscapes/fine/
│       ├── bundles/
│       │   ├── cityscapes-fine-903a9059cb8c-988632a4f631.tar.gz
│       │   └── ...tar.gz.receipt.json
│       └── v1/                       # prepared train tree and preparation receipt
├── exports/                          # currently empty
├── manifests/
│   └── cityscapes/fine/v1/           # real audit, class, group and split evidence
├── private_inputs/
│   ├── leftImg8bit_trainvaltest.zip
│   ├── gtFine_trainvaltest.zip
│   ├── idd-20k-I.tar.gz
│   ├── idd-20k-II.tar.gz
│   ├── bdd100k.zip                   # Kaggle mirror; provisional only
│   └── PIDNet_S_Cityscapes_val.pt
└── reports/                          # currently empty
```

The real Cityscapes evidence reports 2,975 training images, 1,885 groups, all 19
Cityscapes train IDs, and a 6,987,575,819-byte reusable training bundle containing 5,950
files. The bundle receipt binds dataset manifest
`903a9059cb8c…13377cedc`, split manifest `988632a4f631…ea4bbc1f`, and bundle SHA-256
`a96bb496c33c…e21eea04`.

The second read-only inspection after upload found all requested source archives in
`private_inputs/`. Drive-reported sizes are 11,592,327,197 bytes for Cityscapes images,
252,567,705 bytes for Cityscapes labels, 19,895,565,205 and 5,912,499,808 bytes for IDD
Part I/II, and 8,166,127,525 bytes for the Kaggle BDD ZIP. These are presence/size facts,
not archive-integrity evidence; the preflight notebook must still compute and compare
hashes before preparation.

The existing semantic compatibility receipt belongs to historical project commit
`a786522…76984`, selected the isolated Python 3.11 path, and passed five-model checkpoint
reload on an A100. It is valid historical engineering evidence but cannot authorize a
new commit; the new notebook must create a fresh receipt for the reviewed commit.

## Additive target — no migration

Existing folders and archives remain where they are. The revised preflight searches
`private_inputs/` as an accepted immutable input location and creates only missing
additive output roots:

```text
EdgeGuard/
├── archives/
│   ├── bdd100k/       # two official semantic packages when acquired
│   ├── idd20k/        # official Part I and Part II
│   └── ...            # final-only datasets remain gated
├── bundles/           # new BDD/IDD canonical bundles
├── campaigns/
│   ├── EG-REAL-001/   # preserved
│   └── semantic-first-cs-idd-v1/<commit>/
├── downloads/         # bounded review ZIPs
├── quarantine/kaggle/bdd100k/
└── source/            # optional immutable code handoff
```

Cityscapes is not copied or recompressed. The staging command prefers a new canonical
bundle when present; otherwise it accepts only the exact pinned legacy Cityscapes bundle
and receipt above. Any size, manifest, split, filename, file-count, or SHA-256 drift
fails closed. BDD and IDD continue through the new canonical bundle contract.

The uploaded `private_inputs/bdd100k.zip` is inventoried as a provenance-limited
engineering package and produces
`bdd100k.kaggle_mirror.prepared.tar`. It cannot overwrite or masquerade as the future
official `bdd100k.prepared.tar`. It may be staged for audit, but it cannot produce a
scientific manifest and is excluded from HPO/model selection. The active scientific
source set is therefore Cityscapes + IDD20K until official BDD packages exist.
