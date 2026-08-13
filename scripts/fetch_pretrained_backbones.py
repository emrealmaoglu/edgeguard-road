"""Download the ImageNet-classification backbones the committed manifests declare.

The manifests in `configs/pretrained/` are the source of truth: they are committed with
a fixed `checkpoint_path`, `checkpoint_sha256` and `access_date`, and this script only
materialises the bytes those manifests already describe. It never rewrites a manifest.

That direction matters. `pretrained_manifest_sha256` is one of the fields baked into a
training run's immutable identity (`mmseg_runtime.compute_run_identity`), so a manifest
regenerated per session -- with a moving `access_date`, say -- would change its own hash
every session and silently invalidate every Drive checkpoint. Committing the manifest and
verifying against it keeps that hash stable for the life of the campaign.

Only ImageNet *classification* backbones are permitted (`source_task` is checked by
`mmseg_runtime._verified_pretrained_checkpoint`). Cityscapes-pretrained segmentation
weights would have seen the evaluation data and are not usable as an initialisation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from edgeguard.serialization import canonical_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "configs/pretrained"
_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, staging.open("wb") as sink:  # noqa: S310
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            sink.write(block)
    staging.replace(destination)


def materialise(manifest_path: Path) -> dict[str, object]:
    """Ensure one manifest's checkpoint exists locally with its declared hash."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = str(payload["model"])
    expected = str(payload["checkpoint_sha256"])
    destination = Path(str(payload["checkpoint_path"]))
    source_url = str(payload["source_url"])

    if destination.is_file():
        actual = _sha256_file(destination)
        if actual == expected:
            return {"model": model, "status": "already_present", "path": str(destination)}
        # A wrong-hash file is corruption or a changed upstream artifact, never something
        # to train on: drop it and re-fetch, then fail loudly below if it is still wrong.
        destination.unlink()

    _fetch(source_url, destination)
    actual = _sha256_file(destination)
    if actual != expected:
        raise ValueError(
            f"{model}: downloaded checkpoint hash {actual} does not match the committed "
            f"manifest hash {expected}; refusing to use it"
        )
    return {"model": model, "status": "downloaded", "path": str(destination)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=MANIFEST_ROOT,
        help="directory of committed edgeguard_pretrained_initialization manifests",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifests = sorted(args.manifest_root.glob("*.json"))
    if not manifests:
        raise FileNotFoundError(f"no pretrained manifests under {args.manifest_root}")
    results = [materialise(path) for path in manifests]
    print(
        canonical_json(
            {
                "schema_version": "1.0",
                "record_type": "edgeguard_pretrained_backbone_fetch",
                "results": results,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
