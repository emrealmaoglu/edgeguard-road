# Colab failure reporting

Both delivery notebooks install one persistent exception reporter after the exact Git
checkout is known. Clone/update failures use a smaller standard-library bootstrap report;
all later unhandled Python and failed subprocess errors use the full reporter.

## Durable locations

```text
MyDrive/EdgeGuard/failures/
├── bootstrap/<failure-id>/
│   ├── failure.json
│   └── failure-report.zip
├── data-preflight/<commit>/<failure-id>/
└── semantic-first-cs-idd-v1/<commit>/<failure-id>/
```

Each full report contains the notebook and stage identity, UTC time, project commit,
exception type/message/traceback, Python/platform identity, disk usage, bounded context,
and hashes for included small diagnostic files. Dataset bytes, checkpoints, ONNX/TensorRT
files, environment variables and files over the configured limits are excluded. Common
token/password/API-key forms are redacted from both JSON and archived text logs.

Every post-bootstrap subprocess is streamed through a project-owned command logger. Its
redacted live output is also written under `/content/edgeguard-command-logs/` and included
in the bounded failure ZIP. A non-zero exit therefore reports the actual final output lines,
not only a generic `CalledProcessError`.

Notebook subprocesses execute with the exact checkout as their working directory and with
the checkout's `src/` on `PYTHONPATH`. Active CLI config defaults are also anchored to the
repository containing each script, so Colab's initial `/content` directory cannot redirect
relative configuration lookup.

Archive inventory hashing is observational. If mounted Drive interrupts a large sequential
read, inventory records `hash_status=read_error` and continues. This does not waive archive
integrity: preparation copies the archive to `/content` with bounded retries, and the
dataset-specific preparation gate still verifies the complete local archive against its
pinned digest before extraction.

The failure folder is append-only. `LATEST.txt` is only a convenience pointer to the
newest immutable report. Its value is path-validated before use.

## What to do after an error

1. Do not change library versions or repeat speculative installation commands.
2. Note the stage printed in `EDGEGUARD FAILURE REPORT`.
3. In the notebook settings cell set `DOWNLOAD_LATEST_FAILURE_REPORT=True`.
4. Run the notebook's final output/download cell manually. It downloads the latest
   `failure-report.zip`; the same package remains durable in Drive.
5. Provide that ZIP for diagnosis. Preserve the failed campaign directory until the root
   cause is understood.

Runtime-cascade failures also preserve compatibility receipts and small install logs at
`campaigns/<campaign>/<commit>/runtime-compatibility/`. A report is engineering evidence,
not permission to weaken hashes, scientific data gates or model acceptance thresholds.
