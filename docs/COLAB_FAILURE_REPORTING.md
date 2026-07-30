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
