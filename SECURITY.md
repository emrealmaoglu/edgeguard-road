# Security and Safety Policy

EdgeGuard-Road is an offline academic prototype and must not be used to control a
vehicle or make safety-critical decisions.

## Sensitive material

- Never commit credentials, tokens, SSH keys, `.env` files, private dataset links, or
  personally identifying data.
- Use `.env.example` only for empty variable-name documentation.
- Revoke and rotate any credential that is accidentally exposed; do not preserve it
  in issue text or logs.

## Device operations

Automated agents may not connect to Jetson or run privileged/system-changing
commands. JetPack, CUDA, TensorRT, storage, power, and networking changes require a
human operator and a separate reviewed procedure.

## Reporting

Report repository security concerns privately to the project owner. Do not include
secrets or restricted dataset samples in a report.
