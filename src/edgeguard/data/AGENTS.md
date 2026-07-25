# Data subsystem rules

- Preserve every dataset's native labels and record explicit mappings.
- Missing or unavailable labels are not background.
- Split sequences atomically; sequence leakage across roles is prohibited.
- Resize categorical masks with nearest-neighbor interpolation only.
- Test RGB/BGR order, image/mask geometry, label IDs, duplicates, and corrupt files.
- Do not download or expose dataset samples without human-approved license and role.
