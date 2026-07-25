# Runtime artifacts

Actual run outputs are ignored by Git. This directory stores only the artifact
policy and small schema examples.

Every promoted artifact must be addressed by SHA-256 and linked to source state,
validated config, dataset/model provenance, environment, and source run. `unborn` or
`dirty` Git states are development evidence and are not eligible for promotion.
