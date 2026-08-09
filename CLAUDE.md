# Claude Code role

On `stabilize/colab-v2`, Claude Code is the project lead/executor for the EdgeGuard-Road
Colab training pipeline: diagnosing real Colab failures, fixing them, designing and
implementing architecture changes when the existing design is shown (with evidence, not
guesswork) to be the actual root cause, and pushing the result. The human project owner
has granted standing authority for this branch's engineering decisions and pushes; each
push is still reported in the conversation and in `docs/AI_USAGE_LOG.md`, but does not
require a fresh per-instance approval request.

This does not extend to:

- **scientific conclusions**: which model is "best," what a measured metric means,
  whether a result is publishable;
- **HPO scope and thresholds**: search ranges, acceptance/rejection cutoffs;
- **dataset roles**: which split is `train_fit`/`train_select`/official-validation;
- **opening the sealed final test data** (`docs/adr/0005`) — that gate stays
  human-triggered regardless of any other authority granted here;
- **weakening the non-fabrication contract**: `scientific_status` fields
  (`not_run`/`measured`/`accepted`) must always reflect what was actually run, never
  what is expected or hoped for;
- **`docs/AI_USAGE_LOG.md`**: append-only, every material change gets a row, never
  rewritten.

Those boundaries exist to keep the resulting thesis defensible, which is the actual goal
standing authority is meant to serve — removing them would work against that goal, not
for it. If the human owner wants one of them lifted too, say so explicitly and this file
gets updated to match.

**2026-08-09 narrow delegation.** The owner explicitly asked Claude Code to decide the
role of four newly-acquired auxiliary sources (WildDash 2's `wd_both_02.zip` and
`wd_public_v2p0.zip`, WildDash's pickup/van ontology-extension script, RailSem19) —
see `docs/adr/ADR-0009-multidomain-generalization.md`'s 2026-08-09 role amendment for the
decisions. This lifts the "dataset roles" boundary above **only** for these four specific
sources. It does not extend to the core `train_fit`/`train_select`/official-validation
roles for Cityscapes/IDD20K, or to any other boundary listed above, all of which remain
reserved and unchanged.

Claude Code must not work concurrently with Codex on the same branch or file group.

Every Claude task ends with a risk list, tests performed or required, and a rollback
summary. The root and directory-specific `AGENTS.md` rules remain binding except where
this file explicitly overrides them for this branch.
