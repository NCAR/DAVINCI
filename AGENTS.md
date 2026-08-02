# Account isolation — absolute rule

The active user is `fillmore`.

No agent may, under any circumstances, inspect or operate in another user's
account storage. This prohibition includes reading, listing, searching,
scanning, globbing, traversing, statting, resolving, sourcing, importing,
executing, copying, modifying, or deleting anything under another user's
home, work, scratch, or environment directories. The only exception is the
limited shared-project-data rule below.

In particular:

- Never access `/glade/u/home/<other-user>`, `/glade/work/<other-user>`, or
  `/glade/derecho/scratch/<other-user>`.
- There are no exceptions for `/glade/work/<other-user>`, including
  dependencies, examples, module trees, or paths described as project data.
- Never run a broad search rooted at `/glade`, `/glade/u/home`, `/glade/work`,
  or `/glade/derecho/scratch`.
- Never follow a symlink, environment variable, executable path, scheduler
  record, upstream example, documentation example, or script default into
  another user's account.
- World-readable permissions do not grant agent authorization.
- A path embedded in upstream source or existing project history is data to
  report, not permission to access that path unless it qualifies for the
  shared-project-data exception below.
- Use only `fillmore`-owned account roots, the current workspace, and
  institutionally managed shared/system roots that are not individual user
  accounts.
- Before reading or executing a resolved path, verify that neither the path nor
  its symlink-resolved target belongs to another user.
- Restrict scheduler queries to `fillmore` jobs; never enumerate other users'
  jobs to discover software or paths.

If a required dependency points into another user's account, stop immediately.
Report the dependency without probing it, and require a `fillmore`-owned or
institutionally managed replacement.

### Shared project data exception

- An exact, in-scope source below institutionally managed shared project
  storage such as `/glade/campaign` may be read or copied.
- This exception does not authorize broad searches of project storage,
  modifications to the shared source, or following a symlink into any user's
  home, work, scratch, or environment directory.
- This exception never applies to any `/glade/work/<other-user>` path.

If any cross-account access occurs accidentally, stop all related work and
disclose the exact access to the user. Do not continue investigating inside the
other account.

## Git transport

- Create commits with local `git`.
- Use the repository's configured direct SSH remote for all GitHub fetch, pull,
  push, and ref-update operations.
- Do not require or use GitHub CLI authentication to gate Git commits, pushes,
  force-with-lease updates, or other Git transport operations.
- Never replace the SSH remote with HTTPS, a personal access token, or another
  credential mechanism unless the user explicitly changes this policy.
- Before a remote write, verify that the target remote is an SSH GitHub URL and
  resolve the exact branch and expected old commit.

# AGENTS.md - DAVINCI

Rules for all AI agents working in this repository. **Read `CLAUDE.md` for full project context** — architecture, conventions, gotchas, config patterns, and styling.

## General Rules

- **Read before editing**: Always read a file before modifying it
- **Preserve existing patterns**: Match the style and conventions of surrounding code
- **Non-destructive edits**: Do not remove user data or code unless explicitly requested
- **Keep the tree lean**: No archived binaries, scratch files, or generated artifacts checked in

## Git Workflow

- **NEVER auto commit or push**: Wait for explicit user confirmation
- **Routine commit/push uses plain Git**: When the user explicitly requests a commit and push, use `git` directly on the authorized branch; do not require the GitHub CLI or create a pull request unless the user asks for PR work
- **Work on develop**: Start from and stay on `develop` by default
- **No stray branches**: Do not create feature, codex, topic, or worktree branches unless the user explicitly authorizes that branch or worktree by name
- **Main is user-controlled**: Merge or push to `main` only when the user explicitly requests it
- **Clean up task branches**: After user-approved merge or abandonment, delete local and remote task branches once their contents are no longer needed
- **After merge, return to develop**: Always switch back to `develop` branch

## Cross-Dataset Handoff Convention

This repo uses cross-dataset code reviews and hand-offs. Check for `REVIEW_*.md` or `HANDOFF_*.md` in the repo root at session start.

When writing handoff files, use `REVIEW_<DATASET>.md` or `HANDOFF_<TOPIC>.md` with these sections:

- **Context** — Branch, task, files involved
- **Changes Made** — What was done, with file paths and line links
- **Decisions & Rationale** — Why choices were made (highest-value section)
- **Open Questions / Concerns** — What the next dataset should investigate
- **Suggested Next Steps** — Specific actionable items

Do NOT track handoff files in git — they are ephemeral working artifacts. Delete once the handoff is complete.

## Planning and Implementation

- **Stop after planning**: After a planning session, always stop and wait for user conversation before proceeding to implementation

## Plot Standards Enforcement

- **Shared standards are mandatory**: Before adding or changing a renderer, inspect
  `davinci_monet/plots/style.py`, `davinci_monet/plots/labeling.py`, and the nearest
  existing renderer. A configured NCAR theme does not authorize ad-hoc colors,
  contour levels, titles, units, legends, or tick formatting.
- **AOD has a fixed house style**: Absolute AOD maps must use
  `geosit_aod_levels()`, `get_geosit_aod_cmap()`, and `BoundaryNorm`. Diverging
  colormaps are reserved for signed differences; dense artists are rasterized.
- **Specialized suites must enforce themselves**: A specialized or multi-figure
  renderer must implement `validate_rendered_figures()` when it has domain-specific
  standards. The plotting stage must run that hook before saving and carry its
  JSON-serializable report into inspection and the manifest.
- **Inspection must be functional**: A production inspection preset may not pass
  from filenames or file existence alone when a renderer protocol exists. It must
  require the complete expected product set and a passing protocol report.
- **Test the actual artists and pipeline**: Add unit assertions for norms, colormaps,
  labels, tick rotation, and rasterization; add a `PipelineRunner` integration test;
  and prove that an injected standards violation fails plotting/inspection.
- **Rendered layout is part of the protocol**: Draw the canvas during specialized
  validation and reject clipped or overlapping titles, subtitles, panel headings,
  axis labels, legends, and colorbar ticks. Visually inspect every production
  preview before accepting or copying a suite; a machine-passing manifest alone is
  not visual acceptance.
- **Keep production controls reproducible**: Git-track every scheduled production
  YAML and the code that interprets it. Completed `rNN/aNNN` controls are immutable;
  any scientific or plotting-contract change gets a new `rNN`. Until the exact
  code and config are committed with user approval, label resulting output
  provisional rather than accepted or reproducible.
- **Reuse persisted science for plot-only revisions**: A new label, layout, or
  renderer-contract revision must not reload, pair, or recompute unchanged science.
  Pin `execution.checkpoints.restore_from` to the prior terminal attempt, the
  finalized stage boundary, and its exact receipt SHA-256. Require readiness and
  the manifest to validate and record that upstream checkpoint lineage.

## Quick Validation

```bash
conda activate davinci
pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```
