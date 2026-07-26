# DAVINCI Attempt Execution

## Identity

DAVINCI uses three nested operational identities:

```text
<run.id ending rNN>/aNNN/eNNN
```

- `rNN` is the immutable scientific/config revision in `run.id`.
- `aNNN` is one fresh attempt with one exact config, DAVINCI code-tree hash,
  numerical runtime, and source inventory.
- `eNNN` is one process or PBS job within that attempt. DAVINCI allocates it
  automatically.

An interrupted or failed execution does not require a new attempt. A completed
attempt is closed. Changed config, code, or source identity requires a new
attempt or revision rather than an override.

## Fresh

The configured `execution.attempt_root` must be missing or empty and end in
`aNNN`. `analysis.output_dir` and `analysis.log_dir` must be its `output/` and
`logs/` children.

```bash
conda activate davinci
davinci validate CONTROL.yaml --strict --readiness
davinci run CONTROL.yaml
```

Do not pre-create `output/`, `logs/`, scheduler log files, or PBS stdout beneath
the attempt root. DAVINCI initializes the append-only attempt store before any
pipeline output.

## Resume Plan and Resume

The plan is read-only and validates exact configuration, code, numerical
runtime, source, receipt, and object identities:

```bash
davinci validate CONTROL.yaml --strict --readiness --resume
davinci run CONTROL.yaml --resume-plan
davinci run CONTROL.yaml --resume
```

`restored` means the exact receipt and every referenced object are valid.
`computed` means no receipt exists. `recomputed` includes changed identity,
invalid dependency closure, corrupt bytes, or an operator restart.

To deliberately recompute one stage/item and its downstream closure without
deleting history:

```bash
davinci run CONTROL.yaml --resume --restart-from analyses:aod_basis
```

Every replacement receipt is appended as `rNNN.json`; finalized prior
generations remain immutable.

## Interruption and Crash Behavior

`SIGINT` and `SIGTERM` produce an `interrupted` execution record after the
currently published receipts remain durable. A process that dies without a
terminal record is marked `abandoned` by the next execution. Resume reuses only
finalized, identity-matched, checksum-valid checkpoints.

## PBS

Use the fresh template only for a missing/empty attempt. Use the resume template
for the same incomplete attempt after reviewing `--resume-plan`. Scheduler
stdout and stderr belong under a revision-level `scheduler-logs/` directory,
outside `aNNN`, so PBS cannot make a fresh attempt nonempty before DAVINCI
starts. Render `__SUBMISSION_TAG__` to a unique UTC timestamp or other immutable
submission label so repeated resume jobs never overwrite scheduler logs.

Rendering a PBS file is reversible and does not authorize `qsub`. Submission
requires an explicit user request. Automated tests must never contact PBS.

## Closeout Evidence

Inspect:

- `attempt.json` for exact identities, runtime versions, Git commit, and
  terminal attempt state;
- `executions/eNNN/{started,finished}.json` for scheduler/process history;
- `checkpoints/<stage>/.../rNNN.json` for item lineage and dispositions;
- `state/events.jsonl` for interruption and publication history;
- `output/manifest.json` for final stage/product/completion evidence; and
- required NetCDF, CSV, PDF/PNG, inspection JSON/Markdown/previews.

Production is complete only when `run.completion` passed and the refreshed
manifest records the latest terminal execution. Scheduler completion alone is
not scientific or operational completion.
