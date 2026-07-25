# DAVINCI Pipeline Resumption Plan

**Status:** Approved 2026-07-25
**Date:** 2026-07-25
**Goal:** Resume an interrupted DAVINCI pipeline from the latest valid stage or
stage item without silently changing its scientific or execution identity.

## Outcome

DAVINCI production runs will be able to survive scheduler interruption,
walltime exhaustion, host failure, or process termination without restarting
all completed work. A resumed execution will:

1. validate the attempt identity;
2. inventory every available checkpoint;
3. reconstruct the pipeline context from valid checkpoints;
4. recompute the first missing or invalid item and everything that depends on
   it;
5. record which work was computed and which work was restored; and
6. finish through saved datasets, plots, inspection, completion, and the final
   manifest without an interactive gate.

This is checkpoint/resume support, not a cross-run scientific cache. The first
implementation will keep checkpoint reuse inside one attempt.

## Terminology and Directory Model

Use these terms consistently:

- **Run revision** — the immutable scientific/configuration identity declared
  by `run.id`, ending in `-rNN` for production.
- **Attempt** — one intended pipeline result for a run revision, named `aNNN`.
  An attempt owns its checkpoints and final output.
- **Execution** — one local or scheduler invocation of an attempt, named
  `eNNN`. An interrupted attempt can have multiple executions.
- **Stage** — one ordered pipeline stage such as `load_sources`, `analyses`, or
  `pairing`.
- **Item** — an independently checkpointable unit inside a stage, such as one
  source, analysis node, pair, statistics group, saved product, or plot.

Recommended layout:

```text
<runs-root>/<run.id>/<attempt.id>/
  attempt.json
  executions/
    e001.json
    e002.json
  state/
    events.jsonl
    snapshot.json
  checkpoints/
    load_sources/
      stage.json
      items/<source>.json
    analyses/
      stage.json
      items/<analysis>.json
    ...
  objects/
    sha256/<object-id>/
      summary.json
      chunk-00000.nc
      ...
  output/
  logs/
```

An attempt can be resumed in place because checkpoint and execution
publication is append-only. Finalized checkpoint bytes are never overwritten.
A changed configuration, code identity, or source inventory requires a new
attempt.

## Required Invariants

1. **Fresh and resume are different operations.** A fresh run requires a new,
   empty attempt root. `--resume` requires an existing, incomplete attempt
   root.
2. **Completed attempts are closed.** DAVINCI refuses to resume an attempt
   whose production manifest is completed.
3. **Identity mismatch is a hard stop.** There is no force flag that silently
   accepts changed scientific inputs, configuration, or code.
4. **Only finalized receipts are reusable.** Temporary files, partial object
   collections, running items, and failed items are ignored.
5. **Writes are atomic.** Data are written to a temporary location, flushed,
   checksummed, and renamed before the receipt is finalized.
6. **Restoration precedes computation.** The runner or stage checks for a valid
   checkpoint before invoking expensive work.
7. **Dependencies are explicit.** An item identity includes the identities of
   every upstream item it consumed.
8. **Invalidation is downstream.** Invalidating one item invalidates its
   dependents, not unrelated branches.
9. **Resume provenance is visible.** The final manifest distinguishes
   `computed`, `restored`, `recomputed`, `skipped`, and `failed`.
10. **Production completion is unchanged.** Restored work satisfies a required
    analysis or product only after the same integrity and completion checks as
    newly computed work.
11. **No automatic deletion.** Production checkpoints remain available until
    an explicit cleanup operation removes the attempt.
12. **No interactive execution gate.** PBS executions run through deterministic
    inspection and completion.

## Identity Contract

### Attempt identity

Capture in `attempt.json`:

- run ID and run kind;
- normalized complete configuration hash;
- DAVINCI version and Git commit;
- dirty-source-tree hash, when applicable;
- Python version and relevant numerical-library versions;
- source inventory identity;
- pipeline checkpoint schema version; and
- initial execution timestamp and host.

### Stage-item identity

Each item receipt contains:

- stage name and stage checkpoint version;
- item key;
- normalized relevant configuration;
- upstream checkpoint IDs;
- source file inventory used by the item;
- implementation/code identity;
- output geometry, dimensions, coordinates, variables, dtypes, and time
  coverage;
- content checksums;
- producing execution ID and timestamps; and
- publication state.

Source inventory should initially use canonical path, file size, and
nanosecond modification time, plus any authoritative checksum already
available from the source. Full hashing of multi-terabyte raw collections is
not required for the first implementation. Every checkpointed derived dataset
will receive strong content checksums.

The first implementation should use the conservative whole-source-tree code
hash already used by analysis artifacts. A later compatibility-version scheme
may allow an unrelated plot-code change to preserve a data checkpoint, but
that optimization must not weaken the initial safety contract.

## Storage Contract

Reuse the existing artifact mechanisms instead of introducing an unrelated
serialization stack:

- xarray datasets: chunked NetCDF collections plus a summary and strong
  checksums;
- small structured results: canonical JSON;
- tables: CSV or Parquet only when the consuming stage already has a stable
  table contract;
- plots and inspection products: original files plus per-file checksums;
- configuration/context deltas: canonical JSON; and
- external summary output: persisted Markdown/JSON response metadata, never a
  repeated provider call when the receipt remains valid.

The object store is attempt-local in the first release. Content IDs deduplicate
the same dataset within the attempt. Cross-attempt or cross-run caching is a
separate feature with different retention and provenance requirements.

## Resume Granularity

The top-level stage is not sufficient because several stages contain multiple
expensive independent items. Support both stage and item receipts.

| Stage | Checkpoint unit | State restored |
|---|---|---|
| `load_sources` | each configured source | `SourceData`, normalized dataset, source metadata |
| `analyses` | each analysis DAG node | derived `SourceData`, analysis status, artifact receipts |
| `plot_suites` | whole stage | expanded typed plot configuration |
| `pairing` | each configured pair | paired dataset and pair metadata |
| `statistics` | each pair or descriptive source | canonical metric result |
| `save_results` | each logical saved product | file path, role, and checksum |
| `plotting` | each logical plot specification | all files produced by that plot and their checksums |
| `summary` | whole stage | summary file, response metadata, and input identities |
| `inspection` | whole stage | inspection report and preview receipts |
| `completion` | whole stage | deterministic completion checks |
| `manifest` | terminal rebuild | final aggregation; never used as an upstream data checkpoint |

Within a currently running item, work may still be lost. For example, a single
plot specification that emits 50 mode maps becomes reusable only after all 50
files and its receipt are finalized. Finer renderer-level checkpointing can be
added later without changing the stage/item model.

## Runner and Stage Architecture

### Checkpoint manager

Add a runtime checkpoint manager responsible for:

- opening or creating the attempt;
- allocating monotonically increasing execution IDs;
- computing and validating identities;
- finding valid receipts;
- writing and loading objects;
- atomically finalizing receipts;
- recording state events;
- reconstructing `PipelineContext`; and
- producing a resume plan before computation.

The manager should live under a focused package such as:

```text
davinci_monet/pipeline/checkpoints/
  models.py
  identity.py
  store.py
  manager.py
  codecs.py
```

`PipelineContext` gains a runtime-only checkpoint-manager reference. The
manager itself is not serialized into a checkpoint.

### Stage boundary

Before a stage executes, the runner asks the manager whether the completed
stage receipt and its dependencies are valid. If valid, the stage codec
restores its context delta and `StageResult`. Otherwise the stage executes.
After a successful stage, the runner atomically publishes the stage receipt.

### Item boundary

Data-heavy multi-item stages call a common item API around their existing
loops:

```text
restore valid item
    or
compute item -> serialize object -> finalize item receipt
```

The check must occur before source loading, analysis `analyze_inputs`, pairing,
statistics, rendering, or external summary invocation.

Stage-specific codecs own reconstruction because a generic pickle of
`PipelineContext` would be unsafe, non-portable, and opaque.

### Failure and signals

- A stage exception records a failed execution and leaves finalized prior
  receipts untouched.
- `SIGTERM`, `SIGINT`, and scheduler termination record the execution as
  interrupted when the process receives enough time to run a handler.
- `SIGKILL` may prevent execution finalization, but already finalized item and
  stage receipts remain usable.
- Temporary paths from a dead execution are ignored and may be reported by
  status/cleanup tooling.
- The final manifest is still written on ordinary failures when possible, but
  resume state does not depend on that final manifest.

## Configuration and CLI

Add an operational top-level configuration block:

```yaml
execution:
  attempt_root: ${DAVINCI_RUN_ROOT}
  checkpoints:
    mode: required       # required | best_effort | off
    granularity: item    # item | stage
    loaded_sources: true
    retain: all
```

Rules:

- production requires `mode: required`, `granularity: item`, and
  `retain: all`;
- smoke and preflight runs may use `best_effort`;
- examples may use `off`;
- checkpoint policy is part of attempt identity;
- the resume decision is an invocation choice, not stored in the scientific
  YAML.

CLI additions:

```text
davinci run CONTROL.yaml
davinci run CONTROL.yaml --resume
davinci run CONTROL.yaml --resume-plan
davinci run CONTROL.yaml --resume --restart-from STAGE[:ITEM]
```

- `--resume-plan` performs validation and prints what would be restored,
  recomputed, or blocked without executing.
- `--restart-from` ignores the selected receipt and its downstream dependents;
  it does not delete historical receipts.
- `davinci run --help` documents fresh-run, resume, and restart behavior.
- Readiness validation checks attempt-root semantics, checkpoint policy,
  writeability, supported codecs, and the absence of interactive stages.

PBS resubmission uses the same control file and attempt root with `--resume`.
Each PBS job becomes a new execution record inside the attempt.

## Manifest and Reporting

Extend the manifest with:

- run, attempt, and execution IDs;
- attempt and checkpoint schema identities;
- execution history including PBS job IDs when available;
- per-stage and per-item disposition;
- checkpoint paths and checksums;
- upstream lineage;
- invalidated or recomputed items and reasons;
- interruption/failure history; and
- final checkpoint inventory.

Terminal and Markdown reporting should show a concise resume plan before work
starts and a final count of computed, restored, and recomputed items.

## Implementation Phases

### Phase 1 — Contracts and attempt journal

1. Add typed execution/checkpoint configuration.
2. Define attempt, execution, receipt, object, and resume-plan schemas with
   explicit schema versions.
3. Implement atomic JSON receipt/event publication and attempt locking.
4. Add identity and dependency validation.
5. Add CLI flags and a read-only resume plan.
6. Update readiness validation and terminology documentation.

This phase does not yet skip pipeline computation.

### Phase 2 — Stage-boundary resume

1. Add the checkpoint manager to `PipelineRunner` and `PipelineContext`.
2. Define stage checkpoint codecs.
3. Restore stage results and context deltas before stage execution.
4. Rebuild the terminal manifest from restored and computed evidence.
5. Add execution signal/failure recording.

This establishes end-to-end resume semantics using small synthetic stages.

### Phase 3 — Dataset-bearing item checkpoints

1. Checkpoint each loaded source.
2. Move analysis reuse checks before `analyze_inputs`; checkpoint every
   topologically completed analysis.
3. Checkpoint each completed pairing job, including intermediate target-grid
   paired data.
4. Reuse the existing chunked-NetCDF artifact writer/loader and consolidate
   duplicate identity/checksum code.
5. Ensure restored datasets reconstruct `SourceData` and paired-data wrappers
   exactly enough for downstream stages.

This phase addresses the principal computation and memory costs.

### Phase 4 — Downstream products

1. Checkpoint statistics by pair/source.
2. Add checksummed saved-product receipts.
3. Checkpoint each logical plot and skip rendering when every declared output
   validates.
4. Persist summary request identity and response before allowing a repeat
   provider call.
5. Restore inspection and completion evidence.
6. Extend the final manifest and pipeline log.

### Phase 5 — Production and PBS hardening

1. Add subprocess interruption and signal tests.
2. Exercise a multi-execution PBS-style resume locally.
3. Add a compact EOF/projection/wavelet production integration run.
4. Update the production-run skill to construct checkpoint-ready YAML and
   explain resume commands in its help output.
5. Update PBS submission templates to record execution and scheduler IDs.
6. Run the full repository quality gate in the `davinci` environment.

## Test Design Requiring Approval Before Implementation

Per repository policy, implementation must not begin until this test design is
reviewed and approved.

### 1. Runner stage-boundary flow

Exercise `PipelineRunner.run` with three real checkpoint-aware synthetic
stages. The first writes an xarray source, the second derives data, and the
third consumes it. Terminate after the second stage, start a new runner against
the same attempt, and assert:

- the first two execute methods are not called again;
- their datasets and `StageResult` values are reconstructed;
- the third stage receives the reconstructed context; and
- the final manifest records two restored stages and one computed stage.

### 2. Analysis DAG item flow

Run the real `AnalysesStage` with three dependent registered test analyses and
inject interruption during the third. Resume through the same pipeline entry
point and assert that the first two analysis adapters are not invoked, the
third receives restored inputs, and downstream analysis-status/artifact
metadata is complete.

### 3. Source and pairing flow

Load two compact on-disk NetCDF sources and create two real pairs, including an
intermediate-grid pair. Interrupt after the first pair finalizes. Resume and
assert that valid source and first-pair objects are loaded from checkpoints,
only the second pair strategy executes, and both pairs are available to
statistics.

### 4. Downstream file flow

Run statistics, save, plotting, summary with a fake provider, inspection,
completion, and manifest. Interrupt after each stage in parameterized cases.
Resume and verify that:

- valid CSVs and plots are not rewritten;
- a checksummed missing or changed file invalidates its logical product;
- the fake summary provider is not called twice;
- inspection consumes restored plot receipts; and
- completion passes only when every declared product validates.

### 5. Identity denial flow

Through the CLI/pipeline entry point, vary one identity dimension at a time:
normalized configuration, raw source inventory, upstream object checksum, code
hash, and checkpoint schema version. Assert that `--resume-plan` identifies the
first invalid node and normal `--resume` refuses attempt-level identity
mismatches before any stage executes.

### 6. Corruption and atomicity flow

Create truncated NetCDF chunks, checksum mismatches, orphan temporary
directories, an unfinalized item receipt, and a truncated state snapshot.
Assert that no partial data are restored, valid unrelated branches remain
reusable, and the append-only event journal can rebuild the latest valid state.

### 7. Process interruption flow

Launch the CLI in a subprocess, wait for an item receipt, send `SIGTERM`, and
resume in a second subprocess. Assert execution history records interrupted
then completed executions and that finalized prior work is not repeated.

### 8. Production EOF/wavelet flow

Run a compact synthetic production configuration through preprocessing, EOF,
projection, wavelet filtering, plotting, inspection, completion, and manifest.
Interrupt at each stage boundary in separate parameterized cases. Every
resumed run must produce the same final scientific datasets and logical
products as an uninterrupted reference run, with matching content checksums
and explicit restored lineage.

## Acceptance Criteria

- An identical interrupted attempt resumes automatically from its latest valid
  work.
- No expensive stage item executes before checkpoint lookup.
- Any changed attempt identity is rejected before computation.
- Corrupt or partial checkpoints are never consumed.
- Restored datasets are scientifically identical to uninterrupted outputs.
- Production completion cannot distinguish restored from computed work except
  through provenance.
- A scheduler interruption never introduces an approval prompt.
- Help and readiness output explain exactly what will happen.
- The full `pytest`, `mypy`, `black --check`, and `isort --check` gate passes in
  the `davinci` conda environment.

## Explicit Non-Goals for the First Release

- Reusing checkpoints across run revisions or unrelated attempts.
- Distributed checkpoint coordination across concurrent writers.
- Mid-operation snapshots inside one numerical kernel or renderer call.
- Automatic checkpoint garbage collection.
- Bypassing identity checks with a force or compatibility flag.
- Supporting legacy run nomenclature or deprecated environment aliases.
