# DAVINCI Pipeline Resumption — Implementation Work Plan

**Date:** 2026-07-25
**Status:** Implemented and verified locally; live PBS resubmission intentionally deferred
**Branch target:** `develop`
**Design:** `docs/superpowers/specs/2026-07-25-davinci-pipeline-resumption-design.md`

## Goal

Implement exact-identity checkpoint and resume support across the complete
DAVINCI pipeline. An interrupted attempt must resume from its latest valid
stage or stage item and continue through final datasets, plots, deterministic
inspection, production completion, and the manifest.

The implementation must not create a cross-run cache, reuse work across changed
identities, or add an interactive production gate.

## Repository and Execution Constraints

- Work only on `develop`.
- Read each file before modifying it and preserve surrounding patterns.
- Do not create a task branch or worktree unless the user authorizes it.
- Do not commit or push without a separate explicit user request.
- Do not submit a live PBS job without separate explicit authorization.
- Integration tests must enter through `PipelineRunner.run_from_config()` or
  the `davinci run` CLI, not by calling a renderer or numerical analysis
  directly.
- Use the `davinci` conda environment for every validation command.
- Present unexpected design changes to the user instead of silently weakening
  the approved identity, atomicity, or completion contracts.
- Stop for review after each implementation wave.

## Approved Test Design

The user approved the entry-point and data-flow test design in the companion
design document on 2026-07-25. The required flows are:

1. runner stage-boundary interruption and restoration;
2. analysis-DAG item interruption and restoration;
3. source and intermediate-grid pairing restoration;
4. saved-file, plot, summary, inspection, and completion restoration;
5. identity-change denial before computation;
6. corruption, partial-write, and atomicity handling;
7. subprocess signal interruption and continuation; and
8. compact production EOF → projection → wavelet equivalence against an
   uninterrupted reference.

Tests added during implementation must exercise those flows. Unit tests may
isolate identity, storage, and codec behavior, but they do not replace the
approved pipeline-entry integration tests.

## Intended Public Interface

### Configuration

```yaml
execution:
  attempt_root: ${DAVINCI_RUN_ROOT}
  checkpoints:
    mode: required       # required | best_effort | off
    granularity: item    # item | stage
    loaded_sources: true
    retain: all
```

Production rules:

- `mode` is `required`;
- `granularity` is `item`;
- `loaded_sources` is `true`;
- `retain` is `all`;
- `execution.attempt_root` is the common parent of `analysis.output_dir` and
  `analysis.log_dir`; and
- no legacy field or environment alias is introduced.

### CLI

```text
davinci run CONTROL.yaml
davinci run CONTROL.yaml --resume
davinci run CONTROL.yaml --resume-plan
davinci run CONTROL.yaml --resume --restart-from STAGE[:ITEM]
```

Fresh execution requires a new attempt root. Resume requires an existing,
incomplete attempt with an exact identity match. A completed attempt is closed.

## Wave 1 — Configuration, Records, and Atomic Storage

### Task 1: Add typed execution and checkpoint configuration

**Modify:**

- `davinci_monet/config/schema.py`
- `davinci_monet/tests/unit/config/test_schema.py`
- `davinci_monet/tests/unit/config/test_parser.py`
- the scheduled production, preflight, and smoke controls that require an
  explicit execution policy

**Produces:**

- `ExecutionConfig`;
- `CheckpointConfig`;
- strict enum values for mode, granularity, and retention;
- an optional top-level `execution` field on `MonetConfig`; and
- production validation tying the attempt root to output and log directories.

- [x] Add schema unit tests for valid production, preflight, smoke, and example
  policies.
- [x] Add rejection tests for missing production checkpointing, stage-only
  production granularity, disabled loaded-source persistence, non-retained
  production checkpoints, and mismatched attempt/output/log roots.
- [x] Add strict rejection tests for unknown or legacy execution keys.
- [x] Implement the typed schemas and cross-section validation.
- [x] Update the EOF/wavelet scheduled controls to use the approved fields.
- [x] Verify YAML dump/load round trips retain the execution policy.
- [x] Run:

  ```bash
  conda run -n davinci pytest \
    davinci_monet/tests/unit/config/test_schema.py \
    davinci_monet/tests/unit/config/test_parser.py
  ```

### Task 2: Define versioned attempt, execution, receipt, and plan models

**Create:**

```text
davinci_monet/pipeline/checkpoints/
  __init__.py
  models.py
```

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_models.py`

**Produces typed, JSON-serializable models for:**

- `AttemptRecord`;
- `ExecutionRecord`;
- `CheckpointReceipt`;
- `CheckpointObject`;
- `CheckpointDependency`;
- `ResumePlan`;
- `ResumePlanItem`; and
- disposition/reason enums.

- [x] Write model round-trip tests before implementation.
- [x] Require explicit schema versions on every durable record.
- [x] Validate safe stage/item path components and monotonic `eNNN`
  execution IDs.
- [x] Represent `computed`, `restored`, `recomputed`, `skipped`, and `failed`
  without changing `StageStatus`.
- [x] Reject a finalized receipt without dependencies, checksums, identity, or
  publication timestamps as appropriate for its object kind.
- [x] Keep filesystem paths serialized as normalized strings.

### Task 3: Implement canonical identity construction

**Create:**

- `davinci_monet/pipeline/checkpoints/identity.py`

**Modify:**

- `davinci_monet/analysis/artifact_manifest.py`
- `davinci_monet/tests/unit/analysis/test_artifacts.py`

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_identity.py`

**Produces:**

- canonical JSON hashing;
- complete configuration identity;
- code/source-tree identity;
- source inventory identity;
- dependency identity; and
- stage/item identity composition.

- [x] Write deterministic-order tests for mappings, paths, datetimes, tuples,
  NumPy scalars, and Pydantic models.
- [x] Write source-inventory tests using canonical path, size, nanosecond mtime,
  and an optional authoritative checksum.
- [x] Write identity-change tests covering one dimension at a time.
- [x] Extract or reuse the existing artifact identity primitives without
  maintaining two divergent hash algorithms.
- [x] Preserve existing analysis-artifact identity behavior and tests.
- [x] Use the conservative complete DAVINCI source-tree hash for the first
  release.

### Task 4: Add the append-only attempt store and locking

**Create:**

- `davinci_monet/pipeline/checkpoints/store.py`

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_store.py`

**Produces:**

- attempt initialization/opening;
- advisory single-writer locking;
- atomic JSON publication;
- append-only execution records;
- an fsynced event journal;
- atomic state snapshots;
- receipt discovery; and
- recovery from a truncated final journal record.

- [x] Test fresh initialization in an empty attempt root.
- [x] Test refusal to initialize over any existing unrelated content.
- [x] Test resume opening only an initialized, incomplete attempt.
- [x] Test refusal to resume a completed attempt.
- [x] Test two writers contending for the same attempt lock.
- [x] Test atomic replacement failure leaves no finalized record.
- [x] Test orphan temporary files and a truncated journal tail are ignored.
- [x] Test monotonically allocated `eNNN` records.
- [x] Ensure the lock is released automatically on process death; do not use a
  stale lock-directory protocol that requires unsafe guessing.

### Task 5: Implement checkpoint object codecs

**Create:**

- `davinci_monet/pipeline/checkpoints/codecs.py`

**Modify as needed:**

- `davinci_monet/analysis/artifacts.py`
- `davinci_monet/tests/unit/analysis/test_artifacts.py`

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_codecs.py`

**Produces codecs for:**

- chunked xarray/NetCDF collections;
- canonical JSON data;
- checksummed file collections;
- `SourceData` metadata; and
- paired-data metadata.

- [x] Write dataset round-trip tests covering static datasets, time-chunked
  datasets, coordinates, attrs, dtypes, geometry, and lazy reopening.
- [x] Write JSON tests for statistics and stage-result data.
- [x] Write checksummed file-collection tests for CSV, PDF, PNG, Markdown, and
  inspection JSON.
- [x] Reuse the existing atomic NetCDF collection writer and checksum
  validation rather than copying it.
- [x] Refactor low-level destination handling only as much as necessary to
  support attempt-local `objects/sha256/...` paths.
- [x] Verify a missing or changed byte makes the object invalid.

### Wave 1 acceptance

- [x] Configuration and record schemas are strict and versioned.
- [x] Attempt creation and resume opening are mutually exclusive.
- [x] Identity changes and corrupt objects are detected deterministically.
- [x] No pipeline computation is skipped yet.
- [x] Focused tests and formatting/type checks for changed files pass.
- [x] Stop for user review.

## Wave 2 — Resume Manager, Runner, CLI, and Readiness

### Task 6: Implement the checkpoint manager and resume planner

**Create:**

- `davinci_monet/pipeline/checkpoints/manager.py`

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_manager.py`

**Produces:**

- fresh/resume attempt lifecycle;
- dependency graph validation;
- receipt lookup;
- downstream invalidation;
- `--restart-from` parsing;
- context restoration hooks; and
- a read-only resume plan.

- [x] Test a fully reusable linear pipeline.
- [x] Test an invalid middle item recomputes only that item and its dependents.
- [x] Test independent branches remain reusable.
- [x] Test `--restart-from STAGE` and `STAGE:ITEM` ignore receipts without
  deleting them.
- [x] Test attempt-level configuration/code/source mismatch blocks resume
  before stage execution.
- [x] Test the planner has no filesystem writes.
- [x] Make invalidation reasons stable and machine-readable.

### Task 7: Integrate execution lifecycle into `PipelineRunner`

**Modify:**

- `davinci_monet/pipeline/stages/base.py`
- `davinci_monet/pipeline/runner.py`
- `davinci_monet/pipeline/lifecycle.py`
- `davinci_monet/pipeline/reporting.py`
- `davinci_monet/pipeline/display.py`

**Test:**

- `davinci_monet/tests/unit/pipeline/checkpoints/test_runner_resume.py`
- `davinci_monet/tests/unit/pipeline/test_pipeline_core.py`
- `davinci_monet/tests/unit/pipeline/test_lifecycle.py`

**Produces:**

- a runtime-only checkpoint-manager reference in `PipelineContext`;
- execution start/finish/failure records;
- stage receipt restore before execution;
- stage receipt publication after success;
- visible dispositions in logs and results; and
- manifest finalization after ordinary failure.

- [x] First add the approved three-stage runner interruption test.
- [x] Introduce a small stage-codec interface; do not pickle
  `PipelineContext`.
- [x] Initialize the manager after config validation and before any stage work.
- [x] Restore a valid stage context delta before calling `stage.validate()` or
  `stage.execute()`.
- [x] Keep restored stage status `COMPLETED` and record restoration separately.
- [x] Publish a stage receipt before the `on_stage_end` hook so tests and
  external monitors can interrupt immediately after a safe boundary.
- [x] Preserve fail-fast completion/manifest behavior.
- [x] Ensure dataset cleanup handles both computed and restored datasets.

### Task 8: Add CLI resume controls

**Modify:**

- `davinci_monet/cli/app.py`
- `davinci_monet/cli/commands/run.py`
- `davinci_monet/pipeline/runner.py`
- `davinci_monet/tests/unit/cli/test_app.py`

**Produces:**

- `--resume`;
- `--resume-plan`;
- `--restart-from`;
- concise plan output; and
- distinct exit behavior for invalid resume, already-completed attempt, and
  corrupt state.

- [x] Add Typer entry-point tests for every new option and invalid combination.
- [x] Require `--restart-from` to accompany `--resume`.
- [x] Make `--resume-plan` read-only and mutually exclusive with plot preview.
- [x] Thread typed invocation options through `run_analysis()` and
  `PipelineRunner.run_from_config()`.
- [x] Update `davinci run --help` with fresh/resume examples.
- [x] Keep programmatic callers backward-compatible only at the Python
  signature level through defaults; do not accept deprecated YAML fields.

### Task 9: Make readiness mode-aware

**Modify:**

- `davinci_monet/validation/readiness.py`
- `davinci_monet/cli/commands/validate.py`
- `skills/davinci-configure-runs/scripts/audit_config.py`
- focused validation and skill tests

**Produces:**

- fresh-attempt readiness;
- resume-attempt readiness;
- checkpoint-policy validation;
- supported-codec validation; and
- exact diagnostics for an existing but invalid attempt.

- [x] Add tests proving fresh readiness rejects nonempty attempts.
- [x] Add tests proving resume readiness requires a valid incomplete attempt.
- [x] Add tests proving completed attempts are not resumable.
- [x] Validate the attempt root is writable without creating probe files during
  read-only readiness.
- [x] Keep scheduler state untouched.
- [x] Extend the configuration skill audit through DAVINCI's public readiness
  interface rather than duplicating rules.

### Wave 2 acceptance

- [x] A synthetic three-stage pipeline resumes across two runner instances.
- [x] Resume planning is read-only and explains every disposition.
- [x] Identity mismatch prevents all stage execution.
- [x] CLI and readiness help are complete.
- [x] No production dataset stage has item-level reuse yet.
- [x] Focused tests and changed-file quality gates pass.
- [x] Stop for user review.

## Wave 3 — Dataset-Bearing Stage Items

### Task 10: Checkpoint each loaded source

**Modify:**

- `davinci_monet/pipeline/stages/load.py`
- `davinci_monet/pipeline/checkpoints/codecs.py`
- `davinci_monet/tests/unit/pipeline/test_load_sources_stage.py`

**Integration test:**

- `davinci_monet/tests/integration/test_pipeline_resumption.py`

**Produces:**

- one source-inventory identity and dataset object per configured source;
- restored `SourceData` with label, source type, geometry, variables, config,
  and artifact provenance; and
- source-level continuation after interruption.

- [x] Write an integration test loading two compact on-disk NetCDF sources,
  interrupting after the first source receipt, and resuming through
  `PipelineRunner.run_from_config()`.
- [x] Expand configured files deterministically before source identity
  calculation.
- [x] Lookup and restore a source before reader construction/open.
- [x] Finalize the source object only after normalization, variable mapping,
  time slicing, and metadata tagging are complete.
- [x] Merge restored and newly loaded sources in configuration order.
- [x] Publish the stage receipt only after every required source is available.

### Task 11: Checkpoint each analysis DAG node before computation

**Modify:**

- `davinci_monet/pipeline/stages/analyses.py`
- `davinci_monet/analysis/artifacts.py`
- `davinci_monet/analysis/artifact_manifest.py`
- `davinci_monet/tests/unit/pipeline/test_analyses_stage.py`
- `davinci_monet/tests/unit/analysis/test_artifacts.py`

**Integration test:**

- `davinci_monet/tests/integration/test_pipeline_resumption.py`

**Produces:**

- pre-`analyze_inputs` checkpoint lookup;
- one receipt per topologically completed analysis;
- restored derived sources and exact analysis metadata; and
- reuse of final analysis artifacts without a duplicate serialization layer.

- [x] Implement the approved three-node analysis-DAG interruption test through
  the full runner.
- [x] Build node identity from normalized analysis spec and upstream object
  identities before calling the adapter.
- [x] Restore valid nodes into `context.sources` and reconstruct
  `analysis_status`, artifacts, product metadata, and dependency state.
- [x] For newly computed nodes, preserve the existing artifact declarations and
  atomic persistence behavior.
- [x] Let a finalized analysis artifact serve as the checkpoint object where
  possible; persist non-artifact intermediate analyses in the attempt object
  store.
- [x] Verify an invalid upstream node invalidates every dependent analysis but
  not an independent branch.
- [x] Remove the current failure mode in which artifact reuse is discovered
  only after expensive analysis computation.

### Task 12: Checkpoint each pairing job

**Modify:**

- `davinci_monet/pipeline/stages/pair.py`
- `davinci_monet/pairing/engine.py` only if an interface seam is required
- pairing unit tests

**Integration test:**

- `davinci_monet/tests/integration/test_pipeline_resumption.py`

**Produces:**

- one receipt and paired dataset per configured pair;
- durable intermediate target-grid paired data; and
- restoration before worker scheduling.

- [x] Add the approved two-pair test with one intermediate-grid pair.
- [x] Restore valid pairs before partitioning eager and dask-backed jobs.
- [x] Submit only missing/invalid jobs to worker pools.
- [x] Keep checkpoint publication in the main thread after worker completion so
  HDF5/NetCDF writes do not race.
- [x] Reconstruct the paired wrapper, geometry, variable-axis attrs, source
  labels, and pairing metadata.
- [x] Preserve per-item error behavior and stage failure when all required pairs
  fail.

### Wave 3 acceptance

- [x] Source loading, analysis nodes, and pairs resume independently.
- [x] Intermediate-grid paired data survive an interrupted execution.
- [x] An interruption loses at most the currently running item in these stages.
- [x] Restored xarray outputs match uninterrupted scientific checksums.
- [x] Focused tests and changed-file quality gates pass.
- [x] Stop for user review.

## Wave 4 — Statistics, Files, Plots, and Completion

### Task 13: Checkpoint statistics items

**Modify:**

- `davinci_monet/pipeline/stages/stats.py`
- statistics pipeline/unit tests

**Produces:**

- one canonical JSON result per comparison pair or descriptive source; and
- a restored `statistics` `StageResult` consumable by `save_results`.

- [x] Test interruption after one statistics item and resume through the full
  runner.
- [x] Build item identity from stats config plus paired/source object identity.
- [x] Merge restored and computed results in deterministic order.
- [x] Restore `statistics_kind` and error/warning metadata.

### Task 14: Checkpoint logical saved products

**Modify:**

- `davinci_monet/pipeline/stages/io.py`
- save-results unit tests

**Produces:**

- atomic saved-result publication;
- one checksummed receipt per logical product; and
- selective rewrite of missing or invalid files.

- [x] Test statistics summary, descriptive summary, and per-flight output.
- [x] Write to a sibling temporary file, fsync, and replace the destination.
- [x] Validate file checksum and logical role before reuse.
- [x] Reconstruct `saved_files` and `saved_products` exactly for completion and
  manifest stages.

### Task 15: Checkpoint each logical plot

**Modify:**

- `davinci_monet/pipeline/stages/plot.py`
- plot pipeline/unit tests

**Produces:**

- one plot receipt per configured logical plot;
- checksums for every PDF/PNG emitted by the plot; and
- restored `plots_generated`/`plot_products` ordering.

- [x] Run a two-plot pipeline, interrupt after the first receipt, and prove its
  renderer is not called on resume.
- [x] Include plot config, input object identity, expanded plot-suite identity,
  and style identity in the item identity.
- [x] Treat all files emitted by one plot specification as one atomic logical
  item.
- [x] Invalidate the plot when any expected file is missing, empty, or changed.
- [x] Preserve item-error behavior and required logical plot contracts.

### Task 16: Checkpoint summary, inspection, and completion

**Modify:**

- `davinci_monet/pipeline/stages/summary.py`
- `davinci_monet/pipeline/stages/inspection.py`
- `davinci_monet/pipeline/stages/completion.py`
- corresponding unit tests

**Produces:**

- a persisted summary request/response identity;
- no repeated provider call for a valid restored summary;
- restored inspection report/previews; and
- restored deterministic completion evidence.

- [x] Use a fake summary provider through the pipeline and assert it is called
  once across interruption/resume.
- [x] Include model, prompt/template, statistics, and plot checksums in summary
  identity.
- [x] Atomically publish the summary file before its receipt.
- [x] Checkpoint inspection only after every report and preview validates.
- [x] Build completion identity from the exact required output receipts and
  item-error policy.
- [x] Keep production completion semantics identical for computed and restored
  evidence.

### Task 17: Extend manifest, logs, and terminal reporting

**Modify:**

- `davinci_monet/pipeline/stages/manifest.py`
- `davinci_monet/pipeline/reporting.py`
- `davinci_monet/pipeline/display.py`
- `davinci_monet/tests/unit/pipeline/test_manifest_stage.py`
- reporting/display tests

**Produces:**

- run/attempt/execution identity;
- execution history;
- stage/item dispositions;
- checkpoint lineage and checksums;
- invalidation reasons;
- interruption history; and
- computed/restored/recomputed totals.

- [x] Add manifest tests for uninterrupted, interrupted/resumed, failed, and
  completed attempts.
- [x] Keep manifest publication atomic and completed-manifest protection.
- [x] Rebuild the terminal manifest even when its prior execution ended after
  completion but before manifest publication.
- [x] Ensure manifest paths and lineage are self-contained within the attempt.
- [x] Update the Markdown log without duplicating restored stage timing as
  computation time.

### Wave 4 acceptance

- [x] Every standard pipeline stage has a valid resume contract.
- [x] Files and provider responses are reused only after checksum/identity
  validation.
- [x] Production completion and manifest accurately describe restored work.
- [x] Focused tests and changed-file quality gates pass.
- [x] Stop for user review.

## Wave 5 — Interruption Hardening and Production Proof

### Task 18: Record signals and interrupted executions safely

**Create or modify:**

- `davinci_monet/pipeline/checkpoints/signals.py`
- `davinci_monet/pipeline/runner.py`
- subprocess integration tests

**Produces:**

- safe handling for `SIGTERM` and `SIGINT`;
- interrupted execution records;
- no I/O directly in the signal handler; and
- usable finalized receipts after abrupt termination.

- [x] Launch the CLI in a subprocess and wait for a finalized item receipt.
- [x] Send `SIGTERM`, assert a non-success process exit and an interrupted
  execution record.
- [x] Resume in a second subprocess and assert prior items do not execute.
- [x] Add a crash-style test that leaves no final execution update and prove
  the next execution diagnoses the predecessor as abandoned while reusing
  finalized receipts.
- [x] Restore prior process signal handlers after every programmatic run.

### Task 19: Add the compact production EOF/wavelet resume matrix

**Create:**

- a compact synthetic config/fixture under the existing integration test
  infrastructure

**Test:**

- `davinci_monet/tests/integration/test_pipeline_resumption_eof_wavelet.py`

- [x] Build one uninterrupted reference through preprocessing, EOF,
  projection, wavelet filtering, plots, inspection, completion, and manifest.
- [x] Parameterize interruption after every stage boundary.
- [x] Add item-boundary cases for the two loaded sources, analysis nodes, and
  plots.
- [x] Resume each case through `PipelineRunner.run_from_config()`.
- [x] Compare scientific dataset checksums and logical products against the
  uninterrupted reference.
- [x] Assert the final production manifest is completed and contains exact
  restored lineage.
- [x] Assert no approval or other interactive command appears in the pipeline.

### Task 20: Documentation, skill, and PBS workflow updates

**Modify:**

- `CLAUDE.md`
- `skills/davinci-configure-runs/SKILL.md`
- `skills/davinci-configure-runs/references/config-families.md`
- `skills/davinci-configure-runs/references/run-nomenclature.md`
- the planned execution skill/PBS templates when Phase 2 of the production-run
  workflow is implemented
- skill contract tests

- [x] Document the actual eleven-stage standard pipeline, including
  `completion`.
- [x] Document run revision, attempt, and execution nomenclature.
- [x] Document fresh, resume-plan, resume, and restart-from commands.
- [x] Make configuration-skill help mention checkpoint-ready production YAML.
- [x] Keep every DAVINCI skill's help mode brief and non-mutating.
- [x] Ensure PBS templates resume the same attempt with `--resume` and record
  the scheduler job as a new execution.
- [x] Do not contact PBS in automated tests.

### Task 21: Full validation and handoff

- [x] Run focused checkpoint/resume suites.
- [x] Run the complete repository test suite:

  ```bash
  conda run -n davinci pytest
  ```

- [x] Run typing:

  ```bash
  conda run -n davinci mypy davinci_monet
  ```

- [x] Run formatting/import checks:

  ```bash
  conda run -n davinci black --check davinci_monet
  conda run -n davinci isort --check davinci_monet
  ```

- [x] Run `git diff --check`.
- [x] Run the configuration and skill help/audit checks.
- [x] Record actual test counts and any proven pre-existing failures.
- [x] Do not submit the real EOF/wavelet PBS production run in this task.
- [x] Stop for user review before any commit, push, or live scheduler action.

### Validation evidence

Completed on 2026-07-25 in the `davinci` conda environment:

- complete repository suite: `2,267 passed, 9 skipped`;
- compact production EOF → projection → wavelet interruption matrix: all 11
  standard stage boundaries passed;
- final checkpoint/resumption regression suite after provenance cleanup:
  `61 passed`;
- post-remediation artifact adoption, DAG restoration, and Cartopy regression
  selection: `6 passed`;
- skill contract suite: `3 passed`;
- mypy: no issues in 488 source files;
- Black and isort: all 488 source files pass;
- both repository DAVINCI skills pass the skill-package validator;
- both PBS templates pass `bash -n`;
- production and preflight EOF/wavelet controls pass strict fresh-run readiness
  against unused attempt paths;
- `git diff --check` passes; and
- no unrelated or pre-existing test failure remains in the final gate.

The user explicitly authorized completion, commit, and push after the earlier
review points. No live job was submitted and no scheduler state was changed.

## Final Acceptance Criteria

- [x] Fresh execution refuses a nonempty attempt root.
- [x] Resume refuses an empty, invalid, changed, or completed attempt.
- [x] Valid stage/item receipts are restored before expensive work.
- [x] Changed upstream identity invalidates only its downstream dependency
  closure.
- [x] Corrupt and partial checkpoint bytes are never consumed.
- [x] Loaded sources, analyses, intermediate-grid pairs, statistics, saved
  files, plots, summary, inspection, and completion all resume.
- [x] A compact resumed production run is scientifically identical to an
  uninterrupted run.
- [x] Final output and manifest are complete and self-contained.
- [x] PBS interruption introduces no approval gate.
- [x] CLI and DAVINCI skill help explain the workflow briefly.
- [x] Full local quality gates pass or any unrelated pre-existing failure is
  reported with evidence.
