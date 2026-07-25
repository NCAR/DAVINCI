# DAVINCI Production Run Workflows — Design Spec

**Date:** 2026-07-25
**Status:** Approved for implementation planning
**Scope:** Machine-enforced production-run identity and completion contracts, plus three
repository-local skills that configure, execute, and close out complete DAVINCI runs.

## Context

A DAVINCI production run is expected to finish without an interactive pause and leave durable
products for inspection and downstream use. Depending on the analysis, those products include
derived datasets, tables, plots, inspection previews, logs, and a final manifest.

The current `analysis.execution_contract` is narrower than that operational definition. It proves
that named analysis nodes exist with expected types, but it does not prove that:

- terminal analysis results will be persisted;
- every required dataset, table, and plot was written;
- inspection covered the exact required plots;
- non-fatal per-item errors are acceptable for the run;
- a production output root identifies one immutable execution attempt; or
- a config called "production" is distinct from a preflight or example.

The current aerosol EOF/wavelet controls also demonstrate a nomenclature problem: production-facing
names contain `fable` even though the Fable AI model is not part of the run, and a production control
uses the `.example.yaml` suffix. Production controls are not examples.

## Goals

1. Define one machine-readable production contract from YAML validation through final manifest.
2. Make production jobs noninteractive after submission and capable of running to completion under
   PBS without an agent approval gate.
3. Give runs, config revisions, and execution attempts distinct, consistent identities.
4. Land every contracted dataset, table, plot, inspection artifact, log, and manifest.
5. Introduce concise, deterministic skills for configuration, execution, and closeout.
6. Give every DAVINCI skill a brief, read-only help mode and every bundled executable ordinary
   `--help`.

## Non-Goals

- Declaring scientific acceptance solely from pipeline completion.
- Submitting CESM or other model simulations.
- Hiding scheduler failures or automatically retrying scientifically different configurations.
- Preserving obsolete production-facing `fable` or `.example` names through compatibility aliases.
- Making every development or example config satisfy the production contract.

## Decisions

### D1 — DAVINCI-prefixed skill names

Create these repository-local skills:

1. `davinci-configure-runs`
2. `davinci-execute-runs`
3. `davinci-closeout-runs`

Place each skill at `skills/<skill-name>/` and name its folder exactly after the skill. Keep
`SKILL.md` concise; put detailed policy in one-level-deep `references/` files and deterministic
operations in `scripts/`.

### D2 — Skills orchestrate; DAVINCI owns invariants

Production correctness must not depend on invoking a skill. Pydantic schemas, the DAVINCI CLI,
pipeline stages, and manifests own the enforceable rules. Skills select the correct operations,
invoke those interfaces, interpret results, and apply project guardrails.

### D3 — Run classes are explicit

Every scheduled control must declare exactly one run kind:

- `production`: complete durable workflow with a strict completion contract;
- `preflight`: bounded representative run that may deliberately stop before final products;
- `smoke`: minimal runtime wiring check, not scientific evidence;
- `example`: reusable incomplete template that is not directly submittable.

Only `example` controls may use `.example.yaml`. A production, preflight, or smoke control must use
`.yaml` with its run kind represented in the YAML and, where useful, its filename.

### D4 — Scientific identity, revision, and attempt are separate

Use lowercase kebab-case.

Production config and run ID:

```text
<science>-<sources>-<period>-<workflow>-rNN
```

For the current EOF/wavelet workflow:

```text
aod-merra2-myd08-aqua-2008-eof-wavelet-r01
```

The tracked config is:

```text
analyses/aerosol-tuning/configs/aod-merra2-myd08-aqua-2008-eof-wavelet-r01.yaml
```

`rNN` identifies an immutable scientific/config revision. An execution attempt is allocated
separately by the execution workflow (`a001`, `a002`, ...), and scheduler identity is recorded
against that attempt. Re-running a revision must not overwrite an earlier attempt.

Use `fable` only when the Fable AI model is genuinely an input, method, or evaluated target.
Historical immutable Fable evidence may retain its identity; generic aerosol/EOF/wavelet production
controls, environment variables, run roots, and reports must not use it. Do not add compatibility
aliases for renamed production controls.

### D5 — Replace the narrow execution contract with a production completion contract

Add an optional root `run:` block. Existing non-scheduled configs may omit it. Scheduled production
controls use this shape:

```yaml
run:
  id: aod-merra2-myd08-aqua-2008-eof-wavelet-r01
  kind: production
  completion:
    required_analyses:
      model_daily: aod_preprocess
      aqua_daily: aod_preprocess
      aod_basis: eof
      obs_pcs: eof_projection
      filtered_pcs: wavelet_filter
    required_artifacts:
      - analysis: aod_basis
        role: basis_fit
      - analysis: obs_pcs
        role: projection_fit
      - analysis: filtered_pcs
        role: wavelet_filter
    required_saved_files: []
    required_plots:
      - basis_scree
      - projected_pc1
      - filtered_pc1
      - filtered_pc1_scalogram
    inspection:
      required: true
      presets: [eof_wavelet]
    allow_item_errors: false
```

Final field names may be adjusted to match the existing artifact vocabulary, but the semantics are
fixed:

- every required analysis exists, has the declared type, and has `required: true`;
- every required artifact is finalized, exists, and passes its recorded integrity checks;
- every required saved file exists and is nonempty;
- every required logical plot produced all configured final formats;
- required inspection ran with the declared presets and passed;
- no undeclared missing output can coexist with a completed production manifest; and
- item errors fail production unless the contract explicitly permits them.

Migrate all scheduled configs, then remove `analysis.execution_contract` without a compatibility
alias.

### D6 — Production validation has structural and readiness layers

`davinci validate` continues to perform strict schema and reference validation. Add a production
readiness mode with human-readable and JSON output. It must verify:

- valid run ID, kind, and filename policy;
- complete production contract;
- resolvable environment variables and source paths;
- source coverage of the requested time window, where the reader exposes coverage;
- terminal analysis nodes required by the workflow have durable artifact declarations;
- plot and inspection references resolve;
- production output/attempt placement cannot overwrite an existing completed attempt; and
- the execution path contains no interactive approval operation.

The readiness report is evidence that a config is structured to complete, not evidence that it has
already completed.

### D7 — Add a runtime completion stage

Add `CompletionStage` after `InspectionStage` and before `ManifestStage`. It evaluates the production
contract against pipeline results and the filesystem.

Keep existing result keys such as `plots_generated` for compatibility within the current codebase,
and add a logical `plot_products` mapping from plot config key to emitted paths. The completion stage
uses logical identities, not filename guessing.

If completion verification fails:

- mark the stage failed;
- write the reasons into the manifest;
- set manifest status to `failed`;
- preserve any prior completed manifest according to the existing atomic-publication behavior; and
- never publish a separate "complete" marker.

### D8 — Approval occurs before submission, never during the job

`davinci-execute-runs` resolves all choices and approvals before `qsub`. A direct user request to
submit is submission authorization; do not insert a redundant confirmation. If the user requested
only validation or inspection, do not submit.

The PBS script must be fully noninteractive. It activates the `davinci` environment through a
resolved executable path, receives all required environment values at submission, runs the full
pipeline, and exits nonzero on contract failure. It must not call an AI agent, request approval, or
wait for conversational input.

Queued or running is not complete. The durable attempt record stores the config identity, revision,
attempt ID, config hash, code revision, environment identity, PBS job ID, paths, and timestamps.

### D9 — Pipeline completion and scientific acceptance remain distinct

`davinci-closeout-runs` establishes operational completion from:

- terminal PBS history and zero exit status;
- clean DAVINCI pipeline logs;
- a completed production manifest;
- exact required artifacts and integrity records;
- exact required plots;
- passing automated inspection; and
- a complete attempt record.

Scientific acceptance additionally requires human or explicitly authorized multimodal inspection
against the analysis-specific checklist. Report the two states separately.

### D10 — Every skill has deterministic help mode

Every `skills/*/SKILL.md` must include a `## Help Mode` section with this behavior:

- trigger on `$<skill> help`, `$<skill> --help`, or `$<skill> -h`;
- give help precedence over the workflow;
- do not use tools, read bundled references, inspect inputs, or change state;
- return only four to six brief bullets describing supported operations; and
- stop.

Every bundled executable must implement `-h/--help`, return zero, and briefly describe its inputs,
outputs, mutations, and important safety boundary.

## Skill Boundaries

### `davinci-configure-runs`

- Classify the intended run.
- Select the nearest supported config family.
- Create or minimally adapt strict YAML when requested.
- Assign compliant run/config identity.
- Construct the completion contract.
- Validate sources, dependency graph, terminal artifacts, plots, and inspection.
- Emit a readiness report.
- Stop before submission.

### `davinci-execute-runs`

- Consume a versioned non-example config and passing readiness report.
- Allocate an immutable attempt root.
- Capture config, code, and environment identity.
- Render and inspect the noninteractive PBS script.
- Submit only when authorized.
- Record the PBS job ID and report queued/running state accurately.
- Never introduce a mid-job approval gate.

### `davinci-closeout-runs`

- Inspect terminal PBS history.
- Audit logs, manifest, completion result, and exact outputs.
- Regenerate deterministic inspection artifacts when explicitly requested.
- Distinguish operational completion from scientific acceptance.
- Record or report closeout evidence without resubmitting.

## Acceptance Criteria

- Production YAML cannot validate while masquerading as an example or using a reserved unrelated
  model name.
- A production pipeline cannot complete without every contracted dataset, file, plot, and
  inspection result.
- Nonfatal item errors cannot yield a completed production manifest by default.
- Each execution attempt has an immutable output root and durable PBS identity.
- No production PBS script contains an interactive approval path.
- All three DAVINCI skills validate structurally and satisfy the help contract.
- The current MERRA-2/MYD08 Aqua EOF-projection/wavelet workflow uses scientific nomenclature and
  passes the production readiness validator before any resubmission.
