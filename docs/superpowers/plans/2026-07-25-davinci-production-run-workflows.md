# DAVINCI Production Run Workflows — Implementation Plan

**Date:** 2026-07-25
**Status:** Phase 1 implemented; Phase 2 and Phase 3 remain planned
**Branch target:** `develop`
**Design:** `docs/superpowers/specs/2026-07-25-davinci-production-run-workflows-design.md`

## Scope and sequencing

Implement in three reviewable phases:

1. **Foundation and configuration:** run nomenclature, production completion schema, readiness
   validation, runtime completion verification, and `davinci-configure-runs`.
2. **Execution:** immutable attempts and `davinci-execute-runs`.
3. **Closeout:** scheduler/output audit and `davinci-closeout-runs`.

Stop for user review after each phase. Do not submit a live PBS job during implementation or
testing without separate, explicit authorization.

## Approved test design

The user approved the following entry-point-oriented design on 2026-07-25:

- **Schema unit tests:** call `MonetConfig.model_validate()` with production, preflight, smoke, and
  example documents. Exercise run identity, clean-break contract parsing, required analysis
  semantics, output references, inspection requirements, and forbidden production names.
- **CLI integration tests:** invoke the Typer `davinci validate` entry point against temporary YAML
  and source files. Exercise strict production readiness, filename policy, environment expansion,
  output-root safety, readable diagnostics, and JSON reports.
- **Pipeline integration tests:** call `PipelineRunner.run_from_config()` with synthetic
  EOF → projection → wavelet data. Exercise the real load, analyses, artifact publication,
  plotting, inspection, completion, and manifest stages. Assert durable datasets, exact logical
  plots, inspection products, and a completed manifest.
- **Failure-path pipeline integration tests:** inject a missing artifact, a plot renderer failure,
  an inspection failure, and a per-item error through the same pipeline entry point. Assert a
  failed completion stage and failed manifest, never a false completed production run.
- **PBS workflow tests:** run submission scripts in dry-run mode or with a fake scheduler adapter.
  Assert immutable attempt allocation, environment capture, command construction, and no
  interactive command. Do not contact PBS in automated tests.
- **Closeout tests:** feed terminal scheduler fixtures plus temporary run directories to the audit
  command. Exercise success, nonzero exit, missing scheduler history, missing contract output,
  stale manifest, and incomplete inspection.
- **Skill contract tests:** validate each skill folder, statically enforce help-mode wording, and
  run every bundled executable with `--help`.

Implementation must not replace the pipeline integration tests with direct analysis or renderer
calls.

## Phase 1 — Foundation and `davinci-configure-runs`

### Task 1: Inventory and migrate production-facing nomenclature

- [x] Inventory `fable`, `.example.yaml`, production, preflight, smoke, run-root, and environment
  names across tracked configs, scripts, docs, and tests.
- [x] Classify every `fable` occurrence as:
  - genuinely tied to the Fable AI model or immutable historical evidence; or
  - generic aerosol/EOF/wavelet workflow nomenclature that must be renamed.
- [x] Rename the current scheduled controls:
  - `analyses/aerosol-tuning/configs/fable-merra2-aqua-2008.example.yaml`
    → `aod-merra2-myd08-aqua-2008-eof-wavelet-r01.yaml`;
  - `analyses/aerosol-tuning/configs/fable-merra2-aqua-2008-eof-preflight.example.yaml`
    → `aod-merra2-myd08-aqua-2008-eof-wavelet-preflight.yaml`.
- [x] Rename production-facing environment variables and run roots to DAVINCI/science names.
- [x] Update exact references in `FABLE_PLAN.md`, `CLAUDE.md`, tests, and relevant scripts.
- [x] Do not leave aliases, duplicate configs, or redirects for the obsolete production names.

### Task 2: Add run and completion schemas

Modify:

- `davinci_monet/config/schema.py`
- `davinci_monet/tests/unit/config/test_schema.py`

- [x] Add strict schemas for `run.id`, `run.kind`, and `run.completion`.
- [x] Add typed required-artifact, required-file, required-plot, and inspection contracts.
- [x] Validate the kebab-case run-ID/revision grammar.
- [x] Require production analyses named in the contract to exist, match type, and set
  `required: true`.
- [x] Require production inspection to be enabled and required.
- [x] Require at least one durable output category for production.
- [x] Migrate scheduled configs from `analysis.execution_contract`.
- [x] Remove `AnalysisExecutionContract` and its parsing/validation without a compatibility alias.
- [x] Run:

  ```bash
  conda run -n davinci pytest davinci_monet/tests/unit/config/test_schema.py
  ```

### Task 3: Add production readiness validation

Modify:

- `davinci_monet/cli/app.py`
- `davinci_monet/cli/commands/validate.py`
- focused CLI tests under `davinci_monet/tests/unit/cli/` or the existing CLI test location

Add a focused validator module if `commands/validate.py` would otherwise become too large.

- [x] Add human-readable and JSON readiness reports.
- [x] Validate config filename against run kind and ID.
- [x] Reject `.example.yaml` for production, preflight, and smoke.
- [x] Reject `fable` in production identity unless an explicit Fable-model field establishes that
  it is scientifically part of the run.
- [x] Resolve required environment variables and source paths.
- [x] Reuse reader/source-coverage helpers where they exist; report unsupported coverage checks
  explicitly rather than claiming they passed.
- [x] Validate terminal analysis persistence, logical plot references, and inspection presets.
- [x] Reject an unsafe output target that would overwrite a completed attempt.
- [x] Keep readiness read-only.
- [x] Run focused CLI tests through Typer's command entry point.

### Task 4: Preserve logical output identities

Modify:

- `davinci_monet/pipeline/stages/plot.py`
- `davinci_monet/pipeline/stages/io.py`
- `davinci_monet/pipeline/stages/manifest.py`
- corresponding unit tests

- [x] Keep `plots_generated` and add `plot_products: {plot_name: [paths...]}`.
- [x] Give saved result files stable logical identities rather than requiring basename guessing.
- [x] Carry analysis artifact roles, saved files, and logical plots into the manifest.
- [x] Verify paths exist and are nonempty before recording them as final products.
- [x] Preserve atomic manifest publication and completed-manifest protection.

### Task 5: Add runtime completion verification

Create:

- `davinci_monet/pipeline/stages/completion.py`
- `davinci_monet/tests/unit/pipeline/test_completion_stage.py`

Modify:

- `davinci_monet/pipeline/stages/factory.py`
- `davinci_monet/pipeline/stages/__init__.py`
- `davinci_monet/pipeline/stages/manifest.py`
- pipeline integration tests

- [x] Insert `CompletionStage` after inspection and before manifest in both standard factories.
- [x] Skip cleanly when no production completion contract exists.
- [x] Verify required analysis status, artifact integrity, saved files, logical plots, inspection,
  and item-error policy.
- [x] Return precise machine-readable failure reasons.
- [x] Ensure fail-fast still executes `ManifestStage`.
- [x] Make the manifest status reflect completion-stage failure.
- [x] Add the approved success and failure pipeline integration tests through
  `PipelineRunner.run_from_config()`.

### Task 6: Create `davinci-configure-runs`

Create with the skill-creator initializer:

```text
skills/davinci-configure-runs/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   ├── config-families.md
│   └── run-nomenclature.md
└── scripts/
    └── audit_config.py
```

- [x] Run `init_skill.py`; do not hand-create the initial skeleton.
- [x] Keep construction and review steps in `SKILL.md`.
- [x] Put detailed naming and config-family routing in references without duplicating it.
- [x] Make `audit_config.py` call DAVINCI's public validation interfaces rather than reimplementing
  schema rules.
- [x] Implement fixed, read-only help mode for the skill.
- [x] Implement `audit_config.py --help`.
- [x] End the workflow before submission.
- [x] Generate `agents/openai.yaml` from the finished skill.
- [x] Run `quick_validate.py` and focused skill contract tests.

### Phase 1 acceptance

- [x] The renamed EOF/wavelet production config passes strict readiness.
- [x] Its preflight config is unmistakably non-production.
- [x] The synthetic full pipeline lands every contracted artifact and passes completion.
- [x] Each injected missing/error case produces a failed manifest.
- [x] `$davinci-configure-runs help` is brief and non-mutating.
- [ ] Full local code gates pass.
- [x] Stop for user review before Phase 2.

Validation note: `pytest` passed all 2,175 tests with 9 skips; Black, isort, diff checks, and mypy
over every changed Python surface passed. The repository-wide mypy gate remains open only for the
two pre-existing Cartopy `Axes.get_extent` errors in
`davinci_monet/tests/unit/plots/test_spatial_bias_grid.py`.

## Phase 2 — `davinci-execute-runs`

### Task 7: Add immutable attempt records

Create a small typed module and tests under the existing run/provenance package, or introduce
`davinci_monet/runs/` if no suitable owner exists.

- [ ] Allocate `aNNN` atomically beneath the run revision root.
- [ ] Refuse to reuse an attempt output directory.
- [ ] Record run ID, attempt ID, config path/hash, Git commit, dirty-state policy, environment
  identity, PBS job ID, paths, and timestamps.
- [ ] Publish attempt metadata atomically.
- [ ] Never overwrite a completed attempt.

### Task 8: Create `davinci-execute-runs`

Create:

```text
skills/davinci-execute-runs/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   └── pbs-execution.md
└── scripts/
    └── submit_run_pbs.py
```

- [ ] Require a non-example config and passing production readiness.
- [ ] Resolve the `davinci` environment executable before submission.
- [ ] Render a complete noninteractive PBS script with all environment inputs.
- [ ] Provide `--dry-run` and `--help`.
- [ ] Treat an explicit submit request as authorization; otherwise remain read-only.
- [ ] Do not place an approval command, prompt, agent call, or conversational dependency in the PBS
  script.
- [ ] Capture and persist the PBS job ID.
- [ ] Report queued/running as queued/running, never complete.
- [ ] Test with a fake scheduler adapter; do not submit live PBS jobs in automated tests.
- [ ] Validate the skill and stop for user review before any live smoke submission.

## Phase 3 — `davinci-closeout-runs`

### Task 9: Add deterministic run audit

Create a DAVINCI CLI command or public audit module that:

- [ ] reads the attempt record;
- [ ] accepts terminal scheduler evidence;
- [ ] verifies PBS exit status and DAVINCI logs;
- [ ] validates manifest status and completion details;
- [ ] rechecks required artifacts, saved files, plots, and inspection paths; and
- [ ] returns structured operational-completion evidence.

Test with scheduler and filesystem fixtures. Missing scheduler history is unknown, not success.

### Task 10: Create `davinci-closeout-runs`

Create:

```text
skills/davinci-closeout-runs/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   └── completion-and-acceptance.md
└── scripts/
    └── audit_run.py
```

- [ ] Support status, audit, inspection handoff, and closeout reporting.
- [ ] Never submit or resubmit.
- [ ] Distinguish operational completion from scientific acceptance.
- [ ] Require explicit authorization before regenerating or promoting products.
- [ ] Implement fixed help mode and executable `--help`.
- [ ] Validate the skill and its deterministic audit script.

## Task 11: Enforce the skill contract repository-wide

Create focused tests at `davinci_monet/tests/unit/skills/test_skill_contracts.py`:

- [ ] discover every `skills/*/SKILL.md`;
- [ ] verify folder/frontmatter name agreement and `davinci-` prefix;
- [ ] require the standard help triggers and non-mutating help instructions;
- [ ] require four to six help bullets;
- [ ] run every bundled executable with `--help` and require exit zero; and
- [ ] run the skill-creator `quick_validate.py` against all DAVINCI skills.

## Task 12: Documentation and full validation

- [ ] Update `CLAUDE.md` config naming and production execution sections.
- [ ] Update `analyses/README.md` to distinguish example, smoke, preflight, and production controls.
- [ ] Document the three skill boundaries without duplicating their detailed references.
- [ ] Run:

  ```bash
  conda run -n davinci pytest
  conda run -n davinci mypy davinci_monet
  conda run -n davinci black --check davinci_monet
  conda run -n davinci isort --check davinci_monet
  git diff --check
  ```

- [ ] Record actual results. Do not infer live PBS readiness from mocked scheduler tests.
- [ ] With separate user authorization, perform one bounded PBS smoke run before returning a
  production config to the queue.

## Final acceptance

- [ ] Production configuration, execution, and closeout are independently invokable and auditable.
- [ ] Production jobs have no mid-run approval gate.
- [ ] Completed manifests prove exact contracted outputs, not merely absence of stage failure.
- [ ] Run revisions and attempts cannot overwrite one another.
- [ ] Production controls contain neither `.example` nor unrelated `fable` nomenclature.
- [ ] All DAVINCI skills begin with `davinci-` and provide deterministic help.
- [ ] Scientific acceptance remains a separate, explicit decision after operational completion.
