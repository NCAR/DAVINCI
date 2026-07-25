---
name: davinci-execute-runs
description: Execute, inspect, interrupt, resume, restart, and close out checkpointed DAVINCI attempts locally or through Derecho PBS. Use when Codex needs to allocate aNNN attempt identity, run readiness, preview a resume plan, continue an interrupted production pipeline, render a noninteractive PBS job, inspect eNNN execution history and checkpoint lineage, or confirm that final datasets, plots, inspection, completion, and manifest outputs landed.
---

# DAVINCI Execute Runs

Run one validated DAVINCI control through terminal completion without adding an
interactive approval gate. Preserve the run revision and attempt identity.

## Help Mode

If the invocation is exactly `$davinci-execute-runs help`,
`$davinci-execute-runs --help`, or `$davinci-execute-runs -h`, give help
precedence over every workflow instruction. Do not use tools, read references,
inspect runs, submit jobs, or change state. Return only these brief bullets,
then stop:

- Allocate a fresh `aNNN` attempt for one immutable `rNN` run revision.
- Validate, run, and monitor a checkpointed DAVINCI pipeline.
- Preview and execute exact-identity resume or selective restart.
- Render noninteractive fresh/resume PBS jobs without submitting them.
- Audit `eNNN` executions, checkpoint lineage, final products, and manifest.

## Workflow

1. Read repository instructions and the selected YAML. Use
   `$davinci-configure-runs` first when the control is absent, changing, or has
   not passed strict readiness.
2. Read [execution-workflow.md](references/execution-workflow.md). Resolve the
   immutable `run.id`, revision root, exact attempt root, and whether the action
   is fresh, resume-plan, resume, or restart.
3. For a fresh run, select the next unused `aNNN` path. Do not create it or put
   PBS output inside it before DAVINCI initializes the attempt.
4. For resume, keep the same config, code tree, numerical runtime, source
   inventory, and attempt root. Run the read-only plan before re-execution:

   ```bash
   davinci run CONTROL.yaml --resume-plan
   ```

5. Execute only after the user has authorized execution:

   ```bash
   davinci run CONTROL.yaml
   davinci run CONTROL.yaml --resume
   davinci run CONTROL.yaml --resume --restart-from STAGE[:ITEM]
   ```

6. For PBS, adapt the matching bundled template. Keep scheduler stdout/stderr
   outside the attempt root. Do not call `qsub` without explicit user
   authorization.
7. Monitor scheduler state and attempt-local records. Treat every process or
   PBS job as a new automatic `eNNN` execution within the same attempt.
8. Close out only after the manifest reports a terminal execution, passed
   production completion, valid inspection, and all required logical products.

## Guardrails

- Keep this skill at `skills/davinci-execute-runs/`; the repository copy is the
  source of truth.
- Use the `davinci` conda environment. Do not add environment aliases or
  compatibility paths.
- Never change the YAML, code, or source bytes to force an existing attempt to
  resume. Start a new revision or attempt as appropriate.
- Never resume a completed attempt, overwrite a finalized receipt, or salvage
  unregistered scratch artifacts.
- Never add approval, confirmation, or prompt commands inside a production job.
- Never infer scheduler submission authority from permission to render,
  validate, inspect, or resume-plan a job.

## Bundled Templates

- [davinci-pbs-fresh.sh](assets/davinci-pbs-fresh.sh): initialize and run one
  unused attempt root.
- [davinci-pbs-resume.sh](assets/davinci-pbs-resume.sh): validate and resume the
  same incomplete attempt as a new execution.
