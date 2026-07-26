---
name: davinci-configure-runs
description: Construct, adapt, and audit DAVINCI YAML run controls for example, smoke, preflight, or production workflows. Use when Codex needs to classify a DAVINCI run, choose a tracked config family, assign compliant run nomenclature, build a production completion contract, validate source and analysis references, or decide whether a config is structured to run to completion before PBS submission.
---

# DAVINCI Configure Runs

Build or review one strict, scientifically named DAVINCI control and produce readiness evidence.
End before attempt allocation or scheduler submission.

## Help Mode

If the invocation is exactly `$davinci-configure-runs help`, `$davinci-configure-runs --help`, or
`$davinci-configure-runs -h`, give help precedence over every workflow instruction. Do not use tools,
read bundled references, inspect inputs, or change state. Return only these brief bullets, then stop:

- Classify a run as example, smoke, preflight, or production.
- Select and minimally adapt the nearest tracked YAML config family.
- Assign scientific run IDs, revisions, filenames, and output-root conventions.
- Build and audit production completion contracts and durable outputs.
- Add exact checkpoint policy and `aNNN` attempt paths.
- Run strict fresh/resume readiness validation and stop before PBS submission.

## Workflow

1. Read the repository `AGENTS.md`, `CLAUDE.md`, and any root `REVIEW_*.md` or `HANDOFF_*.md`
   before changing a config.
2. Read [run-nomenclature.md](references/run-nomenclature.md), classify the requested run, and state
   the classification. Do not silently promote a smoke or preflight control to production.
3. Read [config-families.md](references/config-families.md), inspect the nearest tracked control,
   and reuse its reader, analysis, plotting, and inspection patterns.
4. For an audit request, remain read-only. For an explicit construction or edit request, make the
   smallest YAML change that satisfies the intended science and run class.
5. For production, enumerate every required analysis, durable artifact role, saved logical file,
   logical plot, and inspection preset in `run.completion`. Require all named analyses.
6. Add `execution.attempt_root` and the required item/source-retaining checkpoint policy. Keep
   output and logs beneath that exact `aNNN`.
7. Run `scripts/audit_config.py <config>` from the `davinci` environment. Use `--resume` only for
   an initialized incomplete attempt and `--json` for machine-readable evidence.
8. Report the run kind and ID, config path, readiness failures or skips, and the exact next safe
   action. Stop before allocating an attempt, rendering PBS, calling `qsub`, or resubmitting.

## Guardrails

- Keep this skill in the repository at `skills/davinci-configure-runs/`; the tracked repository is
  the source of truth.
- Treat `run.completion` as operational output evidence, not scientific acceptance.
- Do not add environment aliases, obsolete production `fable` names, or
  `analysis.execution_contract` compatibility.
- Do not claim source time coverage passed when the reader lacks a common pre-open coverage check.
- Do not reuse a nonempty production attempt root or overwrite completed output.
- Do not submit or requeue a job from this skill.
