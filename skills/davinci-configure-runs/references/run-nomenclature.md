# DAVINCI Run Nomenclature

## Run kinds

| Kind | Purpose | Filename policy | Completion contract |
| --- | --- | --- | --- |
| `example` | Reusable, intentionally incomplete template | Ends in `.example.yaml` | Forbidden |
| `smoke` | Minimal runtime and environment wiring check | `.yaml`, never `.example.yaml` | Forbidden |
| `preflight` | Bounded representative scientific workflow | `.yaml`, normally ends `-preflight.yaml` | Forbidden |
| `production` | Complete durable workflow for operational use | Exactly `<run.id>.yaml` | Required |

Every scheduled control declares `run.id` and `run.kind`. Use lowercase kebab-case. A production
identity follows:

```text
<science>-<sources>-<period>-<workflow>-rNN
```

The current production identity is:

```text
aod-merra2-myd08-aqua-2008-eof-wavelet-r01
```

`rNN` is the immutable scientific/config revision. A later execution attempt is a separate `aNNN`
record beneath the revision root. Do not put an attempt number in `run.id`, reuse an earlier
attempt output directory, or overwrite a completed attempt.

Each process or PBS submission within one incomplete attempt is an automatic
`eNNN` execution. Resume keeps the same `rNN/aNNN` identity and increments only
`eNNN`. A changed config, code tree, or source inventory is not the same
attempt. A changed Python or numerical-library runtime also requires a new
attempt because it can change scientific results.

Use `fable` only when the Fable AI model is genuinely an input, method, or evaluated target.
Historical Fable synthetic evidence keeps its established identity. Generic aerosol, EOF, wavelet,
environment-variable, run-root, and report names use DAVINCI/scientific terminology. Do not add
aliases for obsolete production names.

Use the `davinci` conda environment. Do not add compatibility aliases.
