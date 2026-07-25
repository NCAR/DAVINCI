# DAVINCI Config Families

Select the nearest tracked control and edit minimally. Read the selected YAML before changing it.

## General evaluation

Controls under `analyses/<campaign>/configs/` normally define:

- `analysis` time, output, logging, workflow, and style;
- typed `sources`;
- optional `pairs`, statistics, and plots;
- optional `run` identity for scheduled controls.

Reusable templates may end in `.example.yaml`. Do not infer that an existing example is ready for
production.

## Aerosol EOF/wavelet

Use these tracked controls:

- Production:
  `analyses/aerosol-tuning/configs/aod-merra2-myd08-aqua-2008-eof-wavelet-r01.yaml`
- EOF-only representative preflight:
  `analyses/aerosol-tuning/configs/aod-merra2-myd08-aqua-2008-eof-wavelet-preflight.yaml`

The production dependency chain is raw MERRA-2 and MODIS → daily preprocessing → EOF basis →
observation projection → wavelet filter. Its durable artifact roles are:

| Analysis type | Required role |
| --- | --- |
| `eof` | `basis_fit` |
| `eof_projection` | `projection_fit` |
| `wavelet_filter` | `wavelet_filter` |

Name logical plots by their YAML keys in `run.completion.required_plots`; never infer required plots
from filenames. Keep inspection enabled and required, and make its preset set exactly match the
completion contract.

## Production construction checklist

Before readiness validation, confirm:

1. The filename exactly matches the revisioned production `run.id`.
2. Every `required_analyses` entry exists, has the declared type, and sets `required: true`.
3. Every terminal scientific output has a required durable artifact role.
4. Every required saved file and plot uses the logical identity emitted by the pipeline.
5. Inspection is enabled, required, and uses registered presets.
6. `analysis.output_dir` and `analysis.log_dir` are sibling `output/` and `logs/` paths under one
   unique, empty attempt root.
7. Source environment variables resolve and source globs match files.
8. The job path is noninteractive and contains no approval gate.

Run strict readiness after structural review. A skipped source-coverage check is an explicit
limitation, not a pass.
