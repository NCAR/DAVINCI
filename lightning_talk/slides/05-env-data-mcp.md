# Slide 5 — POWER as tools for agents: the KBase env-data-mcp server

New slide (not in the template). Answers "Innovation & Future Vision": what
emerging technologies (AI, automation) unlock when paired with POWER data.
Time: ~55 s

## Slide title

POWER as tools for agents: the KBase env-data-mcp server

## On the slide

- **env-data-mcp** (KBase incubator, Apache-2.0): an MCP server that exposes
  environmental data — weather, soil, atmospheric composition, satellite — as
  tools callable by any MCP-compatible assistant or workflow. Six sources
  working (NASA POWER, SSURGO, SoilGrids, GBIF, TROPOMI, OpenAQ), three
  prototyped (OCO-2, EMIT, ESS-DIVE)
- **POWER's two halves are two tool families**: `nasa_power_merra2_*`
  (meteorology) and `nasa_power_syn1deg_*` (radiation), each with
  `available_variables`, `point_query`, `bbox_query`; hourly → climatology;
  read from the **POWER AWS Open Data Zarr stores** — anonymous, no key
- **Guardrails a tool can carry**
  - inputs validated (lat/lon ranges, bbox order, date order); every response
    validated against one schema
  - a `_meta` block on every call: source, **license and citation**,
    `query_params` echoed verbatim "so any result can be reproduced from a log
    record", latency, `success` / `error`, unavailable variables
  - errors come back as data (`success: false`), not crashes — the agent sees
    what went wrong
  - a runtime guard: a benchmark-fitted timing model estimates cost; above
    `max_runtime_s` the tool returns a warning instead of data
  - cached variable lists checked against the live services in CI; hundreds of
    mocked unit tests plus live integration tests; a GUI to run the same tools
    by hand
- **Complementary to DAVINCI**: the MCP answers an agent's question at a point
  or a box; DAVINCI runs the campaign. Two independent paths to POWER — REST
  API + NetCDF cache, and AWS Zarr — and one reproducibility check waiting to
  be run: same site, both paths, do the values match?

## Visual (later)

Left: tool-call diagram — agent → MCP server → POWER Zarr on AWS → `{data,
_meta}` back, with the `_meta` fields listed. Right: DAVINCI's path (API →
cache → pipeline) drawn in parallel, both meeting at "POWER".

## Content notes

- Repo: github.com/kbaseincubator/env-data-mcp, v0.1.0, 35 commits
  (2026-05-13 → 2026-08-28), primary author Matt Dawson. Confirm how to
  attribute (KBase / DOE, and the person) before the slide is shown.
- "POWER API" in the loose sense only: the MCP reads POWER's AWS Open Data
  Zarr stores (`power_merra2_*_spatial_utc.zarr`,
  `power_syn1deg_*_spatial_utc.zarr`), not the REST API. Say "POWER data as
  tools", not "the POWER API".
- Point queries return the nearest grid cell (`argmin`), the same convention
  the REST point endpoint uses (leg B). Bbox queries add one buffer cell
  outside each edge for interpolation.
- Coverage strings are the MCP's own: MERRA-2 "1980–present", SYN1deg
  "2001–present". POWER's REST record starts 1981 (met) / 1984 (solar).
- Timing model: `t = α + β_n·n_days + β_a·area_deg2`, fitted at Yakima,
  Manaus and Frankfurt; coefficients committed as `timing_model.json`;
  default `max_runtime_s` is 30 s.
- Test counts by `def test_` grep: 595 unit, 187 integration; the README says
  "250+ unit tests". Say "hundreds", not a number.
- The hackweek's MCP session (Joe Hamman, Jason Gilman) included an MCP
  evaluation demo (github.com/jasongilman/mcp-eval-demo) — a natural reference
  if evals come up in questions.
