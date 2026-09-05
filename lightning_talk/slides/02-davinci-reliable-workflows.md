# Slide 2 — DAVINCI: model evaluation as a guardrailed AI workflow

Template slides merged: "Introduction" + "The Problem". Establishes who we are,
the problem, and the reliable-AI thread that runs through the talk.
Time: ~55 s

## Slide title

Model evaluation as a guardrailed AI workflow

## On the slide

- **DAVINCI** (Data Analysis and Visual Intelligence for Climate/Chemistry):
  open-source model-evaluation toolkit from NSF NCAR. Pairs model output with
  observations of **any geometry** (stations, aircraft tracks, profiles,
  satellite swaths, grids); 30+ readers from global climate (CESM/CAM) and
  reanalysis (MERRA-2) to regional air quality (CheMPAS-A); in
  production in the NASA CERES project (CAM radiative fluxes vs SARB)
- **The hackweek's spectrum**: "from open-ended chat-based coding assistants to
  fully guardrailed gen-AI workflows that use agent skills and MCP servers."
  DAVINCI is built for the guardrailed end
- **Division of labour**: the agent writes a *config*, not code → Pydantic
  validation rejects what it gets wrong → the pipeline computes every number
  deterministically → the LLM reads the statistics and figures and interprets;
  it is told not to invent numbers, and the stage is non-fatal
- **Provenance by construction**: run log with a stage table, manifest,
  statistics CSV, cached inputs with deterministic keys, PDF figures —
  "AI-assisted outputs … must be traceable" (NASA ESDS AI strategy)
- **The weak link is the reference**: every evaluation rebuilds it from sparse,
  gappy station networks or archives that need logins, staging and regridding;
  a global model and a regional forecast checked against *different* references
  have biases that cannot be compared

## Visual (later)

Left: a one-line spectrum (chat → skills → MCP tools → guardrailed pipeline)
with DAVINCI at the right end. Right: pipeline strip — config → validate →
load / pair / stats / plots → LLM summary — with the artifact each stage leaves.

## Content notes

- Hackweek: Responsible Gen-AI for NASA Earthdata, UW eScience Institute,
  Seattle, 24–28 Aug 2026 (responsible-genai.hackweek.io). The spectrum quote
  is the site's own phrasing.
- NASA ESDS AI strategy (earthdata.nasa.gov/about/ai-strategy): "We do not
  sacrifice accuracy or traceability for speed"; "AI-assisted outputs in our
  systems must be traceable".
- "Do not invent numbers that are not present in the provided statistics or
  visible in the figures" is verbatim from the summary stage's system prompt
  (`davinci_monet/ai/summarizer.py`).
- 1,600+ tests on synthetic data, no live network in the suite. GitHub Actions
  is currently disabled for the repo, so do not say "CI green".
- CheMPAS-A is named per the abstract. The source registry has no MPAS reader
  yet (POWER.md backlog), so the slide describes reach, not a shipped reader;
  be ready for the question.
- Merging Intro and Problem is what makes room for the MCP slide within six.
  Splitting back to seven is a cut-and-paste.
