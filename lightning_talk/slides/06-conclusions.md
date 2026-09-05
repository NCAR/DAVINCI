# Slide 6 — Conclusions

Template slide: "Conclusions: Most important take aways or recommendations for
improvement". Also answers "Innovation & Future Vision".
Time: ~45 s

## Slide title

Take-aways, recommendations, and what comes next

## On the slide

**Take-aways**
- Reliable AI workflows keep the model out of the numbers: agents write
  configs and call tools; validated pipelines compute; the LLM interprets
  traceable outputs
- POWER is the reference that is always there — any site, any region, 1981 to
  present — and now reaches every model DAVINCI reads and any MCP-capable agent
- Provenance is not symmetric: POWER meteorology is MERRA-2 (a traceability
  check); POWER solar is CERES / SRB (an evaluation). DAVINCI reads both
  parents natively, so any comparison closes back to source observations

**Recommendations for POWER**
- Per-parameter source attribution in the response (`header.sources` is per
  request today)
- Time conventions stated up front: LST default, hourly stamped at the interval
  start, monthly "YYYY13" annual means
- Hourly regional access on the API — or point API users at the AWS Zarr
  stores, which already serve it
- Uncertainty served as parameters, and a machine-readable dataset description
  (GeoCroissant-style) so agents can *discover* POWER, not just query it

**Next**
- One POWER, two paths: a standing DAVINCI ↔ env-data-mcp cross-check
  (REST API vs AWS Zarr) as a reproducibility test
- DAVINCI as an MCP tool: "evaluate this model against POWER at these sites" —
  config in, manifest and figures out, with a human review point on the plan
- CAM and regional air-quality campaigns wherever the model output lives;
  POWER meteorology at every AERONET / Pandora site via a virtual-station
  generator. Open source: github.com/NCAR/DAVINCI

## Visual (later)

Three columns matching the three headings; footer with the GitHub URL, the
Zenodo DOI (10.5281/zenodo.21180438), and contact.

## Content notes

- GeoCroissant (Rajat Shinde) and AI-ready data (Wei Ji Leong) were hackweek
  tutorials; keep the mention light unless the audience knows them.
- "Human review point on the plan" echoes the hackweek's Research > Plan >
  Implement pattern (HumanLayer, via the workflow-patterns tutorial): the plan
  is the cheapest place to catch a mistake.
- The CAM campaign is designed (`power-cam.example.yaml` in POWER.md) but
  blocked on data; say "next".
- DAVINCI-as-MCP-tool and the cross-check are proposals, not work done. Say so
  if asked.
