---
title: 'DAVINCI: Type-Safe, Geometry-Aware Model and Observation Evaluation in NASA CERES Production Processing'
tags:
  - Python
  - atmospheric chemistry
  - aerosol
  - model evaluation
  - air quality
  - xarray
  - satellite remote sensing
  - radiation budget
authors:
  - name: David Fillmore
    orcid: 0000-0001-7726-4352
    affiliation: 1
affiliations:
  - name: National Center for Atmospheric Research, Boulder, CO, USA
    index: 1
date: 2 July 2026
bibliography: paper.bib
nocite: '@fillmore_davinci'
---

# Summary

DAVINCI (Data Analysis and Visual Intelligence for Climate/Chemistry) is a
Python package for evaluating climate and atmospheric composition model output
against observations. It runs in the production environment of the NASA Clouds
and the Earth's Radiant Energy System (CERES) [@wielicki_ceres_1996] compute
infrastructure at NASA Langley Research Center (LaRC), and on the Derecho
supercomputer at the NCAR-Wyoming Supercomputing Center (NWSC). At LaRC it
provides quality assessment of observation products, model verification, and
diagnostics of radiative inputs for the satellite-constrained aerosol system
behind the CERES Surface and Atmospheric Radiation Budget (SARB) flux
computation [@kato_sarb_2013]. DAVINCI was built by and for AI agents: coding
agents implemented it under the author's design and review (see the AI usage
disclosure), and its operating surfaces are as legible to agents as to
analysts: validated YAML control files run by one command, completion gates a
machine can check, provenance manifests.

Beyond that deployment, DAVINCI combines validated configuration,
geometry-aware pairing, evaluation statistics, and plotting styled for
publication in a staged workflow built on xarray. Models and observations are
handled uniformly as sources distinguished by data geometry rather than type,
so paired evaluation, observation-only field campaign workflows, satellite
swath-to-grid evaluation, and derived analyses (gridded diagnostics, empirical
orthogonal function [EOF] decomposition, continuous wavelet analysis) share
one software stack.

# Statement of need

Atmospheric model evaluation requires pairing model output with observations
that span fundamentally different data geometries: fixed surface stations
(points), aircraft flight tracks, vertical profiles, satellite swaths, and
gridded products. Existing workflows are often fragmented by observation type,
with pairing code duplicated ad hoc per campaign or satellite product, hard to
reproduce, extend, or share.

Operational deployment sharpens these requirements. When evaluation runs
inside a processing system, as it does for the aerosol inputs to CERES SARB,
analyses must be reviewable as configuration rather than custom scripts,
finish with checkable evidence that every expected product exists, and
regenerate deterministically as upstream data and code change.

DAVINCI addresses both needs with a unified evaluation runtime driven by
configuration. One YAML control file declares sources, pairs, derived
analyses, variable mappings, plot requests, and statistics; a single command
validates the configuration, then loads, pairs, analyzes, plots, inspects,
and writes logs and a run manifest.

Target users include atmospheric chemistry model developers, air quality and
field campaign scientists, and operational processing groups.

# State of the field

Existing open-source tooling for atmospheric composition evaluation offers
strong building blocks but no unified runtime. The monet and monetio libraries
[@baker_monet] provide readers for model formats, retrieval from observation
networks, and low-level I/O; DAVINCI uses them for data access and supplies
the evaluation layer above: pairing dispatch driven by data geometry,
configuration validated by Pydantic schemas, observation-only execution, and
satellite swath-to-grid binning. Domain libraries such as MetPy [@metpy]
provide meteorological analysis and visualization but likewise no such
evaluation runtime. DAVINCI's contribution is not a new metric or I/O library;
it is a software design that makes heterogeneous evaluation workflows easier
to configure, extend, reuse, and operate unattended.

# Software design

DAVINCI is organized around composable subsystems. The configuration layer
loads YAML control files, expands environment variables, and validates
structure before runtime; every input is declared in a single `sources:`
block, and evaluations are binary `pairs:` whose `x:`/`y:` axes carry plot
metadata but do not drive pairing. The pipeline layer executes named stages
(loading, derived analyses, pairing, statistics, plotting, inspection,
manifest) with a shared context, so paired, observation-only, and
analysis-only runs share one execution model. The pairing layer uses a
`PairingEngine` with geometry-specific strategies for point, track, profile,
swath, and grid data, including swath-to-grid binning for satellite Level 2
products, accelerated with numba and propagating retrieval uncertainty. For
each pair the reference dataset is chosen by geometry precedence, so a gridded
model is sampled onto observation locations. Reader coverage includes surface
networks (AirNow, AQS, AERONET, OpenAQ, Pandora), ICARTT aircraft data,
ozonesondes, satellite Level 2 and 3 products including MODIS and VIIRS Deep
Blue aerosol retrievals [@levy_modis_2013; @hsu_viirs_2019]; model readers
support CMAQ, WRF-Chem, UFS, CESM, and generic NetCDF. A download layer
automates staging of satellite granules from NASA Earthdata via
`earthaccess`.

A layer of derived analyses extends the same workflow to diagnostics computed
without pairing. Analyses run in dependency order; each result registers as a
source in its own right, so existing plotters render it and one analysis can
consume another. The workhorse is the gridded analysis type: source variables
are bound to semantic roles (*observation*, *first guess*, *analysis*, *mask*)
and grouped in time (daily, monthly, whole-period). Fields are reduced through
a sandboxed formula language permitting arithmetic and a fixed reduction
vocabulary (means, masked `active_mean`, count-weighted means, occurrence
fractions) while blocking attribute access, subscripts, and imports, so
formulas embedded in configuration are safe to execute. Results are written as
NetCDF and rendered through plot suites, presets that expand into consistent,
publication-styled map sets. Two further analyses serve research use: EOF
decomposition with North's rule [@north_eof_1982] for eigenvalue separation,
and Morlet continuous wavelet analysis [@torrence_compo_1998] with AR(1)
red-noise significance, applicable to any one-dimensional series including an
EOF principal component.

Three closing stages support unattended operation. An inspection stage
verifies every expected final product exists, renders previews, and can be
marked *required* so a run fails when a deliverable is missing. A manifest
stage records the run's products, plots, inspection results, and stage
outcomes in machine-readable form. An optional, always non-fatal AI summary
stage, the "Visual Intelligence" of the name, sends statistics and selected
plot images to a multimodal large language model (Anthropic Claude by default,
with an OpenRouter provider option) and writes back a structured Markdown
brief.

# Research impact statement

DAVINCI's impact is realized, not prospective: it operates inside NASA CERES
production processing.

## Production use in NASA CERES processing

The CERES project measures Earth's radiation budget from broadband radiometers
on multiple satellites [@wielicki_ceres_1996]; its SARB working group computes
surface and in-atmosphere radiative fluxes, which require specified aerosol
optical properties [@kato_sarb_2013]. CERES editions have long constrained
those aerosols with satellite-assimilated aerosol fields
[@collins_match_2001], whose optical depths and clear-sky radiative fluxes
were evaluated for CERES Edition 4.1 in @fillmore_ceres_2022. The current
modernization, CERES-SARB-CAM7, constrains the CAM7 atmosphere model (CESM3,
successor to CESM2 [@danabasoglu_cesm2_2020]) with a merged MODIS/VIIRS
aerosol optical depth (AOD) analysis: Level 2 retrievals [@levy_modis_2013;
@hsu_viirs_2019] are quality-controlled, binned to the model grid in
three-hour windows, merged with inverse-variance weighting, and applied inside
CAM7 so the aerosol direct radiative effect tracks observations; aerosol
optics are then tapped from the radiation code in each of the 26 SARB
shortwave and longwave bands.

DAVINCI is the evaluation and verification layer of this system, in two modes.
As a *library*, the observation pipeline imports `davinci_monet` for data
staging from NASA Earthdata, MODIS and VIIRS Level 2 reading, swath-to-grid
binning with uncertainty propagation, and AERONET colocation statistics that
validate the merged product; both the VIIRS reader and the uncertainty
handling were contributed upstream as general capabilities. As a *runtime*,
the operational diagnostics are three checked-in DAVINCI control files with no
project-specific analysis code: DAVINCI owns the analysis machinery, and the
application repository owns only YAML. The three analyses:

- **Observation product quality assessment.** Monthly means of the merged MODIS/VIIRS 550 nm AOD, weighted by
  observation count, its propagated uncertainty, valid pixel counts, and the fraction of three-hour windows with valid
  retrievals, the acceptance view of the observation product before it
  constrains the model.
- **Assimilation-style model verification.** Daily and multiday diagnostics of the in-model AOD constraint from model
  history fields, cast in data assimilation roles: observation, first guess
  (the AOD before correction), and analysis (after correction). The observation mean, increment, residual,
  and active fraction are computed through masked reductions over the
  actively constrained columns.
- **SARB band aerosol optics.** Column AOD, single-scatter albedo, and
  asymmetry parameter for the visible SARB band, and column extinction and
  scattering for the longwave window, derived from layer optics through vertical integrals in the formula
  language, the direct quality check on the
  radiative inputs delivered to SARB.

Every run ends with a required inspection gate over the final products and a
manifest recording them; the accepted SARB optics deliverable regenerates from
its control file in under a minute. \autoref{fig:ceres-aod} shows one such
product, the five-day mean analyzed AOD from the verification analysis.
Multimodal review of the rendered products, by analysts or the AI summary
stage, is part of the project's verification practice.

![Five-day mean CAM7 analyzed AOD at 550 nm (1--5 July 2008) from the production nudging verification analysis, after the MODIS AOD constraint is applied.\label{fig:ceres-aod}](gallery/ceres_cam7_analyzed_aod_5day.png){ width=80% }

## Additional applications

Two earlier workflows show the package's breadth: **ASIA-AQ**, a paired
evaluation of CESM/CAM-chem against four observation networks (AirNow surface,
AERONET AOD, Pandora NO$_2$ columns, DC-8 aircraft) over East and Southeast
Asia [@crawford_asia_aq], and a **MODIS AOD** swath-to-grid evaluation of
Terra and Aqua Level 2 AOD against two CAM6 variants during the December 2019
Australian bushfire event. Representative outputs appear in
\autoref{fig:asia-aq} and \autoref{fig:modis}.

![PM$_{2.5}$ spatial bias (Model $-$ Observations) at AirNow stations during ASIA-AQ (February 2024).\label{fig:asia-aq}](gallery/asia-aq_pm25_spatial_bias.png){ width=80% }

![MODIS Terra+Aqua L2 AOD (left) and two CAM6 variants (center, right), 21 December 2019, Australian bushfire event; swath pixels binned to the model grid by numba-accelerated aggregation.\label{fig:modis}](gallery/modis_aod_comparison.png)

These workflows ship as configurations, scripts, and gallery outputs in the
repository; they are evidence of breadth, not new scientific results. Some
depend on external datasets or credentials, so not every analysis reproduces
out of the box; the repository makes configuration and acquisition paths
explicit, and 1,700+ tests on synthetic data verify pipeline correctness
independent of external data.

# AI usage disclosure

DAVINCI was designed by the author and implemented entirely by AI coding
agents: Anthropic Claude models (Claude 4.5 through 4.8) wrote 100% of the
code, and Claude Fable 5 performed the final design and code review. The work
spanned approximately 100 interactive sessions, roughly 1,000 conversational
turns, and roughly 200 hours of AI inference time. The same tools assisted paper drafting and editorial revision. The author reviewed and
accepted every change; ran the relevant tests and reviewed the resulting
outputs; and takes full responsibility for the accuracy, originality,
licensing compliance, and final content of both the software and the paper.

# Acknowledgements

This work was supported by the National Center for Atmospheric Research, which
is a major facility sponsored by the National Science Foundation under
Cooperative Agreement No. 1852977.

DAVINCI builds on the monet and monetio packages [@baker_monet] developed at
the NOAA Air Resources Laboratory.

# Bibliography
