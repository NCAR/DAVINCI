# Slide 4 — Results: four decades, a correctness test, and a bias

Template slide: "Results to Date: What is the result and impact …". Also
answers "Real-World Impact" (what changed, who benefits).
Time: ~75 s

## Slide title

Results: four decades, a correctness test, and a bias

## On the slide

**1. The record — POWER only, no staged data**
- Monthly surface shortwave and 2 m temperature, 1981–2024, at Boulder,
  Barrow (Utqiaġvik), Mauna Loa, South Pole
- Anomalies vs the 1991–2020 normal, 12-month mean: a 450 W m⁻² polar seasonal
  cycle becomes a few W m⁻² of signal; Arctic warming at Barrow is visible with
  no further processing

**2. A pre-registered correctness test — not a result**
- POWER T2M *is* MERRA-2 T2M, so the pair must agree; threshold fixed before
  looking (|MB| < 0.05 K, R > 0.999)
- Hourly, four CONUS sites, Feb 2024: N = 2688, MB = 0.000 K,
  RMSE = 0.003 K, R = 1.000 → the reader is validated to 3 mK; POWER point
  mode is the nearest MERRA-2 cell

**3. The evaluation that test makes interpretable**
MERRA-2 SWGDN vs POWER ALLSKY_SFC_SW_DWN (CERES-derived, independent of the
GEOS radiation scheme), Feb 2024

| Leg | N | MB (W m⁻²) | R |
|---|---|---|---|
| point, hourly | 2688 | +9.1 | 0.96 — the diurnal cycle, not skill |
| point, daily | 112 | +9.1 | **0.81** |
| regional CONUS, daily, 1° | 14500 | +7.2 | 0.56 |

- **MERRA-2 surface shortwave is high by 7–9 W m⁻² (NMB +5 to +6 %)** in two
  legs computed independently — the direction expected if GEOS under-predicts
  cloud

**Why the workflow matters**: an ad-hoc check of the same comparison gave
R = 0.61; the pipeline gives 0.81. Only pipeline numbers are cited, and the AI
summary can only read them

**Impact**: the shortwave bias of a widely used reanalysis, quantified against
an independent reference by one config — and the same config now checks every
model DAVINCI reads. Beneficiaries: radiation / cloud / aerosol developers; the
regional air-quality community driven by MERRA-2; POWER, which gains an
independent consistency check on its parents

## Visual (later)

Left: T2M anomaly panel (`t2m_anomaly_record.pdf`), optionally the shortwave
anomaly beneath it. Right: daily density scatter and the CONUS bias map; the
table as an inset.

## Content notes

- Leg A/B numbers are from the 2026-07-15 run (POWER.md "RESULTS"). Re-run
  `power-merra2.example.yaml` (staged MERRA-2 under `${POWER_DATA}`) to
  regenerate the scatter and bias map; `analyses/power/output/` holds only the
  long-record run now.
- The R = 0.613 ad-hoc number is documented in POWER.md as an arbitrary 10-day
  single-cell `xr.align` sample that was "simply wrong"; the pipeline value for
  the same comparison is 0.812. Say "ad-hoc check", not "bug".
- NMB was recorded as a range (+5.4 to +6.4 %) across legs; the slide quotes
  the range.
- The spatial-bias map draws the whole globe for CONUS data (known limitation);
  crop or fix before it goes on a slide.
- Solar starts 1984, not 1981; pre-2000 solar is SRB-era, not CERES. Say "four
  decades".
- Barrow's anomaly rises from roughly −2 to −3 K (early 1980s) to +1 to +3 K
  (after 2015), read off the figure; compute decadal means through the
  pipeline before quoting a number aloud.
- Mauna Loa's ~−20 W m⁻² solar dip in 1989–91 precedes Pinatubo (June 1991);
  do not attribute it.
