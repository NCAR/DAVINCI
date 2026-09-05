# Slide 3 — POWER in DAVINCI: the reference that is always there

Template slide: "POWER Usage: What data/services are being incorporated into
your workflow?" Also answers "How You Work With POWER" (where it makes the
biggest difference; what it solves faster / more reliably) and the abstract's
tools question (the API).
Time: ~50 s

## Slide title

POWER in DAVINCI: `type: power`

## On the slide

- **Services**: POWER API temporal **point** and **regional** endpoints;
  hourly / daily / monthly; NetCDF; community RE; UTC. The **Parameter
  Dictionary** keys the variable catalog
- **One source type, two shapes**: `sites:` → virtual stations (POINT);
  `bbox:` → regional grid (GRID). Both pair with every model reader through
  the same geometry-based engine
- **Reliable by construction**
  - live fetch + local NetCDF cache: first run online, every rerun offline and
    reproducible; `davinci-stage-power` pre-stages the same requests
  - units normalized to SI on read, and the reader **asserts POWER's unit
    string** before scaling — upstream drift fails loudly instead of silently
    corrupting statistics
  - API behaviour measured once, then encoded as tests: LST default → UTC
    always requested; monthly "YYYY13" annual-mean rows dropped; no
    hourly/regional endpoint → refused up front; per-parameter native grids
    (solar 1°, meteorology 0.5° × 0.625°)
- **Where it makes the difference**: a complete evaluation is a ~30-line
  config — no login, no granule staging; any site, any region, 1981 to present

```yaml
sources:
  power:
    type: power
    temporal: hourly
    sites:
      - {name: "Boulder, CO",   latitude: 40.02,  longitude: -105.27}
      - {name: "Bondville, IL", latitude: 40.052, longitude: -88.373}
    variables: {ALLSKY_SFC_SW_DWN: {}, T2M: {}}
  merra2:
    type: merra2
    files: ${POWER_DATA}/MERRA2_tavg1/*/*.nc4
    variables: {SWGDN: {units: "W m-2"}, T2M: {units: K}}

pairs:
  merra2_vs_power_swdn:
    x: {source: power,  variable: ALLSKY_SFC_SW_DWN}
    y: {source: merra2, variable: SWGDN}
```

## Visual (later)

YAML on the left. Right: API → NetCDF cache → reader (unit assertion) →
pairing → stats / plots. Trim the YAML to the `power` source and the pair if it
cannot be read from the back of the room.

## Content notes

- Snippet trimmed from `analyses/power/configs/power-merra2.example.yaml`.
  Bondville replaces Table Mountain, which shares Boulder's MERRA-2 cell.
- Every API fact was measured against the live API on 2026-07-15 (POWER.md,
  "API facts — VERIFIED").
