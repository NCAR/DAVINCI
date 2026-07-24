# ASIA-AQ Analysis

Model evaluation for the NASA ASIA-AQ (Airborne and Satellite Investigation of Asian Air Quality) campaign.

## Campaign Overview

ASIA-AQ was an international cooperative field study conducted from January-March 2024 to address local air quality challenges across eastern Asia. The campaign deployed multiple aircraft (including NASA's DC-8 and G-III) over four countries: South Korea, Philippines, Taiwan, and Thailand.

**Science Goals:**
- Satellite validation and interpretation (especially GEMS geostationary sensor)
- Emissions quantification and verification
- Model evaluation
- Aerosol and ozone chemistry

**Data Sources:**
- [NASA ESPO ASIA-AQ](https://espo.nasa.gov/asia-aq/)
- [NASA ASDC ASIA-AQ Data](https://asdc.larc.nasa.gov/project/ASIA-AQ)
- [CASEI Campaign Archive](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ)

## Model Data

**Model:** CESM/CAM-chem with FC (full chemistry) configuration, nudged to meteorology

**Case:** `f.e3b06m.FCnudged.t6s.01x01.01`

**Resolution:** 0.1° x 0.1° (450 x 500 grid), 32 vertical levels

**Domain:** 0°-45°N, 90°-140°E (covers Southeast Asia, Korea, Taiwan, S. China, S. Japan)

**Period:** February 1-29, 2024 (hourly output, full month)

**Location:** `~/Data/ASIA-AQ/CAM/`

**Key Variables:**
| Variable | Description | Unit Conversion |
|----------|-------------|-----------------|
| O3 | Ozone | × 1e9 (mol/mol → ppb) |
| NO2 | Nitrogen dioxide | × 1e9 (mol/mol → ppb) |
| CO | Carbon monoxide | × 1e9 (mol/mol → ppb) |
| PM25 | PM2.5 | × 1.2e9 (kg/kg → µg/m³) |
| AODVISdn | Aerosol optical depth (550 nm) | none |

## Observation Data

| Type | Source | Variables | Sites | Status |
|------|--------|-----------|-------|--------|
| Surface | AirNow | PM2.5, O3, NO2, CO | 36 US Embassy monitors | ✓ Complete |
| Surface | AERONET | AOD (500 nm) | 68 sites | ✓ Complete |
| Column | Pandora | Tropospheric NO2 | 13 spectrometers | ✓ Complete |
| Aircraft | DC-8 (ICARTT) | O3, NO2, CO | 17 flights | ✓ Complete |

### Pandora Sites
Seoul, Seoul-SNU, Incheon-ESC, Yongin, Seosan, Ulsan, Suwon-USW, Busan (South Korea), Bangkok (Thailand), Vientiane (Laos), Singapore-NUS, Banting (Malaysia), Palau

### DC-8 Aircraft Variables
| Variable | Instrument | Description |
|----------|------------|-------------|
| O3_ROZE_STCLAIR | ROZE | Ozone (ppb) |
| NO2_CANOE_STCLAIR | CANOE | NO2 (pptv → ppb with scale 0.001) |
| CO_DACOM_DISKIN | DACOM | CO (ppb) |

## Directory Structure

```
asia-aq/
├── README.md
├── configs/
│   └── asia-aq-airnow.example.yaml  # Pipeline config (portable template)
├── scripts/
│   ├── download_observations.py    # Download all obs (AirNow, AERONET, Pandora)
│   ├── download_airnow.py          # AirNow data download (standalone)
│   └── run_evaluation.py           # Run pipeline (with NO2 column preprocessing)
├── data/                           # Observation data (NetCDF)
│   ├── airnow_asiaq_*.nc           # AirNow surface observations
│   ├── AERONET_L15_*.nc            # AERONET AOD
│   ├── pandora_no2_column_*.nc     # Pandora NO2 columns
│   └── cesm_no2_column_*.nc        # Precomputed model NO2 columns
├── output/                         # Plots and statistics
├── logs/                           # Pipeline logs (timestamped)
└── misc/                           # Exploratory scripts
```

## Setup

Set the environment variables for data and analysis directories:

```bash
# Model and raw observation data (required)
export ASIA_AQ_DATA=~/Data/ASIA-AQ

# Analysis directory (set automatically by run_evaluation.py)
export ASIA_AQ_ANALYSIS=/path/to/analyses/asia-aq
```

If `ASIA_AQ_DATA` is not set, scripts default to `~/Data/ASIA-AQ`.
The `ASIA_AQ_ANALYSIS` variable is set automatically when running `run_evaluation.py`.

## Usage

**Download observations:**
```bash
# All observations (AirNow, AERONET, Pandora)
python scripts/download_observations.py

# Or individual sources
python scripts/download_airnow.py
davinci get aeronet -s 2024-02-01 -e 2024-02-29 -d data
```

**Run the evaluation pipeline:**
```bash
python scripts/run_evaluation.py
```

Or via CLI:
```bash
davinci run configs/asia-aq-airnow.example.yaml
```

## Results (February 1-29, 2024)

### Surface Species (AirNow)

| Species | N | Mean Obs | Mean Model | R | NMB |
|---------|---|----------|------------|-----|------|
| PM2.5 | 1,008 | 37.3 µg/m³ | 48.2 µg/m³ | 0.21 | +29% |
| O3 | 152 | 13.0 ppb | 7.1 ppb | 0.48 | -45% |
| NO2 | 54 | 17.1 ppb | 42.8 ppb | 0.43 | +150% |
| CO | 133 | 1,376 ppb | 1,002 ppb | 0.07 | -27% |

### Aerosol Optical Depth (AERONET)

| Variable | N | Mean Obs | Mean Model | R | NMB |
|----------|---|----------|------------|-----|------|
| AOD (500nm) | 8,150 | 0.38 | 0.20 | 0.51 | -46% |

### Tropospheric NO2 Column (Pandora)

| Variable | N | Mean Obs | Mean Model | R | NMB |
|----------|---|----------|------------|-----|------|
| NO2 Column | 8,886 | 1.56×10⁻⁴ mol/m² | 2.90×10⁻⁴ mol/m² | 0.57 | +86% |

### DC-8 Aircraft

| Variable | N | Mean Obs | Mean Model | R | NMB |
|----------|---|----------|------------|-----|------|
| O3 (ROZE) | 3,248 | 37.7 ppb | 54.3 ppb | 0.42 | +44% |
| NO2 (CANOE) | 3,255 | 0.98 ppb | 1.25 ppb | 0.67 | +28% |
| CO (DACOM) | 3,244 | 169 ppb | 123 ppb | 0.77 | -27% |

*Note: Statistics computed with 3D vertical interpolation to aircraft altitude.*

## Output Files

**Statistics:**
- `output/statistics_summary.csv` - Evaluation metrics (N, MB, RMSE, R, NMB, NME, IOA)
- `output/statistics_per_flight.csv` - Per-flight metrics for aircraft data (enable via `per_flight: true` in stats config)

**Surface plots:**
- `*_scatter.png` - Model vs obs scatter plots with regression
- `*_timeseries.png` - Time series with uncertainty bands (mean ± std)
- `*_spatial_bias.png` - Spatial bias maps with city labels

**DC-8 Aircraft plots:**
- `dc8_*_scatter.png` - Model vs aircraft scatter plots
- `dc8_*_track_3d.png` - 3D flight track with bias coloring

**Logs:**
- `logs/pipeline_YYYYMMDD_HHMMSS.md` - Pipeline execution logs with timing

## Key Findings

1. **NO2 biases**: Model overpredicts NO2 at surface (+150%), in column (+86%), and aloft (+28%)
2. **AOD underprediction**: Model underpredicts aerosol loading by 46%
3. **O3 high bias**: Surface underpredicted (-45%), free troposphere overpredicted (+44%)
4. **CO low bias**: Aircraft CO shows -27% bias but best correlation (R=0.77)
5. **Pandora correlation**: NO2 column has good correlation (R=0.57) among all species
6. **Per-flight variability**: Statistics vary significantly by flight date (see `statistics_per_flight.csv`)

See the [wiki](https://github.com/NCAR/DAVINCI/wiki/ASIA-AQ-Analysis) for detailed analysis and interpretation.
