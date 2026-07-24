# DAVINCI Analyses

Real-data model evaluation analyses using DAVINCI.

## Directory Structure

```
analyses/
├── README.md           # This file
└── asia-aq/            # ASIA-AQ campaign analysis
    ├── README.md       # Campaign overview and data sources
    ├── configs/        # YAML configuration files
    ├── scripts/        # Python analysis scripts
    └── output/         # Generated plots and statistics
```

## Analyses

| Analysis | Campaign | Model | Period | Status |
|----------|----------|-------|--------|--------|
| [asia-aq](asia-aq/) | NASA ASIA-AQ | CESM/CAM-chem | Feb 2024 | In Progress |

## Adding New Analyses

1. Create a new directory: `analyses/<name>/`
2. Add a README.md with campaign/project overview
3. Create subdirectories: `configs/`, `scripts/`, `output/`
4. Add exploration scripts to understand the data
5. Create YAML configs for model-observation pairing
6. Run analyses using CLI or Python scripts

## Running Analyses

```bash
# Activate environment
conda activate davinci

# Download / preprocess observation data
python analyses/asia-aq/scripts/download_airnow.py
python analyses/asia-aq/scripts/preprocess_pandora.py

# Derive the model NO2 column from CESM 3D output (for the Pandora comparison)
python analyses/asia-aq/scripts/compute_no2_column.py

# Run the evaluation pipeline
davinci run analyses/asia-aq/configs/asia-aq-airnow.example.yaml
```

## Data Locations

Model data is typically stored externally and referenced by path in configs:
- ASIA-AQ CESM: `~/Data/ASIA-AQ/`

Observation data should be downloaded to `analyses/<name>/data/` or referenced externally.
