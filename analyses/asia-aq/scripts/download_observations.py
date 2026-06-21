#!/usr/bin/env python
"""
Download and process observation data for the ASIA-AQ domain.

This script downloads/processes:
- AirNow data (US Embassy monitors)
- AERONET AOD data
- Pandora NO2 column data (from local raw files)

Domain: 0-45N, 90-140E
Period: full ASIA-AQ campaign window, 2024-01-29 to 2024-04-01
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Suppress warnings during data download (monetio generates many deprecation warnings)
warnings.filterwarnings("ignore")

# Data directory from env var or default to ~/Data/ASIA-AQ
ASIA_AQ_DATA = Path(os.environ.get("ASIA_AQ_DATA", Path.home() / "Data" / "ASIA-AQ"))

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ASIA-AQ domain
BBOX = {
    "lat_min": 0,
    "lat_max": 45,
    "lon_min": 90,
    "lon_max": 140,
}

# Date range — full ASIA-AQ campaign (Philippines -> Korea -> Thailand)
START_DATE = "2024-01-29"
END_DATE = "2024-04-01"


def download_airnow():
    """Download AirNow data using monetio."""
    from monetio.obs import airnow

    print("=" * 60)
    print("Downloading AirNow data")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Domain: {BBOX['lat_min']}-{BBOX['lat_max']}N, {BBOX['lon_min']}-{BBOX['lon_max']}E")
    print()

    # Download data
    df = airnow.add_data(pd.date_range(START_DATE, END_DATE))

    if df is None or df.empty:
        print("No data retrieved!")
        return None

    print(f"Total records globally: {len(df)}")

    # Filter to ASIA-AQ domain
    asia_mask = (
        (df["latitude"] >= BBOX["lat_min"])
        & (df["latitude"] <= BBOX["lat_max"])
        & (df["longitude"] >= BBOX["lon_min"])
        & (df["longitude"] <= BBOX["lon_max"])
    )
    df = df[asia_mask].copy()

    print(f"Records in ASIA-AQ domain: {len(df)}")

    if df.empty:
        print("No data in ASIA-AQ domain!")
        return None

    # Summary
    print(f"\n--- Sites ({len(df['siteid'].unique())}) ---")
    sites = (
        df.groupby(["siteid", "site", "latitude", "longitude"]).size().reset_index(name="records")
    )
    for _, row in sites.iterrows():
        print(f"  {row['site']:30s} ({row['latitude']:.2f}N, {row['longitude']:.2f}E)")

    print("\n--- Variables ---")
    key_vars = ["OZONE", "PM2.5", "PM10", "NO2", "CO", "SO2"]
    for var in key_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            if valid > 0:
                print(
                    f"  {var:8s}: {valid:6d} values, range: {df[var].min():.1f} - {df[var].max():.1f}"
                )

    return df


def airnow_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """Convert AirNow DataFrame to xarray Dataset."""
    if df is None or df.empty:
        return None

    data_vars = ["OZONE", "PM2.5", "PM10", "NO2", "CO", "SO2"]
    available_vars = [v for v in data_vars if v in df.columns]

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.round("h")

    sites = df.groupby("siteid").first()[["site", "latitude", "longitude"]].reset_index()
    site_ids = sites["siteid"].tolist()
    times = pd.to_datetime(df["time"].unique()).sort_values()

    data_dict = {}
    for var in available_vars:
        var_name = var.lower().replace(".", "")
        pivoted = df.pivot_table(index="time", columns="siteid", values=var, aggfunc="first")
        pivoted = pivoted.reindex(columns=site_ids)
        data_dict[var_name] = pivoted.values

    ds = xr.Dataset(
        {name: (["time", "site"], data) for name, data in data_dict.items()},
        coords={
            "time": times,
            "site": site_ids,
            "latitude": ("site", sites["latitude"].values),
            "longitude": ("site", sites["longitude"].values),
            "site_name": ("site", sites["site"].values),
        },
    )

    ds.attrs["source"] = "AirNow (US EPA)"
    ds.attrs["domain"] = (
        f"{BBOX['lat_min']}-{BBOX['lat_max']}N, {BBOX['lon_min']}-{BBOX['lon_max']}E"
    )
    ds.attrs["geometry"] = "point"
    ds.attrs["created"] = pd.Timestamp.now().isoformat()

    var_attrs = {
        "ozone": {"long_name": "Ozone", "units": "ppb"},
        "pm25": {"long_name": "PM2.5", "units": "ug/m3"},
        "pm10": {"long_name": "PM10", "units": "ug/m3"},
        "no2": {"long_name": "Nitrogen Dioxide", "units": "ppb"},
        "co": {"long_name": "Carbon Monoxide", "units": "ppm"},
        "so2": {"long_name": "Sulfur Dioxide", "units": "ppb"},
    }
    for var, attrs in var_attrs.items():
        if var in ds:
            ds[var].attrs.update(attrs)

    return ds


def download_aeronet():
    """Download AERONET data using monetio."""
    from monetio.obs import aeronet

    print()
    print("=" * 60)
    print("Downloading AERONET data")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print()

    dates = pd.date_range(START_DATE, END_DATE)

    # Download Level 1.5 data (more coverage than L2.0)
    # latlonbox order: [lat_min, lon_min, lat_max, lon_max] (lower-left to upper-right)
    df = aeronet.add_data(
        dates=dates,
        product="AOD15",
        latlonbox=(BBOX["lat_min"], BBOX["lon_min"], BBOX["lat_max"], BBOX["lon_max"]),
    )

    if df is None or df.empty:
        print("No AERONET data retrieved!")
        return None

    print(f"Total records: {len(df)}")
    print(f"Sites: {df['siteid'].nunique()}")

    # Show available wavelengths
    aod_cols = [c for c in df.columns if c.startswith("aod_")]
    print(f"AOD wavelengths: {', '.join(aod_cols)}")

    return df


def aeronet_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """Convert AERONET DataFrame to xarray Dataset."""
    if df is None or df.empty:
        return None

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.round("h")

    sites = df.groupby("siteid").first()[["latitude", "longitude"]].reset_index()
    site_ids = sites["siteid"].tolist()
    times = pd.to_datetime(df["time"].unique()).sort_values()

    # Key AOD variables
    aod_vars = ["aod_500nm", "aod_440nm", "aod_675nm"]
    available_vars = [v for v in aod_vars if v in df.columns]

    data_dict = {}
    for var in available_vars:
        pivoted = df.pivot_table(index="time", columns="siteid", values=var, aggfunc="first")
        # pivot_table drops all-NaN hours (dropna=True), which can differ per
        # wavelength; reindex to the full time/site axes so every variable has a
        # consistent (time, site) shape matching the dataset coords.
        pivoted = pivoted.reindex(index=times, columns=site_ids)
        data_dict[var] = pivoted.values

    ds = xr.Dataset(
        {name: (["time", "site"], data) for name, data in data_dict.items()},
        coords={
            "time": times,
            "site": site_ids,
            "latitude": ("site", sites["latitude"].values),
            "longitude": ("site", sites["longitude"].values),
        },
    )

    ds.attrs["source"] = "AERONET"
    ds.attrs["product"] = "AOD Level 1.5"
    ds.attrs["geometry"] = "point"
    ds.attrs["created"] = pd.Timestamp.now().isoformat()

    for var in available_vars:
        ds[var].attrs = {"long_name": f"AOD at {var.split('_')[1]}", "units": "1"}

    return ds


def process_pandora():
    """Process Pandora NO2 column data from raw L2 files."""
    from davinci_monet.datasets.surface.pandora import PandoraReader

    print()
    print("=" * 60)
    print("Processing Pandora NO2 column data")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print()

    pandora_dir = ASIA_AQ_DATA / "Pandora"
    if not pandora_dir.exists():
        print(f"Pandora data directory not found: {pandora_dir}")
        return None

    # Find all L2 files
    files = sorted(pandora_dir.glob("Pandora*_L2_*.txt"))
    print(f"Found {len(files)} Pandora files")

    # Filter to ASIA-AQ domain sites (exclude Boulder, CO)
    asia_files = [f for f in files if "Boulder" not in f.name]
    print(f"ASIA-AQ region files: {len(asia_files)}")

    if not asia_files:
        print("No Pandora files for ASIA-AQ region!")
        return None

    # Read using the PandoraReader
    reader = PandoraReader()
    ds = reader.open(
        asia_files,
        quality_flag_max=1,  # High + medium quality
        solar_zenith_max=80.0,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # Filter to domain
    lat_mask = (ds.latitude >= BBOX["lat_min"]) & (ds.latitude <= BBOX["lat_max"])
    lon_mask = (ds.longitude >= BBOX["lon_min"]) & (ds.longitude <= BBOX["lon_max"])
    domain_mask = lat_mask & lon_mask

    valid_sites = ds.site.values[domain_mask.values]
    ds = ds.sel(site=valid_sites)

    print(f"\nSites in domain: {len(ds.site)}")
    for site in ds.site.values:
        lat = float(ds.sel(site=site).latitude)
        lon = float(ds.sel(site=site).longitude)
        n_obs = int(ds.sel(site=site)["no2_trop_column"].count())
        print(f"  {site:25s}: ({lat:.2f}N, {lon:.2f}E) - {n_obs} obs")

    print(f"\nTotal observations: {int(ds['no2_trop_column'].count())}")

    return ds


def main():
    """Main download routine."""
    print("=" * 70)
    print("ASIA-AQ Observation Data Download/Processing")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("=" * 70)
    print()

    # 1. AirNow
    df_airnow = download_airnow()
    if df_airnow is not None:
        ds_airnow = airnow_to_dataset(df_airnow)
        if ds_airnow is not None:
            nc_file = DATA_DIR / f"airnow_asiaq_{START_DATE}_{END_DATE}.nc"
            ds_airnow.to_netcdf(nc_file)
            print(f"\nSaved: {nc_file}")

            csv_file = DATA_DIR / f"airnow_asiaq_{START_DATE}_{END_DATE}.csv"
            df_airnow.to_csv(csv_file, index=False)
            print(f"Saved: {csv_file}")

    # 2. AERONET
    df_aeronet = download_aeronet()
    if df_aeronet is not None:
        ds_aeronet = aeronet_to_dataset(df_aeronet)
        if ds_aeronet is not None:
            nc_file = (
                DATA_DIR
                / f"AERONET_L15_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.nc"
            )
            ds_aeronet.to_netcdf(nc_file)
            print(f"\nSaved: {nc_file}")

    # 3. Pandora
    ds_pandora = process_pandora()
    if ds_pandora is not None:
        nc_file = (
            DATA_DIR
            / f"pandora_no2_column_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.nc"
        )
        ds_pandora.to_netcdf(nc_file)
        print(f"\nSaved: {nc_file}")

    print()
    print("=" * 70)
    print("Download/processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
