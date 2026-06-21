#!/usr/bin/env python
"""
Download AirNow data for the ASIA-AQ domain.

AirNow provides data from US Embassy/Consulate air quality monitors
in Asian cities. No API key required for this access method.

Domain: 0-45°N, 90-140°E
Period: full ASIA-AQ campaign window, 2024-01-29 to 2024-04-01
        (one 00 UTC hourly snapshot per day; daily cadence, not 24 h mean)

Coverage includes:
- Beijing, Guangzhou, Shenyang (China)
- Bangkok (Thailand) - extensive network
- Manila (Philippines)
- Hanoi (Vietnam)
- Singapore
- Kuala Lumpur (Malaysia)
- Dhaka (Bangladesh)
- Rangoon (Myanmar)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

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

# Date range — full ASIA-AQ campaign (Philippines -> Korea -> Thailand).
# pd.date_range defaults to daily freq, so add_data fetches the 00 UTC hour of
# each day: one daily snapshot per site (matches the original Feb file cadence).
START_DATE = "2024-01-29"
END_DATE = "2024-04-01"


def download_airnow():
    """Download AirNow data using monetio."""
    from monetio.obs import airnow

    print(f"Downloading AirNow data for {START_DATE} to {END_DATE}...")
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
    print("\n--- Sites ---")
    sites = (
        df.groupby(["siteid", "site", "latitude", "longitude"]).size().reset_index(name="records")
    )
    print(f"  {len(sites)} unique sites")
    for _, row in sites.iterrows():
        print(f"    {row['site']:30s} ({row['latitude']:.2f}N, {row['longitude']:.2f}E)")

    print("\n--- Variables ---")
    key_vars = ["OZONE", "PM2.5", "PM10", "NO2", "CO", "SO2"]
    for var in key_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            if valid > 0:
                print(
                    f"  {var:8s}: {valid:4d} values, range: {df[var].min():.1f} - {df[var].max():.1f}"
                )

    print(f"\n--- Time Range ---")
    print(f"  {df['time'].min()} to {df['time'].max()}")

    return df


def dataframe_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """Convert AirNow DataFrame to xarray Dataset for pairing."""
    if df is None or df.empty:
        return None

    # Key variables to include
    data_vars = ["OZONE", "PM2.5", "PM10", "NO2", "CO", "SO2"]
    available_vars = [v for v in data_vars if v in df.columns]

    # Round times to nearest hour to avoid single-site outlier times
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.round("h")

    # Get unique sites
    sites = df.groupby("siteid").first()[["site", "latitude", "longitude"]].reset_index()
    site_ids = sites["siteid"].tolist()

    # Create time index
    times = pd.to_datetime(df["time"].unique()).sort_values()

    # Use vectorized pivot_table for fast conversion
    data_dict = {}
    for var in available_vars:
        var_name = var.lower().replace(".", "")
        pivoted = df.pivot_table(index="time", columns="siteid", values=var, aggfunc="first")
        # Reindex to ensure consistent site ordering
        pivoted = pivoted.reindex(columns=site_ids)
        data_dict[var_name] = pivoted.values

    # Create dataset
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

    # Add attributes
    ds.attrs["source"] = "AirNow (US EPA)"
    ds.attrs["domain"] = (
        f"{BBOX['lat_min']}-{BBOX['lat_max']}N, {BBOX['lon_min']}-{BBOX['lon_max']}E"
    )
    ds.attrs["geometry"] = "point"
    ds.attrs["created"] = pd.Timestamp.now().isoformat()

    # Variable attributes
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


def main():
    """Main download routine."""
    print("=" * 60)
    print("AirNow Data Download for ASIA-AQ")
    print("=" * 60)
    print()

    # Download data
    df = download_airnow()

    if df is None or df.empty:
        print("\nNo data to save.")
        return

    # Save raw CSV
    csv_file = DATA_DIR / f"airnow_asiaq_{START_DATE}_{END_DATE}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV: {csv_file}")

    # Convert to xarray and save NetCDF
    print("\nConverting to xarray Dataset...")
    ds = dataframe_to_dataset(df)

    if ds is not None:
        nc_file = DATA_DIR / f"airnow_asiaq_{START_DATE}_{END_DATE}.nc"
        ds.to_netcdf(nc_file)
        print(f"Saved NetCDF: {nc_file}")

        print("\nDataset summary:")
        print(ds)
    else:
        print("Could not convert to xarray format.")

    print("\n" + "=" * 60)
    print("Download complete.")


if __name__ == "__main__":
    main()
