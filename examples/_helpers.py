"""Shared helper functions for DAVINCI examples.

This module provides common utilities used across all plot examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from davinci_monet.tests.synthetic.datasets import (
    create_dataset_dataset,
    create_gridded_geometries,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
)
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig

if TYPE_CHECKING:
    import matplotlib.figure

# Default output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "plots"


def save_figure(fig: matplotlib.figure.Figure, name: str, output_dir: Path | None = None) -> None:
    """Save figure as PNG (300 DPI) and PDF.

    Parameters
    ----------
    fig
        Matplotlib figure to save.
    name
        Base filename (without extension).
    output_dir
        Output directory. Defaults to examples/output/plots.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def create_paired_surface_data(
    n_sites: int = 30,
    variables: list[str] | None = None,
    bias_range: float = 8.0,
    noise: float = 3.0,
    seed: int = 42,
) -> xr.Dataset:
    """Create paired x-y surface data.

    Parameters
    ----------
    n_sites
        Number of dataset sites.
    variables
        Variables to include. Defaults to ["O3", "PM25", "NO2"].
    bias_range
        Range of spatially-varying bias (-bias_range to +bias_range).
    noise
        Random noise standard deviation.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Paired dataset with x_* and y_* variables.
    """
    if variables is None:
        variables = ["O3", "PM25", "NO2"]

    domain = Domain(lat_min=25, lat_max=50, lon_min=-125, lon_max=-65)
    time_config = TimeConfig(start="2024-07-01", end="2024-07-15", freq="1h")

    # Create datasets
    geometry = create_point_geometries(
        n_sites=n_sites,
        variables=variables,
        domain=domain,
        time_config=time_config,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 1)

    # Create spatially-varying bias based on longitude
    # Western sites: negative bias, Eastern sites: positive bias
    lons = geometry["longitude"].values
    lon_normalized = (lons - lons.min()) / (lons.max() - lons.min())  # 0 to 1
    site_bias = bias_range * (2 * lon_normalized - 1)  # -bias_range to +bias_range

    data_vars = {}
    for var in variables:
        x_data = geometry[var].values  # shape: (time, site)
        # Apply site-specific bias (broadcast across time)
        bias_2d = np.broadcast_to(site_bias, x_data.shape)
        y_data = x_data + bias_2d + rng.normal(0, noise, x_data.shape)
        y_data = np.clip(y_data, 0, None)

        data_vars[f"x_{var.lower()}"] = (["time", "site"], x_data.astype(np.float32))
        data_vars[f"y_{var.lower()}"] = (["time", "site"], y_data.astype(np.float32))

    paired = xr.Dataset(
        data_vars,
        coords={
            "time": geometry["time"],
            "site": geometry["site_id"].values,
            "latitude": ("site", geometry["latitude"].values),
            "longitude": ("site", geometry["longitude"].values),
        },
        attrs={"title": "Paired Surface Data", "geometry": "point"},
    )

    return paired


def create_paired_track_data(
    n_flights: int = 6,
    points_per_flight: int = 240,
    variables: list[str] | None = None,
    bias_range: float = 10.0,
    seed: int = 42,
) -> xr.Dataset:
    """Create paired x-y aircraft track data.

    Parameters
    ----------
    n_flights
        Number of flights.
    points_per_flight
        Data points per flight.
    variables
        Variables to include. Defaults to ["O3"].
    bias_range
        Range of altitude-dependent bias. Low altitude: negative,
        high altitude: positive.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Paired dataset with flight coordinate and altitude.
    """
    if variables is None:
        variables = ["O3"]

    domain = Domain(lat_min=30, lat_max=45, lon_min=-105, lon_max=-90)
    rng = np.random.default_rng(seed)

    all_data: dict[str, list] = {
        "time": [],
        "latitude": [],
        "longitude": [],
        "altitude": [],
        "flight": [],
    }
    for var in variables:
        all_data[f"x_{var.lower()}"] = []
        all_data[f"y_{var.lower()}"] = []

    base_date = np.datetime64("2024-07-01")

    for flight_num in range(n_flights):
        n_points = points_per_flight
        flight_date = base_date + np.timedelta64(flight_num // 2, "D")
        am_pm = "AM" if flight_num % 2 == 0 else "PM"
        flight_id = f"{str(flight_date)[:10]}-{am_pm}"

        start_hour = 8 if flight_num % 2 == 0 else 14
        times = flight_date + np.arange(n_points) * np.timedelta64(1, "m")
        times = times + np.timedelta64(start_hour, "h")

        # Flight path: spiral pattern
        center_lat = domain.center[1] + 2 * rng.normal()
        center_lon = domain.center[0] + 3 * rng.normal()
        t = np.linspace(0, 4 * np.pi, n_points)

        radius = 0.5 + t / (4 * np.pi) * 1.5
        lats = center_lat + radius * np.sin(t) + 0.02 * rng.normal(size=n_points)
        lons = center_lon + radius * np.cos(t) * 1.3 + 0.02 * rng.normal(size=n_points)

        # Altitude profile
        alt_profile = np.concatenate(
            [
                np.linspace(500, 8000, n_points // 3),
                np.ones(n_points // 3) * 8000 + 500 * rng.normal(size=n_points // 3),
                np.linspace(8000, 1000, n_points - 2 * (n_points // 3)),
            ]
        )
        alt_profile = np.clip(alt_profile, 300, 12000)

        all_data["time"].extend(times)
        all_data["latitude"].extend(lats)
        all_data["longitude"].extend(lons)
        all_data["altitude"].extend(alt_profile)
        all_data["flight"].extend([flight_id] * n_points)

        for var in variables:
            # O3 increases with altitude
            base_val = 30 + 0.005 * alt_profile + 5 * rng.normal(size=n_points)
            x_val = np.clip(base_val, 10, 120)
            # Altitude-dependent bias: negative at low altitude, positive at high
            # Crossover at ~4000m
            alt_bias = bias_range * (alt_profile - 4000) / 4000  # -bias_range to +bias_range
            y_val = x_val + alt_bias + 3 * rng.normal(size=n_points)
            y_val = np.clip(y_val, 10, 130)

            all_data[f"x_{var.lower()}"].extend(x_val)
            all_data[f"y_{var.lower()}"].extend(y_val)

    # Build dataset
    n_total = len(all_data["time"])
    data_vars = {}
    for var in variables:
        data_vars[f"x_{var.lower()}"] = (
            ["time"],
            np.array(all_data[f"x_{var.lower()}"], dtype=np.float32),
        )
        data_vars[f"y_{var.lower()}"] = (
            ["time"],
            np.array(all_data[f"y_{var.lower()}"], dtype=np.float32),
        )

    paired = xr.Dataset(
        data_vars,
        coords={
            "time": np.array(all_data["time"], dtype="datetime64[ns]"),
            "latitude": ("time", np.array(all_data["latitude"], dtype=np.float32)),
            "longitude": ("time", np.array(all_data["longitude"], dtype=np.float32)),
            "altitude": ("time", np.array(all_data["altitude"], dtype=np.float32)),
            "flight": ("time", all_data["flight"]),
        },
        attrs={"title": "Paired Aircraft Track Data", "geometry": "track"},
    )

    return paired


def create_paired_profile_data(
    n_profiles: int = 10,
    n_levels: int = 50,
    variables: list[str] | None = None,
    bias_range: float = 10.0,
    seed: int = 42,
) -> xr.Dataset:
    """Create paired x-y profile data.

    Parameters
    ----------
    n_profiles
        Number of sonde profiles.
    n_levels
        Vertical levels per profile.
    variables
        Variables to include. Defaults to ["O3"].
    bias_range
        Range of level-dependent bias. Lower levels: negative,
        upper levels: positive.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Paired dataset with time and level dimensions.
    """
    if variables is None:
        variables = ["O3"]

    domain = Domain(lat_min=25, lat_max=50, lon_min=-125, lon_max=-65)
    time_config = TimeConfig(start="2024-07-01", end="2024-07-15", freq="1D")

    geometry = create_profile_geometries(
        n_profiles=n_profiles,
        n_levels=n_levels,
        variables=variables,
        domain=domain,
        time_config=time_config,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 1)

    # Create level-dependent bias: negative at surface, positive aloft
    levels = geometry["level"].values
    level_normalized = (levels - levels.min()) / (levels.max() - levels.min())
    level_bias = bias_range * (2 * level_normalized - 1)  # -bias_range to +bias_range

    data_vars = {}
    for var in variables:
        x_data = geometry[var].values  # shape: (time, level)
        # Apply level-specific bias (broadcast across time)
        bias_2d = np.broadcast_to(level_bias, x_data.shape)
        y_data = x_data + bias_2d + rng.normal(0, 2, x_data.shape)
        y_data = np.clip(y_data, 0, None)

        data_vars[f"x_{var.lower()}"] = (["time", "level"], x_data.astype(np.float32))
        data_vars[f"y_{var.lower()}"] = (["time", "level"], y_data.astype(np.float32))

    paired = xr.Dataset(
        data_vars,
        coords={
            "time": geometry["time"],
            "level": geometry["level"],
            "latitude": geometry["latitude"],
            "longitude": geometry["longitude"],
        },
        attrs={"title": "Paired Profile Data", "geometry": "profile"},
    )

    return paired


def create_paired_swath_data(
    n_scans: int = 100,
    n_pixels: int = 60,
    variables: list[str] | None = None,
    bias_range: float = 0.4,
    seed: int = 42,
) -> xr.Dataset:
    """Create paired x-y satellite swath data.

    Parameters
    ----------
    n_scans
        Number of scanlines.
    n_pixels
        Pixels per scanline.
    variables
        Variables to include. Defaults to ["NO2"].
    bias_range
        Range of spatially-varying bias (multiplicative).
        Creates east-west gradient from -bias_range to +bias_range.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Paired dataset with scanline and pixel dimensions.
    """
    if variables is None:
        variables = ["NO2"]

    domain = Domain(lat_min=25, lat_max=50, lon_min=-100, lon_max=-70)
    time_config = TimeConfig(start="2024-07-01", end="2024-07-02", freq="1D")

    geometry = create_swath_geometries(
        n_scans=n_scans,
        n_pixels=n_pixels,
        variables=variables,
        domain=domain,
        time_config=time_config,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 1)

    # Create spatially-varying bias based on longitude
    lons = geometry["longitude"].values  # shape: (scanline, pixel)
    lon_normalized = (lons - lons.min()) / (lons.max() - lons.min())
    spatial_bias = bias_range * (2 * lon_normalized - 1)  # -bias_range to +bias_range

    data_vars = {}
    for var in variables:
        x_data = geometry[var].values
        # Apply spatially-varying multiplicative bias
        y_data = x_data * (1.0 + spatial_bias) + rng.normal(0, 0.3, x_data.shape)
        y_data = np.clip(y_data, 0, None)

        data_vars[f"x_{var.lower()}"] = (["scanline", "pixel"], x_data.astype(np.float32))
        data_vars[f"y_{var.lower()}"] = (["scanline", "pixel"], y_data.astype(np.float32))

    # Copy qa_flag
    data_vars["qa_flag"] = (["scanline", "pixel"], geometry["qa_flag"].values)

    paired = xr.Dataset(
        data_vars,
        coords={
            "time": geometry["time"],
            "scanline": geometry["scanline"],
            "pixel": geometry["pixel"],
            "latitude": geometry["latitude"],
            "longitude": geometry["longitude"],
        },
        attrs={"title": "Paired Satellite Swath Data", "geometry": "swath"},
    )

    return paired


def create_paired_gridded_data(
    variables: list[str] | None = None,
    bias_range: float = 0.3,
    seed: int = 42,
) -> xr.Dataset:
    """Create paired x-y gridded (L3) data.

    Parameters
    ----------
    variables
        Variables to include. Defaults to ["NO2"].
    bias_range
        Range of spatially-varying bias (multiplicative).
        Creates diagonal gradient pattern across domain.
    seed
        Random seed.

    Returns
    -------
    xr.Dataset
        Paired dataset with time, lat, lon dimensions.
    """
    if variables is None:
        variables = ["NO2"]

    domain = Domain(lat_min=25, lat_max=50, lon_min=-125, lon_max=-65, n_lat=25, n_lon=60)
    time_config = TimeConfig(start="2024-07-01", end="2024-07-08", freq="1D")

    geometry = create_gridded_geometries(
        variables=variables,
        domain=domain,
        time_config=time_config,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 1)

    # Create spatially-varying bias: diagonal pattern (SW negative, NE positive)
    lats = geometry["lat"].values
    lons = geometry["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Normalize to 0-1
    lat_norm = (lat_grid - lat_grid.min()) / (lat_grid.max() - lat_grid.min())
    lon_norm = (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min())

    # Diagonal pattern: combine lat and lon
    spatial_bias_2d = bias_range * (lat_norm + lon_norm - 1)  # -bias_range to +bias_range

    data_vars = {}
    for var in variables:
        x_data = geometry[var].values  # shape: (time, lat, lon)
        # Broadcast spatial bias across time dimension
        spatial_bias_3d = np.broadcast_to(spatial_bias_2d, x_data.shape)
        y_data = x_data * (1.0 + spatial_bias_3d) + rng.normal(0, 0.2, x_data.shape)
        y_data = np.clip(y_data, 0, None)

        data_vars[f"x_{var.lower()}"] = (["time", "lat", "lon"], x_data.astype(np.float32))
        data_vars[f"y_{var.lower()}"] = (["time", "lat", "lon"], y_data.astype(np.float32))

    paired = xr.Dataset(
        data_vars,
        coords={
            "time": geometry["time"],
            "lat": geometry["lat"],
            "lon": geometry["lon"],
        },
        attrs={"title": "Paired Gridded Data (L3)", "geometry": "grid"},
    )

    return paired
