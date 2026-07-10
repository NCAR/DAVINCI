"""Synthetic native MMR inputs and independent optical closure fields."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_contracts import GRAVITY, SyntheticTuningSpec, named_rng
from davinci_monet.tests.synthetic._aerosol_inputs import attrs
from davinci_monet.tests.synthetic._aerosol_oracles import (
    extinction_coefficients,
    log_time_interpolation_oracle,
    optical_aod_oracle,
)


def make_mmr(
    spec: SyntheticTuningSpec,
    hash_value: str,
    daily_time: pd.DatetimeIndex,
    native_lat: np.ndarray,
    native_lon: np.ndarray,
    native_model_daily: np.ndarray,
    r_native_daily: np.ndarray,
) -> tuple[dict[str, xr.Dataset], xr.Dataset]:
    selected_days = daily_time[2 : 2 + spec.mmr_days].normalize()
    mmr_time = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(day, day + pd.Timedelta(hours=21), freq="3h").values
                for day in selected_days
            ]
        )
    )
    model_at_mmr = (
        log_time_interpolation_oracle(native_model_daily + spec.log_epsilon, daily_time, mmr_time)
        - spec.log_epsilon
    )
    model_at_mmr = np.maximum(model_at_mmr, 0.0)
    ratio_at_mmr = log_time_interpolation_oracle(r_native_daily, daily_time, mmr_time)
    lev = np.array([20000.0, 50000.0, 80000.0, 100000.0])
    dp_1d = np.array([12000.0, 18000.0, 30000.0, 40000.0])
    t_index = np.arange(mmr_time.size)[:, None, None, None]
    lat_rad = np.deg2rad(native_lat)[None, None, :, None]
    lon_rad = np.deg2rad(native_lon)[None, None, None, :]
    level_index = np.arange(lev.size)[None, :, None, None]
    rh = 0.42 + 0.16 * np.sin(2.0 * np.pi * t_index / 8.0 + lon_rad)
    rh = rh + 0.08 * np.cos(lat_rad) - 0.035 * level_index
    rh = np.broadcast_to(rh, (mmr_time.size, lev.size, native_lat.size, native_lon.size)).copy()
    rh = np.clip(rh, 0.08, 0.92)
    dp = np.broadcast_to(
        dp_1d[None, :, None, None],
        (mmr_time.size, lev.size, native_lat.size, native_lon.size),
    ).copy()
    rng = named_rng(spec.master_seed, "mmr_perturbations")
    profile = rng.uniform(0.7, 1.3, size=(len(spec.species), lev.size))
    profile *= np.linspace(0.65, 1.35, lev.size)[None, :]
    profile *= np.arange(1, len(spec.species) + 1)[:, None]
    profile /= profile.sum()
    mixing_ratio = np.broadcast_to(
        profile[:, None, :, None, None],
        (len(spec.species), mmr_time.size, lev.size, native_lat.size, native_lon.size),
    ).copy()
    kappa = extinction_coefficients(rh, len(spec.species))
    raw_aod = np.sum(kappa * mixing_ratio * dp[None, ...] / GRAVITY, axis=(0, 2))
    mixing_ratio *= (model_at_mmr / raw_aod)[:, None, :, :][None, ...]

    variables: dict[str, Any] = {
        "RH": (("time", "lev", "lat", "lon"), rh.astype(np.float32), {"units": "1"}),
        "DELP": (("time", "lev", "lat", "lon"), dp.astype(np.float32), {"units": "Pa"}),
        "T": (
            ("time", "lev", "lat", "lon"),
            (
                245.0 + 12.0 * level_index + 2.0 * np.cos(lat_rad) + 0.0 * t_index + 0.0 * lon_rad
            ).astype(np.float32),
            {"units": "K", "long_name": "unrelated synthetic temperature"},
        ),
        "ORO": (
            ("lat", "lon"),
            (
                700.0
                * np.maximum(
                    0.0,
                    np.cos(np.deg2rad(native_lat))[:, None]
                    * np.cos(np.deg2rad(native_lon))[None, :],
                )
            ).astype(np.float32),
            {"units": "m"},
        ),
    }
    for species_index, name in enumerate(spec.species):
        variables[name] = (
            ("time", "lev", "lat", "lon"),
            mixing_ratio[species_index].astype(np.float32),
            {"units": "kg kg-1", "synthetic_role": "scaled_aerosol"},
        )
    for gas_index, name in enumerate(("SO2", "DMS", "MSA")):
        gas = (
            (gas_index + 1)
            * 1.0e-10
            * (
                1.0
                + 0.03
                * rng.normal(size=(mmr_time.size, lev.size, native_lat.size, native_lon.size))
            )
        )
        variables[name] = (
            ("time", "lev", "lat", "lon"),
            gas.astype(np.float32),
            {"units": "kg kg-1", "synthetic_role": "unscaled_gas"},
        )
    combined = xr.Dataset(
        variables,
        coords={"time": mmr_time.values, "lev": lev, "lat": native_lat, "lon": native_lon},
        attrs=attrs(spec, hash_value, "fitting_input:mmr"),
    )
    combined["lev"].attrs = {
        "units": "Pa",
        "positive": "down",
        "note": "pressure increases with index; surface is last",
    }
    for name in (*spec.species, "SO2", "DMS", "MSA"):
        combined[name].encoding["_FillValue"] = np.float32(-9.999e15)
    if spec.scenario == "writer_ci":
        combined[spec.species[0]].values[0, 0, 0, 0] = np.nan
        combined["SO2"].values[0, 0, 0, 1] = np.nan

    serialized_kappa = extinction_coefficients(
        np.asarray(combined["RH"], dtype=np.float64), len(spec.species)
    )
    baseline = optical_aod_oracle(combined, spec.species)
    scaled = combined.copy(deep=True)
    ratio_da = xr.DataArray(
        ratio_at_mmr,
        dims=("time", "lat", "lon"),
        coords={"time": combined["time"], "lat": native_lat, "lon": native_lon},
    )
    for name in spec.species:
        scaled[name] = scaled[name] * ratio_da
    scaled_aod = optical_aod_oracle(scaled, spec.species)

    mmr: dict[str, xr.Dataset] = {}
    for day in selected_days:
        key = str(day.date())
        day_slice = combined.sel(time=str(day.date())).copy(deep=True)
        day_slice.attrs = attrs(spec, hash_value, f"fitting_input:mmr:{key}")
        mmr[key] = day_slice
    optical_truth = xr.Dataset(
        {
            "kappa": (
                ("species", "mmr_time", "lev", "native_lat", "native_lon"),
                serialized_kappa,
                {"units": "m2 kg-1"},
            ),
            "layer_weight": (
                ("mmr_time", "lev", "native_lat", "native_lon"),
                dp / GRAVITY,
                {"units": "kg m-2"},
            ),
            "baseline_optical_aod": (
                ("mmr_time", "native_lat", "native_lon"),
                baseline.values,
                {"units": "1"},
            ),
            "scaled_optical_aod": (
                ("mmr_time", "native_lat", "native_lon"),
                scaled_aod.values,
                {"units": "1"},
            ),
            "r_3hour_true": (
                ("mmr_time", "native_lat", "native_lon"),
                ratio_at_mmr,
                {"units": "1"},
            ),
        },
        coords={
            "species": list(spec.species),
            "mmr_time": mmr_time.values,
            "lev": lev,
            "native_lat": native_lat,
            "native_lon": native_lon,
        },
    )
    return mmr, optical_truth
