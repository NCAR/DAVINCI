from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.gridded_reductions import product_summary


@dataclass(frozen=True)
class ProductArtifactResult:
    product: str
    analysis_path: Path
    summary_path: Path


def _field_arrays(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    return {str(name): ds[name] for name in ds.data_vars}


def _netcdf_safe_attrs(attrs: dict[Any, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, bool):
            safe[str(key)] = str(value)
        else:
            safe[str(key)] = value
    return safe


def _netcdf_safe_dataset(ds: xr.Dataset) -> xr.Dataset:
    writable = ds.copy(deep=False)
    writable.attrs = _netcdf_safe_attrs(dict(ds.attrs))
    for name in writable.data_vars:
        writable[name].attrs = _netcdf_safe_attrs(dict(writable[name].attrs))
    for name in writable.coords:
        writable.coords[name].attrs = _netcdf_safe_attrs(dict(writable.coords[name].attrs))
    return writable


def write_product_artifacts(
    output_dir: str | Path, product: str, ds: xr.Dataset
) -> ProductArtifactResult:
    root = Path(output_dir) / "products" / product
    root.mkdir(parents=True, exist_ok=True)
    analysis_path = root / "analysis.nc"
    summary_path = root / "summary.json"
    encoding: dict[str, dict[str, Any]] = {}
    for raw_name, da in ds.data_vars.items():
        name = str(raw_name)
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
    _netcdf_safe_dataset(ds).to_netcdf(analysis_path, engine="netcdf4", encoding=encoding)
    summary = {
        "product": product,
        "attrs": {str(k): str(v) for k, v in ds.attrs.items()},
        "groups": [str(v) for v in ds["group"].values] if "group" in ds.coords else [],
        "fields": product_summary(_field_arrays(ds)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return ProductArtifactResult(
        product=product, analysis_path=analysis_path, summary_path=summary_path
    )


def load_product_dataset(path: str | Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds:
        return ds.load()
