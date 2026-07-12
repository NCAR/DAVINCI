"""Fit-window selection and provenance helpers for EOF analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import EOFSpec


def fit_time_text(value: Any) -> str:
    """Render a fit boundary with stable second precision."""
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="s"))
    return str(value)


def _fit_from_artifact(
    data: xr.DataArray, artifact: xr.Dataset, fit_split: str
) -> tuple[xr.DataArray, str]:
    if "fit_mask" in artifact and "split" in artifact:
        raise ValueError(
            "EOF fit_artifact must contain exactly one selector; found both "
            "'fit_mask' and 'split'"
        )
    if "fit_mask" in artifact:
        selector = artifact["fit_mask"]
        selector_kind = "fit_mask"
        selection = "fit_mask"
    elif "split" in artifact:
        selector = artifact["split"]
        selector_kind = "split"
        selection = fit_split
    else:
        raise ValueError("EOF fit_artifact must contain a time-indexed 'fit_mask' or 'split'")
    if selector.dims != ("time",):
        raise ValueError("EOF fit_artifact selector must have exactly the ('time',) dimension")

    aligned = selector.reindex(time=data["time"])
    values = np.asarray(aligned.values)
    if selector_kind == "fit_mask" and values.dtype.kind in "biu":
        keep = values.astype(bool)
    elif selector_kind == "fit_mask" and values.dtype.kind in "fc":
        keep = np.isfinite(values) & (values != 0)
    elif selector_kind == "fit_mask":
        keep = np.asarray([value is True for value in values], dtype=bool)
    else:
        keep = values.astype(str) == fit_split
    return data.isel(time=np.flatnonzero(keep)), selection


def select_fit_data(
    data: xr.DataArray, spec: EOFSpec, fit_artifact: xr.Dataset | None
) -> tuple[xr.DataArray, str]:
    """Select the immutable EOF fit subset from a window or split artifact."""
    if "time" not in data.dims:
        raise ValueError("EOF variable must have a time dimension")
    if spec.fit_window is not None:
        fit = data.sel(time=slice(spec.fit_window.start, spec.fit_window.end))
        selection = "window"
    elif spec.fit_artifact is not None:
        if fit_artifact is None:
            raise ValueError("EOF fit_artifact requires the named artifact input")
        fit, selection = _fit_from_artifact(data, fit_artifact, spec.fit_split)
    else:
        fit = data
        selection = "all"
    if int(fit.sizes.get("time", 0)) < 2:
        raise ValueError("EOF fit selection must contain at least two time samples")
    return fit, selection


def _safe_metadata_value(value: Any) -> str | int | float | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return value
    return None


def copy_input_metadata(output: xr.Dataset, data: xr.Dataset, field: xr.DataArray) -> None:
    """Copy the input hashes and preprocessing contract into EOF provenance."""
    metadata = dict(data.attrs)
    metadata.update(field.attrs)
    explicit = {
        "grid_cell_convention",
        "grid_resolution",
        "log_epsilon",
        "regrid_method",
    }
    for raw_key, raw_value in metadata.items():
        key = str(raw_key)
        if not (key.endswith("_hash") or key in explicit):
            continue
        value = _safe_metadata_value(raw_value)
        if value is not None:
            output.attrs[f"eof_input_{key}"] = value


def copy_fit_artifact_metadata(output: xr.Dataset, artifact: xr.Dataset) -> None:
    """Copy immutable fit-artifact hashes into EOF provenance."""
    for raw_key, raw_value in artifact.attrs.items():
        key = str(raw_key)
        if not key.endswith("_hash"):
            continue
        value = _safe_metadata_value(raw_value)
        if value is not None:
            output.attrs[f"eof_fit_artifact_{key}"] = value


__all__ = [
    "copy_fit_artifact_metadata",
    "copy_input_metadata",
    "fit_time_text",
    "select_fit_data",
]
