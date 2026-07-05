"""WRF-Chem model reader.

This module provides the WRFChemReader class for reading Weather Research and
Forecasting with Chemistry (WRF-Chem) model output files.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    promote_to_coords,
    retry_transient_open,
    standardize_dims,
    validate_file_list,
)

# monetio's WRF-Chem reader accepts these kwargs but xarray.open_dataset does
# not. When the monetio path fails (e.g. wrf-python / netCDF4 version
# incompatibility) and the reader falls back to xarray, these are stripped to
# avoid TypeError from xarray's backend.
_MONETIO_ONLY_KWARGS = ("mech", "convert_to_ppb", "surf_only", "surf_only_nc")


@source_registry.register("wrfchem")
class WRFChemReader:
    """Reader for WRF-Chem model output.

    Reads WRF-Chem output files (wrfout_*), handling various chemical
    mechanisms and output configurations.

    Examples
    --------
    >>> reader = WRFChemReader()
    >>> ds = reader.open(["wrfout_d01_2024-01-01_00:00:00"])
    >>> print(ds.dims)
    Frozen({'time': 24, 'z': 35, 'lat': 100, 'lon': 120})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "wrfchem"

    @property
    def geometry(self) -> DataGeometry:
        """Dataset output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem output files.

        Parameters
        ----------
        file_paths
            Paths to WRF-Chem output files (wrfout_*).
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options passed to monetio or xarray.

        Returns
        -------
        xr.Dataset
            WRF-Chem data with standardized dimensions (time, z, lat, lon).
        """
        file_list = validate_file_list(file_paths, source_label="WRF-Chem")

        # Try monetio first; fall back to plain xarray if monetio's wrf-python
        # path isn't usable (e.g. wrf-python ↔ netCDF4 incompatibility raising
        # NotImplementedError("Dataset is not picklable") from wrf-python's
        # internal copy.copy on a netCDF4.Dataset).
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except (ImportError, NotImplementedError) as e:
            xarray_kwargs = {k: v for k, v in kwargs.items() if k not in _MONETIO_ONLY_KWARGS}
            dropped = [k for k in kwargs if k in _MONETIO_ONLY_KWARGS]
            warnings.warn(
                "monetio WRF-Chem reader unavailable "
                f"({type(e).__name__}: {e}); falling back to xarray. "
                f"Dropped monetio-only kwargs: {dropped}. "
                "Raw WRF-Chem variables are returned without mech-aware "
                "decoding — e.g. `o3` will be in ppmv (not ppb) and derived "
                "diagnostics will not be added. Use `unit_scale` in the "
                "config to compensate.",
                UserWarning,
                stacklevel=2,
            )
            ds = self._open_with_xarray(file_list, variables, **xarray_kwargs)

        # Standardize dimensions
        ds = self._standardize_dataset(ds)

        # Drop timesteps where chemistry diagnostics are identically zero.
        # In the operational AQ_WATCH cycle the hour-0 wrfout is an IC dump
        # written before any chemistry tendency step, so PM2_5_DRY/PM10 are
        # exactly zero across the entire grid. Pairing such steps against
        # surface geometry silently biases stats negative and produces a 0-to-real
        # discontinuity in timeseries plots.
        ds = self._drop_uninitialized_chem_steps(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem files using monetio.

        Parameters
        ----------
        file_paths
            WRF-Chem file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw WRF-Chem dataset.
        """
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._wrfchem_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WRF-Chem files using xarray.

        Parameters
        ----------
        file_paths
            WRF-Chem file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw WRF-Chem dataset.
        """

        def _open() -> xr.Dataset:
            if len(file_paths) > 1:
                # WRF files store time as the `Times` char-array (not a
                # coord variable on `Time`), so combine="by_coords" cannot
                # infer a concat dim. Use nested concat along Time, with
                # file paths sorted to provide chronological order.
                ds = xr.open_mfdataset(
                    [str(f) for f in sorted(file_paths, key=str)],
                    combine="nested",
                    concat_dim="Time",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_paths[0]), **kwargs)

            if variables is not None:
                available = [v for v in variables if v in ds.data_vars]
                if available:
                    # Preserve WRF housekeeping variables that
                    # _standardize_dataset relies on (Times is the
                    # char-array time encoding; XLAT/XLONG are the lat/
                    # lon coords). These get dropped or set as coords by
                    # standardize_dataset after the user's selection.
                    keep_aux = [v for v in ("Times", "XLAT", "XLONG") if v in ds.variables]
                    ds = ds[available + [v for v in keep_aux if v not in available]]

            return ds

        return retry_transient_open(_open, context="Opening WRF-Chem files")

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize WRF-Chem dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw WRF-Chem dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset.
        """
        # WRF dimension renames
        ds = standardize_dims(
            ds,
            {
                "Time": "time",
                "bottom_top": "z",
                "south_north": "y",
                "west_east": "x",
                "bottom_top_stag": "z_stag",
                "south_north_stag": "y_stag",
                "west_east_stag": "x_stag",
            },
        )

        # Decode WRF's `Times` char-array variable into a datetime coord on
        # the `time` dim. monetio's path normally handles this; in the xarray
        # fallback we have to do it ourselves so downstream pairing can align
        # dataset times with dataset times.
        if "Times" in ds.variables and "time" in ds.dims and "time" not in ds.coords:
            times_bytes = ds["Times"].values
            try:
                time_strs = [
                    t.decode("ascii") if isinstance(t, bytes) else str(t)
                    for t in times_bytes.tolist()
                ]
                # WRF format: 'YYYY-MM-DD_HH:MM:SS' — replace '_' for parsing
                import numpy as _np

                times_np = _np.array(
                    [_np.datetime64(s.replace("_", "T"), "ns") for s in time_strs],
                    dtype="datetime64[ns]",
                )
                ds = ds.assign_coords(time=("time", times_np))
                ds = ds.drop_vars("Times")
            except (ValueError, TypeError, AttributeError):
                # If decoding fails, leave Times alone — better than crashing
                pass

        # Handle WRF lat/lon (XLAT, XLONG). WRF replicates these across the
        # Time dim even though they are static; squeeze that out so downstream
        # pairing strategies see 2D (y, x) lat/lon as expected.
        def _squeeze_time(da: xr.DataArray) -> xr.DataArray:
            if "time" in da.dims and da.sizes["time"] > 0:
                return da.isel(time=0, drop=True)
            return da

        if "XLAT" in ds.data_vars or "XLAT" in ds.coords:
            if "XLAT" in ds.data_vars:
                ds = ds.set_coords("XLAT")
            if "lat" not in ds.coords:
                ds = ds.assign_coords(lat=_squeeze_time(ds["XLAT"]))

        if "XLONG" in ds.data_vars or "XLONG" in ds.coords:
            if "XLONG" in ds.data_vars:
                ds = ds.set_coords("XLONG")
            if "lon" not in ds.coords:
                ds = ds.assign_coords(lon=_squeeze_time(ds["XLONG"]))

        # Handle latitude/longitude if present
        ds = promote_to_coords(ds, ("latitude", "longitude"))

        return ds

    # Chemistry diagnostics computed during the chemistry tendency step.
    # These are exactly zero in the hour-0 IC dump and become populated only
    # after the first chemistry step. Used by _drop_uninitialized_chem_steps.
    _CHEM_DIAGNOSTICS: tuple[str, ...] = ("PM2_5_DRY", "PM10")

    def _drop_uninitialized_chem_steps(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop timesteps where chemistry diagnostics are identically zero.

        WRF-Chem diagnostics like ``PM2_5_DRY`` and ``PM10`` are computed
        inside the chemistry tendency routine. The hour-0 wrfout file is
        written from initial conditions before any chemistry step has run,
        so these diagnostics are exactly zero across the entire grid. Other
        timesteps in the cycle are populated normally.

        Parameters
        ----------
        ds
            Standardized dataset (post :meth:`_standardize_dataset`).

        Returns
        -------
        xr.Dataset
            Dataset with all-zero diagnostic timesteps removed. Returns the
            input unchanged when no chemistry diagnostic is present or all
            timesteps look populated.
        """
        if "time" not in ds.dims:
            return ds

        # Find timesteps where ANY tracked diagnostic is all-zero across its
        # non-time dimensions.
        import numpy as np

        bad_steps: set[int] = set()
        triggering_var: str | None = None
        for diag in self._CHEM_DIAGNOSTICS:
            if diag not in ds.data_vars:
                continue
            da = ds[diag]
            other_dims = [d for d in da.dims if d != "time"]
            if not other_dims:
                continue
            max_per_t = da.max(dim=other_dims).values
            zero_t = np.where(max_per_t == 0)[0]
            if zero_t.size:
                bad_steps.update(int(i) for i in zero_t)
                if triggering_var is None:
                    triggering_var = diag

        if not bad_steps:
            return ds

        keep = np.array([i for i in range(ds.sizes["time"]) if i not in bad_steps], dtype=int)
        bad_times = ds["time"].values[sorted(bad_steps)]
        warnings.warn(
            f"WRF-Chem: dropping {len(bad_steps)} timestep(s) where "
            f"{triggering_var} is identically zero across the grid. This is "
            f"the hour-0 IC dump before any chemistry tendency step has run; "
            f"the diagnostic is not populated. Dropped times: "
            f"{[str(t) for t in bad_times]}. For analysis that needs these "
            f"hours, supply the previous cycle's +24h forecast file instead.",
            UserWarning,
            stacklevel=2,
        )
        return ds.isel(time=keep)
