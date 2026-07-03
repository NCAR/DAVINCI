"""Intermediate-gridding pairing strategy using numba-accelerated binning.

Bins satellite L2 swath pixels onto a uniform (time, lon, lat) grid,
then pairs with the y data on the same grid. This is the recommended
strategy for all L2 satellite products at scale.

Ported from the MELODIES-MONET intermediate grid approach
(grid_util.update_data_grid).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.coordinates import (
    match_longitude_convention,
    normalize_post_load_coordinates,
    vertical_dim_name,
    wrap_longitudes,
)
from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.grid_binning import bin_swath_to_grid, edges_from_centers, normalize_grid
from davinci_monet.pairing.strategies.base import BasePairingStrategy

_UNIT_TO_M = {"m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0, "ft": 0.3048, "feet": 0.3048}
# ``z`` is included because the generic reader standardizes ``altitude`` ->
# ``z`` (COMMON_COORDINATE_ALIASES); the native-altitude branch still gates on
# length units, so a ``z`` that is a pressure/sigma level (no length units)
# falls through to the pressure branch or the no-vertical error.
_ALT_NAMES = {"altitude", "alt", "height", "geometric_height", "z"}
_GEOPOT_NAMES = {"z3", "zg", "geopotential_height", "geopotential_height_msl"}
_PRESSURE_NAMES = {"lev", "level", "plev", "pressure", "p"}
_PRESSURE_HPA_UNITS = {"hpa", "mb", "millibar", "hectopascal"}
_PRESSURE_PA_UNITS = {"pa", "pascal"}


class IntermediateGridStrategy(BasePairingStrategy):
    """Pairing strategy that bins satellite swath data onto a uniform grid.

    The strategy:
    1. Defines a target grid (from the y source, resolution, or explicit dims)
    2. Bins all swath pixels into grid cells using numba-accelerated loop
    3. Normalizes (mean = sum / count)
    4. Aligns the y data on the same grid
    5. Returns paired dataset with x, y, and pixel counts

    Examples
    --------
    >>> strategy = IntermediateGridStrategy()
    >>> paired = strategy.pair_sources(
    ...     y_data, satellite_data,
    ...     grid_mode="match_dataset",
    ...     time_resolution="1D",
    ... )
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return SWATH geometry."""
        return DataGeometry.SWATH

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair swath data with the y source by binning onto a common grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with swath dimensions and 2D lat/lon.
        y_data
            The y source. Dataset with dims (time, [z], lat, lon).
        radius_of_influence
            Not used (kept for interface compatibility).
        time_tolerance
            Not used (temporal binning is controlled by time_resolution).
        vertical_method
            Not used.
        horizontal_method
            Not used.
        **kwargs
            Strategy-specific options:

            - grid_mode : str
                Grid definition mode: "match_dataset", "resolution", or
                "explicit". Default "match_dataset".
            - time_resolution : str
                Pandas frequency string for temporal binning (e.g. "1D").
                Default "1D".
            - resolution : float
                Grid spacing in degrees (for grid_mode="resolution").
            - ntime, nlat, nlon : int
                Explicit grid dimensions (for grid_mode="explicit").
            - min_sample_count : int
                Minimum pixel count per cell. Cells below this are
                masked to NaN. Default 1.
            - x_var : str
                The x variable name to bin.
            - y_var : str
                The y variable name to extract.

        Returns
        -------
        xr.Dataset
            Paired dataset on common grid with x, y, and count
            variables.
        """
        x_data = normalize_post_load_coordinates(x_data)
        y_data = normalize_post_load_coordinates(y_data)

        if kwargs.get("horizontal_res") is not None:
            return self._pair_symmetric(
                x_data,
                x_data_var=kwargs.get("x_var"),
                y_data=y_data,
                y_data_var=kwargs.get("y_var"),
                x_source=kwargs.get("x_source"),
                y_source=kwargs.get("y_source"),
                horizontal_res=float(kwargs["horizontal_res"]),
                extent=kwargs.get("extent"),
                time_resolution=kwargs.get("time_resolution", "1D"),
                min_sample_count=int(kwargs.get("min_sample_count", 1)),
                vertical=kwargs.get("vertical"),
            )

        grid_mode = kwargs.get("grid_mode", "match_dataset")
        time_resolution = kwargs.get("time_resolution", "1D")
        min_sample_count = kwargs.get("min_sample_count", 1)
        x_var = kwargs.get("x_var") or kwargs.get("x_var")
        y_var = kwargs.get("y_var") or kwargs.get("y_var")

        # Extract surface if the y source has a vertical dimension
        y_proc = y_data
        dim_name = vertical_dim_name(y_proc)
        if dim_name is not None:
            y_proc = self._extract_surface(y_proc, dim_name)

        # Get y coordinates
        y_lat, y_lon = self._get_y_coords(y_proc)

        # Determine analysis time range from the y source
        if "time" in y_proc.dims:
            y_times = pd.DatetimeIndex(y_proc["time"].values)
            t_start = y_times.min()
            t_end = y_times.max()
        else:
            raise PairingError("The y source must have a time dimension")

        # Build target grid (filter kwargs to avoid duplicating named params)
        consumed = {
            "grid_mode",
            "time_resolution",
            "min_sample_count",
            "x_var",
            "y_var",
            "x_var",
            "y_var",
        }
        grid_kwargs = {k: v for k, v in kwargs.items() if k not in consumed}
        time_edges, lon_edges, lat_edges, time_centers, lon_centers, lat_centers = self._build_grid(
            grid_mode=grid_mode,
            y_lat=y_lat,
            y_lon=y_lon,
            t_start=t_start,
            t_end=t_end,
            time_resolution=time_resolution,
            **grid_kwargs,
        )

        ntime = len(time_centers)
        nlon = len(lon_centers)
        nlat = len(lat_centers)

        # Determine which x variable to bin
        if x_var is None:
            data_var_names = list(x_data.data_vars)
            if len(data_var_names) == 0:
                raise PairingError("No data variables in the x source")
            x_var = data_var_names[0]

        # Extract flat arrays from the x source
        x_lat, x_lon = self._get_x_coords(x_data)
        lat_flat = x_lat.values.flatten().astype(np.float64)
        lon_flat = x_lon.values.flatten().astype(np.float64)
        data_flat = x_data[x_var].values.flatten().astype(np.float64)

        lon_flat = match_longitude_convention(lon_flat, lon_edges)

        # Get x timestamps as epoch seconds, aligned to the same
        # flattening order as ``data_flat`` (i.e. the binned x variable).
        time_flat = self._get_x_timestamps(x_data, len(data_flat), align_var=x_var)

        # Allocate accumulation arrays
        count_grid = np.zeros((ntime, nlon, nlat), dtype=np.int32)
        data_grid = np.zeros((ntime, nlon, nlat), dtype=np.float64)

        # Bin swath pixels onto grid (numba fast path)
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            time_flat,
            lon_flat,
            lat_flat,
            data_flat,
            count_grid,
            data_grid,
        )

        # Normalize: sum → mean
        normalize_grid(count_grid, data_grid)

        # Apply min_sample_count filter
        if min_sample_count > 1:
            data_grid[count_grid < min_sample_count] = np.nan

        # Build datetime coordinates from time centers (epoch seconds)
        time_coords = pd.to_datetime(time_centers, unit="s")

        # Create gridded x dataset
        x_gridded = xr.Dataset(
            {
                f"x_{x_var}": (["time", "lon", "lat"], data_grid.astype(np.float32)),
                "sample_count": (["time", "lon", "lat"], count_grid),
            },
            coords={
                "time": time_coords,
                "lon": lon_centers,
                "lat": lat_centers,
            },
        )

        # Extract y variable on same grid
        if y_var is None:
            y_data_vars = list(y_proc.data_vars)
            if len(y_data_vars) == 0:
                raise PairingError("No data variables in the y source")
            y_var = str(y_data_vars[0])

        y_on_grid = self._align_y_to_grid(
            y_proc,
            y_var,
            time_coords,
            lon_centers,
            lat_centers,
            y_lat,
            y_lon,
            grid_mode,
        )

        # Create paired output
        paired = x_gridded.copy()
        paired[f"y_{y_var}"] = y_on_grid

        return paired

    def _pair_symmetric(
        self,
        x_data: xr.Dataset,
        *,
        x_data_var: str | None,
        y_data: xr.Dataset,
        y_data_var: str | None,
        x_source: str | None,
        y_source: str | None,
        horizontal_res: float,
        extent: tuple[float, float, float, float] | None,
        time_resolution: str,
        min_sample_count: int,
        vertical: Any = None,
    ) -> xr.Dataset:
        """Bin BOTH sources onto a common uniform (time, lon, lat) grid and pair."""
        if vertical is not None:
            return self._pair_symmetric_3d(
                x_data,
                x_data_var=x_data_var,
                y_data=y_data,
                y_data_var=y_data_var,
                x_source=x_source,
                y_source=y_source,
                horizontal_res=horizontal_res,
                extent=extent,
                time_resolution=time_resolution,
                min_sample_count=min_sample_count,
                vertical=vertical,
            )

        x_var = x_data_var or str(list(x_data.data_vars)[0])
        y_var = y_data_var or str(list(y_data.data_vars)[0])
        # Phase 1 is 2-D: reduce any vertical dim to the surface for both sources.
        x_proc = self._reduce_to_surface(x_data)
        y_proc = self._reduce_to_surface(y_data)

        lon_centers, lat_centers, lon_edges, lat_edges = self._uniform_horizontal_grid(
            [x_proc, y_proc], horizontal_res, extent
        )
        time_centers_epoch, time_edges, time_coords = self._uniform_time_grid(
            [x_proc, y_proc], time_resolution
        )

        x_grid, x_count = self._bin_one_source(
            x_proc,
            x_var,
            time_edges,
            lon_edges,
            lat_edges,
            len(time_centers_epoch),
            len(lon_centers),
            len(lat_centers),
            min_sample_count,
        )
        y_grid, y_count = self._bin_one_source(
            y_proc,
            y_var,
            time_edges,
            lon_edges,
            lat_edges,
            len(time_centers_epoch),
            len(lon_centers),
            len(lat_centers),
            min_sample_count,
        )

        paired = xr.Dataset(
            {
                f"x_{x_var}": (["time", "lon", "lat"], x_grid.astype(np.float32)),
                f"y_{y_var}": (["time", "lon", "lat"], y_grid.astype(np.float32)),
                "x_sample_count": (["time", "lon", "lat"], x_count),
                "y_sample_count": (["time", "lon", "lat"], y_count),
            },
            coords={"time": time_coords, "lon": lon_centers, "lat": lat_centers},
        )
        paired[f"x_{x_var}"].attrs.update({"axis": "x", "source_label": x_source or ""})
        paired[f"y_{y_var}"].attrs.update({"axis": "y", "source_label": y_source or ""})
        paired.attrs.update({"created_by": "davinci_monet", "paired": True})
        return paired

    def _pair_symmetric_3d(
        self,
        x_data: xr.Dataset,
        *,
        x_data_var: str | None,
        y_data: xr.Dataset,
        y_data_var: str | None,
        x_source: str | None,
        y_source: str | None,
        horizontal_res: float,
        extent: tuple[float, float, float, float] | None,
        time_resolution: str,
        min_sample_count: int,
        vertical: Any,
    ) -> xr.Dataset:
        """Bin BOTH sources onto a common (time, lon, lat, alt) grid and pair."""
        x_var = x_data_var or str(list(x_data.data_vars)[0])
        y_var = y_data_var or str(list(y_data.data_vars)[0])

        # ``vertical`` arrives as a plain dict at runtime (config is re-dumped to
        # dicts in the pipeline); handle the model form defensively as well.
        if isinstance(vertical, dict):
            units = str(vertical.get("units", "m"))
            vres = float(vertical["res"])
            vextent = vertical.get("extent")
        else:
            units = str(getattr(vertical, "units", "m"))
            vres = float(vertical.res)
            vextent = getattr(vertical, "extent", None)

        x_alt = self._source_altitude(x_data, x_var, units)
        y_alt = self._source_altitude(y_data, y_var, units)

        lon_centers, lat_centers, lon_edges, lat_edges = self._uniform_horizontal_grid(
            [x_data, y_data], horizontal_res, extent
        )
        time_centers, time_edges, time_coords = self._uniform_time_grid(
            [x_data, y_data], time_resolution
        )
        alt_centers, alt_edges = self._uniform_vertical_grid([x_alt, y_alt], vres, vextent)

        shape = (len(time_centers), len(lon_centers), len(lat_centers), len(alt_centers))
        xg, xc = self._bin_one_source_3d(
            x_data,
            x_var,
            x_alt,
            time_edges,
            lon_edges,
            lat_edges,
            alt_edges,
            shape,
            min_sample_count,
        )
        yg, yc = self._bin_one_source_3d(
            y_data,
            y_var,
            y_alt,
            time_edges,
            lon_edges,
            lat_edges,
            alt_edges,
            shape,
            min_sample_count,
        )
        dims = ["time", "lon", "lat", "alt"]
        paired = xr.Dataset(
            {
                f"x_{x_var}": (dims, xg.astype(np.float32)),
                f"y_{y_var}": (dims, yg.astype(np.float32)),
                "x_sample_count": (dims, xc),
                "y_sample_count": (dims, yc),
            },
            coords={
                "time": time_coords,
                "lon": lon_centers,
                "lat": lat_centers,
                "alt": alt_centers,
            },
        )
        paired["alt"].attrs["units"] = units
        paired[f"x_{x_var}"].attrs.update({"axis": "x", "source_label": x_source or ""})
        paired[f"y_{y_var}"].attrs.update({"axis": "y", "source_label": y_source or ""})
        paired.attrs.update({"created_by": "davinci_monet", "paired": True})
        return paired

    def _bin_one_source_3d(
        self,
        ds: xr.Dataset,
        var: str,
        alt: xr.DataArray,
        time_edges: np.ndarray,
        lon_edges: np.ndarray,
        lat_edges: np.ndarray,
        alt_edges: np.ndarray,
        shape: tuple[int, int, int, int],
        min_sample_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from davinci_monet.pairing.grid_binning import bin_points_to_grid_4d

        time_flat, lon_flat, lat_flat, data_flat = self._flatten_to_points(ds, var)
        da = ds[var]
        # IMPORTANT: build ``alt_flat`` with the IDENTICAL broadcast/transpose/
        # flatten pattern as the lat/lon flatten in ``_flatten_to_points`` so the
        # i-th altitude aligns element-wise with the i-th value/lat/lon/time.
        alt_flat = alt.broadcast_like(da).transpose(*da.dims).values.astype(np.float64).flatten()
        lon_flat = match_longitude_convention(lon_flat, lon_edges)
        count = np.zeros(shape, dtype=np.int32)
        acc = np.zeros(shape, dtype=np.float64)
        bin_points_to_grid_4d(
            time_edges,
            lon_edges,
            lat_edges,
            alt_edges,
            time_flat,
            lon_flat,
            lat_flat,
            alt_flat,
            data_flat,
            count,
            acc,
        )
        normalize_grid(count, acc)
        if min_sample_count > 1:
            acc[count < min_sample_count] = np.nan
        return acc, count

    def _uniform_vertical_grid(
        self, alt_arrays: list[xr.DataArray], res: float, extent: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if extent is not None:
            z0, z1 = float(extent[0]), float(extent[1])
        else:
            mins = [float(np.nanmin(a.values)) for a in alt_arrays]
            maxs = [float(np.nanmax(a.values)) for a in alt_arrays]
            z0, z1 = min(mins), max(maxs)
        edges = self._span_edges(z0, z1, res)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, edges

    def _reduce_to_surface(self, ds: xr.Dataset) -> xr.Dataset:
        dim_name = vertical_dim_name(ds)
        if dim_name is not None:
            return self._extract_surface(ds, dim_name)
        return ds

    def _source_altitude(self, ds: xr.Dataset, var: str, units: str) -> xr.DataArray:
        """Per-datum altitude (broadcast to ``ds[var]``'s dims) in ``units``.

        Order: native geometric altitude (length units) -> geopotential height
        (length units) -> pressure (US Std Atm). Errors if the dataset supplies
        none — it is the dataset's responsibility to carry a usable vertical.
        """
        from davinci_monet.pairing.strategies.track import pressure_to_altitude

        da = ds[var]
        tu = units.lower()
        if tu not in _UNIT_TO_M:
            raise PairingError(
                f"Unsupported vertical units '{units}'; use one of {sorted(_UNIT_TO_M)}"
            )
        tfac = _UNIT_TO_M[tu]

        # 1. native geometric altitude — a coord/var with length units
        for name in list(da.coords) + [v for v in ds.variables if v != var]:
            if str(name).lower() in _ALT_NAMES:
                cand = ds[name]
                su = str(cand.attrs.get("units", "")).lower()
                if su in _UNIT_TO_M:
                    return (cand * (_UNIT_TO_M[su] / tfac)).broadcast_like(da)

        # 2. geopotential height — a data variable with length units
        for name in ds.data_vars:
            if str(name).lower() in _GEOPOT_NAMES:
                cand = ds[name]
                su = str(cand.attrs.get("units", "m")).lower()
                if su in _UNIT_TO_M:
                    return (cand * (_UNIT_TO_M[su] / tfac)).broadcast_like(da)

        # 3. pressure vertical coordinate -> altitude (US Std Atm)
        for name in list(da.dims) + list(da.coords):
            if str(name).lower() in _PRESSURE_NAMES and name in ds.coords:
                cand = ds[name]
                su = str(cand.attrs.get("units", "")).lower()
                if su in _PRESSURE_HPA_UNITS or su in _PRESSURE_PA_UNITS:
                    p_hpa = cand if su in _PRESSURE_HPA_UNITS else cand / 100.0
                    alt_m = xr.DataArray(
                        pressure_to_altitude(np.asarray(p_hpa.values, dtype=float)),
                        dims=p_hpa.dims,
                        coords=p_hpa.coords,
                    )
                    return (alt_m * (1.0 / tfac)).broadcast_like(da)

        raise PairingError(
            f"Source variable '{var}' has no usable vertical coordinate for a 3-D "
            f"altitude grid; supply geometric altitude (m), geopotential height (m), "
            f"or pressure (hPa). Found dims: {list(da.dims)}"
        )

    def _flatten_to_points(
        self, ds: xr.Dataset, var: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten a source variable to (time_epoch, lon, lat, value) flat arrays.

        Uses ``broadcast_like`` so point/track/swath/grid all reduce uniformly:
        lat/lon (and time) are broadcast against the data variable's dims, then
        flattened in the variable's dim order (consistent C-order across arrays).
        """
        da = ds[var]
        lat, lon = self._get_x_coords(ds)
        order = da.dims
        data_flat = da.transpose(*order).values.astype(np.float64).flatten()
        lat_flat = lat.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
        lon_flat = lon.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
        if "time" in ds.coords or "time" in ds.dims:
            t = ds["time"]
            tvals = t.values
            if np.issubdtype(tvals.dtype, np.datetime64):
                epoch = tvals.astype("datetime64[s]").astype(np.float64)
            else:
                epoch = np.asarray(tvals, dtype=np.float64)
            epoch_da = xr.DataArray(epoch, dims=t.dims)
            time_flat = (
                epoch_da.broadcast_like(da).transpose(*order).values.astype(np.float64).flatten()
            )
        else:
            time_flat = np.zeros_like(data_flat)
        return time_flat, lon_flat, lat_flat, data_flat

    def _bin_one_source(
        self,
        ds: xr.Dataset,
        var: str,
        time_edges: np.ndarray,
        lon_edges: np.ndarray,
        lat_edges: np.ndarray,
        ntime: int,
        nlon: int,
        nlat: int,
        min_sample_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_flat, lon_flat, lat_flat, data_flat = self._flatten_to_points(ds, var)
        lon_flat = match_longitude_convention(lon_flat, lon_edges)
        count = np.zeros((ntime, nlon, nlat), dtype=np.int32)
        acc = np.zeros((ntime, nlon, nlat), dtype=np.float64)
        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges, time_flat, lon_flat, lat_flat, data_flat, count, acc
        )
        normalize_grid(count, acc)
        if min_sample_count > 1:
            acc[count < min_sample_count] = np.nan
        return acc, count

    def _uniform_horizontal_grid(
        self,
        datasets: list[xr.Dataset],
        res: float,
        extent: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if extent is not None:
            lon0, lon1, lat0, lat1 = (float(v) for v in extent)
        else:
            lons: list[float] = []
            lats: list[float] = []
            for ds in datasets:
                lat, lon = self._get_x_coords(ds)
                lon_values = wrap_longitudes(lon.values)
                lons.append(float(np.nanmin(lon_values)))
                lons.append(float(np.nanmax(lon_values)))
                lats.append(float(np.nanmin(lat.values)))
                lats.append(float(np.nanmax(lat.values)))
            lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
        # Build edges directly from the span so the grid always COVERS the data
        # extent with full ``res``-width cells — even when the span is smaller
        # than ``res`` (a single cell still spans a full ``res`` and contains the
        # data). Deriving centers then edges (the old path) could collapse a
        # small span to one center whose ``edges_from_centers`` window was too
        # narrow, silently dropping edge points.
        lon_edges = self._span_edges(lon0, lon1, res)
        lat_edges = self._span_edges(lat0, lat1, res)
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
        return lon_centers, lat_centers, lon_edges, lat_edges

    @staticmethod
    def _span_edges(lo: float, hi: float, res: float) -> np.ndarray:
        """Uniform bin edges of width ``res`` covering ``[lo, hi]`` (always ≥1 cell)."""
        n = max(1, int(np.ceil((hi - lo) / res - 1e-9)))
        return lo + res * np.arange(n + 1, dtype=np.float64)

    def _uniform_time_grid(
        self, datasets: list[xr.Dataset], time_resolution: str
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for ds in datasets:
            if "time" in ds.coords or "time" in ds.dims:
                ti = pd.DatetimeIndex(np.atleast_1d(ds["time"].values).ravel())
                starts.append(ti.min())
                ends.append(ti.max())
        if not starts:
            t0 = pd.Timestamp("1970-01-01")
            rng: pd.DatetimeIndex = pd.DatetimeIndex([t0])
        else:
            rng = pd.date_range(min(starts), max(ends), freq=time_resolution)
            if len(rng) < 1:
                rng = pd.DatetimeIndex([min(starts)])
        centers = rng.values.astype("datetime64[s]").astype(np.float64)
        return centers, edges_from_centers(centers), pd.to_datetime(centers, unit="s")

    def _build_grid(
        self,
        grid_mode: str,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        time_resolution: str,
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Build target grid edges and centers.

        Returns
        -------
        tuple
            (time_edges, lon_edges, lat_edges,
             time_centers, lon_centers, lat_centers)
            All edges/centers as numpy arrays. Time values are in
            epoch seconds (float64).
        """
        # Time grid: from analysis window + resolution
        time_range = pd.date_range(t_start, t_end, freq=time_resolution)
        if len(time_range) < 1:
            time_range = pd.DatetimeIndex([t_start])
        time_centers_epoch = time_range.values.astype("datetime64[s]").astype(np.float64)
        time_edges = edges_from_centers(time_centers_epoch)

        if grid_mode == "match_dataset":
            if y_lat.ndim != 1 or y_lon.ndim != 1:
                raise PairingError(
                    "match_dataset grid mode requires 1D y lat/lon " "(rectilinear grid)"
                )
            lat_centers = np.sort(y_lat.values.astype(np.float64))
            lon_centers = np.sort(wrap_longitudes(y_lon.values.astype(np.float64)))
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        elif grid_mode == "resolution":
            res = kwargs.get("resolution")
            if res is None:
                raise PairingError("resolution must be specified for grid_mode='resolution'")
            res = float(res)
            lat_centers = np.arange(-90 + res / 2, 90, res)
            lon_centers = np.arange(0 + res / 2, 360, res)
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        elif grid_mode == "explicit":
            nlat = kwargs.get("nlat")
            nlon = kwargs.get("nlon")
            if nlat is None or nlon is None:
                raise PairingError("nlat and nlon must be specified for grid_mode='explicit'")
            lat_centers = np.linspace(-90, 90, int(nlat))
            lon_centers = np.linspace(0, 360, int(nlon), endpoint=False)
            lat_edges = edges_from_centers(lat_centers)
            lon_edges = edges_from_centers(lon_centers)

        else:
            raise PairingError(f"Unknown grid_mode: {grid_mode}")

        return time_edges, lon_edges, lat_edges, time_centers_epoch, lon_centers, lat_centers

    def _get_x_timestamps(
        self, x_data: xr.Dataset, n_pixels: int, align_var: str | None = None
    ) -> np.ndarray:
        """Extract x timestamps as flat epoch-second array.

        Parameters
        ----------
        x_data
            The x source.
        n_pixels
            Total number of pixels (for broadcasting).
        align_var
            Name of the data variable whose pixel order ``data_flat`` follows.
            The time coordinate is broadcast against this variable so per-pixel
            timestamps line up with the binned values. Defaults to the first
            data variable.

        Returns
        -------
        np.ndarray
            Flat float64 array of epoch seconds, one per pixel.
        """
        if "time" in x_data.coords:
            time_da = x_data["time"]
            time_vals = time_da.values
            if np.issubdtype(time_vals.dtype, np.datetime64):
                epoch = time_vals.astype("datetime64[s]").astype(np.float64)
                if epoch.ndim == 0:
                    # Scalar time → broadcast to all pixels
                    return np.full(n_pixels, float(epoch), dtype=np.float64)
                # Dim-aware broadcast: a swath ``time`` is commonly per-scanline
                # (dims ``(scanline,)``) while data is per-pixel (dims
                # ``(scanline, pixel)``). broadcast_like expands ``time`` along
                # the data var's dims regardless of axis order, where positional
                # ``np.broadcast_to`` would fail (trailing-axis rule).
                var_name = align_var if align_var in x_data.data_vars else None
                if var_name is None and x_data.data_vars:
                    var_name = str(list(x_data.data_vars)[0])
                if var_name is not None:
                    # ``epoch`` was already converted via .values above (numpy
                    # path), so reuse it directly without recomputing.
                    epoch_da = xr.DataArray(epoch, dims=time_da.dims)
                    broadcast = epoch_da.broadcast_like(x_data[var_name])
                    # Match the C-order .flatten() used for the data values.
                    epoch_flat = broadcast.transpose(*x_data[var_name].dims).values.flatten()
                    if len(epoch_flat) == n_pixels:
                        return epoch_flat.astype(np.float64)
                return np.full(n_pixels, float(epoch.mean()), dtype=np.float64)
            else:
                return np.full(n_pixels, float(time_vals.flat[0]), dtype=np.float64)

        # No time coordinate — use a single default timestamp
        return np.zeros(n_pixels, dtype=np.float64)

    def _align_y_to_grid(
        self,
        y_data: xr.Dataset,
        y_var: str,
        time_coords: pd.DatetimeIndex,
        lon_centers: np.ndarray,
        lat_centers: np.ndarray,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        grid_mode: str,
    ) -> xr.DataArray:
        """Extract y variable aligned to the target grid.

        For match_dataset mode, the grid is already the y grid so
        we just select the nearest times. For other modes, we interpolate
        spatially.

        Returns
        -------
        xr.DataArray
            The y data on (time, lon, lat) grid.
        """
        var_data = y_data[y_var]

        # Select nearest y times
        if "time" in var_data.dims:
            var_data = var_data.sel(time=time_coords, method="nearest")
            var_data = var_data.assign_coords(time=time_coords)

        if grid_mode == "match_dataset":
            # Already on the y grid — just ensure dim order is (time, lon, lat)
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
            # Transpose to (time, lon, lat) to match our output convention
            dim_order: list[str] = []
            if "time" in var_data.dims:
                dim_order.append("time")
            dim_order.extend([str(lon_dim), str(lat_dim)])
            var_data = var_data.transpose(*dim_order)
            # Rename dims to match paired output
            rename_map = {}
            if lat_dim != "lat":
                rename_map[lat_dim] = "lat"
            if lon_dim != "lon":
                rename_map[lon_dim] = "lon"
            if rename_map:
                var_data = var_data.rename(rename_map)
            return var_data.astype(np.float32)
        else:
            # Interpolate the y source to target grid
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
            var_interp = var_data.interp(
                {lat_dim: lat_centers, lon_dim: lon_centers},
                method="nearest",
            )
            # Rename dims
            rename_map = {}
            if lat_dim != "lat":
                rename_map[lat_dim] = "lat"
            if lon_dim != "lon":
                rename_map[lon_dim] = "lon"
            if rename_map:
                var_interp = var_interp.rename(rename_map)
            # Ensure (time, lon, lat) order
            dim_order = []
            if "time" in var_interp.dims:
                dim_order.append("time")
            dim_order.extend(["lon", "lat"])
            var_interp = var_interp.transpose(*dim_order)
            return var_interp.astype(np.float32)


# Back-compat alias (the strategy generalized beyond swath in 2026-06).
SwathGridStrategy = IntermediateGridStrategy
