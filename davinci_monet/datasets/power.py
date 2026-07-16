"""NASA POWER reader.

Reads NASA POWER (https://power.larc.nasa.gov/) solar and meteorological
parameters. Unlike the granule-staged readers, POWER is a REST API: this
reader fetches at pipeline runtime through
:mod:`davinci_monet.io.download.power`, which caches each response as NetCDF,
so reruns are offline and free.

Three config modes, exactly one of:

* ``sites:``  -> POINT ``(time, site)``  -- "virtual stations"
* ``bbox:``   -> GRID  ``(time, lat, lon)``, on the parameter's **parent grid**
* ``files:``  -> GRID, opened from previously staged NetCDF, no network

Regional responses are **not** on a common 0.5 deg grid: each parameter comes
back on its parent's native grid -- solar on CERES SYN1deg's 1.0 deg, met on
MERRA-2's 0.5 x 0.625 (measured 2026-07-15). Pairing two POWER parameters
against each other spatially therefore needs regridding.

Provenance
----------
POWER is **not ground truth**. Its solar parameters derive from CERES
SYN1deg/FLASHFlux (and an earlier parent before the CERES era -- the record
starts 1984 while SYN1deg starts 2000); its meteorology *is* MERRA-2/GEOS,
served on MERRA-2's own grid rather than regridded. So evaluating MERRA-2
against POWER solar is a genuine comparison, while evaluating MERRA-2 against
POWER ``T2M`` is circular -- a traceability check only. See POWER.md.

Units
-----
POWER's native units vary by **(parameter, temporal level)**, not parameter
alone -- daily solar is ``kW-hr/m^2/day`` while hourly solar is ``Wh/m^2``.
:data:`POWER_CATALOG` is keyed accordingly and every entry's ``native_units``
was measured against the live API on 2026-07-15. The reader asserts the
response's units against the catalog before scaling, so an upstream unit
change fails loudly instead of silently corrupting statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.download.power import fetch_to_cache, plan_requests
from davinci_monet.io.reader_utils import select_variables, set_geometry_attr

logger = logging.getLogger(__name__)

#: POWER's documented missing-data sentinel (JSON ``header.fill_value``).
FILL_VALUE = -999.0

#: 1 kWh/m^2/day = 1000 Wh spread over 24 h = 41.667 W/m^2.
KWH_M2_DAY_TO_W_M2 = 1000.0 / 24.0
#: 1 MJ/m^2/day = 1e6 J over 86400 s = 11.574 W/m^2 (community AG; unverified).
MJ_M2_DAY_TO_W_M2 = 1.0e6 / 86400.0


@dataclass(frozen=True)
class PowerVariable:
    """Catalog entry: what POWER sends, and how to get it to SI.

    ``value_si = value_native * scale + offset``
    """

    native_units: str
    units: str
    scale: float = 1.0
    offset: float = 0.0


def _radiation(temporal: str) -> PowerVariable:
    """Radiative flux entry for ``temporal`` (units differ by level)."""
    if temporal == "hourly":
        # Wh/m^2 accumulated over one hour *is* W/m^2 -- scale 1, not 3600.
        return PowerVariable(native_units="Wh/m^2", units="W m-2", scale=1.0)
    return PowerVariable(native_units="kW-hr/m^2/day", units="W m-2", scale=KWH_M2_DAY_TO_W_M2)


_TEMPERATURE = PowerVariable(native_units="C", units="K", offset=273.15)
_RADIATION_PARAMS = ("ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN")

#: (parameter, temporal) -> conversion. ``None`` temporal means "any level".
#: Every native_units string here was measured against the live API.
POWER_CATALOG: dict[tuple[str, str | None], PowerVariable] = {
    **{
        (param, temporal): _radiation(temporal)
        for param in _RADIATION_PARAMS
        for temporal in ("hourly", "daily", "monthly")
    },
    ("T2M", None): _TEMPERATURE,
    # T2M_MAX/T2M_MIN are daily aggregates -- the hourly endpoint rejects them.
    ("T2M_MAX", "daily"): _TEMPERATURE,
    ("T2M_MAX", "monthly"): _TEMPERATURE,
    ("T2M_MIN", "daily"): _TEMPERATURE,
    ("T2M_MIN", "monthly"): _TEMPERATURE,
    ("RH2M", None): PowerVariable(native_units="%", units="%"),
    ("WS10M", None): PowerVariable(native_units="m/s", units="m s-1"),
    ("WS50M", None): PowerVariable(native_units="m/s", units="m s-1"),
    ("PS", None): PowerVariable(native_units="kPa", units="Pa", scale=1000.0),
    # Left as a depth rate: mm/day is the conventional met unit and no model
    # comparison in v1 needs kg m-2 s-1.
    ("PRECTOTCORR", None): PowerVariable(native_units="mm/day", units="mm day-1"),
}


def catalog_entry(param: str, temporal: str) -> PowerVariable | None:
    """Return the catalog entry for ``param`` at ``temporal``, if catalogued."""
    entry = POWER_CATALOG.get((param, temporal))
    if entry is None:
        entry = POWER_CATALOG.get((param, None))
    return entry


@source_registry.register("power")
class POWERReader:
    """Reader for NASA POWER, via the POWER REST API or staged NetCDF."""

    def __init__(self) -> None:
        # Resolved during open(); the load stage reads .geometry afterwards.
        self._geometry = DataGeometry.GRID

    @property
    def name(self) -> str:
        """Return reader name."""
        return "power"

    @property
    def geometry(self) -> DataGeometry:
        """POINT for ``sites:``, GRID for ``bbox:``/``files:``.

        Unlike the fixed-geometry readers this is config-dependent, so it is
        only meaningful after :meth:`open`. The load stage reads it there.
        """
        return self._geometry

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        temporal: str = "daily",
        community: str = "RE",
        cache_dir: str | Path | None = None,
        sites: Sequence[Mapping[str, Any]] | None = None,
        bbox: Mapping[str, float] | None = None,
        time_range: tuple[Any, Any] | None = None,
        force: bool = False,
        offline: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open POWER data and standardize it to DAVINCI conventions.

        Parameters
        ----------
        file_paths
            Staged POWER NetCDF. Mutually exclusive with ``sites``/``bbox``.
        variables
            POWER parameter names, e.g. ``["T2M"]``.
        temporal
            ``hourly``, ``daily`` or ``monthly``.
        time_range
            ``(start, end)``, injected by the load stage from
            ``analysis.start_time``/``end_time``. Required for fetch modes.

        Returns
        -------
        xr.Dataset
            POINT ``(time, site)`` for ``sites:``, else GRID ``(time, lat, lon)``,
            with geometry tagged and units normalized to SI.
        """
        selected = [
            name
            for name, value in (("files", list(file_paths)), ("sites", sites), ("bbox", bbox))
            if value
        ]
        if len(selected) != 1:
            raise ValueError(
                "POWER source needs exactly one of files:, sites: or bbox:; "
                f"got {selected or 'none'}."
            )
        mode = selected[0]
        params = list(variables) if variables else []

        if mode == "files":
            ds = self._open_files(file_paths, params)
            self._geometry = DataGeometry.GRID
        else:
            if time_range is None:
                raise ValueError(
                    "POWER fetch modes need a time range; set analysis.start_time "
                    "and analysis.end_time."
                )
            if not params:
                raise ValueError("POWER source needs at least one variable.")
            start, end = time_range
            requests = plan_requests(
                temporal,
                "point" if mode == "sites" else "regional",
                params,
                start=start,
                end=end,
                sites=sites,
                bbox=bbox,
                community=community,
            )
            paths = [
                fetch_to_cache(req, cache_dir or _default_cache(), force=force, offline=offline)
                for req in requests
            ]
            if mode == "sites":
                ds = self._assemble_points(requests, paths)
                self._geometry = DataGeometry.POINT
            else:
                ds = self._assemble_grid(paths)
                self._geometry = DataGeometry.GRID

        ds = self._decode_monthly_time(ds)
        ds = self._center_hourly_time(ds, temporal)
        ds = self._normalize(ds, temporal)
        return set_geometry_attr(ds, self._geometry)

    def _open_files(self, file_paths: Sequence[str | Path], params: Sequence[str]) -> xr.Dataset:
        """Open staged POWER NetCDF without touching the network."""
        paths = [str(p) for p in file_paths]
        ds = (
            xr.open_mfdataset(paths, combine="by_coords")
            if len(paths) > 1
            else xr.open_dataset(paths[0])
        )
        return select_variables(ds, list(params) or None)

    def _assemble_points(self, requests: Sequence[Any], paths: Sequence[Path]) -> xr.Dataset:
        """Build ``(time, site)`` from per-site ``(time, lat=1, lon=1)`` responses.

        The API has no site dimension -- a point response has the same shape as
        a regional one with degenerate lat/lon -- so the site axis is ours to
        construct. Parameter chunks for one site are merged before concat.
        """
        per_site: dict[str, list[xr.Dataset]] = {}
        coords: dict[str, tuple[float, float]] = {}
        for req, path in zip(requests, paths):
            site = req.site or f"{req.latitude}_{req.longitude}"
            ds = xr.open_dataset(path).squeeze(["lat", "lon"], drop=True)
            per_site.setdefault(site, []).append(ds)
            coords[site] = (float(req.latitude), float(req.longitude))

        names = list(per_site)
        merged = [xr.merge(per_site[name]) for name in names]
        out = xr.concat(merged, dim="site")
        out = out.assign_coords(
            site=("site", names),
            latitude=("site", [coords[n][0] for n in names]),
            longitude=("site", [coords[n][1] for n in names]),
        )
        return out

    def _decode_monthly_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Turn the monthly endpoint's ``YYYYMM`` integers into real datetimes.

        Two quirks, both measured against the live API:

        * ``time`` arrives as **int64 YYYYMM**, not datetime64, so without this
          every downstream time operation -- selection, resampling, the x axis
          of a plot -- is operating on the integer 198101.
        * Each year carries a **13th "month"**: ``YYYY13`` is that year's annual
          mean. Left in, it is an extra data point every 13th step that is not a
          month at all, quietly contaminating any series or statistic.

        The annual means are dropped rather than surfaced; a caller who wants
        them can resample the monthly series, which is unambiguous.
        """
        if "time" not in ds.coords or np.issubdtype(ds["time"].dtype, np.datetime64):
            return ds

        stamps = np.asarray(ds["time"].values).astype("int64")
        months = stamps % 100
        annual = months == 13
        if annual.any():
            logger.debug("Dropping %d POWER annual-mean (YYYY13) entries", int(annual.sum()))
            ds = ds.isel(time=~annual)
            stamps = stamps[~annual]

        years, months = stamps // 100, stamps % 100
        decoded = np.array([np.datetime64(f"{y:04d}-{m:02d}", "ns") for y, m in zip(years, months)])
        return ds.assign_coords(time=("time", decoded))

    def _center_hourly_time(self, ds: xr.Dataset, temporal: str) -> xr.Dataset:
        """Move hourly stamps from the interval start to its midpoint.

        POWER labels an hourly mean at the interval **start** (``2024020100``
        is the 00:00-01:00 mean). Model 1-hour averages -- MERRA-2 ``tavg1``,
        and the same convention elsewhere -- label the **midpoint** (00:30).
        Both describe the same hour; only the label differs.

        This is not cosmetic. Pairing is nearest-neighbour in time, so an
        unshifted POWER 00:00 sits *equidistant* between MERRA-2's 23:30 and
        00:30 and the tie can break either way. Measured at Boulder over 72 h:
        aligned, POWER and MERRA-2 ``T2M`` agree to **RMSE 0.003 K**; off by one
        hour, **1.16 K**. That is the whole margin between a traceability check
        that proves the reader and one that looks like a phase bug.

        Hourly only: daily and monthly means are conventionally labelled at the
        period start, and models follow suit, so they are left alone.
        """
        if temporal != "hourly" or "time" not in ds.coords:
            return ds
        shifted = ds.assign_coords(time=ds["time"] + np.timedelta64(30, "m"))
        shifted["time"].attrs.update(
            {
                "cell_methods": "time: mean (interval: 1 hour)",
                "comment": (
                    "POWER labels hourly means at the interval start; shifted +30 min "
                    "to the interval midpoint to match tavg1-style model conventions."
                ),
            }
        )
        return shifted

    def _assemble_grid(self, paths: Sequence[Path]) -> xr.Dataset:
        """Merge regional responses, stitching bbox tiles back into one field.

        A domain wider than 10 deg on either axis has to be requested as tiles
        (the API rejects anything larger), so the pieces must be reassembled
        here. Adjacent tiles can return a shared edge row -- the met grid
        includes its bbox boundary while the solar grid does not -- so
        duplicate coordinates are dropped rather than left to become a
        double-counted row in a bias map.
        """
        datasets = [xr.open_dataset(p) for p in paths]
        if len(datasets) == 1:
            return datasets[0]
        merged = xr.combine_by_coords(datasets, combine_attrs="override")
        if not isinstance(merged, xr.Dataset):  # pragma: no cover - defensive
            merged = merged.to_dataset()
        for dim in ("lat", "lon"):
            if dim in merged.dims:
                _, keep = np.unique(merged[dim].values, return_index=True)
                if len(keep) != merged.sizes[dim]:
                    logger.debug(
                        "Dropping %d duplicate %s values from tiled POWER grid",
                        merged.sizes[dim] - len(keep),
                        dim,
                    )
                    merged = merged.isel({dim: np.sort(keep)})
        return merged

    def _normalize(self, ds: xr.Dataset, temporal: str) -> xr.Dataset:
        """Mask fill values, then convert each variable to canonical SI units.

        Masking precedes scaling on purpose: -999 * 41.667 is not a missing
        value, it is a plausible-looking wrong number.
        """
        for name in list(ds.data_vars):
            var = ds[name]
            native = str(var.attrs.get("units", ""))
            entry = catalog_entry(str(name), temporal)

            masked = var.where(var != FILL_VALUE)

            if entry is None:
                logger.warning(
                    "POWER parameter %r is not in POWER_CATALOG; passing values through "
                    "with the response's own units (%r). Add a catalog entry to get SI "
                    "normalization and a units check.",
                    name,
                    native,
                )
                ds[name] = masked.assign_attrs(var.attrs)
                continue

            if native != entry.native_units:
                raise ValueError(
                    f"POWER {name} at temporal={temporal!r} arrived with units {native!r} "
                    f"but the catalog expected {entry.native_units!r}. Refusing to scale: "
                    f"an upstream unit change must not silently corrupt statistics. "
                    f"Update POWER_CATALOG after verifying against the live API."
                )

            converted = masked * entry.scale + entry.offset
            ds[name] = converted.assign_attrs({**var.attrs, "units": entry.units})
            # valid_min/max arrive in native units; they no longer describe the
            # converted values, so drop rather than mislead downstream masking.
            for attr in ("valid_min", "valid_max"):
                ds[name].attrs.pop(attr, None)
        return ds


def _default_cache() -> Path:
    from davinci_monet.io.download.power import DEFAULT_CACHE_DIR

    return DEFAULT_CACHE_DIR


__all__ = ["POWERReader", "POWER_CATALOG", "PowerVariable", "catalog_entry"]
