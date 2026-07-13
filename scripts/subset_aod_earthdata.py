#!/usr/bin/env python3
"""Create compact AOD subsets from the Earthaccess scratch staging tree.

MERRA-2 hourly ``TOTEXTTAU`` is averaged to one daily field. MODIS D3 keeps
the combined 550-nm daily mean, while MODIS L2 keeps that field, its QA and
algorithm flags, and the swath geolocation/time arrays needed by DAVINCI.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from netCDF4 import Dataset, date2num
from pyhdf.SD import SD, SDC

DEFAULT_SOURCE_ROOT = Path("/glade/derecho/scratch/fillmore/DAVINCI-AOD/raw")
DEFAULT_OUTPUT_ROOT = Path("/glade/work/fillmore/Data/CERES-SARB-CAM7/AOD_SUBSETS")
D3_AOD = "AOD_550_Dark_Target_Deep_Blue_Combined_Mean"
L2_AOD = "AOD_550_Dark_Target_Deep_Blue_Combined"
L2_QA = f"{L2_AOD}_QA_Flag"
L2_ALGORITHM = f"{L2_AOD}_Algorithm_Flag"
L2_FIELDS = ("Latitude", "Longitude", "Scan_Start_Time", L2_AOD, L2_QA, L2_ALGORITHM)
FILL_FLOAT = np.float32(9.96921e36)
FILL_FLAG = np.int8(-127)


@dataclass(frozen=True)
class Product:
    key: str
    platform: str
    prefix: str
    level: str

    @property
    def is_merra2(self) -> bool:
        return self.key == "merra2"

    @property
    def is_l2(self) -> bool:
        return self.level == "L2"


@dataclass(frozen=True)
class Bounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= self.lat_min)
            & (lat <= self.lat_max)
            & (lon >= self.lon_min)
            & (lon <= self.lon_max)
        )


@dataclass(frozen=True)
class Result:
    product: str
    day: date
    status: str
    source: Path | None
    output: Path | None
    message: str = ""


PRODUCTS = {
    "merra2": Product("merra2", "MERRA2", "MERRA2_", "daily"),
    "terra": Product("terra", "Terra", "MOD08_D3", "D3"),
    "aqua": Product("aqua", "Aqua", "MYD08_D3", "D3"),
    "terra-l2": Product("terra-l2", "Terra", "MOD04_L2", "L2"),
    "aqua-l2": Product("aqua-l2", "Aqua", "MYD04_L2", "L2"),
}
DEFAULT_PRODUCTS = tuple(PRODUCTS)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def parse_products(value: str) -> list[Product]:
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not keys or keys == ["all"]:
        keys = list(DEFAULT_PRODUCTS)
    unknown = sorted(set(keys) - PRODUCTS.keys())
    if unknown:
        choices = ", ".join(PRODUCTS)
        raise argparse.ArgumentTypeError(
            f"unknown product(s): {', '.join(unknown)}; choose from {choices}"
        )
    return [PRODUCTS[key] for key in keys]


def iter_dates(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def product_inputs(root: Path, product: Product, day: date) -> list[Path]:
    if product.is_merra2:
        directory = root / "GMAO" / "MERRA2" / f"{day:%Y}" / f"{day:%m}"
        pattern = f"MERRA2_*.tavg1_2d_aer_Nx.{day:%Y%m%d}.nc4"
    else:
        directory = root / "MODIS" / product.platform / "C61" / f"{day:%Y}" / f"{day:%j}"
        pattern = f"{product.prefix}.A{day:%Y%j}.*.hdf"
    return sorted(directory.glob(pattern))


def output_path(root: Path, product: Product, day: date, source: Path) -> Path:
    if product.is_merra2:
        name = f"MERRA2_TOTEXTTAU_daily.{day:%Y%m%d}.nc4"
        return root / "GMAO" / "MERRA2" / f"{day:%Y}" / f"{day:%m}" / name
    directory = root / "MODIS" / product.platform / "C61" / f"{day:%Y}" / f"{day:%j}"
    return directory / f"{source.stem}.AOD550.nc4"


def atomic_netcdf(target: Path, writer: Callable[[Path], None], overwrite: bool) -> str:
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("writer produced an empty output")
        with Dataset(tmp) as check:
            if not check.variables:
                raise RuntimeError("output has no variables")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return "written"


def coordinate_indices(values: np.ndarray, lower: float, upper: float, name: str) -> np.ndarray:
    indices = np.flatnonzero((values >= lower) & (values <= upper))
    if indices.size == 0:
        raise ValueError(f"{name} bounds do not intersect the source grid")
    return indices


def set_global_attrs(
    output: Dataset, *, title: str, source: Path, product: Product, day: date
) -> None:
    output.setncatts(
        {
            "title": title,
            "source_file": str(source),
            "source_product": product.prefix,
            "platform": product.platform,
            "collection": "6.1" if not product.is_merra2 else "M2T1NXAER 5.12.4",
            "date": day.isoformat(),
            "history": f"created {datetime.now(timezone.utc).isoformat()} by {Path(__file__).name}",
            "Conventions": "CF-1.8",
        }
    )


def create_grid_coordinates(output: Dataset, day: date, lat: np.ndarray, lon: np.ndarray) -> None:
    output.createDimension("time", 1)
    output.createDimension("lat", lat.size)
    output.createDimension("lon", lon.size)
    time_units = "days since 1970-01-01 00:00:00 UTC"
    time_var = output.createVariable("time", "f8", ("time",))
    time_var[:] = date2num(datetime.combine(day, time(12), tzinfo=timezone.utc), time_units)
    time_var.setncatts({"standard_name": "time", "units": time_units, "calendar": "standard"})
    lat_var = output.createVariable("lat", "f4", ("lat",))
    lon_var = output.createVariable("lon", "f4", ("lon",))
    lat_var[:] = lat.astype("f4")
    lon_var[:] = lon.astype("f4")
    lat_var.setncatts({"standard_name": "latitude", "units": "degrees_north"})
    lon_var.setncatts({"standard_name": "longitude", "units": "degrees_east"})


def write_merra2(
    source: Path, target: Path, product: Product, day: date, bounds: Bounds | None, overwrite: bool
) -> str:
    with Dataset(source) as raw:
        lat_all = np.asarray(raw.variables["lat"][:])
        lon_all = np.asarray(raw.variables["lon"][:])
        if bounds is None:
            lat_indices = np.arange(lat_all.size)
            lon_indices = np.arange(lon_all.size)
        else:
            lat_indices = coordinate_indices(lat_all, bounds.lat_min, bounds.lat_max, "latitude")
            lon_indices = coordinate_indices(lon_all, bounds.lon_min, bounds.lon_max, "longitude")
        lat_slice = slice(int(lat_indices[0]), int(lat_indices[-1]) + 1)
        lon_slice = slice(int(lon_indices[0]), int(lon_indices[-1]) + 1)
        hourly = np.ma.asarray(raw.variables["TOTEXTTAU"][:, lat_slice, lon_slice])
        daily = np.ma.mean(hourly, axis=0, dtype=np.float64).astype("f4")
        lat = lat_all[lat_slice]
        lon = lon_all[lon_slice]
        sample_count = hourly.shape[0]

    def writer(tmp: Path) -> None:
        with Dataset(tmp, "w", format="NETCDF4") as output:
            create_grid_coordinates(output, day, lat, lon)
            aod = output.createVariable(
                "TOTEXTTAU",
                "f4",
                ("time", "lat", "lon"),
                fill_value=FILL_FLOAT,
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, min(180, lat.size), min(288, lon.size)),
            )
            aod[0] = daily
            aod.setncatts(
                {
                    "long_name": "Daily mean total aerosol extinction optical thickness at 550 nm",
                    "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
                    "units": "1",
                    "source_sample_count": sample_count,
                }
            )
            set_global_attrs(
                output,
                title="MERRA-2 daily mean total 550-nm aerosol optical thickness",
                source=source,
                product=product,
                day=day,
            )

    return atomic_netcdf(target, writer, overwrite)


def hdf_field(hdf: SD, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if name not in hdf.datasets():
        raise KeyError(f"required SDS is missing: {name}")
    sds = hdf.select(name)
    try:
        return np.asarray(sds[:]), dict(sds.attributes())
    finally:
        sds.endaccess()


def physical_values(raw: np.ndarray, attrs: dict[str, Any]) -> np.ma.MaskedArray:
    mask = np.zeros(raw.shape, dtype=bool)
    if "_FillValue" in attrs:
        mask |= raw == attrs["_FillValue"]
    valid_range = attrs.get("valid_range")
    if valid_range is not None and len(valid_range) == 2:
        mask |= (raw < valid_range[0]) | (raw > valid_range[1])
    values = raw.astype("f8")
    values = values * float(attrs.get("scale_factor", 1.0)) + float(attrs.get("add_offset", 0.0))
    return np.ma.array(values, mask=mask)


def write_modis_d3(
    source: Path, target: Path, product: Product, day: date, bounds: Bounds | None, overwrite: bool
) -> str:
    hdf = SD(str(source), SDC.READ)
    try:
        raw_aod, attrs = hdf_field(hdf, D3_AOD)
        lon, _ = hdf_field(hdf, "XDim")
        lat, _ = hdf_field(hdf, "YDim")
    finally:
        hdf.end()
    aod = physical_values(raw_aod, attrs).astype("f4")
    if bounds is not None:
        lat_indices = coordinate_indices(lat, bounds.lat_min, bounds.lat_max, "latitude")
        lon_indices = coordinate_indices(lon, bounds.lon_min, bounds.lon_max, "longitude")
        aod = aod[np.ix_(lat_indices, lon_indices)]
        lat = lat[lat_indices]
        lon = lon[lon_indices]

    def writer(tmp: Path) -> None:
        with Dataset(tmp, "w", format="NETCDF4") as output:
            create_grid_coordinates(output, day, lat, lon)
            variable = output.createVariable(
                D3_AOD,
                "f4",
                ("time", "lat", "lon"),
                fill_value=FILL_FLOAT,
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, min(180, lat.size), min(360, lon.size)),
            )
            variable[0] = aod
            variable.setncatts(
                {
                    "long_name": attrs.get("long_name", "Combined MODIS 550-nm AOD daily mean"),
                    "units": "1",
                    "valid_min": np.float32(0.0),
                    "valid_max": np.float32(5.0),
                }
            )
            set_global_attrs(
                output,
                title=f"MODIS {product.platform} daily combined 550-nm aerosol optical depth",
                source=source,
                product=product,
                day=day,
            )

    return atomic_netcdf(target, writer, overwrite)


def write_modis_l2(
    source: Path, target: Path, product: Product, day: date, bounds: Bounds | None, overwrite: bool
) -> str:
    hdf = SD(str(source), SDC.READ)
    try:
        fields = {name: hdf_field(hdf, name) for name in L2_FIELDS}
    finally:
        hdf.end()
    values = {name: physical_values(raw, attrs) for name, (raw, attrs) in fields.items()}
    lat = values["Latitude"]
    lon = values["Longitude"]
    in_bounds = np.isfinite(lat.filled(np.nan)) & np.isfinite(lon.filled(np.nan))
    if bounds is not None:
        in_bounds &= bounds.mask(lat.filled(np.nan), lon.filled(np.nan))
        rows, columns = np.nonzero(in_bounds)
        if rows.size == 0:
            return "outside-bounds"
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        column_slice = slice(int(columns.min()), int(columns.max()) + 1)
        values = {name: value[row_slice, column_slice] for name, value in values.items()}
        in_bounds = in_bounds[row_slice, column_slice]
        for name in (L2_AOD, L2_QA, L2_ALGORITHM):
            values[name] = np.ma.masked_where(~in_bounds, values[name])
    along, across = values[L2_AOD].shape

    def writer(tmp: Path) -> None:
        with Dataset(tmp, "w", format="NETCDF4") as output:
            output.createDimension("Cell_Along_Swath", along)
            output.createDimension("Cell_Across_Swath", across)
            dims = ("Cell_Along_Swath", "Cell_Across_Swath")
            chunks = (min(128, along), min(135, across))
            for name, standard_name, units in (
                ("Latitude", "latitude", "degrees_north"),
                ("Longitude", "longitude", "degrees_east"),
            ):
                variable = output.createVariable(
                    name,
                    "f4",
                    dims,
                    fill_value=FILL_FLOAT,
                    zlib=True,
                    complevel=4,
                    chunksizes=chunks,
                )
                variable[:] = values[name].astype("f4")
                variable.setncatts({"standard_name": standard_name, "units": units})
            scan = output.createVariable(
                "Scan_Start_Time",
                "f8",
                dims,
                fill_value=np.float64(FILL_FLOAT),
                zlib=True,
                complevel=4,
                chunksizes=chunks,
            )
            scan[:] = values["Scan_Start_Time"]
            scan.setncatts(
                {
                    "long_name": "TAI time at start of scan",
                    "units": "seconds since 1993-01-01 00:00:00",
                }
            )
            aod = output.createVariable(
                L2_AOD, "f4", dims, fill_value=FILL_FLOAT, zlib=True, complevel=4, chunksizes=chunks
            )
            aod[:] = values[L2_AOD].astype("f4")
            aod.setncatts(
                {
                    "long_name": fields[L2_AOD][1].get("long_name", "Combined MODIS 550-nm AOD"),
                    "units": "1",
                    "coordinates": "Latitude Longitude Scan_Start_Time",
                }
            )
            for name, long_name in (
                (L2_QA, "Combined aerosol confidence flag"),
                (L2_ALGORITHM, "Combined AOD algorithm flag"),
            ):
                flag = output.createVariable(
                    name,
                    "i1",
                    dims,
                    fill_value=FILL_FLAG,
                    zlib=True,
                    complevel=4,
                    chunksizes=chunks,
                )
                flag[:] = values[name].filled(FILL_FLAG).astype("i1")
                flag.setncatts(
                    {
                        "long_name": long_name,
                        "units": "1",
                        "coordinates": "Latitude Longitude Scan_Start_Time",
                    }
                )
            set_global_attrs(
                output,
                title=f"MODIS {product.platform} L2 combined 550-nm aerosol optical depth subset",
                source=source,
                product=product,
                day=day,
            )

    return atomic_netcdf(target, writer, overwrite)


def process_product(
    root: Path,
    output_root: Path,
    product: Product,
    day: date,
    bounds: Bounds | None,
    overwrite: bool,
) -> list[Result]:
    inputs = product_inputs(root, product, day)
    if not inputs:
        return [Result(product.key, day, "missing", None, None, "no source files")]
    if not product.is_l2 and len(inputs) > 1:
        inputs = [inputs[-1]]
    processor = (
        write_merra2 if product.is_merra2 else write_modis_l2 if product.is_l2 else write_modis_d3
    )
    results = []
    for source in inputs:
        target = output_path(output_root, product, day, source)
        try:
            status = processor(source, target, product, day, bounds, overwrite)
            output = target if status != "outside-bounds" else None
            results.append(Result(product.key, day, status, source, output))
            print(f"{product.key} {day}: {status} {target if output else source.name}")
        except Exception as exc:
            results.append(Result(product.key, day, "error", source, target, str(exc)))
            print(f"ERROR: {product.key} {day} {source}: {exc}", file=sys.stderr)
    return results


def write_manifest(path: Path, results: Sequence[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "product",
                "date",
                "status",
                "source_size_bytes",
                "output_size_bytes",
                "source",
                "output",
                "message",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "product": result.product,
                    "date": result.day,
                    "status": result.status,
                    "source_size_bytes": result.source.stat().st_size if result.source else "",
                    "output_size_bytes": (
                        result.output.stat().st_size
                        if result.output and result.output.exists()
                        else ""
                    ),
                    "source": result.source or "",
                    "output": result.output or "",
                    "message": result.message,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, required=True, help="First UTC date")
    parser.add_argument("--end", type=parse_date, required=True, help="Last UTC date")
    parser.add_argument(
        "--products",
        default="all",
        help="Comma-separated merra2,terra,aqua,terra-l2,aqua-l2 list (default: all)",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lat-min", type=float)
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    products = parse_products(args.products)
    bound_values = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if any(value is not None for value in bound_values):
        if not all(value is not None for value in bound_values):
            raise SystemExit("all four latitude/longitude bounds are required")
        bounds = Bounds(*bound_values)
        if not (-90 <= bounds.lat_min <= bounds.lat_max <= 90):
            raise SystemExit("latitude bounds must be ordered within [-90, 90]")
        if not (-180 <= bounds.lon_min <= bounds.lon_max <= 180):
            raise SystemExit("longitude bounds must be ordered within [-180, 180]")
    else:
        bounds = None
    if not args.source_root.is_dir():
        raise SystemExit(f"source root does not exist: {args.source_root}")

    os.umask(0o027)
    results = [
        result
        for day in iter_dates(args.start, args.end)
        for product in products
        for result in process_product(
            args.source_root, args.output_root, product, day, bounds, args.overwrite
        )
    ]
    label = "-".join(product.key for product in products)
    manifest = (
        args.output_root / "manifests" / f"subset_{label}_{args.start:%Y%m%d}_{args.end:%Y%m%d}.tsv"
    )
    write_manifest(manifest, results)
    print(f"Manifest: {manifest}")
    errors = sum(result.status == "error" for result in results)
    missing = sum(result.status == "missing" for result in results)
    if errors:
        print(f"ERROR: {errors} source file(s) failed", file=sys.stderr)
        return 1
    if missing and not args.allow_missing:
        print(f"ERROR: {missing} product-day(s) missing", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
