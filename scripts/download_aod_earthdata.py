#!/usr/bin/env python3
"""Download MERRA-2 and MODIS daily or Level-2 AOD inputs through Earthaccess.

The script is intended for short, queue-managed date ranges. It filters CMR
results by the date encoded in each filename because daily MODIS searches can
include the first granule after the requested interval.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

DEFAULT_ROOT = Path("/glade/derecho/scratch/fillmore/DAVINCI-AOD/raw")


@dataclass(frozen=True)
class Product:
    key: str
    short_name: str
    version: str
    platform: str
    available_from: date
    filename_pattern: re.Pattern[str]

    def filename_match(self, filename: str) -> re.Match[str] | None:
        return self.filename_pattern.search(filename)

    def granule_date(self, filename: str) -> date | None:
        match = self.filename_match(filename)
        if match is None:
            return None
        value = match.group("date")
        date_format = "%Y%j" if len(value) == 7 else "%Y%m%d"
        return datetime.strptime(value, date_format).date()

    def granule_identity(self, filename: str) -> str | None:
        match = self.filename_match(filename)
        if match is None:
            return None
        acquisition_time = match.groupdict().get("time", "")
        return f"{match.group('date')}:{acquisition_time}"

    def destination(self, root: Path, day: date) -> Path:
        if self.key == "merra2":
            return root / "GMAO" / "MERRA2" / f"{day:%Y}" / f"{day:%m}"
        return root / "MODIS" / self.platform / "C61" / f"{day:%Y}" / f"{day:%j}"


PRODUCTS = {
    "merra2": Product(
        key="merra2",
        short_name="M2T1NXAER",
        version="5.12.4",
        platform="MERRA2",
        available_from=date(1980, 1, 1),
        filename_pattern=re.compile(r"\.tavg1_2d_aer_Nx\.(?P<date>\d{8})\.nc4$"),
    ),
    "terra": Product(
        key="terra",
        short_name="MOD08_D3",
        version="6.1",
        platform="Terra",
        available_from=date(2000, 2, 24),
        filename_pattern=re.compile(r"MOD08_D3\.A(?P<date>\d{7})\.061\.\d+\.hdf$"),
    ),
    "aqua": Product(
        key="aqua",
        short_name="MYD08_D3",
        version="6.1",
        platform="Aqua",
        available_from=date(2002, 7, 4),
        filename_pattern=re.compile(r"MYD08_D3\.A(?P<date>\d{7})\.061\.\d+\.hdf$"),
    ),
    "terra-l2": Product(
        key="terra-l2",
        short_name="MOD04_L2",
        version="6.1",
        platform="Terra",
        available_from=date(2000, 2, 24),
        filename_pattern=re.compile(r"MOD04_L2\.A(?P<date>\d{7})\.(?P<time>\d{4})\.061\.\d+\.hdf$"),
    ),
    "aqua-l2": Product(
        key="aqua-l2",
        short_name="MYD04_L2",
        version="6.1",
        platform="Aqua",
        available_from=date(2002, 7, 4),
        filename_pattern=re.compile(r"MYD04_L2\.A(?P<date>\d{7})\.(?P<time>\d{4})\.061\.\d+\.hdf$"),
    ),
}

DEFAULT_PRODUCT_KEYS = ("merra2", "terra", "aqua")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def iter_dates(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def external_url(granule: Any) -> str:
    links = [url for url in granule.data_links(access="external") if url.startswith("http")]
    if not links:
        raise RuntimeError("CMR granule has no external HTTP data link")
    return links[0]


def granule_size_mb(granule: Any) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return float(granule.size())
        except (AttributeError, TypeError, ValueError):
            return 0.0


def search_product(earthaccess: Any, product: Product, start: date, end: date) -> list[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        results = earthaccess.search_data(
            short_name=product.short_name,
            version=product.version,
            temporal=(f"{start.isoformat()}T00:00:00Z", f"{end.isoformat()}T23:59:59Z"),
        )

    by_identity: dict[str, list[Any]] = {}
    for granule in results:
        filename = Path(urlparse(external_url(granule)).path).name
        day = product.granule_date(filename)
        identity = product.granule_identity(filename)
        if day is not None and identity is not None and start <= day <= end:
            by_identity.setdefault(identity, []).append(granule)

    selected: list[Any] = []
    for identity in sorted(by_identity):
        candidates = by_identity[identity]
        # CMR can expose more than one production revision for a day or L2
        # acquisition. The final filename timestamp sorts in production order.
        selected.append(max(candidates, key=lambda item: external_url(item)))
    return selected


def write_row(writer: csv.DictWriter[str], **values: object) -> None:
    writer.writerow(values)


def stage_product(
    earthaccess: Any,
    product: Product,
    start: date,
    end: date,
    root: Path,
    writer: csv.DictWriter[str] | None,
    dry_run: bool,
) -> tuple[int, int, float]:
    granules = search_product(earthaccess, product, start, end)
    by_day: dict[date, list[Any]] = {}
    for granule in granules:
        filename = Path(urlparse(external_url(granule)).path).name
        day = product.granule_date(filename)
        if day is not None:
            by_day.setdefault(day, []).append(granule)

    expected_start = max(start, product.available_from)
    expected_days = list(iter_dates(expected_start, end)) if expected_start <= end else []
    missing = [day for day in expected_days if day not in by_day]
    total_mb = sum(granule_size_mb(granule) for granule in granules)
    print(
        f"{product.key}: {len(granules)} granule(s), {len(missing)} missing day(s), "
        f"{total_mb / 1024:.2f} GiB"
    )

    if dry_run:
        return len(granules), len(missing), total_mb

    assert writer is not None
    downloaded = 0
    for day in expected_days:
        day_granules = by_day.get(day, [])
        if not day_granules:
            write_row(
                writer,
                product=product.key,
                date=day.isoformat(),
                status="missing",
                cmr_size_mb="",
                local_size_bytes="",
                path="",
                source_url="",
            )
            continue

        for granule in day_granules:
            url = external_url(granule)
            filename = Path(urlparse(url).path).name
            destination = product.destination(root, day)
            target = destination / filename
            status = "existing"
            if not target.is_file() or target.stat().st_size == 0:
                destination.mkdir(parents=True, exist_ok=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    paths = earthaccess.download([granule], str(destination), threads=1)
                if not target.is_file() and paths:
                    target = Path(paths[0])
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"download did not produce a non-empty file: {target}")
                status = "downloaded"
                downloaded += 1

            write_row(
                writer,
                product=product.key,
                date=day.isoformat(),
                status=status,
                cmr_size_mb=f"{granule_size_mb(granule):.6f}",
                local_size_bytes=target.stat().st_size,
                path=target,
                source_url=url,
            )
            print(f"{product.key} {day}: {status} {target}")

    return downloaded, len(missing), total_mb


def selected_products(value: str) -> list[Product]:
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not keys or keys == ["all"]:
        keys = list(DEFAULT_PRODUCT_KEYS)
    unknown = sorted(set(keys) - PRODUCTS.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown product(s): {', '.join(unknown)}; choose from {', '.join(PRODUCTS)}"
        )
    return [PRODUCTS[key] for key in keys]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, required=True, help="First UTC date")
    parser.add_argument("--end", type=parse_date, required=True, help="Last UTC date")
    parser.add_argument(
        "--products",
        default="all",
        help=(
            "Comma-separated merra2,terra,aqua,terra-l2,aqua-l2 list "
            "(default all excludes opt-in L2 products)"
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Raw staging root")
    parser.add_argument("--allow-missing", action="store_true", help="Exit successfully with gaps")
    parser.add_argument("--dry-run", action="store_true", help="Search and estimate only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    products = selected_products(args.products)

    try:
        import earthaccess
    except ImportError as exc:
        raise SystemExit("earthaccess is required; activate the davinci environment") from exc

    auth = earthaccess.login(strategy="netrc")
    if not bool(getattr(auth, "authenticated", False)):
        raise SystemExit("Earthdata authentication from ~/.netrc failed")

    manifest = None
    manifest_handle = None
    if not args.dry_run:
        manifest_dir = args.root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        product_label = "-".join(product.key for product in products)
        manifest = manifest_dir / (
            f"manifest_{product_label}_{args.start:%Y%m%d}_{args.end:%Y%m%d}.tsv"
        )
        manifest_handle = manifest.open("w", newline="", encoding="utf-8")

    fieldnames = [
        "product",
        "date",
        "status",
        "cmr_size_mb",
        "local_size_bytes",
        "path",
        "source_url",
    ]
    writer = None
    if manifest_handle is not None:
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        manifest_handle.flush()

    missing_total = 0
    try:
        for product in products:
            _, missing, _ = stage_product(
                earthaccess, product, args.start, args.end, args.root, writer, args.dry_run
            )
            missing_total += missing
            if manifest_handle is not None:
                manifest_handle.flush()
    finally:
        if manifest_handle is not None:
            manifest_handle.close()

    if manifest is not None:
        print(f"Manifest: {manifest}")
    if missing_total and not args.allow_missing:
        print(
            f"ERROR: {missing_total} product-day(s) were missing; rerun with --allow-missing "
            "only after reviewing the manifest",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
