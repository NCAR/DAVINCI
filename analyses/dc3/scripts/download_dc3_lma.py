#!/usr/bin/env python
"""Download DC3 Lightning Mapping Array (LMA) data from NCAR EOL.

OKLMA (Oklahoma Lightning Mapping Array) NetCDF grids must be manually
ordered from the NCAR EOL data archive.

Dataset: NCAR EOL dataset 353.202
URL: https://data.eol.ucar.edu/dataset/353.202

After ordering, download the NetCDF grid files to ~/Data/DC3/lightning/oklma/.

Usage:
    python download_dc3_lma.py [--data-dir ~/Data/DC3]

This script creates the target directory structure and provides
instructions for manual data ordering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_URL = "https://data.eol.ucar.edu/dataset/353.202"

# LMA networks available for DC3
LMA_NETWORKS = {
    "oklma": {
        "name": "Oklahoma Lightning Mapping Array",
        "dataset": "353.202",
        "format": "CF-compliant NetCDF grids",
        "variables": ["flash_extent_density", "source_density", "flash_init_density"],
        "subdir": "lightning/oklma",
    },
    "colma": {
        "name": "Colorado Lightning Mapping Array",
        "dataset": "353.201",
        "format": "CF-compliant NetCDF grids",
        "variables": ["flash_extent_density", "source_density", "flash_init_density"],
        "subdir": "lightning/colma",
    },
    "nalma": {
        "name": "North Alabama Lightning Mapping Array",
        "dataset": "353.200",
        "format": "CF-compliant NetCDF grids",
        "variables": ["flash_extent_density", "source_density", "flash_init_density"],
        "subdir": "lightning/nalma",
    },
}


def setup_lma_dirs(data_dir: Path) -> None:
    """Create directory structure for LMA data."""
    for network_id, info in LMA_NETWORKS.items():
        target_dir = data_dir / info["subdir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {target_dir}")


def check_existing_data(data_dir: Path) -> dict[str, list[Path]]:
    """Check for existing LMA data files."""
    found: dict[str, list[Path]] = {}
    for network_id, info in LMA_NETWORKS.items():
        target_dir = data_dir / info["subdir"]
        if target_dir.exists():
            nc_files = sorted(target_dir.glob("*.nc"))
            if nc_files:
                found[network_id] = nc_files
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and check DC3 LMA data directories")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Data" / "DC3",
        help="Root data directory (default: ~/Data/DC3)",
    )
    args = parser.parse_args()

    print("DC3 Lightning Mapping Array (LMA) Data Setup")
    print("=" * 55)
    print()

    # Create directories
    print("Creating directory structure...")
    setup_lma_dirs(args.data_dir)
    print()

    # Check for existing data
    existing = check_existing_data(args.data_dir)
    if existing:
        print("Existing LMA data found:")
        for net_id, files in existing.items():
            print(f"  {net_id.upper()}: {len(files)} NetCDF files")
            for f in files[:3]:
                print(f"    {f.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        print()

    # Print download instructions
    print("LMA data must be manually ordered from NCAR EOL.")
    print("The data is not available for automated download.")
    print()
    print("Instructions:")
    print("-" * 55)

    for network_id, info in LMA_NETWORKS.items():
        target_dir = args.data_dir / info["subdir"]
        n_existing = len(existing.get(network_id, []))

        status = f" ({n_existing} files on disk)" if n_existing else ""
        print(f"\n  {info['name']}{status}")
        print(f"    Dataset: https://data.eol.ucar.edu/dataset/{info['dataset']}")
        print(f"    Format:  {info['format']}")
        print(f"    Save to: {target_dir}")
        print(f"    Variables: {', '.join(info['variables'])}")

    print()
    print("Benchmark case: 29-30 May 2012 Oklahoma supercell")
    print(f"  Primary dataset: {DATA_URL}")
    print()
    print("After downloading, run the analysis with:")
    print("  davinci run analyses/dc3/configs/dc3-geometry-lma-gemini.yaml")


if __name__ == "__main__":
    main()
