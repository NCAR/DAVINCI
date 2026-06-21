#!/usr/bin/env python
"""Organize ASIA-AQ plots into tidy subdirectories.

The DAVINCI pipeline writes every figure flat into ``analysis.output_dir`` (no
native per-source/per-flight subdir support), so run this AFTER each pipeline
run to sort the output:

  Plots/ASIA-AQ/
    AirNow/                 <- airnow_*.{png,pdf}
    AERONET/                <- aeronet_*.{png,pdf}
    DC8/
      2024-02-06/           <- dc8_*_2024-02-06.{png,pdf}
      2024-02-07/
      ...

Idempotent: files already in their subfolders are left alone, so it is safe to
re-run repeatedly.

Usage:
    python analyses/asia-aq/scripts/organize_plots.py
"""

import re
import shutil
from pathlib import Path

BASE = Path("/glade/work/fillmore/Plots/ASIA-AQ")
EXTS = {".png", ".pdf"}
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Surface plots: route by source-label prefix -> subdir name.
SURFACE_MAP = {"airnow_": "AirNow", "aeronet_": "AERONET"}


def _move(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_dir / src.name))


def organize_surface() -> int:
    """Move flat surface figures into per-source subdirs. Returns count moved."""
    moved = 0
    for f in sorted(BASE.glob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        for prefix, sub in SURFACE_MAP.items():
            if f.name.startswith(prefix):
                _move(f, BASE / sub)
                moved += 1
                break
    return moved


def organize_dc8() -> int:
    """Move flat DC-8 figures into per-flight (date) subdirs. Returns count moved."""
    dc8 = BASE / "DC8"
    if not dc8.is_dir():
        return 0
    moved = 0
    # glob("*") is non-recursive: only files directly in DC8/, not in date subdirs.
    for f in sorted(dc8.glob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        m = DATE_RE.search(f.name)
        if m:
            _move(f, dc8 / m.group(1))
            moved += 1
    return moved


def main() -> None:
    n_surface = organize_surface()
    n_dc8 = organize_dc8()
    print(f"Surface figures organized into AirNow/ + AERONET/: {n_surface}")
    print(f"DC-8 figures organized into per-flight subfolders:  {n_dc8}")


if __name__ == "__main__":
    main()
