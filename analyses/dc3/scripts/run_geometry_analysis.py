#!/usr/bin/env python
"""Run DC3 dataset-only analysis pipeline.

Uses DAVINCI's geometry-only pipeline mode to generate flight track maps,
vertical profiles, time series, histograms, and summary statistics from
DC3 aircraft merge data.

Usage:
    python run_geometry_analysis.py [config_name]

Examples:
    python run_geometry_analysis.py dc3-geometry-dc8
    python run_geometry_analysis.py dc3-geometry-gv
    python run_geometry_analysis.py dc3-geometry-all-aircraft
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set analysis directory
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("DC3_ANALYSIS", str(ANALYSIS_DIR))
os.environ.setdefault("DC3_DATA", str(Path.home() / "Data" / "DC3"))


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "dc3-geometry-dc8"
    config_path = ANALYSIS_DIR / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        available = sorted(ANALYSIS_DIR.glob("configs/*.yaml"))
        print("Available configs:")
        for c in available:
            print(f"  {c.stem}")
        sys.exit(1)

    print("=" * 70)
    print("DC3 Dataset-Only Analysis")
    print("=" * 70)
    print(f"\nConfig:       {config_path.name}")
    print(f"DC3_DATA:     {os.environ['DC3_DATA']}")
    print(f"DC3_ANALYSIS: {os.environ['DC3_ANALYSIS']}")
    print()

    from davinci_monet.pipeline.runner import run_analysis

    result = run_analysis(str(config_path))

    print()
    print("=" * 70)
    if result.success:
        print(f"Analysis complete in {result.total_duration_seconds:.1f}s")
        print(f"Stages completed: {', '.join(result.completed_stages)}")
    else:
        print("Analysis failed!")
        for failed in result.failed_stages:
            print(f"  {failed.stage_name}: {failed.error}")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
