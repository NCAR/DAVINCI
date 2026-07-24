#!/usr/bin/env python
"""Run MODIS AOD vs CAM6 evaluation pipeline.

Reads MODIS L2 Terra+Aqua granules, bins onto the CAM6 grid, pairs
with dataset AODVIS, and generates statistics.

Usage:
    python run_evaluation.py

Or use the CLI directly:
    davinci run ../configs/modis-aod-cam6-gemini.yaml
"""

from pathlib import Path

from davinci_monet.pipeline.runner import run_analysis

CONFIG = Path(__file__).parent.parent / "configs" / "modis-aod-cam6-gemini.yaml"

if __name__ == "__main__":
    result = run_analysis(str(CONFIG))
    if result.success:
        print(f"\nPipeline completed in {result.total_duration_seconds:.1f}s")
    else:
        print(f"\nPipeline failed: {result.error}")
        raise SystemExit(1)
