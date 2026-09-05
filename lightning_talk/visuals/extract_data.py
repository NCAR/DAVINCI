"""Run the talk's pipeline config and write its outputs as tidy CSV for the cards.

Every number the cards show comes from ``PipelineRunner`` context objects:
``context.sources[<key>].data`` (an xarray Dataset). Nothing is recomputed here.

    python lightning_talk/visuals/extract_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "configs" / "power-annual-anomaly.yaml"
DATA = HERE / "data"


def main() -> None:
    os.environ.setdefault("POWER_ANALYSIS", str(HERE))
    os.environ.setdefault("POWER_CACHE", str(REPO / "analyses" / "power" / "cache"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    from davinci_monet.pipeline.runner import run_analysis

    result = run_analysis(str(CONFIG))
    if not result.success or result.context is None:
        raise SystemExit(f"pipeline failed: {[s.name for s in result.stage_results if not s.success]}")
    DATA.mkdir(exist_ok=True)
    for key, var, fname in (
        ("t2m_annual_anomaly", "T2M", "t2m_annual_anomaly.csv"),
        ("power_annual", "T2M", "t2m_annual.csv"),
        ("power_annual", "ALLSKY_SFC_SW_DWN", "swdn_annual.csv"),
    ):
        ds = result.context.sources[key].data
        da = ds[var]
        df = da.to_dataframe(name=var).reset_index()
        df["year"] = pd.to_datetime(df["time"]).dt.year
        df = df[["site", "year", var]].sort_values(["site", "year"])
        df.to_csv(DATA / fname, index=False)
        print(f"{fname}: {len(df)} rows, sites={sorted(df.site.unique())}, "
              f"years {df.year.min()}-{df.year.max()}, units={da.attrs.get('units')}")
    clim = result.context.sources["t2m_annual_anomaly"].data["T2M_climatology"]
    print("baseline means (K):", {str(s): round(float(v), 2) for s, v in zip(clim.site.values, clim.values.ravel())})


if __name__ == "__main__":
    main()
