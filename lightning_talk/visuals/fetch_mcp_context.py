"""Fetch real environmental context for one field site through the KBase env-data-mcp
server, over the MCP protocol (stdio), exactly as an agent would.

Every response is saved verbatim (data + _meta) to data/mcp/<tool>.json; the card is
built from those files only.

Run with the server's own environment:
    cd ~/EarthData/env-data-mcp && uv run python <this file>
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT = Path(__file__).resolve().parent / "data" / "mcp"
SERVER_DIR = Path.home() / "EarthData" / "env-data-mcp"

# The server's own benchmark site: Yakima River Valley, WA (semi-arid, agricultural).
LAT, LON = 46.2531882, -119.4768203

CALLS = [
    ("nasa_power_merra2_point_query", dict(
        latitude=LAT, longitude=LON, start_date="1991-01-01", end_date="2020-12-31",
        temporal_resolution="climatology",
        variables=["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS10M", "GWETROOT"],
        max_runtime_s=300)),
    ("nasa_power_syn1deg_point_query", dict(
        latitude=LAT, longitude=LON, start_date="2001-01-01", end_date="2020-12-31",
        temporal_resolution="climatology",
        variables=["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "ALLSKY_SFC_LW_DWN"],
        max_runtime_s=300)),
    ("nasa_power_merra2_point_query__daily2023", dict(
        latitude=LAT, longitude=LON, start_date="2023-01-01", end_date="2023-12-31",
        temporal_resolution="daily", variables=["T2M", "PRECTOTCORR"], max_runtime_s=300)),
    ("soilgrids_point_query", dict(latitude=LAT, longitude=LON, radius_km=1.0, max_runtime_s=120)),
    ("ssurgo_soil_profile_point_query", dict(latitude=LAT, longitude=LON, max_runtime_s=120)),
    ("ssurgo_area_summary_point_query", dict(latitude=LAT, longitude=LON, max_runtime_s=120)),
    ("gbif_occurrence_point_query", dict(
        latitude=LAT, longitude=LON, start_date="2023-01-01", end_date="2023-12-31",
        radius_km=10.0, limit=5000, max_runtime_s=180)),
    ("tropomi_point_query", dict(
        latitude=LAT, longitude=LON, start_date="2023-07-01", end_date="2023-07-31", max_runtime_s=300)),
]


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(command="uv", args=["run", "env-data-mcp"], cwd=str(SERVER_DIR))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            (OUT / "_tools.json").write_text(json.dumps(
                [{"name": t.name, "description": (t.description or "")[:200]} for t in tools.tools], indent=1))
            print(f"server exposes {len(tools.tools)} tools")
            for name, args in CALLS:
                tool = name.split("__")[0]
                t0 = time.perf_counter()
                try:
                    raw = await session.call_tool(tool, arguments=args)
                    text = raw.content[0].text
                    result = json.loads(text)
                except Exception as exc:  # noqa: BLE001 - record, keep going
                    result = {"data": [], "_meta": {"success": False, "error": repr(exc)}}
                wall = time.perf_counter() - t0
                meta = result.get("_meta", {})
                (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
                n = meta.get("total_records_returned")
                print(f"{name:42s} success={meta.get('success')} records={n} "
                      f"latency={meta.get('latency_s')} wall={wall:.1f}s "
                      f"{('ERROR: ' + str(meta.get('error'))[:160]) if not meta.get('success') else ''}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
