"""'A field site in context' — one card built only from env-data-mcp tool responses
saved by fetch_mcp_context.py (data/mcp/*.json).

    python lightning_talk/visuals/make_mcp_card.py
"""

from __future__ import annotations

import json
import re
from datetime import date
from collections import Counter
from pathlib import Path

import numpy as np

from cardkit import BOX_W, PLOT_H, THEMES, Card, render

HERE = Path(__file__).resolve().parent
MCP = HERE / "data" / "mcp"
OUT = HERE / "cards"
LAT, LON = 46.2531882, -119.4768203
MONTHS = "JFMAMJJASOND"


def load(name: str) -> tuple[list[dict], dict]:
    r = json.loads((MCP / f"{name}.json").read_text())
    return [rec | {"_geom": g} for g in r["data"] for rec in g["records"]], r["_meta"]


def monthly(records: list[dict], var: str) -> np.ndarray:
    out = np.full(12, np.nan)
    for rec in records:
        m = re.fullmatch(r"month-(\d\d)", str(rec.get("date", "")))
        if m and 1 <= int(m.group(1)) <= 12 and rec.get(var) is not None:
            out[int(m.group(1)) - 1] = float(rec[var])
    return out


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def build() -> Path:
    t = THEMES["biome"]
    clim, meta_clim = load("nasa_power_merra2_point_query")
    sun, meta_sun = load("nasa_power_syn1deg_point_query")
    daily, meta_daily = load("nasa_power_merra2_point_query__daily2023")
    soilg, meta_soilg = load("soilgrids_point_query")
    ssp, meta_ssp = load("ssurgo_soil_profile_point_query")
    ssa, meta_ssa = load("ssurgo_area_summary_point_query")
    gbif, meta_gbif = load("gbif_occurrence_point_query")
    trop, meta_trop = load("tropomi_point_query")
    metas = [meta_clim, meta_sun, meta_daily, meta_soilg, meta_ssp, meta_ssa, meta_gbif, meta_trop]
    n_calls = len(metas)
    latency = sum(float(m["latency_s"]) for m in metas)

    # --- POWER climate backbone -------------------------------------------------
    tmax, tmin, tmean = monthly(clim, "T2M_MAX"), monthly(clim, "T2M_MIN"), monthly(clim, "T2M")
    rain = monthly(clim, "PRECTOTCORR")                # mm/day, monthly climatology
    sw = monthly(sun, "ALLSKY_SFC_SW_DWN")             # W m-2, monthly climatology
    d23 = sorted((rec["date"], rec["T2M"]) for rec in daily if rec.get("T2M") is not None)
    doy = np.array([date.fromisoformat(d).timetuple().tm_yday for d, _ in d23])
    t23 = np.array([v for _, v in d23], float)

    # --- soil: SSURGO dominant component + SoilGrids nearest 0-5 cm pixel -------
    comp = ssp[0]
    muname = comp["_geom"].get("muname", "")
    def near(recs):  # nearest SoilGrids pixel to the site
        return min(recs, key=lambda r: (r["_geom"]["latitude"] - LAT) ** 2 + (r["_geom"]["longitude"] - LON) ** 2)
    sg = near(soilg)

    # --- life: GBIF 2023 within 10 km ---------------------------------------------
    n_occ = len(gbif)
    species = Counter(r.get("species") for r in gbif if r.get("species"))
    classes = Counter(r.get("class") for r in gbif)
    birds_pct = 100 * classes.get("Aves", 0) / max(n_occ, 1)

    # --- air: Sentinel-5P CO column, July 2023 ---------------------------------------
    co = np.array([r["OFFL-L2_CO"] for r in trop if isinstance(r.get("OFFL-L2_CO"), (int, float))], float)

    # ============================ SVG ============================
    W, H = BOX_W, PLOT_H
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    lx0, lx1 = 64, 690  # left panel x-extent (plot area)

    def X(day_of_year: float) -> float:
        return lx0 + (day_of_year - 1) / 364 * (lx1 - lx0)

    def Xm(mi: int) -> float:  # month centre
        return X(15.5 + mi * 30.4)

    def panel(y0, y1, label, src, vmin, vmax, ticks, fmt):
        s.append(f'<text x="{lx0}" y="{y0 - 12}" fill="{t.text}" font-family="HNC" font-size="24" letter-spacing="1.5">{label}</text>')
        s.append(f'<text x="{lx1}" y="{y0 - 12}" text-anchor="end" fill="{t.muted}" font-family="Helvetica Neue" font-size="17">{src}</text>')
        def Y(v):
            return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)
        for tv in ticks:
            s.append(f'<line x1="{lx0}" y1="{Y(tv):.1f}" x2="{lx1}" y2="{Y(tv):.1f}" stroke="{t.grid}" stroke-width="1.5"/>')
            s.append(f'<text x="{lx0 - 12}" y="{Y(tv) + 8:.1f}" text-anchor="end" fill="{t.accent}" font-family="HNC" font-size="22">{fmt(tv)}</text>')
        return Y

    # temperature: 2023 daily over the 1991-2020 monthly max/min band
    Y = panel(34, 250, "TEMPERATURE · 2023 DAILY", "NASA POWER · MERRA-2", -20, 40, (-20, 0, 20, 40), lambda v: f"{v:+d}°".replace("+0°", "0°").replace("-", "−"))
    band = " ".join(f"{'M' if i == 0 else 'L'}{Xm(i):.1f},{Y(tmax[i]):.1f}" for i in range(12))
    band += " " + " ".join(f"L{Xm(i):.1f},{Y(tmin[i]):.1f}" for i in range(11, -1, -1)) + " Z"
    s.append(f'<path d="{band}" fill="rgba(255,255,255,0.16)"/>')
    s.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{Xm(i):.1f},{Y(tmean[i]):.1f}" for i in range(12)) +
             f'" fill="none" stroke="{t.reference}" stroke-width="3" stroke-dasharray="2 7" stroke-linecap="round"/>')
    s.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{X(d):.1f},{Y(v):.1f}" for i, (d, v) in enumerate(zip(doy, t23))) +
             f'" fill="none" stroke="{t.series}" stroke-width="3.5" stroke-linejoin="round"/>')
    s.append(f'<text x="{lx1 - 6}" y="{Y(tmin[11]) - 8:.1f}" text-anchor="end" fill="{t.muted}" font-family="Helvetica Neue" font-size="16">band: 1991–2020 normal, daily max to min</text>')

    # sunlight
    Y = panel(300, 400, "SUNLIGHT", "NASA POWER · CERES SYN1deg", 0, 400, (0, 200, 400), lambda v: f"{v:d}")
    area = " ".join(f"{'M' if i == 0 else 'L'}{Xm(i):.1f},{Y(sw[i]):.1f}" for i in range(12))
    s.append(f'<path d="M{Xm(0):.1f},{Y(0):.1f} ' + area.replace("M", "L", 1) + f' L{Xm(11):.1f},{Y(0):.1f} Z" fill="rgba(245,185,66,0.22)"/>')
    s.append(f'<path d="{area}" fill="none" stroke="{t.series2}" stroke-width="3.5" stroke-linejoin="round"/>')
    s.append(f'<text x="{lx1 - 6}" y="{Y(sw[6]) + 34:.1f}" text-anchor="end" fill="{t.muted}" font-family="Helvetica Neue" font-size="16">W m⁻²</text>')

    # rain
    Y = panel(444, 530, "RAIN", "NASA POWER · MERRA-2", 0, 2, (0, 1, 2), lambda v: f"{v:d}")
    bw = (lx1 - lx0) / 12 * 0.62
    for i in range(12):
        s.append(f'<rect x="{Xm(i) - bw / 2:.1f}" y="{Y(rain[i]):.1f}" width="{bw:.1f}" height="{Y(0) - Y(rain[i]):.1f}" rx="3" fill="{t.accent}"/>')
    s.append(f'<text x="{lx1 - 6}" y="{Y(2) + 22:.1f}" text-anchor="end" fill="{t.muted}" font-family="Helvetica Neue" font-size="16">mm / day</text>')
    for i, ch in enumerate(MONTHS):
        s.append(f'<text x="{Xm(i):.1f}" y="{H - 4}" text-anchor="middle" fill="{t.accent}" font-family="HNC" font-size="24">{ch}</text>')

    # --- right column: tiles -----------------------------------------------------
    rx, rw = 740, W - 740
    tiles = [
        ("SOIL", esc(muname.split(",")[0].upper()),
         [f"{comp['taxorder']} · {float(comp['sandtotal_r']):.0f} % sand · pH {float(comp['ph1to1h2o_r']):.1f}",
          f"{comp['drainagecl'].lower()} · SOC {sg['soc_0-5cm_mean']:.0f} g kg⁻¹ (0–5 cm)"],
         "USDA SSURGO · ISRIC SoilGrids"),
        ("LIFE", f"{len(species)} SPECIES",
         [f"{n_occ:,} observations within 10 km in 2023",
          f"{birds_pct:.0f} % birds · {esc(species.most_common(1)[0][0])} most seen"],
         "GBIF"),
        ("AIR", f"CO {co.mean():.3f} mol m⁻²",
         [f"column mean, {co.size} clear days, July 2023", "NO₂, SO₂, CH₄, HCHO, O₃ on request"],
         "Sentinel-5P TROPOMI"),
    ]
    th, tg, ty = 128, 14, 8
    for i, (lab, big, lines, src) in enumerate(tiles):
        y = ty + i * (th + tg)
        s.append(f'<rect x="{rx}" y="{y}" width="{rw}" height="{th}" rx="14" fill="{t.node}" stroke="{t.edge}" stroke-width="2"/>')
        s.append(f'<text x="{rx + 22}" y="{y + 32}" fill="{t.accent}" font-family="HNC" font-size="22" letter-spacing="2">{lab}</text>')
        s.append(f'<text x="{rx + rw - 18}" y="{y + 30}" text-anchor="end" fill="{t.muted}" font-family="Helvetica Neue" font-size="15">{src}</text>')
        s.append(f'<text x="{rx + 22}" y="{y + 74}" fill="{t.text}" font-family="HNC" font-size="40">{big}</text>')
        for j, line in enumerate(lines):
            s.append(f'<text x="{rx + 22}" y="{y + 98 + j * 21}" fill="{t.accent}" font-family="Helvetica Neue" font-size="17">{line}</text>')
    # hero under the tiles: annual rainfall from the POWER climatology (mm/day x days in month)
    days = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    annual_rain = float(np.nansum(rain * days))
    hy = ty + 3 * (th + tg) + 8
    s.append(f'<text x="{rx + rw}" y="{hy + 70}" text-anchor="end" fill="{t.text}" font-family="HNC" font-size="84" letter-spacing="-1">{annual_rain:.0f} mm</text>')
    s.append(f'<text x="{rx + rw}" y="{hy + 108}" text-anchor="end" fill="{t.accent}" font-family="HNC" font-size="30" letter-spacing="2">OF RAIN A YEAR</text>')
    s.append("</svg>")

    card = Card(
        title="A field site in context",
        subtitle="Yakima River Valley, Washington · KBase env-data-mcp",
        plot_svg="\n".join(s), theme=t, title_px=92,
        footnote=("Every value is a tool response from github.com/kbaseincubator/env-data-mcp for 46.253 °N, "
                  "119.477 °W (the server's benchmark site). POWER climatologies via the AWS Zarr stores; "
                  "SSURGO dominant component (90 %); SoilGrids nearest 250 m pixel; GBIF human observations; "
                  "TROPOMI OFFL L2 CO. Latency is the sum of the servers' reported latency_s."),
    )
    print(f"annual rain={annual_rain:.0f} mm | calls={n_calls} latency={latency:.1f}s | soil: {muname} | species={len(species)} occ={n_occ} birds={birds_pct:.0f}% | CO={co.mean():.4f} n={co.size}")
    print("clim T2M:", np.round(tmean, 1)); print("SW:", np.round(sw, 0)); print("rain:", np.round(rain, 2))
    return render(card.html(), OUT / "card_field_site_context.png")


if __name__ == "__main__":
    print(build())
