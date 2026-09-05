"""Data cards for the talk, from the CSVs written by extract_data.py.

    python lightning_talk/visuals/make_cards.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cardkit import THEMES, Card, render, stripes_svg, trend_card_svg

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "cards"
ACCESSED = "2026-09-05"
SITES_N_TO_S = ["Barrow, AK", "Boulder, CO", "Mauna Loa, HI", "South Pole"]
ROW_LABELS = {"Barrow, AK": "Barrow · 71 °N", "Boulder, CO": "Boulder · 40 °N",
              "Mauna Loa, HI": "Mauna Loa · 20 °N", "South Pole": "South Pole · 90 °S"}


def arctic_trend_card(df: pd.DataFrame) -> Path:
    t = THEMES["arctic"]
    d = df[df.site == "Barrow, AK"]
    svg, change = trend_card_svg(d.year.values, d.T2M.values, t, "Barrow, Alaska", unit_suffix="°",
                                 y_range=(-4.5, 4.5), y_step=2)  # leaves the lower right free for the hero
    sign = "+" if change >= 0 else "−"
    card = Card(
        title="A warming Arctic",
        subtitle="Annual 2 m temperature vs the 1991–2020 average · 71 °N",
        plot_svg=svg, theme=t,
        hero=f"{sign}{abs(change):.1f}°C", hero_unit="since 1981", hero_top=509,
        footnote=("Annual means of NASA POWER monthly 2 m temperature (MERRA-2), as departures from the "
                  f"1991–2020 average. White line: linear trend 1981–2024; the number is its change over "
                  f"the period. Accessed {ACCESSED}."),
    )
    print(f"Barrow trend change 1981-2024: {change:+.2f} K")
    return render(card.html(), OUT / "card_arctic_trend.png")


def stripes_card(df: pd.DataFrame) -> Path:
    t = THEMES["ink"]
    rows = []
    for site in SITES_N_TO_S:
        d = df[df.site == site]
        rows.append((ROW_LABELS[site], d.year.values, d.T2M.values))
    limit = float(np.ceil(np.nanpercentile(np.abs(df.T2M.values), 98) * 2) / 2)  # to nearest 0.5
    svg = stripes_svg(rows, t, limit)
    card = Card(
        title="Four decades, four sites",
        subtitle="Annual 2 m temperature vs the 1991–2020 average",
        plot_svg=svg, theme=t,
        footnote=(f"NASA POWER monthly 2 m temperature (MERRA-2), annual means as departures from each site's "
                  f"1991–2020 average. One colour scale for all sites, ±{limit:g} °C, after Ed Hawkins' "
                  f"warming stripes. Accessed {ACCESSED}."),
    )
    print(f"stripes colour limit: ±{limit:g} K")
    return render(card.html(), OUT / "card_stripes_four_sites.png")


if __name__ == "__main__":
    df = pd.read_csv(DATA / "t2m_annual_anomaly.csv")
    for p in (arctic_trend_card(df), stripes_card(df)):
        print(p)
