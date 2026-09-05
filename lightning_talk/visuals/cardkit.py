"""cardkit — broadcast-style graphic cards for the POWER lightning talk.

Renders HTML/SVG cards to PNG with Playwright/Chromium. The style follows
``climate-central-style-notes.md``: a 16:9 frame with every element inside a
centred 60 % x 80 % content box, a condensed all-caps headline, one message,
one hero number, a source line, and a text lockup. Fonts are macOS system
faces (Helvetica Neue and its Condensed Bold/Black) that PowerPoint on this
Mac also has, so slide titles can match.

This module knows nothing about DAVINCI or POWER; it draws what it is given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

W, H = 1920, 1080
BOX_LEFT, BOX_TOP, BOX_W, BOX_H = 384, 108, 1152, 864  # 20 % / 10 % / 60 % / 80 %
PLOT_TOP, PLOT_H = 200, 560  # plot slot inside the box (below title + subtitle)

# ---------------------------------------------------------------------------
# Themes — dark, topic-coloured fields. Colours from the style notes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    background: str          # CSS background for <body>
    text: str = "#ffffff"
    accent: str = "#aac7e2"  # subtitle, axis labels, lockup
    muted: str = "rgba(170,199,226,0.62)"  # source line
    grid: str = "#1e4d7c"    # hairline gridlines, one step off the surface
    series: str = "#f0803c"  # the one data series
    series2: str = "#f5b942"
    reference: str = "#ffffff"  # trend / normal line
    node: str = "rgba(255,255,255,0.08)"  # diagram box fill
    edge: str = "rgba(255,255,255,0.35)"  # diagram box stroke


THEMES = {
    # Flat card blue (Climate Central's cooling-demand card).
    "card": Theme("card", "#17416b", grid="#285d8f", accent="#82bee6",
                  muted="rgba(130,190,230,0.62)"),
    # Navy briefing field with a faint warm glow bottom-right: cold place, warming.
    "arctic": Theme(
        "arctic",
        "radial-gradient(900px 620px at 88% 96%, rgba(240,128,60,0.22), rgba(240,128,60,0) 70%),"
        " linear-gradient(180deg, #021f3d 0%, #043058 100%)",
        grid="#1b4a7a",
    ),
    # Plain navy for diagrams and neutral cards.
    "navy": Theme("navy", "linear-gradient(180deg, #022749 0%, #032d54 100%)", grid="#1b4a7a"),
    # Near-black for stripes: the colour is the graphic.
    "ink": Theme("ink", "#161417", accent="#d9d9d9", muted="rgba(217,217,217,0.6)", grid="#2a2a2a"),
    # Deep green-teal for ecosystem / field-site context.
    "biome": Theme(
        "biome",
        "radial-gradient(900px 620px at 10% 100%, rgba(240,128,60,0.14), rgba(240,128,60,0) 70%),"
        " linear-gradient(180deg, #0a2a2e 0%, #10403d 100%)",
        accent="#9fd6c8", muted="rgba(159,214,200,0.62)", grid="#1f5150",
        node="rgba(255,255,255,0.07)", edge="rgba(159,214,200,0.35)",
    ),
}

# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

CSS = """
@font-face {{ font-family: "HNC";  src: local("HelveticaNeue-CondensedBold"); }}
@font-face {{ font-family: "HNCK"; src: local("HelveticaNeue-CondensedBlack"); }}
html, body {{ margin: 0; width: {W}px; height: {H}px; overflow: hidden; }}
body {{ background: {bg}; color: {text}; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       -webkit-font-smoothing: antialiased; }}
.box {{ position: absolute; left: {bl}px; top: {bt}px; width: {bw}px; height: {bh}px; }}
.title {{ font: 700 {title_px}px/0.95 "HNC"; text-transform: uppercase; letter-spacing: 0.5px; margin: 0; }}
.subtitle {{ font: 400 40px/1.2 "Helvetica Neue"; color: {accent}; margin: 14px 0 0; }}
.plot {{ position: absolute; left: 0; top: {plot_top}px; width: {bw}px; height: {plot_h}px; }}
.plot svg {{ display: block; }}
.hero {{ position: absolute; right: 0; top: {hero_top}px; text-align: right; line-height: 0.9; }}
.hero .n {{ font: 700 140px/0.9 "HNC"; letter-spacing: -2px; }}
.hero .u {{ font: 700 40px/1 "HNC"; letter-spacing: 3px; text-transform: uppercase; margin-top: 6px; }}
.foot {{ position: absolute; left: 0; bottom: 0; font: 300 19px/1.4 "Helvetica Neue"; color: {muted};
         max-width: 780px; }}
.lockup {{ position: absolute; right: 0; bottom: 0; text-align: right; color: {accent};
           font: 700 22px/1.3 "HNC"; letter-spacing: 3px; text-transform: uppercase; }}
.lockup .data {{ font: 400 17px/1.3 "Helvetica Neue"; letter-spacing: 0.5px; color: {muted};
                 text-transform: none; }}
"""


@dataclass
class Card:
    title: str
    subtitle: str
    plot_svg: str
    theme: Theme = THEMES["navy"]
    hero: str | None = None
    hero_unit: str | None = None
    hero_top: int = 430
    footnote: str = ""
    show_footnote: bool = False  # fine print is added as PowerPoint annotations instead
    credit: str = ""  # fine print; goes into PowerPoint annotations
    lockup: str = "NSF NCAR &middot; DAVINCI"
    title_px: int = 96
    extra_css: str = ""

    def html(self) -> str:
        t = self.theme
        css = CSS.format(W=W, H=H, bg=t.background, text=t.text, accent=t.accent, muted=t.muted,
                         bl=BOX_LEFT, bt=BOX_TOP, bw=BOX_W, bh=BOX_H, plot_top=PLOT_TOP,
                         plot_h=PLOT_H, hero_top=self.hero_top, title_px=self.title_px)
        hero = ""
        if self.hero:
            unit = f'<div class="u">{self.hero_unit}</div>' if self.hero_unit else ""
            hero = f'<div class="hero"><div class="n">{self.hero}</div>{unit}</div>'
        foot = f'<div class="foot">{self.footnote}</div>' if (self.show_footnote and self.footnote) else ""
        credit = f'<div class="data">{self.credit}</div>' if self.credit else ""
        return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}{self.extra_css}</style></head>
<body><div class="box">
  <h1 class="title">{self.title}</h1>
  <div class="subtitle">{self.subtitle}</div>
  <div class="plot">{self.plot_svg}</div>
  {hero}
  {foot}
  <div class="lockup">{credit}{self.lockup}</div>
</div></body></html>"""


def render(html: str, out_png: Path, scale: int = 2) -> Path:
    """Render *html* at 1920 x 1080 CSS px, *scale* x for a crisp PowerPoint image."""
    from playwright.sync_api import sync_playwright

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=scale)
        page.set_content(html)
        page.wait_for_timeout(150)  # let local() fonts settle
        page.screenshot(path=str(out_png))
        browser.close()
    return out_png


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    ticks = np.arange(lo, hi + step / 2, step)
    return [round(float(t), 6) for t in ticks]


def _fmt_tick(v: float) -> str:
    s = f"{v:+.1f}" if abs(v) < 10 and v != int(v) else f"{v:+.0f}"
    return s.replace("-", "−").replace("+0.0", "0").replace("+0", "0") if v == 0 else s.replace("-", "−")


def linear_trend(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope and intercept, ignoring NaNs. (The one number a card
    adds to pipeline output; the footnote must say so.)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept)


def trend_card_svg(
    years: Sequence[int],
    values: Sequence[float],
    theme: Theme,
    place: str,
    *,
    unit_suffix: str = "",
    y_pad: tuple[float, float] = (0.15, 0.15),  # (below, above) as fractions of the data span
    y_range: tuple[float, float] | None = None,  # explicit axis limits (win over y_pad)
    y_step: float | None = None,                # explicit tick step
    show_trend: bool = True,
    width: int = BOX_W,
    height: int = PLOT_H,
) -> tuple[str, float]:
    """Climate Central-style local trend plot: thin series with dots, straight
    reference trend, first/last year and the place name under the axis.
    Returns (svg, trend_change_over_period)."""
    years = np.asarray(years, int)
    vals = np.asarray(values, float)
    ok = np.isfinite(vals)
    ml, mr, mt, mb = 96, 40, 20, 100  # margins: y labels left, years/place below
    pw, ph = width - ml - mr, height - mt - mb
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    span = vmax - vmin
    if y_range is not None:
        lo, hi = y_range
        step = y_step or (_nice_ticks(lo, hi)[1] - _nice_ticks(lo, hi)[0])
        ticks = [round(float(v), 6) for v in np.arange(math.ceil(lo / step) * step, hi + step / 2, step)]
    else:
        ticks = _nice_ticks(vmin - y_pad[0] * span, vmax + y_pad[1] * span)
        lo, hi = ticks[0], ticks[-1]
    x0, x1 = years.min(), years.max()

    def X(v: float) -> float:
        return ml + (v - x0) / (x1 - x0) * pw

    def Y(v: float) -> float:
        return mt + (hi - v) / (hi - lo) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    # gridlines + y labels (condensed face, accent colour)
    for t in ticks:
        y = Y(t)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" stroke="{theme.grid}" stroke-width="1.5"/>')
        out.append(f'<text x="{ml - 18}" y="{y + 11:.1f}" text-anchor="end" fill="{theme.accent}" '
                   f'font-family="HNC" font-size="32">{_fmt_tick(t)}{unit_suffix}</text>')
    # series
    pts = [(X(x), Y(v)) for x, v, k in zip(years, vals, ok) if k]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    out.append(f'<path d="{path}" fill="none" stroke="{theme.series}" stroke-width="5" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')
    for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{theme.series}"/>')
    # trend
    change = float("nan")
    if show_trend:
        slope, intercept = linear_trend(years, vals)
        ya, yb = slope * x0 + intercept, slope * x1 + intercept
        change = yb - ya
        out.append(f'<line x1="{X(x0):.1f}" y1="{Y(ya):.1f}" x2="{X(x1):.1f}" y2="{Y(yb):.1f}" '
                   f'stroke="{theme.reference}" stroke-width="9" stroke-linecap="round"/>')
    # x labels: first year, place, last year
    yb_ = height - 22
    out.append(f'<text x="{ml}" y="{yb_}" fill="{theme.accent}" font-family="HNC" font-size="44">{x0}</text>')
    out.append(f'<text x="{ml + pw / 2:.1f}" y="{yb_}" text-anchor="middle" fill="{theme.text}" '
               f'font-family="HNC" font-size="48" letter-spacing="1">{place.upper()}</text>')
    out.append(f'<text x="{width - mr}" y="{yb_}" text-anchor="end" fill="{theme.accent}" font-family="HNC" font-size="44">{x1}</text>')
    out.append("</svg>")
    return "\n".join(out), change


# Hawkins' warming-stripes colours: ColorBrewer Blues (dark->light) + Reds (light->dark).
STRIPE_COLOURS = [
    "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
    "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d",
]


def stripe_colour(v: float, limit: float) -> str:
    """Map an anomaly in [-limit, +limit] onto the 16 stripe bins; NaN -> gray."""
    if not np.isfinite(v):
        return "#4a4a4a"
    f = (v + limit) / (2 * limit)
    i = int(np.clip(math.floor(f * len(STRIPE_COLOURS)), 0, len(STRIPE_COLOURS) - 1))
    return STRIPE_COLOURS[i]


def stripes_svg(
    rows: Sequence[tuple[str, Sequence[int], Sequence[float]]],
    theme: Theme,
    limit: float,
    *,
    width: int = BOX_W,
    height: int = PLOT_H,
    label_w: int = 310,
    gap: int = 18,
) -> str:
    """Stacked warming stripes, one row per site, common colour scale."""
    n = len(rows)
    row_h = (height - 70 - gap * (n - 1)) / n
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for r, (label, years, vals) in enumerate(rows):
        years = np.asarray(years, int)
        vals = np.asarray(vals, float)
        y = r * (row_h + gap)
        sw = (width - label_w) / len(years)
        out.append(f'<text x="{label_w - 22}" y="{y + row_h / 2 + 14:.1f}" text-anchor="end" fill="{theme.text}" '
                   f'font-family="HNC" font-size="34">{label.upper()}</text>')
        for i, (yr, v) in enumerate(zip(years, vals)):
            out.append(f'<rect x="{label_w + i * sw:.2f}" y="{y:.1f}" width="{sw + 0.6:.2f}" height="{row_h:.1f}" '
                       f'fill="{stripe_colour(v, limit)}"/>')
    yb = height - 16
    y0, y1 = int(np.asarray(rows[0][1]).min()), int(np.asarray(rows[0][1]).max())
    out.append(f'<text x="{label_w}" y="{yb}" fill="{theme.accent}" font-family="HNC" font-size="44">{y0}</text>')
    out.append(f'<text x="{width}" y="{yb}" text-anchor="end" fill="{theme.accent}" font-family="HNC" font-size="44">{y1}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Diagram primitives (boxes and arrows in the same voice)
# ---------------------------------------------------------------------------


def svg_open(width: int = BOX_W, height: int = PLOT_H, theme: Theme = THEMES["navy"]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{theme.accent}"/></marker></defs>')


def node(x: float, y: float, w: float, h: float, label: str, theme: Theme, *,
         sub: str = "", fill: str | None = None, stroke: str | None = None, size: int = 34,
         text: str | None = None, rx: int = 14) -> str:
    fill = fill or theme.node
    stroke = stroke or theme.edge
    text = text or theme.text
    cy = y + h / 2
    lines = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>']
    if sub:
        lines.append(f'<text x="{x + w / 2}" y="{cy - 4}" text-anchor="middle" fill="{text}" font-family="HNC" font-size="{size}">{label}</text>')
        lines.append(f'<text x="{x + w / 2}" y="{cy + 30}" text-anchor="middle" fill="{theme.accent}" font-family="Helvetica Neue" font-size="20">{sub}</text>')
    else:
        lines.append(f'<text x="{x + w / 2}" y="{cy + size * 0.36}" text-anchor="middle" fill="{text}" font-family="HNC" font-size="{size}">{label}</text>')
    return "\n".join(lines)


def arrow(x1: float, y1: float, x2: float, y2: float, theme: Theme, *, width: float = 4, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="10 10"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{theme.accent}" stroke-width="{width}" '
            f'stroke-linecap="round" marker-end="url(#arr)"{dash}/>')


def label(x: float, y: float, s: str, theme: Theme, *, size: int = 26, anchor: str = "middle",
          colour: str | None = None, family: str = "Helvetica Neue", weight: str = "400") -> str:
    colour = colour or theme.accent
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{colour}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}">{s}</text>')
