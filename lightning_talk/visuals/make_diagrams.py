"""Diagram cards for the talk: same frame, same voice as the data cards.

Run in the davinci env from the repo root:
    python lightning_talk/visuals/make_diagrams.py
Writes PNGs to lightning_talk/visuals/cards/.
"""

from __future__ import annotations

from pathlib import Path

from cardkit import THEMES, BOX_W, PLOT_H, Card, arrow, label, node, render, svg_open

OUT = Path(__file__).parent / "cards"


def pipeline_diagram() -> Path:
    """Agent -> config -> validated -> pipeline computes -> outputs -> LLM reads."""
    t = THEMES["navy"]
    s = [svg_open(theme=t)]
    y, h, w, gap = 190, 150, 216, 18
    xs = [i * (w + gap) for i in range(5)]
    names = [("AGENT", "writes the request"), ("CONFIG", "YAML, validated"),
             ("PIPELINE", "load · pair · stats · plot"), ("OUTPUTS", "CSV · PDF · manifest"),
             ("LLM", "reads, then writes")]
    for x, (n, sub) in zip(xs, names):
        strong = n in ("PIPELINE", "OUTPUTS")
        s.append(node(x, y, w, h, n, t, sub=sub,
                      fill="rgba(240,128,60,0.16)" if strong else None,
                      stroke=t.series if strong else None))
    for a, b in zip(xs[:-1], xs[1:]):
        s.append(arrow(a + w + 4, y + h / 2, b - 6, y + h / 2, t, width=3))
    # validation gate: a badge on the CONFIG -> PIPELINE arrow, captioned above the row
    gx = xs[1] + w + gap / 2
    s.append(f'<circle cx="{gx}" cy="{y + h / 2}" r="24" fill="{t.series}"/>')
    s.append(f'<text x="{gx}" y="{y + h / 2 + 12}" text-anchor="middle" fill="#022749" font-family="HNC" font-size="34">✓</text>')
    s.append(f'<line x1="{gx}" y1="{y + h / 2 - 26}" x2="{gx}" y2="{y - 34}" stroke="{t.series}" stroke-width="2"/>')
    s.append(label(gx, y - 44, "rejects what the agent gets wrong", t, size=22))
    # brackets: computed (pipeline + outputs) vs interpreted (LLM)
    by = y + h + 44
    s.append(f'<path d="M{xs[2]},{by} v16 h{xs[3] + w - xs[2]} v-16" fill="none" stroke="{t.series}" stroke-width="3"/>')
    s.append(label((xs[2] + xs[3] + w) / 2, by + 56, "COMPUTED · TRACEABLE", t, size=30, colour=t.series, family="HNC"))
    s.append(label((xs[2] + xs[3] + w) / 2, by + 90, "every number comes from here", t, size=22))
    s.append(f'<path d="M{xs[4]},{by} v16 h{w} v-16" fill="none" stroke="{t.accent}" stroke-width="3"/>')
    s.append(label(xs[4] + w, by + 56, "INTERPRETS ONLY", t, size=30, family="HNC", anchor="end"))
    s.append(label(xs[4] + w, by + 90, "cannot invent a number", t, size=22, anchor="end"))
    s.append("</svg>")
    card = Card(
        title="The model never touches the numbers",
        subtitle="A DAVINCI evaluation as a guardrailed AI workflow",
        plot_svg="\n".join(s), theme=t,
        footnote="The language model is given the run's statistics and figures and is instructed not to "
                 "invent numbers that are not in them. The stage is optional and never fatal.",
        credit="", lockup="NSF NCAR &middot; DAVINCI",
    )
    return render(card.html(), OUT / "diagram_pipeline.png")


def two_paths_diagram() -> Path:
    """DAVINCI (REST API + cache + pipeline) and the KBase MCP (tools + AWS Zarr) both reach POWER."""
    t = THEMES["navy"]
    s = [svg_open(theme=t)]
    # central POWER node with its two halves
    cx, cw, cy, ch = BOX_W / 2 - 150, 300, 60, 440
    s.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="18" fill="rgba(240,128,60,0.16)" stroke="{t.series}" stroke-width="3"/>')
    s.append(label(cx + cw / 2, cy + 64, "NASA POWER", t, size=46, colour=t.text, family="HNC"))
    s.append(f'<line x1="{cx + 30}" y1="{cy + 100}" x2="{cx + cw - 30}" y2="{cy + 100}" stroke="{t.edge}" stroke-width="2"/>')
    s.append(label(cx + cw / 2, cy + 190, "MERRA-2", t, size=40, colour=t.text, family="HNC"))
    s.append(label(cx + cw / 2, cy + 226, "meteorology", t, size=24))
    s.append(f'<line x1="{cx + 30}" y1="{cy + 262}" x2="{cx + cw - 30}" y2="{cy + 262}" stroke="{t.edge}" stroke-width="2"/>')
    s.append(label(cx + cw / 2, cy + 340, "CERES", t, size=40, colour=t.text, family="HNC"))
    s.append(label(cx + cw / 2, cy + 376, "solar", t, size=24))
    # left column: DAVINCI path; right column: MCP path
    w, h, gap = 300, 78, 22
    left_x, right_x = 0, BOX_W - w
    left = [("YAML CONFIG", ""), ("REST API · NETCDF", ""), ("LOCAL CACHE", "reruns offline"), ("PAIR · STATS · PLOTS", "")]
    right = [("AGENT", ""), ("TOOL CALL", "point or bbox"), ("AWS ZARR", "anonymous"), ("{ DATA, _META }", "license · citation · query")]
    top = 96
    for col_x, items, head in ((left_x, left, "DAVINCI"), (right_x, right, "KBASE ENV-DATA-MCP")):
        s.append(label(col_x + w / 2, top - 30, head, t, size=30, colour=t.text, family="HNC"))
        for i, (n, sub) in enumerate(items):
            y = top + i * (h + gap)
            s.append(node(col_x, y, w, h, n, t, sub=sub, size=28) if sub else node(col_x, y, w, h, n, t, size=28))
            if i < len(items) - 1:
                s.append(arrow(col_x + w / 2, y + h + 3, col_x + w / 2, y + h + gap - 4, t, width=3))
    # POWER -> REST API (left row 2) and POWER -> AWS Zarr (right row 3)
    ry1 = top + 1 * (h + gap) + h / 2
    ry2 = top + 2 * (h + gap) + h / 2
    s.append(arrow(cx - 6, cy + 190, left_x + w + 8, ry1, t))
    s.append(arrow(cx + cw + 6, cy + 300, right_x - 8, ry2, t))
    s.append("</svg>")
    card = Card(
        title="Two paths to POWER",
        subtitle="A reproducible campaign, or a tool an agent can call",
        plot_svg="\n".join(s), theme=t,
        footnote="Left: DAVINCI's power source fetches the API and caches NetCDF. Right: the KBase env-data-mcp "
                 "server reads POWER's AWS Open Data Zarr stores and returns provenance with every call.",
        credit="", lockup="NSF NCAR &middot; DAVINCI",
    )
    return render(card.html(), OUT / "diagram_two_paths.png")


if __name__ == "__main__":
    for p in (pipeline_diagram(), two_paths_diagram()):
        print(p)
