# Talk visuals

Broadcast-style graphic cards for the POWER lightning talk, rendered as
3840 × 2160 PNGs (16:9) for direct placement in PowerPoint. Style reference:
`climate-central-style-notes.md`.

## Regenerate

Run from the repo root in the `davinci` env (Playwright + Chromium are installed there):

```bash
cd lightning_talk/visuals
python extract_data.py     # runs configs/power-annual-anomaly.yaml through PipelineRunner -> data/*.csv
python make_cards.py       # data cards -> cards/card_*.png
python make_diagrams.py    # diagram cards -> cards/diagram_*.png

# KBase env-data-mcp card: fetch through the MCP protocol (server env), then draw
cd ~/EarthData/env-data-mcp && uv run python ~/EarthSystem/DAVINCI/lightning_talk/visuals/fetch_mcp_context.py
cd - && python make_mcp_card.py   # -> cards/card_field_site_context.png
```

`extract_data.py` needs the POWER cache at `analyses/power/cache` (already populated);
it makes no network requests when the cache is warm.

## Pieces

| File | Role |
|---|---|
| `cardkit.py` | frame, themes, HTML shell, Playwright render, SVG helpers (trend plot, stripes, diagram boxes/arrows) |
| `configs/power-annual-anomaly.yaml` | annual 2 m temperature anomalies at four sites, vs the 1991–2020 normal |
| `extract_data.py` | pipeline run → tidy CSV (`data/`); nothing recomputed outside the pipeline |
| `make_cards.py` | data cards |
| `make_diagrams.py` | diagram cards |
| `fetch_mcp_context.py` | calls eight env-data-mcp tools over stdio for the Yakima site; saves each response with its `_meta` to `data/mcp/` |
| `make_mcp_card.py` | the field-site context card, from those responses only |

## Rules the cards follow

- Fine print (source, method, access date) is kept in each card's `footnote` but not
  rendered (`Card.show_footnote=False`); it goes into PowerPoint annotations instead.
- Every number on a card comes from `PipelineRunner` output. The one exception is the
  hero number on trend cards: a least-squares trend fitted in `cardkit.linear_trend`,
  which the footnote states.
- Fonts are macOS system faces that PowerPoint on this Mac also has: Helvetica Neue
  Condensed Bold (titles, numbers, years) and Helvetica Neue (subtitles, footnotes).
  On Windows, PowerPoint will substitute; embed fonts in the deck if it travels.
- All content sits inside the centred 60 % × 80 % box; the outer gutters carry only
  background, so the cards survive lower-thirds and 4:3 crops.
- Colours: one warm series (`#f0803c`) and a white reference line on a dark themed
  field. Do not put the orange and the gold (`#f5b942`) on the same chart as two
  series — they fail the palette validator's normal-vision floor (ΔE 13.6 < 15).
