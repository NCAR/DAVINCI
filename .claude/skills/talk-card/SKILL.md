---
name: talk-card
description: Build a broadcast-style 16:9 graphic card (Climate Central look) from DAVINCI pipeline output or saved MCP tool responses, render it to PNG + vector PDF with lightning_talk/visuals/cardkit.py, check it, and deliver it. Use when asked for a talk graphic, slide graphic, public-media graphic, "card", stripes, trend card, or diagram card for the POWER lightning talk or any presentation of DAVINCI / POWER / env-data-mcp results.
---

# Talk cards

A card is one 1920 x 1080 frame that tells one story with almost no words, dropped
into PowerPoint as an image. The mechanics live in `lightning_talk/visuals/cardkit.py`;
this skill is the procedure and the checks that, if skipped, cost a render cycle each.

Style reference (measured, not guessed): `lightning_talk/visuals/climate-central-style-notes.md`.
Worked examples: `make_cards.py` (trend card, stripes), `make_diagrams.py` (diagram
cards), `make_mcp_card.py` (context tiles + small multiples).

## Procedure

1. **Get the numbers from a run, never from a side computation.**
   - DAVINCI data: write or reuse a config, run it, and read datasets from
     `result.context.sources[<key>].data` (see `extract_data.py`). Annual means,
     anomalies, resampling all belong in the config (`resample:`, `analyses:`), not
     in the card script.
   - MCP data: call the tools over the protocol (see `fetch_mcp_context.py`) and save
     every response verbatim with its `_meta`; the card reads only those files.
   - The only numbers a card script may add: a least-squares trend for a hero
     (`cardkit.linear_trend`) and simple aggregates of what was returned (a count, a
     mean, a sum over months). State them in `footnote`.
2. **Pick the form** from the chart grammar in the style notes: local trend card,
   record-vs-normal line, stripes, anomaly map, context tiles, diagram. One message
   per card; the title is the message in plain words, the subtitle names the metric.
3. **Build** with `Card(...)` + a plot SVG from the helpers (`trend_card_svg`,
   `stripes_svg`, `node`/`arrow`/`label` for diagrams) and `render()`. Put the script
   in `lightning_talk/visuals/`, outputs in `cards/` (git-ignored).
4. **Render, then look at the PNG at full size** (Read the file). The validator
   checks colours, nothing checks layout. Fix, re-render, look again.
5. **Deliver**: PNG + PDF land in `cards/`; copy both to
   `~/Library/Mobile Documents/com~apple~CloudDocs/Claude/lightning_talk/` and
   `~/Desktop/POWER/graphics/`. Commit scripts only (never the PNG/PDF/data).

## Rules

- **Frame**: everything inside the centred 60 % x 80 % box (`BOX_*` in cardkit); the
  gutters carry only background. Title all caps condensed, subtitle sentence case,
  plot, one hero number bottom-right, lockup bottom-right.
- **Hero = a fact about the subject** ("+2.9 °C since 1981", "252 mm of rain a
  year"). Never plumbing (call counts, latency, keys). If there is no honest hero,
  leave it out; the tiles or plot carry the numbers.
- **Fine print off**: `Card.show_footnote=False`, `credit=""`. Keep the footnote text
  in the script; it becomes a PowerPoint annotation.
- **Fonts**: Helvetica Neue Condensed Bold (`HNC`) for titles, numbers, years, labels;
  Helvetica Neue for subtitles and small text. Both are macOS system faces PowerPoint
  has; do not introduce web fonts.
- **Colour**: one warm series `#f0803c` + white reference on a dark themed field
  (`THEMES`: card, arctic, navy, ink, biome). A second data series is pale blue or
  white, never the gold `#f5b942` next to the orange (fails the normal-vision floor).
  Text wears text tokens; only marks wear the series colour.
- **Axes**: no axis lines, hairline gridlines, y ticks in the condensed face, x labels
  are the first and last year only, place name centred under the axis.
- **Baseline named** in the subtitle ("vs the 1991–2020 average") and the period
  visible on the axis.
- **Stripes**: one common colour scale across sites (state the limit), Hawkins'
  Blues + Reds; labels carry the latitude, not the state code.

## Collision checklist (each cost a render cycle)

- **Subtitle wraps** to two lines → it collides with the plot. At 40 px the limit is
  about 55 characters. Cut words, do not shrink the font.
- **Hero vs the last-year label**: the hero block (number ~135 px + unit ~46 px) must
  end above the year row. Give the plot explicit `y_range` / `y_step` so the lower
  right is empty, then set `hero_top` (509 worked with y_range (-4.5, 4.5)).
- **Auto y-range with padding** can pull ticks to odd values (−10°) and squash the
  data; prefer explicit `y_range` for hero cards.
- **Row labels clipped at the left**: raise `label_w` in `stripes_svg` before
  shrinking the font.
- **Bracket / caption labels overlapping** in diagrams: right-align the rightmost
  label (`anchor="end"`), shorten the middle one, keep captions ≥ 40 px above the row.
- **Panel headers vs annotations**: a small-multiple's title, source tag, and any
  in-plot note need three separate positions; put the note inside the band it
  describes.
- **Node sub-labels overflow** a 216 px node above ~20 px font; shorten or drop.

## Regenerate everything

```bash
cd lightning_talk/visuals
python extract_data.py && python make_cards.py && python make_diagrams.py
cd ~/EarthData/env-data-mcp && uv run python ~/EarthSystem/DAVINCI/lightning_talk/visuals/fetch_mcp_context.py && cd -
python make_mcp_card.py
```

Requirements (already in the `davinci` env): Playwright 1.62 + Chromium, `uv`,
`nodejs` (for the `dataviz` skill's palette validator when adding a colour).
