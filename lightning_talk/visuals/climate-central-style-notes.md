# Climate Central graphic style — notes for the lightning talk visuals

Purpose: pin down, from measurement, the graphical style Climate Central uses for
broadcast- and web-ready graphics, so we can define skills that produce the same
kind of graphic from DAVINCI/POWER data. This is a description of a *style*; we
do not reuse their assets, logo, or background art.

Studied 2026-09-05. Page: https://www.climatecentral.org/climate-briefings/july-2026
(sections "Global Climate", "Additional Graphics", "Relevant Climate Matters") plus
one Climate Matters package page (More U.S. Fire Weather). Reference copies of the
images live only in the session scratchpad; re-fetch from the URLs below.

## Graphics examined

| # | Graphic | Form | URL (images.ctfassets.net/cxgxgstp8r5d/…) |
|---|---|---|---|
| 1 | Sea Surface Temperatures, July 2026 | global anomaly map, diverging | `7eog03eej8901sdwtUYl3t/…/SST_anomalies_Global_July-2026.png` |
| 2 | Global Mean Temperature by Day | current year vs normal with all-years envelope | `4HEYok4Cl8J9BGPeD2KnBk/…/Tave_daily_Global_July-2026.png` |
| 3 | Fire Weather Days (average) | U.S. choropleth, sequential, stepped legend | `4T5xUnffOiUdMFPdNTwQRE/…/2026FireWeather_Average_en_title_lg.jpg` |
| 4 | Change in Fire Weather Days | U.S. choropleth, diverging | `696oWtwOr8fzikm7D8cLhd/…/2026FireWeather_Change_en_title_lg.jpg` |
| 5 | Fire Weather Days, Southeast Desert Basins CA | local annual series + trend + hero number | `4NhYBuhxsLGIaUWiIGiFVs/…/2026FireWeather_Local_palmsprings_en_title_lg.jpg` |
| 6 | More Warm Summer Nights, Austin | local annual series + trend + hero number | `7iaPU6BdK0I7frVx7EIhbX/…/2026WarmSummerNights_NightsAbove_austin_en_title_lg.jpg` |
| 7 | Rising Cooling Demand, Boise | local annual series + trend + hero number | `iqQLko4MZ4R9jhUT1G4j2/…/2026CDD_boise_en_title_lg.jpg` |
| 8 | Global warming stripes 1850–2021 | stripes, no axes | `4coKFI6Q0LNZj7kjgvTTbR/…/20222021GlobalTemps_Stripes_en_title_lg.jpg` |
| 9 | Wildfire Risk to Homes, California | state choropleth + boxed hero statement, photo background | `8qCx6NP0ZWJfOpico9IHF/…/2024WildfireRisk_State_CA_en_title_lg.jpg` |

Program context (their own words): Climate Matters produces "free weekly climate
reporting materials in English and Spanish, localized for 245+ U.S. cities and media
markets" for "meteorologists and journalists"; "production-ready, localized visuals
just a click away". File names encode the variants: `<year><Topic>_<Metric>_<city>_<lang>_<title|notitle>_<size>`.
Every graphic is a *parameterised template*: the city, the metric, the language and
the title toggle are inputs. That is the thing to copy.

## Anatomy — what every card has

1. **Headline title**, all caps, bold condensed sans, white (or pale accent), top-left of the content box.
2. **Subtitle** in sentence case naming the metric ("Annual cooling degree days"), in the theme's accent colour.
3. **The plot**, occupying most of the content box, with the minimum of chrome.
4. **The takeaway**: one hero number ("+99%", "+44 NIGHTS", "+69 DAYS") bottom-right of the plot, or a boxed statement ("38.3 MILLION PEOPLE …").
5. **Location** centred under the x-axis, with only the **start and end year** as x labels.
6. **Source line**: definition of the metric, dataset, and "Accessed: <date>", small, low-contrast, bottom-left.
7. **Logo lockup** bottom-right, same baseline as the source line.
8. **Themed background**: a dark, topic-coloured field (flat, gradient, or a low-contrast photo/texture) that sits *behind* the content box and bleeds to the frame edge.

No axis lines, no tick marks, no box around the plot, no legend when there is a
single series. The white trend line is the only "annotation" on the local cards.

## Geometry (measured on the 1600 × 900 renders)

| Element | Measured | Rule |
|---|---|---|
| Frame | 1600 × 900 (16:9); briefing graphics 1600 × 1039–1090 | Produce 1920 × 1080 for TV; 16:9 for web |
| Content box | title left edge x = 320 (20 %), title top y ≈ 90–97 (10 %), logo right edge x = 1280 (80 %), source baseline y ≈ 810 (90 %) | **All text and data inside a centred 60 % × 80 % box**; the outer gutters carry only background |
| Title | cap height ≈ 55 px (≈ 6 % of frame height) | one line; two if unavoidable |
| Subtitle | ≈ 28 px (≈ 3 %) | one line |
| Hero number | ≈ 90 px (≈ 10 %), unit word beneath at ≈ 30 px | bottom-right, inside the plot's right margin |
| Axis labels | ≈ 26 px, start/end year ≈ 34 px | |
| Source line | ≈ 14 px (≈ 1.5 %), two lines max | |
| Logo | ≈ 270 × 24 px | bottom-right, right edge at 80 % |
| Stripes card | title omitted; start year / place / end year on one row under the stripes | |

The 20 % side gutters are what make the card survive a 4:3 centre crop and lower-third
overlays; keep them even for web.

## Typography

- **Titles and big numbers**: a bold *condensed* grotesque, all caps, tight tracking.
  Looks like **Oswald** (unconfirmed — the website itself uses Work Sans and Maitree,
  which are not the graphic faces). Open-licence stand-ins: Oswald, Barlow Condensed,
  Roboto Condensed.
- **Subtitle, ticks, source line**: a rounded humanist sans, regular weight, sentence
  case. Looks like Nunito / Nunito Sans. Stand-ins: Nunito Sans, Source Sans 3, or
  the project's Poppins (NCAR brand) if we want one family for body text.
- Numbers on axes use the condensed face; the source line uses the body face.
- Hierarchy is carried by **size and case**, not by colour: title white, subtitle
  accent, everything else muted.

## Colour (sampled)

Backgrounds are dark and *themed by topic*; text is white or a pale tint of the theme;
the data series is a warm orange on every dark theme; the trend line is white.

| Role | Blue card (Boise) | Navy briefing (SST, daily) | Ember (fire) | Night sky (warm nights) | Stripes |
|---|---|---|---|---|---|
| Background | `#17416b` (flat) | `#022749` (flat) | `#200806` with ember glow texture | gradient `#020308` (top) → `#590c07` (bottom) | `#161417` |
| Title | white | white / `#aac7e2` pale blue | white | white | white |
| Subtitle / accent | `#82bee6` | `#aac7e2` | orange `#d77214` | muted rose-gray | — |
| Axis text | `#73b0d6` | pale blue | tan | rose-gray | white (years) |
| Gridlines | `#285d8f` hairline | hairline one step off surface | hairline | hairline | none |
| Series | orange `#e07a5a`-ish with dots | gold current year, dotted white normal, gray-olive envelope, red end dot | orange `#d77214` | orange-red `#dc6041` | — |
| Trend / reference | white 6 px | — | white | white | — |
| Hero number | white | — | white | white | — |
| Source line | `#82bee6` at ~60 % | pale blue at ~60 % | tan at ~60 % | rose-gray | gray |

Map ramps:
- Sequential (fire weather days): white `#ffffff` at 0 → ColorBrewer **Reds** to `#67000d` (the darkest Reds step, measured exactly), 7 classes on a log-ish scale (0, 1, 7, 14, 28, 56); **"No data" = `#9b9698` gray** as a labelled class at the left of the legend bar.
- Diverging (change in fire weather days): ColorBrewer **RdBu**-style, white at zero, symmetric labels (−56 … +56), same gray no-data class.
- Global SST anomaly: blue–white–red with **dark-red saturation at +4 °C**, land flat gray `#696969`, no coastlines drawn beyond the land fill, Robinson-style projection, colourbar with the poles named ("Colder", "Warmer") not just numbered.
- Warming stripes: Hawkins' blue–red stripes, no axes.

Notes against the data-viz method (`dataviz` skill): the diverging maps put *white*
at the midpoint on a dark surface, which reads as "zero" because the surface is dark;
on a light surface the method's neutral-gray midpoint applies. The single orange
series on a dark surface clears 3:1 easily; the pale-blue subtitle on `#17416b` is
~4.5:1. Their categorical needs are tiny (one series + one reference), which is why
the cards look calm.

## Chart grammar — the reusable forms

| Form | What it is | DAVINCI/POWER analogue |
|---|---|---|
| **Local trend card** (5, 6, 7) | annual values as a thin orange line with small dots; a straight white least-squares trend; hero number = trend change over the period; x labels = first and last year only; place name centred | POWER monthly → annual at a site: T2M, ALLSKY_SFC_SW_DWN; MERRA-2 − POWER bias per year |
| **Record-vs-normal line** (2) | current year in gold; the climatological normal as a dotted white line; the envelope of all prior years as a translucent gray band; red dot at the latest value; annotation inside the plot | POWER 2024 daily/monthly vs the 1991–2020 normal, envelope 1984–2023 |
| **Anomaly map** (1) | global or regional field vs a named baseline; diverging ramp with named poles; period label under the map; gray land | POWER regional anomaly; MERRA-2 − POWER bias map (spatial_bias) |
| **Choropleth with stepped legend** (3, 4) | classed sequential or diverging ramp; legend bar *above* the map with class edges; explicit no-data class | POWER regional grid (bin to classes) |
| **Stripes** (8) | one colour per year, no axes, three labels | POWER T2M anomaly at Barrow 1981–2024; SWdn at Mauna Loa 1984–2024 |
| **Boxed hero statement** (9) | a small map plus a white card with a number and a sentence | "MERRA-2 is 9 W m⁻² too bright" with the CONUS bias map |

## Story rules they follow

- **One message per graphic.** The title is the message in plain words ("MORE WARM
  SUMMER NIGHTS", "RISING COOLING DEMAND"); the subtitle is the metric; the hero
  number is the evidence. Never a title that only names a variable.
- **Name the baseline** in the graphic ("compared to the 1991–2020 average") and the
  period ("1970–2025"); never leave the reader to infer the reference.
- **Source and access date on every graphic**, plus a one-sentence metric definition.
- **Localisable by construction**: the place name is a parameter; the same template
  renders 245 cities. For us: the site (Boulder, Barrow, Mauna Loa, South Pole) is
  the parameter.
- **Theme follows topic**, not dataset: heat → warm reds, cooling demand → blue,
  fire → ember, nights → night sky. Backgrounds are atmospheric but never compete
  with the data (blurred, dark, low contrast).
- Titled and untitled variants are published so broadcasters can add their own
  lower-third; sizes `lg` and smaller; English and Spanish.

## What to mimic, what not to

Mimic: the 60 % × 80 % content box and 16:9 frame; the title/subtitle/hero/source/logo
anatomy; condensed all-caps headline typography; single warm series + white trend on a
dark themed field; named baselines; stepped legends with an explicit no-data class;
the parameterised-template idea.

Do not copy: the Climate Central name, logo or lockup; their background photographs
and ember/starfield textures; their exact palettes as a "brand". Our lockup is NSF
NCAR / DAVINCI with a "Data: NASA POWER" credit line, and POWER's requested
attribution ("These data were obtained from the NASA Langley Research Center (LaRC)
POWER Project …") goes in the source line where the graphic uses POWER data.

## As a design-system instance for the `dataviz` method

Parameters the method needs, proposed from the measurements above (to be validated
with `scripts/validate_palette.js` per surface before use):

| Parameter | Proposed value |
|---|---|
| Surfaces (dark only; these graphics have no light mode) | navy `#022749`; card blue `#17416b`; ember `#200806`; night `#0b0a12` |
| Primary ink | `#ffffff` |
| Secondary ink | pale tint of the theme (`#aac7e2` on navy/blue; warm tan on ember) |
| Muted / source line | secondary ink at 60 % |
| Gridline | one step off the surface (`#285d8f` on card blue), 1 px, solid |
| Categorical order | 1 orange `#e07a5a`, 2 gold `#f5b942`, 3 pale blue `#82bee6`, 4 white (reference/trend), then the method's defaults |
| Sequential hue | Reds (white → `#67000d`) on maps; blue for anything on a light page |
| Diverging pair | blue ↔ red, white midpoint on dark surfaces, gray on light |
| No-data | `#9b9698` as a labelled class |
| Status | not used |
| Typeface | condensed bold sans for title/hero (Oswald or Barlow Condensed); body sans for the rest (Nunito Sans or Poppins) |

The validator needs Node, which is not on this machine's PATH (see the open items).

## Proposed skills (for discussion — nothing built yet)

1. **`broadcast-card` (design system + anatomy).** The parameters above, the frame
   and content-box geometry, the anatomy checklist, and the story rules. It is the
   style layer every card inherits; it does not know about DAVINCI.
2. **`card-recipes` (the six forms).** One recipe per row of the chart-grammar table:
   inputs (a tidy series or field, place, units, baseline, period), the message
   template for the title, how the hero number is computed (trend × span, latest
   anomaly, mean bias), and the checks (baseline named, source line present, x has two
   labels, one message).
3. **`davinci-to-card` (data hand-off).** How to get the numbers out of a DAVINCI run
   (statistics CSV, paired NetCDF, manifest) into the recipe inputs without
   recomputing anything — the reliable-workflow rule from the talk applies to the
   graphics too: every number on a card comes from pipeline output.

Two skills would also work (fold 3 into 2). Skill 1 is the one that needs the most
care and the validator.

## Generation options (for discussion)

| Option | How | Pros | Cons |
|---|---|---|---|
| A. Matplotlib in DAVINCI | a `context: broadcast` style preset + a `cards` renderer family; PNG 1920 × 1080 and PDF | same pipeline, data already there, reproducible, no new toolchain | typography and gradients are laborious; condensed fonts must be installed; text layout is fiddly |
| B. HTML/SVG cards rendered headlessly | a card template in HTML + CSS (web fonts, gradients, exact type); data as JSON from DAVINCI; render to PNG with Playwright (Python package drives Chromium) | best typographic control; matches the `dataviz` method's HTML-first approach; easy titled/untitled and language variants | new toolchain (Playwright + Chromium download); charts drawn in SVG/D3 or embedded from A |
| C. Hybrid | Matplotlib renders only the data layer (map, line) to SVG; the HTML shell adds title, subtitle, hero, source, logo, background; Playwright renders | keeps the science plotting in DAVINCI; gives the shell the web's typography | two steps to keep in sync |
| D. Claude Design canvas | artboards per graphic for hand refinement after A/B/C produce the data layers | fastest way to iterate layout by eye | not reproducible from the pipeline; final assets only |

Recommendation to discuss: **C** for the talk (six to eight graphics, high visual
bar, data must stay pipeline-computed), with **A** only for pieces that should live
in DAVINCI permanently. B if we decide to draw the plots in SVG as well.

## Candidate graphics for the six slides

| Slide | Graphic | Form |
|---|---|---|
| 2 | the chat → skills → MCP → guardrailed-pipeline spectrum | diagram card, not data |
| 3 | the config as a card, with the API → cache → reader → pairing flow | diagram card |
| 4 | Barrow 2 m temperature anomaly 1981–2024 | local trend card (hero: trend change) or stripes |
| 4 | Barrow + South Pole shortwave in antiphase | record-vs-normal line, or stripes pair |
| 4 | MERRA-2 − POWER surface shortwave, CONUS, Feb 2024 | anomaly map + boxed hero "+9 W m⁻²" |
| 5 | agent → MCP → POWER Zarr → `{data, _meta}` | diagram card |
| 6 | none, or the stripes as a closing image | — |

## Open items

- Node 26.8.1 is now in the `davinci` env (conda-forge `nodejs`, installed
  2026-09-05), so the `dataviz` palette validator can run.
- Confirm the title typeface (Oswald is the guess) before we buy into it. None of
  Oswald, Poppins, Nunito or Barlow is installed on this machine (Matplotlib is
  already falling back from Poppins); Helvetica Neue is present, and its Condensed
  Bold face is a no-install stand-in for the headline style.
- The Observable collection `@climatecentral/climate-services` (rate-limited when
  fetched) may expose their chart code and exact palette; worth one look.
- Decide the lockup: NSF NCAR + DAVINCI wordmarks, and where "Data: NASA POWER" sits.
- Playwright 1.62.0 (Python, pip `--no-deps`; `greenlet` and `pyee` from conda-forge)
  and its Chromium headless shell are installed in the `davinci` env (2026-09-05);
  a 1920 x 1080 render test passed. Not yet recorded in `environment.yml`.
