# ASCII geographic maps

This optional viewer has its own script, Python API, and YAML controls. It is
independent of DAVINCI's analysis CLI, registries, and pipeline: no existing
component imports it. It reads a single NetCDF file with numeric fields on
one-dimensional geographic latitude/longitude axes, including MERRA-2.

Activate the `davinci` environment, then run from the checkout:

```bash
python scripts/terminal_map.py /path/to/fields.nc4
# Equivalent installed-module entry point:
python -m davinci_monet.terminal_map /path/to/fields.nc4
```

The default logical grid is 180 latitude rows by 360 longitude columns. Each
character represents one interpolated cell. Curses measures the active terminal
in rows/columns, reserves room for labels and the legend, and shows a movable
viewport. It measures again after a resize. Use `f` to fit the whole globe in the
window, or `g` to enter a grid such as `90 180`. Terminal characters are generally
taller than they are wide; this is a character-cell geographic display, with no
attempt to correct font aspect ratio.

## Configuration

YAML is optional. Defaults < YAML < explicit command-line options. The tracked
[example](../examples/terminal-map.example.yaml) contains every setting:

```bash
export MERRA2_ROOT=/Volumes/Io
python scripts/terminal_map.py --config examples/terminal-map.example.yaml
# Open a new macOS Terminal, sized in character cells:
python scripts/terminal_map.py -c examples/terminal-map.example.yaml --new-window 40 120 --fit
```

```yaml
source:
  filename: ${MERRA2_ROOT}/MERRA2_tavg1/slv_Nx/MERRA2_400.tavg1_2d_slv_Nx.20260401.nc4
  variable: T2M
  indices: {time: 0}
display:
  grid: {lat: 180, lon: 360}
  interpolation: linear
  fit: false
  bins: 10
  limits: null
  mask_below: null
terminal:
  new_window: false
  rows: 40
  columns: 120
  font_size: 12
```

Unknown/duplicate YAML keys, invalid types, and unresolved path environment
variables are errors. `${VAR}`, `$VAR`, and `~` expand in `source.filename`.
Relative YAML filenames are relative to the YAML file; a positional CLI filename
is relative to the working directory. Only `source.filename` is a data path; the
viewer never searches directories or follows file globs. `._` resource forks are
rejected.

`terminal.rows` and `terminal.columns` size a **new** macOS Terminal window when
`terminal.new_window: true`; they do not override the current terminal's detected
size. `--current-window` overrides YAML window creation. The new process uses the
same Python interpreter and package location, without requiring conda activation
in the new shell. macOS may ask for permission to control Terminal. Other systems
can use the current-terminal curses interface; window creation is macOS-specific.
`terminal.font_size` / `--font-size` sets only the new window's font (12 points by
default), making the default 40-by-120 window fit typical screens. Physical screen
limits can still constrain larger requests; the launcher reports both requested
and actual dimensions. It does not change your saved Terminal profile.

## Display and selection

Numeric fields use ten equal-width bins by default, labeled `0` through `9`.
`display.bins` / `--bins` can choose 1-10. The legend gives interval boundaries and
the file's native units. Lower boundaries are included; the maximum is included in
the last bin. Automatic limits come from the entire selected native slice and
stay fixed while panning, resizing, or changing the display grid. Changing the
variable, time, or level recalculates automatic limits. Set `limits: [MIN, MAX]`
or `--limits MIN MAX` to hold a scale across different slices; tails are clipped
to the first/last digit. A constant field uses `0`.

`-` means missing, nonfinite, out of source coverage, or explicitly masked. For
terrain elevations, `mask_below: 0` / `--mask-below 0` hides below-sea-level values,
including below-sea-level land. It is not a general ocean mask. Zero stays visible.
There is no automatic field-unit conversion, vertical interpolation, or time
averaging. Labels preserve the file's metadata in printable ASCII, rather than
using Matplotlib/LaTeX typesetting.

MERRA-2 `LWI` is a special categorical view: `1` is land, `2` is ice, and water
is `-`. Nearest-neighbor interpolation preserves those codes despite packing
roundoff. This provides a recognizable geography check using the files on `Io`;
it is not elevation or a static ocean mask. The available meteorology's `H1000`
and `H500` are pressure-level heights, not terrain. Actual MERRA-2 topography
requires the separate constants file (`PHIS`, surface geopotential).

The script initially selects the first supported variable unless one is given.
All non-horizontal dimensions are selectable. Initial time/member indices are
zero. The shared CF-aware surface helper selects native surface levels, including
MERRA-2's last `Nv` layer and first `Np` pressure level. All displayed indices are
zero-based. Select negative indices on the CLI/Python API as with Python arrays.

| Key | Action |
| --- | --- |
| Arrows or `h j k l` | Pan one cell; longitude wraps and latitude clamps |
| PageUp / PageDown | Pan north/south half a viewport |
| Home | Center the viewport at latitude/longitude zero |
| `v` | Variable picker (`/` filters; arrows/PageUp/PageDown scroll) |
| `t`, `z`, `d` | Pick time, native vertical level, or another dimension |
| `,` / `.` | Previous/next time |
| `[` / `]` | Previous/next native level |
| `g` | Enter latitude rows and longitude columns |
| `+` / `-` | Double/halve logical grid resolution |
| `f` | Toggle whole-globe fit and the requested logical grid |
| `i` | Toggle linear/nearest (LWI remains nearest) |
| `b`, `m` | Set bin limits / masking cutoff; blank restores automatic / disables |
| `?`, `q` | Help / quit; Esc cancels a menu or quits the map |

## Examples on Io

```bash
# Continent shapes:
python scripts/terminal_map.py /Volumes/Io/MERRA2_inst3/aer_Nv/MERRA2_300.inst3_3d_aer_Nv.20080701.nc4 --variable LWI --fit
# Surface temperature:
python scripts/terminal_map.py /Volumes/Io/MERRA2_tavg1/slv_Nx/MERRA2_400.tavg1_2d_slv_Nx.20260401.nc4 --variable T2M
# Humidity on native level 71 (MERRA-2 layer 72, nearest the surface):
python scripts/terminal_map.py /Volumes/Io/MERRA2_inst3/aer_Nv/MERRA2_300.inst3_3d_aer_Nv.20080701.nc4 --variable RH --level 71
# List variables without loading their data, or make a bounded text snapshot:
python scripts/terminal_map.py /path/to/file.nc --list
python scripts/terminal_map.py /path/to/file.nc --variable T2M --print --fit --size 35 100
```

Other CLI selections include `--time N` and repeatable `--index DIM=N`.
`--auto-limits`, `--no-mask`, and `--no-fit` clear the corresponding YAML controls.
Explicit dimension selections that do not belong to the chosen variable are
errors. In the interactive variable picker, new variables receive fresh defaults.

## Python API

```python
from davinci_monet.terminal_map import DisplayGrid, MapDataset, MapView, render_frame

with MapDataset.open("/path/to/file.nc4") as dataset:
    view = MapView(dataset, "T2M", grid=DisplayGrid(90, 180), indices={"time": 0})
    print("\n".join(render_frame(view, rows=35, columns=100)))
```

`MapDataset(xarray_dataset)` also accepts an existing dataset; its caller retains
ownership. Only the selected 2-D native slice is loaded. Panning uses cached
characters, while grid changes reinterpolate the cached native slice. Source
longitudes are normalized and sorted; global seams interpolate cyclically.
Regional grids crossing the dateline are unwrapped across their data interval.
Regional gaps, unsampled polar latitudes, and NaNs are not extrapolated or filled.
Curvilinear, projected, unstructured, and single-point grids are rejected with a
clear diagnostic. Interpolation is for visual exploration, not conservative
scientific regridding.

The viewer uses NumPy, xarray, SciPy (interpolation), and the optional CLI uses
PyYAML/Pydantic plus Python's curses module. These are provided by the existing
`davinci` environment. It does not import Matplotlib or register a plotter/stage.
