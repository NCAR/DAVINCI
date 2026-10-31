"""Optional ASCII maps; no DAVINCI pipeline component depends on this package.

Use ``python -m davinci_monet.terminal_map --help`` for the standalone viewer.
The public data and rendering API does not import curses or Matplotlib.
"""

from .data import DisplayGrid, MapDataset
from .render import BinScale, MapView, Viewport, ascii_cells, render_frame

__all__ = [
    "BinScale",
    "DisplayGrid",
    "MapDataset",
    "MapView",
    "Viewport",
    "ascii_cells",
    "render_frame",
]
